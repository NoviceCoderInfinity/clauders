"""Score judge input pairs with one LoRA judge (GPU).

usage: python -m stage3_judges.score_lora --judge judge_context_1 --inputs WORK_DIR/stage3/inputs/test_us_context.parquet \
           [--out-dir WORK_DIR/stage3/scores/judge_context_1] [--shard 0 --nshard 1] [--limit N]
Input parquet: s1_id, cand_id, text_a, text_b. Output: <out-dir>/<input stem>/s<shard>of<nshard>p<k>.parquet (s1_id, cand_id, llm_logit).
Adapters: WEIGHTS_DIR/qwen_judges/<judge>/ (adapter_config.json, adapter_model.safetensors, score_head.safetensors).
"""
import argparse
import os
import time

import polars as pl
import torch

import config
from stage3_judges import judge_model as JM

PART = 400_000

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge", required=True, choices=sorted(JM.JUDGES))
    ap.add_argument("--inputs", required=True, nargs="+")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshard", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0, help="score only the first N rows of each input (checks)")
    a = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = True
    kind, maxlen, _, _ = JM.JUDGES[a.judge]
    adapter = a.adapter or str(config.JUDGES_DIR / a.judge)
    out_dir = a.out_dir or str(config.WORK_DIR / "stage3" / "scores" / a.judge)
    backbone, head, remap, tok, pad = JM.load_scoring_model(adapter)
    B = JM.Batcher(tok, pad, maxlen)
    for inp in a.inputs:
        stem = os.path.basename(inp).removesuffix(".parquet")
        df = pl.read_parquet(inp)
        if a.limit:
            df = df.head(a.limit)
        df = df.with_row_index("_i").filter(pl.col("_i") % a.nshard == a.shard).drop("_i")
        for k, s in enumerate(range(0, df.height, PART)):
            dest = f"{out_dir}/{stem}/s{a.shard}of{a.nshard}p{k}.parquet"
            if os.path.exists(dest):
                continue
            part = df.slice(s, PART); t = time.time()
            logit = JM.score(backbone, head, B.tokenize(part), B.batch, remap=remap)
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            part.select("s1_id", "cand_id").with_columns(pl.Series("llm_logit", logit)).write_parquet(dest)
            print(f"{dest}: {part.height:,} pairs in {time.time() - t:.0f}s", flush=True)
