"""Mean judge score over several judges (CPU). The combiners read the mean as llm_logit.

usage: python -m stage3_judges.merge_scores --name mean_six --judges judge_plain_1,judge_plain_2,judge_context_1,judge_context_2,judge_context_3,judge_context_4
Reads WORK_DIR/stage3/scores/<judge>/<split>[_context]/*.parquet, writes WORK_DIR/stage3/scores/<name>/<split>/part.parquet.
"""
import argparse
import glob
import os

import polars as pl

import config



def load(root, judge, split):
    fs = sorted(glob.glob(f"{root}/{judge}/{split}/*.parquet")) or sorted(glob.glob(f"{root}/{judge}/{split}_context/*.parquet"))
    assert fs, (judge, split)
    q = pl.concat([pl.read_parquet(f) for f in fs]).select(pl.col("s1_id").cast(pl.Utf8), pl.col("cand_id").cast(pl.Utf8),
                                                           pl.col("llm_logit").cast(pl.Float64).alias(judge))
    assert q.select("s1_id", "cand_id").is_duplicated().sum() == 0, (judge, split)
    return q


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--judges", required=True)
    ap.add_argument("--root", default=str(config.WORK_DIR / "stage3" / "scores"))
    a = ap.parse_args()
    judges = a.judges.split(",")
    # every split scored by the first judge (test_<country>, plus hold_<country> when the combiner is retrained)
    splits = sorted({os.path.basename(d).removesuffix("_context") for d in glob.glob(f"{a.root}/{judges[0]}/*") if os.path.isdir(d)})
    for sp in splits:
        qs = {j: load(a.root, j, sp) for j in judges}
        q = qs[judges[0]]
        for j in judges[1:]:
            q = q.join(qs[j], on=["s1_id", "cand_id"])
        assert all(q.height == qs[j].height for j in judges), sp
        os.makedirs(f"{a.root}/{a.name}/{sp}", exist_ok=True)
        q.select("s1_id", "cand_id", pl.mean_horizontal(judges).alias("llm_logit")).write_parquet(f"{a.root}/{a.name}/{sp}/part.parquet")
        print(a.name, sp, q.height, flush=True)
