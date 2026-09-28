"""Encode one text column of a texts parquet with the fine-tuned e5-small retriever, shard i of n, to fp16 .npy (L2-normalised).
The retriever is loaded with the pruned vocabulary (common/encoders.py); embeddings are identical to the full model's.

usage: python encode.py <model_dir> <items> <field> <shard> <nshards> <outdir>
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, polars as pl, torch
from sentence_transformers import SentenceTransformer
import config
from common.encoders import prune_sentence_transformer

model, items, field, shard, nsh, outdir = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5]), sys.argv[6]
W = str(config.WORK_DIR)
os.makedirs(outdir, exist_ok=True)
d = pl.read_parquet(f"{W}/data/texts/{items}.parquet", columns=[field])
n = d.height
lo, hi = shard * n // nsh, (shard + 1) * n // nsh
txt = d[field][lo:hi].to_list()
m = SentenceTransformer(model, device="cuda")
m.max_seq_length = 64
dt = torch.float16 if torch.cuda.get_device_capability()[0] >= 7 else torch.float32
if dt == torch.float16:
    m.half()
prune_sentence_transformer(m)
t = time.time()
out = np.empty((len(txt), m.get_sentence_embedding_dimension()), dtype=np.float16)
CH = 200_000
for i in range(0, len(txt), CH):
    out[i:i + CH] = m.encode(txt[i:i + CH], batch_size=512, normalize_embeddings=True, convert_to_numpy=True).astype(np.float16)
    print(f"{items}/{field} shard {shard}: {min(i + CH, len(txt)):,}/{len(txt):,}  {min(i + CH, len(txt)) / (time.time() - t):,.0f} rec/s", flush=True)
np.save(f"{outdir}/{items}.{field}.{shard}-of-{nsh}.npy", out)
print(f"done {items}/{field} shard {shard}: {len(txt):,} in {time.time() - t:.0f}s ({len(txt) / (time.time() - t):,.0f} rec/s)")
