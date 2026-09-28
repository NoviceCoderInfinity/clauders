"""e5-small cross-encoder scores as a table: pairs_<items>.parquet (qi, di) + score shards <items>.<k>-of-<n>.npy (in shard order)
-> parquet qi, di, ce (float32).
usage: python cross_encoder_table.py <pairs.parquet> <scores_dir> <items> <out.parquet>
"""
import sys, glob
import numpy as np, polars as pl

pairs, scores_dir, items, out = sys.argv[1:5]
P = pl.read_parquet(pairs, columns=["qi", "di"])
fs = sorted(glob.glob(f"{scores_dir}/{items}.*-of-*.npy"), key=lambda f: int(f.split(".")[-2].split("-")[0]))
s = np.concatenate([np.load(f) for f in fs]).astype(np.float32)
assert len(s) == P.height, (len(s), P.height)
P.with_columns(pl.Series("ce", s)).write_parquet(out)
print(out, P.height)
