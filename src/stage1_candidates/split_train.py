"""Training-data splits (seed 42).

Source 1 split per country: ENC 85% (retriever training), MT 5% (matcher training), HOLD 10% (validation).
Search index for HOLD and MT = ALL train Source 1 of the country (same scale as test).
Pool = matched records of HOLD + matched records of MT + ALL unmatched train records.

Outputs WORK_DIR/data/split/:
  s1_split.parquet : entity_id, country, split in {ENC, MT, HOLD}, sub (53% subset of HOLD/MT)
  pool.parquet     : entity_id (Source 2/3), src, country, s1 (true Source 1 or null), s1_split (HOLD/MT/null)
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import polars as pl
import config

WORK = str(config.WORK_DIR)
PQ, OUT = f"{WORK}/data/pq", f"{WORK}/data/split"
os.makedirs(OUT, exist_ok=True)
SEED = config.SEED
HOLD_FRAC, MT_FRAC, SUB_FRAC = 0.10, 0.05, 0.53
rng = np.random.default_rng(SEED)

s1 = pl.read_parquet(f"{PQ}/train_s1.parquet", columns=["entity_id", "country"]).sort("entity_id")
parts = []
for (c,), g in s1.group_by(["country"], maintain_order=True):
    ids = g["entity_id"].to_numpy()
    perm = rng.permutation(len(ids))
    n_hold = int(round(HOLD_FRAC * len(ids)))
    n_mt = int(round(MT_FRAC * len(ids)))
    split = np.empty(len(ids), dtype=object)
    split[perm[:n_hold]] = "HOLD"
    split[perm[n_hold:n_hold + n_mt]] = "MT"
    split[perm[n_hold + n_mt:]] = "ENC"
    sub = np.zeros(len(ids), bool)
    for grp in ("HOLD", "MT"):
        idx = np.where(split == grp)[0]
        sub[rng.choice(idx, int(round(SUB_FRAC * len(idx))), replace=False)] = True
    parts.append(pl.DataFrame({"entity_id": ids, "country": [c] * len(ids), "split": split.tolist(), "sub": sub}))
sp = pl.concat(parts)
sp.write_parquet(f"{OUT}/s1_split.parquet")

pairs = pl.read_parquet(f"{PQ}/train_pairs.parquet")  # s1, m, src
s23 = pl.concat([pl.read_parquet(f"{PQ}/train_s{i}.parquet", columns=["entity_id", "country"]).with_columns(pl.lit(i).alias("src"))
                 for i in (2, 3)])
unmatched = s23.join(pairs.select(pl.col("m").alias("entity_id")), on="entity_id", how="anti").sort("entity_id")

# ---- test-scale pool
lab = pairs.join(sp.select(pl.col("entity_id").alias("s1"), pl.col("split").alias("s1_split")), on="s1")
matched = lab.filter(pl.col("s1_split") != "ENC").join(
    s23.select(pl.col("entity_id").alias("m"), "country"), on="m").select(
    pl.col("m").alias("entity_id"), "src", "country", "s1", "s1_split")
pool = pl.concat([matched, unmatched.select("entity_id", "src", "country",
                                            pl.lit(None, dtype=pl.Utf8).alias("s1"), pl.lit(None, dtype=pl.Utf8).alias("s1_split"))])
pool.write_parquet(f"{OUT}/pool.parquet")

print("split sizes:", dict(sp.group_by("split").len().iter_rows()))
print(f"retriever-train pairs (ENC Source 1 only): {lab.filter(pl.col('s1_split') == 'ENC').height:,}")
