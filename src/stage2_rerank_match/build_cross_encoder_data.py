"""Data for the e5-small cross-encoder.

Training (no overlap with anything the LightGBM matcher trains on or anything evaluated):
  - ENC hard-negative triples from retriever mining (read by train_cross_encoder.py)
  - negatives: candidate pairs of unmatched lookalike records in group look_ce (a seeded half of the matcher's lookalike group);
    the LightGBM matcher then uses only look_mt2.
Scoring sets (text files + candidate pairs): pool_q, test_q.
Outputs WORK_DIR/data/split/pool_groups.parquet and WORK_DIR/data/cross_encoder/.
"""
import os, sys, glob
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, polars as pl
import config

W = str(config.WORK_DIR)
OUT = f"{W}/data/cross_encoder"
os.makedirs(OUT, exist_ok=True)
SEED = config.SEED
TAG = os.environ.get("CAND_TAG", "main")

# training files only when the training-pool candidates exist (TRAIN=1 runs); otherwise scoring files for test_q only
TRAINING = os.path.exists(f"{W}/data/texts/pool_q.parquet") and any(f.startswith("pairs.pool_q.") for f in os.listdir(f"{W}/data/cand/{TAG}"))
SETS = ("pool_q", "test_q") if TRAINING else ("test_q",)

# lookalike groups of the unmatched pool records: look_mt (matcher) / look_v (validation); look_mt -> look_ce / look_mt2
if TRAINING:
    tq = pl.read_parquet(f"{W}/data/texts/pool_q.parquet", columns=["id", "s1_split", "text_full"]).with_row_index("qi")
    rng = np.random.default_rng(SEED)
    g = np.where(rng.random(tq.height) < 0.5, "look_mt", "look_v")
    tq = tq.with_columns(pl.when(pl.col("s1_split").is_null()).then(pl.Series(g)).otherwise(pl.col("s1_split")).alias("grp"))
    rng2 = np.random.default_rng(SEED + 5)
    sub = np.where(rng2.random(tq.height) < 0.5, "look_ce", "look_mt2")
    tq = tq.with_columns(pl.when(pl.col("grp") == "look_mt").then(pl.Series(sub)).otherwise(pl.col("grp")).alias("grp2"))
    tq.select("qi", "grp", "grp2").write_parquet(f"{W}/data/split/pool_groups.parquet")
    print(dict(tq.group_by("grp2").len().iter_rows()))

    s1t = pl.read_parquet(f"{W}/data/texts/s1_train.parquet", columns=["text_full"]).with_row_index("di")
    pp = pl.concat([pl.read_parquet(f, columns=["qi", "di"]) for f in sorted(glob.glob(f"{W}/data/cand/{TAG}/pairs.pool_q.*.parquet"))])
    look = pp.join(tq.filter(pl.col("grp2") == "look_ce").select("qi", pl.col("text_full").alias("q")), on="qi").join(
        s1t.select("di", pl.col("text_full").alias("p")), on="di")
    look = look.sample(min(look.height, 1_000_000), seed=SEED).select("q", "p", pl.lit(0).alias("label"))
    look.write_parquet(f"{OUT}/train_lookalike_negatives.parquet")
    print(f"lookalike negatives for the cross-encoder: {look.height:,}")

# scoring texts + pairs
for it in SETS:
    fs = sorted(f for f in os.listdir(f"{W}/data/cand/{TAG}") if f.startswith(f"pairs.{it}."))
    P = pl.concat([pl.read_parquet(f"{W}/data/cand/{TAG}/{f}", columns=["qi", "di"]) for f in fs])
    P.write_parquet(f"{OUT}/pairs_{it}.parquet")
    pl.read_parquet(f"{W}/data/texts/{it}.parquet", columns=["text_full"]).write_parquet(f"{OUT}/text_{it}.parquet")
    print(f"{it}: {P.height:,} pairs")
for dit in (("s1_train", "s1_test") if TRAINING else ("s1_test",)):
    pl.read_parquet(f"{W}/data/texts/{dit}.parquet", columns=["text_full"]).write_parquet(f"{OUT}/text_{dit}.parquet")
