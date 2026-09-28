"""Retriever training data.

WORK_DIR/data/encoder_train/enc_pairs.parquet: every true pair whose Source 1 is in the ENC split
   s1, s1_row (row in s1_train texts), country, grp (country|state of the Source 1, for same-area batches),
   q_full, q_name, q_addr, p (passage text), src
Also copies texts/s1_train.parquet and texts/v_q.parquet for evaluation and hard-negative mining.
"""
import os, sys, json, shutil
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import polars as pl
import config

W = str(config.WORK_DIR)
OUT = f"{W}/data/encoder_train"
os.makedirs(OUT, exist_ok=True)

sp = pl.read_parquet(f"{W}/data/split/s1_split.parquet")
pairs = pl.read_parquet(f"{W}/data/pq/train_pairs.parquet").join(
    sp.filter(pl.col("split") == "ENC").select(pl.col("entity_id").alias("s1")), on="s1")
s1t = pl.read_parquet(f"{W}/data/texts/s1_train.parquet").with_row_index("s1_row")
n23 = pl.concat([pl.read_parquet(f"{W}/data/norm/train_s{i}.parquet") for i in (2, 3)])

# state of each Source 1 = first address component that is a known state form (targets of the learned state table)
smap = json.load(open(f"{W}/data/state_map.json"))
states = {v for k, v in smap.items() if k.split("|", 1)[1] != v}
n1 = pl.read_parquet(f"{W}/data/norm/train_s1.parquet", columns=["entity_id", "country", "addr_clean"])
grp = [c + "|" + next((x for x in a.split(", ") if x in states), "") for c, a in zip(n1["country"], n1["addr_clean"])]
n1 = n1.with_columns(pl.Series("grp", grp)).select(pl.col("entity_id").alias("s1"), "grp")

d = (pairs.join(n23.select(pl.col("entity_id").alias("m"), "name_latin", "addr_clean"), on="m")
     .join(s1t.select(pl.col("id").alias("s1"), "s1_row", "country", pl.col("text_full").alias("p")), on="s1")
     .join(n1, on="s1")
     .select("s1", "s1_row", "country", "grp",
             ("query: " + pl.col("name_latin") + " | " + pl.col("addr_clean")).alias("q_full"),
             ("query: " + pl.col("name_latin")).alias("q_name"),
             ("query: " + pl.col("addr_clean")).alias("q_addr"), "p", "src"))
d.write_parquet(f"{OUT}/enc_pairs.parquet", compression="zstd")
print(f"enc_pairs: {d.height:,} pairs, {d['s1'].n_unique():,} Source 1; groups: {d['grp'].n_unique()}")
for f in ("s1_train", "v_q"):
    shutil.copy(f"{W}/data/texts/{f}.parquet", f"{OUT}/{f}.parquet")
