"""Texts for ENC-split matched records (listwise reranker training lists): 2.0M sampled, seed 42. Training data only.
Output WORK_DIR/data/texts/enc_q.parquet."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import polars as pl
import config

W = str(config.WORK_DIR)
sp = pl.read_parquet(f"{W}/data/split/s1_split.parquet").filter(pl.col("split") == "ENC").select(pl.col("entity_id").alias("s1"), "country")
pairs = pl.read_parquet(f"{W}/data/pq/train_pairs.parquet")
e = pairs.join(sp, on="s1").sample(2_000_000, seed=42)
n23 = pl.concat([pl.read_parquet(f"{W}/data/norm/train_s{i}.parquet").select("entity_id", "name_latin", "addr_clean") for i in (2, 3)])
e = e.join(n23, left_on="m", right_on="entity_id").rename({"m": "id"})
p = "query: "
e = e.with_columns((pl.lit(p) + pl.col("name_latin") + " | " + pl.col("addr_clean")).alias("text_full"),
                   (pl.lit(p) + pl.col("name_latin")).alias("text_name"), (pl.lit(p) + pl.col("addr_clean")).alias("text_addr"))
e.select("id", "country", "text_full", "text_name", "text_addr", "src", "s1").write_parquet(f"{W}/data/texts/enc_q.parquet")
print(e.height, e.group_by("country").len().rows())
