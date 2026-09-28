"""Encoder input texts (one parquet per item set) under WORK_DIR/data/texts/.

default run:
  s1_train  : all train Source 1 (passages)
  s1_test   : all test Source 1 (passages)
  v_q       : HOLD matched records (queries), with true s1 (retriever evaluation)
POOL=1 run:
  pool_q    : training pool (HOLD + MT matched records + all unmatched records), with s1 / s1_split
  test_q    : all test Source 2 then Source 3 records (queries)
Columns: id, country, text_full ("<prefix>name_latin | addr_clean"), text_name, text_addr, (src, s1)
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import polars as pl
import config

W = str(config.WORK_DIR)
OUT = f"{W}/data/texts"
os.makedirs(OUT, exist_ok=True)


def texts(df, kind):
    p = "passage: " if kind == "p" else "query: "
    return df.with_columns(
        (pl.lit(p) + pl.col("name_latin") + " | " + pl.col("addr_clean")).alias("text_full"),
        (pl.lit(p) + pl.col("name_latin")).alias("text_name"),
        (pl.lit(p) + pl.col("addr_clean")).alias("text_addr"))


cols = ["id", "country", "text_full", "text_name", "text_addr"]
n23 = pl.concat([pl.read_parquet(f"{W}/data/norm/train_s{i}.parquet") for i in (2, 3)])

if not os.environ.get("POOL"):
    for sp in ("train", "test"):
        d = pl.read_parquet(f"{W}/data/norm/{sp}_s1.parquet").rename({"entity_id": "id"})
        texts(d, "p").select(cols).write_parquet(f"{OUT}/s1_{sp}.parquet")

    pool = pl.read_parquet(f"{W}/data/split/pool.parquet").filter(pl.col("s1_split") == "HOLD")
    v = pool.join(n23.select("entity_id", "name_latin", "addr_clean"), on="entity_id").rename({"entity_id": "id"})
    texts(v, "q").select(cols + ["src", "s1"]).write_parquet(f"{OUT}/v_q.parquet")

    for f in ("s1_train.parquet", "s1_test.parquet", "v_q.parquet"):
        d = pl.read_parquet(f"{OUT}/{f}")
        print(f, d.height, dict(d.group_by("country").len().iter_rows()), "| e.g.", d["text_full"][0])
else:
    # full training pool and all test Source 2/3 records, for candidates
    p = pl.read_parquet(f"{W}/data/split/pool.parquet")
    pq = p.join(n23.select("entity_id", "name_latin", "addr_clean"), on="entity_id").rename({"entity_id": "id"})
    texts(pq, "q").select(cols + ["src", "s1", "s1_split"]).write_parquet(f"{OUT}/pool_q.parquet")
    t23 = pl.concat([pl.read_parquet(f"{W}/data/norm/test_s{i}.parquet").with_columns(pl.lit(i).alias("src")) for i in (2, 3)])
    texts(t23.rename({"entity_id": "id"}), "q").select(cols + ["src"]).write_parquet(f"{OUT}/test_q.parquet")
    for f in ("pool_q", "test_q"):
        d = pl.read_parquet(f"{OUT}/{f}.parquet")
        print(f, d.height, dict(d.group_by("country").len().iter_rows()))
