"""Test-side tables consumed by the later stages (all written to WORK_DIR/stage2/):

  test_record_index.parquet     qi, entity_id, src, country     qi = row in concat(test_source2, test_source3)
  test_source1_index.parquet    di, entity_id, country          di = row in test_source1
  matcher_accepted_pairs.parquet  s1, v, di, country, qi        pairs accepted by the first-stage matcher (matching_results.tsv)
  test_candidates.parquet       every test candidate pair (row order = listwise reranker test scores):
      qi, di, ce (e5-small cross-encoder), ce2 (listwise reranker), ce3 (name specialist), ce4 (address specialist),
      <score>_rank / _gap (to the record's best) / _pnull (softmax over the record's candidates + null logit 0),
      src, country, matcher_accept (1 = accepted by the first-stage matcher), ncand

usage: python candidate_table.py --first-stage <dir with matching_results.tsv> --listwise <test_ce.parquet> --name <test_ce.parquet>
                                 --address <test_ce.parquet> --cross-encoder <test_ce.parquet>
"""
import os, sys, argparse, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import polars as pl
import config

ap = argparse.ArgumentParser()
ap.add_argument("--first-stage", required=True); ap.add_argument("--listwise", required=True); ap.add_argument("--name", required=True)
ap.add_argument("--address", required=True); ap.add_argument("--cross-encoder", required=True)
ap.add_argument("--out", default=str(config.WORK_DIR / "stage2"))
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); T0 = time.time()
RAW = config.TEST_DIR


def rd(p):
    return pl.read_csv(p, separator="\t", quote_char=None, infer_schema=False, missing_utf8_is_empty_string=True)


# ---- index maps
t1 = rd(f"{RAW}/test_source1.tsv"); t2 = rd(f"{RAW}/test_source2.tsv"); t3 = rd(f"{RAW}/test_source3.tsv")
q = pl.concat([t2.with_columns(pl.lit(2).alias("src")), t3.with_columns(pl.lit(3).alias("src"))]).with_row_index("qi").with_columns(pl.col("qi").cast(pl.Int32))
d = t1.with_row_index("di").with_columns(pl.col("di").cast(pl.Int32))
q.select("qi", "entity_id", "src", "country").write_parquet(f"{a.out}/test_record_index.parquet")
d.select("di", "entity_id", "country").write_parquet(f"{a.out}/test_source1_index.parquet")
print("rows: records", q.height, "Source 1", d.height, flush=True)

# ---- first-stage accepted pairs
m = pl.read_csv(f"{a.first_stage}/matching_results.tsv", separator="\t", infer_schema=False, quote_char=None, missing_utf8_is_empty_string=True)
m = m.with_columns(pl.col("matched_entity_ids").str.split(",")).explode("matched_entity_ids").filter(pl.col("matched_entity_ids") != "")
m = m.select(pl.col("source1_entity_id").alias("s1"), pl.col("matched_entity_ids").alias("v"))
m = m.join(d.select(pl.col("entity_id").alias("s1"), "di", "country"), on="s1").join(q.select(pl.col("entity_id").alias("v"), "qi"), on="v")
m.write_parquet(f"{a.out}/matcher_accepted_pairs.parquet")
print("first-stage accepted pairs", m.height, flush=True)

# ---- candidate table
X = pl.read_parquet(a.listwise).rename({"ce": "ce2"})
for nm, path in (("ce3", a.name), ("ce4", a.address), ("ce", a.cross_encoder)):
    c = pl.read_parquet(path)
    assert (c["qi"] == X["qi"]).all() and (c["di"] == X["di"]).all()
    X = X.with_columns(c["ce"].cast(pl.Float32).alias(nm))
for nm in ("ce", "ce2", "ce3", "ce4"):
    X = X.with_columns(pl.col(nm).rank("ordinal", descending=True).over("qi").cast(pl.Int16).alias(f"{nm}_rank"),
                       (pl.col(nm).max().over("qi") - pl.col(nm)).alias(f"{nm}_gap"),
                       (pl.col(nm).exp() / (pl.col(nm).exp().sum().over("qi") + 1.0)).alias(f"{nm}_pnull"))
X = X.join(q.select("qi", "src", "country"), on="qi", how="left")
S = m.select("qi", "di", pl.lit(1, pl.Int8).alias("matcher_accept"))
X = X.join(S, on=["qi", "di"], how="left").with_columns(pl.col("matcher_accept").fill_null(0))
X = X.with_columns(pl.len().over("qi").alias("ncand"))
print(X.group_by("country").agg(pl.len(), pl.col("matcher_accept").sum(), pl.col("qi").n_unique().alias("recs")))
X.write_parquet(f"{a.out}/test_candidates.parquet")
print("done", round(time.time() - T0))
