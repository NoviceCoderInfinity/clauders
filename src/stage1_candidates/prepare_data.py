"""Raw competition TSVs -> Parquet; explode the training ground truth into (Source 1, matched record) pairs; check row counts.

Outputs WORK_DIR/data/pq/: {train,test}_s{1,2,3}.parquet (entity_id, business_name, business_address, country),
train_pairs.parquet (s1, m, src).
Only the training files are checked against known row counts; the test files are converted as they are and their counts
are printed, since the pipeline must not depend on what the test set contains.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import polars as pl
import config

RAW = str(config.DATA_DIR)
PQ = f"{config.WORK_DIR}/data/pq"
os.makedirs(PQ, exist_ok=True)

EXPECTED = {"train_s1": 2_206_821, "train_s2": 5_034_616, "train_s3": 5_285_603}   # training files only
EXPECTED_COUNTRY = {"train_s1": {"US": 1_323_633, "India": 883_188}}
EXPECTED_PAIRS = 7_638_365


def read_tsv(path):
    df = pl.read_csv(path, separator="\t", quote_char=None, infer_schema=False)
    # empty string -> null
    return df.with_columns([pl.when(pl.col(c) == "").then(None).otherwise(pl.col(c)).alias(c) for c in df.columns])


ok = True
for split in ("train", "test"):
    for s in (1, 2, 3):
        key = f"{split}_s{s}"
        df = read_tsv(f"{RAW}/{split}/{split}_source{s}.tsv")
        assert df.columns == ["entity_id", "business_name", "business_address", "country"], df.columns
        assert df["entity_id"].n_unique() == df.height, f"{key}: duplicate entity_id"
        df.write_parquet(f"{PQ}/{key}.parquet")
        byc = dict(df.group_by("country").len().iter_rows())
        line = f"{key}: {df.height:,}"
        if key in EXPECTED:
            good = df.height == EXPECTED[key]
            ok &= good
            line += " OK" if good else " MISMATCH (exp %d)" % EXPECTED[key]
        line += f"  countries={byc}  null_addr={df['business_address'].null_count():,} null_name={df['business_name'].null_count():,}"
        if key in EXPECTED_COUNTRY:
            cgood = all(byc.get(c) == n for c, n in EXPECTED_COUNTRY[key].items())
            ok &= cgood
            line += "  country-counts " + ("OK" if cgood else "MISMATCH")
        print(line, flush=True)

gt = read_tsv(f"{RAW}/train/train_ground_truth.tsv")
print("ground truth columns", gt.columns, "rows", gt.height)
pairs = (gt.select(pl.col("source1_entity_id").alias("s1"),
                   pl.col("matched_entity_ids").str.split(",").alias("m"))
           .explode("m")
           .with_columns(pl.col("m").str.strip_chars())
           .filter(pl.col("m").is_not_null() & (pl.col("m") != "")))
n_empty = gt.filter(pl.col("matched_entity_ids").is_null()).height
good = pairs.height == EXPECTED_PAIRS
ok &= good
uniq = pairs["m"].n_unique() == pairs.height
ok &= uniq
print(f"train pairs: {pairs.height:,} {'OK' if good else 'MISMATCH'}; m unique: {uniq}; Source 1 with no match: {n_empty:,} ({n_empty/gt.height:.2%})")

# every m exists in train Source 2/3, every s1 in train Source 1
s1 = pl.read_parquet(f"{PQ}/train_s1.parquet", columns=["entity_id"])
s23 = pl.concat([pl.read_parquet(f"{PQ}/train_s{i}.parquet", columns=["entity_id"]).with_columns(pl.lit(i).alias("src")) for i in (2, 3)])
pairs = pairs.join(s23.rename({"entity_id": "m"}), on="m", how="left")
print("pairs with m not found in Source 2/3:", pairs["src"].null_count(), "| s1 not in Source 1:", pairs.join(s1.rename({"entity_id": "s1"}), on="s1", how="anti").height)
print("pairs by src:", dict(pairs.group_by("src").len().iter_rows()))
print("ground-truth rows == train Source 1 rows:", gt.height == EXPECTED["train_s1"])
pairs.select("s1", "m", "src").write_parquet(f"{PQ}/train_pairs.parquet")
print("ALL CHECKS", "PASS" if ok else "FAIL")
