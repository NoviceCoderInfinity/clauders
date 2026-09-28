"""Write output/matching_results.tsv from the decision sets of both branches.

Inputs: the pairs of the countries with training labels (stage 4) and of the countries without (stage 5). Checks: every
record matched at most once, at most 5 Source 2 and 6 Source 3 matches (11 in all) per Source 1 entity, one row per test
Source 1 entity (empty when nothing is matched). Rows follow test_source1.tsv; ids in a row are sorted.

usage: python -m stage6_output.assemble [--labelled ...] [--unlabelled ...] [--out OUTPUT_DIR/matching_results.tsv]
"""
import argparse
from pathlib import Path

import polars as pl

import config
from common.country_route import UNLABELLED, labelled_branches, with_branch
from common.io import read_tsv, write_tsv

CAP = {"S2": 5, "S3": 6}
MAX_MATCHES = 11


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labelled", default=str(config.WORK_DIR / "stage4" / "us_india_pairs.parquet"))
    ap.add_argument("--unlabelled", default=str(config.WORK_DIR / "stage5" / "unlabelled_pairs.parquet"))
    ap.add_argument("--test-dir", default=str(config.TEST_DIR))
    ap.add_argument("--out", default=str(config.OUTPUT_DIR / "matching_results.tsv"))
    a = ap.parse_args()
    key = ["source1_entity_id", "entity_id"]
    s1 = with_branch(read_tsv(f"{a.test_dir}/test_source1.tsv", columns=["entity_id", "country"]).rename({"entity_id": "source1_entity_id"}))
    lab = pl.read_parquet(a.labelled).select(key).join(s1.filter(pl.col("branch").is_in(labelled_branches())), on="source1_entity_id", how="semi")
    unl = pl.read_parquet(a.unlabelled).select(key).join(s1.filter(pl.col("branch") == UNLABELLED), on="source1_entity_id", how="semi")

    d = pl.concat([lab, unl]).unique()
    assert d["entity_id"].n_unique() == d.height, "a record is matched to two Source 1 entities"
    per_src = d.with_columns(pl.col("entity_id").str.slice(0, 2).alias("src")).group_by("source1_entity_id", "src").len()
    assert per_src.filter(pl.col("len") > pl.col("src").replace_strict(CAP, return_dtype=pl.Int64)).height == 0, "per-source cap"
    assert d.height == 0 or d.group_by("source1_entity_id").len()["len"].max() <= MAX_MATCHES

    g = d.group_by("source1_entity_id").agg(pl.col("entity_id").sort().str.join(",").alias("matched_entity_ids"))
    out = s1.select("source1_entity_id").join(g, on="source1_entity_id", how="left").with_columns(pl.col("matched_entity_ids").fill_null(""))
    assert out.height == s1.height
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    write_tsv(out, a.out)
    cnt = d.join(s1, on="source1_entity_id").group_by("country").len().sort("country")
    print(f"pairs by country {dict(cnt.iter_rows())}, total {d.height:,}; {out.height:,} rows -> {a.out}")


if __name__ == "__main__":
    main()
