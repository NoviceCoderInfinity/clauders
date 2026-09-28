"""Uncertainty zone for US and India (CPU).

A Source 2 / Source 3 record is in the zone when its top cross-encoder candidate (ce2 rank 1) has a no-match-normalised
probability pn1 in [0.001, 0.999), or pn1 >= 0.3 with pn1 - pn2 < 0.5 (pn2 = the second candidate). Only zone records are
re-decided by the judges and the combiners; the stage-2 matcher decides every other record.

usage: python -m stage3_judges.zone
Reads  WORK_DIR/stage2/test_candidates.parquet, WORK_DIR/stage2/first_stage/test_scores.parquet (matcher p)
Writes WORK_DIR/stage3/test_zone_records.parquet (qi) and WORK_DIR/stage3/test_zone_candidates.parquet (every candidate of a
zone record with the cross-encoder columns and the matcher probability p).
"""
import polars as pl

import config
from common.country_route import UNLABELLED, with_branch


def zone_records(X):
    """X: candidates with qi, ce2_rank, ce2_pnull. Returns the zone records (qi)."""
    t1 = X.filter(pl.col("ce2_rank") == 1).select("qi", pl.col("ce2_pnull").alias("pn1"))
    t2 = X.filter(pl.col("ce2_rank") == 2).select("qi", pl.col("ce2_pnull").alias("pn2"))
    T = t1.join(t2, on="qi", how="left").with_columns(pl.col("pn2").fill_null(0.0))
    return T.filter(((pl.col("pn1") >= 0.001) & (pl.col("pn1") < 0.999)) | ((pl.col("pn1") >= 0.3) & (pl.col("pn1") - pl.col("pn2") < 0.5))).select("qi")


if __name__ == "__main__":
    d2 = config.WORK_DIR / "stage2"
    C = with_branch(pl.read_parquet(d2 / "test_candidates.parquet"), config.TEST_DIR)
    C = C.filter(pl.col("branch") != UNLABELLED).drop("matcher_accept", "branch")
    zq = zone_records(C)
    d3 = config.WORK_DIR / "stage3"; d3.mkdir(parents=True, exist_ok=True)
    zq.write_parquet(d3 / "test_zone_records.parquet")
    p = pl.read_parquet(d2 / "first_stage" / "test_scores.parquet", columns=["qi", "di", "p"])
    Z = C.join(zq, on="qi", how="semi").join(p, on=["qi", "di"], how="left")
    Z.write_parquet(d3 / "test_zone_candidates.parquet")
    print(f"zone records {zq.height:,}; zone candidates {Z.height:,}; missing p {Z['p'].null_count():,}", flush=True)
