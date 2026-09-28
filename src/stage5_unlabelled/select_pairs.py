"""Decision set for the countries without training labels (branch common/country_route.UNLABELLED; on this test set: France).

These countries have no labels to train on, so their pairs are decided by the unseen-country matcher of stage 2
(lightgbm/matcher_france, trained on the US and India training labels with country-agnostic features: string and retrieval
features plus the core-name reranker, no e5-small cross-encoder), applied to their normalised candidates
(stage1_candidates/france_canon.py). Rule (predict_first_stage.py): each record keeps its best Source 1 candidate when
p >= 0.60; caps 5 Source 2 / 6 Source 3 per Source 1 entity. No decision rule, word list or statistic is fitted on test records here.

usage: python -m stage5_unlabelled.select_pairs [--accepted WORK_DIR/stage2/matcher_accepted_pairs.parquet] [--out ...]
Writes (source1_entity_id, entity_id) for every accepted pair of the UNLABELLED branch.
"""
import argparse
from pathlib import Path

import polars as pl

import config
from common.country_route import UNLABELLED, with_branch


def main():
    w = config.WORK_DIR
    ap = argparse.ArgumentParser()
    ap.add_argument("--accepted", default=str(w / "stage2" / "matcher_accepted_pairs.parquet"),
                    help="first-stage accepted pairs from candidate_table.py: (s1, v, ...) or (qi, di, ...)")
    ap.add_argument("--stage2-dir", default=str(w / "stage2"), help="index maps, used when --accepted has only qi / di")
    ap.add_argument("--out", default=str(w / "stage5" / "unlabelled_pairs.parquet"))
    a = ap.parse_args()
    acc = pl.read_parquet(a.accepted)
    if "s1" not in acc.columns:
        qm = pl.read_parquet(f"{a.stage2_dir}/test_record_index.parquet").select("qi", pl.col("entity_id").alias("v"))
        dm = pl.read_parquet(f"{a.stage2_dir}/test_source1_index.parquet").select("di", pl.col("entity_id").alias("s1"))
        acc = acc.join(qm, on="qi").join(dm, on="di")
    acc = with_branch(acc)
    out = acc.filter(pl.col("branch") == UNLABELLED).select(pl.col("s1").alias("source1_entity_id"), pl.col("v").alias("entity_id"))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    out.write_parquet(a.out)
    print(f"unlabelled-branch pairs {out.height:,} for {out['source1_entity_id'].n_unique():,} Source 1 entities -> {a.out}")


if __name__ == "__main__":
    main()
