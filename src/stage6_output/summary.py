"""Print the candidate-set table of the methodology document (Appendix B) from the two output files.

usage: python -m stage6_output.summary [--output OUTPUT_DIR] [--test-dir DATA_DIR/test]
"""
import argparse

import polars as pl

import config
from common.io import explode_ids, read_tsv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default=str(config.OUTPUT_DIR))
    ap.add_argument("--test-dir", default=str(config.TEST_DIR))
    a = ap.parse_args()
    s1 = read_tsv(f"{a.test_dir}/test_source1.tsv", columns=["entity_id", "country"]).rename({"entity_id": "source1_entity_id"})
    cand = explode_ids(read_tsv(f"{a.output}/candidate_pairs.tsv"), "source1_entity_id", "candidate_entity_ids").join(s1, on="source1_entity_id")
    match = explode_ids(read_tsv(f"{a.output}/matching_results.tsv"), "source1_entity_id", "matched_entity_ids").join(s1, on="source1_entity_id")
    n1 = dict(s1.group_by("country").len().iter_rows()); nc = dict(cand.group_by("country").len().iter_rows())
    nm = dict(match.group_by("country").len().iter_rows())
    print("| Country | Source 1 entities | Candidate pairs | Per entity | Matched pairs |\n|---|---|---|---|---|")
    for c in sorted(n1, key=lambda c: -nc.get(c, 0)):
        print(f"| {c} | {n1[c]:,} | {nc.get(c, 0):,} | {nc.get(c, 0) / n1[c]:.2f} | {nm.get(c, 0):,} |")
    T1, TC, TM = sum(n1.values()), sum(nc.values()), sum(nm.values())
    print(f"| **Total** | **{T1:,}** | **{TC:,}** | **{TC / T1:.2f}** | **{TM:,}** |")


if __name__ == "__main__":
    main()
