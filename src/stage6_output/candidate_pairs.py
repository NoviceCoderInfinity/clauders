"""Write output/candidate_pairs.tsv: exactly the pairs the matching models run inference on.

Blocking (stage 1) gives each Source 2 / Source 3 record a list of Source 1 candidates of the same country (dense top-20
within 0.15 cosine of the best, merge_candidates.py). The stage-2 matchers score every pair of these lists, and every later
decision (uncertainty zone, LLM judges, combiner) chooses among them, so the candidate set is:
  - countries with training labels: every row of WORK_DIR/stage2/test_candidates.parquet (scored by the cross-encoders, the
    listwise rerankers and matcher_us_india);
  - countries without training labels (branch UNLABELLED): every pair scored by the unseen-country matcher
    (WORK_DIR/stage2/first_stage/test_scores.parquet; their candidates come from the normalised texts of france_canon.py).
Nothing is added from the decisions. The script checks that every decided pair is a candidate and stops otherwise.

usage: python -m stage6_output.candidate_pairs [--decisions a.parquet,b.parquet] [--out OUTPUT_DIR/candidate_pairs.tsv]
"""
import argparse
from pathlib import Path

import polars as pl

import config
from common.country_route import UNLABELLED, labelled_branches, with_branch
from common.io import read_tsv, write_tsv


def scored_pairs(stage2_dir, unlabelled_scores):
    """(source1_entity_id, entity_id) of every pair a matcher scored."""
    qm = pl.read_parquet(f"{stage2_dir}/test_record_index.parquet").select("qi", "entity_id")
    dm = with_branch(pl.read_parquet(f"{stage2_dir}/test_source1_index.parquet")).select(
        "di", pl.col("entity_id").alias("source1_entity_id"), "branch")
    lab = pl.scan_parquet(f"{stage2_dir}/test_candidates.parquet").select("qi", "di").collect() \
        .join(dm.filter(pl.col("branch").is_in(labelled_branches())).select("di"), on="di", how="semi")
    unl = pl.read_parquet(unlabelled_scores, columns=["qi", "di"]).join(dm.filter(pl.col("branch") == UNLABELLED).select("di"), on="di", how="semi")
    return pl.concat([lab, unl]).unique().join(qm, on="qi").join(dm, on="di").select("source1_entity_id", "entity_id")


def main():
    w = config.WORK_DIR
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage2-dir", default=str(w / "stage2"))
    ap.add_argument("--unlabelled-scores", default=str(w / "stage2" / "first_stage" / "test_scores.parquet"),
                    help="qi, di, ... of every pair the first-stage matchers scored (the UNLABELLED-branch rows are used)")
    ap.add_argument("--decisions", default=",".join(str(w / p) for p in ("stage4/us_india_pairs.parquet", "stage5/unlabelled_pairs.parquet")))
    ap.add_argument("--test-dir", default=str(config.TEST_DIR))
    ap.add_argument("--out", default=str(config.OUTPUT_DIR / "candidate_pairs.tsv"))
    a = ap.parse_args()
    key = ["source1_entity_id", "entity_id"]
    pairs = scored_pairs(a.stage2_dir, a.unlabelled_scores)
    d = pl.concat([pl.read_parquet(p).select(key) for p in a.decisions.split(",")]).unique()
    missing = d.join(pairs, on=key, how="anti").height
    assert missing == 0, f"{missing} decided pairs are not in the scored candidate set"
    s1 = read_tsv(f"{a.test_dir}/test_source1.tsv", columns=["entity_id", "country"]).rename({"entity_id": "source1_entity_id"})
    g = pairs.group_by("source1_entity_id").agg(pl.col("entity_id").sort().str.join(",").alias("candidate_entity_ids"))
    out = s1.select("source1_entity_id").join(g, on="source1_entity_id", how="left").with_columns(pl.col("candidate_entity_ids").fill_null(""))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    write_tsv(out, a.out)
    per = pairs.join(s1, on="source1_entity_id").group_by("country").len().sort("country")
    print(f"{pairs.height:,} candidate pairs ({pairs.height / out.height:.2f} per Source 1 entity) by country {dict(per.iter_rows())}; "
          f"all {d.height:,} decided pairs are candidates -> {a.out}")


if __name__ == "__main__":
    main()
