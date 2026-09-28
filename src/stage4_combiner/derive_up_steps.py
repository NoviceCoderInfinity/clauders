"""Derive the lookalike house-number offsets (features.UP_STEPS) from the TRAINING labels of the hold-out split.

Lookalike records copy a real business on the same street and move the house number up by a fixed offset. The offsets are
measured on labelled training pairs only (the hold-out Source 1 entities of the training data, WORK_DIR/stage4/hold):
for every hold-out zone candidate whose two addresses carry a house number on the same street (features._street_1, street-word similarity >= 0.75 as in the same_st feature), count
the pairs and the true matches per step d = record number - Source 1 number. A step d in 1..MAX_STEP is a lookalike offset
when it is frequent (at least MIN_PAIRS pairs) and rarely a true match (match rate below MAX_MATCH_RATE, against about 65%
for an unchanged number). No test record is read.

usage: python -m stage4_combiner.derive_up_steps [--hold-dir WORK_DIR/stage4/hold] [--check]
  --check  exit with an error when the derived set differs from features.UP_STEPS
"""
import argparse
import collections
import sys

import polars as pl
from rapidfuzz.distance import Indel

import config
from stage4_combiner import features as FT

MAX_STEP = 50
MIN_PAIRS = 1000
MAX_MATCH_RATE = 0.25


def step_table(hold_dir):
    """(step, n_pairs, match_rate) over same-street hold-out zone candidates, steps -MAX_STEP..MAX_STEP."""
    texts = pl.read_parquet(f"{hold_dir}/hold_zone_texts.parquet")
    labels = pl.read_parquet(f"{hold_dir}/hold_cross_encoder.parquet", columns=["qi", "di", "label"]).unique(["qi", "di"])
    d = texts.join(labels, on=["qi", "di"])
    n, pos = collections.Counter(), collections.Counter()
    for ta, tb, y in zip(d["text_a"].to_list(), d["text_b"].to_list(), d["label"].to_list()):
        _, a = (ta.split(" | ", 1) + [""])[:2]
        _, b = (tb.split(" | ", 1) + [""])[:2]
        b = b.split(" || ")[0]
        h1, s1 = FT._street_1(a)
        hv, sv = FT._street_1(b)
        if h1 is None or hv is None or not s1 or not sv or Indel.normalized_similarity(s1, sv) < 0.75:
            continue
        step = hv - h1
        if abs(step) <= MAX_STEP:
            n[step] += 1
            pos[step] += int(y)
    return pl.DataFrame([(s, n[s], pos[s] / n[s]) for s in sorted(n)], schema=["step", "n_pairs", "match_rate"], orient="row")


def up_steps(table):
    t = table.filter((pl.col("step") >= 1) & (pl.col("n_pairs") >= MIN_PAIRS) & (pl.col("match_rate") < MAX_MATCH_RATE))
    return set(t["step"].to_list())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hold-dir", default=str(config.WORK_DIR / "stage4" / "hold"))
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    table = step_table(a.hold_dir)
    derived = up_steps(table)
    same = table.filter(pl.col("step") == 0)
    print(f"same number: {same['n_pairs'].item():,} pairs, match rate {same['match_rate'].item():.3f}")
    print(table.filter((pl.col("step") >= 1) & (pl.col("n_pairs") >= MIN_PAIRS)))
    print(f"derived lookalike offsets {sorted(derived)}; features.UP_STEPS {sorted(FT.UP_STEPS)}")
    if a.check and derived != FT.UP_STEPS:
        sys.exit("the offsets derived from the training labels differ from features.UP_STEPS")


if __name__ == "__main__":
    main()
