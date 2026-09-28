"""Verify that the shipped judge training sets and hold-out tables contain only TRAINING records with their training labels.

The judge training sets (<dir>/stage3/train/{plain,context}.parquet, built by build_training_sets.py), the hold-out judge inputs (stage3/inputs/hold_*) and
the hold-out tables of the combiner (stage4/hold/, built by stage4_combiner/build_hold_tables.py) come from the training split only. This
script checks, from the competition files alone:
  1. every Source 1 id is in train_source1.tsv and every record id in train_source2/3.tsv (no test id anywhere);
  2. every label equals the ground truth (1 exactly when the record is in the Source 1 entity's train_ground_truth.tsv list);
  3. every text equals the training record's "business_name | business_address" text (stage3_judges.texts.text);
  4. the context set holds the same pairs as the plain set, with the record's other candidates appended to text_b.
Exits with an error when a check fails.

usage: python -m stage3_judges.check_training_sets --stored WORK_DIR   (run_stage3.sh runs it with TRAIN=1)
"""
import argparse
import os
import sys

import polars as pl

import config
from common.io import explode_ids, read_tsv
from stage3_judges.texts import text


def training_tables(train_dir):
    rd = lambda f: read_tsv(f"{train_dir}/{f}", columns=["entity_id", "business_name", "business_address"])
    txt = lambda d: d.select("entity_id", text(pl.col("business_name"), pl.col("business_address")).alias("text"))
    s1 = txt(rd("train_source1.tsv"))
    rec = txt(pl.concat([rd("train_source2.tsv"), rd("train_source3.tsv")]))
    gt = explode_ids(read_tsv(f"{train_dir}/train_ground_truth.tsv"), "source1_entity_id", "matched_entity_ids") \
        .select(pl.col("source1_entity_id").alias("s1_id"), pl.col("entity_id").alias("cand_id"), pl.lit(1, pl.Int8).alias("truth"))
    return s1, rec, gt


def check_pairs(name, d, s1, rec, gt, ok):
    """d: s1_id, cand_id, label and optionally text_a / text_b."""
    bad_s1 = d.join(s1, left_on="s1_id", right_on="entity_id", how="anti").height
    bad_rec = d.join(rec, left_on="cand_id", right_on="entity_id", how="anti").height
    lab = d.join(gt, on=["s1_id", "cand_id"], how="left").with_columns(pl.col("truth").fill_null(0))
    bad_lab = lab.filter(pl.col("label") != pl.col("truth")).height
    bad_txt = 0
    if "text_a" in d.columns:
        t = d.join(s1.rename({"entity_id": "s1_id", "text": "ta"}), on="s1_id").join(rec.rename({"entity_id": "cand_id", "text": "tb"}), on="cand_id")
        bad_txt = t.filter((pl.col("text_a") != pl.col("ta")) | ~pl.col("text_b").str.starts_with(pl.col("tb"))).height
    good = bad_s1 == 0 and bad_rec == 0 and bad_lab == 0 and bad_txt == 0
    print(f"{name}: {d.height:,} rows; non-training Source 1 ids {bad_s1}, non-training record ids {bad_rec}, "
          f"labels differing from the ground truth {bad_lab}, texts differing from the training records {bad_txt} -> {'OK' if good else 'FAIL'}")
    return ok and good


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stored", default=str(config.WORK_DIR), help="folder with stage3/train, stage3/inputs and stage4/hold")
    a = ap.parse_args()
    s1, rec, gt = training_tables(config.TRAIN_DIR)
    ok = True
    sets = {}
    for name in ("plain", "context"):
        f = f"{a.stored}/stage3/train/{name}.parquet"
        if os.path.exists(f):
            sets[name] = pl.read_parquet(f)
            ok = check_pairs(f"stage3/train/{name}", sets[name], s1, rec, gt, ok)
    if len(sets) == 2:
        p, c = sets["plain"], sets["context"]
        same = p.select("s1_id", "cand_id", "label", "split", "text_a").equals(c.select("s1_id", "cand_id", "label", "split", "text_a"))
        ctx = (c["text_b"].str.starts_with(p["text_b"] + " || other candidates: ")).all()
        print(f"context set = plain set + other candidates: {same and ctx}")
        ok = ok and same and ctx
    hold = f"{a.stored}/stage4/hold"
    if os.path.isdir(hold):
        # hold-out tables index Source 1 entities by di (source1_split) and records by qi (record_pool)
        sm = pl.read_parquet(f"{hold}/source1_split.parquet").select("di", pl.col("entity_id").alias("s1_id"))
        pm = pl.read_parquet(f"{hold}/record_pool.parquet").select("qi", pl.col("entity_id").alias("cand_id"))
        ids = lambda d: d.join(sm, on="di", how="left").join(pm, on="qi", how="left")
        # tables without a label column take the ground truth as label, so only their ids and texts are tested
        with_truth = lambda d: d.join(gt.select("s1_id", "cand_id", pl.col("truth").alias("label")), on=["s1_id", "cand_id"], how="left") \
            .with_columns(pl.col("label").fill_null(0))
        for name in ("hold_cross_encoder", "matcher_scores"):
            d = ids(pl.read_parquet(f"{hold}/{name}.parquet", columns=["qi", "di", "label"]))
            ok = check_pairs(f"stage4/hold/{name}", d.select("s1_id", "cand_id", "label"), s1, rec, gt, ok)
        ok = check_pairs("stage4/hold/hold_zone_texts", with_truth(ids(pl.read_parquet(f"{hold}/hold_zone_texts.parquet"))), s1, rec, gt, ok)
        inputs = f"{a.stored}/stage3/inputs"
        for f in sorted(os.listdir(inputs)) if os.path.isdir(inputs) else []:
            d = pl.read_parquet(f"{inputs}/{f}").with_columns(pl.col("s1_id").cast(pl.Int32).alias("di"), pl.col("cand_id").cast(pl.Int32).alias("qi"))
            ok = check_pairs(f"stage3/inputs/{f}", with_truth(ids(d.drop("s1_id", "cand_id"))), s1, rec, gt, ok)
        ok = check_pairs("stage4/hold/judge_train_pairs", pl.read_parquet(f"{hold}/judge_train_pairs.parquet"), s1, rec, gt, ok)
    print("ALL CHECKS", "PASS" if ok else "FAIL")
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
