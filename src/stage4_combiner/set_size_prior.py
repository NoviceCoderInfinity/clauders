"""Training histogram H of (Source 2 matches, Source 3 matches) per Source 1 entity, used by the set-size prior.

H[n2, n3] = (number of training Source 1 entities with n2 Source 2 and n3 Source 3 matches + 0.5) / total, counts capped at 15.
Every training Source 1 entity is counted, including entities without matches.
usage: python -m stage4_combiner.set_size_prior [--out WORK_DIR/stage4/set_size_prior.npy]
"""
import argparse
import numpy as np
import polars as pl

import config
from common.io import read_tsv, explode_ids


def build(train_dir=config.TRAIN_DIR):
    s1 = read_tsv(train_dir / "train_source1.tsv", columns=["entity_id"])
    gt = explode_ids(read_tsv(train_dir / "train_ground_truth.tsv"), "source1_entity_id", "matched_entity_ids")
    cnt = gt.with_columns(pl.col("entity_id").str.starts_with("S2").alias("is2")).group_by("source1_entity_id") \
        .agg(n2=pl.col("is2").sum(), n3=(~pl.col("is2")).sum())
    cnt = s1.rename({"entity_id": "source1_entity_id"}).join(cnt, on="source1_entity_id", how="left").fill_null(0)
    H = np.zeros((16, 16))
    for n2, n3, c in cnt.group_by("n2", "n3").len().iter_rows():
        H[min(n2, 15), min(n3, 15)] += c
    return (H + 0.5) / H.sum()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(config.WORK_DIR / "stage4" / "set_size_prior.npy"))
    a = ap.parse_args()
    H = build()
    np.save(a.out, H)
    print("wrote", a.out, H.shape)
