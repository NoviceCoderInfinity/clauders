"""Build the LoRA judges' training sets from the TRAIN=1 outputs of stages 1-2 (CPU).

Every pair is a (training Source 1 entity, training Source 2 / 3 record) pair with its label from train_ground_truth.tsv.
No hold-out (HOLD) Source 1 entity is used, so the hold-out split stays clean for the combiner and the validation scores.

Pairs (the same kind of pairs the judges score at test time: a record against its best retrieval candidates):
  ENC records   data/texts/enc_q.parquet (matched records of ENC-split entities, make_list_queries.py) with their e5 retrieval
                top 20 (data/cand/<tag>/dense_text_full.enc_q.<country>.npz): the first TOP_K candidates within 0.15 cosine of
                the record's best
  MT records    data/texts/pool_q.parquet records of the matcher-training groups MT and look_mt2 (data/split/pool_groups.parquet,
                build_cross_encoder_data.py) with their merged candidates (data/cand/<tag>/pairs.pool_q.<country>.parquet):
                the TOP_K first by retrieval rank
label = 1 when the candidate is the record's true Source 1 entity. Pairs whose Source 1 entity is in the HOLD split are dropped.
The pairs are deduplicated, shuffled (seed 0) and cut to SIZE rows; VAL_PER_MILLE of the Source 1 entities (by hash) form the
validation split that train_lora.py reports on.

Texts: text_a = "business_name | business_address" of the Source 1 entity, text_b = the record's (stage3_judges.texts.text).
  plain.parquet    those texts
  context.parquet  the same rows, text_b + " || other candidates: ..." (build_inputs.context: the record's other rows)
Columns: s1_id, cand_id, label, split (train | val), text_a, text_b. Written to WORK_DIR/stage3/train/.

usage: python -m stage3_judges.build_training_sets [--size 1093841] [--top-k 3]
"""
import argparse
import glob

import numpy as np
import polars as pl

import config
from common.io import read_tsv
from stage3_judges.build_inputs import context
from stage3_judges.texts import text

W = config.WORK_DIR
DELTA = 0.15
VAL_PER_MILLE = 27


def enc_pairs(tag, top_k):
    """(di, rid, true s1) for the ENC records' top retrieval candidates."""
    eq = pl.read_parquet(W / "data" / "texts" / "enc_q.parquet", columns=["id", "s1"]).with_row_index("qi")
    rows = []
    for f in sorted(glob.glob(str(W / "data" / "cand" / tag / "dense_text_full.enc_q.*.npz"))):
        z = np.load(f); idx, sc = z["idx"][:, :top_k], z["score"][:, :top_k].astype(np.float32)
        keep = (z["score"][:, :1].astype(np.float32) - sc <= DELTA) & (idx >= 0)
        r, c = np.nonzero(keep)
        rows.append(pl.DataFrame({"qi": z["qi"][r].astype(np.uint32), "di": idx[r, c].astype(np.int64)}))
    assert rows, "no dense_text_full.enc_q.*.npz: run run_stage1.sh with TRAIN=1"
    return pl.concat(rows).join(eq, on="qi").select("di", pl.col("id").alias("rid"), pl.col("s1").alias("true_s1"))


def mt_pairs(tag, top_k):
    """(di, rid, true s1) for the matcher-training pool records' first retrieval candidates."""
    pq = pl.read_parquet(W / "data" / "texts" / "pool_q.parquet", columns=["id", "s1"]).with_row_index("qi")
    grp = pl.read_parquet(W / "data" / "split" / "pool_groups.parquet").select("qi", "grp2")
    keep_q = grp.filter(pl.col("grp2").is_in(["MT", "look_mt2"])).select(pl.col("qi").cast(pl.UInt32))
    fs = sorted(glob.glob(str(W / "data" / "cand" / tag / "pairs.pool_q.*.parquet")))
    assert fs, "no pairs.pool_q.*.parquet: run run_stage1.sh with TRAIN=1"
    P = pl.concat([pl.read_parquet(f, columns=["qi", "di", "r_df"]) for f in fs]).filter(pl.col("r_df") <= top_k) \
        .with_columns(pl.col("qi").cast(pl.UInt32), pl.col("di").cast(pl.Int64))
    return P.join(keep_q, on="qi", how="semi").join(pq, on="qi").select("di", pl.col("id").alias("rid"), pl.col("s1").alias("true_s1"))


def build(size, top_k, tag="main", seed=0):
    s1 = pl.read_parquet(W / "data" / "texts" / "s1_train.parquet", columns=["id"]).with_row_index("di") \
        .with_columns(pl.col("di").cast(pl.Int64)).rename({"id": "s1_id"})
    split = pl.read_parquet(W / "data" / "split" / "s1_split.parquet").select(pl.col("entity_id").alias("s1_id"), "split")
    P = pl.concat([enc_pairs(tag, top_k), mt_pairs(tag, top_k)]).join(s1, on="di")
    P = P.join(split.filter(pl.col("split") == "HOLD").select("s1_id"), on="s1_id", how="anti")
    P = P.select("di", "s1_id", pl.col("rid").alias("cand_id"), (pl.col("s1_id") == pl.col("true_s1")).fill_null(False).cast(pl.Int8).alias("label"))
    P = P.unique(["s1_id", "cand_id"], keep="first").sort("s1_id", "cand_id")
    P = P.sample(fraction=1.0, shuffle=True, seed=seed).head(size)
    P = P.with_columns(pl.when((pl.col("s1_id").hash(seed) % 1000) < VAL_PER_MILLE).then(pl.lit("val")).otherwise(pl.lit("train")).alias("split"))
    rd = lambda f: read_tsv(config.TRAIN_DIR / f, columns=["entity_id", "business_name", "business_address"])
    tx = lambda d: d.select("entity_id", text(pl.col("business_name"), pl.col("business_address")).alias("t"))
    ta = tx(rd("train_source1.tsv")).rename({"entity_id": "s1_id", "t": "text_a"})
    tb = tx(pl.concat([rd("train_source2.tsv"), rd("train_source3.tsv")])).rename({"entity_id": "cand_id", "t": "text_b"})
    P = P.join(ta, on="s1_id", how="left", maintain_order="left").join(tb, on="cand_id", how="left", maintain_order="left")
    assert P["text_a"].null_count() == 0 and P["text_b"].null_count() == 0, "pairs whose texts are missing from the training files"
    plain = P.select("s1_id", "cand_id", "label", "split", "text_a", "text_b")
    # context() keys the Source 1 entity by its integer row (di); the ids are restored afterwards
    C = context(P.select(pl.col("di").cast(pl.Utf8).alias("s1_id"), "cand_id", "text_a", "text_b"))
    ctx = plain.with_columns(C["text_b"].alias("text_b"))
    return plain, ctx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=1_093_841, help="number of pairs kept (train + val)")
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--tag", default="main")
    a = ap.parse_args()
    plain, ctx = build(a.size, a.top_k, a.tag)
    out = W / "stage3" / "train"; out.mkdir(parents=True, exist_ok=True)
    plain.write_parquet(out / "plain.parquet"); ctx.write_parquet(out / "context.parquet")
    print(f"judge training pairs: {plain.height:,} {[(sp, lab, n) for sp, lab, n in plain.group_by('split', 'label').len().sort('split', 'label').iter_rows()]} -> {out}")


if __name__ == "__main__":
    main()
