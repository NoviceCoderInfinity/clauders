"""Build the hold-out tables of the combiner and the hold-out judge inputs from the TRAIN=1 outputs of stages 1-2 (CPU).

These tables are made of TRAINING records only (the hold-out split of the training Source 1 entities, split_train.py). They
are what the combiner is trained and evaluated on (hold_table.py, train_combiner.py, evaluate_holdout.py).

Inputs (WORK_DIR, all written by run_stage1.sh / run_stage2.sh with TRAIN=1):
  data/split/s1_split.parquet                      training split of the Source 1 entities (split_train.py)
  data/texts/s1_train.parquet, pool_q.parquet      row order = di (Source 1) and qi (training pool records) (make_texts.py)
  matchers/us_india/v_scores.parquet               matcher probabilities on the validation rows: HOLD records and the
                                                   look_v lookalike records (train_matcher.py)
  reranker/{listwise_full,specialist_name,specialist_address}/pool_ce.parquet   reranker scores on the pool pairs (qi, di, ce)
  stage3/train/plain.parquet                       the judges' training pairs (stage3_judges/build_training_sets.py)
Writes WORK_DIR/stage4/hold/:
  source1_split.parquet     di, entity_id, country, split
  record_pool.parquet       qi, entity_id, src, country, s1, s1_split
  matcher_scores.parquet    qi, di, src, label, grp (HOLD | D_v = look_v), country, p
  hold_cross_encoder.parquet  matcher_scores + ce2 / ce3 / ce4 (listwise, name, address) with rank, gap to the record's best,
                              no-match softmax, and p_rank
  judge_train_pairs.parquet the judges' training pairs (hold_table.py leaves their Source 1 entities out)
  hold_candidates.parquet   hold_table.build()
  hold_zone_texts.parquet   qi, di, text_a, text_b for every uncertainty-zone row of hold_candidates
and WORK_DIR/stage3/inputs/hold_<country>.parquet: the hold-out judge inputs = the zone rows among the listwise top 3 of each
record, which are exactly the rows whose judge score the combiner reads (features.judge_feats).

usage: python -m stage4_combiner.build_hold_tables
"""
import os
import shutil

import polars as pl

import config
from common.country_route import labelled_branches
from common.io import read_tsv
from stage3_judges.texts import text

W = config.WORK_DIR
H = W / "stage4" / "hold"
RERANKERS = (("ce2", "listwise_full"), ("ce3", "specialist_name"), ("ce4", "specialist_address"))


def base_tables():
    """source1_split, record_pool, matcher_scores, hold_cross_encoder, judge_train_pairs."""
    sp = pl.read_parquet(W / "data" / "split" / "s1_split.parquet").select("entity_id", "split")
    s1 = pl.read_parquet(W / "data" / "texts" / "s1_train.parquet", columns=["id", "country"]).with_row_index("di") \
        .with_columns(pl.col("di").cast(pl.Int32))
    s1 = s1.join(sp, left_on="id", right_on="entity_id", how="left", maintain_order="left")
    assert s1["split"].null_count() == 0
    s1.select("di", pl.col("id").alias("entity_id"), "country", "split").write_parquet(H / "source1_split.parquet")

    pq = pl.read_parquet(W / "data" / "texts" / "pool_q.parquet", columns=["id", "src", "country", "s1", "s1_split"]).with_row_index("qi")
    pq.select(pl.col("qi").cast(pl.Int32), pl.col("id").alias("entity_id"), pl.col("src").cast(pl.Int32), "country", "s1", "s1_split") \
        .write_parquet(H / "record_pool.parquet")

    v = pl.read_parquet(W / "matchers" / "us_india" / "v_scores.parquet")
    ms = v.select(pl.col("qi").cast(pl.Int32), pl.col("di").cast(pl.Int32), pl.col("src").cast(pl.Int32), pl.col("label").cast(pl.Int8),
                  pl.col("grp").replace({"look_v": "D_v"}), "country", pl.col("p").cast(pl.Float64))
    assert set(ms["grp"].unique().to_list()) <= {"HOLD", "D_v"}, ms["grp"].unique().to_list()
    ms.write_parquet(H / "matcher_scores.parquet")

    X = ms
    for name, folder in RERANKERS:
        C = pl.read_parquet(W / "reranker" / folder / "pool_ce.parquet").select(
            pl.col("qi").cast(pl.Int32), pl.col("di").cast(pl.Int32), pl.col("ce").cast(pl.Float32).alias(name))
        X = X.join(C, on=["qi", "di"], how="left")
        # validation pairs outside the reranker's scored pairs keep a null score (their rank / gap / pnull are null too)
        print(f"{folder}: {X[name].null_count():,} of {X.height:,} validation pairs without a reranker score", flush=True)
        X = X.with_columns(pl.col(name).rank("ordinal", descending=True).over("qi").cast(pl.Int16).alias(f"{name}_rank"),
                           (pl.col(name).max().over("qi") - pl.col(name)).alias(f"{name}_gap"),
                           (pl.col(name).exp() / (pl.col(name).exp().sum().over("qi") + 1.0)).alias(f"{name}_pnull"))
    X = X.with_columns(pl.col("p").rank("ordinal", descending=True).over("qi").alias("p_rank"))
    X.write_parquet(H / "hold_cross_encoder.parquet")

    shutil.copy(W / "stage3" / "train" / "plain.parquet", H / "judge_train_pairs.parquet")


def raw_texts():
    """(entity_id, text) of the training Source 1 entities and of the training Source 2 / 3 records."""
    rd = lambda f: read_tsv(config.TRAIN_DIR / f, columns=["entity_id", "business_name", "business_address"])
    tx = lambda d: d.select("entity_id", text(pl.col("business_name"), pl.col("business_address")).alias("text"))
    return tx(rd("train_source1.tsv")), tx(pl.concat([rd("train_source2.tsv"), rd("train_source3.tsv")]))


def zone_tables():
    """hold_zone_texts (every zone row of hold_candidates) and the hold-out judge inputs (zone rows among the listwise top 3)."""
    hc = pl.read_parquet(H / "hold_candidates.parquet", columns=["qi", "di", "inzone", "ce2_rank", "ce2_pnull", "country"])
    sm = pl.read_parquet(H / "source1_split.parquet").select("di", pl.col("entity_id").alias("sid"))
    pm = pl.read_parquet(H / "record_pool.parquet").select("qi", pl.col("entity_id").alias("vid"))
    t1, t23 = raw_texts()
    Z = hc.filter(pl.col("inzone") == 1).join(pm, on="qi").join(sm, on="di") \
        .join(t1.rename({"entity_id": "sid", "text": "text_a"}), on="sid", how="left") \
        .join(t23.rename({"entity_id": "vid", "text": "text_b"}), on="vid", how="left")
    Z.select("qi", "di", "text_a", "text_b").write_parquet(H / "hold_zone_texts.parquet")
    out = W / "stage3" / "inputs"; out.mkdir(parents=True, exist_ok=True)
    for c in labelled_branches():
        D = Z.filter((pl.col("country") == c) & (pl.col("ce2_rank") <= 3)).sort("qi", "di").select(
            pl.col("di").cast(pl.Utf8).alias("s1_id"), pl.col("qi").cast(pl.Utf8).alias("cand_id"), "text_a", "text_b")
        D.write_parquet(out / f"hold_{c.lower()}.parquet")
        print(f"hold judge inputs {c}: {D.height:,}", flush=True)
    print(f"hold zone texts: {Z.height:,}", flush=True)


def main():
    H.mkdir(parents=True, exist_ok=True)
    base_tables()
    from stage4_combiner import hold_table
    X = hold_table.build(H)
    X.write_parquet(H / "hold_candidates.parquet")
    print(f"hold_candidates: {X.height:,} rows, zone rows {X['inzone'].sum():,}, trainable {X['trainable'].sum():,}", flush=True)
    zone_tables()


if __name__ == "__main__":
    main()
