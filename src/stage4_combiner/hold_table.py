"""Hold-out candidate table for the US/India combiners (CPU).

Every candidate of the reduced hold-out pool (holdout.HoldOut.Vx) with the stage-2 cross-encoder columns, the matcher
probability features (features.matcher_feats), the half (TUNE / REPORT), the zone flag (stage3_judges.zone rule) and a
trainable flag: rows whose Source 1 entity or (Source 1, record) pair was used to train the judges are excluded from combiner
training and from the hold-out scores.

usage: python -m stage4_combiner.hold_table
Inputs (WORK_DIR/stage4/hold/): hold_cross_encoder.parquet (hold-out candidates with ce2 / ce3 / ce4 columns, label, grp,
country, p), source1_split.parquet, record_pool.parquet (qi, entity_id), matcher_scores.parquet, judge_train_pairs.parquet
(s1_id, cand_id, split). judge_train_pairs holds every pair of the six judges' training sets (stage3/train/plain, 1,093,841
rows, none of whose Source 1 entities is in the hold-out split) plus 120,195 further pairs of hold-out entities that are
in none of the six judges' training sets. The exclusion below removes the Source 1 entities of its first 500,000 "train" rows; for the six
judges this only removes more hold-out entities than needed, so no hold-out entity seen by a judge is scored or trained on.
Writes WORK_DIR/stage4/hold/hold_candidates.parquet.
"""
import polars as pl

import config
from stage3_judges.zone import zone_records
from stage4_combiner import features as FT
from stage4_combiner.holdout import HoldOut


def judge_training(hold_dir):
    T = pl.read_parquet(f"{hold_dir}/judge_train_pairs.parquet").filter(pl.col("split") == "train").head(500000)
    s1m = pl.read_parquet(f"{hold_dir}/source1_split.parquet"); pm = pl.read_parquet(f"{hold_dir}/record_pool.parquet")
    excluded_s1 = s1m.join(T.select(pl.col("s1_id").alias("entity_id")).unique(), on="entity_id", how="semi").select("di")
    excluded_pairs = T.select(pl.col("s1_id").alias("entity_id"), pl.col("cand_id").alias("vid")).join(s1m.select("di", "entity_id"), on="entity_id") \
        .join(pm.select("qi", pl.col("entity_id").alias("vid")), on="vid").select("qi", "di")
    return excluded_s1, excluded_pairs


def build(hold_dir=None):
    hold_dir = hold_dir or config.WORK_DIR / "stage4" / "hold"
    excluded_s1, excluded_pairs = judge_training(hold_dir)
    H = HoldOut(hold_dir, excluded_s1=excluded_s1)
    XF = pl.read_parquet(f"{hold_dir}/hold_cross_encoder.parquet")
    zq = zone_records(XF)
    X = FT.matcher_feats(XF.join(H.Vx.select("qi", "di"), on=["qi", "di"])).join(H.S1.select("di", "half"), on="di", how="left")
    X = X.join(zq.with_columns(pl.lit(1).alias("inzone")), on="qi", how="left").with_columns(pl.col("inzone").fill_null(0))
    bp = excluded_pairs.with_columns(pl.lit(1).alias("_bp")); bd = excluded_s1.with_columns(pl.lit(1).alias("_bd"))
    X = X.join(bp, on=["qi", "di"], how="left").join(bd, on="di", how="left")
    return X.with_columns((pl.col("_bp").is_null() & pl.col("_bd").is_null()).alias("trainable")).drop("_bp", "_bd")


if __name__ == "__main__":
    X = build()
    out = config.WORK_DIR / "stage4" / "hold" / "hold_candidates.parquet"
    X.write_parquet(out)
    print(f"wrote {out}: {X.height:,} rows, zone rows {X['inzone'].sum():,}, trainable {X['trainable'].sum():,}", flush=True)
