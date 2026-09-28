"""Train a US/India combiner on the hold-out split and score the test zone (CPU, a few minutes per run).

Preset (judge score = mean of the listed judges; features from features.py):
  first_pass      judges plain_1, plain_2, context_1..4     BASE + LIST + TEXT + LOOKALIKE_1                        seeds 1..6

usage: python -m stage4_combiner.train_combiner --preset first_pass --seeds 1,2,3,4,5,6 --out WORK_DIR/stage4/first_pass
       python -m stage4_combiner.train_combiner --preset first_pass --seeds 1,2,3,4,5,6 --out ... --models WEIGHTS_DIR/lightgbm/first_pass
         (--models: score the test zone with released models instead of training; no hold-out step)
Inputs (WORK_DIR; the stage4/hold tables and hold_* scores only when training):
  stage4/hold/hold_candidates.parquet   hold-out candidates with stage-2 features, label, half, country, inzone, trainable
  stage4/hold/hold_zone_texts.parquet   (qi, di, text_a, text_b) for the hold-out zone candidates
  stage3/test_zone_candidates.parquet   test zone candidates with stage-2 features (stage3_judges/zone.py)
  stage2/test_record_index.parquet, stage2/test_source1_index.parquet
  stage3/scores/<judge>/<split>/*.parquet  judge scores (s1_id = di, cand_id = qi), split in hold_<country> (training only) and
                                           test_<country>, one per country with training labels
Writes <out>/test_zone_p.parquet (qi, di, p), <out>/hold_newp.parquet and the LightGBM models.
"""
import argparse
import glob
import json
import os
import time

import numpy as np
import polars as pl

import config
from common.country_route import labelled_branches
from common.io import read_tsv
from stage3_judges.texts import text
from stage4_combiner import crossfit as CF, features as FT

SIX = ["judge_plain_1", "judge_plain_2", "judge_context_1", "judge_context_2", "judge_context_3", "judge_context_4"]
PRESETS = {
    "first_pass": dict(judges=SIX, lookalike=1),
}
# judge-score splits: one per country with training labels (build_inputs.py names them test_<country> / hold_<country>)
HOLD_SPLITS = tuple(f"hold_{c.lower()}" for c in labelled_branches())
TEST_SPLITS = tuple(f"test_{c.lower()}" for c in labelled_branches())


def load_scores(judge_dir, splits):
    """(qi, di, llm_logit) from a judge's score parts; plain or context input folders."""
    fs = []
    for sp in splits:
        sub = sp if glob.glob(f"{judge_dir}/{sp}/*.parquet") else sp + "_context"
        fs += sorted(glob.glob(f"{judge_dir}/{sub}/*.parquet"))
    assert fs, (judge_dir, splits)
    return pl.concat([pl.read_parquet(f) for f in fs]).select(pl.col("s1_id").cast(pl.Int32).alias("di"), pl.col("cand_id").cast(pl.Int32).alias("qi"),
                                                               "llm_logit")


def mean_scores(scores_root, judges, splits):
    """Mean judge score over `judges` (all judges score the same rows)."""
    q = None
    for j in judges:
        s = load_scores(f"{scores_root}/{j}", splits).select("qi", "di", pl.col("llm_logit").cast(pl.Float64).alias(j))
        q = s if q is None else q.join(s, on=["qi", "di"])
    return q.select("qi", "di", pl.mean_horizontal(judges).alias("llm_logit"))


def add_extra(Z, preset, texts, scores_root, splits):
    """Lookalike-signature features of the preset (features.LOOKALIKE_1)."""
    feats = []
    if preset["lookalike"] == 1:
        Z = FT.add_lookalike_1(Z, texts); feats += FT.LOOKALIKE_1
    return Z, feats


def test_texts(Zt, work):
    rd = lambda f: read_tsv(config.TEST_DIR / f, columns=["entity_id", "business_name", "business_address"])
    txt = lambda d: d.select("entity_id", text(pl.col("business_name"), pl.col("business_address")).alias("text"))
    ta = pl.read_parquet(work / "stage2" / "test_source1_index.parquet", columns=["di", "entity_id"]).join(txt(rd("test_source1.tsv")), on="entity_id") \
        .select("di", pl.col("text").alias("text_a"))
    vq = Zt.select("qi").unique().join(pl.read_parquet(work / "stage2" / "test_record_index.parquet", columns=["qi", "entity_id"]), on="qi")
    tb = vq.join(pl.concat([txt(rd("test_source2.tsv")), txt(rd("test_source3.tsv"))]), on="entity_id").select("qi", pl.col("text").alias("text_b"))
    return Zt.select("qi", "di").join(ta, on="di", how="left").join(tb, on="qi", how="left") \
        .with_columns(pl.col("text_a").fill_null(""), pl.col("text_b").fill_null(""))


def load_models(models_dir, seeds):
    """Released cross-fitted models, in the order crossfit() returns them: TUNE-trained seeds, then REPORT-trained seeds."""
    import lightgbm as lgb
    return [lgb.Booster(model_file=f"{models_dir}/model_train{half}_s{sd}.txt") for half in ("TUNE", "REPORT") for sd in seeds]


def run(preset_name, seeds, out, work, scores_root, judge_mean=None, models_dir=None):
    t0 = time.time(); preset = PRESETS[preset_name]; os.makedirs(out, exist_ok=True)
    judge_scores = (lambda splits: load_scores(judge_mean, splits)) if judge_mean else (lambda splits: mean_scores(scores_root, preset["judges"], splits))
    feats = FT.BASE + FT.LIST + FT.TEXT
    if models_dir:
        models = load_models(models_dir, seeds)
        F = feats + (FT.LOOKALIKE_1 if preset["lookalike"] == 1 else [])
        assert models[0].feature_name() == F, "released models were trained on other features"
        return score_test(preset_name, preset, seeds, out, work, scores_root, judge_scores, models, F, t0)
    # hold-out
    Q = judge_scores(HOLD_SPLITS)
    X0 = pl.read_parquet(work / "stage4" / "hold" / "hold_candidates.parquet").drop("llm_logit", "qrk", "qgap", "qpnull", strict=False)
    X0 = FT.judge_feats(X0.join(Q, on=["qi", "di"], how="left"))
    htexts = pl.read_parquet(work / "stage4" / "hold" / "hold_zone_texts.parquet")
    Z = FT.add_text(FT.add_listwise(X0.filter(pl.col("inzone") == 1)), htexts)
    Z, extra = add_extra(Z, preset, htexts, scores_root, HOLD_SPLITS)
    F = feats + extra
    X = pl.concat([Z, X0.filter(pl.col("inzone") == 0)], how="diagonal_relaxed")
    P, models = CF.crossfit(X, F, seeds)
    zq = X.filter(pl.col("inzone") == 1).select("qi").unique()
    P.join(zq, on="qi", how="semi").select("qi", "di", pl.col("pnew").fill_null(0.0).alias("p")).write_parquet(f"{out}/hold_newp.parquet")
    for i, b in enumerate(models):
        b.save_model(f"{out}/model_train{('TUNE', 'REPORT')[i // len(seeds)]}_s{seeds[i % len(seeds)]}.txt")
    print(f"[{preset_name} {seeds}] {len(F)} features, {len(models)} models [{time.time() - t0:.0f}s]", flush=True)
    return score_test(preset_name, preset, seeds, out, work, scores_root, judge_scores, models, F, t0)


def score_test(preset_name, preset, seeds, out, work, scores_root, judge_scores, models, F, t0):
    """Score every test zone candidate with the mean of the combiner models; writes <out>/test_zone_p.parquet."""
    Zt = pl.read_parquet(work / "stage3" / "test_zone_candidates.parquet")
    Zt = FT.matcher_feats(FT.judge_feats(Zt.join(judge_scores(TEST_SPLITS), on=["qi", "di"], how="left")))
    Zt = FT.add_listwise(Zt)
    T = test_texts(Zt, work)
    Zt = FT.add_text(Zt, T)
    Zt, _ = add_extra(Zt, preset, T, scores_root, TEST_SPLITS)
    S = Zt.filter(pl.col("llm_logit").is_not_null())
    S = S.select("qi", "di").with_columns(pl.Series("p", CF.predict(models, S, F)))
    Zt.select("qi", "di").join(S, on=["qi", "di"], how="left").with_columns(pl.col("p").fill_null(0.0)).write_parquet(f"{out}/test_zone_p.parquet")
    json.dump({"preset": preset_name, "seeds": list(seeds), "features": F, "judges": preset["judges"]}, open(f"{out}/combiner.json", "w"), indent=1)
    print(f"[{preset_name} {seeds}] test zone pairs {Zt.height:,}, scored {S.height:,} [{time.time() - t0:.0f}s]", flush=True)


if __name__ == "__main__":
    from pathlib import Path
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", required=True, choices=sorted(PRESETS))
    ap.add_argument("--seeds", required=True, help="e.g. 1,2,3")
    ap.add_argument("--out", required=True)
    ap.add_argument("--work", default=str(config.WORK_DIR))
    ap.add_argument("--scores", default=None, help="judge score root (default WORK_DIR/stage3/scores)")
    ap.add_argument("--judge-mean", default=None, help="precomputed judge-mean score folder (merge_scores.py) instead of averaging here")
    ap.add_argument("--models", default=None, help="released model folder: score the test zone with these models instead of training")
    a = ap.parse_args()
    work = Path(a.work).resolve()
    run(a.preset, tuple(int(s) for s in a.seeds.split(",")), a.out, work, a.scores or str(work / "stage3" / "scores"), a.judge_mean, a.models)
