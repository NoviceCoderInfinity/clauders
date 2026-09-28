"""Cross-fitted LightGBM combiner on the hold-out split.

The hold-out Source 1 entities are split into two halves. For every seed, one model is trained on the zone candidates of each
half and predicts the other half, so every hold-out prediction is out-of-sample. On test, the prediction is the mean over all
(2 halves x seeds) models. Training rows: zone candidates with a judge score whose Source 1 entity was not used to train the
judges (column `trainable`).
"""
import numpy as np
import polars as pl
import lightgbm as lgb

NT = 16
PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=31, min_data_in_leaf=100, feature_fraction=0.9, bagging_fraction=0.8,
              bagging_freq=1, lambda_l2=1.0, num_threads=NT, seed=42, verbose=-1, deterministic=True)
ROUNDS = 400


def crossfit(X, feats, seeds, rounds=ROUNDS):
    """X: hold-out candidates with feats, label, half (TUNE / REPORT), inzone, trainable, llm_logit.
    Returns (hold-out predictions (qi, di, pnew), models in order [TUNE-trained seeds..., REPORT-trained seeds...])."""
    tr_filter = (pl.col("inzone") == 1) & pl.col("llm_logit").is_not_null() & pl.col("trainable")
    te_filter = (pl.col("inzone") == 1) & pl.col("llm_logit").is_not_null()
    TR = X.filter(tr_filter).sort("qi", "di"); TE = X.filter(te_filter).sort("qi", "di")
    parts, models = [], []
    for trh, teh in (("TUNE", "REPORT"), ("REPORT", "TUNE")):
        tr = TR.filter(pl.col("half") == trh); te = TE.filter(pl.col("half") == teh)
        xt = te.select(feats).to_numpy(); pred = np.zeros(te.height)
        for sd in seeds:
            ds = lgb.Dataset(tr.select(feats).to_numpy(), tr["label"].to_numpy(), feature_name=feats, free_raw_data=False)
            b = lgb.train(dict(PARAMS, seed=sd), ds, rounds)
            pred += b.predict(xt, num_threads=NT) / len(seeds); models.append(b)
        parts.append(te.select("qi", "di").with_columns(pl.Series("pnew", pred)))
    return pl.concat(parts), models


def predict(models, X, feats):
    return np.mean([b.predict(X.select(feats).to_numpy(), num_threads=NT) for b in models], axis=0)
