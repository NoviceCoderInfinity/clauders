"""LightGBM matcher on the pair features plus any set of cross-encoder / reranker scores.
  train: MT records + look_mt2 lookalikes, pairs whose Source 1 is not HOLD;   validation: HOLD records + look_v lookalikes
usage: python train_matcher.py <out_dir> [--ce name=path.parquet ...] [--no_cross_encoder] [--drop_feats a,b]
  each ce parquet: qi, di, ce (logit) over pool_q candidate pairs. Derived per score: ce, ce_rank, ce_gap (to the record's best),
  ce_pnull (softmax over the record's candidates + null logit 0).
  --no_cross_encoder: leave out the e5-small cross-encoder score (feature "ce").
Writes <out_dir>/v_scores.parquet, lgbm.txt, summary.json (feature list, best iteration). Seeds fixed (42).
Models used: matcher_us_india = --ce ce2=<listwise full> --ce ce3=<name specialist> --ce ce4=<address specialist>;
             matcher_france   = --no_cross_encoder --ce ce2=<core-name reranker>.
"""
import os, sys, json, glob, argparse, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, polars as pl, lightgbm as lgb
import config

W = str(config.WORK_DIR)
ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("--ce", action="append", default=[])
ap.add_argument("--no_cross_encoder", action="store_true"); ap.add_argument("--drop_feats", default="")
ap.add_argument("--threads", type=int, default=32); ap.add_argument("--feat_tag", default="main")
ap.add_argument("--leaves", type=int, default=255); ap.add_argument("--min_leaf", type=int, default=200)
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); T0 = time.time()
tq = pl.read_parquet(f"{W}/data/texts/pool_q.parquet", columns=["country"]).with_row_index("qi")
g2 = pl.read_parquet(f"{W}/data/split/pool_groups.parquet").select("qi", pl.col("grp2").alias("grp"))
s1 = pl.read_parquet(f"{W}/data/texts/s1_train.parquet", columns=["id"]).with_row_index("di").join(
    pl.read_parquet(f"{W}/data/split/s1_split.parquet").select(pl.col("entity_id").alias("id"), pl.col("split").alias("s1_split_d")), on="id").drop("id")
X = pl.concat([pl.read_parquet(f) for f in sorted(glob.glob(f"{W}/data/feat/{a.feat_tag}/pool_q.*.parquet"))], how="vertical_relaxed")
X = X.join(g2, on="qi").filter(pl.col("grp") != "look_ce").join(s1, on="di").join(tq, on="qi")


def add_ce(X, name, path):
    C = pl.read_parquet(path).select("qi", "di", pl.col("ce").cast(pl.Float32).alias(name))
    X = X.join(C, on=["qi", "di"], how="left")
    X = X.with_columns(
        pl.col(name).rank("ordinal", descending=True).over("qi").cast(pl.Int16).alias(f"{name}_rank"),
        (pl.col(name).max().over("qi") - pl.col(name)).alias(f"{name}_gap"),
        (pl.col(name).exp() / (pl.col(name).exp().sum().over("qi") + 1.0)).alias(f"{name}_pnull"))
    return X


def cross_encoder_scores(items):
    """e5-small cross-encoder logits aligned with data/cross_encoder/pairs_<items>.parquet (shards concatenated in order)."""
    P = pl.read_parquet(f"{W}/data/cross_encoder/pairs_{items}.parquet")
    fs = sorted(glob.glob(f"{W}/cross_encoder/scores/{items}.*-of-*.npy"), key=lambda f: int(f.split(".")[-2].split("-")[0]))
    return P.with_columns(pl.Series("ce", np.concatenate([np.load(f) for f in fs]).astype(np.float32)))


if not a.no_cross_encoder:
    cache = f"{W}/cross_encoder/pool_ce.parquet"
    if not os.path.exists(cache):
        cross_encoder_scores("pool_q").write_parquet(cache)
    X = add_ce(X, "ce", cache)
for spec in a.ce:
    nm, path = spec.split("=", 1); X = add_ce(X, nm, path)
drop = {"qi", "di", "label", "grp", "s1_split_d", "country"} | set(filter(None, a.drop_feats.split(",")))
feats = [c for c in X.columns if c not in drop]
tr = X.filter(pl.col("grp").is_in(["MT", "look_mt2"]) & (pl.col("s1_split_d") != "HOLD"))
va = X.filter(pl.col("grp").is_in(["HOLD", "look_v"]))
print(f"[{time.time()-T0:.0f}s] train {tr.height:,} pairs ({tr['label'].sum():,} pos), validation {va.height:,}; {len(feats)} feats", flush=True)
es = (tr["qi"].hash(42) % 10 == 0).to_numpy()
params = dict(objective="binary", learning_rate=0.1, num_leaves=a.leaves, min_data_in_leaf=a.min_leaf, feature_fraction=0.8, bagging_fraction=0.7,
              bagging_freq=1, lambda_l2=1.0, num_threads=a.threads, seed=42, verbose=-1, deterministic=True, force_row_wise=True)
dtr = lgb.Dataset(tr.filter(~pl.Series(es)).select(feats).to_numpy(), tr.filter(~pl.Series(es))["label"].to_numpy(), feature_name=feats)
des = lgb.Dataset(tr.filter(pl.Series(es)).select(feats).to_numpy(), tr.filter(pl.Series(es))["label"].to_numpy(), reference=dtr)
bst = lgb.train(params, dtr, 2000, valid_sets=[des], callbacks=[lgb.early_stopping(100, verbose=False)])
bst.save_model(f"{a.out}/lgbm.txt")
imp = sorted(zip(feats, bst.feature_importance("gain")), key=lambda x: -x[1]); tot = sum(v for _, v in imp)
top = [(k, round(v / tot * 100, 1)) for k, v in imp[:12]]
print(f"[{time.time()-T0:.0f}s] {bst.best_iteration} trees; top gain: {top}", flush=True)
p = bst.predict(va.select(feats).to_numpy(), num_iteration=bst.best_iteration)
va.select("qi", "di", "src", "label", "grp", "country").with_columns(pl.Series("p", p)).write_parquet(f"{a.out}/v_scores.parquet")
json.dump({"feats": feats, "iters": bst.best_iteration}, open(f"{a.out}/summary.json", "w"), indent=1)
print("done", a.out)
