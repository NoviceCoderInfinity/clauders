"""First-stage test predictions and files.
usage: python predict_first_stage.py <out_dir> --seen <lgb_dir>:<t>:<m> --unseen <lgb_dir>:<t>:<m> --ce_test name=path ...
  seen   : countries with training labels (common/country_route.py)
  unseen : every country without training labels (the UNLABELLED branch)
  each lgb_dir has lgbm.txt + summary.json (feature list, best iteration). Score files: parquet qi, di, ce over the test candidate pairs
  (ce = e5-small cross-encoder, ce2 = listwise reranker or, for unseen countries, the core-name reranker, ce3 / ce4 = name / address
  specialists).
Decision: each record goes to its best Source 1 if p >= t and p - second best >= m; caps 5 Source 2 / 6 Source 3 per Source 1.
Writes <out_dir>/matching_results.tsv, candidate_pairs.tsv, test_scores.parquet (qi, di, src, p), test_stats.json.
"""
import os, sys, json, glob, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, polars as pl, lightgbm as lgb
import config
from common.country_route import UNLABELLED, labelled_branches, with_branch
from common.decision import decide_m

W = str(config.WORK_DIR)
ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("--seen", required=True); ap.add_argument("--unseen", required=True)
ap.add_argument("--ce_test", action="append", default=[]); ap.add_argument("--feat_tag", default="main")
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
tq = pl.read_parquet(f"{W}/data/texts/test_q.parquet", columns=["id", "country", "src"]).with_row_index("qi")
s1 = pl.read_parquet(f"{W}/data/texts/s1_test.parquet", columns=["id", "country"]).with_row_index("di")
X = pl.concat([pl.read_parquet(f) for f in sorted(glob.glob(f"{W}/data/feat/{a.feat_tag}/test_q.*.parquet"))], how="vertical_relaxed")
X = with_branch(X.join(tq.select("qi", "country"), on="qi"), config.TEST_DIR)
CE = {}
for spec in a.ce_test:
    nm, path = spec.split("=", 1); CE[nm] = pl.read_parquet(path)


def add_ce(X, name, C):
    X = X.join(C.select("qi", "di", pl.col("ce").cast(pl.Float32).alias(name)), on=["qi", "di"], how="left")
    return X.with_columns(pl.col(name).rank("ordinal", descending=True).over("qi").cast(pl.Int16).alias(f"{name}_rank"),
                          (pl.col(name).max().over("qi") - pl.col(name)).alias(f"{name}_gap"),
                          (pl.col(name).exp() / (pl.col(name).exp().sum().over("qi") + 1.0)).alias(f"{name}_pnull"))


def run(spec, rows):
    """spec = <lgb_dir>:t:m"""
    d, t, m = spec.split(":"); t, m = float(t), float(m)
    S = json.load(open(f"{d}/summary.json")); feats = S["feats"]
    Y = rows
    for nm in ("ce", "ce2", "ce3", "ce4"):
        if nm in feats: Y = add_ce(Y, nm, CE[nm])
    bst = lgb.Booster(model_file=f"{d}/lgbm.txt")
    p = bst.predict(Y.select(feats).to_numpy(), num_iteration=S["iters"])
    Y = Y.select("qi", "di", "src").with_columns(pl.Series("p", p))
    return Y, decide_m(Y, t, m)


seen = X.filter(pl.col("branch").is_in(labelled_branches())); unseen = X.filter(pl.col("branch") == UNLABELLED)
Ys, Ds = run(a.seen, seen) if seen.height else (None, None)
Yu, Du = run(a.unseen, unseen) if unseen.height else (None, None)
Y = pl.concat([p for p in (Ys, Yu) if p is not None]); D = pl.concat([p for p in (Du, Ds) if p is not None])
Y.write_parquet(f"{a.out}/test_scores.parquet")
qid = tq.select("qi", pl.col("id").alias("m")); did = s1.select("di", pl.col("id").alias("s1"))
M = D.join(qid, on="qi").join(did, on="di").group_by("s1").agg(pl.col("m").sort().str.join(","))
C = Y.select("qi", "di").join(qid, on="qi").join(did, on="di").group_by("s1").agg(pl.col("m").unique().sort().str.join(","))
allS1 = s1.select(pl.col("id").alias("s1"))
for frame, col, fn in ((M, "matched_entity_ids", "matching_results.tsv"), (C, "candidate_entity_ids", "candidate_pairs.tsv")):
    out = allS1.join(frame, on="s1", how="left").with_columns(pl.col("m").fill_null("")).rename({"s1": "source1_entity_id", "m": col})
    out.write_csv(f"{a.out}/{fn}", separator="\t", quote_style="never")
stats = {}
for c in s1["country"].unique().sort().to_list():
    sc = s1.filter(pl.col("country") == c)
    dd = D.join(sc.select("di"), on="di")
    rec_c = tq.filter(pl.col("country") == c).height
    stats[c] = {"s1": sc.height, "pred_per_s1": round(dd.height / sc.height, 3),
                "empty_share": round(1 - dd["di"].n_unique() / sc.height, 4), "records_matched_share": round(dd.height / rec_c, 4)}
json.dump({"seen": a.seen, "unseen": a.unseen, "per_country": stats}, open(f"{a.out}/test_stats.json", "w"), indent=1)
print(json.dumps(stats))
