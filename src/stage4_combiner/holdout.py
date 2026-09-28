"""Hold-out evaluation for the countries with training labels: expected F0.5 of a decision set on the hold-out Source 1 entities.

The hold-out Source 1 entities (split == "HOLD" in the stage-1 split map) are divided into two halves by a fixed hash
(TUNE / REPORT). Their candidate pool is reduced to the size of the test pool per country (random sample of the other training entities, seed 42;
the size comes from test row counts, test_sizes(), and this pool is also the one the combiner is trained on), and records
whose best matcher candidate is a filler entity are dropped. For a decision set D, per Source 1 entity:
  tp = accepted true pairs, n = true matches, false positives split into lookalike ("D_v" group) and other rows;
  F0.5 uses the false-positive counts re-weighted to test density (w_d for lookalike rows, w_o[country] for the others),
  averaged over 20 negative-binomial draws (seed 42). Scores are reported per half and country and as a test-share mix.
HoldOut.evaluate_D() scores a decision set. evaluate_holdout.py uses it to score the deployed decision (stage-2 rule 0.50 / 0.30
outside the zone; set-size prior, combiner rule (t, m) and caps 5 / 6 inside) over the (t, m) grid; the rule 0.65 / 0.40 is the
best on the TUNE half. (grid() and hybrid_decide() below are the same scan without the set-size prior.)

Inputs (WORK_DIR/stage4/hold/): source1_split.parquet (di, entity_id, split, country), matcher_scores.parquet
(qi, di, src, p, label, grp: stage-2 matcher probabilities on the hold-out pool) and DATA_DIR/train/train_ground_truth.tsv.
"""
import numpy as np
import polars as pl

import config



def test_sizes(test_dir=config.TEST_DIR, train_dir=config.TRAIN_DIR):
    """Label-free sizes read from the files (no labels, no record content):
    n_s1[c]  number of test Source 1 entities of each country with training labels (sets the hold-out pool size per country);
    w_d      false-merge weight of lookalike records: test unmatched Source 2/3 records per Source 1 entity (from the file row
             counts, assuming the training rate of matched records per entity) over half the training rate."""
    from common.country_route import labelled_branches, with_branch
    from common.io import read_tsv
    s1 = with_branch(read_tsv(f"{test_dir}/test_source1.tsv", columns=["country"]))
    n_s1 = {c: s1.filter(pl.col("branch") == c).height for c in labelled_branches()}
    n23 = sum(read_tsv(f"{test_dir}/test_source{i}.tsv", columns=["entity_id"]).height for i in (2, 3))
    tr_s1 = read_tsv(f"{train_dir}/train_source1.tsv", columns=["entity_id"]).height
    tr_23 = sum(read_tsv(f"{train_dir}/train_source{i}.tsv", columns=["entity_id"]).height for i in (2, 3))
    gt = read_tsv(f"{train_dir}/train_ground_truth.tsv")["matched_entity_ids"].fill_null("")
    tr_m = int(gt.str.split(",").list.eval(pl.element().filter(pl.element() != "")).list.len().sum())
    m_per_s1 = tr_m / tr_s1
    test_unm = (n23 - m_per_s1 * s1.height) / s1.height
    train_unm = (tr_23 - tr_m) / tr_s1
    return n_s1, test_unm / (train_unm * 0.5)


GRID_T = (0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8)
GRID_M = (0.2, 0.3, 0.4, 0.5)


def f05(tp, fp, n):
    tp, fp, n = (np.asarray(x, float) for x in (tp, fp, n)); npred = tp + fp
    with np.errstate(divide="ignore", invalid="ignore"):
        P = np.where(npred > 0, tp / npred, 0.); R = np.where(n > 0, tp / n, 0.)
        F = np.where(P + R > 0, 1.25 * P * R / (.25 * P + R), 0.)
    return np.where(n == 0, (npred == 0).astype(float), F)


def decide_m(P, t, m=0.0, cap2=5, cap3=6):
    """Best Source 1 candidate per record with p >= t and margin >= m, then caps per Source 1 entity."""
    srt = P.sort(["qi", "p"], descending=[False, True])
    k = srt.with_columns(pl.col("p").cum_count().over("qi").alias("_k"))
    best = k.filter(pl.col("_k") == 1).join(k.filter(pl.col("_k") == 2).select("qi", pl.col("p").alias("_p2")), on="qi", how="left")
    best = best.with_columns(pl.col("_p2").fill_null(0.0)).filter((pl.col("p") >= t) & (pl.col("p") - pl.col("_p2") >= m))
    best = best.with_columns(pl.col("p").rank("ordinal", descending=True).over(["di", "src"]).alias("_r"))
    return best.filter(((pl.col("src") == 2) & (pl.col("_r") <= cap2)) | ((pl.col("src") == 3) & (pl.col("_r") <= cap3))).drop("_r", "_k", "_p2")


def hybrid_decide(Y, zq, t, m):
    """Y: qi, di, src, p (matcher), pnew (combiner, zone rows). Outside the zone the matcher rule (0.5, 0.3), inside the combiner
    rule (t, m); caps 5 Source 2 / 6 Source 3 jointly on the decision score."""
    Yn = Y.join(zq, on="qi", how="anti").select("qi", "di", "src", "p")
    Yz = Y.join(zq, on="qi", how="semi").select("qi", "di", "src", pl.col("pnew").fill_null(0.0).alias("p"))
    Dn = decide_m(Yn, 0.5, 0.3, cap2=10**6, cap3=10**6); Dz = decide_m(Yz, t, m, cap2=10**6, cap3=10**6)
    D = pl.concat([Dn.select("qi", "di", "src", "p"), Dz.select("qi", "di", "src", "p")])
    D = D.with_columns(pl.col("p").rank("ordinal", descending=True).over(["di", "src"]).alias("_r"))
    return D.filter(((pl.col("src") == 2) & (pl.col("_r") <= 5)) | ((pl.col("src") == 3) & (pl.col("_r") <= 6))).select("qi", "di")


class HoldOut:
    def __init__(self, hold_dir=None, train_dir=config.TRAIN_DIR, excluded_s1=None):
        hold_dir = hold_dir or config.WORK_DIR / "stage4" / "hold"
        S1 = pl.read_parquet(f"{hold_dir}/source1_split.parquet").rename({"entity_id": "id"})
        self.S1 = S1.with_columns(pl.when(pl.col("id").hash(42) % 2 == 0).then(pl.lit("TUNE")).otherwise(pl.lit("REPORT")).alias("half"))
        self.HOLD = self.S1.filter(pl.col("split") == "HOLD")
        g = pl.read_csv(f"{train_dir}/train_ground_truth.tsv", separator="\t", quote_char=None, infer_schema=False)
        g = g.with_columns(pl.col("matched_entity_ids").str.split(",")).explode("matched_entity_ids").filter(pl.col("matched_entity_ids") != "")
        g = g.select(pl.col("source1_entity_id").alias("s1"), pl.col("matched_entity_ids").alias("m"))
        self.TRUTH = g.join(self.HOLD.select(pl.col("id").alias("s1"), "di"), on="s1").group_by("di").agg(pl.len().alias("n"))
        self.n_s1, self.w_d = test_sizes(train_dir=train_dir)
        keep = [self.HOLD.select("di")]
        for c, n in self.n_s1.items():
            oth = self.S1.filter((pl.col("country") == c) & (pl.col("split") != "HOLD")).select("di")
            keep.append(oth.sample(n - self.HOLD.filter(pl.col("country") == c).height, seed=42))
        self.KEEP = pl.concat(keep)
        V = pl.read_parquet(f"{hold_dir}/matcher_scores.parquet")
        tgt = V.filter(pl.col("grp") == "D_v").sort(["qi", "p"], descending=[False, True]).unique("qi", keep="first").select("qi", "di")
        self.DROP = tgt.join(self.KEEP, on="di", how="anti").select("qi")
        self.Vx = V.select("qi", "di", "src", "p", "label", "grp").join(self.KEEP, on="di").join(self.DROP, on="qi", how="anti")
        S1x = self.S1.join(self.KEEP, on="di")
        nh = dict(self.HOLD.group_by("country").len().iter_rows()); ns = dict(S1x.group_by("country").len().iter_rows())
        self.w_o = {c: ns[c] / nh[c] for c in ns}
        self.scored = self.HOLD if excluded_s1 is None else self.HOLD.join(excluded_s1, on="di", how="anti")

    def per_s1(self, D):
        D = D.join(self.scored.select("di"), on="di")
        agg = D.group_by("di").agg((pl.col("label") == 1).sum().alias("tp"),
                                   ((pl.col("label") == 0) & (pl.col("grp") == "D_v")).sum().alias("xd"),
                                   ((pl.col("label") == 0) & (pl.col("grp") == "HOLD")).sum().alias("xo"))
        return self.scored.select("di", "country", "half").join(self.TRUTH, on="di", how="left").join(agg, on="di", how="left").fill_null(0)

    def f_est(self, T, rng, R=20):
        tp, n, xd, xo = (T[c].to_numpy().astype(float) for c in ("tp", "n", "xd", "xo"))
        c = T["country"].to_numpy(); wo = np.vectorize(self.w_o.get)(c)
        out = np.zeros((R, len(tp))); strat = np.char.add(c.astype(str), (n == 0).astype(str))
        for s in np.unique(strat):
            msk = strat == s; Z = 0
            for x, w in ((xd[msk], np.full(msk.sum(), self.w_d)), (xo[msk], wo[msk])):
                mu, v = x.mean(), x.var()
                if v <= mu * 1.0001 or mu == 0: lam = np.full((R, msk.sum()), mu)
                else:
                    k = mu * mu / (v - mu); lam = rng.gamma(k + x, 1.0 / (k / mu + 1.0), size=(R, msk.sum()))
                Z = Z + x + rng.poisson((w - 1) * lam)
            out[:, msk] = f05(np.broadcast_to(tp[msk], (R, msk.sum())), Z, np.broadcast_to(n[msk], (R, msk.sum())))
        return out.mean(0)

    def evaluate_D(self, D):
        if "label" not in D.columns:
            D = D.select("qi", "di").join(self.Vx.select("qi", "di", "label", "grp"), on=["qi", "di"])
        T = self.per_s1(D)
        F = self.f_est(T, np.random.default_rng(42))
        d = T.select("di", "country", "half").with_columns(pl.Series("F", F))
        r = {(h, c): f for h, c, f in d.group_by("half", "country").agg(pl.col("F").mean()).iter_rows()}
        out = dict(r)
        for h in ("TUNE", "REPORT"):
            out[(h, "mix")] = sum(r[(h, c)] * self.n_s1[c] for c in self.n_s1) / sum(self.n_s1.values())
        return out

    def grid(self, X, P):
        """X: hold-out table (qi, di, src, p, inzone); P: combiner predictions (qi, di, pnew). Returns the (t, m) grid."""
        zq = X.filter(pl.col("inzone") == 1).select("qi").unique()
        Y = X.select("qi", "di", "src", "p").join(P, on=["qi", "di"], how="left")
        rows = []
        for t in GRID_T:
            for m in GRID_M:
                r = self.evaluate_D(hybrid_decide(Y, zq, t, m))
                rows.append({"t": t, "m": m, **{f"{h}_{c}": r[(h, c)] for h in ("TUNE", "REPORT") for c in (*self.n_s1, "mix")}})
        return pl.DataFrame(rows).sort("TUNE_mix", descending=True)
