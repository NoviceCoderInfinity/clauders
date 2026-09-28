"""US and India decision on the uncertainty zone.

Inputs are candidate rows indexed by qi (a Source 2 / Source 3 record) and di (a Source 1 entity):
  zone rows       (qi, di, src, p)  every candidate of a zone record, p = combiner probability (0 when unscored)
  outside rows    (qi, di, src)     pairs accepted by the stage-2 matcher for records outside the zone
The decision:
  1. set-size prior (scale 0.5): the logit of an uncertain zone pair (0.03 <= p <= 0.97) moves by
     scale * (log H(n2 + d2, n3 + d3) / H(n2, n3) - c0), where (n2, n3) are the Source 1 entity's OTHER confident
     matches by source (outside-zone accepted pairs + zone pairs with p >= 0.5, each capped at 4), (d2, d3) is the
     candidate's own source, H is the training histogram of (Source 2, Source 3) match counts per Source 1 entity
     (set_size_prior.py), and c0 centres the shift (p-weighted mean over uncertain rows);
  2. rule: each zone record keeps its best Source 1 candidate when p >= 0.65 and p - (second best p) >= 0.40;
  3. caps: at most 5 Source 2 and 6 Source 3 records per Source 1 entity (outside-zone pairs rank first).
"""
import numpy as np
import polars as pl

T_ACCEPT, M_ACCEPT, PRIOR_SCALE = 0.65, 0.40, 0.5
CAP_S2, CAP_S3 = 5, 6


def logit(p):
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return np.log(p / (1 - p))


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def apply_prior(A, outside, H, scale=PRIOR_SCALE):
    """A: zone rows (qi, di, src, p). outside: (qi, di, src). Returns A with p moved by the set-size prior."""
    conf = pl.concat([outside.select("di", "src"), A.filter(pl.col("p") >= 0.5).select("di", "src")])
    tot = conf.group_by("di").agg(c2=(pl.col("src") == 2).sum(), c3=(pl.col("src") == 3).sum())
    X = A.join(tot, on="di", how="left", maintain_order="left").with_columns(
        pl.col("c2").fill_null(0), pl.col("c3").fill_null(0), (pl.col("src") == 2).alias("is2"), (pl.col("p") >= 0.5).alias("conf"))
    X = X.with_columns(n2o=(pl.col("c2") - (pl.col("conf") & pl.col("is2")).cast(pl.UInt32)).clip(0, 4),
                       n3o=(pl.col("c3") - (pl.col("conf") & ~pl.col("is2")).cast(pl.UInt32)).clip(0, 4))
    n2 = X["n2o"].to_numpy().astype(int); n3 = X["n3o"].to_numpy().astype(int); s2 = X["is2"].to_numpy().astype(int)
    g = np.log(H[n2 + s2, n3 + (1 - s2)] / H[n2, n3]); p = X["p"].to_numpy().astype(np.float64)
    unc = (p >= 0.03) & (p <= 0.97); w = p * unc; c0 = float((g * w).sum() / w.sum())
    return X.select("qi", "di", "src").with_columns(pl.Series("p", np.where(unc, sigmoid(logit(p) + scale * (g - c0)), p)))


def best_per_record(Z, t=T_ACCEPT, m=M_ACCEPT):
    """Each record's best Source 1 candidate, kept when p >= t and p - (second best) >= m."""
    srt = Z.sort(["qi", "p"], descending=[False, True]).with_columns(pl.col("p").cum_count().over("qi").alias("_k"))
    best = srt.filter(pl.col("_k") == 1).join(srt.filter(pl.col("_k") == 2).select("qi", pl.col("p").alias("_p2")), on="qi", how="left")
    return best.with_columns(pl.col("_p2").fill_null(0.0)).filter((pl.col("p") >= t) & (pl.col("p") - pl.col("_p2") >= m)).drop("_k", "_p2")


def apply_caps(D):
    D = D.with_columns(pl.col("p").rank("ordinal", descending=True).over(["di", "src"]).alias("_r"))
    return D.filter(((pl.col("src") == 2) & (pl.col("_r") <= CAP_S2)) | ((pl.col("src") == 3) & (pl.col("_r") <= CAP_S3))).drop("_r")


def decide(A, outside, H, scale=PRIOR_SCALE, t=T_ACCEPT, m=M_ACCEPT):
    """Full zone decision. Returns accepted (qi, di, src, p): outside-zone pairs (p = 1) + decided zone pairs, after caps."""
    if scale > 0:
        A = apply_prior(A, outside, H, scale)
    Dz = best_per_record(A, t, m).select("qi", "di", "src", "p")
    D = pl.concat([outside.select("qi", "di", "src").with_columns(pl.lit(1.0).alias("p")), Dz])
    return apply_caps(D)


def zone_inputs(candidates, zone_records, zone_p):
    """candidates: (qi, di, src, country, matcher_accept) for US and India; zone_records: (qi);
    zone_p: combiner output (qi, di, p) for zone rows. Returns (zone rows, outside accepted rows)."""
    inz = pl.col("qi").is_in(zone_records["qi"].implode())
    A = candidates.filter(inz).select("qi", "di", "src").join(zone_p.select("qi", "di", "p"), on=["qi", "di"], how="left") \
        .with_columns(pl.col("p").fill_null(0.0))
    outside = candidates.filter(~inz & (pl.col("matcher_accept") == 1)).select("qi", "di", "src")
    return A, outside
