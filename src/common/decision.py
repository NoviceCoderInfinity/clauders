"""Vendor-to-Source-1 decision rule.

Each Source 2 / Source 3 record (qi) goes to its highest-p Source 1 candidate (di) if p >= t and p exceeds the record's
second-best candidate by at least m; then per Source 1 entity keep the top cap2 Source 2 and top cap3 Source 3 records by p.
"""
import polars as pl


def decide_m(P, t, m=0.0, cap2=5, cap3=6):
    """P: qi, di, src, p  ->  accepted (qi, di)."""
    srt = P.sort(["qi", "p"], descending=[False, True])
    k = srt.with_columns(pl.col("p").cum_count().over("qi").alias("_k"))
    best = k.filter(pl.col("_k") == 1).join(k.filter(pl.col("_k") == 2).select("qi", pl.col("p").alias("_p2")), on="qi", how="left")
    best = best.with_columns(pl.col("_p2").fill_null(0.0)).filter((pl.col("p") >= t) & (pl.col("p") - pl.col("_p2") >= m))
    best = best.with_columns(pl.col("p").rank("ordinal", descending=True).over(["di", "src"]).alias("_r"))
    return best.filter(((pl.col("src") == 2) & (pl.col("_r") <= cap2)) | ((pl.col("src") == 3) & (pl.col("_r") <= cap3))).select("qi", "di")
