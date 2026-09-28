"""Combiner features for the US and India zone candidates. The same code runs on the hold-out split and on test.

Rows are candidates (qi = Source 2/3 record, di = Source 1 entity) of zone records, with the stage-2 columns
p (matcher probability), ce2 / ce3 / ce4 (cross-encoder logits) with *_rank, *_gap, *_pnull, src, and llm_logit (judge score).
Texts are "business_name | business_address" for both sides (text_a = Source 1, text_b = record).

Feature groups
  BASE        18 matcher, cross-encoder and judge features
  LIST        10 listwise features (value minus the best other candidate of the record, top-1 votes)
  TEXT        21 name / address evidence features, including same-name Source 1 siblings in the record's list
  LOOKALIKE_1  9 lookalike signatures (first-pass combiner)

UP_STEPS, the house-number offsets of lookalike records, is measured on the training labels: derive_up_steps.py recomputes it
from the hold-out split of the training data, and run_stage4.sh stops when the two differ.
"""
import re
import unicodedata

import numpy as np
import polars as pl
from rapidfuzz.distance import Indel

BASE = ["lp", "prk", "pgap", "p2nd", "ncand", "src", "llm_logit", "qrk", "qgap", "qpnull", "ce2", "ce2_rank", "ce2_gap", "ce2_pnull",
        "ce3", "ce3_pnull", "ce4", "ce4_pnull"]
LIST = ["ce3_rank", "ce3_gap", "ce4_rank", "ce4_gap", "qmarg", "pmarg", "ce2marg", "qsm", "n_top1", "lsum_z"]
_TEXT_ALL = ["vaddr", "v_nnum", "nm_eq", "nm_eq_raw", "sfx_eq", "punct_eq", "nsib", "nsib3", "vsib3", "num_cov", "num_first_eq", "post_eq",
             "tok_cov", "tok_jac", "d_sib_tok", "d_sib_num", "d_sib_llm", "d_sib_ce4", "d_sib_p", "d_sib_snz", "d_oth3_tok", "d_oth3_num",
             "s_top1_n", "d_sib_top1n"]
# counts of how many records point at a Source 1 entity are not used: on the hold-out split they separate hold-out Source 1
# entities from the filler entities of the reduced candidate pool
TEXT = [f for f in _TEXT_ALL if f not in ("d_sib_snz", "s_top1_n", "d_sib_top1n")]
LOOKALIKE_1 = ["same_st", "hshift", "sh_dec_up", "sh_dec_down", "trunc1", "dropfirst", "swap1", "r1combo", "noaddr_nsib"]
UP_STEPS = {1, 2, 3, 4, 5, 7, 9, 11, 13, 21}
LEGAL = set("inc incorporated llc corp corporation co company ltd limited lp llp pllc pc plc pvt private opc the".split())


# ---------------------------------------------------------------- base
def judge_feats(X):
    """llm_logit kept for the cross-encoder top 3 of each record; rank, gap and softmax share within the record."""
    X = X.with_columns(pl.when(pl.col("ce2_rank") <= 3).then(pl.col("llm_logit")).otherwise(None).alias("llm_logit"))
    return X.with_columns(pl.col("llm_logit").rank("ordinal", descending=True).over("qi").alias("qrk"),
                          (pl.col("llm_logit").max().over("qi") - pl.col("llm_logit")).alias("qgap"),
                          (pl.col("llm_logit").exp() / (pl.col("llm_logit").exp().sum().over("qi") + 1.0)).alias("qpnull"))


def matcher_feats(X):
    lg = (pl.col("p").clip(1e-6, 1 - 1e-6) / (1 - pl.col("p").clip(1e-6, 1 - 1e-6))).log()
    X = X.with_columns(lg.alias("lp"), pl.col("p").max().over("qi").alias("pmax"), pl.col("p").rank("ordinal", descending=True).over("qi").alias("prk"),
                       pl.len().over("qi").alias("ncand"))
    X = X.with_columns((pl.col("pmax") - pl.col("p")).alias("pgap"))
    p2nd = X.filter(pl.col("prk") == 2).select("qi", pl.col("p").alias("p2nd"))
    return X.join(p2nd, on="qi", how="left").with_columns(pl.col("p2nd").fill_null(0.0))


# ---------------------------------------------------------------- listwise
def _marg(c, g="qi"):
    """value minus the best OTHER value in the group (positive only for the top row)."""
    mx = pl.col(c).max().over(g); second = pl.col(c).top_k(2).min().over(g)
    return pl.when(pl.col(c) >= mx).then(pl.col(c) - second).otherwise(pl.col(c) - mx)


def add_listwise(Z):
    Z = Z.with_columns(_marg("llm_logit").alias("qmarg"), _marg("p").alias("pmarg"), _marg("ce2").alias("ce2marg"),
                       (pl.col("llm_logit") - pl.col("llm_logit").max().over("qi")).exp().alias("_e"))
    return Z.with_columns((pl.col("_e") / pl.col("_e").sum().over("qi")).alias("qsm"),
                          ((pl.col("ce2_rank") == 1).cast(pl.Int8) + (pl.col("ce3_rank") == 1).cast(pl.Int8) + (pl.col("ce4_rank") == 1).cast(pl.Int8)
                           + (pl.col("prk") == 1).cast(pl.Int8) + (pl.col("qrk") == 1).fill_null(False).cast(pl.Int8)).alias("n_top1"),
                          (pl.col("ce2") + pl.col("ce3") + pl.col("ce4")).alias("lsum_z")).drop("_e")


# ---------------------------------------------------------------- text evidence
SUFFIX = r"\b(llc|inc|ltd|pvt|private|limited|corp|corporation|co|company|pc|plc|llp|lp|partners|the)\b"


def _split(c, a, b):
    parts = c.str.splitn(" | ", 2)
    return parts.struct.field("field_0").alias(a), parts.struct.field("field_1").fill_null("").alias(b)


def _norm_name(c):
    return c.str.to_lowercase().str.replace_all(SUFFIX, "").str.replace_all(r"[^\p{L}\p{N}]", "")


def _toks(c):
    return c.str.to_lowercase().str.extract_all(r"[\p{L}]{3,}").list.unique()


def _nums(c):
    return c.str.extract_all(r"\d+").list.eval(pl.element().str.strip_chars_start("0")).list.eval(pl.element().filter(pl.element() != "")).list.unique()


def _first_num(c):
    return c.str.extract(r"(\d+)", 1).str.strip_chars_start("0")


def _oth_max(c, g):
    """max of c over OTHER rows of group g (null if none)."""
    mx = pl.col(c).max().over(g); second = pl.col(c).top_k(2).min().over(g); n = pl.col(c).count().over(g)
    return pl.when(n < 2).then(None).when(pl.col(c) >= mx).then(second).otherwise(mx)


def add_text(Z, texts):
    """texts: (qi, di, text_a, text_b). Adds TEXT columns (and placeholders for the unused Source 1 counts)."""
    Z = Z.with_columns(pl.lit(0).alias("s_nz_same"))
    s_top1 = pl.DataFrame({"di": [], "s_top1_n": []}, schema={"di": pl.Int32, "s_top1_n": pl.UInt32})
    T = texts.select("qi", "di", *_split(pl.col("text_b"), "vn", "va"), *_split(pl.col("text_a"), "sn", "sa"))
    T = T.with_columns(
        (pl.col("va").str.strip_chars() != "").cast(pl.Int8).alias("vaddr"),
        _norm_name(pl.col("vn")).alias("vnn"), _norm_name(pl.col("sn")).alias("snn"),
        pl.col("vn").str.to_lowercase().str.replace_all(r"\s+", " ").str.strip_chars().alias("_vl"),
        pl.col("sn").str.to_lowercase().str.replace_all(r"\s+", " ").str.strip_chars().alias("_sl"),
        pl.col("vn").str.to_lowercase().str.extract_all(SUFFIX).list.sort().list.join(" ").alias("_vs"),
        pl.col("sn").str.to_lowercase().str.extract_all(SUFFIX).list.sort().list.join(" ").alias("_ss"),
        pl.col("vn").str.replace_all(r"[\p{L}\p{N}\s]", "").alias("_vp"), pl.col("sn").str.replace_all(r"[\p{L}\p{N}\s]", "").alias("_sp"),
        _nums(pl.col("va")).alias("_vnum"), _nums(pl.col("sa")).alias("_snum"),
        _toks(pl.col("va")).alias("_vt"), _toks(pl.col("sa")).alias("_st"),
        _first_num(pl.col("va")).alias("_vf"), _first_num(pl.col("sa")).alias("_sf"))
    inter_n = pl.col("_vnum").list.set_intersection(pl.col("_snum")).list.len()
    inter_t = pl.col("_vt").list.set_intersection(pl.col("_st")).list.len()
    post = lambda c: c.list.eval(pl.element().filter(pl.element().str.len_chars().is_in([5, 6])))
    T = T.with_columns(
        pl.col("_vnum").list.len().alias("v_nnum"),
        (pl.col("vnn") == pl.col("snn")).cast(pl.Int8).alias("nm_eq"),
        (pl.col("_vl") == pl.col("_sl")).cast(pl.Int8).alias("nm_eq_raw"),
        (pl.col("_vs") == pl.col("_ss")).cast(pl.Int8).alias("sfx_eq"),
        (pl.col("_vp") == pl.col("_sp")).cast(pl.Int8).alias("punct_eq"),
        pl.when(pl.col("_vnum").list.len() > 0).then(inter_n / pl.col("_vnum").list.len()).otherwise(None).alias("num_cov"),
        pl.when(pl.col("_vf").is_null() | pl.col("_sf").is_null()).then(None).otherwise((pl.col("_vf") == pl.col("_sf")).cast(pl.Int8)).alias("num_first_eq"),
        pl.when(post(pl.col("_vnum")).list.len() > 0).then(
            (post(pl.col("_vnum")).list.set_intersection(post(pl.col("_snum"))).list.len() > 0).cast(pl.Int8)).otherwise(None).alias("post_eq"),
        pl.when(pl.col("_vt").list.len() > 0).then(inter_t / pl.col("_vt").list.len()).otherwise(None).alias("tok_cov"),
        pl.when(pl.col("_vt").list.len() > 0).then(
            inter_t / (pl.col("_vt").list.len() + pl.col("_st").list.len() - inter_t)).otherwise(None).alias("tok_jac"))
    T = T.select("qi", "di", "vaddr", "v_nnum", "nm_eq", "nm_eq_raw", "sfx_eq", "punct_eq", "num_cov", "num_first_eq", "post_eq", "tok_cov", "tok_jac", "snn")
    Z = Z.join(T, on=["qi", "di"], how="left").join(s_top1, on="di", how="left").with_columns(pl.col("s_top1_n").fill_null(0))
    Z = Z.with_columns(pl.col("vaddr").max().over("qi"), pl.col("v_nnum").max().over("qi"))
    Z = Z.with_columns((pl.len().over(["qi", "snn"]) - 1).alias("nsib"))
    Z = Z.with_columns(pl.when(pl.col("ce2_rank") <= 3).then(pl.col("snn")).otherwise(None).alias("_snn3"))
    Z = Z.with_columns(pl.when(pl.col("_snn3").is_not_null()).then(pl.len().over(["qi", "_snn3"]) - 1).otherwise(None).alias("nsib3"))
    Z = Z.with_columns((pl.col("nsib3").max().over("qi").fill_null(0) > 0).cast(pl.Int8).alias("vsib3"))
    Z = Z.with_columns(pl.col("tok_cov").fill_null(-1.0).alias("_tk"), pl.col("num_cov").fill_null(-1.0).alias("_nm"),
                       pl.col("llm_logit").alias("_ll"), pl.col("s_nz_same").cast(pl.Float64).alias("_sz"), pl.col("s_top1_n").cast(pl.Float64).alias("_s1"))
    for src, nm in (("_tk", "d_sib_tok"), ("_nm", "d_sib_num"), ("_ll", "d_sib_llm"), ("ce4", "d_sib_ce4"), ("p", "d_sib_p"), ("_sz", "d_sib_snz"),
                    ("_s1", "d_sib_top1n")):
        Z = Z.with_columns((pl.col(src) - _oth_max(src, ["qi", "_snn3"])).alias(nm))
    Z = Z.with_columns(pl.when(pl.col("ce2_rank") <= 3).then(pl.lit(1)).otherwise(None).alias("_in3"))
    Z = Z.with_columns((pl.col("_tk") - _oth_max("_tk", ["qi", "_in3"])).alias("d_oth3_tok"), (pl.col("_nm") - _oth_max("_nm", ["qi", "_in3"])).alias("d_oth3_num"))
    return Z.drop("_tk", "_nm", "_ll", "_sz", "_s1", "_snn3", "_in3", "snn")


# ---------------------------------------------------------------- lookalike signatures, first form
_STOP_1 = set(("street st road rd avenue ave av drive dr lane ln court ct way boulevard blvd place pl circle cir highway hwy parkway pkwy "
               "suite ste unit apt no floor flat plot near opp opposite shop house door").split())


def _n1(s):
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode().lower())).strip()


def _street_1(a):
    for p in (a or "").split(","):
        if re.search(r"\d", p):
            q = _n1(p); m = re.search(r"\d+", q)
            return (int(m.group()) if m else None), " ".join(w for w in q.split() if not re.search(r"\d", w) and w not in _STOP_1 and len(w) > 1)
    return None, None


def _row_1(ta, tb):
    sn, _, sa = (ta or "").partition(" | "); vn, _, va = (tb or "").partition(" | ")
    A = [w for w in _n1(sn).split() if w not in LEGAL]; B = [w for w in _n1(vn).split() if w not in LEGAL]
    sA, sB = set(A), set(B)
    swap = 0
    if A and len(A) == len(B) and len(sA - sB) == 1 and len(sB - sA) == 1:
        swap = int(Indel.normalized_similarity(next(iter(sA - sB)), next(iter(sB - sA))) < 0.6)
    gained = int(bool(sB - sA)) if sA and sB else 0
    h1, s1 = _street_1(sa); hv, sv = _street_1(va)
    same = np.nan; sh = np.nan; up = dn = tr = df_ = np.nan; r1 = np.nan
    if h1 is not None and hv is not None and s1 is not None and sv is not None:
        same = float(Indel.normalized_similarity(s1, sv) >= 0.75 if (s1 and sv) else 0)
        d = hv - h1; sh = float(max(-50, min(50, d)))
        up = float(d in UP_STEPS); dn = float(-d in UP_STEPS)
        tr = float(h1 >= 10 and hv == h1 // 10); df_ = float(h1 >= 10 and str(hv) == str(h1)[1:])
        r1 = float(same == 1 and d != 0 and (swap == 1 or gained == 1))
    return same, sh, up, dn, tr, df_, float(swap), r1, float(va.strip() == "")


def add_lookalike_1(Z, texts):
    t = texts.select("qi", "di", "text_a", "text_b")
    arr = np.array([_row_1(a, b) for a, b in zip(t["text_a"].to_list(), t["text_b"].to_list())], dtype=np.float64)
    F = t.select("qi", "di").with_columns([pl.Series(n, arr[:, i]) for i, n in enumerate(LOOKALIKE_1)])
    Z = Z.join(F, on=["qi", "di"], how="left")
    return Z.with_columns((pl.col("noaddr_nsib") * (pl.col("nsib").cast(pl.Float64) + 1)).alias("noaddr_nsib"))
