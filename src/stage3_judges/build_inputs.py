"""Plain and context judge inputs for the uncertainty zone of the countries with training labels (CPU).

For every zone record, its cross-encoder top-3 Source 1 candidates are judged. text_a = Source 1 text, text_b = record text,
both "business_name | business_address". Rows are ordered by tier (records with top-1 pn1 in [0.005, 0.995) or a low margin
first), then qi, di, and finally by di with that order kept inside each di. One file per country: s1_id = di, cand_id = qi.

Context inputs (*_context) add the record's competing Source 1 candidates to text_b:
  text_b + " || other candidates: " + the text_a of the record's other judged Source 1 candidates (ordered by di, at most 3,
  each cut to 90 characters, joined by " ; "), or "none" when the record has no other judged candidate.
They are built for every plain input file in WORK_DIR/stage3/inputs (test, and the hold-out inputs when TRAIN=1 retrains the
combiner).

usage: python -m stage3_judges.build_inputs
Reads WORK_DIR/stage2/test_candidates.parquet, test_record_index.parquet, test_source1_index.parquet and the test TSVs;
writes WORK_DIR/stage3/inputs/test_<country>.parquet for every country with training labels (test_us, test_india).
"""
import polars as pl

import config
from common.country_route import UNLABELLED, labelled_branches, with_branch
from common.io import read_tsv
from stage3_judges.texts import text


OTHER_CUT, MAX_OTHERS = 90, 3


def context(plain):
    """plain: s1_id, cand_id, text_a, text_b (one row per judged pair). Returns the same rows with the context text_b."""
    P = plain.with_row_index("_row").with_columns(pl.col("s1_id").cast(pl.Int64).alias("_di"))
    O = P.select("cand_id", "_di", pl.col("text_a").str.slice(0, OTHER_CUT).alias("_t"))
    pairs = P.select("_row", "cand_id", "_di").join(O, on="cand_id", suffix="_o").filter(pl.col("_di") != pl.col("_di_o"))
    others = pairs.sort(["_row", "_di_o"]).group_by("_row", maintain_order=True).agg(pl.col("_t").head(MAX_OTHERS).str.join(" ; ").alias("_others"))
    P = P.join(others, on="_row", how="left").sort("_row")
    return P.select("s1_id", "cand_id", "text_a",
                    (pl.col("text_b") + " || other candidates: " + pl.col("_others").fill_null("none")).alias("text_b"))


def build():
    d2 = config.WORK_DIR / "stage2"
    X = with_branch(pl.read_parquet(d2 / "test_candidates.parquet", columns=["qi", "di", "ce2_rank", "ce2_pnull", "country"]), config.TEST_DIR)
    X = X.filter(pl.col("branch") != UNLABELLED)
    t1 = X.filter(pl.col("ce2_rank") == 1).select("qi", pl.col("ce2_pnull").alias("pn1"))
    t2 = X.filter(pl.col("ce2_rank") == 2).select("qi", pl.col("ce2_pnull").alias("pn2"))
    T = t1.join(t2, on="qi", how="left").with_columns(pl.col("pn2").fill_null(0.0))
    lowm = (pl.col("pn1") >= 0.3) & (pl.col("pn1") - pl.col("pn2") < 0.5)
    z = T.filter(((pl.col("pn1") >= 0.001) & (pl.col("pn1") < 0.999)) | lowm).with_columns(
        pl.when(((pl.col("pn1") >= 0.005) & (pl.col("pn1") < 0.995)) | lowm).then(0).otherwise(1).alias("tier"))
    P = X.join(z.select("qi", "tier"), on="qi").filter(pl.col("ce2_rank") <= 3).select("qi", "di", "tier", "country", "branch")
    qm = pl.read_parquet(d2 / "test_record_index.parquet").select("qi", pl.col("entity_id").alias("vid"))
    dm = pl.read_parquet(d2 / "test_source1_index.parquet").select("di", pl.col("entity_id").alias("sid"))
    rd = lambda f: read_tsv(config.TEST_DIR / f, columns=["entity_id", "business_name", "business_address"])
    txt = lambda d: d.select("entity_id", text(pl.col("business_name"), pl.col("business_address")).alias("text"))
    ta = txt(rd("test_source1.tsv")).rename({"entity_id": "sid", "text": "text_a"})
    tb = txt(pl.concat([rd("test_source2.tsv"), rd("test_source3.tsv")])).rename({"entity_id": "vid", "text": "text_b"})
    P = P.join(qm, on="qi").join(dm, on="di").join(ta, on="sid", how="left").join(tb, on="vid", how="left")
    return P.sort("tier", "qi", "di")


if __name__ == "__main__":
    P = build(); out = config.WORK_DIR / "stage3" / "inputs"; out.mkdir(parents=True, exist_ok=True)
    for c, name in ((c, f"test_{c.lower()}") for c in labelled_branches()):
        D = P.filter(pl.col("branch") == c).sort("di", maintain_order=True).select(
            pl.col("di").cast(pl.Utf8).alias("s1_id"), pl.col("qi").cast(pl.Utf8).alias("cand_id"), "text_a", "text_b")
        D.write_parquet(out / f"{name}.parquet"); print(name, D.height, flush=True)
    for f in sorted(out.glob("*.parquet")):
        name = f.stem
        if not name.endswith("_context"):
            C = context(pl.read_parquet(f)); C.write_parquet(out / f"{name}_context.parquet"); print(f"{name}_context", C.height, flush=True)
