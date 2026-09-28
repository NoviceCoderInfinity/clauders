"""Canonicaliser for the branch of countries without training labels (common/country_route.UNLABELLED).

Runs on every test record whose country has no training labels (on this test set: France). Rows of the
countries with training labels are copied through unchanged.

What the rewrite does, and nothing else. An address is split on ", ". In each piece, street-type
abbreviations are expanded (R. -> rue, BD -> boulevard, PL -> place, AV -> avenue, RTE -> route),
"N°" is dropped, and a house number written in parentheses is lifted out of the parentheses.
On the name, the decoration "(FRANCE)" and the abbreviation "Ets" (établissements) are removed.
These are generic French spelling rules, written after reading the unlabelled test inputs of the
country without training data; no table of place names is used. The row order is kept, so the candidate indexes qi and
di do not move.

Builds a parallel work dir FRANCE_WORK_DIR (default WORK_DIR/france) whose test norm/texts differ from the main one only on unlabelled-branch
rows (same row order, so qi/di are unchanged); the other files are symlinked (the rules never fire outside that branch, so no model
is retrained). Then the retrieval / candidate / feature steps are rerun there (see run_stage1.sh).
usage: python france_canon.py
"""
import os, sys, re
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import polars as pl
import config
from common.country_route import UNLABELLED, with_branch

W = str(config.WORK_DIR)
WF = os.environ.get("FRANCE_WORK_DIR", f"{W}/france")
ABBR = [(re.compile(r"(?<![\w.])r\.?(?=\s)"), "rue"), (re.compile(r"(?<![\w.])bd\.?(?=\s)"), "boulevard"),
        (re.compile(r"(?<![\w.])pl\.?(?=\s)"), "place"), (re.compile(r"(?<![\w.])av\.?(?=\s)"), "avenue"),
        (re.compile(r"(?<![\w.])rte\.?(?=\s)"), "route"), (re.compile(r"n°\s*"), ""), (re.compile(r"\((\d+[a-z]?)\)"), r"\1")]
NAME_DROP = re.compile(r"\(\s*france\s*\)|(?<!\w)ets\.?(?!\w)")


def fr_addr(a):
    """Expand French street-type abbreviations in every comma-separated address piece."""
    if not a: return a
    out = []
    for c in a.split(", "):
        c = c.strip()
        for r, s in ABBR: c = r.sub(s, c)
        c = re.sub(r"\s+", " ", c).strip()
        if c: out.append(c)
    return ", ".join(out)


def fr_name(n):
    """Drop the "(FRANCE)" decoration and the "Ets" abbreviation from a business name."""
    if not n: return n
    return re.sub(r"\s+", " ", NAME_DROP.sub(" ", n)).strip() or n


if __name__ == "__main__":
    for d in ("norm", "texts"): os.makedirs(f"{WF}/data/{d}", exist_ok=True)
    for x in os.listdir(f"{W}/data"):
        if x not in ("norm", "texts", "emb", "cand", "feat", "reranker", "cross_encoder") and not os.path.exists(f"{WF}/data/{x}"):
            os.symlink(f"{W}/data/{x}", f"{WF}/data/{x}")
    for f in os.listdir(f"{W}/data/norm"):
        src, dst = f"{W}/data/norm/{f}", f"{WF}/data/norm/{f}"
        if f.startswith("test_"):
            d = with_branch(pl.read_parquet(src), config.TEST_DIR); fr = pl.col("branch") == UNLABELLED
            d = d.with_columns(pl.when(fr).then(pl.col("addr_clean").map_elements(fr_addr, return_dtype=pl.String)).otherwise(pl.col("addr_clean")).alias("addr_clean"),
                               pl.when(fr).then(pl.col("name_clean").map_elements(fr_name, return_dtype=pl.String)).otherwise(pl.col("name_clean")).alias("name_clean"),
                               pl.when(fr).then(pl.col("name_latin").map_elements(fr_name, return_dtype=pl.String)).otherwise(pl.col("name_latin")).alias("name_latin"))
            d.drop("branch").write_parquet(dst)
            f0 = pl.read_parquet(src).join(d.select("entity_id", "branch"), on="entity_id").filter(pl.col("branch") == UNLABELLED)
            f1 = d.filter(pl.col("branch") == UNLABELLED)
            print(f, "unlabelled-branch rows", f1.height, "addr changed", (f0["addr_clean"] != f1["addr_clean"]).sum(), "name changed", (f0["name_latin"] != f1["name_latin"]).sum())
        elif not os.path.exists(dst):
            os.symlink(src, dst)

    # test-side texts, same format as make_texts.py (only unlabelled-branch rows differ)
    def texts(df, p):
        return df.with_columns((pl.lit(p) + pl.col("name_latin") + " | " + pl.col("addr_clean")).alias("text_full"),
                               (pl.lit(p) + pl.col("name_latin")).alias("text_name"), (pl.lit(p) + pl.col("addr_clean")).alias("text_addr"))
    cols = ["id", "country", "text_full", "text_name", "text_addr"]
    s1 = texts(pl.read_parquet(f"{WF}/data/norm/test_s1.parquet").rename({"entity_id": "id"}), "passage: ").select(cols)
    ref = pl.read_parquet(f"{W}/data/texts/s1_test.parquet", columns=["id", "text_full"])
    assert (s1["id"] == ref["id"]).all(); s1.write_parquet(f"{WF}/data/texts/s1_test.parquet")
    t23 = pl.concat([pl.read_parquet(f"{WF}/data/norm/test_s{i}.parquet").with_columns(pl.lit(i).alias("src")) for i in (2, 3)])
    tq = texts(t23.rename({"entity_id": "id"}), "query: ").select(cols + ["src"])
    ref = pl.read_parquet(f"{W}/data/texts/test_q.parquet", columns=["id", "text_full", "country"])
    assert (tq["id"] == ref["id"]).all(); tq.write_parquet(f"{WF}/data/texts/test_q.parquet")
    branched = with_branch(ref.select("id", "country"), config.TEST_DIR)
    nf = (branched["branch"] != UNLABELLED).to_list()
    print("test_q: labelled-branch rows identical", (tq.filter(nf)["text_full"] == ref.filter(nf)["text_full"]).all(),
          "unlabelled-branch rows changed", (tq.filter([not x for x in nf])["text_full"] != ref.filter([not x for x in nf])["text_full"]).mean())
    for f in ("s1_train", "pool_q"):
        if not os.path.exists(f"{WF}/data/texts/{f}.parquet"): os.symlink(f"{W}/data/texts/{f}.parquet", f"{WF}/data/texts/{f}.parquet")
    print("done")
