"""String and retrieval features for every (record, Source 1) candidate pair.

usage: python features.py <q_items> <d_items> <cand_tag>
Reads WORK_DIR/data/cand/<tag>/pairs.<q_items>.<country>.parquet, writes WORK_DIR/data/feat/<tag>/<q_items>.<country>.parquet
Country is never a feature (France has no training labels).
"""
import os, sys, re, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from multiprocessing import Pool
import numpy as np, polars as pl
from rapidfuzz import fuzz
from rapidfuzz.process import cpdist
import config

W = str(config.WORK_DIR)
qit, dit, tag = sys.argv[1:4]
OUT = f"{W}/data/feat/{tag}"
os.makedirs(OUT, exist_ok=True)
SMAP = json.load(open(f"{W}/data/state_map.json"))
# Legal forms, filler words and country names, removed to form the core name ("Acme India Pvt Ltd" -> "acme").
CORE_DROP = set("""inc incorporated llc l.l.c ltd limited pvt private corp corporation co company pc p.c pllc lp llp plc
sas sarl sa sci sasu eurl s.a.s s.a s.a.r.l ets services service center centre partners group groupe holdings holding
the and & of france india usa international enterprises""".split())
LEET = str.maketrans("013457", "oleast")
_MIX = re.compile(r"(?=\S*\d)(?=\S*[a-z])\S+")


def core(s):
    return " ".join(t for t in s.split() if t.strip(".,") not in CORE_DROP)


def leet(s):
    return _MIX.sub(lambda m: m.group(0).translate(LEET), s)


def meta(items):
    """normalised fields aligned to the rows of data/texts/<items>.parquet"""
    t = pl.read_parquet(f"{W}/data/texts/{items}.parquet", columns=["id", "country"] + (["src"] if items != dit else []))
    if items.startswith("s1_"):
        sp = items.split("_")[1]
        m = pl.read_parquet(f"{W}/data/norm/{sp}_s1.parquet").rename({"entity_id": "id"})
        m = m.with_columns(pl.lit(1).alias("src"))
    else:
        sp = "test" if items.startswith("test") else "train"
        m = pl.concat([pl.read_parquet(f"{W}/data/norm/{sp}_s{i}.parquet") for i in (2, 3)]).rename({"entity_id": "id"})
    m = m.select("id", "name_latin", "addr_clean", "nums", "is_alias", "is_domain", "is_indic", "addr_null")
    return t.join(m, on="id", how="left", maintain_order="left")


def _py_feats(args):
    qa, da, qn, dn, ctry = args
    out = np.zeros((len(qa), 9), np.float32)
    for i in range(len(qa)):
        a, b = qa[i], da[i]
        qt, dt = set(a.replace(",", " ").split()), set(b.replace(",", " ").split())
        out[i, 0] = len(qt & dt) / len(qt) if qt else -1                       # share of record address tokens in the Source 1 address
        x, y = qn[i].split(), dn[i].split()
        if x and y:
            m1 = x[0] if x else ""
            out[i, 1] = -1 if not x else float(m1 in y)                        # record main number appears in Source 1
            out[i, 2] = float(x[0] == y[0])                                    # main numbers equal
            out[i, 3] = float(bool(set(x) & set(y)))                           # any number shared
            out[i, 4] = float(any((u != v) and (u.startswith(v) or v.startswith(u) or u.endswith(v) or v.endswith(u))
                                  for u in x[:2] for v in y[:2]))              # prefix / suffix (truncated numbers)
        else:
            out[i, 1:5] = -1
        # state_match: record components known to the state table -> mapped value present in the Source 1 components?
        dc = set(c.strip() for c in b.split(","))
        known = [SMAP[k] for k in (f"{ctry[i]}|{c.strip()}" for c in a.split(",")) if k in SMAP] if a else []
        out[i, 5] = -1 if not known else sum(v in dc for v in known) / len(known)
        out[i, 6] = len(dc)
        # numbers of the record that are absent from the Source 1 address (lookalikes often differ by one house number)
        xs, ys = set(x), set(y)
        out[i, 7] = len(xs - ys) if xs else -1
        out[i, 8] = len(xs - ys) / len(xs) if xs else -1
    return out


def main():
    q, d = meta(qit), meta(dit)
    # popularity in the Source 1 index (same name / same address), per country
    d = d.with_row_index("di")
    d = d.with_columns(pl.len().over(["country", "name_latin"]).alias("pop_name"),
                       pl.len().over(["country", "addr_clean"]).alias("pop_addr"))
    lab = None
    tq = pl.read_parquet(f"{W}/data/texts/{qit}.parquet")
    if "s1" in tq.columns:
        lab = tq.select(pl.col("s1")).with_row_index("qi").join(d.select(pl.col("id").alias("s1"), pl.col("di").alias("true_di")),
                                                                 on="s1", how="left")
    q = q.with_row_index("qi")
    pool = Pool(18)
    for f in sorted(os.listdir(f"{W}/data/cand/{tag}")):
        if not (f.startswith(f"pairs.{qit}.") and f.endswith(".parquet")):
            continue
        c = f[len(f"pairs.{qit}."):-len(".parquet")]   # the country label (may itself contain dots)
        t = time.time()
        P = pl.read_parquet(f"{W}/data/cand/{tag}/{f}")
        # record-side competition (same for validation and test: every record searches the full index)
        P = P.with_columns(
            pl.col("cos_f").rank("ordinal", descending=True).over("qi").alias("rank_f"),
            pl.col("cos_n").rank("ordinal", descending=True).over("qi").alias("rank_n"),
            pl.col("cos_a").rank("ordinal", descending=True).over("qi").alias("rank_a"),
            (pl.col("cos_f").max().over("qi") - pl.col("cos_f")).alias("gap_f"),
            (pl.col("cos_n").max().over("qi") - pl.col("cos_n")).alias("gap_n"),
            (pl.col("cos_a").max().over("qi") - pl.col("cos_a")).alias("gap_a"),
            (pl.col("cos_f").max().over("qi") - pl.col("cos_f").top_k(2).min().over("qi")).alias("gap12_f"),
            pl.len().over("qi").alias("n_cand"))
        P = P.join(q.select("qi", "src", pl.col("name_latin").alias("qn"), pl.col("addr_clean").alias("qa"),
                            pl.col("nums").alias("qnum"), "is_alias", "is_domain", "is_indic", "addr_null"), on="qi", how="left")
        P = P.join(d.select("di", pl.col("name_latin").alias("dn"), pl.col("addr_clean").alias("da"),
                            pl.col("nums").alias("dnum"), "pop_name", "pop_addr"), on="di", how="left")
        qn, dn = P["qn"].to_list(), P["dn"].to_list()
        qa, da = P["qa"].to_list(), P["da"].to_list()
        feats = {
            "n_tset": cpdist(qn, dn, scorer=fuzz.token_set_ratio, workers=-1),
            "n_tsort": cpdist(qn, dn, scorer=fuzz.token_sort_ratio, workers=-1),
            "n_partial": cpdist(qn, dn, scorer=fuzz.partial_ratio, workers=-1),
            "n_ratio": cpdist(qn, dn, scorer=fuzz.ratio, workers=-1),
            "n_nospace": cpdist([x.replace(" ", "") for x in qn], [x.replace(" ", "") for x in dn], scorer=fuzz.ratio, workers=-1),
            "n_core_tset": cpdist([core(x) for x in qn], [core(x) for x in dn], scorer=fuzz.token_set_ratio, workers=-1),
            "n_leet_tsort": cpdist([leet(x) for x in qn], dn, scorer=fuzz.token_sort_ratio, workers=-1),
            "a_tset": cpdist(qa, da, scorer=fuzz.token_set_ratio, workers=-1),
            "a_ratio": cpdist(qa, da, scorer=fuzz.token_sort_ratio, workers=-1),
        }
        qnum, dnum = P["qnum"].to_list(), P["dnum"].to_list()
        ctry = [c] * len(qa)
        CH = 200_000
        py = np.concatenate(pool.map(_py_feats, [(qa[i:i + CH], da[i:i + CH], qnum[i:i + CH], dnum[i:i + CH], ctry[i:i + CH])
                                                 for i in range(0, len(qa), CH)]))
        names = ["a_contain", "num_main_in", "num_main_eq", "num_any", "num_prefix", "state_match", "d_ncomp", "num_missing", "num_missing_frac"]
        P = P.with_columns([pl.Series(k, v.astype(np.float32)) for k, v in feats.items()] +
                           [pl.Series(n, py[:, j]) for j, n in enumerate(names)] + [
            (pl.col("is_alias").cast(pl.Boolean) & (pl.col("qn") == pl.col("dn"))).cast(pl.Int8).alias("alias_eq"),
            (pl.col("is_domain").cast(pl.Boolean) & (pl.col("qn").str.replace_all(" ", "") == pl.col("dn").str.replace_all(" ", ""))
             ).cast(pl.Int8).alias("domain_eq"),
            (pl.col("qnum") == "").cast(pl.Int8).alias("q_no_nums"),
            pl.col("qn").str.len_chars().alias("len_qn"), pl.col("dn").str.len_chars().alias("len_dn"),
            pl.col("qa").str.len_chars().alias("len_qa"), pl.col("da").str.len_chars().alias("len_da")])
        if lab is not None:
            P = P.join(lab.select("qi", "true_di"), on="qi", how="left").with_columns(
                (pl.col("di") == pl.col("true_di")).fill_null(False).cast(pl.Int8).alias("label")).drop("true_di")
        P = P.drop("qn", "dn", "qa", "da", "qnum", "dnum")
        P.write_parquet(f"{OUT}/{qit}.{c}.parquet")
        print(f"features {qit} {c}: {P.height:,} pairs x {P.width} cols in {time.time() - t:.0f}s"
              + (f"; positives {P['label'].sum():,}" if lab is not None else ""), flush=True)
    pool.close()


if __name__ == "__main__":
    main()
