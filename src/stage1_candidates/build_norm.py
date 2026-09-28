"""Normalise all six source files; fit the transliteration and state tables on training pairs (ENC + MT Source 1 only);
add name_latin.

Outputs:
  WORK_DIR/data/norm/{train,test}_s{1,2,3}.parquet
     entity_id, country, name_clean, name_latin, addr_clean, nums, is_alias, is_domain, is_indic, addr_null
  WORK_DIR/data/translit.json, WORK_DIR/data/state_map.json
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from collections import Counter, defaultdict
from multiprocessing import Pool
import polars as pl
import config
from common.normalize import norm_name, norm_addr, to_latin

WORK = str(config.WORK_DIR)
PQ, NORM = f"{WORK}/data/pq", f"{WORK}/data/norm"
os.makedirs(NORM, exist_ok=True)
FILES = [f"{sp}_s{i}" for sp in ("train", "test") for i in (1, 2, 3)]


def _work(args):
    names, addrs = args
    out = []
    for n, a in zip(names, addrs):
        nc, al, dm, ind = norm_name(n)
        ac, nums, an = norm_addr(a)
        out.append((nc, ac, nums, al, dm, ind, an))
    return out


def normalise(df, pool):
    names, addrs = df["business_name"].to_list(), df["business_address"].to_list()
    CH = 20000
    chunks = [(names[i:i + CH], addrs[i:i + CH]) for i in range(0, len(names), CH)]
    rows = [r for part in pool.imap(_work, chunks) for r in part]
    cols = list(zip(*rows)) if rows else [[]] * 7
    return df.select("entity_id", "country").with_columns(
        pl.Series("name_clean", cols[0], pl.Utf8), pl.Series("addr_clean", cols[1], pl.Utf8),
        pl.Series("nums", cols[2], pl.Utf8), pl.Series("is_alias", cols[3], pl.Int8),
        pl.Series("is_domain", cols[4], pl.Int8), pl.Series("is_indic", cols[5], pl.Int8),
        pl.Series("addr_null", cols[6], pl.Int8))


def main():
    with Pool(18) as pool:
        for f in FILES:
            df = pl.read_parquet(f"{PQ}/{f}.parquet")
            out = normalise(df, pool)
            out.write_parquet(f"{NORM}/{f}.parquet")
            print(f"normalised {f}: {out.height:,}", flush=True)

    # ---------------- tables fitted on training pairs (ENC + MT Source 1) only; HOLD kept for checking
    split = pl.read_parquet(f"{WORK}/data/split/s1_split.parquet").select(pl.col("entity_id").alias("s1"), "split")
    pairs = pl.read_parquet(f"{PQ}/train_pairs.parquet").join(split, on="s1")
    s1 = pl.read_parquet(f"{NORM}/train_s1.parquet").select(pl.col("entity_id").alias("s1"),
                                                            pl.col("name_clean").alias("n1"), pl.col("addr_clean").alias("a1"))
    s23 = pl.concat([pl.read_parquet(f"{NORM}/train_s{i}.parquet") for i in (2, 3)]).select(
        pl.col("entity_id").alias("m"), pl.col("name_clean").alias("nm"), pl.col("addr_clean").alias("am"), "is_indic", "country")
    P = pairs.join(s1, on="s1").join(s23, on="m")
    fit, hold = P.filter(pl.col("split") != "HOLD"), P.filter(pl.col("split") == "HOLD")

    # transliteration: positional token alignment where token counts agree, majority vote per script token
    cnt = defaultdict(Counter)
    for nm, n1 in fit.filter(pl.col("is_indic") == 1).select("nm", "n1").iter_rows():
        a, b = nm.split(" "), n1.split(" ")
        if len(a) == len(b):
            for x, y in zip(a, b):
                cnt[x][y] += 1
    tmap = {x: c.most_common(1)[0][0] for x, c in cnt.items()}
    json.dump(tmap, open(f"{WORK}/data/translit.json", "w"), ensure_ascii=False)
    print(f"\ntranslit table: {len(tmap):,} script tokens")
    h = hold.filter(pl.col("is_indic") == 1)
    lat = [to_latin(x, tmap) for x in h["nm"].to_list()]
    exact = sum(a == b for a, b in zip(lat, h["n1"].to_list()))
    print(f"HOLD Indic-script names: {h.height:,}; exact Source 1 name recovery {exact/max(h.height, 1):.1%}")

    # state table: frequent digit-free Source 2/3 components -> most co-occurring Source 1 component, keep if >=80%
    comp_n, co = Counter(), defaultdict(Counter)
    # keyed by "country|component": codes clash across countries (TN = Tennessee / Tamil Nadu)
    for am, a1, ctry in fit.filter(pl.col("am") != "").select("am", "a1", "country").iter_rows():
        c1 = set(a1.split(", "))
        for c in set(am.split(", ")):
            if not any(ch.isdigit() for ch in c):
                comp_n[(ctry, c)] += 1
                co[(ctry, c)].update(c1)
    smap = {}
    for key, n in comp_n.items():
        if n >= 500:
            c = key[1]
            t, k = co[key].most_common(1)[0]
            if co[key][c] / n >= 0.80:      # identity wins when it also clears the bar (cities etc.)
                t, k = c, co[key][c]
            if k / n >= 0.80:
                smap[f"{key[0]}|{c}"] = t
    json.dump(smap, open(f"{WORK}/data/state_map.json", "w"), ensure_ascii=False)
    nonid = {k: v for k, v in smap.items() if k.split("|", 1)[1] != v}
    print(f"state table: {len(smap):,} entries ({len(nonid):,} non-identity)")

    # add name_latin to every file
    for f in FILES:
        df = pl.read_parquet(f"{NORM}/{f}.parquet")
        df = df.with_columns(pl.col("name_clean").map_elements(lambda s: to_latin(s, tmap), return_dtype=pl.Utf8).alias("name_latin"))
        df.select("entity_id", "country", "name_clean", "name_latin", "addr_clean", "nums",
                  "is_alias", "is_domain", "is_indic", "addr_null").write_parquet(f"{NORM}/{f}.parquet")
    print("name_latin added", flush=True)


if __name__ == "__main__":
    main()
