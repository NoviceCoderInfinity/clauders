"""Candidate pairs per country (blocking).

usage: python merge_candidates.py <embdir> <q_items> <d_items> <cand_tag> [K=20] [delta=0.15]
Candidates = fine-tuned e5 full-text top-K Source 1 of the same country, minus pairs whose cosine is more than
`delta` below the record's best (chosen on train: -0.002 pt recall, 20 -> 3.4 pairs per record).
BM25 name/address ranks and scores are joined as features where the pair is in BM25's top-20.
Output: WORK_DIR/data/cand/<tag>/pairs.<q_items>.<country>.parquet
   qi, di, r_df, cos_f, cos_n, cos_a, r_bn, r_ba (0 = not in BM25 top-20), s_bn, s_ba
"""
import os, sys, glob, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, polars as pl, torch
import config

embdir, qit, dit, tag = sys.argv[1:5]
K = int(sys.argv[5]) if len(sys.argv) > 5 else 20
DELTA = float(sys.argv[6]) if len(sys.argv) > 6 else 0.15
W = str(config.WORK_DIR)
C = f"{W}/data/cand/{tag}"
dev = torch.device("cuda")


def load_emb(items, field):
    fs = sorted(glob.glob(f"{embdir}/{items}.{field}.*-of-*.npy"), key=lambda f: int(f.split(".")[-2].split("-")[0]))
    return np.concatenate([np.load(f) for f in fs])


q = pl.read_parquet(f"{W}/data/texts/{qit}.parquet", columns=["country"])
E = {}
for c in q["country"].unique().sort().to_list():
    t = time.time()
    z = np.load(f"{C}/dense_text_full.{qit}.{c}.npz")
    idx, sc = z["idx"][:, :K], z["score"][:, :K].astype(np.float32)
    keep = (sc[:, :1] - sc <= DELTA) & (idx >= 0)
    if not keep.any():   # a country with no Source 1 entity: no candidate pairs, so no pairs file
        print(f"{qit} {c}: no Source 1 entities of this country, no candidates", flush=True)
        continue
    rows, cols = np.nonzero(keep)
    P = pl.DataFrame({"qi": z["qi"][rows].astype(np.int32), "di": idx[rows, cols].astype(np.int32),
                      "r_df": (cols + 1).astype(np.int16), "cos_f": sc[rows, cols]})
    for ch in ("bn", "ba"):
        b = np.load(f"{C}/bm25_{'name' if ch == 'bn' else 'addr'}.{qit}.{c}.npz")
        n, k = b["idx"].shape
        B = pl.DataFrame({"qi": np.repeat(b["qi"], k).astype(np.int32), "di": b["idx"].ravel().astype(np.int32),
                          f"r_{ch}": np.tile(np.arange(1, k + 1, dtype=np.int16), n),
                          f"s_{ch}": b["score"].ravel().astype(np.float32)}).filter(pl.col("di") >= 0)
        P = P.join(B, on=["qi", "di"], how="left").with_columns(pl.col(f"r_{ch}").fill_null(0), pl.col(f"s_{ch}").fill_null(0))
    if not E:
        E = {(s, f): load_emb(it, f) for s, it in (("q", qit), ("d", dit)) for f in ("text_name", "text_addr")}
    qi, di = P["qi"].to_numpy(), P["di"].to_numpy()
    for f, nm in (("text_name", "cos_n"), ("text_addr", "cos_a")):
        out = np.empty(len(qi), np.float32)
        for lo in range(0, len(qi), 2_000_000):
            a = torch.from_numpy(E[("q", f)][qi[lo:lo + 2_000_000]].astype(np.float32)).to(dev)
            b = torch.from_numpy(E[("d", f)][di[lo:lo + 2_000_000]].astype(np.float32)).to(dev)
            out[lo:lo + 2_000_000] = (a * b).sum(1).cpu().numpy()
        P = P.with_columns(pl.Series(nm, out))
    P.sort(["qi", "r_df"]).write_parquet(f"{C}/pairs.{qit}.{c}.parquet")
    print(f"{qit} {c}: {len(z['qi']):,} records -> {P.height:,} pairs ({P.height / len(z['qi']):.2f}/record) in {time.time() - t:.0f}s",
          flush=True)
