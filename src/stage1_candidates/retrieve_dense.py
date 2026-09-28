"""Exact GPU nearest-neighbour search, per country (open set: whatever countries appear).

usage: python retrieve_dense.py <embdir> <q_items> <d_items> <field> <K> <tag>
Saves WORK_DIR/data/cand/<tag>/dense_<field>.<q_items>.<country>.npz : qi, idx (global d row, -1 pad), score
"""
import os, sys, glob, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, polars as pl, torch
import config

embdir, qit, dit, field, K, tag = sys.argv[1:7]
K = int(K)
W = str(config.WORK_DIR)
OUT = f"{W}/data/cand/{tag}"
os.makedirs(OUT, exist_ok=True)


def load(items):
    fs = sorted(glob.glob(f"{embdir}/{items}.{field}.*-of-*.npy"), key=lambda f: int(f.split(".")[-2].split("-")[0]))
    return np.concatenate([np.load(f) for f in fs])


Q, D = load(qit), load(dit)
q = pl.read_parquet(f"{W}/data/texts/{qit}.parquet", columns=["id", "country"]).with_row_index("qi")
d = pl.read_parquet(f"{W}/data/texts/{dit}.parquet", columns=["id", "country"]).with_row_index("di")
assert len(Q) == q.height and len(D) == d.height, (len(Q), q.height, len(D), d.height)
dev = torch.device("cuda")
for c in q["country"].unique().sort().to_list():
    t = time.time()
    qc, dc = q.filter(pl.col("country") == c), d.filter(pl.col("country") == c)
    dids = dc["di"].to_numpy()
    if len(dids) == 0:   # a country with no Source 1 entity: its records have no candidates (retrieval stays inside the country)
        np.savez(f"{OUT}/dense_{field}.{qit}.{c}.npz", qi=qc["qi"].to_numpy(), idx=np.full((qc.height, K), -1, np.int32),
                 score=np.zeros((qc.height, K), np.float16))
        continue
    P = torch.from_numpy(D[dids].astype(np.float32)).to(dev)
    qi = qc["qi"].to_numpy()
    idx = np.empty((len(qi), K), np.int32); sc = np.empty((len(qi), K), np.float16)
    B = max(256, int(1.5e9 // (4 * P.shape[0])))       # keep the score block around 1.5 GB
    for lo in range(0, len(qi), B):
        x = torch.from_numpy(Q[qi[lo:lo + B]].astype(np.float32)).to(dev)
        s, i = torch.topk(x @ P.T, min(K, P.shape[0]), dim=1)
        idx[lo:lo + B] = dids[i.cpu().numpy()]
        sc[lo:lo + B] = s.cpu().numpy().astype(np.float16)
    np.savez(f"{OUT}/dense_{field}.{qit}.{c}.npz", qi=qi, idx=idx, score=sc)
    el = time.time() - t
    print(f"{field} {qit}->{dit} {c}: {len(qi):,} q x {len(dids):,} d in {el:.0f}s ({len(qi)/el:,.0f} q/s)", flush=True)
