"""BM25 name / address candidates, top-K per record, per country (open set).
usage: python retrieve_sparse.py <q_items> <d_items> <K> <tag>   -> WORK_DIR/data/cand/<tag>/bm25_{name,addr}.<q_items>.<country>.npz"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, polars as pl
import config
from common.sparse_retrieval import retrieve

W = str(config.WORK_DIR)
qit, dit, K, tag = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
OUT = f"{W}/data/cand/{tag}"; os.makedirs(OUT, exist_ok=True)
strip = lambda s: s.str.replace(r"^(query|passage): ", "")
q = pl.read_parquet(f"{W}/data/texts/{qit}.parquet", columns=["country", "text_name", "text_addr"]).with_row_index("qi")
d = pl.read_parquet(f"{W}/data/texts/{dit}.parquet", columns=["country", "text_name", "text_addr"]).with_row_index("di")
for c in q["country"].unique().sort().to_list():
    qc, dc = q.filter(pl.col("country") == c), d.filter(pl.col("country") == c)
    dids = dc["di"].to_numpy()
    if len(dids) == 0:   # a country with no Source 1 entity: its records have no candidates
        for ch in ("bm25_name", "bm25_addr"):
            np.savez(f"{OUT}/{ch}.{qit}.{c}.npz", qi=qc["qi"].to_numpy(), idx=np.full((qc.height, K), -1, np.int32),
                     score=np.zeros((qc.height, K), np.float16))
        continue
    for ch, field in (("bm25_name", "text_name"), ("bm25_addr", "text_addr")):
        idx, sc, st = retrieve(strip(qc[field]).to_list(), strip(dc[field]).to_list(), "bm25", K)
        np.savez(f"{OUT}/{ch}.{qit}.{c}.npz", qi=qc["qi"].to_numpy(),
                 idx=np.where(idx >= 0, dids[np.maximum(idx, 0)], -1).astype(np.int32), score=sc.astype(np.float16))
        print(f"{ch} {qit} {c}: {qc.height:,} q x {dc.height:,} d  build {st['build_s']:.0f}s search {st['search_s']:.0f}s", flush=True)
