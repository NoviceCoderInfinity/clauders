"""Sparse retrieval on CPU: BM25 (word level) and TF-IDF (char n-grams), multiprocess top-k.

retrieve(q_texts, d_texts, kind, k) -> (idx[n,k] int32 into d_texts, -1 pad; score[n,k] float32)
"""
import os, time
import numpy as np
import scipy.sparse as sp
from multiprocessing import Pool
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

_G = {}
TOK = r"[^\s,|]+"


def _topk_chunk(args):
    lo, hi, k = args
    S = (_G["Q"][lo:hi] @ _G["DT"]).tocsr()
    idx = np.full((hi - lo, k), -1, np.int32)
    sc = np.zeros((hi - lo, k), np.float32)
    for r in range(hi - lo):
        a, b = S.indptr[r], S.indptr[r + 1]
        if a == b:
            continue
        d, c = S.data[a:b], S.indices[a:b]
        if b - a > k:
            p = np.argpartition(-d, k)[:k]
            d, c = d[p], c[p]
        o = np.argsort(-d)
        idx[r, :len(o)] = c[o]
        sc[r, :len(o)] = d[o]
    return lo, idx, sc


def build(q_texts, d_texts, kind, df_cap):
    if kind == "bm25":
        cv = CountVectorizer(token_pattern=TOK, lowercase=False, dtype=np.float32)
        D = cv.fit_transform(d_texts).tocsc()
        N = D.shape[0]
        df = np.diff(D.indptr)
        keep = np.where(df <= df_cap * N)[0]        # very common tokens carry ~no BM25 weight; drop for speed
        D = D[:, keep].tocsr()
        df = df[keep]
        idf = np.log(1 + (N - df + 0.5) / (df + 0.5)).astype(np.float32)
        dl = np.asarray(D.sum(1)).ravel()
        k1, b = 1.2, 0.75
        D.data = D.data * (k1 + 1) / (D.data + k1 * (1 - b + b * np.repeat(dl / dl.mean(), np.diff(D.indptr))))
        D = D @ sp.diags(idf)
        vocab = {t: i for i, t in enumerate(np.array(cv.get_feature_names_out())[keep])}
        cq = CountVectorizer(token_pattern=TOK, lowercase=False, vocabulary=vocab, binary=True, dtype=np.float32)
        Q = cq.transform(q_texts)
    else:  # tfidf char_wb 2-4
        tv = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), lowercase=False, sublinear_tf=True, dtype=np.float32,
                             max_df=df_cap)
        D = tv.fit_transform(d_texts)
        Q = tv.transform(q_texts)
    return Q.tocsr(), D.T.tocsr()


def retrieve(q_texts, d_texts, kind, k=50, df_cap=0.01, procs=18, chunk=2000):
    t = time.time()
    Q, DT = build(q_texts, d_texts, kind, df_cap)
    _G["Q"], _G["DT"] = Q, DT
    t_build = time.time() - t
    n = Q.shape[0]
    idx = np.full((n, k), -1, np.int32)
    sc = np.zeros((n, k), np.float32)
    with Pool(procs) as pool:   # fork: workers share Q, DT copy-on-write
        for lo, i, s in pool.imap_unordered(_topk_chunk, [(lo, min(lo + chunk, n), k) for lo in range(0, n, chunk)]):
            idx[lo:lo + len(i)] = i
            sc[lo:lo + len(s)] = s
    return idx, sc, {"build_s": t_build, "search_s": time.time() - t - t_build, "q_per_s": n / (time.time() - t - t_build)}
