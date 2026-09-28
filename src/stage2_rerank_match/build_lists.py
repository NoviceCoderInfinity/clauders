"""Listwise reranker training lists (training data only; no HOLD/MT records, no look_v/look_mt2 lookalikes).
  ENC records (enc_q, 2M sampled): retriever top-20 over all train Source 1 of the country, within 0.15 of best, first K=8.
  look_ce lookalikes: their candidate lists (pairs_pool_q), target = none.
Easy ENC lists (single candidate that is the true Source 1) are down-sampled to 30%.
Outputs WORK_DIR/data/reranker/lists.parquet: q, cands (list[str]), tgt (int, -1 none), kind ('enc'|'look'), country, src
        and WORK_DIR/data/reranker/pool_pairs.parquet: pool candidate pairs minus look_ce (scored for matcher training).
"""
import os, sys, glob
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, polars as pl
import config

W = str(config.WORK_DIR); K = 8
TAG = os.environ.get("CAND_TAG", "main")
O = f"{W}/data/reranker"; os.makedirs(O, exist_ok=True)
strip = lambda c: pl.col(c).str.replace(r"^(query|passage): ", "")
s1 = pl.read_parquet(f"{W}/data/texts/s1_train.parquet", columns=["id", "text_full"]).with_row_index("di").with_columns(strip("text_full"))
s1txt = s1["text_full"]
eq = pl.read_parquet(f"{W}/data/texts/enc_q.parquet", columns=["text_full", "s1", "country", "src"]).with_row_index("qi").with_columns(strip("text_full"))
eq = eq.join(s1.select(pl.col("id").alias("s1"), pl.col("di").alias("tdi")), on="s1")
rows = []
for f in sorted(glob.glob(f"{W}/data/cand/{TAG}/dense_text_full.enc_q.*.npz")):
    z = np.load(f); idx, sc = z["idx"], z["score"].astype(np.float32)
    keep = (sc[:, :1] - sc <= 0.15) & (idx >= 0)
    rows.append(pl.DataFrame({"qi": z["qi"].astype(np.uint32),
                              "cdi": [r[k][:K].tolist() for r, k in zip(idx, keep)]}))
E = pl.concat(rows).join(eq, on="qi")
tdi = E["tdi"].to_numpy(); cdi = E["cdi"].to_list()
tgt = np.array([c.index(t) if t in c else -1 for c, t in zip(cdi, tdi)])
E = E.with_columns(pl.Series("tgt", tgt), pl.col("cdi").list.len().alias("nc"))
easy = (E["nc"] == 1) & (E["tgt"] == 0)
rng = np.random.default_rng(42)
E = E.filter(~easy | pl.Series(rng.random(E.height) < 0.3))
print(f"ENC lists {E.height:,}; true-in-list {np.mean(E['tgt'].to_numpy() >= 0):.4f}; mean cands {E['nc'].mean():.2f}")
# lookalikes look_ce (same groups as build_cross_encoder_data.py)
tq = pl.read_parquet(f"{W}/data/texts/pool_q.parquet", columns=["text_full", "s1_split", "country", "src"]).with_row_index("qi")
r1 = np.random.default_rng(42); g = np.where(r1.random(tq.height) < 0.5, "look_mt", "look_v")
r2 = np.random.default_rng(47); sub = np.where(r2.random(tq.height) < 0.5, "look_ce", "look_mt2")
tq = tq.with_columns(pl.when(pl.col("s1_split").is_null()).then(pl.Series(g)).otherwise(pl.col("s1_split")).alias("grp"))
tq = tq.with_columns(pl.when(pl.col("grp") == "look_mt").then(pl.Series(sub)).otherwise(pl.col("grp")).alias("grp2"))
tq.select("qi", "grp", "grp2").write_parquet(f"{W}/data/split/pool_groups.parquet")
look = tq.filter(pl.col("grp2") == "look_ce").with_columns(strip("text_full"))
# candidate pairs of the pool (rows are in (qi, dense rank) order within each record)
P = pl.read_parquet(f"{W}/data/cross_encoder/pairs_pool_q.parquet", columns=["qi", "di"])
P = P.group_by("qi", maintain_order=True).agg(pl.col("di").head(K).alias("cdi"))
look = look.join(P, on="qi").with_columns(pl.lit(-1).alias("tgt"), pl.col("cdi").cast(pl.List(pl.Int64)))
print(f"lookalike lists {look.height:,}; mean cands {look['cdi'].list.len().mean():.2f}")
out = pl.concat([
    E.with_columns(pl.col("cdi").cast(pl.List(pl.Int64))).select(pl.col("text_full").alias("q"), "cdi", "tgt", pl.lit("enc").alias("kind"), "country", "src"),
    look.select(pl.col("text_full").alias("q"), "cdi", pl.col("tgt").cast(pl.Int64), pl.lit("look").alias("kind"), "country", "src")])
out = out.with_columns(pl.col("cdi").list.eval(pl.element()).alias("cdi"))
cands = [[s1txt[i] for i in c] for c in out["cdi"].to_list()]
out = out.with_columns(pl.Series("cands", cands)).drop("cdi")
out.write_parquet(f"{O}/lists.parquet")
print(out.group_by("kind", "country").len().sort("kind", "country").rows())

# pool candidate pairs minus look_ce lookalikes: reranker scores for matcher training and validation
g2 = pl.read_parquet(f"{W}/data/split/pool_groups.parquet").select("qi", "grp2")
PP = pl.read_parquet(f"{W}/data/cross_encoder/pairs_pool_q.parquet")
PP.join(g2, on="qi").filter(pl.col("grp2") != "look_ce").select("qi", "di").write_parquet(f"{O}/pool_pairs.parquet")
print("pool_pairs", PP.height)
