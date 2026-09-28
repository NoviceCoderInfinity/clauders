"""Fine-tune intfloat/multilingual-e5-small for Source 2/3 -> Source 1 retrieval.

usage: python train_retriever.py --data WORK_DIR/data/encoder_train --out <dir>   (final model: <dir>/phase2)
Phase 1: multiple-negatives ranking loss (in-batch negatives) on a 3M-pair subsample, batches from one country and state.
Mining: phase-1 model retrieves top-10 Source 1 for 1M training queries; highest-ranked non-true Source 1 = hard negative.
Phase 2: the same loss on (query, positive, hard negative) triples.
Evaluation after each phase: recall@{1,5,10,20} of HOLD records against ALL train Source 1 of the same country.
Checkpoints every ~20 min and at phase end to <out>/latest; a restarted run resumes from there.
"""
import os, sys, json, time, math, shutil, argparse, random
import numpy as np, polars as pl, torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--model", default="intfloat/multilingual-e5-small")
ap.add_argument("--p1_pairs", type=int, default=3_000_000)
ap.add_argument("--hn_queries", type=int, default=1_000_000)
ap.add_argument("--bs", type=int, default=512)
ap.add_argument("--lr", type=float, default=5e-5)
ap.add_argument("--scale", type=float, default=20.0)
ap.add_argument("--ckpt_min", type=float, default=20.0)
ap.add_argument("--eval_only", action="store_true")
args = ap.parse_args()
SEED = 42
torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
dev = torch.device("cuda")
BF16 = torch.cuda.get_device_capability()[0] >= 8
os.makedirs(args.out, exist_ok=True)
T0 = time.time()
log = lambda *a: print(f"[{(time.time() - T0) / 60:6.1f} min]", *a, flush=True)
log("device", torch.cuda.get_device_name(0), "bf16" if BF16 else "fp32")

pairs = pl.read_parquet(f"{args.data}/enc_pairs.parquet")
s1t = pl.read_parquet(f"{args.data}/s1_train.parquet", columns=["id", "country", "text_full"])
vq = pl.read_parquet(f"{args.data}/v_q.parquet", columns=["id", "country", "src", "s1", "text_full"])

# ------------------------------------------------------------------ model
state_f = f"{args.out}/state.json"
state = json.load(open(state_f)) if os.path.exists(state_f) else {"phase": 1, "step": 0, "done": []}
src_model = f"{args.out}/latest" if os.path.exists(f"{args.out}/latest") else args.model
log("loading", src_model, "state", state)
model = SentenceTransformer(src_model, device="cuda")
model.max_seq_length = 64
tok = model.tokenizer


def enc_batch(texts):
    f = tok(texts, padding=True, truncation=True, max_length=64, return_tensors="pt")
    f = {k: v.to(dev, non_blocking=True) for k, v in f.items()}
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=BF16):
        e = model(f)["sentence_embedding"]
    return F.normalize(e.float(), dim=-1)


@torch.no_grad()
def encode(texts, bs=1024):
    model.eval()
    out = torch.empty((len(texts), model.get_sentence_embedding_dimension()), dtype=torch.float16, device=dev)
    order = np.argsort([len(t) for t in texts])           # length-sorted for less padding
    for i in range(0, len(texts), bs):
        idx = order[i:i + bs]
        out[torch.from_numpy(idx).to(dev)] = enc_batch([texts[j] for j in idx]).half()
    model.train()
    return out


@torch.no_grad()
def search(Q, qc, P, pc, k):
    """top-k rows of P for each row of Q, restricted to the same country. returns np int64 [nq,k] (global P rows)."""
    res = np.full((Q.shape[0], k), -1, np.int64)
    for c in np.unique(qc):
        qi = np.where(qc == c)[0]; pi = np.where(pc == c)[0]
        Pc = P[torch.from_numpy(pi).to(dev)]
        for lo in range(0, len(qi), 1024):
            s = Q[torch.from_numpy(qi[lo:lo + 1024]).to(dev)] @ Pc.T
            res[qi[lo:lo + 1024]] = pi[torch.topk(s.float(), min(k, len(pi)), dim=1).indices.cpu().numpy()]
    return res


EV = s1t


def evaluate(tag):
    t = time.time()
    s1t = EV
    P = encode(s1t["text_full"].to_list())
    Q = encode(vq["text_full"].to_list())
    top = search(Q, vq["country"].to_numpy(), P, s1t["country"].to_numpy(), 20)
    row = dict(zip(s1t["id"].to_list(), range(s1t.height)))
    true = np.array([row[s] for s in vq["s1"].to_list()])
    hit = top == true[:, None]
    rank = np.where(hit.any(1), hit.argmax(1) + 1, 0)
    res = {}
    for key, m in [("ALL", np.ones(len(true), bool))] + [
            (f"{c} S{s}", (vq["country"].to_numpy() == c) & (vq["src"].to_numpy() == s))
            for c in sorted(vq["country"].unique().to_list()) for s in (2, 3)]:
        res[key] = {f"R@{k}": round(float(((rank[m] > 0) & (rank[m] <= k)).mean()) * 100, 2) for k in (1, 5, 10, 20)}
    log(f"EVAL {tag} ({time.time() - t:.0f}s):", json.dumps(res))
    json.dump(res, open(f"{args.out}/eval_{tag}.json", "w"), indent=1)
    return res


if args.eval_only:
    evaluate("only"); sys.exit()


# ------------------------------------------------------------------ batching
def make_batches(df, bs, seed):
    """Batches of `bs` rows from one country, same state where possible; spreads repeats of an S1 apart."""
    rng = np.random.default_rng(seed)
    df = df.with_columns(pl.Series("r", rng.random(df.height))).with_columns(
        pl.col("r").rank("ordinal").over("s1").alias("occ"))
    batches, rest = [], {}
    for (g, c), part in df.group_by(["grp", "country"], maintain_order=True):
        idx = part.sort(["occ", "r"])["i"].to_numpy()
        n_full = len(idx) // bs
        batches += [idx[k * bs:(k + 1) * bs] for k in range(n_full)]
        rest.setdefault(c, []).append(idx[n_full * bs:])
    for c, parts in rest.items():                          # leftovers: same country, mixed states
        idx = np.concatenate(parts); rng.shuffle(idx)
        batches += [idx[k:k + bs] for k in range(0, len(idx) - bs + 1, bs)]
    order = rng.permutation(len(batches))
    return [batches[i] for i in order]


def query_texts(df, seed):
    """field dropout: 15% name only, 10% address only (only when the address is non-empty)."""
    u = np.random.default_rng(seed).random(df.height)
    has_addr = (df["q_addr"] != "query: ").to_numpy()
    full, name, addr = df["q_full"].to_list(), df["q_name"].to_list(), df["q_addr"].to_list()
    return [name[i] if u[i] < 0.15 else (addr[i] if u[i] < 0.25 and has_addr[i] else full[i]) for i in range(df.height)]


def save(phase, step, final=False):
    t = time.time()
    tmp = f"/tmp/ckpt_{phase}_{step}"
    model.save(tmp)
    dst = f"{args.out}/latest"
    if os.path.exists(dst + ".new"):
        shutil.rmtree(dst + ".new")
    shutil.copytree(tmp, dst + ".new")
    if os.path.exists(dst):
        shutil.rmtree(dst)
    os.rename(dst + ".new", dst)
    if final:
        fd = f"{args.out}/phase{phase}"
        if os.path.exists(fd):
            shutil.rmtree(fd)
        shutil.copytree(tmp, fd)
    shutil.rmtree(tmp)
    state.update({"phase": phase, "step": step})
    json.dump(state, open(state_f, "w"))
    log(f"checkpoint phase {phase} step {step}{' (final)' if final else ''} saved in {time.time() - t:.0f}s")


def train(phase, q, p, s1ids, hn=None, hn_s1=None):
    batches = make_batches(phase_df, args.bs, SEED + phase)
    total = len(batches)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    warm = max(1, int(0.05 * total))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min((s + 1) / warm, max(0.0, (total - s) / (total - warm))))
    start = state["step"] if state["phase"] == phase else 0
    for _ in range(start):
        sched.step()
    log(f"phase {phase}: {total} batches of {args.bs}, starting at {start}")
    model.train()
    last_ck, t, seen = time.time(), time.time(), 0
    for step in range(start, total):
        b = batches[step]
        qe = enc_batch([q[i] for i in b])
        pe = enc_batch([p[i] for i in b])
        sid = torch.tensor(s1ids[b], device=dev)
        cols, col_s1 = pe, sid
        if hn is not None:
            cols = torch.cat([pe, enc_batch([hn[i] for i in b])])
            col_s1 = torch.cat([sid, torch.tensor(hn_s1[b], device=dev)])
        logits = qe @ cols.T * args.scale
        same = sid[:, None] == col_s1[None, :]            # another copy of the query's own Source 1 is not a negative
        same[torch.arange(len(b)), torch.arange(len(b))] = False
        logits = logits.masked_fill(same, -1e4)
        loss = F.cross_entropy(logits, torch.arange(len(b), device=dev))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        seen += len(b)
        if step % 100 == 0 or step == total - 1:
            log(f"p{phase} step {step + 1}/{total} loss {loss.item():.4f} lr {sched.get_last_lr()[0]:.2e} "
                f"{seen / (time.time() - t):,.0f} pairs/s")
        if time.time() - last_ck > args.ckpt_min * 60:
            save(phase, step + 1); last_ck = time.time()
    save(phase, total, final=True)


s1_row_ids = pairs["s1_row"].to_numpy().astype(np.int64)
pairs = pairs.with_row_index("i")

# ------------------------------------------------------------------ phase 1
if 1 not in state["done"]:
    evaluate("phase0") if state["step"] == 0 else None
    rng = np.random.default_rng(SEED)
    sel = np.sort(rng.choice(pairs.height, min(args.p1_pairs, pairs.height), replace=False))
    phase_df = pairs[sel].with_columns(pl.int_range(len(sel)).alias("i"))
    q = query_texts(phase_df, SEED)
    train(1, q, phase_df["p"].to_list(), phase_df["s1_row"].to_numpy().astype(np.int64))
    state["done"].append(1); state["step"] = 0; json.dump(state, open(state_f, "w"))
    evaluate("phase1")

# ------------------------------------------------------------------ mining + phase 2
if 2 not in state["done"]:
    hn_f = f"{args.out}/hard_negatives.parquet"
    if os.path.exists(hn_f):
        phase_df = pl.read_parquet(hn_f)
    else:
        t = time.time()
        rng = np.random.default_rng(SEED + 7)
        sel = np.sort(rng.choice(pairs.height, min(args.hn_queries, pairs.height), replace=False))
        phase_df = pairs[sel].with_columns(pl.int_range(len(sel)).alias("i"))
        enc_rows = np.unique(s1_row_ids)                    # index = retriever-train Source 1 only (no HOLD/MT leakage)
        P = encode([s1t["text_full"][int(r)] for r in enc_rows])
        Q = encode(phase_df["q_full"].to_list())
        top = search(Q, phase_df["country"].to_numpy(), P, s1t["country"].to_numpy()[enc_rows], 10)
        top = np.where(top >= 0, enc_rows[np.maximum(top, 0)], -1)
        true = phase_df["s1_row"].to_numpy()
        hn = np.full(len(true), -1, np.int64); hn_rank = np.zeros(len(true), np.int64)
        for r in range(len(true)):
            for k in range(10):
                if top[r, k] >= 0 and top[r, k] != true[r]:
                    hn[r] = top[r, k]; hn_rank[r] = k + 1; break
        ok = hn >= 0
        phase_df = phase_df.filter(pl.Series(ok)).with_columns(
            pl.Series("hn_row", hn[ok]), pl.Series("hn_rank", hn_rank[ok]),
            pl.Series("hn", [s1t["text_full"][int(x)] for x in hn[ok]])).with_columns(pl.int_range(int(ok.sum())).alias("i"))
        phase_df.write_parquet(hn_f)
        log(f"mined {phase_df.height:,} hard negatives in {time.time() - t:.0f}s; true S1 at rank 1 for "
            f"{(top[:, 0] == true).mean():.1%}; hard-negative rank distribution "
            f"{dict(zip(*np.unique(hn_rank[ok], return_counts=True)))}")
        log("examples:", phase_df.select("q_full", "p", "hn").head(5).rows())
    q = query_texts(phase_df, SEED + 2)
    train(2, q, phase_df["p"].to_list(), phase_df["s1_row"].to_numpy().astype(np.int64),
          phase_df["hn"].to_list(), phase_df["hn_row"].to_numpy().astype(np.int64))
    state["done"].append(2); state["step"] = 0; json.dump(state, open(state_f, "w"))
    evaluate("phase2")
log("ALL DONE")
