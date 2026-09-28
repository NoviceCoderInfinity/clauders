"""Listwise cross-encoder reranker (bge-reranker-v2-m3 or multilingual-e5-base backbone).

train: python listwise_reranker.py train --backbone <hf id> --lists <lists.parquet> --out <dir> [--max_q N] [--k 8] [--bs_q 32] [--lr 3e-5] [--field ...]
   lists.parquet: q (record text), cands (list[str] Source 1 texts, candidate order), tgt (index of the true Source 1 in cands, -1 = none)
   loss = listwise softmax over [candidate logits..., null logit 0] (target = true Source 1 or null) + 0.5 * pointwise BCE per pair
score: python listwise_reranker.py score --model <dir> --pairs <pairs.parquet qi,di> --qtext <parquet text_full> --dtext <parquet text_full> --out <file.npy>
   multi-GPU (all visible), length-sorted batches, fp16 logits aligned with pairs rows; the model is loaded with the pruned
   vocabulary (common/encoders.py), so the scores are identical to the full model's.
--field: 'full' (name | address), 'name' (text before " | "), 'addr' (after), 'core' (name without legal-form / filler words + address)
Seeds fixed (42). bf16 autocast.
"""
import os, sys, time, json, argparse, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, polars as pl, torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from common.encoders import ensure_pruned, remap_batch

T0 = time.time()
log = lambda *a: print(f"[{(time.time() - T0) / 60:6.1f} min]", *a, flush=True)
strip = lambda s: s.str.replace(r"^(query|passage): ", "")
MAXLEN = 128


def load_model(path, dev):
    tok = AutoTokenizer.from_pretrained(path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    m = AutoModelForSequenceClassification.from_pretrained(path, num_labels=1, dtype=torch.float32)
    if m.config.pad_token_id is None:
        m.config.pad_token_id = tok.pad_token_id
    return tok, m.to(dev)


FIELD = "full"
CORE_DROP = set("""inc incorporated llc l.l.c ltd limited pvt private corp corporation co company pc p.c pllc lp llp plc
sas sarl sa sci sasu eurl s.a.s s.a s.a.r.l ets services service center centre partners group groupe holdings holding
the and & of france india usa international enterprises""".split())   # = features.py CORE_DROP
def view(x):
    if FIELD == "full": return x
    n, _, a = x.partition(" | ")
    if FIELD == "core":
        core = " ".join(w for w in n.split() if w.strip(".,()-") not in CORE_DROP) or n
        return core + " | " + a
    return n if FIELD == "name" else (a or "<no address>")


def enc(tok, a, b):
    a = [view(x) for x in a]; b = [view(y) for y in b]
    return tok(a, b, padding=True, truncation=True, max_length=MAXLEN, return_tensors="pt")


def train(args):
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    world = int(os.environ.get("WORLD_SIZE", "1")); rank = int(os.environ.get("RANK", "0"))
    if world > 1:
        import torch.distributed as dist
        dist.init_process_group("nccl"); torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dev = torch.device(f"cuda:{int(os.environ.get('LOCAL_RANK', '0'))}")
    L = pl.read_parquet(args.lists)
    if args.max_q and L.height > args.max_q:
        L = L.sample(args.max_q, seed=42)
    L = L.sample(fraction=1.0, shuffle=True, seed=args.seed)
    if world > 1:   # each rank trains on its own shard of lists (same number of steps per rank)
        n = (L.height // world) * world
        L = L[:n].gather_every(world, offset=rank)
    q, C, tg = L["q"].to_list(), L["cands"].to_list(), L["tgt"].to_list()
    tok, model = load_model(args.backbone, dev)
    core = model
    if world > 1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[dev.index])
    if args.grad_ckpt:
        core.gradient_checkpointing_enable()
    nb = math.ceil(len(q) / args.bs_q) * args.epochs
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    warm = max(1, int(0.05 * nb))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min((s + 1) / warm, max(0.0, (nb - s) / max(1, nb - warm))))
    log(f"train {args.backbone}: {len(q):,} lists, {sum(len(c) for c in C):,} pairs, {nb} steps, positives-in-list {np.mean(np.array(tg) >= 0):.3f}")
    model.train(); step = 0; t = time.time(); npair = 0
    for ep in range(args.epochs):
        for lo in range(0, len(q), args.bs_q):
            qa, qb, seg, tgt = [], [], [], []
            for i in range(lo, min(lo + args.bs_q, len(q))):
                cs = C[i][:args.k]
                qa += [q[i]] * len(cs); qb += cs; seg.append(len(cs))
                tgt.append(tg[i] if tg[i] < len(cs) else -1)
            f = {k: v.to(dev, non_blocking=True) for k, v in enc(tok, qa, qb).items()}
            with torch.autocast("cuda", dtype=torch.bfloat16):
                s = model(**f).logits.squeeze(-1).float()
            # listwise with null (vectorised): pad each list to kmax with -inf, append null logit 0
            B, kmax = len(seg), max(seg)
            owner = torch.repeat_interleave(torch.arange(B, device=dev), torch.tensor(seg, device=dev))
            pos = torch.cat([torch.arange(n, device=dev) for n in seg])
            M = s.new_full((B, kmax + 1), float("-inf")); M[owner, pos] = s; M[:, kmax] = 0.0
            yt = torch.tensor([y if y >= 0 else kmax for y in tgt], device=dev)
            lab = torch.zeros_like(s); has = torch.tensor([y >= 0 for y in tgt], device=dev)
            first = torch.tensor(np.cumsum([0] + seg[:-1]), device=dev)
            lab[(first + yt.clamp(max=kmax - 1))[has & (yt < kmax)]] = 1.0
            loss = F.cross_entropy(M, yt) + 0.5 * F.binary_cross_entropy_with_logits(s, lab)
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
            step += 1; npair += len(s)
            if rank == 0 and (step % 200 == 0 or step == nb):
                log(f"step {step}/{nb} loss {loss.item():.4f}  {npair / (time.time() - t):,.0f} pairs/s")
    if rank == 0:
        os.makedirs(args.out, exist_ok=True)
        core.save_pretrained(f"{args.out}/model"); tok.save_pretrained(f"{args.out}/model")
        json.dump(vars(args) | {"lists_per_rank": L.height, "world": world, "minutes": (time.time() - T0) / 60}, open(f"{args.out}/train_args.json", "w"), indent=1)
        log("saved", args.out)
    if world > 1:
        import torch.distributed as dist
        dist.barrier(); dist.destroy_process_group()


@torch.no_grad()
def _score_worker(rank, world, args, ret):
    global FIELD
    FIELD = args.field
    dev = torch.device(f"cuda:{rank}")
    tok, model = load_model(args.model, dev); model.eval()
    vp = ensure_pruned(model, tok)
    P = pl.read_parquet(args.pairs, columns=["qi", "di"])
    if args.nparts > 1:   # job-level shard (several jobs score disjoint slices; concatenate in part order)
        P = P[args.part * P.height // args.nparts:(args.part + 1) * P.height // args.nparts]
    qt = strip(pl.read_parquet(args.qtext, columns=["text_full"])["text_full"])
    dt = strip(pl.read_parquet(args.dtext, columns=["text_full"])["text_full"])
    lo, hi = rank * P.height // world, (rank + 1) * P.height // world
    P = P[lo:hi]
    a = qt.gather(P["qi"]).to_list(); b = dt.gather(P["di"]).to_list()
    order = np.argsort([len(x) + len(y) for x, y in zip(a, b)])
    out = np.empty(len(a), np.float16); t = time.time()
    for j in range(0, len(order), args.bs):
        ix = order[j:j + args.bs]
        f = {k: v.to(dev) for k, v in remap_batch(vp, enc(tok, [a[i] for i in ix], [b[i] for i in ix])).items()}
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out[ix] = model(**f).logits.squeeze(-1).float().cpu().numpy()
        if rank == 0 and (j // args.bs) % 500 == 0:
            log(f"gpu0 {j:,}/{len(order):,}  {j / (time.time() - t + 1e-9):,.0f} pairs/s")
    np.save(f"{args.out}.part{rank}.npy", out)


def score(args):
    world = torch.cuda.device_count()
    torch.multiprocessing.spawn(_score_worker, args=(world, args, None), nprocs=world, join=True)
    s = np.concatenate([np.load(f"{args.out}.part{r}.npy") for r in range(world)])
    np.save(args.out, s)
    for r in range(world): os.remove(f"{args.out}.part{r}.npy")
    log(f"scored {len(s):,} pairs on {world} GPUs -> {args.out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["train", "score"])
    ap.add_argument("--backbone"); ap.add_argument("--lists"); ap.add_argument("--out")
    ap.add_argument("--max_q", type=int, default=0); ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--bs_q", type=int, default=32); ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--epochs", type=int, default=1); ap.add_argument("--grad_ckpt", action="store_true")
    ap.add_argument("--model"); ap.add_argument("--pairs"); ap.add_argument("--qtext"); ap.add_argument("--dtext")
    ap.add_argument("--bs", type=int, default=1024)
    ap.add_argument("--part", type=int, default=0); ap.add_argument("--nparts", type=int, default=1)
    ap.add_argument("--field", default="full", choices=["full", "name", "addr", "core"]); ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    FIELD = a.field
    log("GPUs", torch.cuda.device_count(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "-")
    train(a) if a.mode == "train" else score(a)
