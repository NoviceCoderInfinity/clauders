"""e5-small cross-encoder: train on GPU 0, then score all candidate pairs on every visible GPU.

Model: intfloat/multilingual-e5-small (MIT) + 1-logit head, input = (record text, Source 1 text), max 128 tokens.
Train: ENC hard-negative triples (positive + hard negative) + lookalike negatives; 1 epoch, BCE, bf16.
Scoring loads the model with the pruned vocabulary (common/encoders.py); scores are identical to the full model's.
usage: python train_cross_encoder.py --data WORK_DIR/data/cross_encoder --hn <retriever out>/hard_negatives.parquet --out <dir>
       [--skip_train --model <dir>]   (score with an existing model)
Out: <out>/model, <out>/scores/<set>.<shard>-of-<n>.npy (float16 logits aligned with pairs_<set>.parquet)
"""
import os, sys, time, json, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, polars as pl, torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from common.encoders import ensure_pruned, remap_batch

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--hn", default=None)
ap.add_argument("--out", required=True)
ap.add_argument("--model", default=None, help="existing model dir to score with (default: <out>/model)")
ap.add_argument("--bs", type=int, default=256)
ap.add_argument("--lr", type=float, default=3e-5)
ap.add_argument("--skip_train", action="store_true")
args = ap.parse_args()
BASE = "intfloat/multilingual-e5-small"
SETS = [("pool_q", "s1_train"), ("test_q", "s1_test")]
if os.environ.get("CE_SETS"):   # e.g. "test_q:s1_test"
    SETS = [tuple(x.split(":")) for x in os.environ["CE_SETS"].split(",")]
MODEL = args.model or f"{args.out}/model"
T0 = time.time()
log = lambda *a: print(f"[{(time.time() - T0) / 60:6.1f} min]", *a, flush=True)
strip = lambda s: s.str.replace(r"^(query|passage): ", "")
os.makedirs(f"{args.out}/scores", exist_ok=True)


class Pairs(Dataset):
    def __init__(self, a, b, y=None):
        self.a, self.b, self.y = a, b, y

    def __len__(self):
        return len(self.a)

    def __getitem__(self, i):
        return i


def make_collate(tok, a, b, y):
    def collate(idx):
        f = tok([a[i] for i in idx], [b[i] for i in idx], padding=True, truncation=True, max_length=128, return_tensors="pt")
        if y is not None:
            f["labels"] = torch.tensor([y[i] for i in idx], dtype=torch.float32)
        return f
    return collate


def train():
    dev = torch.device("cuda:0")
    hn = pl.read_parquet(args.hn, columns=["q_full", "p", "hn"])
    look = pl.read_parquet(f"{args.data}/train_lookalike_negatives.parquet")
    q = strip(hn["q_full"]).to_list()
    a = q + q + strip(look["q"]).to_list()
    b = strip(hn["p"]).to_list() + strip(hn["hn"]).to_list() + strip(look["p"]).to_list()
    y = [1.0] * hn.height + [0.0] * hn.height + [0.0] * look.height
    rng = np.random.default_rng(42)
    perm = rng.permutation(len(a))
    n_ho = 20000
    ho, tr = perm[:n_ho], perm[n_ho:]
    tok = AutoTokenizer.from_pretrained(BASE)
    model = AutoModelForSequenceClassification.from_pretrained(BASE, num_labels=1).to(dev)
    dl = DataLoader(Pairs(tr, tr), batch_size=args.bs, shuffle=False, num_workers=8, multiprocessing_context="fork", collate_fn=make_collate(
        tok, [a[i] for i in tr], [b[i] for i in tr], [y[i] for i in tr]), sampler=None)
    total = len(dl)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    warm = max(1, int(0.05 * total))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min((s + 1) / warm, max(0.0, (total - s) / (total - warm))))
    log(f"train: {len(tr):,} pairs ({sum(y[i] for i in tr):,.0f} positives), {total} steps")
    model.train(); t = time.time(); seen = 0
    for step, f in enumerate(dl):
        f = {k: v.to(dev, non_blocking=True) for k, v in f.items()}
        lab = f.pop("labels")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logit = model(**f).logits.squeeze(-1)
        loss = F.binary_cross_entropy_with_logits(logit.float(), lab)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
        seen += len(lab)
        if step % 200 == 0 or step == total - 1:
            log(f"step {step + 1}/{total} loss {loss.item():.4f} {seen / (time.time() - t):,.0f} pairs/s")
    model.save_pretrained(f"{args.out}/model"); tok.save_pretrained(f"{args.out}/model")
    # held-out check on 20k training-distribution pairs
    from sklearn.metrics import roc_auc_score
    s = score(model, tok, None, dev, [a[i] for i in ho], [b[i] for i in ho])
    auc = roc_auc_score([y[i] for i in ho], s)
    log(f"held-out (20k training-distribution pairs) AUC {auc:.5f}")
    json.dump({"holdout_auc": auc, "train_pairs": len(tr)}, open(f"{args.out}/train_stats.json", "w"))


@torch.no_grad()
def score(model, tok, vp, dev, a, b, bs=1024):
    """vp: vocabulary pruner of the loaded model (None = full-vocabulary model, as right after training)."""
    model.eval()
    dl = DataLoader(Pairs(a, b), batch_size=bs, num_workers=6, multiprocessing_context="fork", collate_fn=make_collate(tok, a, b, None))
    out = np.empty(len(a), np.float16); k = 0
    for f in dl:
        if vp is not None:
            f = remap_batch(vp, f)
        f = {kk: v.to(dev, non_blocking=True) for kk, v in f.items()}
        with torch.autocast("cuda", dtype=torch.bfloat16):
            s = model(**f).logits.squeeze(-1).float().cpu().numpy()
        out[k:k + len(s)] = s; k += len(s)
    return out


def infer(rank, world):
    dev = torch.device(f"cuda:{rank}")
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL, dtype=torch.float32)
    vp = ensure_pruned(model, tok)
    model = model.to(dev)
    for qs, ds in SETS:
        fn = f"{args.out}/scores/{qs}.{rank}-of-{world}.npy"
        if os.path.exists(fn):
            continue
        P = pl.read_parquet(f"{args.data}/pairs_{qs}.parquet")
        lo, hi = rank * P.height // world, (rank + 1) * P.height // world
        P = P[lo:hi]
        qt = strip(pl.read_parquet(f"{args.data}/text_{qs}.parquet")["text_full"]).to_list()
        dt = strip(pl.read_parquet(f"{args.data}/text_{ds}.parquet")["text_full"]).to_list()
        t = time.time()
        s = score(model, tok, vp, dev, [qt[i] for i in P["qi"].to_list()], [dt[i] for i in P["di"].to_list()])
        np.save(fn + ".tmp.npy", s); os.replace(fn + ".tmp.npy", fn)
        log(f"gpu{rank}: scored {qs} [{lo:,}:{hi:,}] {len(s):,} pairs in {time.time() - t:.0f}s ({len(s) / (time.time() - t):,.0f}/s)")


if __name__ == "__main__":
    log("GPUs:", torch.cuda.device_count(), torch.cuda.get_device_name(0))
    if not args.skip_train and not os.path.exists(f"{MODEL}/config.json"):
        train()
    world = torch.cuda.device_count()
    torch.multiprocessing.spawn(infer, args=(world,), nprocs=world, join=True)
    log("scores:", sorted(os.listdir(f"{args.out}/scores")))
