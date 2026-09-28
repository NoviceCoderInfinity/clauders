"""Train one LoRA judge (GPU; 8 GPUs with torchrun; one epoch over the 1,064,434 training rows of plain or context).

usage: torchrun --standalone --nproc_per_node 8 -m stage3_judges.train_lora --judge judge_context_1 \
           --train WORK_DIR/stage3/train/context.parquet --out WORK_DIR/models/qwen_judges/judge_context_1 [--seed 0]
Training file: s1_id, cand_id, text_a, text_b, label, split (train | val). The two training sets (plain: Source 1 text vs
record text; context: the same pairs with the record's other candidates appended, build_inputs.context) are data tables
(WORK_DIR/stage3/train/, built by build_training_sets.py). Every row is a pair of TRAINING records with its label from train_ground_truth.tsv;
check_training_sets.py verifies this. Recipe: LoRA rank and alpha from the JUDGES table (rank 16 / alpha 32, except judge_context_3 at rank 32 / alpha 64), dropout 0.05 on every
attention / linear-attention / MLP projection of the text tower, trainable fp32 head (last real token), BCE, AdamW lr 2e-4,
global batch 256 split across GPUs (gradients averaged), 5% warmup + linear decay, one epoch, length-bucketed shuffle,
gradient clipping 1.0. Writes the adapter and score_head.safetensors.
"""
import argparse
import json
import math
import os
import time

import numpy as np
import polars as pl
import torch
import torch.distributed as dist

import config
from stage3_judges import judge_model as JM


def auc_ll(y, s):
    from sklearn.metrics import roc_auc_score, log_loss
    p = 1 / (1 + np.exp(-s.astype(np.float64)))
    return round(float(roc_auc_score(y, s)), 5), round(float(log_loss(y, np.clip(p, 1e-7, 1 - 1e-7))), 5)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge", required=True, choices=sorted(JM.JUDGES))
    ap.add_argument("--train", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--epochs", type=float, default=1.0)
    a = ap.parse_args()
    _, maxlen, rank, alpha = JM.JUDGES[a.judge]
    out = a.out or str(config.JUDGES_DIR / a.judge)
    torch.backends.cuda.matmul.allow_tf32 = True; torch.manual_seed(a.seed)
    WORLD = int(os.environ.get("WORLD_SIZE", 1)); RANK = int(os.environ.get("RANK", 0)); LRANK = int(os.environ.get("LOCAL_RANK", 0))
    if WORLD > 1:
        torch.cuda.set_device(LRANK); dist.init_process_group("nccl")
    log = lambda *x: print(time.strftime("%H:%M:%S"), *x, flush=True) if RANK == 0 else None
    from peft import LoraConfig, get_peft_model
    from safetensors.torch import save_file
    from transformers import get_linear_schedule_with_warmup
    base, tok, pad = JM.load_base()
    B = JM.Batcher(tok, pad, maxlen)
    df = pl.read_parquet(a.train); tr = df.filter(pl.col("split") == "train"); va = df.filter(pl.col("split") == "val")
    tk_tr, tk_va = B.tokenize(tr), B.tokenize(va); y_tr = tr["label"].to_numpy().astype(np.float32)
    log("train pairs", tr.height, "val", va.height)
    cfg = LoraConfig(task_type="SEQ_CLS", r=rank, lora_alpha=alpha, lora_dropout=0.05,
                     target_modules=r".*language_model.*\.(" + "|".join(JM.TARGETS) + r")$", modules_to_save=["score"])
    model = get_peft_model(base, cfg).to(JM.dev)
    for p in model.parameters():
        if p.requires_grad: p.data = p.data.float()
    inner = model.base_model.model; backbone, head = JM.text_backbone(inner), inner.score
    rng = np.random.default_rng(a.seed); batches = []
    for ep in range(int(math.ceil(a.epochs))):
        perm = rng.permutation(len(y_tr)); mb = a.batch * 64
        for i in range(0, len(perm), mb):
            c = perm[i:i + mb]; c = c[np.argsort(tk_tr[2][c], kind="stable")]
            batches += [c[k:k + a.batch] for k in range(0, len(c), a.batch)]
    rng.shuffle(batches); steps = int(len(batches) * a.epochs / math.ceil(a.epochs)); batches = batches[:steps]
    batches = [b[RANK::WORLD] for b in batches]
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0, fused=True)
    sch = get_linear_schedule_with_warmup(opt, int(0.05 * steps), steps); lossf = torch.nn.BCEWithLogitsLoss()
    ytr = torch.from_numpy(y_tr); model.train()
    for step in range(steps):
        ids, msk, last = B.batch(tk_tr, batches[step])
        loss = lossf(JM.forward(backbone, head, ids.to(JM.dev), msk.to(JM.dev), last.to(JM.dev)), ytr[batches[step]].to(JM.dev))
        loss.backward()
        if WORLD > 1:
            flat = torch.cat([p.grad.reshape(-1) for p in params]); dist.all_reduce(flat); flat /= WORLD; o = 0
            for p in params:
                n = p.numel(); p.grad.copy_(flat[o:o + n].view_as(p)); o += n
        torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); sch.step(); opt.zero_grad(set_to_none=True)
        if (step + 1) % 200 == 0:
            log(f"step {step + 1}/{steps} loss {loss.item():.4f}")
    if RANK != 0:
        raise SystemExit(0)
    model.eval()
    vm = auc_ll(va["label"].to_numpy(), JM.score(backbone, head, tk_va, B.batch))
    log("validation (AUC, logloss)", vm)
    os.makedirs(out, exist_ok=True); model.save_pretrained(out)
    save_file({"weight": head.modules_to_save["default"].weight.detach().float().cpu().contiguous()}, f"{out}/score_head.safetensors")
    json.dump({"judge": a.judge, "train": os.path.basename(a.train), "n_train": tr.height, "lr": a.lr, "batch": a.batch, "maxlen": maxlen,
               "r": rank, "alpha": alpha, "seed": a.seed, "val": vm}, open(f"{out}/judge_meta.json", "w"), indent=1)
