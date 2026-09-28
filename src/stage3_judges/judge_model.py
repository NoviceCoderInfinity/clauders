"""Qwen3.5-4B judge: shared tokenisation, batching, forward pass and model loading.

Architecture: the text tower of Qwen/Qwen3.5-4B (Apache-2.0) with LoRA adapters on every attention,
linear-attention (Gated-DeltaNet) and MLP projection, and an fp32 linear head on the hidden state of the last real
token. Five adapters are rank 16 with alpha 32. judge_context_3, the rank-32 judge, is alpha 64, which is the
alpha stored in its adapter. Dropout is 0.05. Input: "A: <Source 1 text>" and "\nB: <record text>" as a tokenizer pair, longest-first truncation to MAXLEN tokens,
right padding to a multiple of 16 tokens (pads come after the last real token, so causal layers never see them), bf16 autocast.
A text is "business_name | business_address" (the address part is omitted when empty).

Scoring loads the base model by hub id at a pinned revision, removes the vision tower, keeps only the embedding rows of the
token ids that occur in the competition data (common/vocab_prune.py), merges the adapter and loads the fp32 head from
score_head.safetensors; token ids are remapped after tokenising and padding. Scores equal those of the full model.
"""
import itertools
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import polars as pl
import torch

import config
from common.vocab_prune import VocabPrune, drop_vision
from stage3_judges.texts import text  # noqa: F401  (re-exported)

BASE = os.environ.get("QWEN_BASE", "Qwen/Qwen3.5-4B")
BASE_REVISION = os.environ.get("QWEN_REVISION")          # pinned revision, see README.md
PADMULT = 16
TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "out_proj",
           "gate_proj", "up_proj", "down_proj"]
dev = "cuda"

# name -> (input kind, MAXLEN, LoRA rank, LoRA alpha). plain inputs = Source 1 text vs record text; context inputs add
# the record's other candidates (build_inputs.py). judge_context_3 is the only rank-32 adapter, and its saved adapter
# uses alpha 64. Every other judge is rank 16 with alpha 32.
JUDGES = {
    "judge_plain_1": ("plain", 128, 16, 32),
    "judge_plain_2": ("plain", 128, 16, 32),
    "judge_context_1": ("context", 192, 16, 32),
    "judge_context_2": ("context", 192, 16, 32),
    "judge_context_3": ("context", 192, 32, 64),
    "judge_context_4": ("context", 256, 16, 32),
}


def load_tokenizer():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(BASE, revision=BASE_REVISION)
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    return tok, pad


class Batcher:
    def __init__(self, tok, pad, maxlen):
        self.tok, self.pad, self.maxlen = tok, pad, maxlen
        self.pool = ThreadPoolExecutor(8)

    def tokenize(self, df):
        A = ("A: " + df["text_a"].fill_null("")).to_list(); B = ("\nB: " + df["text_b"].fill_null("")).to_list(); C = 20000
        encs = list(self.pool.map(lambda i: self.tok(A[i:i + C], B[i:i + C], truncation="longest_first", max_length=self.maxlen)["input_ids"],
                                  range(0, len(A), C)))
        return pack([x for e in encs for x in e])

    def batch(self, tk, j):
        return make_batch(tk, j, self.pad)


def pack(ids):
    L = np.fromiter((len(x) for x in ids), dtype=np.int64, count=len(ids))
    flat = np.fromiter(itertools.chain.from_iterable(ids), dtype=np.int32, count=int(L.sum()))
    off = np.zeros(len(L), dtype=np.int64); off[1:] = np.cumsum(L)[:-1]
    return flat, off, L


def make_batch(tk, j, pad):
    flat, off, L = tk; m = int(L[j].max()); m = -(-m // PADMULT) * PADMULT
    arr = np.full((len(j), m), pad, dtype=np.int64); msk = np.zeros((len(j), m), dtype=np.int64)
    for r, k in enumerate(j):
        arr[r, :L[k]] = flat[off[k]:off[k] + L[k]]; msk[r, :L[k]] = 1
    return torch.from_numpy(arr), torch.from_numpy(msk), torch.from_numpy(L[j] - 1)


def text_backbone(m):
    return m.model.language_model if hasattr(m.model, "language_model") else m.model


def forward(backbone, head, ids, msk, last):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        h = backbone(input_ids=ids, attention_mask=msk, use_cache=False).last_hidden_state
    return head(h[torch.arange(h.shape[0], device=ids.device), last].float()).squeeze(-1)


@torch.no_grad()
def score(backbone, head, tk, batch_fn, remap=None, bs=1024):
    """Scores packed token ids in length-sorted batches. remap: VocabPrune.remap for a pruned model."""
    order = np.argsort(tk[2], kind="stable"); out = np.zeros(len(order), dtype=np.float32)
    for i in range(0, len(order), bs):
        j = order[i:i + bs]; ids, msk, last = batch_fn(tk, j)
        if remap is not None:
            ids = remap(ids)
        out[j] = forward(backbone, head, ids.to(dev), msk.to(dev), last.to(dev)).float().cpu().numpy()
    return out


def load_base():
    from transformers import AutoModelForSequenceClassification
    tok, pad = load_tokenizer()
    base = AutoModelForSequenceClassification.from_pretrained(BASE, revision=BASE_REVISION, num_labels=1, dtype=torch.bfloat16,
                                                              attn_implementation="sdpa")
    base.config.pad_token_id = pad; base.config.use_cache = False
    return base, tok, pad


def load_scoring_model(adapter_dir):
    """Base + merged LoRA adapter + fp32 head, vision tower removed, embeddings pruned. Returns (backbone, head, remap, tok, pad)."""
    from peft import PeftModel
    from safetensors.torch import load_file
    base, tok, pad = load_base()
    m = PeftModel.from_pretrained(base, str(adapter_dir)).merge_and_unload()
    hw = load_file(f"{adapter_dir}/score_head.safetensors")["weight"]
    m.score = torch.nn.Linear(hw.shape[1], 1, bias=False); m.score.weight.data = hw.float()
    m = m.to(dev).eval(); drop_vision(m)
    vp = VocabPrune(config.KEPT_QWEN); bb = text_backbone(m)
    bb.embed_tokens = vp.prune(bb.embed_tokens)
    return bb, m.score, vp.remap, tok, pad
