"""Inference-time use of the XLM-R / e5 encoders with the pruned vocabulary (outputs identical to the full models).

The fine-tuned encoders in WEIGHTS_DIR/encoders/ are stored already pruned (word-embedding rows = kept_token_ids/xlmr.npy,
config.vocab_size = number of kept ids, original tokenizer files alongside; see vocab_prune.save_pruned_checkpoint).
Every inference load in stages 1-2 goes through these helpers:
  - a checkpoint that still has the full vocabulary (e.g. straight out of training) is pruned in memory;
  - token ids are remapped right after tokenisation; an id outside the kept set raises instead of changing a score.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from common.vocab_prune import VocabPrune

_VX = None


def xlmr_pruner():
    global _VX
    if _VX is None:
        _VX = VocabPrune(config.KEPT_XLMR)
    return _VX


def _embeddings(model):
    """The token-embedding module of a BERT / XLM-R encoder or of its sequence-classification wrapper."""
    base = getattr(model, getattr(model, "base_model_prefix", ""), model)
    return base.embeddings


def ensure_pruned(model, tok):
    """Prune the word embeddings in place unless the checkpoint is already pruned; returns the pruner for remapping ids."""
    vp = xlmr_pruner()
    vp.check_specials(tok)
    emb = _embeddings(model)
    if emb.word_embeddings.num_embeddings != len(vp.kept):
        emb.word_embeddings = vp.prune(emb.word_embeddings)
        model.config.vocab_size = len(vp.kept)
    return vp


def remap_batch(vp, features):
    """features: tokenizer output (dict of tensors) -> same dict with input_ids renumbered to the pruned vocabulary."""
    features["input_ids"] = vp.remap(features["input_ids"])
    return features


def prune_sentence_transformer(st_model):
    """sentence-transformers model: pruned word embeddings and remapped input_ids.
    encode() tokenises through preprocess() in sentence-transformers >= 5.7 (tokenize() calls preprocess() there) and through
    tokenize() before; exactly one of them is wrapped, so ids are remapped once."""
    first = st_model[0]
    vp = ensure_pruned(first.auto_model, first.tokenizer)
    name = "preprocess" if hasattr(st_model, "preprocess") else "tokenize"
    fn = getattr(st_model, name)

    def pruned(texts, *args, **kwargs):
        return remap_batch(vp, fn(texts, *args, **kwargs))
    setattr(st_model, name, pruned)
    return st_model
