"""Vocabulary pruning for every transformer in the pipeline (outputs identical to the full-vocabulary models).

1. Qwen3.5-4B: the vision tower is removed (text input never reaches it); the multi-token-prediction head is never loaded.
2. Every model keeps only the token-embedding rows whose ids occur in the competition data, tokenised exactly as the pipeline
   does (kept_token_ids/qwen.npy for Qwen, kept_token_ids/xlmr.npy for all XLM-R / e5 models, which share one vocabulary).
   Token ids are remapped right after tokenisation. An embedding lookup of a kept id returns the same vector as before, so
   every score is unchanged. If an input ever produces an id outside the kept set, remap() raises instead of silently
   changing a score.

Usage (Qwen judge):
    vp = VocabPrune(config.KEPT_QWEN)
    model = ...load + merge LoRA...; drop_vision(model)
    backbone.embed_tokens = vp.prune(backbone.embed_tokens)
    input_ids = vp.remap(input_ids)            # after tokenising and padding; the pad id is a kept id
Usage (XLM-R cross-encoder / e5 encoder):
    vp = VocabPrune(config.KEPT_XLMR); vp.check_specials(tok)   # ids 0..3 keep their positions, so padding_idx is unchanged
    emb = model.roberta.embeddings (or model.embeddings for an e5 encoder); emb.word_embeddings = vp.prune(emb.word_embeddings)
    input_ids = vp.remap(input_ids)
"""
import numpy as np
import torch


class VocabPrune:
    def __init__(self, kept_path):
        self.kept = np.load(kept_path).astype(np.int64)
        assert np.all(np.diff(self.kept) > 0), "kept ids must be sorted and unique"
        self.lut = np.full(int(self.kept.max()) + 1, -1, dtype=np.int64)
        self.lut[self.kept] = np.arange(len(self.kept))
        self._lut_t = torch.from_numpy(self.lut)

    def remap(self, ids):
        """ids: torch.Tensor, np.ndarray or list of lists (ragged ok). Raises if any id is outside the kept set."""
        if isinstance(ids, torch.Tensor):
            lut = self._lut_t.to(ids.device)
            if int(ids.max()) >= len(lut): raise ValueError("token id outside the kept vocabulary")
            out = lut[ids]
        elif isinstance(ids, np.ndarray):
            if int(ids.max()) >= len(self.lut): raise ValueError("token id outside the kept vocabulary")
            out = self.lut[ids]
        else:
            return [self.remap(np.asarray(x, dtype=np.int64)).tolist() for x in ids]
        if (out < 0).any(): raise ValueError("token id outside the kept vocabulary: the output would differ; regenerate the kept ids")
        return out

    def prune(self, emb: torch.nn.Embedding) -> torch.nn.Embedding:
        w = emb.weight.data[torch.from_numpy(self.kept).to(emb.weight.device)].clone()
        pad = emb.padding_idx
        return torch.nn.Embedding.from_pretrained(w, freeze=True, padding_idx=None if pad is None else int(self.lut[pad]))

    def check_specials(self, tok):
        """XLM-R computes position ids from input_ids != pad_token_id: the pad id must map to itself."""
        for t in tok.all_special_ids:
            if t < len(self.lut) and self.lut[t] >= 0 and t <= 3: assert self.lut[t] == t, (t, self.lut[t])
        assert self.lut[tok.pad_token_id] == tok.pad_token_id, "pad id moved; set config.pad_token_id = lut[pad] or keep ids 0..3"


def drop_vision(model):
    """Remove Qwen3.5's vision tower from a loaded Qwen3_5For* model (text-only inputs never reach it)."""
    inner = getattr(model, "model", model)
    if hasattr(inner, "visual"):
        del inner.visual
    return model


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def save_pruned_checkpoint(src, out, kept_path, kind="seq_cls", dtype=None):
    """Write a full fine-tuned XLM-R / e5 checkpoint in pruned on-disk form: word-embedding rows = kept ids,
    config.vocab_size = number of kept ids, the original tokenizer files alongside (ids are remapped after tokenising).
    kind: "seq_cls" (cross-encoder / reranker), "encoder" (AutoModel), "sentence_transformer" (sentence-transformers dir).
    dtype: torch dtype to store (default: keep the checkpoint's dtype)."""
    vp = VocabPrune(kept_path)
    if kind == "sentence_transformer":
        from sentence_transformers import SentenceTransformer
        m = SentenceTransformer(src, device="cpu")
        am = m[0].auto_model
        vp.check_specials(m[0].tokenizer)
        am.embeddings.word_embeddings = vp.prune(am.embeddings.word_embeddings)
        am.config.vocab_size = len(vp.kept)
        if dtype is not None:
            m.to(dtype)
        m.save(out)
        return out
    from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
    tok = AutoTokenizer.from_pretrained(src)
    vp.check_specials(tok)
    cls = AutoModelForSequenceClassification if kind == "seq_cls" else AutoModel
    m = cls.from_pretrained(src) if dtype is None else cls.from_pretrained(src, dtype=dtype)
    base = getattr(m, m.base_model_prefix, m)
    base.embeddings.word_embeddings = vp.prune(base.embeddings.word_embeddings)
    m.config.vocab_size = len(vp.kept)
    m.save_pretrained(out)
    tok.save_pretrained(out)
    return out
