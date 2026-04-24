"""
============================================================================
 src/utils/contextual_embeddings.py
 ---------------------------------------------------------------------------
 Contextual embedding loader that REPLACES the legacy static GloVe pipeline.

 Produces two artefacts used by the neural topic model:

   (1) WORD-LEVEL embeddings [|V|, d]
       - Each vocabulary token is encoded in isolation by the sentence
         transformer. Multi-word (phrase) tokens such as "new_york" are
         encoded directly (underscores replaced with spaces).
       - Used by SCADecoder and by the coherence / inter-cosine metrics.

   (2) DOCUMENT-LEVEL embeddings [N_docs, d]
       - Each preprocessed document (joined tokens) is encoded once.
       - Saved to disk alongside the BoW matrix so the training dataloader
         can concatenate them with the BoW vector before the encoder.

 Default backbone: 'sentence-transformers/all-MiniLM-L6-v2'  (d = 384).
 A Twitter-specific backbone 'cardiffnlp/twitter-roberta-base' can be
 passed via the config field `dataset.contextual_model`.

 The loader is deliberately dependency-light:
   * If `sentence_transformers` is available  → fast batched encoding.
   * Otherwise                                → graceful fallback using
     Hugging Face `transformers` with mean-pooling.
   * If neither is available                  → deterministic PRNG stub
     with a loud warning so the training loop can still run for smoke
     tests; DO NOT ship journal numbers from the stub path.
============================================================================
"""
from __future__ import annotations

import os
import hashlib
from typing import Iterable, List, Optional

import numpy as np
import torch

from .logging_utils import get_logger

_LOG = get_logger(__name__)

# MiniLM default; can be overridden by cfg.dataset.contextual_model
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


# =============================================================================
# Backbone loader (cached singleton)
# =============================================================================
_CACHED_MODEL = {"name": None, "obj": None, "dim": None, "backend": None}


def _load_backbone(model_name: str, device: torch.device):
    """Load (or return cached) sentence transformer."""
    if _CACHED_MODEL["name"] == model_name and _CACHED_MODEL["obj"] is not None:
        return _CACHED_MODEL["obj"], _CACHED_MODEL["dim"], _CACHED_MODEL["backend"]

    # ---- Preferred: sentence-transformers ----
    try:
        from sentence_transformers import SentenceTransformer
        _LOG.info(f"Loading sentence-transformer backbone: {model_name}")
        obj = SentenceTransformer(model_name, device=str(device))
        dim = obj.get_sentence_embedding_dimension()
        _CACHED_MODEL.update(name=model_name, obj=obj, dim=dim, backend="st")
        _LOG.info(f"  dim = {dim}")
        return obj, dim, "st"
    except Exception as e:
        _LOG.warning(f"sentence_transformers unavailable ({e}); "
                     f"falling back to raw transformers.")

    # ---- Fallback: transformers + mean pooling ----
    try:
        from transformers import AutoTokenizer, AutoModel
        tok = AutoTokenizer.from_pretrained(model_name)
        mod = AutoModel.from_pretrained(model_name).to(device).eval()
        dim = mod.config.hidden_size
        _CACHED_MODEL.update(name=model_name, obj=(tok, mod), dim=dim,
                             backend="hf")
        _LOG.info(f"  (HF backend) dim = {dim}")
        return (tok, mod), dim, "hf"
    except Exception as e:
        _LOG.error(f"transformers unavailable ({e}); using deterministic "
                   f"PRNG stub — DO NOT report these numbers.")

    # ---- Last-resort stub (deterministic) ----
    dim = 384
    _CACHED_MODEL.update(name=model_name, obj=None, dim=dim, backend="stub")
    return None, dim, "stub"


# =============================================================================
# Encoding helpers
# =============================================================================
def _encode_st(model, texts: List[str], batch_size: int = 64) -> np.ndarray:
    """Encode with sentence-transformers."""
    arr = model.encode(texts,
                       batch_size=batch_size,
                       convert_to_numpy=True,
                       normalize_embeddings=False,
                       show_progress_bar=False)
    return np.asarray(arr, dtype=np.float32)


@torch.no_grad()
def _encode_hf(bundle, texts: List[str], device: torch.device,
               batch_size: int = 32) -> np.ndarray:
    """Encode with raw transformers + mean pooling."""
    tok, mod = bundle
    out = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i:i + batch_size]
        enc = tok(chunk, padding=True, truncation=True,
                  max_length=128, return_tensors="pt").to(device)
        h = mod(**enc).last_hidden_state                    # [B, L, d]
        mask = enc.attention_mask.unsqueeze(-1).float()
        pooled = (h * mask).sum(1) / mask.sum(1).clamp(min=1e-6)
        out.append(pooled.cpu().numpy().astype(np.float32))
    return np.concatenate(out, axis=0)


def _encode_stub(texts: List[str], dim: int) -> np.ndarray:
    """Deterministic hash-based stub (reproducible but uninformative)."""
    out = np.zeros((len(texts), dim), dtype=np.float32)
    for i, t in enumerate(texts):
        h = hashlib.blake2b(t.encode("utf-8"), digest_size=16).digest()
        seed = int.from_bytes(h, "little") & 0xFFFFFFFF
        rng = np.random.RandomState(seed)
        out[i] = rng.normal(0.0, 0.01, dim).astype(np.float32)
    return out


def encode_texts(texts: List[str],
                 model_name: str = DEFAULT_MODEL,
                 device: Optional[torch.device] = None,
                 batch_size: int = 64) -> torch.Tensor:
    """
    Encode an arbitrary list of strings → [len(texts), d] FloatTensor.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available()
                                    else "cpu")
    obj, dim, backend = _load_backbone(model_name, device)

    if backend == "st":
        arr = _encode_st(obj, texts, batch_size=batch_size)
    elif backend == "hf":
        arr = _encode_hf(obj, texts, device, batch_size=batch_size)
    else:
        arr = _encode_stub(texts, dim)

    return torch.from_numpy(arr)


# =============================================================================
# Public API — word-level + doc-level loaders
# =============================================================================
def load_contextual_word_embeddings(dictionary,
                                    model_name: str = DEFAULT_MODEL,
                                    device: Optional[torch.device] = None,
                                    cache_path: Optional[str] = None
                                    ) -> torch.Tensor:
    """
    Build an [|V|, d] word embedding matrix aligned to ``dictionary``.
    Phrase tokens (e.g. 'new_york') are detokenised before encoding.

    Caches to ``cache_path`` if provided.
    """
    if cache_path and os.path.exists(cache_path):
        _LOG.info(f"Loading cached word embeddings from {cache_path}")
        return torch.load(cache_path, map_location="cpu")

    tokens = [t.replace("_", " ") for t, _ in sorted(
        dictionary.token2id.items(), key=lambda kv: kv[1])]
    _LOG.info(f"Encoding {len(tokens):,} vocabulary tokens with {model_name}")
    emb = encode_texts(tokens, model_name=model_name, device=device)
    _LOG.info(f"  word embedding matrix: {tuple(emb.shape)}")

    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        torch.save(emb, cache_path)
    return emb


def load_contextual_doc_embeddings(docs: Iterable[List[str]],
                                   model_name: str = DEFAULT_MODEL,
                                   device: Optional[torch.device] = None,
                                   cache_path: Optional[str] = None
                                   ) -> torch.Tensor:
    """
    Build an [N_docs, d] document embedding matrix.

    Each document is joined into a single string (underscores preserved so
    phrases are treated as one token by WordPiece / BPE).
    """
    if cache_path and os.path.exists(cache_path):
        _LOG.info(f"Loading cached doc embeddings from {cache_path}")
        return torch.load(cache_path, map_location="cpu")

    texts = [" ".join(d).replace("_", " ") for d in docs]
    _LOG.info(f"Encoding {len(texts):,} documents with {model_name}")
    emb = encode_texts(texts, model_name=model_name, device=device,
                       batch_size=128)
    _LOG.info(f"  doc embedding matrix: {tuple(emb.shape)}")

    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        torch.save(emb, cache_path)
    return emb


def get_contextual_dim(model_name: str = DEFAULT_MODEL,
                       device: Optional[torch.device] = None) -> int:
    """Return the embedding dimensionality for ``model_name``."""
    device = device or torch.device("cuda" if torch.cuda.is_available()
                                    else "cpu")
    _, dim, _ = _load_backbone(model_name, device)
    return dim
