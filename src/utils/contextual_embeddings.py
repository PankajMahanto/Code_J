"""
============================================================================
 src/utils/contextual_embeddings.py
 ---------------------------------------------------------------------------
 Sentence-Transformer driven contextual embeddings.

 Replaces static GloVe embeddings with dense contextual representations from
 a transformer (default: ``sentence-transformers/all-MiniLM-L6-v2``).

 Two outputs are produced:

   • doc_embeds   : [D, ctx_dim]  — one embedding per training document
                                    (concatenated with the BoW vector before
                                    the encoder).
   • word_embeds  : [V, ctx_dim]  — one embedding per vocabulary token
                                    (used by the SCAD decoder in place of
                                    the old GloVe matrix).

 The model is loaded lazily, the embeddings are encoded once, cached on disk,
 and then re-used for subsequent runs.
============================================================================
"""
from __future__ import annotations

import os
import hashlib
import pickle
from typing import List, Tuple

import numpy as np
import torch

from gensim.corpora import Dictionary

from .logging_utils import get_logger

_LOG = get_logger(__name__)


def _hash_key(items: List[str], model_name: str) -> str:
    h = hashlib.sha1(model_name.encode("utf-8"))
    for s in items:
        h.update(b"\x1f")
        h.update(s.encode("utf-8", errors="ignore"))
    return h.hexdigest()[:16]


def _load_st_model(model_name: str):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        raise ImportError(
            "sentence-transformers is required. Install with "
            "`pip install sentence-transformers`."
        ) from e
    _LOG.info(f"Loading SentenceTransformer: {model_name}")
    return SentenceTransformer(model_name)


def encode_documents(docs: List[List[str]],
                     model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                     cache_dir: str | None = None,
                     batch_size: int = 64,
                     device: str | None = None) -> torch.Tensor:
    """
    Encode every document (already tokenised) as a single dense vector.

    Tokens are joined back to a string with single spaces — phrase tokens
    (``new_york``) are converted to plain bigrams (``new york``) so the
    transformer's tokenizer behaves naturally.
    """
    flat = [" ".join(t.replace("_", " ") for t in d) for d in docs]
    key = _hash_key(flat, model_name)

    if cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)
        cache = os.path.join(cache_dir, f"doc_ctx_{key}.npy")
        if os.path.exists(cache):
            _LOG.info(f"Loading cached document embeddings: {cache}")
            return torch.from_numpy(np.load(cache))

    model = _load_st_model(model_name)
    if device is not None:
        model = model.to(device)
    arr = model.encode(flat, batch_size=batch_size, convert_to_numpy=True,
                       show_progress_bar=True, normalize_embeddings=True)
    arr = arr.astype(np.float32)

    if cache_dir is not None:
        np.save(cache, arr)
        _LOG.info(f"Saved document embeddings cache: {cache}  shape={arr.shape}")

    return torch.from_numpy(arr)


def encode_vocabulary(dictionary: Dictionary,
                      model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                      cache_dir: str | None = None,
                      batch_size: int = 256,
                      device: str | None = None) -> torch.Tensor:
    """
    Encode every vocabulary token as a dense vector. Phrase tokens are
    converted from ``new_york`` to ``new york``.
    """
    tokens = [t for _, t in sorted((i, w) for w, i in dictionary.token2id.items())]
    surface = [t.replace("_", " ") for t in tokens]
    key = _hash_key(surface, model_name)

    if cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)
        cache = os.path.join(cache_dir, f"vocab_ctx_{key}.npy")
        if os.path.exists(cache):
            _LOG.info(f"Loading cached vocabulary embeddings: {cache}")
            return torch.from_numpy(np.load(cache))

    model = _load_st_model(model_name)
    if device is not None:
        model = model.to(device)
    arr = model.encode(surface, batch_size=batch_size, convert_to_numpy=True,
                       show_progress_bar=False, normalize_embeddings=True)
    arr = arr.astype(np.float32)

    if cache_dir is not None:
        np.save(cache, arr)
        _LOG.info(f"Saved vocabulary embeddings cache: {cache}  shape={arr.shape}")

    return torch.from_numpy(arr)


def get_ctx_dim(model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> int:
    """Return the embedding dimension of the requested transformer model."""
    model = _load_st_model(model_name)
    return int(model.get_sentence_embedding_dimension())
