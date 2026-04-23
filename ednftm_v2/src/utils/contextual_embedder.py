"""
============================================================================
 src/utils/contextual_embedder.py
 ---------------------------------------------------------------------------
 Contextual embedding utilities — replaces the static GloVe pipeline.

 Uses `sentence-transformers` (e.g. `all-MiniLM-L6-v2`, 384-d) to produce:
   1. Document-level contextual vectors  → concatenated with BoW before the
      encoder (Contextual Topic Model; Bianchi et al., 2021).
   2. Vocabulary-level contextual vectors → initial decoder word embeddings
      that replace the GloVe 100-d lookup used previously.

 Both are cached on disk so training reruns do not pay the encoder cost.
============================================================================
"""
from __future__ import annotations

import os
from typing import Iterable, List, Optional

import numpy as np
import torch

from .logging_utils import get_logger

_LOG = get_logger(__name__)

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def _get_sbert(model_name: str, device: Optional[str] = None):
    """Lazy import so the rest of the codebase still works without the dep."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "sentence-transformers is required. Install with:\n"
            "    pip install sentence-transformers"
        ) from exc
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    _LOG.info(f"Loading sentence-transformer: {model_name} on {device}")
    return SentenceTransformer(model_name, device=device)


def encode_documents(docs: List[List[str]],
                     cache_path: Optional[str] = None,
                     model_name: str = DEFAULT_MODEL,
                     batch_size: int = 64,
                     device: Optional[str] = None) -> torch.Tensor:
    """
    Encode each preprocessed document as a single dense contextual vector.

    Tokens are joined with a space to recover a natural-language string
    before being passed through the transformer.

    Returns
    -------
    emb : torch.FloatTensor  shape = (n_docs, sbert_dim)
    """
    if cache_path is not None and os.path.exists(cache_path):
        _LOG.info(f"  loading cached doc embeddings: {cache_path}")
        return torch.from_numpy(np.load(cache_path)).float()

    sbert = _get_sbert(model_name, device)
    sentences = [" ".join(d) if d else "." for d in docs]
    _LOG.info(f"  encoding {len(sentences):,} docs (batch={batch_size}) ...")
    emb = sbert.encode(
        sentences,
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)

    if cache_path is not None:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        np.save(cache_path, emb)
        _LOG.info(f"  cached doc embeddings → {cache_path}")
    return torch.from_numpy(emb)


def encode_vocabulary(vocab_tokens: Iterable[str],
                      cache_path: Optional[str] = None,
                      model_name: str = DEFAULT_MODEL,
                      batch_size: int = 128,
                      device: Optional[str] = None) -> torch.Tensor:
    """
    Encode every vocabulary token (including multi-word phrases such as
    `prime_minister`) with the sentence-transformer to build the
    decoder's word-embedding matrix.

    Returns
    -------
    emb : torch.FloatTensor  shape = (|V|, sbert_dim)
    """
    if cache_path is not None and os.path.exists(cache_path):
        _LOG.info(f"  loading cached vocab embeddings: {cache_path}")
        return torch.from_numpy(np.load(cache_path)).float()

    sbert = _get_sbert(model_name, device)
    tokens = [t.replace("_", " ") for t in vocab_tokens]
    _LOG.info(f"  encoding {len(tokens):,} vocabulary items ...")
    emb = sbert.encode(
        tokens,
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)

    if cache_path is not None:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        np.save(cache_path, emb)
        _LOG.info(f"  cached vocab embeddings → {cache_path}")
    return torch.from_numpy(emb)


def contextual_dim(model_name: str = DEFAULT_MODEL) -> int:
    """Return the hidden dimension of the configured encoder."""
    sbert = _get_sbert(model_name)
    return int(sbert.get_sentence_embedding_dimension())
