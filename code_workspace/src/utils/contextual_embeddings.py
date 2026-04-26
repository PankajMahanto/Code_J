"""
============================================================================
 src/utils/contextual_embeddings.py
 ---------------------------------------------------------------------------
 Sentence-transformer based contextual embeddings.

 Replaces the static GloVe 100-d vectors with dense contextual embeddings
 from a pretrained sentence-transformer (default: all-MiniLM-L6-v2, 384-d).

 Two artefacts are produced:

   1. token_embeds : [V, ctx_dim]
        One vector per vocabulary token.  Each token is encoded as a short
        natural-language query (the token itself, with underscores expanded
        to spaces for phrase tokens).  This vector replaces GloVe in the
        SCAD decoder's word_embeds buffer.

   2. doc_embeds   : [D, ctx_dim]
        One vector per training document (mean-pooled by the encoder).
        These are concatenated with the BoW input inside the SGP-E encoder
        to give the model rich contextual signal beyond bag-of-words.

 The encoder runs once at preprocessing time and the artefacts are cached
 on disk so subsequent training runs reuse them (`torch.save`).
============================================================================
"""
from __future__ import annotations

import os
from typing import List

import numpy as np
import torch

from .logging_utils import get_logger

_LOG = get_logger(__name__)

_DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"   # 384-d


def _load_st_model(model_name: str, device: torch.device):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        raise ImportError(
            "sentence-transformers is required.  Install with:\n"
            "    pip install sentence-transformers"
        ) from e
    _LOG.info(f"Loading sentence-transformer: {model_name}")
    return SentenceTransformer(model_name, device=str(device))


def encode_vocabulary(dictionary,
                      model_name: str = _DEFAULT_MODEL,
                      device: torch.device | None = None,
                      batch_size: int = 256,
                      cache_path: str | None = None) -> torch.Tensor:
    """
    Returns [V, ctx_dim] token embeddings aligned to ``dictionary``.

    Phrase tokens such as ``new_york`` are first un-mangled to
    ``new york`` so the sentence-transformer can attend to the whole
    bigram/trigram as a meaningful unit.
    """
    if cache_path and os.path.exists(cache_path):
        _LOG.info(f"  loading cached vocab embeddings from {cache_path}")
        return torch.load(cache_path, map_location="cpu")

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = _load_st_model(model_name, device)

    V = len(dictionary)
    tokens: List[str] = [""] * V
    for tok, idx in dictionary.token2id.items():
        tokens[idx] = tok.replace("_", " ")

    _LOG.info(f"  encoding {V:,} vocabulary tokens ...")
    emb = model.encode(
        tokens,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=False,
        show_progress_bar=False,
    ).astype(np.float32)

    out = torch.from_numpy(emb)
    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        torch.save(out, cache_path)
        _LOG.info(f"  cached vocab embeddings -> {cache_path}")
    return out


def encode_documents(docs: List[List[str]],
                     model_name: str = _DEFAULT_MODEL,
                     device: torch.device | None = None,
                     batch_size: int = 128,
                     cache_path: str | None = None) -> torch.Tensor:
    """
    Returns [D, ctx_dim] document embeddings.  Each tokenised doc is
    rejoined with spaces (phrase underscores stripped) before encoding.
    """
    if cache_path and os.path.exists(cache_path):
        _LOG.info(f"  loading cached doc embeddings from {cache_path}")
        return torch.load(cache_path, map_location="cpu")

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = _load_st_model(model_name, device)

    texts = [" ".join(tok.replace("_", " ") for tok in d) for d in docs]
    _LOG.info(f"  encoding {len(texts):,} documents ...")
    emb = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=False,
        show_progress_bar=False,
    ).astype(np.float32)

    out = torch.from_numpy(emb)
    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        torch.save(out, cache_path)
        _LOG.info(f"  cached doc embeddings -> {cache_path}")
    return out


def get_contextual_dim(model_name: str = _DEFAULT_MODEL) -> int:
    """Return the embedding dimensionality of a sentence-transformer."""
    model = _load_st_model(model_name, torch.device("cpu"))
    return int(model.get_sentence_embedding_dimension())
