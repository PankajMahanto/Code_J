"""
============================================================================
 src/utils/contextual_loader.py
 ---------------------------------------------------------------------------
 Contextual embedding loader using sentence-transformers.

 Replaces the old static-GloVe pipeline with two functions:

   • load_contextual_word_embeddings(model_name, dictionary)
       Encodes every vocabulary token (including phrase tokens like
       "new_york") as a single "sentence" through a sentence-transformer.
       Returns [V, d] used by the SCAD decoder for the topic-word geometry.

   • compute_contextual_doc_embeddings(model_name, docs, ...)
       Encodes every document (joined back from its lemmatised tokens) as
       a single sentence. Returns [N, d] used by the SGP-Encoder, which
       concatenates the contextual representation with the BoW vector
       BEFORE the MLP path (Contextual Topic Model architecture).

 Caching
 -------
 Both functions cache the resulting tensor next to the artifact directory
 so we don't pay the encoding cost on every training restart.
============================================================================
"""
from __future__ import annotations

import os
import hashlib
from typing import List, Optional

import numpy as np
import torch
from gensim.corpora import Dictionary

from .logging_utils import get_logger

_LOG = get_logger(__name__)

_DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def _hash_key(items: List[str], salt: str) -> str:
    h = hashlib.sha1(salt.encode("utf-8"))
    for s in items:
        h.update(s.encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()[:16]


def _load_model(model_name: str):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        raise ImportError(
            "sentence-transformers is required. Install with:\n"
            "    pip install sentence-transformers"
        ) from e
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _LOG.info(f"Loading sentence-transformer '{model_name}' on {device} ...")
    return SentenceTransformer(model_name, device=device)


def _normalize_token(tok: str) -> str:
    return tok.replace("_", " ")


def load_contextual_word_embeddings(
    model_name: str,
    dictionary: Dictionary,
    cache_dir: Optional[str] = None,
    batch_size: int = 256,
) -> torch.Tensor:
    """
    Returns a [V, d] tensor where row i is the contextual embedding for
    dictionary.id2token[i].  Phrase tokens "new_york" → "new york".
    """
    V = len(dictionary)
    tokens = [dictionary[i] for i in range(V)]

    # cache lookup
    if cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)
        key = _hash_key(tokens, salt=f"word::{model_name}")
        cache_path = os.path.join(cache_dir, f"ctx_word_emb_{key}.pt")
        if os.path.exists(cache_path):
            _LOG.info(f"Loading cached contextual word embeddings: {cache_path}")
            return torch.load(cache_path, map_location="cpu")

    model = _load_model(model_name)
    sentences = [_normalize_token(t) for t in tokens]
    _LOG.info(f"Encoding {V:,} vocabulary tokens with '{model_name}' ...")
    emb = model.encode(
        sentences,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    emb = torch.from_numpy(np.asarray(emb, dtype=np.float32))

    if cache_dir is not None:
        torch.save(emb, cache_path)
        _LOG.info(f"  cached -> {cache_path}")

    _LOG.info(f"  word embedding matrix: {tuple(emb.shape)}")
    return emb


def compute_contextual_doc_embeddings(
    model_name: str,
    docs: List[List[str]],
    cache_dir: Optional[str] = None,
    batch_size: int = 64,
) -> torch.Tensor:
    """
    Returns [N, d] dense doc embeddings. Each doc is detokenized
    (underscores back to spaces) and passed through sentence-transformer.
    """
    N = len(docs)
    sentences = [" ".join(_normalize_token(w) for w in d) for d in docs]

    if cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)
        key = _hash_key(sentences[:1024], salt=f"doc::{model_name}::{N}")
        cache_path = os.path.join(cache_dir, f"ctx_doc_emb_{key}.pt")
        if os.path.exists(cache_path):
            _LOG.info(f"Loading cached contextual doc embeddings: {cache_path}")
            return torch.load(cache_path, map_location="cpu")

    model = _load_model(model_name)
    _LOG.info(f"Encoding {N:,} documents with '{model_name}' ...")
    emb = model.encode(
        sentences,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    emb = torch.from_numpy(np.asarray(emb, dtype=np.float32))

    if cache_dir is not None:
        torch.save(emb, cache_path)
        _LOG.info(f"  cached -> {cache_path}")

    _LOG.info(f"  doc embedding matrix: {tuple(emb.shape)}")
    return emb


def get_contextual_dim(model_name: str = _DEFAULT_MODEL) -> int:
    """Returns the embedding dimension of the chosen sentence-transformer."""
    model = _load_model(model_name)
    return int(model.get_sentence_embedding_dimension())
