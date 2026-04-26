"""
============================================================================
 src/utils/glove_loader.py
 ---------------------------------------------------------------------------
 Load GloVe word vectors and align them to a Gensim dictionary.
 Handles compound phrase tokens (e.g. "new_york") by averaging the vectors
 of their constituent unigrams when an exact match is unavailable.
============================================================================
"""
from __future__ import annotations

import os
import numpy as np
import torch
from gensim.corpora import Dictionary

from .logging_utils import get_logger

_LOG = get_logger(__name__)


def load_glove_aligned(path: str,
                       dictionary: Dictionary,
                       dim: int = 100) -> torch.Tensor:
    """
    Build an [|V|, dim] embedding matrix from GloVe, aligned to ``dictionary``.

    Strategy
    --------
    1. Load GloVe into a dict.
    2. For every token in ``dictionary``:
         a. Exact lookup in GloVe.
         b. If the token is a phrase (contains '_'), average the vectors of
            its components that ARE in GloVe.
         c. Otherwise, random Gaussian N(0, 0.01²).

    Parameters
    ----------
    path       : path to e.g. glove.6B.100d.txt
    dictionary : fitted Gensim Dictionary (final, filtered)
    dim        : embedding dimensionality

    Returns
    -------
    emb : torch.FloatTensor  shape = (|V|, dim)
    """
    V = len(dictionary)
    emb = np.random.normal(0.0, 0.01, (V, dim)).astype(np.float32)

    if not os.path.exists(path):
        _LOG.warning(f"GloVe file not found at {path} — using random init.")
        return torch.from_numpy(emb)

    _LOG.info(f"Loading GloVe from {path} (dim={dim}) ...")
    glove = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip().split()
            if len(parts) < dim + 1:
                continue
            glove[parts[0]] = np.asarray(parts[1:dim + 1], dtype=np.float32)
    _LOG.info(f"  GloVe vocabulary size: {len(glove):,}")

    covered = 0
    for token, idx in dictionary.token2id.items():
        if token in glove:
            emb[idx] = glove[token]
            covered += 1
        elif "_" in token:
            pieces = token.split("_")
            vecs = [glove[p] for p in pieces if p in glove]
            if vecs:
                emb[idx] = np.mean(vecs, axis=0)
                covered += 1
    pct = 100.0 * covered / max(V, 1)
    _LOG.info(f"  Aligned coverage: {covered:,}/{V:,} ({pct:.1f}%)")
    return torch.from_numpy(emb)
