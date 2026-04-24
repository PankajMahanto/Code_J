"""
============================================================================
 src/utils/pmi.py
 ---------------------------------------------------------------------------
 Sliding-window PMI matrix.
 Used by:
   (a) SGP-E encoder  — as the adjacency matrix for the semantic graph.
   (b) Coherence loss — as a differentiable surrogate for NPMI maximisation.
============================================================================
"""
from __future__ import annotations

from typing import List
import numpy as np
import torch
from gensim.corpora import Dictionary

from .logging_utils import get_logger

_LOG = get_logger(__name__)


def compute_pmi_matrix(docs: List[List[str]],
                       dictionary: Dictionary,
                       window: int = 10,
                       clip: float = 10.0) -> torch.Tensor:
    """
    Compute a sliding-window PMI matrix over the vocabulary.

    Parameters
    ----------
    docs       : tokenised documents (list of list of str)
    dictionary : Gensim dictionary defining the token → id mapping
    window     : context window radius (symmetric, so 2*window neighbours)
    clip       : absolute value clip for numerical stability

    Returns
    -------
    pmi : torch.FloatTensor  shape = (|V|, |V|)
          PMI values clipped to [-clip, +clip].
    """
    V = len(dictionary)
    tok2id = dictionary.token2id

    _LOG.info(f"Computing PMI (vocab={V:,}, window={window}) ...")
    cooc = np.zeros((V, V), dtype=np.float64)

    n_docs = len(docs)
    for di, doc in enumerate(docs):
        if di % 5000 == 0 and di > 0:
            _LOG.info(f"  PMI progress: {di:,}/{n_docs:,}")
        ids = [tok2id[w] for w in doc if w in tok2id]
        for i, wi in enumerate(ids):
            start = max(0, i - window)
            end   = min(len(ids), i + window + 1)
            for wj in ids[start:i] + ids[i + 1:end]:
                cooc[wi, wj] += 1.0

    total = cooc.sum()
    if total == 0:
        total = 1.0
    p_i   = cooc.sum(axis=1) / total
    p_j   = cooc.sum(axis=0) / total
    p_ij  = cooc / total

    pmi = np.log(
        (p_ij + 1e-10) / (p_i[:, None] * p_j[None, :] + 1e-10)
    )
    pmi = np.clip(pmi, -clip, clip)

    _LOG.info(f"  PMI non-zero (positive) entries: {(pmi > 0).sum():,}")
    return torch.tensor(pmi, dtype=torch.float32)
