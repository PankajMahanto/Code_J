"""
============================================================================
 src/evaluation/diversity_metrics.py
 ---------------------------------------------------------------------------
 Topic diversity (Dieng et al., 2020), inter-topic cosine, intra-topic
 cosine metrics.
============================================================================
"""
from __future__ import annotations

from typing import Dict, List
import numpy as np

_EPS = 1e-12


def topic_diversity(topics: List[List[str]], top_n: int = 25) -> float:
    """
    Fraction of unique words across the top-N of all topics.
        TD = |unique(⋃_k topics_k[:top_n])|  /  (K * top_n)
    Target ≥ 0.95 for a publishable model.
    """
    all_w = [w for t in topics for w in t[:top_n]]
    return len(set(all_w)) / len(all_w) if all_w else 0.0


def inter_topic_cosine(topics: List[List[str]],
                       embeddings: Dict[str, np.ndarray],
                       top_n: int = 10) -> float:
    """
    Mean pairwise cosine similarity between topic centroids.
    Lower = more distinct topics.  Target ≤ 0.30.
    """
    centroids = []
    for t in topics:
        vs = [embeddings[w] for w in t[:top_n] if w in embeddings]
        if vs:
            centroids.append(np.mean(vs, axis=0))
    if len(centroids) < 2:
        return 0.0
    C = np.array(centroids)
    C = C / (np.linalg.norm(C, axis=1, keepdims=True) + _EPS)
    S = C @ C.T
    iu = np.triu_indices(len(C), k=1)
    return float(np.mean(S[iu]))


def intra_topic_cosine(topics: List[List[str]],
                       embeddings: Dict[str, np.ndarray],
                       top_n: int = 10) -> float:
    """
    Mean within-topic cosine similarity across the top-N words.

    For each topic we compute the average pairwise cosine similarity between
    the word embedding vectors of its top-N words, then average across
    topics.  Target range [0.85, 0.95] — high values mean each topic's words
    are semantically tight.
    """
    intras = []
    for t in topics:
        vs = [embeddings[w] for w in t[:top_n] if w in embeddings]
        if len(vs) < 2:
            continue
        V = np.array(vs)
        V = V / (np.linalg.norm(V, axis=1, keepdims=True) + _EPS)
        S = V @ V.T
        iu = np.triu_indices(len(V), k=1)
        intras.append(float(np.mean(S[iu])))
    return float(np.mean(intras)) if intras else 0.0
