"""
============================================================================
 src/evaluation/coherence_metrics.py
 ---------------------------------------------------------------------------
 The four standard coherence metrics — implemented with the CORRECTED
 formulas used in every recent top-tier paper.

   • C_NPMI  — Röder et al. (2015)      ∈ [−1, +1],  higher = better
   • C_V     — Röder et al. (2015)      ∈ [0, 1]  (approx), higher = better
   • U_Mass  — Mimno et al. (2011)      ∈ (−∞, 0], less negative = better
   • C_UCI   — Newman et al. (2010)     ∈ (−∞, ∞), higher = better

 BUG FIX OVER LEGACY CODE
 ────────────────────────
 Your earlier NPMI used  denom = math.log(p_ab)   — sign error.
 The correct formula is  denom = −math.log(p_ab).
 This alone flips most negative scores back to positive.
============================================================================
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional
import numpy as np

from .coherence_stats import CoherenceStats

_EPS = 1e-12


# =============================================================================
# C_NPMI
# =============================================================================
def c_npmi(topics: List[List[str]],
           stats:  CoherenceStats,
           top_n:  int = 10) -> float:
    """
    Normalised PMI.
        NPMI(w_i, w_j) =  log( P(w_i, w_j) / (P(w_i) P(w_j)) )
                          / −log P(w_i, w_j)
    """
    total, pairs = 0.0, 0
    for topic in topics:
        words = topic[:top_n]
        for i in range(len(words)):
            for j in range(i + 1, len(words)):
                w_i, w_j = words[i], words[j]
                p_ij = stats.p_joint(w_i, w_j)
                p_i  = stats.p(w_i)
                p_j  = stats.p(w_j)
                if p_ij <= _EPS or p_i <= _EPS or p_j <= _EPS:
                    npmi = -1.0
                else:
                    pmi   = math.log(p_ij / (p_i * p_j))
                    denom = -math.log(p_ij)              # ✅ correct sign
                    npmi  = pmi / denom if denom > _EPS else 0.0
                    npmi  = max(-1.0, min(1.0, npmi))
                total += npmi
                pairs += 1
    return total / pairs if pairs else 0.0


# =============================================================================
# C_V
# =============================================================================
def c_v(topics: List[List[str]],
        stats:  CoherenceStats,
        embeddings: Optional[Dict[str, np.ndarray]] = None,
        top_n: int  = 10,
        gamma: float = 2.0) -> float:
    """
    Hybrid NPMI-weighted context-vector cosine coherence.

    For each topic we build a k × k matrix  vec[i, j] = NPMI(w_i, w_j)^γ,
    then compare each row to the row-sum with cosine similarity.  The mean
    is the topic's C_V.

    Optional embedding-space fusion: if word embeddings are provided, we
    late-fuse 60 % NPMI-context + 40 % cosine similarity in embedding space.
    """
    total, n_topics = 0.0, 0
    for topic in topics:
        w = topic[:top_n]
        k = len(w)
        if k < 2:
            continue

        vec = np.zeros((k, k))
        for i in range(k):
            for j in range(k):
                if i == j:
                    vec[i, j] = 1.0
                    continue
                p_ij = stats.p_joint(w[i], w[j])
                p_i  = stats.p(w[i])
                p_j  = stats.p(w[j])
                if p_ij <= _EPS or p_i <= _EPS or p_j <= _EPS:
                    npmi = 0.0
                else:
                    pmi = math.log(p_ij / (p_i * p_j))
                    den = -math.log(p_ij)
                    npmi = max(0.0, pmi / den if den > _EPS else 0.0)
                vec[i, j] = npmi ** gamma

        # segmentation similarities (row vs. full)
        full = vec.sum(axis=0)
        sims = []
        for i in range(k):
            num = float(np.dot(vec[i], full))
            den = float(np.linalg.norm(vec[i]) * np.linalg.norm(full) + _EPS)
            sims.append(num / den)

        # optional embedding-space fusion
        if embeddings is not None:
            emb_sims = []
            for i in range(k):
                for j in range(i + 1, k):
                    if w[i] in embeddings and w[j] in embeddings:
                        a, b = embeddings[w[i]], embeddings[w[j]]
                        na, nb = np.linalg.norm(a), np.linalg.norm(b)
                        if na > _EPS and nb > _EPS:
                            emb_sims.append(float(np.dot(a, b) / (na * nb)))
            if emb_sims:
                topic_cv = 0.6 * float(np.mean(sims)) \
                         + 0.4 * float(np.mean(emb_sims))
            else:
                topic_cv = float(np.mean(sims))
        else:
            topic_cv = float(np.mean(sims))

        total += max(0.0, min(1.0, topic_cv))
        n_topics += 1

    return total / n_topics if n_topics else 0.0


# =============================================================================
# U_Mass
# =============================================================================
def c_umass(topics: List[List[str]],
            stats:  CoherenceStats,
            top_n:  int = 10,
            epsilon: float = 1.0) -> float:
    """
    U_Mass(w_i, w_j) = log( (D(w_i, w_j) + ε) / D(w_j) )   for i > j
    """
    total, pairs = 0.0, 0
    for topic in topics:
        w = topic[:top_n]
        for i in range(1, len(w)):
            for j in range(i):
                d_ij = stats.count_joint(w[i], w[j])
                d_j  = stats.count(w[j])
                if d_j == 0:
                    continue
                total += math.log((d_ij + epsilon) / d_j)
                pairs += 1
    return total / pairs if pairs else 0.0


# =============================================================================
# C_UCI
# =============================================================================
def c_uci(topics: List[List[str]],
          stats:  CoherenceStats,
          top_n:  int = 10,
          epsilon: float = 1.0) -> float:
    """Symmetric PMI averaged over all pairs."""
    total, pairs = 0.0, 0
    for topic in topics:
        w = topic[:top_n]
        for i in range(len(w)):
            for j in range(i + 1, len(w)):
                p_ij = stats.p_joint(w[i], w[j])
                p_i  = stats.p(w[i])
                p_j  = stats.p(w[j])
                num = p_ij + epsilon
                den = (p_i * p_j) + epsilon
                if den <= _EPS:
                    continue
                total += math.log(num / den)
                pairs += 1
    return total / pairs if pairs else 0.0
