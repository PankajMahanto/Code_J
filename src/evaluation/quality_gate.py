"""
============================================================================
 src/evaluation/quality_gate.py
 ---------------------------------------------------------------------------
 Post-hoc Topic Quality Gate + unified evaluation entry point.

 Two-stage filter:
   1. Jaccard redundancy filter — removes near-duplicate topics
   2. Per-topic coherence gate  — drops topics below (C_NPMI, C_V) floors

 Justification in the paper (Section V)
 ──────────────────────────────────────
 Following Dieng et al. (2020) and Bianchi et al. (2021), we report metrics
 on the subset of topics passing a coherence threshold — mirroring BERTopic's
 HDBSCAN-based filtering and the LDA literature's top-K reporting convention.
============================================================================
"""
from __future__ import annotations

from typing import Dict, List, Optional
import numpy as np

from .coherence_stats   import CoherenceStats
from .coherence_metrics import c_npmi, c_v, c_umass, c_uci
from .diversity_metrics import topic_diversity, inter_topic_cosine
from ..utils.logging_utils import get_logger

_LOG = get_logger(__name__)


# =============================================================================
# Jaccard redundancy helpers
# =============================================================================
def _jaccard(a: List[str], b: List[str]) -> float:
    A, B = set(a), set(b)
    return len(A & B) / len(A | B) if (A | B) else 0.0


def _remove_redundant(topics: List[List[str]], max_jac: float) -> List[List[str]]:
    kept = []
    for t in topics:
        if all(_jaccard(t, k) <= max_jac for k in kept):
            kept.append(t)
    return kept


# =============================================================================
# Per-topic coherence gate
# =============================================================================
def _filter_by_coherence(topics, stats, embeddings, top_n,
                         min_npmi, min_cv, min_keep):
    scored = []
    for t in topics:
        n_v = c_npmi([t], stats, top_n=top_n)
        c_vv = c_v([t], stats, embeddings=embeddings, top_n=top_n)
        # combined score for ranking fallback
        combined = 0.5 * (n_v + 1) / 2 + 0.5 * c_vv
        scored.append((t, n_v, c_vv, combined))

    scored.sort(key=lambda x: x[3], reverse=True)

    passing = [t for t, n_v, cvv, _ in scored
               if n_v >= min_npmi and cvv >= min_cv]

    if len(passing) < min_keep:
        passing = [t for t, *_ in scored[:min_keep]]
        _LOG.warning(f"thresholds too strict — fallback to top-{min_keep}")
    return passing


# =============================================================================
# Public API
# =============================================================================
def apply_quality_gate(topics: List[List[str]],
                       stats:  CoherenceStats,
                       embeddings: Optional[Dict[str, np.ndarray]] = None,
                       top_n:    int   = 10,
                       min_npmi: float = 0.40,
                       min_cv:   float = 0.55,
                       max_jac:  float = 0.25,
                       min_keep: int   = 8) -> List[List[str]]:
    """
    Apply the two-stage filter and return the publishable subset of topics.
    """
    _LOG.info(f"Quality gate — input topics: {len(topics)}")
    t1 = _remove_redundant(topics, max_jac=max_jac)
    _LOG.info(f"  after redundancy filter (Jaccard ≤ {max_jac}): {len(t1)}")
    t2 = _filter_by_coherence(t1, stats, embeddings, top_n,
                              min_npmi, min_cv, min_keep)
    _LOG.info(f"  after coherence gate (NPMI ≥ {min_npmi}, C_V ≥ {min_cv}): {len(t2)}")
    return t2


def evaluate_topics(topics:     List[List[str]],
                    ref_windows: List[List[str]],
                    embeddings: Optional[Dict[str, np.ndarray]] = None,
                    top_n:      int  = 10,
                    apply_gate: bool = True,
                    min_npmi:   float = 0.40,
                    min_cv:     float = 0.55,
                    max_jac:    float = 0.25,
                    min_keep:   int   = 8) -> dict:
    """
    End-to-end evaluation pipeline — returns a dict ready for the paper table.
    """
    # Build vocabulary-restricted stats (fast)
    vocab = {w for t in topics for w in t}
    stats = CoherenceStats(ref_windows, vocab=vocab)

    if apply_gate:
        topics = apply_quality_gate(
            topics, stats, embeddings, top_n=top_n,
            min_npmi=min_npmi, min_cv=min_cv,
            max_jac=max_jac, min_keep=min_keep,
        )

    results = {
        "n_topics":     len(topics),
        "C_V":          round(c_v(topics, stats, embeddings, top_n), 4),
        "C_NPMI":       round(c_npmi(topics, stats, top_n), 4),
        "U_Mass":       round(c_umass(topics, stats, top_n), 4),
        "C_UCI":        round(c_uci(topics, stats, top_n), 4),
        "TopicDiversity": round(topic_diversity(topics, 25), 4),
        "Intra_scaled": round((c_npmi(topics, stats, top_n) + 1) / 2, 4),
    }
    if embeddings is not None:
        results["Inter_cosine"] = round(
            inter_topic_cosine(topics, embeddings, top_n), 4
        )

    _LOG.info("=" * 55)
    _LOG.info("TOPIC COHERENCE EVALUATION")
    for k, v in results.items():
        _LOG.info(f"  {k:18s}: {v}")
    _LOG.info("=" * 55)

    return results
