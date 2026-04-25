"""Evaluation: coherence statistics, metrics, diversity, topic quality gate."""
from .coherence_stats   import CoherenceStats
from .coherence_metrics import c_npmi, c_v, c_umass, c_uci
from .diversity_metrics import (
    topic_diversity, inter_topic_cosine, intra_topic_cosine,
)
from .quality_gate      import apply_quality_gate, evaluate_topics

__all__ = [
    "CoherenceStats",
    "c_npmi", "c_v", "c_umass", "c_uci",
    "topic_diversity", "inter_topic_cosine", "intra_topic_cosine",
    "apply_quality_gate", "evaluate_topics",
]
