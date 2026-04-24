"""
============================================================================
 src/evaluation/coherence_stats.py
 ---------------------------------------------------------------------------
 Precomputes word and word-pair counts from the sliding-window reference
 corpus.  All four coherence metrics (C_NPMI, C_V, U_Mass, C_UCI) reuse
 these statistics — one-time cost, many-time benefit.
============================================================================
"""
from __future__ import annotations

from collections import Counter, defaultdict
from typing import List, Set, Optional


class CoherenceStats:
    """
    Pre-compute P(w) and P(w_i, w_j) from a list of token windows.

    Parameters
    ----------
    reference_windows : list of list of str
        Each inner list is one sliding window (pseudo-document).
    vocab : set of str, optional
        If given, restricts statistics to just this vocabulary — this is
        much faster when you only evaluate topics over a subset of tokens.
    """

    def __init__(self,
                 reference_windows: List[List[str]],
                 vocab: Optional[Set[str]] = None):
        self.N = len(reference_windows)
        self.word_count  : Counter = Counter()
        self.pair_count  : dict    = defaultdict(int)
        self._build(reference_windows, vocab)

    def _build(self, windows, vocab):
        for win in windows:
            uniq = set(win)
            if vocab is not None:
                uniq = uniq & vocab
            uniq = list(uniq)
            for w in uniq:
                self.word_count[w] += 1
            for i in range(len(uniq)):
                for j in range(i + 1, len(uniq)):
                    a, b = sorted([uniq[i], uniq[j]])
                    self.pair_count[(a, b)] += 1

    # ── probability / count accessors ───────────────────────────────────
    def p(self, w: str) -> float:
        return self.word_count.get(w, 0) / max(self.N, 1)

    def p_joint(self, a: str, b: str) -> float:
        key = tuple(sorted([a, b]))
        return self.pair_count.get(key, 0) / max(self.N, 1)

    def count(self, w: str) -> int:
        return self.word_count.get(w, 0)

    def count_joint(self, a: str, b: str) -> int:
        key = tuple(sorted([a, b]))
        return self.pair_count.get(key, 0)
