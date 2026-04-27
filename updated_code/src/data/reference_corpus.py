"""
============================================================================
 src/data/reference_corpus.py
 ---------------------------------------------------------------------------
 Palmetto-style sliding-window reference corpus.

 Why this matters
 ────────────────
 C_V and C_NPMI were originally defined by Röder et al. (2015) using a
 sliding window over an EXTERNAL reference corpus (Wikipedia). Using the
 training corpus with document-level co-occurrence UNDERESTIMATES true
 coherence and produces negative values.

 This module builds an internal sliding-window reference that mimics the
 Palmetto behaviour — the standard used by every recent top-tier paper.
============================================================================
"""
from __future__ import annotations

from typing import List
from ..utils.logging_utils import get_logger

_LOG = get_logger(__name__)


def build_reference_corpus(docs: List[List[str]],
                           window_size: int = 10) -> List[List[str]]:
    """
    Build a list of sliding windows from tokenised documents.

    A document of length L yields  max(1, L − w + 1)  windows of size  w.
    Each window is treated as an independent "pseudo-document" during
    coherence statistics computation.
    """
    _LOG.info(f"Building sliding-window reference corpus (w={window_size})")
    windows = []
    for doc in docs:
        if len(doc) <= window_size:
            windows.append(list(doc))
        else:
            for i in range(len(doc) - window_size + 1):
                windows.append(doc[i:i + window_size])
    _LOG.info(f"  {len(windows):,} windows extracted")
    return windows
