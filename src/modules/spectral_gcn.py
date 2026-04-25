"""
============================================================================
 src/modules/spectral_gcn.py
 ---------------------------------------------------------------------------
 Two-hop spectral graph convolution over the PMI vocabulary graph.

 Forward:
     H1 = ReLU( BN( (Â x) W1 ) )                     # 1-hop aggregation
     H2 = ReLU( BN( (Â Â x) W2 ) ) + skip(x)         # 2-hop + residual

 Â = D^{−½} (A + I) D^{−½}  — symmetric normalised Laplacian of the PMI
 graph; only positive PMI edges are retained (semantic co-occurrence).

 This is Novelty-1 sub-component (a).
============================================================================
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralGraphConv(nn.Module):
    """Two-hop Spectral GCN on the vocabulary-level PMI graph."""

    def __init__(self, vocab_size: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.W1      = nn.Linear(vocab_size, hidden_dim, bias=True)
        self.W2      = nn.Linear(vocab_size, out_dim,    bias=True)
        self.skip    = nn.Linear(vocab_size, out_dim,    bias=False)
        self.bn1     = nn.BatchNorm1d(hidden_dim)
        self.bn2     = nn.BatchNorm1d(out_dim)
        # auxiliary projection of 1-hop features to out_dim (small residual)
        self.project = nn.Linear(hidden_dim, out_dim, bias=False)

    # ── static helper: build Â from a PMI matrix ────────────────────────
    @staticmethod
    def build_adj_norm(pmi_matrix: torch.Tensor) -> torch.Tensor:
        """
        Build the symmetric-normalised adjacency matrix from a PMI matrix.
        Only positive PMI entries are kept.
        """
        A       = (pmi_matrix > 0).float() * pmi_matrix.clamp(min=0)
        A       = A + torch.eye(A.shape[0], device=A.device)
        degree  = A.sum(dim=1).clamp(min=1e-8)
        d_isqrt = degree.pow(-0.5)
        return d_isqrt.unsqueeze(1) * A * d_isqrt.unsqueeze(0)

    # ── forward pass ─────────────────────────────────────────────────────
    def forward(self, x: torch.Tensor, adj_norm: torch.Tensor) -> torch.Tensor:
        """
        x        : [B, V]  bag-of-words
        adj_norm : [V, V]  precomputed Â
        returns  : [B, out_dim]
        """
        # 1-hop
        x1 = x @ adj_norm                                # [B, V]
        h1 = F.relu(self.bn1(self.W1(x1)))               # [B, hidden]

        # 2-hop
        x2 = x1 @ adj_norm                               # [B, V]
        h2 = F.relu(self.bn2(self.W2(x2)))               # [B, out]

        # residual connections
        skip_raw = F.relu(self.skip(x))                  # [B, out]
        proj_h1  = self.project(h1) * 0.1                # [B, out] (small)

        return h2 + proj_h1 + skip_raw
