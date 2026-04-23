"""
============================================================================
 src/models/sgpe_encoder.py  [REWRITE — v3]
 ---------------------------------------------------------------------------
 NOVELTY 1 — Spectral Graph-Infused Hierarchical Poincaré Encoder (SGP-E)

 v3 CHANGES
 ──────────
 • `context_dim` argument — the MLP path now ingests
       concat( log1p(BoW), contextual_sbert_emb )
   while the spectral-GCN path continues to operate on the BoW view
   because it is vocabulary-indexed through the PMI adjacency.
 • tighter logvar clamp [-6, 2] — prevents σ² collapse.
 • input sanitisation unchanged.
============================================================================
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..modules import PoincareBall, SpectralGraphConv, fisher_rao_kl


class SGPEncoder(nn.Module):
    """Spectral + Hyperbolic + Fisher–Rao encoder with contextual fusion."""

    def __init__(self,
                 vocab_size:     int,
                 hidden_dim:     int,
                 topic_dim:      int,
                 capsule_module: nn.Module,
                 context_dim:    int   = 0,
                 dropout_rate:   float = 0.30,
                 poincare_c:     float = 1.00,
                 poincare_scale: float = 0.10,
                 fisher_rao_lam: float = 0.05):
        super().__init__()

        self.K              = topic_dim
        self.vocab_size     = vocab_size
        self.context_dim    = context_dim
        self.poincare       = PoincareBall(c=poincare_c)
        self.poincare_scale = poincare_scale
        self.fisher_rao_lam = fisher_rao_lam
        self.capsule        = capsule_module

        gcn_out   = hidden_dim // 2
        mlp_out   = hidden_dim
        fused_dim = mlp_out + gcn_out

        # (a) Spectral GCN path — vocabulary-indexed, BoW only
        self.gcn = SpectralGraphConv(vocab_size, hidden_dim, gcn_out)
        self.register_buffer("adj_norm", torch.eye(vocab_size))

        # MLP path — BoW concatenated with the sentence-transformer embedding
        self.fc1   = nn.Linear(vocab_size + context_dim, mlp_out)
        self.bn1   = nn.BatchNorm1d(mlp_out)
        self.drop1 = nn.Dropout(dropout_rate)

        # fusion
        self.fc_fuse = nn.Linear(fused_dim, hidden_dim)
        self.bn_fuse = nn.BatchNorm1d(hidden_dim)
        self.drop2   = nn.Dropout(dropout_rate)

        # latent projections
        self.fc_mu     = nn.Linear(hidden_dim, topic_dim)
        self.fc_logvar = nn.Linear(hidden_dim, topic_dim)

    def set_adj_norm(self, pmi_matrix: torch.Tensor) -> None:
        adj = SpectralGraphConv.build_adj_norm(
            pmi_matrix.to(next(self.parameters()).device)
        )
        self.adj_norm = adj

    def forward(self,
                x:           torch.Tensor,
                x_ctx:       Optional[torch.Tensor] = None,
                temperature: float = 1.0):
        # ── input sanitisation ─────────────────────────────────────────
        if not torch.isfinite(x).all():
            x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=0.0)
        if x_ctx is not None and not torch.isfinite(x_ctx).all():
            x_ctx = torch.nan_to_num(x_ctx, nan=0.0, posinf=1.0, neginf=0.0)

        # spectral path operates on the BoW view only
        h_graph = self.gcn(x, self.adj_norm)

        # MLP path: BoW concatenated with the contextual sentence vector
        if self.context_dim > 0 and x_ctx is not None and x_ctx.numel() > 0:
            x_mlp = torch.cat([x, x_ctx], dim=-1)
        else:
            x_mlp = x
        h_mlp = self.drop1(F.relu(self.bn1(self.fc1(x_mlp))))

        h_cat = torch.cat([h_mlp, h_graph], dim=-1)
        h     = self.drop2(F.relu(self.bn_fuse(self.fc_fuse(h_cat))))

        mu     = self.fc_mu(h)
        logvar = self.fc_logvar(h).clamp(min=-6.0, max=2.0)

        std = (0.5 * logvar).exp()
        eps = torch.randn_like(std)
        z_e = mu + eps * std

        z_h = self.poincare.expmap0(z_e * self.poincare_scale)
        z_h = self.poincare.proj(z_h)

        theta = self.capsule(z_h, temperature=temperature)

        kl = fisher_rao_kl(mu, logvar, lam=self.fisher_rao_lam)

        return z_h, theta, mu, logvar, kl
