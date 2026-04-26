"""
============================================================================
 src/models/sgpe_encoder.py  [PATCHED — v2]
 ---------------------------------------------------------------------------
 NOVELTY 1 — Spectral Graph-Infused Hierarchical Poincaré Encoder (SGP-E)

 PATCH NOTES (v2)
 ────────────────
 1. Tighter logvar clamp:  min=-6 (was -10)  — prevents sigma_sq → 0.
 2. Poincaré scale now schedulable from outside (for gentler warmup).
 3. Input sanitisation — replace any NaN in x with 0 before forward pass.
============================================================================
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..modules import PoincareBall, SpectralGraphConv, fisher_rao_kl


class SGPEncoder(nn.Module):
    """Spectral + Hyperbolic + Fisher–Rao encoder."""

    def __init__(self,
                 vocab_size:     int,
                 hidden_dim:     int,
                 topic_dim:      int,
                 capsule_module: nn.Module,
                 dropout_rate:   float = 0.30,
                 poincare_c:     float = 1.00,
                 poincare_scale: float = 0.10,
                 fisher_rao_lam: float = 0.05):
        super().__init__()

        self.K              = topic_dim
        self.poincare       = PoincareBall(c=poincare_c)
        self.poincare_scale = poincare_scale
        self.fisher_rao_lam = fisher_rao_lam
        self.capsule        = capsule_module

        gcn_out   = hidden_dim // 2
        mlp_out   = hidden_dim
        fused_dim = mlp_out + gcn_out

        # (a) Spectral GCN path
        self.gcn = SpectralGraphConv(vocab_size, hidden_dim, gcn_out)
        self.register_buffer("adj_norm", torch.eye(vocab_size))

        # MLP path
        self.fc1   = nn.Linear(vocab_size, mlp_out)
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

    def forward(self, x: torch.Tensor, temperature: float = 1.0):
        # ── Input sanitisation ──
        if not torch.isfinite(x).all():
            x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=0.0)

        h_graph = self.gcn(x, self.adj_norm)
        h_mlp   = self.drop1(F.relu(self.bn1(self.fc1(x))))
        h_cat   = torch.cat([h_mlp, h_graph], dim=-1)
        h       = self.drop2(F.relu(self.bn_fuse(self.fc_fuse(h_cat))))

        # ── PATCH: tighter logvar clamp ──
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
