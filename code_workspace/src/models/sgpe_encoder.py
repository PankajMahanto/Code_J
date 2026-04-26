"""
============================================================================
 src/models/sgpe_encoder.py  [PATCHED — v3 / Contextual Topic Model]
 ---------------------------------------------------------------------------
 NOVELTY 1 — Spectral Graph-Infused Hierarchical Poincaré Encoder (SGP-E)

 v3 PATCH NOTES
 ──────────────
 1. CONTEXTUAL INPUT FUSION
    Encoder now accepts a per-document sentence-transformer embedding
    (``x_ctx``) alongside the BoW vector.  The two signals are
    concatenated *before* the dense MLP path:

        x_combined = concat([log1p(BoW), x_ctx])                [B, V + d_ctx]

    The spectral GCN path still runs on BoW alone (its adjacency is the
    word-word PMI graph), but the encoder's MLP and fusion layers now see
    the dense contextual signal directly.  This dramatically alleviates
    the "short-text starvation" that previously caused mode collapse.

 2. Tighter logvar clamp:  min=-6 (was -10)  — prevents sigma_sq → 0.
 3. Poincaré scale schedulable from outside.
 4. Input sanitisation — replace any NaN in x with 0 before forward pass.
============================================================================
"""
from __future__ import annotations

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
                 dropout_rate:   float = 0.30,
                 poincare_c:     float = 1.00,
                 poincare_scale: float = 0.10,
                 fisher_rao_lam: float = 0.05,
                 ctx_dim:        int   = 0):
        super().__init__()

        self.K              = topic_dim
        self.V              = vocab_size
        self.ctx_dim        = ctx_dim
        self.poincare       = PoincareBall(c=poincare_c)
        self.poincare_scale = poincare_scale
        self.fisher_rao_lam = fisher_rao_lam
        self.capsule        = capsule_module

        gcn_out   = hidden_dim // 2
        mlp_out   = hidden_dim
        fused_dim = mlp_out + gcn_out

        # (a) Spectral GCN path — BoW only (uses PMI adjacency)
        self.gcn = SpectralGraphConv(vocab_size, hidden_dim, gcn_out)
        self.register_buffer("adj_norm", torch.eye(vocab_size))

        # (b) MLP path — receives concat([BoW, ctx]) when ctx_dim > 0
        mlp_in     = vocab_size + ctx_dim
        self.fc1   = nn.Linear(mlp_in, mlp_out)
        self.bn1   = nn.BatchNorm1d(mlp_out)
        self.drop1 = nn.Dropout(dropout_rate)

        # (c) Fusion of GCN + MLP paths
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
                x:     torch.Tensor,
                x_ctx: torch.Tensor | None = None,
                temperature: float = 1.0):
        # ── Input sanitisation ──
        if not torch.isfinite(x).all():
            x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=0.0)

        # GCN path (BoW only)
        h_graph = self.gcn(x, self.adj_norm)

        # MLP path receives concat([BoW, ctx])
        if self.ctx_dim > 0:
            if x_ctx is None:
                x_ctx = torch.zeros(x.size(0), self.ctx_dim,
                                    device=x.device, dtype=x.dtype)
            elif not torch.isfinite(x_ctx).all():
                x_ctx = torch.nan_to_num(x_ctx, nan=0.0,
                                         posinf=1.0, neginf=-1.0)
            x_in = torch.cat([x, x_ctx], dim=-1)
        else:
            x_in = x

        h_mlp   = self.drop1(F.relu(self.bn1(self.fc1(x_in))))
        h_cat   = torch.cat([h_mlp, h_graph], dim=-1)
        h       = self.drop2(F.relu(self.bn_fuse(self.fc_fuse(h_cat))))

        # ── tighter logvar clamp ──
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
