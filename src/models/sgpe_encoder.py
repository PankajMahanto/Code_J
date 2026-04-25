"""
============================================================================
 src/models/sgpe_encoder.py  [v3 — contextual fusion]
 ---------------------------------------------------------------------------
 NOVELTY 1 — Spectral Graph-Infused Hierarchical Poincaré Encoder (SGP-E)

 v3 CHANGES
 ──────────
 1. Static GloVe inputs are gone.  The encoder now fuses three streams:
        (a) Bag-of-Words (raw, log-scaled)         → BoW MLP
        (b) Sentence-Transformer document vector   → Contextual MLP
        (c) Spectral GCN on the PMI graph          → graph signal
    The three streams are concatenated then projected to the latent space.
 2. Tighter logvar clamp (min=-6) — prevents sigma_sq → 0 collapse.
 3. NaN sanitisation on both BoW and contextual inputs.
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
                 ctx_dim:        int,
                 dropout_rate:   float = 0.30,
                 poincare_c:     float = 1.00,
                 poincare_scale: float = 0.10,
                 fisher_rao_lam: float = 0.05):
        super().__init__()

        self.K              = topic_dim
        self.ctx_dim        = ctx_dim
        self.poincare       = PoincareBall(c=poincare_c)
        self.poincare_scale = poincare_scale
        self.fisher_rao_lam = fisher_rao_lam
        self.capsule        = capsule_module

        gcn_out   = hidden_dim // 2
        bow_out   = hidden_dim
        ctx_out   = hidden_dim // 2 if ctx_dim > 0 else 0
        fused_dim = bow_out + gcn_out + ctx_out

        # (a) Spectral GCN path on the PMI graph (BoW input)
        self.gcn = SpectralGraphConv(vocab_size, hidden_dim, gcn_out)
        self.register_buffer("adj_norm", torch.eye(vocab_size))

        # (b) BoW MLP path
        self.fc_bow  = nn.Linear(vocab_size, bow_out)
        self.bn_bow  = nn.BatchNorm1d(bow_out)
        self.drop_bow = nn.Dropout(dropout_rate)

        # (c) Contextual (sentence-transformer) MLP path
        if ctx_dim > 0:
            self.fc_ctx  = nn.Linear(ctx_dim, ctx_out)
            self.bn_ctx  = nn.BatchNorm1d(ctx_out)
            self.drop_ctx = nn.Dropout(dropout_rate)
        else:
            self.fc_ctx = None

        # Fusion
        self.fc_fuse = nn.Linear(fused_dim, hidden_dim)
        self.bn_fuse = nn.BatchNorm1d(hidden_dim)
        self.drop_fuse = nn.Dropout(dropout_rate)

        # Latent projections
        self.fc_mu     = nn.Linear(hidden_dim, topic_dim)
        self.fc_logvar = nn.Linear(hidden_dim, topic_dim)

    def set_adj_norm(self, pmi_matrix: torch.Tensor) -> None:
        adj = SpectralGraphConv.build_adj_norm(
            pmi_matrix.to(next(self.parameters()).device)
        )
        self.adj_norm = adj

    def forward(self,
                x:    torch.Tensor,
                ctx:  torch.Tensor | None = None,
                temperature: float = 1.0):
        # Input sanitisation
        if not torch.isfinite(x).all():
            x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=0.0)

        h_graph = self.gcn(x, self.adj_norm)
        h_bow   = self.drop_bow(F.relu(self.bn_bow(self.fc_bow(x))))

        parts = [h_bow, h_graph]

        if self.fc_ctx is not None:
            if ctx is None:
                raise ValueError("Encoder was built with ctx_dim>0 but ctx=None was passed")
            if not torch.isfinite(ctx).all():
                ctx = torch.nan_to_num(ctx, nan=0.0, posinf=1.0, neginf=0.0)
            h_ctx = self.drop_ctx(F.relu(self.bn_ctx(self.fc_ctx(ctx))))
            parts.append(h_ctx)

        h_cat = torch.cat(parts, dim=-1)
        h     = self.drop_fuse(F.relu(self.bn_fuse(self.fc_fuse(h_cat))))

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
