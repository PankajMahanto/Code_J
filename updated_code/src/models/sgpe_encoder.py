"""
============================================================================
 src/models/sgpe_encoder.py  [REWRITE — Contextual Topic Model variant]
 ---------------------------------------------------------------------------
 NOVELTY 1 — Spectral Graph-Infused Hierarchical Poincaré Encoder (SGP-E)

 NEW: Contextual fusion
 ----------------------
 The encoder now accepts a dense contextual document embedding `x_ctx`
 (e.g. from sentence-transformers/all-MiniLM-L6-v2). Internally:

   • Spectral GCN path operates on the BoW vector ONLY (it requires the
     vocab × vocab adjacency).
   • MLP path receives  concat([log1p(BoW), x_ctx])  — that is the
     traditional Bag-of-Words concatenated with the contextual sentence
     vector, exactly the Contextual Neural Topic Model recipe.
   • The two paths are fused before the latent projection.

 Other v3 patches retained:
   • logvar clamp [-6, 2]
   • Poincaré scale schedulable
   • NaN sanitisation on inputs
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
                 contextual_dim: int = 0,
                 dropout_rate:   float = 0.30,
                 poincare_c:     float = 1.00,
                 poincare_scale: float = 0.10,
                 fisher_rao_lam: float = 0.05):
        super().__init__()

        self.K              = topic_dim
        self.V              = vocab_size
        self.contextual_dim = int(contextual_dim)
        self.poincare       = PoincareBall(c=poincare_c)
        self.poincare_scale = poincare_scale
        self.fisher_rao_lam = fisher_rao_lam
        self.capsule        = capsule_module

        gcn_out   = hidden_dim // 2
        mlp_out   = hidden_dim
        fused_dim = mlp_out + gcn_out

        # (a) Spectral GCN path — operates on the BoW vector only.
        self.gcn = SpectralGraphConv(vocab_size, hidden_dim, gcn_out)
        self.register_buffer("adj_norm", torch.eye(vocab_size))

        # (b) MLP path — accepts concat([log1p(BoW), contextual_doc_embed]).
        mlp_in_dim = vocab_size + self.contextual_dim
        self.fc1   = nn.Linear(mlp_in_dim, mlp_out)
        self.bn1   = nn.BatchNorm1d(mlp_out)
        self.drop1 = nn.Dropout(dropout_rate)

        # (c) fusion
        self.fc_fuse = nn.Linear(fused_dim, hidden_dim)
        self.bn_fuse = nn.BatchNorm1d(hidden_dim)
        self.drop2   = nn.Dropout(dropout_rate)

        # (d) latent projections
        self.fc_mu     = nn.Linear(hidden_dim, topic_dim)
        self.fc_logvar = nn.Linear(hidden_dim, topic_dim)

    def set_adj_norm(self, pmi_matrix: torch.Tensor) -> None:
        adj = SpectralGraphConv.build_adj_norm(
            pmi_matrix.to(next(self.parameters()).device)
        )
        self.adj_norm = adj

    def forward(self,
                x_bow: torch.Tensor,
                x_ctx: torch.Tensor = None,
                temperature: float = 1.0):
        # ── Input sanitisation ──
        if not torch.isfinite(x_bow).all():
            x_bow = torch.nan_to_num(x_bow, nan=0.0, posinf=1.0, neginf=0.0)

        h_graph = self.gcn(x_bow, self.adj_norm)

        if self.contextual_dim > 0:
            assert x_ctx is not None, (
                "Encoder built with contextual_dim>0 but x_ctx is None."
            )
            if not torch.isfinite(x_ctx).all():
                x_ctx = torch.nan_to_num(x_ctx, nan=0.0,
                                         posinf=1.0, neginf=-1.0)
            mlp_in = torch.cat([x_bow, x_ctx], dim=-1)
        else:
            mlp_in = x_bow

        h_mlp = self.drop1(F.relu(self.bn1(self.fc1(mlp_in))))
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
