"""
============================================================================
 src/models/sgpe_encoder.py  [PATCHED - v4]
 ---------------------------------------------------------------------------
 NOVELTY 1 - Spectral Graph-Infused Hierarchical Poincare Encoder (SGP-E)

 v4 CHANGE - Contextual Topic Model Architecture
 -----------------------------------------------
 * The MLP path now accepts a concatenation of [BoW  |  contextual doc
   embedding].  The spectral GCN path still consumes BoW only because
   its Laplacian is vocab-sized.
 * When `contextual_dim = 0` the encoder behaves exactly like v3 (pure
   BoW) so legacy checkpoints remain loadable via strict=False.
 * Keeps: tighter logvar clamp, NaN input sanitisation, schedulable
   Poincare scale.
============================================================================
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..modules import PoincareBall, SpectralGraphConv, fisher_rao_kl


class SGPEncoder(nn.Module):
    """Spectral + Hyperbolic + Fisher-Rao encoder (now contextual-aware)."""

    def __init__(self,
                 vocab_size:     int,
                 hidden_dim:     int,
                 topic_dim:      int,
                 capsule_module: nn.Module,
                 dropout_rate:   float = 0.30,
                 poincare_c:     float = 1.00,
                 poincare_scale: float = 0.10,
                 fisher_rao_lam: float = 0.05,
                 contextual_dim: int   = 0):
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

        # (a) Spectral GCN path - operates on BoW only.
        self.gcn = SpectralGraphConv(vocab_size, hidden_dim, gcn_out)
        self.register_buffer("adj_norm", torch.eye(vocab_size))

        # (b) MLP path - input is [BoW ; contextual] when contextual_dim > 0.
        mlp_in = vocab_size + self.contextual_dim
        self.fc1   = nn.Linear(mlp_in, mlp_out)
        self.bn1   = nn.BatchNorm1d(mlp_out)
        self.drop1 = nn.Dropout(dropout_rate)

        # (c) Fusion MLP
        self.fc_fuse = nn.Linear(fused_dim, hidden_dim)
        self.bn_fuse = nn.BatchNorm1d(hidden_dim)
        self.drop2   = nn.Dropout(dropout_rate)

        # (d) Latent projections
        self.fc_mu     = nn.Linear(hidden_dim, topic_dim)
        self.fc_logvar = nn.Linear(hidden_dim, topic_dim)

    def set_adj_norm(self, pmi_matrix: torch.Tensor) -> None:
        adj = SpectralGraphConv.build_adj_norm(
            pmi_matrix.to(next(self.parameters()).device)
        )
        self.adj_norm = adj

    def forward(self,
                x:     torch.Tensor,
                x_ctx: torch.Tensor = None,
                temperature: float = 1.0):
        """
        x     : [B, V]              BoW (raw or log1p counts)
        x_ctx : [B, ctx_dim] or None  contextual document embedding.
        """
        # Input sanitisation
        if not torch.isfinite(x).all():
            x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=0.0)

        # GCN path: BoW only (vocab-sized Laplacian)
        h_graph = self.gcn(x, self.adj_norm)

        # MLP path: [BoW ; contextual] if available
        if self.contextual_dim > 0:
            if x_ctx is None or x_ctx.numel() == 0:
                # pad with zeros so the model still runs if no ctx passed
                x_ctx = torch.zeros(x.size(0), self.contextual_dim,
                                    device=x.device, dtype=x.dtype)
            if not torch.isfinite(x_ctx).all():
                x_ctx = torch.nan_to_num(x_ctx, nan=0.0, posinf=1.0, neginf=0.0)
            x_mlp_in = torch.cat([x, x_ctx], dim=-1)
        else:
            x_mlp_in = x

        h_mlp = self.drop1(F.relu(self.bn1(self.fc1(x_mlp_in))))
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
        kl    = fisher_rao_kl(mu, logvar, lam=self.fisher_rao_lam)
        return z_h, theta, mu, logvar, kl
