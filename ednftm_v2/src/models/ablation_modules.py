"""
============================================================================
 src/models/ablation_modules.py
 ---------------------------------------------------------------------------
 Vanilla (non-novel) building blocks used to REPLACE each novelty during
 ablation experiments.

     variant       FULL      no_SGPE            no_EMGD                no_SCAD
     ─────────────────────────────────────────────────────────────────────────
     encoder       SGP-E     VanillaMLPEncoder  SGP-E                  SGP-E
     routing       EMGD-CR   EMGD-CR            VanillaDynamicRouting  EMGD-CR
     decoder       SCAD      SCAD               SCAD                   VanillaSoftmaxDecoder
============================================================================
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# Vanilla MLP encoder (no GCN, no Poincaré, standard KL)
# =============================================================================
class VanillaMLPEncoder(nn.Module):
    """Plain 2-layer MLP encoder used when SGP-E is ablated OFF."""

    def __init__(self,
                 vocab_size:     int,
                 hidden_dim:     int,
                 topic_dim:      int,
                 capsule_module: nn.Module,
                 dropout_rate:   float = 0.30,
                 **_):
        super().__init__()
        self.K = topic_dim
        self.capsule = capsule_module

        self.fc1 = nn.Linear(vocab_size, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.drop = nn.Dropout(dropout_rate)

        self.fc_mu     = nn.Linear(hidden_dim, topic_dim)
        self.fc_logvar = nn.Linear(hidden_dim, topic_dim)

    def set_adj_norm(self, _pmi):         # no-op (keeps API parity with SGP-E)
        pass

    def forward(self, x: torch.Tensor,
                x_ctx=None,
                temperature: float = 1.0):
        # VanillaMLPEncoder is the no-novelty baseline: ignore contextual input
        _ = x_ctx
        h = self.drop(F.relu(self.bn1(self.fc1(x))))
        h = F.relu(self.bn2(self.fc2(h)))

        mu     = self.fc_mu(h)
        logvar = self.fc_logvar(h).clamp(min=-10, max=4)

        std = (0.5 * logvar).exp()
        eps = torch.randn_like(std)
        z   = mu + eps * std

        # standard Gaussian KL
        kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(-1).mean()

        theta = self.capsule(z, temperature=temperature)
        return z, theta, mu, logvar, kl


# =============================================================================
# Vanilla dynamic routing (Sabour et al. 2017) — no momentum, no graph, no
# Lorentzian squash, no entropic temperature
# =============================================================================
class VanillaDynamicRouting(nn.Module):
    """Classic dynamic routing, used when EMGD-CR is ablated OFF."""

    def __init__(self,
                 input_dim:     int,
                 topic_dim:     int,
                 routing_iters: int = 3,
                 **_):
        super().__init__()
        self.K             = topic_dim
        self.routing_iters = routing_iters
        self.W = nn.Parameter(
            torch.randn(topic_dim, input_dim) * (1.0 / input_dim ** 0.5)
        )

    @staticmethod
    def _squash(s: torch.Tensor) -> torch.Tensor:
        n2 = (s * s).sum(-1, keepdim=True)
        return (n2 / (1 + n2)) * s / (n2.sqrt() + 1e-9)

    def forward(self, x: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        u = x @ self.W.T                                    # [B, K]
        b = torch.zeros(x.size(0), self.K, device=x.device)
        for _ in range(self.routing_iters):
            c = F.softmax(b, dim=1)                         # [B, K]
            s = u * c                                       # [B, K]
            v = self._squash(s)
            b = b + (u * v).sum(-1, keepdim=True).squeeze(-1).unsqueeze(-1).expand_as(b)
        return F.softmax(b, dim=1)


# =============================================================================
# Vanilla softmax decoder (plain β = softmax(W))
# =============================================================================
class VanillaSoftmaxDecoder(nn.Module):
    """Plain softmax-parameterised β, used when SCAD is ablated OFF."""

    def __init__(self, n_topics: int, vocab_size: int, **_):
        super().__init__()
        self.beta_raw = nn.Parameter(torch.randn(n_topics, vocab_size) * 0.02)

    def forward(self) -> torch.Tensor:
        return F.softmax(self.beta_raw, dim=-1)
