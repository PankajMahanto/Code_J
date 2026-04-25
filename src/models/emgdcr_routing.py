"""
============================================================================
 src/models/emgdcr_routing.py
 ---------------------------------------------------------------------------
 NOVELTY 3 — Entropic Momentum Graph-Diffused Capsule Routing  (EMGD-CR)

 Four sub-contributions over classical Dynamic Routing (Sabour et al., 2017):
   (a) Momentum on routing logits       – EMA across batches
   (b) Annealed entropic temperature    – softmax(b / T), T: T_max → T_min
   (c) Learnable topic-topic graph      – K×K symmetric adjacency spreads
                                          routing signal between related topics
   (d) Lorentzian squash function       – hyperbolic (Lorentz-model) squash

 Why these help
 ──────────────
 (a) Cold-start every batch wastes past routing evidence.  EMA carries it.
 (b) High T at start → soft / exploratory routing (avoids topic collapse).
     Low T late      → sharp, decisive assignments.
 (c) Topics are not independent; diffusion lets related topics co-activate.
 (d) Lorentzian squash has a speed-of-light ceiling at 1 and sharper
     gradient near 0 than tanh — faster early routing.
============================================================================
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class EMGDCapsuleRouting(nn.Module):
    """Entropic Momentum Graph-Diffused Capsule Routing."""

    def __init__(self,
                 input_dim:     int,
                 topic_dim:     int,
                 routing_iters: int   = 5,
                 momentum:      float = 0.90):
        super().__init__()
        self.K             = topic_dim
        self.routing_iters = routing_iters
        self.momentum      = momentum

        # vote (transformation) matrix
        self.W = nn.Parameter(
            torch.randn(topic_dim, input_dim) * (1.0 / input_dim ** 0.5)
        )

        # (c) learnable topic-topic graph (unconstrained — symmetrised later)
        self.A_raw = nn.Parameter(torch.zeros(topic_dim, topic_dim))

        # (b) learnable base temperature (log-scale for positivity)
        self.log_T = nn.Parameter(torch.tensor(0.0))          # T_init = 1.0

        # (d) Lorentzian curvature c (log-scale for positivity)
        self.log_c = nn.Parameter(torch.tensor(0.0))          # c = 1.0

        # (a) momentum buffer (persistent running logit mean, NOT a parameter)
        self.register_buffer("b_running", torch.zeros(topic_dim))

    # ── (c) normalised diffusion matrix ─────────────────────────────────
    def _diffusion_matrix(self) -> torch.Tensor:
        A_sym = (self.A_raw + self.A_raw.T) * 0.5             # symmetric
        return F.softmax(A_sym, dim=-1)                        # row-stochastic

    # ── (d) Lorentzian squash ───────────────────────────────────────────
    def _lorentzian_squash(self, s: torch.Tensor) -> torch.Tensor:
        """
        v = (c·||s||²) / (1 + c·||s||²)  ·  s / ||s||
        ∈ [0, 1) for all c > 0  (light-cone constraint).
        """
        c       = torch.exp(self.log_c).clamp(min=0.01, max=10.0)
        norm_sq = (s * s).sum(dim=-1, keepdim=True).clamp(min=1e-10)
        norm    = norm_sq.sqrt()
        scale   = (c * norm_sq) / (1.0 + c * norm_sq)
        return scale / norm * s

    # ── forward ─────────────────────────────────────────────────────────
    def forward(self, x: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        """
        x           : [B, input_dim]  – hyperbolic input from SGP-E
        temperature : external annealing multiplier (from training schedule)

        Returns
        -------
        theta : [B, K]  topic mixture (row-normalised)
        """
        B = x.size(0)

        # (b) effective temperature
        T_eff = (torch.exp(self.log_T) * temperature).clamp(min=0.05, max=10.0)

        # votes
        u_hat = x @ self.W.T                                   # [B, K]

        # (a) initialise routing logits from momentum buffer
        b = self.b_running.unsqueeze(0).expand(B, -1).clone()  # [B, K]

        A_diff = self._diffusion_matrix()                      # [K, K]

        # routing iterations
        for _ in range(self.routing_iters):
            c_coef    = F.softmax(b / T_eff, dim=1)            # [B, K]
            v         = self._lorentzian_squash(u_hat)         # [B, K]
            agreement = (u_hat * v).sum(dim=-1, keepdim=True)  # [B, 1]
            b = b + agreement * c_coef                         # [B, K]
            # (c) graph diffusion: spread signal to neighbouring topics
            b = b + 0.10 * (b @ A_diff.T)                      # [B, K]

        # (a) update momentum buffer (no gradient)
        with torch.no_grad():
            batch_mean     = b.detach().mean(dim=0)
            self.b_running = (self.momentum * self.b_running
                              + (1.0 - self.momentum) * batch_mean)

        # final tempered softmax
        theta = F.softmax(b / T_eff, dim=1)                    # [B, K]
        return theta
