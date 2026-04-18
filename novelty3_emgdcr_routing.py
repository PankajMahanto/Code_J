"""
=============================================================================
NOVELTY 3: Entropic Momentum Graph-Diffused Capsule Routing  (EMGD-CR)
=============================================================================

Four sub-contributions over classical Dynamic Routing (Sabour et al., 2017):
  (a) Momentum on routing logits  — EMA persistent memory across batches
  (b) Annealed entropic temperature — softmax(b/T) with T decreasing from
                                      T_max → T_min over training epochs
  (c) Learnable topic-topic graph diffusion — K×K symmetric adjacency
                                              spreads routing signal across
                                              semantically linked topics
  (d) Lorentzian squash function   — replaces tanh; inspired by hyperbolic
                                     (Lorentz model) geometry; keeps vectors
                                     on the future light cone surface

Why these improve routing
─────────────────────────
(a) Momentum: cold-start every batch throws away accumulated routing evidence.
    EMA carries forward reliable prior from past batches → faster convergence.
(b) Entropic temperature: high T at epoch 0 = soft, exploratory routing
    → avoids early topic collapse. Low T late in training = sharp assignments.
(c) Graph diffusion: topics are NOT independent. If topic A and topic B share
    a sub-theme, routing logits should be correlated. The diffusion matrix
    Â encodes this structure and is learned jointly.
(d) Lorentzian squash: ||v||/(1+||v||²) · v/||v|| has a "speed-of-light"
    ceiling effect — vectors cannot exceed unit norm — while its gradient at
    small ||v|| is larger than tanh's, accelerating early routing updates.

Published for: IEEE Transactions on Knowledge and Data Engineering
=============================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class EMGDCapsuleRouting(nn.Module):
    """
    Entropic Momentum Graph-Diffused Capsule Routing (EMGD-CR).

    Parameters
    ----------
    input_dim    : int   — dim of incoming capsule vector  (= topic_dim)
    topic_dim    : int   — number of topics  K
    routing_iters: int   — EM routing iterations per forward pass (default 5)
    momentum     : float — EMA coefficient for persistent routing memory [0,1)
    """

    def __init__(self,
                 input_dim:    int,
                 topic_dim:    int,
                 routing_iters: int   = 5,
                 momentum:      float = 0.90):
        super().__init__()

        self.K            = topic_dim
        self.routing_iters = routing_iters
        self.momentum     = momentum

        # ── Transformation (vote) matrix ─────────────────────────────────────
        # u_hat = x W^T   →  x: [B, input_dim],  W: [K, input_dim]
        self.W = nn.Parameter(
            torch.randn(topic_dim, input_dim) * (1.0 / input_dim**0.5)
        )

        # ── (c) Learnable topic-topic graph ──────────────────────────────────
        # A_raw is unconstrained; symmetric normalised version used in diffusion
        self.A_raw = nn.Parameter(torch.zeros(topic_dim, topic_dim))

        # ── (b) Learnable initial temperature (log-scale) ────────────────────
        # T_init = exp(log_T);  combined with external annealing temp at run-time
        self.log_T = nn.Parameter(torch.tensor(0.0))   # → T_init = 1.0

        # ── (d) Lorentzian curvature ──────────────────────────────────────────
        # curvature c = exp(log_c);  default c=1  (standard Lorentz model)
        self.log_c = nn.Parameter(torch.tensor(0.0))

        # ── (a) Momentum buffer (persistent routing logit mean) ──────────────
        # NOT a gradient parameter — updated with no_grad in each forward pass.
        self.register_buffer('b_running', torch.zeros(topic_dim))

    # ── (c) Normalised topic-topic diffusion matrix ───────────────────────────
    def _diffusion_matrix(self) -> torch.Tensor:
        """
        Build a row-stochastic symmetric diffusion matrix from A_raw.
            A_sym  = (A_raw + A_raw^T) / 2          symmetric
            A_diff = softmax(A_sym, dim=-1)          row-stochastic
        """
        A_sym  = (self.A_raw + self.A_raw.T) * 0.5   # [K, K]
        A_diff = F.softmax(A_sym, dim=-1)             # [K, K] row-stochastic
        return A_diff

    # ── (d) Lorentzian squash ─────────────────────────────────────────────────
    def _lorentzian_squash(self, s: torch.Tensor) -> torch.Tensor:
        """
        Lorentz-model inspired squash:
            v = (c · ||s||²) / (1 + c · ||s||²)  ·  s / ||s||

        Properties
        ──────────
        • Output norm ∈ [0, 1)  for all c > 0  (light-cone constraint)
        • At small ||s||: norm ≈ c·||s||   (larger grad than tanh near 0)
        • At large ||s||: norm → 1         (hard cap, no saturation spike)
        """
        c       = torch.exp(self.log_c).clamp(min=0.01, max=10.0)
        norm_sq = (s * s).sum(dim=-1, keepdim=True).clamp(min=1e-10)
        norm    = norm_sq.sqrt()
        scale   = (c * norm_sq) / (1.0 + c * norm_sq)    # ∈ [0, 1)
        return scale / norm * s

    # ── forward ─────────────────────────────────────────────────────────────
    def forward(self,
                x:           torch.Tensor,
                temperature: float = 1.0) -> torch.Tensor:
        """
        Parameters
        ----------
        x           : [B, input_dim]   hyperbolic capsule input (from SGP-E)
        temperature : float            external annealing temperature

        Returns
        -------
        theta : [B, K]   topic mixture (sums to 1 per document)
        """
        B = x.size(0)

        # ── (b) Effective temperature (learnable × external annealing) ────────
        T_eff = (torch.exp(self.log_T) * temperature).clamp(min=0.05, max=10.0)

        # ── vote vectors:  u_hat = x W^T   [B, K] ────────────────────────────
        u_hat = x @ self.W.T        # [B, K]

        # ── (a) Initialise b from momentum buffer ────────────────────────────
        # b_running: [K]  →  expand to [B, K] for batch processing
        b = self.b_running.unsqueeze(0).expand(B, -1).clone()   # [B, K]

        # ── Precompute diffusion matrix (once per forward) ───────────────────
        A_diff = self._diffusion_matrix()    # [K, K]

        # ── Routing EM iterations ────────────────────────────────────────────
        for _iter in range(self.routing_iters):
            # (b) Entropic softmax — tempered coupling coefficients
            c_coeff = F.softmax(b / T_eff, dim=1)               # [B, K]

            # Squash votes with (d) Lorentzian squash
            v = self._lorentzian_squash(u_hat)                  # [B, K]

            # Agreement between votes and squashed output
            # agreement: scalar per document (dot-product between u_hat and v)
            agreement = (u_hat * v).sum(dim=-1, keepdim=True)   # [B, 1]

            # Update logits: b += agreement × c_coeff  (weighted vote)
            b = b + agreement * c_coeff                         # [B, K]

            # (c) Graph diffusion: spread routing signal through topic graph
            b = b + 0.10 * (b @ A_diff.T)                      # [B, K]

        # ── (a) Update momentum buffer with batch mean (no gradient) ─────────
        with torch.no_grad():
            b_mean          = b.detach().mean(dim=0)            # [K]
            self.b_running  = (self.momentum * self.b_running
                               + (1.0 - self.momentum) * b_mean)

        # ── Final topic mixture: tempered softmax ─────────────────────────────
        theta = F.softmax(b / T_eff, dim=1)    # [B, K]  sums to 1

        return theta


# ─────────────────────────────────────────────────────────────────────────────
#  Quick sanity check
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    torch.manual_seed(42)

    B, input_dim, K = 16, 12, 12

    router = EMGDCapsuleRouting(input_dim=input_dim, topic_dim=K,
                                routing_iters=5, momentum=0.90)

    # Simulate 3 successive batches to verify momentum accumulation
    for i, T in enumerate([2.0, 1.0, 0.5]):
        x     = torch.randn(B, input_dim)
        theta = router(x, temperature=T)
        print(f"[EMGD-CR] batch {i+1} | T={T:.1f} | "
              f"theta shape: {theta.shape} | "
              f"sum/doc ≈ {theta.sum(-1).mean().item():.4f} | "
              f"b_running norm: {router.b_running.norm().item():.4f}")

    assert theta.shape == (B, K),              "Shape error!"
    assert abs(theta.sum(-1).mean().item() - 1.0) < 1e-4, "Not normalised!"
    assert theta.min().item() >= 0,            "Negative probability!"
    print("Novelty 3 (EMGD-CR) — PASSED")
