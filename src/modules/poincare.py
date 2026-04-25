"""
============================================================================
 src/modules/poincare.py
 ---------------------------------------------------------------------------
 Riemannian operations on the Poincaré ball model of hyperbolic space
 with curvature −c.

 Reference geometry:  all points live inside the open ball  ||x|| < 1/√c.

 Used by
 -------
 • SGP-E encoder (Novelty 1)      – lifts Euclidean latent z to hyperbolic z_H
 • (Optional)   Lorentzian squash – in EMGD-CR (already self-contained there)
============================================================================
"""
from __future__ import annotations
import torch


class PoincareBall:
    """
    Poincaré ball with curvature −c (c > 0).  Only origin-centred operations
    are needed for our encoder, so we implement:

        • proj     – project any point back inside the ball
        • expmap0  – tangent space (origin) → manifold
        • logmap0  – manifold            → tangent space (origin)
        • mobius_add, dist – generic operations (used nowhere critical yet
                              but handy for ablation / debugging)
    """

    def __init__(self, c: float = 1.0):
        self.c = float(c)

    # ── projection (numerical safety) ───────────────────────────────────
    def proj(self, x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
        max_norm = 1.0 / (self.c ** 0.5) - eps
        norm = x.norm(dim=-1, keepdim=True).clamp(min=1e-15)
        return torch.where(norm >= max_norm, x / norm * max_norm, x)

    # ── exp / log maps at origin ────────────────────────────────────────
    def expmap0(self, v: torch.Tensor, min_norm: float = 1e-15) -> torch.Tensor:
        v_norm  = v.norm(dim=-1, keepdim=True).clamp(min=min_norm)
        c_sqrt  = self.c ** 0.5
        tanh_in = (c_sqrt * v_norm).clamp(max=15.0)
        return torch.tanh(tanh_in) / (c_sqrt * v_norm) * v

    def logmap0(self, x: torch.Tensor, min_norm: float = 1e-15) -> torch.Tensor:
        x_norm   = x.norm(dim=-1, keepdim=True).clamp(min=min_norm)
        c_sqrt   = self.c ** 0.5
        atanh_in = (c_sqrt * x_norm).clamp(max=1.0 - 1e-6)
        return torch.atanh(atanh_in) / (c_sqrt * x_norm) * x

    # ── Möbius addition  ⊕_c ────────────────────────────────────────────
    def mobius_add(self, x: torch.Tensor, y: torch.Tensor,
                   eps: float = 1e-15) -> torch.Tensor:
        c  = self.c
        x2 = (x * x).sum(-1, keepdim=True)
        y2 = (y * y).sum(-1, keepdim=True)
        xy = (x * y).sum(-1, keepdim=True)
        num   = (1 + 2 * c * xy + c * y2) * x + (1 - c * x2) * y
        denom = 1 + 2 * c * xy + (c ** 2) * x2 * y2
        return num / (denom + eps)

    # ── hyperbolic (geodesic) distance ──────────────────────────────────
    def dist(self, x: torch.Tensor, y: torch.Tensor,
             min_norm: float = 1e-15) -> torch.Tensor:
        diff      = self.mobius_add(-x, y)
        diff_norm = diff.norm(dim=-1).clamp(min=min_norm)
        c_sqrt    = self.c ** 0.5
        atanh_in  = (c_sqrt * diff_norm).clamp(max=1.0 - 1e-6)
        return 2.0 / c_sqrt * torch.atanh(atanh_in)
