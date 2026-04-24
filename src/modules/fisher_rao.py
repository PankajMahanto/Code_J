"""
============================================================================
 src/modules/fisher_rao.py  [PATCHED — v2]
 ---------------------------------------------------------------------------
 Fisher–Rao precision-weighted KL divergence.

 PATCH NOTES (v2 — fixes NaN at epoch ~5-10)
 ───────────────────────────────────────────
 Original bug:
     precision = 1.0 / (sigma_sq + 1e-8)
     When sigma_sq → 0 (posterior collapse), precision → 10^8+ and the
     product with kl_per_dim explodes into NaN.

 Fix:
   1. Clamp precision to a max value (default 10.0).
   2. Tighter logvar clamp (min=-8 instead of -10).
   3. Down-weight Fisher-Rao term to 0.05 (was 0.10) — more conservative.
   4. NaN/Inf fallback — return standard KL if something slipped through.
============================================================================
"""
from __future__ import annotations
import torch


def fisher_rao_kl(mu: torch.Tensor,
                  logvar: torch.Tensor,
                  lam: float = 0.05,
                  precision_max: float = 10.0) -> torch.Tensor:
    """
    Natural-gradient KL divergence for a diagonal Gaussian prior.

    Parameters
    ----------
    mu            : [B, K]  posterior mean
    logvar        : [B, K]  posterior log-variance (should be pre-clamped)
    lam           : float   mixing coefficient λ ∈ [0, 1]
    precision_max : float   upper clamp on precision to prevent explosion
    """
    # Extra safety clamp
    logvar   = logvar.clamp(min=-8.0, max=4.0)
    sigma_sq = logvar.exp()

    # Precision is stop-grad; now CLAMPED to prevent explosion
    precision = (1.0 / (sigma_sq.detach() + 1e-6)).clamp(max=precision_max)

    # per-dimension KL
    kl_per_dim = -0.5 * (1.0 + logvar - mu.pow(2) - sigma_sq)

    kl_standard = kl_per_dim.sum(dim=-1).mean()
    kl_fisher   = (precision * kl_per_dim).sum(dim=-1).mean()

    out = kl_standard + lam * kl_fisher

    # NaN/Inf fallback
    if not torch.isfinite(out):
        return kl_standard
    return out
