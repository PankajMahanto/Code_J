"""
============================================================================
 src/training/losses.py  [REWRITE — v3]
 ---------------------------------------------------------------------------
 Loss functions redesigned so GRADIENTS ACTUALLY FLOW.

 WHY v2 FAILED
 ─────────────
 1. `coherence_loss` used `beta.topk()` for index selection → topk is
    NON-DIFFERENTIABLE w.r.t. β's values.  Only the selected positions got
    gradient; everything else was a dead weight.
 2. `reconstruction_loss` applied log_softmax to (θ·β), but (θ·β) is
    already a probability distribution → double normalization destroyed
    gradient magnitude.

 v3 FIX
 ──────
 1. Coherence is now a SOFT weighted sum over ALL words:
      coh_k = sum_{i,j} β[k,i] · β[k,j] · PMI[i,j]
    This is fully differentiable w.r.t. β and scales quadratically with
    the top mass — topic's top words get sharpened automatically.
 2. Reconstruction uses log(θ·β + eps) directly — no double softmax.
 3. ``orthogonal_loss`` penalises off-diagonals of the topic-topic cosine
    similarity matrix to enforce Inter-cosine ≤ 0.30.
============================================================================
"""
from __future__ import annotations
import torch
import torch.nn.functional as F


# =============================================================================
# Reconstruction — direct multinomial NLL on a valid probability
# =============================================================================
def reconstruction_loss(recon_probs: torch.Tensor,
                        x_bow:       torch.Tensor) -> torch.Tensor:
    """
    recon_probs : [B, V]  — already a valid probability (θ · β).
    x_bow       : [B, V]  — raw word counts.

    L = - sum_v  x_bow[b,v] · log(recon_probs[b,v])   (per doc, then mean)
    """
    log_p = torch.log(recon_probs.clamp(min=1e-10))
    return -(x_bow * log_p).sum(dim=-1).mean()


# =============================================================================
# Coherence — FULLY DIFFERENTIABLE soft PMI surrogate
# =============================================================================
def coherence_loss(beta:       torch.Tensor,
                   pmi_matrix: torch.Tensor) -> torch.Tensor:
    """
    Soft PMI coherence — differentiable with full gradient flow.

    For each topic k:
        coh_k = β_k^T · PMI · β_k       (quadratic form)
              = sum_{i, j}  β[k, i] · β[k, j] · PMI[i, j]

    Why this works:
      - If β puts mass on co-occurring words (high PMI[i,j]), coh_k grows.
      - Gradient flows to EVERY β[k, v] position, not just top-N.
      - Topic sharpening happens naturally because β concentrates on the
        high-PMI cluster.

    Returns  −mean(coh_k)  so minimisation maximises coherence.
    """
    # Clamp PMI to a safe range
    P = pmi_matrix.clamp(min=-3.0, max=5.0)

    # coh_k = β_k · P · β_k for each topic k
    # Compute in two matmuls for stability
    Bp = beta @ P                            # [K, V]
    coh_per_topic = (Bp * beta).sum(dim=-1)  # [K]

    return -coh_per_topic.mean()


# =============================================================================
# Diversity — SOFT topic-word entropy + KL between topic pairs
# =============================================================================
def diversity_loss(beta: torch.Tensor) -> torch.Tensor:
    """
    Differentiable diversity penalty — encourages different topics to have
    different word distributions.

    Computed as the SYMMETRIC KL divergence between topic pairs (averaged),
    NEGATED so that minimising drives topics apart.
    """
    K = beta.shape[0]
    eps = 1e-10
    log_beta = torch.log(beta + eps)         # [K, V]

    # KL(i || j) = Σ β_i log(β_i / β_j) = Σ β_i (log β_i - log β_j)
    # For all pairs: kl_matrix[i, j] = sum_v β[i,v] * (log β[i,v] - log β[j,v])
    entropy_i = (beta * log_beta).sum(dim=-1, keepdim=True)   # [K, 1]
    cross_ij  = beta @ log_beta.T                              # [K, K]
    kl_ij     = entropy_i - cross_ij                           # [K, K]

    # Symmetric KL, off-diagonal
    sym_kl = 0.5 * (kl_ij + kl_ij.T)
    mask   = 1.0 - torch.eye(K, device=beta.device)

    # More KL between pairs = more diverse = LOWER diversity_loss is better
    # So return -mean(sym_kl) — minimising increases diversity
    mean_pairwise_kl = (sym_kl * mask).sum() / (K * (K - 1))
    return -mean_pairwise_kl


# =============================================================================
# Redundancy — cosine similarity between topic pairs
# =============================================================================
def redundancy_loss(beta: torch.Tensor) -> torch.Tensor:
    """Cosine similarity penalty (complements diversity loss)."""
    K = beta.shape[0]
    Bn = F.normalize(beta + 1e-12, dim=-1)
    S  = Bn @ Bn.T
    iu = torch.triu_indices(K, K, offset=1)
    return S[iu[0], iu[1]].mean()   # no abs — already non-negative


# =============================================================================
# Orthogonal regularization — enforce decoder topic-word orthogonality
# =============================================================================
def orthogonal_loss(beta: torch.Tensor) -> torch.Tensor:
    """
    Orthogonal regularization on the decoder topic-word matrix.

    Computes the cosine-similarity matrix S = β_n β_n^T (rows L2-normalised)
    and penalises the off-diagonal elements with a Frobenius norm.  Driving
    this term down forces topic rows toward mutual orthogonality, which is
    the mechanism for hitting Inter-cosine ≤ 0.30 while keeping Intra-cosine
    high.

      L_ortho = || S - I ||_F^2 / (K * (K - 1))

    Returns a scalar; minimising it pushes topics apart.
    """
    K = beta.shape[0]
    Bn = F.normalize(beta + 1e-12, dim=-1)
    S = Bn @ Bn.T                                          # [K, K]
    I = torch.eye(K, device=beta.device, dtype=beta.dtype)
    off = (S - I).pow(2)
    return off.sum() / max(1, K * (K - 1))
