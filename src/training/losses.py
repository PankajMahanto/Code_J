"""
============================================================================
 src/training/losses.py  [REWRITE - v4]
 ---------------------------------------------------------------------------
 Loss functions redesigned so GRADIENTS ACTUALLY FLOW.

 v4 ADDS
 -------
 * `orthogonal_regularization(beta)` - cosine-similarity off-diagonal
   penalty on the decoder's topic-word matrix.  Mandatory contributor
   to the Topic Diversity / Inter-cosine targets.

 v3 highlights (retained)
 ------------------------
 * Soft PMI coherence (quadratic form over full beta).
 * Direct multinomial NLL on (theta @ beta).
 * Symmetric-KL diversity + cosine redundancy.
============================================================================
"""
from __future__ import annotations
import torch
import torch.nn.functional as F


# =============================================================================
# Reconstruction - direct multinomial NLL on a valid probability
# =============================================================================
def reconstruction_loss(recon_probs: torch.Tensor,
                        x_bow:       torch.Tensor) -> torch.Tensor:
    """
    recon_probs : [B, V]  - already a valid probability (theta @ beta).
    x_bow       : [B, V]  - raw word counts.

    L = - sum_v  x_bow[b,v] * log(recon_probs[b,v])   (per doc, then mean)
    """
    log_p = torch.log(recon_probs.clamp(min=1e-10))
    return -(x_bow * log_p).sum(dim=-1).mean()


# =============================================================================
# Coherence - FULLY DIFFERENTIABLE soft PMI surrogate
# =============================================================================
def coherence_loss(beta:       torch.Tensor,
                   pmi_matrix: torch.Tensor) -> torch.Tensor:
    """
    Soft PMI coherence - differentiable with full gradient flow.

        coh_k = beta_k^T . PMI . beta_k

    Returns -mean(coh_k) so minimisation maximises coherence.
    """
    P = pmi_matrix.clamp(min=-3.0, max=5.0)
    Bp = beta @ P                            # [K, V]
    coh_per_topic = (Bp * beta).sum(dim=-1)  # [K]
    return -coh_per_topic.mean()


# =============================================================================
# Diversity - symmetric-KL between topic pairs
# =============================================================================
def diversity_loss(beta: torch.Tensor) -> torch.Tensor:
    """
    Differentiable diversity penalty - encourages different topics to have
    different word distributions.  NEGATED so that minimising drives
    topics apart.
    """
    K = beta.shape[0]
    eps = 1e-10
    log_beta = torch.log(beta + eps)         # [K, V]

    entropy_i = (beta * log_beta).sum(dim=-1, keepdim=True)   # [K, 1]
    cross_ij  = beta @ log_beta.T                              # [K, K]
    kl_ij     = entropy_i - cross_ij                           # [K, K]

    sym_kl = 0.5 * (kl_ij + kl_ij.T)
    mask   = 1.0 - torch.eye(K, device=beta.device)

    mean_pairwise_kl = (sym_kl * mask).sum() / (K * (K - 1))
    return -mean_pairwise_kl


# =============================================================================
# Redundancy - cosine similarity between topic pairs
# =============================================================================
def redundancy_loss(beta: torch.Tensor) -> torch.Tensor:
    """Cosine similarity penalty (complements diversity loss)."""
    K = beta.shape[0]
    Bn = F.normalize(beta + 1e-12, dim=-1)
    S  = Bn @ Bn.T
    iu = torch.triu_indices(K, K, offset=1)
    return S[iu[0], iu[1]].mean()


# =============================================================================
# Orthogonal regularization  (NEW in v4)
# =============================================================================
def orthogonal_regularization(weight: torch.Tensor,
                              mode: str = "cosine") -> torch.Tensor:
    """
    Push the rows of a topic weight matrix towards mutual orthogonality.

    * weight : [K, D]  - decoder topic matrix (either the raw parameter or
      the post-softmax topic-word distribution beta).
    * mode   : 'cosine' (default) penalises off-diagonal cosine similarity.
               'gram'   penalises || W W^T - I ||_F^2  (Brock et al. 2016).

    The penalty is ZERO when rows are orthonormal and grows as rows collapse
    - directly enforcing Inter-cosine <= 0.30 and Topic Diversity >= 0.95.
    """
    K = weight.shape[0]
    if K < 2:
        return weight.new_zeros(())

    if mode == "gram":
        W = F.normalize(weight + 1e-12, dim=-1)
        gram = W @ W.T
        target = torch.eye(K, device=weight.device, dtype=weight.dtype)
        return ((gram - target) ** 2).mean()

    # default: cosine off-diagonal
    W   = F.normalize(weight + 1e-12, dim=-1)
    sim = W @ W.T                                  # [K, K]
    off = sim - torch.eye(K, device=weight.device, dtype=weight.dtype)
    # square keeps everything non-negative and penalises large |cos|
    return (off ** 2).sum() / (K * (K - 1))
