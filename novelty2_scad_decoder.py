"""
=============================================================================
NOVELTY 2: Sinkhorn Concept-Anchor Decoder (SCAD)
=============================================================================

Three sub-contributions:
  (a) Learnable concept anchors  — one anchor per topic in embedding space
  (b) Mahalanobis cost matrix    — low-rank PD metric (M = I + UUᵀ)
  (c) Sinkhorn OT in log-space   — entropy-regularised optimal transport
                                   produces spread-out, coherent topic-word
                                   distributions instead of peaked softmax.

Why it improves coherence
─────────────────────────
Standard decoder:  β[k,v] = softmax(t_k · w_v)
  → winner-takes-all: one or two words dominate each topic.

SCAD:              β = Sinkhorn(−C/ε)  where  C[k,v] = Mahal(a_k, w_v)
  → OT pushes mass across ALL vocab positions proportionally to
    semantic distance → richer, more coherent top-K words per topic.
  → Mahalanobis metric is learned end-to-end → warps the word space
    to minimise coherence loss.

Published for: IEEE Transactions on Knowledge and Data Engineering
=============================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SCADecoder(nn.Module):
    """
    Sinkhorn Concept-Anchor Decoder (SCAD).

    Parameters
    ----------
    n_topics    : int   — number of topics K
    vocab_size  : int   — vocabulary size V
    embed_dim   : int   — word embedding dimension d
    word_embeds : Tensor [V, d]  — pretrained GloVe (frozen)
    rank        : int   — rank r of low-rank Mahalanobis factor U  (r = d//4)
    sinkhorn_n  : int   — number of Sinkhorn iterations (default 20)
    """

    def __init__(self,
                 n_topics:    int,
                 vocab_size:  int,
                 embed_dim:   int,
                 word_embeds: torch.Tensor,
                 rank:        int  = None,
                 sinkhorn_n:  int  = 20):
        super().__init__()

        self.K        = n_topics
        self.V        = vocab_size
        self.d        = embed_dim
        self.S_iters  = sinkhorn_n
        rank = rank or max(1, embed_dim // 4)

        # ── (a) Concept anchors ─────────────────────────────────────────────
        # Initialised near the origin; learned via back-prop.
        # Each anchor a_k ∈ ℝ^d represents the geometric centroid of topic k.
        self.concept_anchors = nn.Parameter(
            torch.randn(n_topics, embed_dim) * 0.02
        )

        # ── (b) Low-rank Mahalanobis factor ─────────────────────────────────
        # M = I + U Uᵀ   →  always positive-definite.
        # rank r ≪ d keeps the parameter count manageable.
        self.M_U = nn.Parameter(
            torch.randn(embed_dim, rank) * (1.0 / embed_dim**0.5)
        )

        # ── (c) Sinkhorn regularisation temperature (log-parameterised) ─────
        # ε = softplus(log_eps) + 1e-4; initialised ≈ 0.10
        self.log_eps = nn.Parameter(torch.tensor(-2.3))   # softplus(-2.3)≈0.10

        # ── Topic-specific output scale (learnable) ──────────────────────────
        self.topic_scale = nn.Parameter(torch.ones(n_topics, 1))

        # ── Word embedding projection (small adapter) ───────────────────────
        self.word_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        nn.init.eye_(self.word_proj.weight)   # identity init (no-op initially)

        # Store frozen word embeddings as buffer
        self.register_buffer('word_embeds', word_embeds.clone())

    # ── (b) Mahalanobis cost matrix ──────────────────────────────────────────
    def _mahalanobis_cost(self,
                          A: torch.Tensor,   # [K, d]  concept anchors (normed)
                          B: torch.Tensor    # [V, d]  word embeddings (projected)
                          ) -> torch.Tensor:
        """
        C[k, v] = (a_k − b_v)ᵀ M (a_k − b_v)
        where  M = I + U Uᵀ   (PD by construction).

        Efficient expansion:
            C = diag(AMA') 1ᵀ  +  1 diag(BMB')ᵀ  −  2 AMBᵀ
        """
        # M = I + U Uᵀ
        M  = torch.eye(self.d, device=A.device) + self.M_U @ self.M_U.T  # [d, d]

        MA = A @ M           # [K, d]
        MB = B @ M           # [V, d]

        AA = (MA * A).sum(-1)              # [K]
        BB = (MB * B).sum(-1)              # [V]
        AB = MA @ B.T                      # [K, V]

        cost = AA.unsqueeze(1) + BB.unsqueeze(0) - 2.0 * AB   # [K, V]
        return F.relu(cost)   # numerical safety: no negative distances

    # ── (c) Sinkhorn iterations (log-space, numerically stable) ─────────────
    def _sinkhorn_log(self, log_K: torch.Tensor) -> torch.Tensor:
        """
        Sinkhorn-Knopp in log-space.

        Input
            log_K : [K, V]  log of Gibbs kernel  −C/ε
        Output
            log_P : [K, V]  log of doubly-stochastic transport plan

        Update rules (Sinkhorn in log-domain):
            log_u ← −LSE_v(log_K + log_v)
            log_v ← −LSE_k(log_K + log_u)
        """
        log_u = torch.zeros(self.K, 1,    device=log_K.device)
        log_v = torch.zeros(1,    self.V, device=log_K.device)

        for _ in range(self.S_iters):
            log_u = -torch.logsumexp(log_K + log_v, dim=1, keepdim=True)
            log_v = -torch.logsumexp(log_K + log_u, dim=0, keepdim=True)

        return log_K + log_u + log_v   # [K, V] log-transport plan

    # ── forward ─────────────────────────────────────────────────────────────
    def forward(self) -> torch.Tensor:
        """
        Compute topic-word distributions β ∈ [K, V].

        Returns
        -------
        beta : [K, V]   row-normalised (each topic sums to 1 over vocab)
        """
        # ── project & normalise word embeddings ─────────────────────────────
        w = F.normalize(self.word_proj(self.word_embeds), dim=-1)   # [V, d]

        # ── normalise concept anchors ────────────────────────────────────────
        a = F.normalize(self.concept_anchors, dim=-1)               # [K, d]

        # ── (b) Mahalanobis cost ─────────────────────────────────────────────
        cost = self._mahalanobis_cost(a, w)                         # [K, V]

        # ── (c) Sinkhorn OT ──────────────────────────────────────────────────
        eps      = F.softplus(self.log_eps) + 1e-4                  # scalar > 0
        log_kern = -cost / eps                                       # [K, V]

        log_P    = self._sinkhorn_log(log_kern)                     # [K, V]

        # row-normalise: each topic β[k] sums to 1 over the vocab
        log_beta = log_P - torch.logsumexp(log_P, dim=1, keepdim=True)

        beta = torch.exp(log_beta)                                  # [K, V]

        # ── topic-specific scaling (sharpens or smooths per-topic) ──────────
        scale = F.softplus(self.topic_scale)                        # [K, 1]
        beta  = beta * scale
        beta  = beta / beta.sum(dim=1, keepdim=True)               # re-normalise

        return beta   # [K, V]


# ─────────────────────────────────────────────────────────────────────────────
#  Quick sanity check
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    torch.manual_seed(0)

    K, V, d  = 12, 300, 100
    we       = torch.randn(V, d) * 0.1          # dummy GloVe

    dec = SCADecoder(n_topics=K, vocab_size=V, embed_dim=d, word_embeds=we)
    beta = dec()

    print("[SCAD] beta shape     :", beta.shape)
    print("[SCAD] row sums (≈1)  :", beta.sum(dim=1).mean().item())
    print("[SCAD] min val        :", beta.min().item())
    print("[SCAD] entropy/topic  :",
          -(beta * (beta + 1e-20).log()).sum(dim=1).mean().item())
    assert beta.shape == (K, V),            "Shape error!"
    assert beta.min().item() >= 0,          "Negative probability!"
    assert abs(beta.sum(dim=1).mean().item() - 1.0) < 1e-4, "Not normalised!"
    print("Novelty 2 (SCAD) — PASSED")
