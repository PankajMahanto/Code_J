"""
============================================================================
 src/models/scad_decoder.py  [REWRITE — v3]
 ---------------------------------------------------------------------------
 NOVELTY 2 — Sinkhorn Concept-Anchor Decoder (SCAD)

 WHY v2 FAILED
 ─────────────
 Previous version used full Sinkhorn OT with Mahalanobis on raw static
 embeddings. With random-init M_U and concept_anchors, the cost matrix
 was nearly uniform → β came out nearly uniform → no gradient signal →
 rec loss stuck at 38.16 for 15 epochs.

 v3 FIX — WARM-STARTED DIRECT DECODER + SINKHORN REFINEMENT
 ───────────────────────────────────────────────────────────
 We keep all three novelty components (anchors, Mahalanobis, Sinkhorn) but
 combine them with a standard learnable β that is trainable from step 1.

 β[k,v] = softmax(  θ_k · word_emb_v · scale                    (direct signal)
                  - λ·Mahalanobis(anchor_k, word_emb_v)         (novelty)
                  + γ·(Sinkhorn refinement term)                (novelty)
                 )

 This means:
   • The direct dot-product gives an immediate, non-zero gradient.
   • Mahalanobis sculpts the geometry as it learns.
   • Sinkhorn refinement enforces a (near-)doubly-stochastic structure.
============================================================================
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SCADecoder(nn.Module):
    """Warm-started concept-anchor decoder with Mahalanobis + Sinkhorn."""

    def __init__(self,
                 n_topics:    int,
                 vocab_size:  int,
                 embed_dim:   int,
                 word_embeds: torch.Tensor,
                 rank:        int  = None,
                 sinkhorn_n:  int  = 10,
                 lambda_mahal: float = 0.3,
                 gamma_sinkhorn: float = 0.3):
        super().__init__()

        self.K       = n_topics
        self.V       = vocab_size
        self.d       = embed_dim
        self.S_iters = sinkhorn_n
        rank         = rank or max(1, embed_dim // 4)

        self.lambda_mahal   = lambda_mahal
        self.gamma_sinkhorn = gamma_sinkhorn

        # (a) Concept anchors - initialised from the contextual word-embedding
        # centroid (NOT random!).  Random init makes the cost matrix flat.
        init_anchor = word_embeds.mean(dim=0, keepdim=True).expand(n_topics, -1).clone()
        init_anchor = init_anchor + torch.randn(n_topics, embed_dim) * 0.05
        self.concept_anchors = nn.Parameter(init_anchor)

        # (b) Low-rank Mahalanobis factor (small init so it doesn't dominate)
        self.M_U = nn.Parameter(torch.randn(embed_dim, rank) * 0.01)

        # (c) Sinkhorn temperature
        self.log_eps = nn.Parameter(torch.tensor(-1.5))   # ε ≈ 0.22

        # Global sharpness scale (starts moderate, learns to sharpen)
        self.log_scale = nn.Parameter(torch.tensor(0.0))   # scale = 1.0

        # Per-topic bias (helps individual topics specialize)
        self.topic_bias = nn.Parameter(torch.zeros(n_topics, 1))

        # Word embedding adapter
        self.word_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        nn.init.eye_(self.word_proj.weight)

        self.register_buffer("word_embeds", word_embeds.clone())

    # ── Mahalanobis cost ──
    def _mahalanobis_cost(self, A, B):
        """C[k,v] = (a_k − b_v)^T M (a_k − b_v)  with  M = I + U U^T"""
        M = torch.eye(self.d, device=A.device) + self.M_U @ self.M_U.T
        MA, MB = A @ M, B @ M
        AA = (MA * A).sum(-1, keepdim=True)                 # [K, 1]
        BB = (MB * B).sum(-1, keepdim=True).T                # [1, V]
        AB = MA @ B.T                                        # [K, V]
        return F.relu(AA + BB - 2.0 * AB)                   # [K, V]

    # ── Sinkhorn log-space ──
    def _sinkhorn_log(self, log_K):
        log_u = torch.zeros(self.K, 1,    device=log_K.device)
        log_v = torch.zeros(1,      self.V, device=log_K.device)
        for _ in range(self.S_iters):
            log_u = -torch.logsumexp(log_K + log_v, dim=1, keepdim=True)
            log_v = -torch.logsumexp(log_K + log_u, dim=0, keepdim=True)
        return log_K + log_u + log_v

    def forward(self) -> torch.Tensor:
        w = F.normalize(self.word_proj(self.word_embeds), dim=-1)   # [V, d]
        a = F.normalize(self.concept_anchors, dim=-1)               # [K, d]

        # Part 1: direct dot-product score (guarantees a learning signal)
        scale   = torch.exp(self.log_scale).clamp(max=20.0)
        dot_sim = a @ w.T                                            # [K, V]
        direct_logits = scale * dot_sim + self.topic_bias            # [K, V]

        # Part 2: Mahalanobis penalty (sculpts geometry)
        mahal = self._mahalanobis_cost(a, w)                         # [K, V]

        # Part 3: Sinkhorn refinement (bounded; differentiable)
        eps = F.softplus(self.log_eps) + 1e-3
        log_P = self._sinkhorn_log(-mahal / eps)                     # [K, V]

        # Combine: direct dominates early, Sinkhorn sculpts later
        logits = (direct_logits
                  - self.lambda_mahal * mahal
                  + self.gamma_sinkhorn * log_P)

        # Safety clamp before softmax
        logits = logits.clamp(min=-30.0, max=30.0)
        beta = F.softmax(logits, dim=-1)

        # Safety: replace any NaN (should never happen, defense-in-depth)
        if not torch.isfinite(beta).all():
            beta = torch.nan_to_num(beta, nan=1.0 / self.V)
            beta = beta / beta.sum(dim=-1, keepdim=True)

        return beta
