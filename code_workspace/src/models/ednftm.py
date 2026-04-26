"""
============================================================================
 src/models/ednftm.py  [REWRITE — v3 / Contextual]
 ---------------------------------------------------------------------------
 Full EDNeuFTM-v3 model.

 v3 CHANGES
 ──────────
 1. Accepts a per-document contextual embedding (sentence-transformer)
    that is concatenated with the BoW input inside the SGP-E encoder.
 2. SCAD decoder now consumes sentence-transformer token embeddings in
    place of GloVe; ``embed_dim`` should match the contextual dim
    (e.g. 384 for all-MiniLM-L6-v2).
 3. Reconstruction still uses direct θ·β (no extra softmax).
============================================================================
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .sgpe_encoder   import SGPEncoder
from .emgdcr_routing import EMGDCapsuleRouting
from .scad_decoder   import SCADecoder


class EDNeuFTMv2(nn.Module):
    def __init__(self,
                 vocab_size:    int,
                 topic_dim:     int,
                 embed_dim:     int,
                 hidden_dim:    int,
                 word_embeds:   torch.Tensor,
                 pmi_matrix:    torch.Tensor,
                 ctx_dim:          int   = 0,
                 routing_iters:    int   = 3,
                 routing_momentum: float = 0.90,
                 dropout:          float = 0.30,
                 poincare_c:       float = 1.00,
                 poincare_scale:   float = 0.05,
                 fisher_rao_lam:   float = 0.05,
                 scad_rank:        int   = None,
                 sinkhorn_iters:   int   = 10):
        super().__init__()

        # Novelty 3 first
        self.router = EMGDCapsuleRouting(
            input_dim=topic_dim, topic_dim=topic_dim,
            routing_iters=routing_iters, momentum=routing_momentum,
        )

        # Novelty 1 — now contextual
        self.encoder = SGPEncoder(
            vocab_size=vocab_size, hidden_dim=hidden_dim,
            topic_dim=topic_dim, capsule_module=self.router,
            dropout_rate=dropout,
            poincare_c=poincare_c, poincare_scale=poincare_scale,
            fisher_rao_lam=fisher_rao_lam,
            ctx_dim=ctx_dim,
        )
        self.encoder.set_adj_norm(pmi_matrix)

        # Novelty 2 — sentence-transformer word embeddings
        self.decoder = SCADecoder(
            n_topics=topic_dim, vocab_size=vocab_size,
            embed_dim=embed_dim, word_embeds=word_embeds,
            rank=scad_rank, sinkhorn_n=sinkhorn_iters,
        )

    def forward(self,
                x:     torch.Tensor,
                x_ctx: torch.Tensor | None = None,
                temperature: float = 1.0):
        """
        Returns:
          recon_probs : [B, V]   direct θ · β  (already a probability distribution)
          theta       : [B, K]
          beta        : [K, V]
          mu, logvar, kl
        """
        z_h, theta, mu, logvar, kl = self.encoder(x, x_ctx=x_ctx,
                                                  temperature=temperature)
        beta = self.decoder()                 # [K, V]  rows sum to 1
        recon_probs = theta @ beta            # [B, V]  rows sum to 1 (since θ sums to 1)
        return recon_probs, theta, beta, mu, logvar, kl

    @torch.no_grad()
    def get_beta(self) -> torch.Tensor:
        self.eval()
        return self.decoder().detach().cpu()

    # Convenience accessor used by the orthogonal-regularisation term in
    # the loss.  We treat the decoder's CONCEPT ANCHORS (one row per topic
    # in the contextual embedding space) as the canonical "topic weight
    # matrix" of the decoder.  Penalising off-diagonal cosines on this
    # matrix is what drives Inter-cosine ≤ 0.30 and Intra ≥ 0.85.
    @property
    def topic_weight(self) -> torch.Tensor:
        return self.decoder.concept_anchors
