"""
============================================================================
 src/models/ednftm.py  [v4 — contextual fusion]
 ---------------------------------------------------------------------------
 Full EDNeuFTM-v2 model.

 v4 CHANGES
 ──────────
 • Encoder now accepts a per-document contextual embedding (sentence-
   transformer output) in addition to the BoW vector.  GloVe is gone.
 • The decoder is initialised from contextual word embeddings rather than
   GloVe word vectors.
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

        # Novelty 1
        self.encoder = SGPEncoder(
            vocab_size=vocab_size, hidden_dim=hidden_dim,
            topic_dim=topic_dim, capsule_module=self.router,
            ctx_dim=ctx_dim,
            dropout_rate=dropout,
            poincare_c=poincare_c, poincare_scale=poincare_scale,
            fisher_rao_lam=fisher_rao_lam,
        )
        self.encoder.set_adj_norm(pmi_matrix)

        # Novelty 2
        self.decoder = SCADecoder(
            n_topics=topic_dim, vocab_size=vocab_size,
            embed_dim=embed_dim, word_embeds=word_embeds,
            rank=scad_rank, sinkhorn_n=sinkhorn_iters,
        )

    def forward(self,
                x:   torch.Tensor,
                ctx: torch.Tensor | None = None,
                temperature: float = 1.0):
        """
        Returns:
          recon_probs : [B, V]   direct θ · β
          theta       : [B, K]
          beta        : [K, V]
          mu, logvar, kl
        """
        z_h, theta, mu, logvar, kl = self.encoder(x, ctx=ctx, temperature=temperature)
        beta = self.decoder()
        recon_probs = theta @ beta
        return recon_probs, theta, beta, mu, logvar, kl

    @torch.no_grad()
    def get_beta(self) -> torch.Tensor:
        self.eval()
        return self.decoder().detach().cpu()
