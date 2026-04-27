"""
============================================================================
 src/models/ednftm.py  [REWRITE — Contextual variant]
 ---------------------------------------------------------------------------
 Full EDNeuFTM-v2 model.

 Architectural change
 --------------------
 GloVe (static, 100d) is GONE. We now use sentence-transformer embeddings:
   • word_embeds  : [V, d_ctx]   contextual embedding per vocab token,
                                   used by the SCAD decoder.
   • doc x_ctx    : [B, d_ctx]   per-document contextual sentence vector,
                                   concatenated with the BoW input inside
                                   the SGP-Encoder.

 Reconstruction:
   recon_probs = θ · β   ∈ [0,1]   (already a valid p(v|doc))
   no double softmax — log(recon_probs + eps) is taken directly in the loss.
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
                 contextual_dim:   int   = 0,
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

        # Novelty 1 — encoder with contextual fusion path
        self.encoder = SGPEncoder(
            vocab_size=vocab_size, hidden_dim=hidden_dim,
            topic_dim=topic_dim, capsule_module=self.router,
            contextual_dim=contextual_dim,
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
                x_bow: torch.Tensor,
                x_ctx: torch.Tensor = None,
                temperature: float = 1.0):
        z_h, theta, mu, logvar, kl = self.encoder(
            x_bow, x_ctx=x_ctx, temperature=temperature
        )
        beta = self.decoder()                 # [K, V] rows sum to 1
        recon_probs = theta @ beta            # [B, V]
        return recon_probs, theta, beta, mu, logvar, kl

    @torch.no_grad()
    def get_beta(self) -> torch.Tensor:
        self.eval()
        return self.decoder().detach().cpu()
