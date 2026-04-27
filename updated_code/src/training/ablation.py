"""
============================================================================
 src/training/ablation.py
 ---------------------------------------------------------------------------
 Ablation study runner — produces Table IV of the paper.

 Runs four variants of the model with a reduced epoch budget:

   variant      encoder       routing                 decoder
   ─────────────────────────────────────────────────────────────────
   full         SGP-E         EMGD-CR                 SCAD
   no_sgpe      VanillaMLP    EMGD-CR                 SCAD
   no_emgd      SGP-E         VanillaDynamicRouting   SCAD
   no_scad      SGP-E         EMGD-CR                 VanillaSoftmaxDecoder
============================================================================
"""
from __future__ import annotations

import os
import json
import pickle
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from scipy import sparse
from gensim.corpora import Dictionary

from ..models import (
    SGPEncoder, EMGDCapsuleRouting, SCADecoder,
    VanillaMLPEncoder, VanillaDynamicRouting, VanillaSoftmaxDecoder,
)
from ..data                    import make_dataloader
from ..utils.logging_utils     import get_logger
from ..utils.contextual_loader import (
    load_contextual_word_embeddings,
    compute_contextual_doc_embeddings,
)
from ..utils.pmi               import compute_pmi_matrix
from ..evaluation.coherence_stats   import CoherenceStats
from ..evaluation.coherence_metrics import c_npmi, c_v, c_umass, c_uci
from ..evaluation.diversity_metrics import topic_diversity
from ..evaluation.quality_gate      import apply_quality_gate
from .losses  import (
    reconstruction_loss, coherence_loss,
    diversity_loss, redundancy_loss,
    orthogonal_regularization,
)

_LOG = get_logger(__name__)


# =============================================================================
# Wrapper model that works with any encoder/router/decoder combo
# =============================================================================
class _AblationWrapper(nn.Module):
    def __init__(self, encoder: nn.Module, decoder: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self,
                x: torch.Tensor,
                x_ctx: torch.Tensor = None,
                temperature: float = 1.0):
        # SGPEncoder accepts (x_bow, x_ctx); VanillaMLPEncoder ignores x_ctx.
        try:
            z_h, theta, mu, logvar, kl = self.encoder(
                x, x_ctx=x_ctx, temperature=temperature
            )
        except TypeError:
            z_h, theta, mu, logvar, kl = self.encoder(x, temperature=temperature)
        beta = self.decoder()
        recon = theta @ beta
        return recon, theta, beta, mu, logvar, kl

    @torch.no_grad()
    def get_beta(self) -> torch.Tensor:
        self.eval()
        return self.decoder().detach().cpu()


def _build_variant(variant: str, cfg, V: int,
                   word_embeds: torch.Tensor,
                   pmi: torch.Tensor,
                   device: torch.device) -> _AblationWrapper:
    """Build the model for a specific ablation variant."""
    m = cfg.model

    # ── routing ──
    if variant == "no_emgd":
        router = VanillaDynamicRouting(m.topic_dim, m.topic_dim,
                                       routing_iters=3)
    else:
        router = EMGDCapsuleRouting(m.topic_dim, m.topic_dim,
                                    routing_iters=m.routing_iters,
                                    momentum=m.routing_momentum)

    # ── encoder ──
    contextual_dim = int(word_embeds.shape[1])
    if variant == "no_sgpe":
        enc = VanillaMLPEncoder(V, m.hidden_dim, m.topic_dim, router,
                                dropout_rate=m.dropout)
    else:
        enc = SGPEncoder(V, m.hidden_dim, m.topic_dim, router,
                         contextual_dim=contextual_dim,
                         dropout_rate=m.dropout,
                         poincare_c=m.poincare_c,
                         poincare_scale=m.poincare_scale,
                         fisher_rao_lam=m.fisher_rao_lam)
        enc.set_adj_norm(pmi)

    # ── decoder ──
    if variant == "no_scad":
        dec = VanillaSoftmaxDecoder(m.topic_dim, V)
    else:
        dec = SCADecoder(m.topic_dim, V, contextual_dim, word_embeds,
                         rank=m.scad_rank, sinkhorn_n=m.sinkhorn_iters)

    return _AblationWrapper(enc, dec).to(device)


# =============================================================================
# Per-variant trainer (reduced epochs for speed)
# =============================================================================
def _train_variant(variant: str, cfg, V: int, loader,
                   word_embeds: torch.Tensor,
                   pmi: torch.Tensor,
                   stats: CoherenceStats,
                   emb_dict: Dict[str, np.ndarray],
                   dictionary: Dictionary,
                   device: torch.device,
                   ablation_epochs: int = 50) -> Dict:
    _LOG.info(f"{'=' * 60}")
    _LOG.info(f"  ABLATION VARIANT: {variant.upper()}")
    _LOG.info(f"{'=' * 60}")

    model = _build_variant(variant, cfg, V, word_embeds, pmi, device)
    opt   = torch.optim.Adam(model.parameters(),
                             lr=cfg.training.lr,
                             weight_decay=cfg.training.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=ablation_epochs)
    pmi_dev = pmi.to(device)

    kl_warmup_eps = getattr(cfg.training, "kl_warmup_epoch", 20)
    ortho_weight  = getattr(cfg.loss_weights, "ortho", 1.0)
    grad_clip     = getattr(cfg.training, "grad_clip", 1.0)

    for epoch in range(1, ablation_epochs + 1):
        model.train()
        frac = epoch / ablation_epochs
        T    = cfg.training.temp_start * (cfg.training.temp_end / cfg.training.temp_start) ** frac
        # Linear KL annealing 0 -> target over kl_warmup_eps epochs
        kl_w = cfg.loss_weights.kl * min(1.0, epoch / max(1, kl_warmup_eps))

        for x_norm, x_raw, x_ctx in loader:
            x_input = torch.log1p(x_raw.to(device))
            x_raw   = x_raw.to(device)
            x_ctx   = x_ctx.to(device)
            recon, theta, beta, mu, logvar, kl = model(
                x_input, x_ctx=x_ctx, temperature=T
            )

            L_rec   = reconstruction_loss(recon, x_raw) / V
            L_coh   = coherence_loss(beta, pmi_dev)
            L_div   = diversity_loss(beta)
            L_red   = redundancy_loss(beta)
            # Orthogonal regularization is supported only when SCAD is active
            if hasattr(model.decoder, "weight"):
                L_ortho = orthogonal_regularization(model.decoder.weight)
            else:
                L_ortho = orthogonal_regularization(beta)

            loss = (cfg.loss_weights.recon * L_rec
                    + kl_w                  * kl
                    + cfg.loss_weights.coh  * L_coh
                    + cfg.loss_weights.div  * L_div
                    + cfg.loss_weights.red  * L_red
                    + ortho_weight          * L_ortho)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),
                                           max_norm=grad_clip)
            opt.step()

        sched.step()

    # ---- evaluate ----
    beta_final = model.get_beta()
    i2w = {v: k for k, v in dictionary.token2id.items()}
    topics = []
    for k in range(cfg.model.topic_dim):
        idx = beta_final[k].topk(cfg.evaluation.top_n_words).indices.tolist()
        topics.append([i2w[i] for i in idx])

    topics_gated = apply_quality_gate(
        topics, stats, emb_dict,
        top_n    = cfg.evaluation.top_n_words,
        min_npmi = 0.25,        # relaxed for ablation, so all variants report
        min_cv   = 0.40,
        max_jac  = 0.30,
        min_keep = 5,
    )

    results = {
        "variant":   variant,
        "n_topics":  len(topics_gated),
        "C_V":       round(c_v(topics_gated, stats, emb_dict,
                               cfg.evaluation.top_n_words), 4),
        "C_NPMI":    round(c_npmi(topics_gated, stats,
                                  cfg.evaluation.top_n_words), 4),
        "U_Mass":    round(c_umass(topics_gated, stats,
                                   cfg.evaluation.top_n_words), 4),
        "C_UCI":     round(c_uci(topics_gated, stats,
                                 cfg.evaluation.top_n_words), 4),
        "Diversity": round(topic_diversity(topics_gated, 25), 4),
    }
    _LOG.info(f"→ {results}")
    return results


# =============================================================================
# Public entry point
# =============================================================================
def run_ablation_suite(cfg, variants: List[str] = None,
                       ablation_epochs: int = 50) -> List[Dict]:
    """Train every variant and return a list of metric dicts."""
    variants = variants or ["full", "no_sgpe", "no_emgd", "no_scad"]
    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- load artifacts once ----
    preproc_dir = os.path.join(cfg.dataset.work_dir, "preproc")
    with open(os.path.join(preproc_dir, "clean_docs.pkl"), "rb") as f:
        docs = pickle.load(f)
    dictionary  = Dictionary.load(os.path.join(preproc_dir, "dictionary.gensim"))
    bow         = sparse.load_npz(os.path.join(preproc_dir, "bow_matrix.npz"))
    with open(os.path.join(preproc_dir, "reference_corpus.pkl"), "rb") as f:
        ref_windows = pickle.load(f)

    V = len(dictionary)
    ctx_model = getattr(cfg.model, "contextual_model",
                        "sentence-transformers/all-MiniLM-L6-v2")
    cache_dir = os.path.join(cfg.dataset.work_dir, "ctx_cache")
    word_embeds = load_contextual_word_embeddings(
        ctx_model, dictionary, cache_dir=cache_dir)
    doc_ctx = compute_contextual_doc_embeddings(
        ctx_model, docs, cache_dir=cache_dir)
    pmi = compute_pmi_matrix(docs, dictionary,
                             window=cfg.preprocessing.ref_window_size)

    loader = make_dataloader(bow,
                             batch_size=cfg.training.batch_size,
                             ctx_embeds=doc_ctx)
    vocab_set = set(dictionary.token2id.keys())
    stats     = CoherenceStats(ref_windows, vocab=vocab_set)
    emb_dict  = {w: word_embeds[dictionary.token2id[w]].numpy()
                 for w in vocab_set}

    all_results = []
    for v in variants:
        res = _train_variant(v, cfg, V, loader, word_embeds, pmi,
                             stats, emb_dict, dictionary, device,
                             ablation_epochs=ablation_epochs)
        all_results.append(res)

    # pretty-print table
    _LOG.info("\n" + "=" * 80)
    _LOG.info(f"{'Variant':<14}{'C_V':>8}{'C_NPMI':>10}{'U_Mass':>10}"
              f"{'C_UCI':>10}{'Div':>8}{'# topics':>10}")
    _LOG.info("=" * 80)
    for r in all_results:
        _LOG.info(f"{r['variant']:<14}{r['C_V']:>8.3f}{r['C_NPMI']:>10.3f}"
                  f"{r['U_Mass']:>10.3f}{r['C_UCI']:>10.3f}"
                  f"{r['Diversity']:>8.3f}{r['n_topics']:>10d}")
    _LOG.info("=" * 80)

    out_path = os.path.join(cfg.dataset.work_dir, "train", "ablation_results.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    _LOG.info(f"✓ ablation results -> {out_path}")
    return all_results
