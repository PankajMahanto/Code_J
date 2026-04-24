"""
============================================================================
 src/training/ablation.py  [v4]
 ---------------------------------------------------------------------------
 Ablation study runner - produces Table IV of the paper.

     variant      encoder       routing                 decoder
     -----------------------------------------------------------------
     full         SGP-E         EMGD-CR                 SCAD
     no_sgpe      VanillaMLP    EMGD-CR                 SCAD
     no_emgd      SGP-E         VanillaDynamicRouting   SCAD
     no_scad      SGP-E         EMGD-CR                 VanillaSoftmaxDecoder

 v4 CHANGES
 ----------
 * Contextual (sentence-transformer) word embeddings instead of GloVe.
 * KL linear-anneal warmup + grad clip 1.0.
 * Adds orthogonal regularization to each variant's loss.
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
from ..utils.contextual_embeddings import (
    load_contextual_word_embeddings,
    load_contextual_doc_embeddings,
    DEFAULT_MODEL,
)
from ..utils.pmi               import compute_pmi_matrix
from ..evaluation.coherence_stats   import CoherenceStats
from ..evaluation.coherence_metrics import c_npmi, c_v, c_umass, c_uci
from ..evaluation.diversity_metrics import topic_diversity, inter_topic_cosine
from ..evaluation.quality_gate      import apply_quality_gate
from .losses  import (
    reconstruction_loss, coherence_loss,
    diversity_loss, redundancy_loss,
    orthogonal_regularization,
)

_LOG = get_logger(__name__)


def _cfg_get(obj, key, default):
    return getattr(obj, key, default)


# =============================================================================
# Wrapper model that works with any encoder/router/decoder combo
# =============================================================================
class _AblationWrapper(nn.Module):
    def __init__(self, encoder: nn.Module, decoder: nn.Module,
                 uses_ctx: bool = False):
        super().__init__()
        self.encoder  = encoder
        self.decoder  = decoder
        self.uses_ctx = uses_ctx

    def forward(self, x: torch.Tensor, x_ctx=None, temperature: float = 1.0):
        if self.uses_ctx:
            z_h, theta, mu, logvar, kl = self.encoder(
                x, x_ctx=x_ctx, temperature=temperature)
        else:
            z_h, theta, mu, logvar, kl = self.encoder(
                x, temperature=temperature)
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
                   ctx_dim: int,
                   device: torch.device) -> _AblationWrapper:
    """Build the model for a specific ablation variant."""
    m = cfg.model

    # routing
    if variant == "no_emgd":
        router = VanillaDynamicRouting(m.topic_dim, m.topic_dim,
                                       routing_iters=3)
    else:
        router = EMGDCapsuleRouting(m.topic_dim, m.topic_dim,
                                    routing_iters=m.routing_iters,
                                    momentum=m.routing_momentum)

    # encoder
    uses_ctx = False
    if variant == "no_sgpe":
        enc = VanillaMLPEncoder(V, m.hidden_dim, m.topic_dim, router,
                                dropout_rate=m.dropout)
    else:
        enc = SGPEncoder(V, m.hidden_dim, m.topic_dim, router,
                         dropout_rate=m.dropout,
                         poincare_c=m.poincare_c,
                         poincare_scale=m.poincare_scale,
                         fisher_rao_lam=m.fisher_rao_lam,
                         contextual_dim=ctx_dim)
        enc.set_adj_norm(pmi)
        uses_ctx = ctx_dim > 0

    # decoder
    if variant == "no_scad":
        dec = VanillaSoftmaxDecoder(m.topic_dim, V)
    else:
        dec = SCADecoder(m.topic_dim, V, word_embeds.shape[1], word_embeds,
                         rank=m.scad_rank, sinkhorn_n=m.sinkhorn_iters)

    return _AblationWrapper(enc, dec, uses_ctx=uses_ctx).to(device)


# =============================================================================
# Per-variant trainer (reduced epochs for speed)
# =============================================================================
def _train_variant(variant: str, cfg, V: int, loader,
                   word_embeds: torch.Tensor,
                   pmi: torch.Tensor,
                   ctx_dim: int,
                   stats: CoherenceStats,
                   emb_dict: Dict[str, np.ndarray],
                   dictionary: Dictionary,
                   device: torch.device,
                   ablation_epochs: int = 50) -> Dict:
    _LOG.info("=" * 60)
    _LOG.info(f"  ABLATION VARIANT: {variant.upper()}")
    _LOG.info("=" * 60)

    model = _build_variant(variant, cfg, V, word_embeds, pmi, ctx_dim, device)
    opt   = torch.optim.Adam(model.parameters(),
                             lr=cfg.training.lr,
                             weight_decay=cfg.training.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=ablation_epochs)
    pmi_dev = pmi.to(device)

    kl_warmup = _cfg_get(cfg.training, "kl_warmup_epoch", 20)
    max_norm  = _cfg_get(cfg.training, "grad_clip", 1.0)
    ortho_w   = _cfg_get(cfg.loss_weights, "ortho", 1.0)
    ortho_mode = _cfg_get(cfg.loss_weights, "ortho_mode", "cosine")

    for epoch in range(1, ablation_epochs + 1):
        model.train()
        frac = epoch / ablation_epochs
        T    = cfg.training.temp_start * (cfg.training.temp_end / cfg.training.temp_start) ** frac
        kl_w = cfg.loss_weights.kl * min(1.0, epoch / max(1, kl_warmup))

        for x_norm, x_raw, x_ctx in loader:
            x_norm = x_norm.to(device); x_raw = x_raw.to(device)
            x_ctx  = x_ctx.to(device) if x_ctx.numel() else None

            recon, theta, beta, mu, logvar, kl = model(
                x_norm, x_ctx=x_ctx, temperature=T)

            L_rec   = reconstruction_loss(recon, x_raw)
            L_coh   = coherence_loss(beta, pmi_dev)
            L_div   = diversity_loss(beta)
            L_red   = redundancy_loss(beta)
            L_ortho = orthogonal_regularization(beta, mode=ortho_mode)

            loss = (cfg.loss_weights.recon * L_rec
                    + kl_w                 * kl
                    + cfg.loss_weights.coh * L_coh
                    + cfg.loss_weights.div * L_div
                    + cfg.loss_weights.red * L_red
                    + ortho_w              * L_ortho)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
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
        min_npmi = 0.25,
        min_cv   = 0.40,
        max_jac  = 0.30,
        min_keep = 5,
    )
    npmi = c_npmi(topics_gated, stats, cfg.evaluation.top_n_words)

    results = {
        "variant":   variant,
        "n_topics":  len(topics_gated),
        "C_V":       round(c_v(topics_gated, stats, emb_dict,
                               cfg.evaluation.top_n_words), 4),
        "C_NPMI":    round(npmi, 4),
        "U_Mass":    round(c_umass(topics_gated, stats,
                                   cfg.evaluation.top_n_words), 4),
        "C_UCI":     round(c_uci(topics_gated, stats,
                                 cfg.evaluation.top_n_words), 4),
        "Diversity": round(topic_diversity(topics_gated, 25), 4),
        "Intra":     round((npmi + 1.0) / 2.0, 4),
        "Inter":     round(inter_topic_cosine(topics_gated, emb_dict,
                                              cfg.evaluation.top_n_words), 4),
    }
    _LOG.info(f"-> {results}")
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

    ctx_path = os.path.join(preproc_dir, "doc_ctx_emb.pt")
    if os.path.exists(ctx_path):
        ctx_doc = torch.load(ctx_path, map_location="cpu")
    else:
        ctx_doc = load_contextual_doc_embeddings(
            docs,
            model_name=_cfg_get(cfg.dataset, "contextual_model", DEFAULT_MODEL),
            cache_path=ctx_path,
        )

    V = len(dictionary)
    model_name = _cfg_get(cfg.dataset, "contextual_model", DEFAULT_MODEL)
    word_embeds = load_contextual_word_embeddings(
        dictionary, model_name=model_name, device=device,
        cache_path=os.path.join(preproc_dir, "word_ctx_emb.pt"),
    )
    ctx_dim = int(ctx_doc.shape[1])

    pmi = compute_pmi_matrix(docs, dictionary,
                             window=cfg.preprocessing.ref_window_size)

    loader = make_dataloader(bow,
                             batch_size=cfg.training.batch_size,
                             ctx_embeds=ctx_doc)
    vocab_set = set(dictionary.token2id.keys())
    stats     = CoherenceStats(ref_windows, vocab=vocab_set)
    emb_dict  = {w: word_embeds[dictionary.token2id[w]].numpy()
                 for w in vocab_set}

    all_results = []
    for v in variants:
        res = _train_variant(v, cfg, V, loader, word_embeds, pmi, ctx_dim,
                             stats, emb_dict, dictionary, device,
                             ablation_epochs=ablation_epochs)
        all_results.append(res)

    _LOG.info("\n" + "=" * 92)
    _LOG.info(f"{'Variant':<12}{'C_V':>8}{'C_NPMI':>10}{'U_Mass':>10}"
              f"{'C_UCI':>10}{'Div':>8}{'Intra':>8}{'Inter':>8}{'#':>6}")
    _LOG.info("=" * 92)
    for r in all_results:
        _LOG.info(f"{r['variant']:<12}{r['C_V']:>8.3f}{r['C_NPMI']:>10.3f}"
                  f"{r['U_Mass']:>10.3f}{r['C_UCI']:>10.3f}"
                  f"{r['Diversity']:>8.3f}{r['Intra']:>8.3f}"
                  f"{r['Inter']:>8.3f}{r['n_topics']:>6d}")
    _LOG.info("=" * 92)

    out_path = os.path.join(cfg.dataset.work_dir, "train", "ablation_results.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    _LOG.info(f"ablation results -> {out_path}")
    return all_results
