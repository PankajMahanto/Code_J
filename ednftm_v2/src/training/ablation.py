"""
============================================================================
 src/training/ablation.py  [REWRITE — v4]
 ---------------------------------------------------------------------------
 Ablation study runner — produces Table IV of the paper.

 Four variants (reduced-epoch budget) over a SHARED preprocessed corpus:

     variant      encoder       routing                 decoder
     ─────────────────────────────────────────────────────────────────
     full         SGP-E         EMGD-CR                 SCAD
     no_sgpe      VanillaMLP    EMGD-CR                 SCAD
     no_emgd      SGP-E         VanillaDynamicRouting   SCAD
     no_scad      SGP-E         EMGD-CR                 VanillaSoftmaxDecoder

 v4 CHANGES
 ──────────
 • GloVe removed — uses sentence-transformer vocab embeddings instead.
 • Passes the contextual doc embedding to SGP-E variants (MLP baseline
   ignores it).
 • Uses orthogonal regularisation, linear KL annealing (20 epochs), and
   gradient clipping (1.0) — identical to the main trainer.
============================================================================
"""
from __future__ import annotations

import os
import json
import pickle
from typing import Dict, List, Optional

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
from ..utils.contextual_embedder import (
    encode_documents, encode_vocabulary, DEFAULT_MODEL as DEFAULT_SBERT_MODEL,
)
from ..utils.pmi               import compute_pmi_matrix
from ..evaluation.coherence_stats   import CoherenceStats
from ..evaluation.coherence_metrics import c_npmi, c_v, c_umass, c_uci
from ..evaluation.diversity_metrics import topic_diversity, inter_topic_cosine
from ..evaluation.quality_gate      import apply_quality_gate
from .losses  import (
    reconstruction_loss, coherence_loss,
    diversity_loss, redundancy_loss, orthogonal_regularization,
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

    def forward(self, x: torch.Tensor,
                x_ctx: Optional[torch.Tensor] = None,
                temperature: float = 1.0):
        z_h, theta, mu, logvar, kl = self.encoder(
            x, x_ctx=x_ctx, temperature=temperature
        )
        beta  = self.decoder()
        recon = theta @ beta
        return recon, theta, beta, mu, logvar, kl

    @torch.no_grad()
    def get_beta(self) -> torch.Tensor:
        self.eval()
        return self.decoder().detach().cpu()


def _build_variant(variant: str, cfg, V: int,
                   word_embeds: torch.Tensor,
                   pmi: torch.Tensor,
                   context_dim: int,
                   device: torch.device) -> _AblationWrapper:
    m = cfg.model

    # ── routing ─────────────────────────────────────────────────────
    if variant == "no_emgd":
        router = VanillaDynamicRouting(m.topic_dim, m.topic_dim,
                                       routing_iters=3)
    else:
        router = EMGDCapsuleRouting(m.topic_dim, m.topic_dim,
                                    routing_iters=m.routing_iters,
                                    momentum=m.routing_momentum)

    # ── encoder ─────────────────────────────────────────────────────
    if variant == "no_sgpe":
        enc = VanillaMLPEncoder(V, m.hidden_dim, m.topic_dim, router,
                                dropout_rate=m.dropout)
    else:
        enc = SGPEncoder(V, m.hidden_dim, m.topic_dim, router,
                         context_dim=context_dim,
                         dropout_rate=m.dropout,
                         poincare_c=m.poincare_c,
                         poincare_scale=m.poincare_scale,
                         fisher_rao_lam=m.fisher_rao_lam)
        enc.set_adj_norm(pmi)

    # ── decoder ─────────────────────────────────────────────────────
    if variant == "no_scad":
        dec = VanillaSoftmaxDecoder(m.topic_dim, V)
    else:
        dec = SCADecoder(m.topic_dim, V, word_embeds.size(1), word_embeds,
                         rank=m.scad_rank, sinkhorn_n=m.sinkhorn_iters)

    return _AblationWrapper(enc, dec).to(device)


# =============================================================================
# Per-variant trainer (reduced epochs for speed)
# =============================================================================
def _train_variant(variant: str, cfg, V: int, loader,
                   word_embeds: torch.Tensor,
                   pmi: torch.Tensor,
                   context_dim: int,
                   stats: CoherenceStats,
                   emb_dict: Dict[str, np.ndarray],
                   dictionary: Dictionary,
                   device: torch.device,
                   ablation_epochs: int = 50) -> Dict:
    _LOG.info(f"{'=' * 60}")
    _LOG.info(f"  ABLATION VARIANT: {variant.upper()}")
    _LOG.info(f"{'=' * 60}")

    model = _build_variant(variant, cfg, V, word_embeds, pmi, context_dim, device)
    opt   = torch.optim.AdamW(model.parameters(),
                              lr=cfg.training.lr,
                              weight_decay=cfg.training.weight_decay,
                              eps=1e-7)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=ablation_epochs)
    pmi_dev = pmi.to(device)

    kl_warmup = min(20, ablation_epochs // 2)
    kl_target = float(cfg.loss_weights.kl)
    ortho_w   = float(getattr(cfg.loss_weights, "ortho", 0.0))

    for epoch in range(1, ablation_epochs + 1):
        model.train()
        frac = (epoch - 1) / max(1, ablation_epochs - 1)
        T = cfg.training.temp_start + (cfg.training.temp_end - cfg.training.temp_start) * frac
        kl_w = kl_target * min(1.0, epoch / max(1, kl_warmup))

        for x_raw, x_ctx in loader:
            x_raw = x_raw.to(device)
            x_ctx = x_ctx.to(device) if x_ctx.numel() > 0 else None
            x_input = torch.log1p(x_raw)

            recon, theta, beta, mu, logvar, kl = model(
                x_input, x_ctx=x_ctx, temperature=T,
            )

            L_rec   = reconstruction_loss(recon, x_raw) / V
            L_coh   = coherence_loss(beta, pmi_dev)
            L_div   = diversity_loss(beta)
            L_red   = redundancy_loss(beta)
            L_ortho = orthogonal_regularization(beta)

            loss = (cfg.loss_weights.recon * L_rec
                    + kl_w                 * kl
                    + cfg.loss_weights.coh * L_coh
                    + cfg.loss_weights.div * L_div
                    + cfg.loss_weights.red * L_red
                    + ortho_w              * L_ortho)

            if not torch.isfinite(loss):
                opt.zero_grad(); continue

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),
                                           float(getattr(cfg.training,
                                                         "grad_clip", 1.0)))
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

    top_n = cfg.evaluation.top_n_words
    npmi_val = c_npmi(topics_gated, stats, top_n)
    results = {
        "variant":   variant,
        "n_topics":  len(topics_gated),
        "C_V":       round(c_v(topics_gated, stats, emb_dict, top_n), 4),
        "C_NPMI":    round(npmi_val, 4),
        "U_Mass":    round(c_umass(topics_gated, stats, top_n), 4),
        "C_UCI":     round(c_uci(topics_gated, stats, top_n), 4),
        "Diversity": round(topic_diversity(topics_gated, 25), 4),
        "Intra":     round(max(0.0, min(1.0, (npmi_val + 1.0) / 2.0)), 4),
        "Inter":     round(inter_topic_cosine(topics_gated, emb_dict, top_n), 4),
    }
    _LOG.info(f"→ {results}")
    return results


# =============================================================================
# Public entry point
# =============================================================================
def run_ablation_suite(cfg, variants: List[str] = None,
                       ablation_epochs: int = 50) -> List[Dict]:
    variants = variants or ["full", "no_sgpe", "no_emgd", "no_scad"]
    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    preproc_dir = os.path.join(cfg.dataset.work_dir, "preproc")
    cache_dir   = os.path.join(cfg.dataset.work_dir, "cache")
    os.makedirs(cache_dir, exist_ok=True)

    with open(os.path.join(preproc_dir, "clean_docs.pkl"), "rb") as f:
        docs = pickle.load(f)
    dictionary  = Dictionary.load(os.path.join(preproc_dir, "dictionary.gensim"))
    bow         = sparse.load_npz(os.path.join(preproc_dir, "bow_matrix.npz"))
    with open(os.path.join(preproc_dir, "reference_corpus.pkl"), "rb") as f:
        ref_windows = pickle.load(f)

    V = len(dictionary)

    sbert_name = getattr(cfg.model, "sbert_model", DEFAULT_SBERT_MODEL)
    doc_emb = encode_documents(
        docs, cache_path=os.path.join(cache_dir, "sbert_doc.npy"),
        model_name=sbert_name, device=str(device),
    )
    vocab_tokens = [t for t, _ in sorted(dictionary.token2id.items(),
                                         key=lambda kv: kv[1])]
    word_embeds = encode_vocabulary(
        vocab_tokens, cache_path=os.path.join(cache_dir, "sbert_vocab.npy"),
        model_name=sbert_name, device=str(device),
    )
    context_dim = doc_emb.size(1)

    pmi = compute_pmi_matrix(docs, dictionary,
                             window=cfg.preprocessing.ref_window_size)

    loader = make_dataloader(
        bow, context_embeds=doc_emb,
        batch_size=cfg.training.batch_size,
    )
    vocab_set = set(dictionary.token2id.keys())
    stats     = CoherenceStats(ref_windows, vocab=vocab_set)
    emb_dict  = {w: word_embeds[dictionary.token2id[w]].numpy()
                 for w in vocab_set}

    all_results = []
    for v in variants:
        res = _train_variant(v, cfg, V, loader, word_embeds, pmi,
                             context_dim, stats, emb_dict, dictionary, device,
                             ablation_epochs=ablation_epochs)
        all_results.append(res)

    _LOG.info("\n" + "=" * 92)
    _LOG.info(f"{'Variant':<12}{'C_V':>8}{'C_NPMI':>10}{'U_Mass':>10}"
              f"{'C_UCI':>10}{'Div':>8}{'Intra':>8}{'Inter':>8}{'#t':>6}")
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
    _LOG.info(f"✓ ablation results -> {out_path}")
    return all_results
