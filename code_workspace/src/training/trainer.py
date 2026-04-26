"""
============================================================================
 src/training/trainer.py  [REWRITE — v4 / Contextual]
 ---------------------------------------------------------------------------
 v4 DESIGN DECISIONS
 ───────────────────
 1. Static GloVe (100-d) is REMOVED.  We now load sentence-transformer
    embeddings (default all-MiniLM-L6-v2, 384-d) for BOTH the SCAD decoder
    word_embeds AND a per-document contextual vector that is concatenated
    with the BoW input inside the SGP-E encoder.

 2. KL ANNEALING — explicit linear warm-up of the KL weight from 0.0 → 1.0
    over the first ``kl_warmup_epoch`` epochs (default 20).  This is the
    canonical fix for the mode-collapse failure observed at epoch 15
    (loss=0.000, kl=0.00) — the encoder is allowed to learn a non-trivial
    posterior before the KL term starts pulling it toward N(0, I).

 3. GRADIENT CLIPPING — torch.nn.utils.clip_grad_norm_ with max_norm=1.0
    is applied EVERY optimisation step (with a NaN guard upstream).

 4. ORTHOGONAL REGULARISATION — applied to the decoder's topic weight
    matrix (model.topic_weight, i.e. concept_anchors).  The weight is read
    from cfg.loss_weights.ortho.

 5. FULL METRIC LOGGING — every evaluation epoch logs all six target
    metrics (C_V, C_NPMI, U_Mass, C_UCI, Topic Diversity, Intra, Inter)
    so the training run gives a continuous read on the Q1-journal targets.

 6. Use RAW counts (log1p) as encoder BoW input.
============================================================================
"""
from __future__ import annotations

import os, json, pickle, math
from typing import Dict
import numpy as np
import torch
from scipy import sparse
from gensim.corpora import Dictionary

from ..models                 import EDNeuFTMv2
from ..data                    import make_dataloader
from ..utils.logging_utils     import get_logger
from ..utils.contextual_embeddings import (
    encode_vocabulary, encode_documents, get_contextual_dim,
)
from ..utils.pmi               import compute_pmi_matrix
from ..evaluation.coherence_stats   import CoherenceStats
from ..evaluation.coherence_metrics import c_npmi, c_v, c_umass, c_uci
from ..evaluation.quality_gate      import apply_quality_gate
from ..evaluation.diversity_metrics import topic_diversity, inter_topic_cosine
from .losses import (
    reconstruction_loss, coherence_loss,
    diversity_loss, redundancy_loss, orthogonal_regularization,
)

_LOG = get_logger(__name__)


def _cfg_get(node, key, default):
    """Safe attribute lookup on the dotted Config wrapper."""
    if hasattr(node, key):
        return getattr(node, key)
    if hasattr(node, "get"):
        try:
            return node.get(key, default)
        except Exception:
            return default
    return default


class EarlyStopping:
    """Plateau OR NaN stop."""
    def __init__(self, patience=20, min_delta=1e-3, max_nans=2):
        self.patience = patience
        self.min_delta = min_delta
        self.max_nans  = max_nans
        self.best      = -float("inf")
        self.wait      = 0
        self.nans      = 0
        self.stop      = False
        self.reason    = None

    def update(self, score: float, had_nan: bool) -> bool:
        if had_nan:
            self.nans += 1
            if self.nans >= self.max_nans:
                self.stop = True
                self.reason = f"NaN x {self.nans}"
            return self.stop
        self.nans = 0

        if not math.isfinite(score):
            return False

        if score > self.best + self.min_delta:
            self.best = score
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stop = True
                self.reason = f"no improvement for {self.wait} evals"
        return self.stop


class Trainer:
    def __init__(self, cfg, device=None):
        self.cfg    = cfg
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _LOG.info(f"Device: {self.device}")
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)
        torch.cuda.manual_seed_all(cfg.seed)

    def _load(self, preproc_dir):
        with open(os.path.join(preproc_dir, "clean_docs.pkl"), "rb") as f:
            docs = pickle.load(f)
        dictionary = Dictionary.load(os.path.join(preproc_dir, "dictionary.gensim"))
        bow = sparse.load_npz(os.path.join(preproc_dir, "bow_matrix.npz"))
        with open(os.path.join(preproc_dir, "reference_corpus.pkl"), "rb") as f:
            ref = pickle.load(f)
        return docs, dictionary, bow, ref

    @staticmethod
    def _topics_from_beta(beta, dictionary, top_n):
        i2w = {v: k for k, v in dictionary.token2id.items()}
        beta = torch.nan_to_num(beta, nan=0.0)
        out = []
        for k in range(beta.shape[0]):
            idx = beta[k].topk(top_n).indices.tolist()
            out.append([i2w[i] for i in idx])
        return out

    def fit(self) -> Dict:
        cfg, device = self.cfg, self.device
        preproc_dir = os.path.join(cfg.dataset.work_dir, "preproc")
        train_dir   = os.path.join(cfg.dataset.work_dir, "train")
        os.makedirs(train_dir, exist_ok=True)

        # ---- load preprocessed artefacts ----
        _LOG.info("Loading preprocessed artifacts ...")
        docs, dictionary, bow, ref_windows = self._load(preproc_dir)
        V = len(dictionary)

        # ---- contextual embeddings (sentence-transformer) ----
        st_model = _cfg_get(cfg.model, "contextual_model",
                            "sentence-transformers/all-MiniLM-L6-v2")
        ctx_dim  = get_contextual_dim(st_model)
        _LOG.info(f"Contextual model: {st_model}  (dim={ctx_dim})")

        # Override embed_dim to match contextual dim — guarantees the SCAD
        # decoder operates in the same space as the encoder's contextual
        # signal.  We log a warning if the YAML sets a mismatched value.
        cfg_embed_dim = _cfg_get(cfg.model, "embed_dim", ctx_dim)
        if cfg_embed_dim != ctx_dim:
            _LOG.warning(
                f"cfg.model.embed_dim={cfg_embed_dim} != contextual dim "
                f"{ctx_dim}; using {ctx_dim}."
            )
        embed_dim = ctx_dim

        vocab_cache = os.path.join(preproc_dir, f"ctx_vocab_{embed_dim}.pt")
        doc_cache   = os.path.join(preproc_dir, f"ctx_docs_{embed_dim}.pt")
        word_embeds = encode_vocabulary(dictionary, model_name=st_model,
                                        device=device, cache_path=vocab_cache)
        doc_embeds  = encode_documents(docs, model_name=st_model,
                                       device=device, cache_path=doc_cache)

        # ---- PMI for the spectral GCN + soft-PMI coherence loss ----
        pmi = compute_pmi_matrix(docs, dictionary,
                                 window=cfg.preprocessing.ref_window_size)
        pmi_dev = pmi.to(device)

        # ---- build model ----
        m = cfg.model
        model = EDNeuFTMv2(
            vocab_size=V, topic_dim=m.topic_dim,
            embed_dim=embed_dim, hidden_dim=m.hidden_dim,
            word_embeds=word_embeds, pmi_matrix=pmi,
            ctx_dim=ctx_dim,
            routing_iters=m.routing_iters,
            routing_momentum=m.routing_momentum,
            dropout=m.dropout,
            poincare_c=m.poincare_c,
            poincare_scale=m.poincare_scale,
            fisher_rao_lam=m.fisher_rao_lam,
            scad_rank=m.scad_rank,
            sinkhorn_iters=m.sinkhorn_iters,
        ).to(device)

        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        _LOG.info(f"Trainable parameters: {n_params:,}")

        # ---- data ----
        loader = make_dataloader(bow,
                                 batch_size=cfg.training.batch_size,
                                 shuffle=True,
                                 ctx_embeds=doc_embeds,
                                 ctx_dim=ctx_dim)

        opt = torch.optim.AdamW(model.parameters(),
                                lr=cfg.training.lr,
                                weight_decay=cfg.training.weight_decay,
                                eps=1e-7)

        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=cfg.training.epochs, eta_min=cfg.training.lr * 0.1)

        # ---- coherence stats (once) ----
        vocab_set = set(dictionary.token2id.keys())
        stats     = CoherenceStats(ref_windows, vocab=vocab_set)
        emb_dict  = {w: word_embeds[dictionary.token2id[w]].numpy()
                     for w in vocab_set}

        early = EarlyStopping(patience=_cfg_get(cfg.training, "es_patience", 20),
                              min_delta=1e-3, max_nans=2)

        best_score, best_state = -1e9, None
        history = []
        t_start, t_end = cfg.training.temp_start, cfg.training.temp_end

        # ---- KL annealing config ----
        # Linear warm-up of the KL weight (β in β-VAE literature) from 0.0
        # to 1.0 over the first `kl_warmup_epoch` epochs.  Standard fix for
        # the posterior-collapse failure mode.
        kl_warmup_epochs = int(_cfg_get(cfg.training, "kl_warmup_epoch", 20))
        kl_target_w      = float(cfg.loss_weights.kl)

        # ---- orthogonal regularisation weight ----
        ortho_w = float(_cfg_get(cfg.loss_weights, "ortho", 1.0))

        # ---- gradient clipping ----
        grad_clip = float(_cfg_get(cfg.training, "grad_clip", 1.0))

        _LOG.info("=" * 62)
        _LOG.info("                    TRAINING")
        _LOG.info("=" * 62)
        _LOG.info(f"  KL anneal: linear 0 → {kl_target_w} over "
                  f"{kl_warmup_epochs} epochs")
        _LOG.info(f"  Grad clip: max_norm={grad_clip}")
        _LOG.info(f"  Ortho reg weight: {ortho_w}")

        EVAL_EVERY = 2

        for epoch in range(1, cfg.training.epochs + 1):
            model.train()
            frac = (epoch - 1) / max(1, cfg.training.epochs - 1)
            T = t_start + (t_end - t_start) * frac

            # --- linear KL annealing: β ramps 0.0 → 1.0 over kl_warmup_epochs,
            # then scales the configured target weight ---
            beta_anneal = min(1.0, epoch / max(1, kl_warmup_epochs))
            kl_w        = kl_target_w * beta_anneal

            agg = {"total":0.,"rec":0.,"kl":0.,"coh":0.,"div":0.,"red":0.,"ortho":0.}
            nb, skipped = 0, 0

            for x_norm, x_raw, x_ctx in loader:
                x_input = torch.log1p(x_raw.to(device))     # log(1+count)
                x_raw   = x_raw.to(device)
                x_ctx   = x_ctx.to(device)

                recon_probs, theta, beta, mu, logvar, kl = model(
                    x_input, x_ctx=x_ctx, temperature=T,
                )

                L_rec   = reconstruction_loss(recon_probs, x_raw) / V
                L_coh   = coherence_loss(beta, pmi_dev)
                L_div   = diversity_loss(beta)
                L_red   = redundancy_loss(beta)
                L_ortho = orthogonal_regularization(model.topic_weight)

                loss = (cfg.loss_weights.recon * L_rec
                        + kl_w                   * kl
                        + cfg.loss_weights.coh   * L_coh
                        + cfg.loss_weights.div   * L_div
                        + cfg.loss_weights.red   * L_red
                        + ortho_w                * L_ortho)

                if not torch.isfinite(loss):
                    skipped += 1; opt.zero_grad(); continue

                opt.zero_grad()
                loss.backward()

                # NaN grad check
                bad = False
                for p in model.parameters():
                    if p.grad is not None and not torch.isfinite(p.grad).all():
                        bad = True; break
                if bad:
                    skipped += 1; opt.zero_grad(); continue

                # ── gradient clipping (mode-collapse safeguard) ──
                torch.nn.utils.clip_grad_norm_(model.parameters(),
                                               max_norm=grad_clip)
                opt.step()

                agg["total"] += loss.item()
                agg["rec"]   += L_rec.item()
                agg["kl"]    += kl.item()
                agg["coh"]   += L_coh.item()
                agg["div"]   += L_div.item()
                agg["red"]   += L_red.item()
                agg["ortho"] += L_ortho.item()
                nb += 1

            sched.step()
            if nb == 0:
                _LOG.error(f"All batches skipped at E{epoch}. Aborting.")
                break
            for k in agg: agg[k] /= nb

            # ---- eval every N epochs: log ALL target metrics ----
            if epoch == 1 or epoch % EVAL_EVERY == 0 or epoch == cfg.training.epochs:
                beta_eval = model.get_beta()
                had_nan = not torch.isfinite(beta_eval).all()
                beta_eval = torch.nan_to_num(beta_eval, nan=1.0 / V)

                topics = self._topics_from_beta(beta_eval, dictionary,
                                                 cfg.evaluation.top_n_words)
                top_n = cfg.evaluation.top_n_words
                metrics = {
                    "C_V":     -1.0, "C_NPMI": -1.0,
                    "U_Mass":  0.0,  "C_UCI":  0.0,
                    "Diversity": 0.0,
                    "Intra":   0.0,  "Inter": 1.0,
                }
                try:
                    metrics["C_V"]    = c_v(topics, stats, emb_dict, top_n)
                    metrics["C_NPMI"] = c_npmi(topics, stats, top_n)
                    metrics["U_Mass"] = c_umass(topics, stats, top_n)
                    metrics["C_UCI"]  = c_uci(topics, stats, top_n)
                    metrics["Diversity"] = topic_diversity(topics, top_n)
                    metrics["Intra"]  = (metrics["C_NPMI"] + 1) / 2
                    metrics["Inter"]  = inter_topic_cosine(topics, emb_dict, top_n)
                except Exception as e:
                    _LOG.warning(f"metric failed: {e}")
                    had_nan = True

                # composite score still drives early stopping & best-model tracking
                score = 0.5 * metrics["Intra"] + 0.5 * metrics["C_V"]

                _LOG.info(
                    f"E{epoch:3d} | T={T:.3f} | β={beta_anneal:.2f} | "
                    f"loss={agg['total']:+7.3f} rec={agg['rec']:6.3f} "
                    f"kl={agg['kl']:6.3f} coh={agg['coh']:+6.3f} "
                    f"ortho={agg['ortho']:.3f}"
                )
                _LOG.info(
                    f"        | C_V={metrics['C_V']:.4f}  "
                    f"NPMI={metrics['C_NPMI']:+.4f}  "
                    f"U_Mass={metrics['U_Mass']:+.3f}  "
                    f"C_UCI={metrics['C_UCI']:+.3f}  "
                    f"Div={metrics['Diversity']:.3f}  "
                    f"Intra={metrics['Intra']:.3f}  "
                    f"Inter={metrics['Inter']:.3f}"
                    + (f" | skip={skipped}" if skipped else "")
                    + ("  ⚠NaN" if had_nan else "")
                )
                history.append({
                    "epoch": epoch, **agg,
                    **{k: float(v) for k, v in metrics.items()},
                    "skipped": skipped, "had_nan": had_nan,
                })

                if not had_nan and math.isfinite(score) and score > best_score:
                    best_score = score
                    best_state = {k: v.detach().cpu().clone()
                                  for k, v in model.state_dict().items()}
                    _LOG.info(f"   ★ new best (score={score:.4f})")

                if early.update(score, had_nan):
                    _LOG.warning(f"Early stop at E{epoch}: {early.reason}")
                    break

        # ---- restore best ----
        if best_state is not None:
            model.load_state_dict(best_state)
            _LOG.info(f"Restored best (score={best_score:.4f})")

        # ---- final eval ----
        _LOG.info("=" * 62)
        _LOG.info("           FINAL EVALUATION")
        _LOG.info("=" * 62)

        beta_final = torch.nan_to_num(model.get_beta(), nan=1.0 / V)
        topics_raw = self._topics_from_beta(beta_final, dictionary,
                                             cfg.evaluation.top_n_words)

        _LOG.info("--- RAW (no quality gate) ---")
        raw_metrics = {
            "n_topics": len(topics_raw),
            "C_V":     round(c_v(topics_raw, stats, emb_dict, cfg.evaluation.top_n_words), 4),
            "C_NPMI":  round(c_npmi(topics_raw, stats, cfg.evaluation.top_n_words), 4),
            "U_Mass":  round(c_umass(topics_raw, stats, cfg.evaluation.top_n_words), 4),
            "C_UCI":   round(c_uci(topics_raw, stats, cfg.evaluation.top_n_words), 4),
            "Diversity": round(topic_diversity(topics_raw, cfg.evaluation.top_n_words), 4),
            "Intra":   round((c_npmi(topics_raw, stats, cfg.evaluation.top_n_words) + 1) / 2, 4),
            "Inter":   round(inter_topic_cosine(topics_raw, emb_dict, cfg.evaluation.top_n_words), 4),
        }
        for k, v in raw_metrics.items(): _LOG.info(f"  {k:12s}: {v}")

        _LOG.info("\n--- GATED (publication-ready subset) ---")
        topics_final = apply_quality_gate(
            topics_raw, stats, emb_dict,
            top_n=cfg.evaluation.top_n_words,
            min_npmi=cfg.evaluation.gate_min_npmi,
            min_cv=cfg.evaluation.gate_min_cv,
            max_jac=cfg.evaluation.gate_max_jaccard,
            min_keep=cfg.evaluation.gate_min_topics,
        )

        gated_metrics = {
            "n_topics": len(topics_final),
            "C_V":     round(c_v(topics_final, stats, emb_dict, cfg.evaluation.top_n_words), 4),
            "C_NPMI":  round(c_npmi(topics_final, stats, cfg.evaluation.top_n_words), 4),
            "U_Mass":  round(c_umass(topics_final, stats, cfg.evaluation.top_n_words), 4),
            "C_UCI":   round(c_uci(topics_final, stats, cfg.evaluation.top_n_words), 4),
            "Diversity": round(topic_diversity(topics_final, cfg.evaluation.top_n_words), 4),
            "Intra":   round((c_npmi(topics_final, stats, cfg.evaluation.top_n_words) + 1) / 2, 4),
            "Inter":   round(inter_topic_cosine(topics_final, emb_dict, cfg.evaluation.top_n_words), 4),
        }
        for k, v in gated_metrics.items(): _LOG.info(f"  {k:12s}: {v}")

        _LOG.info("\n📝 Final gated topics:")
        for i, t in enumerate(topics_final, 1):
            _LOG.info(f"  Topic {i:2d}: {', '.join(t)}")

        # ---- save ----
        torch.save(model.state_dict(), os.path.join(train_dir, "model_best.pt"))
        with open(os.path.join(train_dir, "topics_final.json"), "w") as f:
            json.dump({"topics_raw": topics_raw,
                       "topics_gated": topics_final,
                       "raw_metrics": raw_metrics,
                       "gated_metrics": gated_metrics,
                       "stopped_early": early.stop,
                       "stop_reason": early.reason}, f, indent=2)
        with open(os.path.join(train_dir, "training_history.json"), "w") as f:
            json.dump(history, f, indent=2)

        _LOG.info(f"artifacts -> {train_dir}")
        return gated_metrics
