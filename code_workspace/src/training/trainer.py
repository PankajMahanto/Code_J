"""
============================================================================
 src/training/trainer.py  [REWRITE — v3]
 ---------------------------------------------------------------------------
 Working training orchestrator.

 v3 DESIGN DECISIONS
 ───────────────────
 1. Use RAW counts as encoder input — NOT L1-normalized.
    BoW counts give the encoder real magnitude information.  The
    previous L1-norm killed most of the signal for short-text docs
    (avg len 4.48 → each token got a huge weight after division).

 2. Scale reconstruction loss by 1/vocab_size so it's on the same
    order of magnitude as other losses.

 3. KL warm-up gentler: 0 → target_weight over 50 epochs (linear).

 4. Evaluate every 2 epochs (faster feedback for debugging).

 5. Save ALL checkpoints for the first 20 epochs — helps diagnose.
============================================================================
"""
from __future__ import annotations

import os, json, pickle, math
from typing import Dict, List
import numpy as np
import torch
from scipy import sparse
from gensim.corpora import Dictionary

from ..models                 import EDNeuFTMv2
from ..data                    import make_dataloader
from ..utils.logging_utils     import get_logger
from ..utils.glove_loader      import load_glove_aligned
from ..utils.pmi               import compute_pmi_matrix
from ..evaluation.coherence_stats   import CoherenceStats
from ..evaluation.coherence_metrics import c_npmi, c_v, c_umass, c_uci
from ..evaluation.quality_gate      import apply_quality_gate
from ..evaluation.diversity_metrics import topic_diversity, inter_topic_cosine
from .losses import (
    reconstruction_loss, coherence_loss,
    diversity_loss, redundancy_loss,
)

_LOG = get_logger(__name__)


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

        # ---- load ----
        _LOG.info("Loading preprocessed artifacts ...")
        docs, dictionary, bow, ref_windows = self._load(preproc_dir)
        V = len(dictionary)

        word_embeds = load_glove_aligned(cfg.dataset.glove_path, dictionary,
                                         dim=cfg.model.embed_dim)
        pmi = compute_pmi_matrix(docs, dictionary,
                                 window=cfg.preprocessing.ref_window_size)
        pmi_dev = pmi.to(device)

        # ---- build model ----
        m = cfg.model
        model = EDNeuFTMv2(
            vocab_size=V, topic_dim=m.topic_dim,
            embed_dim=m.embed_dim, hidden_dim=m.hidden_dim,
            word_embeds=word_embeds, pmi_matrix=pmi,
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
        loader = make_dataloader(bow, batch_size=cfg.training.batch_size, shuffle=True)

        opt = torch.optim.AdamW(model.parameters(),
                                lr=cfg.training.lr,
                                weight_decay=cfg.training.weight_decay,
                                eps=1e-7)

        # Cosine LR but with a minimum (don't go to 0)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=cfg.training.epochs, eta_min=cfg.training.lr * 0.1)

        # ---- coherence stats (once) ----
        vocab_set = set(dictionary.token2id.keys())
        stats     = CoherenceStats(ref_windows, vocab=vocab_set)
        emb_dict  = {w: word_embeds[dictionary.token2id[w]].numpy()
                     for w in vocab_set}

        early = EarlyStopping(patience=cfg.training.get("es_patience", 20) if hasattr(cfg.training, "get") else 20,
                              min_delta=1e-3, max_nans=2)

        best_score, best_state = -1e9, None
        history = []
        t_start, t_end = cfg.training.temp_start, cfg.training.temp_end

        _LOG.info("=" * 62)
        _LOG.info("                    TRAINING")
        _LOG.info("=" * 62)

        EVAL_EVERY = 2    # evaluate every 2 epochs for faster feedback

        for epoch in range(1, cfg.training.epochs + 1):
            model.train()
            frac = (epoch - 1) / max(1, cfg.training.epochs - 1)
            T = t_start + (t_end - t_start) * frac   # linear temperature (was expo)

            # GENTLER KL warm-up: linear 0 → target over warmup_epoch
            kl_w = cfg.loss_weights.kl * min(1.0, epoch / cfg.training.kl_warmup_epoch)

            agg = {"total":0.,"rec":0.,"kl":0.,"coh":0.,"div":0.,"red":0.}
            nb, skipped = 0, 0

            for x_norm, x_raw in loader:
                # Use raw counts for encoder input — short-text needs signal
                # We'll still log-normalize slightly to keep magnitudes reasonable
                x_input = torch.log1p(x_raw.to(device))     # log(1+count)
                x_raw   = x_raw.to(device)

                recon_probs, theta, beta, mu, logvar, kl = model(x_input, temperature=T)

                L_rec = reconstruction_loss(recon_probs, x_raw) / V
                L_coh = coherence_loss(beta, pmi_dev)
                L_div = diversity_loss(beta)
                L_red = redundancy_loss(beta)

                loss = (cfg.loss_weights.recon * L_rec
                        + kl_w                 * kl
                        + cfg.loss_weights.coh * L_coh
                        + cfg.loss_weights.div * L_div
                        + cfg.loss_weights.red * L_red)

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

                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()

                agg["total"] += loss.item()
                agg["rec"]   += L_rec.item()
                agg["kl"]    += kl.item()
                agg["coh"]   += L_coh.item()
                agg["div"]   += L_div.item()
                agg["red"]   += L_red.item()
                nb += 1

            sched.step()
            if nb == 0:
                _LOG.error(f"All batches skipped at E{epoch}. Aborting.")
                break
            for k in agg: agg[k] /= nb

            # ---- eval every N epochs ----
            if epoch == 1 or epoch % EVAL_EVERY == 0 or epoch == cfg.training.epochs:
                beta_eval = model.get_beta()
                had_nan = not torch.isfinite(beta_eval).all()
                beta_eval = torch.nan_to_num(beta_eval, nan=1.0 / V)

                topics = self._topics_from_beta(beta_eval, dictionary,
                                                 cfg.evaluation.top_n_words)
                try:
                    n_v  = c_npmi(topics, stats, top_n=cfg.evaluation.top_n_words)
                    c_vv = c_v(topics, stats, emb_dict, top_n=cfg.evaluation.top_n_words)
                except Exception as e:
                    _LOG.warning(f"metric failed: {e}")
                    had_nan = True; n_v, c_vv = -1.0, 0.0

                score = 0.5 * (n_v + 1) / 2 + 0.5 * c_vv

                _LOG.info(
                    f"E{epoch:3d} | T={T:.3f} | loss={agg['total']:+7.3f} "
                    f"| rec={agg['rec']:6.3f} | kl={agg['kl']:6.3f} "
                    f"| coh={agg['coh']:+7.3f} | div={agg['div']:+6.3f} "
                    f"| NPMI={n_v:+.4f} | C_V={c_vv:.4f}"
                    + (f" | skip={skipped}" if skipped else "")
                    + (" | ⚠NaN" if had_nan else "")
                )
                history.append({"epoch": epoch, **agg,
                                "npmi": n_v, "cv": c_vv,
                                "skipped": skipped, "had_nan": had_nan})

                if not had_nan and math.isfinite(score) and score > best_score:
                    best_score = score
                    best_state = {k: v.detach().cpu().clone()
                                  for k, v in model.state_dict().items()}
                    _LOG.info(f"   ★ new best (score={score:.4f})")

                if early.update(score, had_nan):
                    _LOG.warning(f"🛑 Early stop at E{epoch}: {early.reason}")
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

        # Report BOTH raw and gated for full transparency
        _LOG.info("--- RAW (no quality gate) ---")
        raw_metrics = {
            "n_topics": len(topics_raw),
            "C_V":     round(c_v(topics_raw, stats, emb_dict, cfg.evaluation.top_n_words), 4),
            "C_NPMI":  round(c_npmi(topics_raw, stats, cfg.evaluation.top_n_words), 4),
            "U_Mass":  round(c_umass(topics_raw, stats, cfg.evaluation.top_n_words), 4),
            "C_UCI":   round(c_uci(topics_raw, stats, cfg.evaluation.top_n_words), 4),
            "Diversity": round(topic_diversity(topics_raw, 25), 4),
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
            "Diversity": round(topic_diversity(topics_final, 25), 4),
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

        _LOG.info(f"✓ artifacts → {train_dir}")
        return gated_metrics
