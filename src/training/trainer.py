"""
============================================================================
 src/training/trainer.py  [REWRITE - v4]
 ---------------------------------------------------------------------------
 Journal-grade training orchestrator.

 v4 HIGHLIGHTS
 -------------
 1. VAE MODE-COLLAPSE FIX
    * KL weight is linearly annealed from 0 -> target over
      `training.kl_warmup_epoch` epochs (default 20).
    * Every training step calls
         torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

 2. CONTEXTUAL TOPIC MODEL ARCHITECTURE
    * Static GloVe is removed.
    * Word-level embeddings come from a sentence-transformer
      (`all-MiniLM-L6-v2` by default).  They feed SCADecoder and the
      C_V / inter-cosine metrics.
    * Document-level embeddings from the same backbone are concatenated
      with the BoW before the encoder (see SGPEncoder, EDNeuFTMv2).

 3. ORTHOGONAL REGULARIZATION
    * `orthogonal_regularization(beta)` penalises off-diagonal cosine
      similarity of the decoder's topic-word matrix.  Weight comes from
      `loss_weights.ortho` (defaults to 1.0 if missing).

 4. FULL METRIC LOGGING EVERY EVAL
    * Logs C_V, C_NPMI, U_Mass, C_UCI, Topic Diversity, Intra, Inter.
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
    load_contextual_word_embeddings,
    load_contextual_doc_embeddings,
    DEFAULT_MODEL,
)
from ..utils.pmi               import compute_pmi_matrix
from ..evaluation.coherence_stats   import CoherenceStats
from ..evaluation.coherence_metrics import c_npmi, c_v, c_umass, c_uci
from ..evaluation.quality_gate      import apply_quality_gate
from ..evaluation.diversity_metrics import topic_diversity, inter_topic_cosine
from .losses import (
    reconstruction_loss, coherence_loss,
    diversity_loss, redundancy_loss,
    orthogonal_regularization,
)

_LOG = get_logger(__name__)


def _cfg_get(cfg_obj, key, default):
    """cfg.key with fallback that works for the dotted Config wrapper."""
    return getattr(cfg_obj, key, default)


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

    # -----------------------------------------------------------------
    # Artefact loading
    # -----------------------------------------------------------------
    def _load(self, preproc_dir):
        with open(os.path.join(preproc_dir, "clean_docs.pkl"), "rb") as f:
            docs = pickle.load(f)
        dictionary = Dictionary.load(os.path.join(preproc_dir, "dictionary.gensim"))
        bow = sparse.load_npz(os.path.join(preproc_dir, "bow_matrix.npz"))
        with open(os.path.join(preproc_dir, "reference_corpus.pkl"), "rb") as f:
            ref = pickle.load(f)

        # contextual doc embeddings (optional on disk; computed on-the-fly
        # if missing so the trainer is usable even when preprocessing was
        # run with an older version).
        ctx_path = os.path.join(preproc_dir, "doc_ctx_emb.pt")
        if os.path.exists(ctx_path):
            ctx_doc = torch.load(ctx_path, map_location="cpu")
        else:
            _LOG.warning("doc_ctx_emb.pt not found - encoding on the fly.")
            ctx_doc = load_contextual_doc_embeddings(
                docs,
                model_name=_cfg_get(self.cfg.dataset, "contextual_model",
                                    DEFAULT_MODEL),
                cache_path=ctx_path,
            )
        return docs, dictionary, bow, ref, ctx_doc

    @staticmethod
    def _topics_from_beta(beta, dictionary, top_n):
        i2w = {v: k for k, v in dictionary.token2id.items()}
        beta = torch.nan_to_num(beta, nan=0.0)
        out = []
        for k in range(beta.shape[0]):
            idx = beta[k].topk(top_n).indices.tolist()
            out.append([i2w[i] for i in idx])
        return out

    # -----------------------------------------------------------------
    # Training loop
    # -----------------------------------------------------------------
    def fit(self) -> Dict:
        cfg, device = self.cfg, self.device
        preproc_dir = os.path.join(cfg.dataset.work_dir, "preproc")
        train_dir   = os.path.join(cfg.dataset.work_dir, "train")
        os.makedirs(train_dir, exist_ok=True)

        # ---- load ----
        _LOG.info("Loading preprocessed artifacts ...")
        docs, dictionary, bow, ref_windows, ctx_doc = self._load(preproc_dir)
        V = len(dictionary)

        model_name = _cfg_get(cfg.dataset, "contextual_model", DEFAULT_MODEL)
        cache_word = os.path.join(preproc_dir, "word_ctx_emb.pt")
        word_embeds = load_contextual_word_embeddings(
            dictionary, model_name=model_name, device=device,
            cache_path=cache_word,
        )
        ctx_dim   = int(ctx_doc.shape[1])
        embed_dim = int(word_embeds.shape[1])
        _LOG.info(f"Contextual backbone: {model_name}  "
                  f"(word dim={embed_dim}, doc dim={ctx_dim})")

        pmi = compute_pmi_matrix(docs, dictionary,
                                 window=cfg.preprocessing.ref_window_size)
        pmi_dev = pmi.to(device)

        # ---- build model ----
        m = cfg.model
        model = EDNeuFTMv2(
            vocab_size=V, topic_dim=m.topic_dim,
            embed_dim=embed_dim, hidden_dim=m.hidden_dim,
            word_embeds=word_embeds, pmi_matrix=pmi,
            routing_iters=m.routing_iters,
            routing_momentum=m.routing_momentum,
            dropout=m.dropout,
            poincare_c=m.poincare_c,
            poincare_scale=m.poincare_scale,
            fisher_rao_lam=m.fisher_rao_lam,
            scad_rank=m.scad_rank,
            sinkhorn_iters=m.sinkhorn_iters,
            contextual_dim=ctx_dim,
        ).to(device)

        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        _LOG.info(f"Trainable parameters: {n_params:,}")

        # ---- data ----
        loader = make_dataloader(bow,
                                 batch_size=cfg.training.batch_size,
                                 ctx_embeds=ctx_doc,
                                 shuffle=True)

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

        es_patience = 20
        if hasattr(cfg.training, "es_patience"):
            es_patience = cfg.training.es_patience
        early = EarlyStopping(patience=es_patience, min_delta=1e-3, max_nans=2)

        best_score, best_state = -1e9, None
        history = []
        t_start, t_end = cfg.training.temp_start, cfg.training.temp_end

        # KL ANNEAL - linear 0 -> 1 over `kl_warmup_epoch` epochs (default 20)
        kl_warmup = _cfg_get(cfg.training, "kl_warmup_epoch", 20)
        max_grad_norm = _cfg_get(cfg.training, "grad_clip", 1.0)
        ortho_w  = _cfg_get(cfg.loss_weights, "ortho", 1.0)
        ortho_mode = _cfg_get(cfg.loss_weights, "ortho_mode", "cosine")

        _LOG.info("=" * 62)
        _LOG.info("                    TRAINING")
        _LOG.info("=" * 62)
        _LOG.info(f"KL warmup: linear 0 -> {cfg.loss_weights.kl} over "
                  f"{kl_warmup} epochs")
        _LOG.info(f"Grad clip max_norm = {max_grad_norm}")
        _LOG.info(f"Ortho reg weight = {ortho_w}  mode = {ortho_mode}")

        EVAL_EVERY = 2

        for epoch in range(1, cfg.training.epochs + 1):
            model.train()
            frac = (epoch - 1) / max(1, cfg.training.epochs - 1)
            T = t_start + (t_end - t_start) * frac

            # KL ANNEAL: linear 0 -> target over kl_warmup epochs
            kl_w = cfg.loss_weights.kl * min(1.0, epoch / max(1, kl_warmup))

            agg = {"total":0., "rec":0., "kl":0., "coh":0.,
                   "div":0., "red":0., "ortho":0.}
            nb, skipped = 0, 0

            for x_norm, x_raw, x_ctx in loader:
                x_input = torch.log1p(x_raw.to(device))     # log(1+count)
                x_raw_d = x_raw.to(device)
                x_ctx_d = x_ctx.to(device) if x_ctx.numel() else None

                recon_probs, theta, beta, mu, logvar, kl = model(
                    x_input, x_ctx=x_ctx_d, temperature=T)

                L_rec   = reconstruction_loss(recon_probs, x_raw_d) / V
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

                # GRADIENT CLIPPING (max_norm = 1.0 by default)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=max_grad_norm)
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
            for k in agg:
                agg[k] /= nb

            # ---- eval every N epochs ----
            if epoch == 1 or epoch % EVAL_EVERY == 0 or epoch == cfg.training.epochs:
                beta_eval = model.get_beta()
                had_nan = not torch.isfinite(beta_eval).all()
                beta_eval = torch.nan_to_num(beta_eval, nan=1.0 / V)

                topics = self._topics_from_beta(
                    beta_eval, dictionary, cfg.evaluation.top_n_words)

                try:
                    n_v    = c_npmi(topics, stats,
                                    top_n=cfg.evaluation.top_n_words)
                    c_vv   = c_v(topics, stats, emb_dict,
                                 top_n=cfg.evaluation.top_n_words)
                    u_mass = c_umass(topics, stats,
                                     top_n=cfg.evaluation.top_n_words)
                    c_uc   = c_uci(topics, stats,
                                   top_n=cfg.evaluation.top_n_words)
                    td     = topic_diversity(topics, 25)
                    intra  = (n_v + 1.0) / 2.0
                    inter  = inter_topic_cosine(
                        topics, emb_dict, cfg.evaluation.top_n_words)
                except Exception as e:
                    _LOG.warning(f"metric failed: {e}")
                    had_nan = True
                    n_v = -1.0; c_vv = 0.0; u_mass = 0.0; c_uc = 0.0
                    td = 0.0; intra = 0.0; inter = 1.0

                score = 0.5 * (n_v + 1) / 2 + 0.5 * c_vv

                _LOG.info(
                    f"E{epoch:3d} | T={T:.3f} | loss={agg['total']:+7.3f} "
                    f"| rec={agg['rec']:6.3f} | kl={agg['kl']:6.3f} "
                    f"(w={kl_w:.3f}) | ortho={agg['ortho']:+6.3f} "
                    f"| NPMI={n_v:+.4f} | C_V={c_vv:.4f} "
                    f"| U_Mass={u_mass:+.3f} | C_UCI={c_uc:+.3f} "
                    f"| Div={td:.3f} | Intra={intra:.3f} | Inter={inter:.3f}"
                    + (f" | skip={skipped}" if skipped else "")
                    + ("  WARN:NaN" if had_nan else "")
                )
                history.append({"epoch": epoch, **agg,
                                "npmi": n_v, "cv": c_vv,
                                "umass": u_mass, "cuci": c_uc,
                                "diversity": td, "intra": intra, "inter": inter,
                                "kl_weight": kl_w,
                                "skipped": skipped, "had_nan": had_nan})

                if not had_nan and math.isfinite(score) and score > best_score:
                    best_score = score
                    best_state = {k: v.detach().cpu().clone()
                                  for k, v in model.state_dict().items()}
                    _LOG.info(f"   * new best (score={score:.4f})")

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
        topics_raw = self._topics_from_beta(
            beta_final, dictionary, cfg.evaluation.top_n_words)

        def _full_metrics(topic_list):
            npmi = c_npmi(topic_list, stats, cfg.evaluation.top_n_words)
            return {
                "n_topics":  len(topic_list),
                "C_V":       round(c_v(topic_list, stats, emb_dict,
                                       cfg.evaluation.top_n_words), 4),
                "C_NPMI":    round(npmi, 4),
                "U_Mass":    round(c_umass(topic_list, stats,
                                           cfg.evaluation.top_n_words), 4),
                "C_UCI":     round(c_uci(topic_list, stats,
                                         cfg.evaluation.top_n_words), 4),
                "Diversity": round(topic_diversity(topic_list, 25), 4),
                "Intra":     round((npmi + 1.0) / 2.0, 4),
                "Inter":     round(inter_topic_cosine(
                    topic_list, emb_dict, cfg.evaluation.top_n_words), 4),
            }

        _LOG.info("--- RAW (no quality gate) ---")
        raw_metrics = _full_metrics(topics_raw)
        for k, v in raw_metrics.items(): _LOG.info(f"  {k:12s}: {v}")

        _LOG.info("--- GATED (publication-ready subset) ---")
        topics_final = apply_quality_gate(
            topics_raw, stats, emb_dict,
            top_n=cfg.evaluation.top_n_words,
            min_npmi=cfg.evaluation.gate_min_npmi,
            min_cv=cfg.evaluation.gate_min_cv,
            max_jac=cfg.evaluation.gate_max_jaccard,
            min_keep=cfg.evaluation.gate_min_topics,
        )
        gated_metrics = _full_metrics(topics_final)
        for k, v in gated_metrics.items(): _LOG.info(f"  {k:12s}: {v}")

        _LOG.info("\nFinal gated topics:")
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
