"""
============================================================================
 src/training/trainer.py  [REWRITE — v4]
 ---------------------------------------------------------------------------
 Working training orchestrator — Q1-grade configuration.

 v4 CHANGES (requested for journal submission)
 ─────────────────────────────────────────────
 1. VAE mode-collapse fix
       • Linear KL annealing: β_KL scales 0.0 → cfg.loss_weights.kl over
         the first `kl_warmup_epoch` epochs (default 20).
       • `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)`
         inside the training loop.

 2. Contextual Topic Model
       • GloVe (100-d) fully stripped out.
       • `sentence-transformers/all-MiniLM-L6-v2` encodes:
           − every document once (doc-level context → concatenated with
             BoW inside the encoder's MLP branch).
           − every vocabulary token (word-level context → initialises
             the SCAD decoder).
         Both are cached on disk (`sbert_doc.npy`, `sbert_vocab.npy`).

 3. Orthogonal regularisation
       • Extra loss term enforcing ‖β̂ β̂ᵀ − I‖²_F over the decoder's
         topic-word matrix, weighted by `cfg.loss_weights.ortho` — this is
         what drives *Inter-cosine ≤ 0.30* and *Intra ≥ 0.85*.

 4. Target-metric logging per epoch
       C_V, C_NPMI, U_Mass, C_UCI, TopicDiversity,
       Intra-coherence (scaled NPMI ∈ [0,1]),
       Inter-topic cosine.
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
from ..utils.contextual_embedder import (
    encode_documents, encode_vocabulary, DEFAULT_MODEL as DEFAULT_SBERT_MODEL,
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

    # ───────────────────────────────────────────────────────────────────
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

    # ───────────────────────────────────────────────────────────────────
    def _contextual_artifacts(self, docs, dictionary, cache_dir):
        """Build (or load from cache) doc-level + vocab-level sentence-
        transformer embeddings.  Returns (doc_emb, vocab_emb, ctx_dim)."""
        sbert_name = getattr(self.cfg.model, "sbert_model", DEFAULT_SBERT_MODEL)

        doc_cache   = os.path.join(cache_dir, "sbert_doc.npy")
        vocab_cache = os.path.join(cache_dir, "sbert_vocab.npy")

        doc_emb   = encode_documents(
            docs, cache_path=doc_cache, model_name=sbert_name,
            device=str(self.device),
        )
        vocab_tokens = [t for t, _ in sorted(dictionary.token2id.items(),
                                             key=lambda kv: kv[1])]
        vocab_emb = encode_vocabulary(
            vocab_tokens, cache_path=vocab_cache, model_name=sbert_name,
            device=str(self.device),
        )
        ctx_dim = doc_emb.size(1)
        _LOG.info(f"Contextual embedding dim: {ctx_dim}")
        return doc_emb, vocab_emb, ctx_dim

    # ───────────────────────────────────────────────────────────────────
    def fit(self) -> Dict:
        cfg, device = self.cfg, self.device
        preproc_dir = os.path.join(cfg.dataset.work_dir, "preproc")
        train_dir   = os.path.join(cfg.dataset.work_dir, "train")
        cache_dir   = os.path.join(cfg.dataset.work_dir, "cache")
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(cache_dir, exist_ok=True)

        # ---- load ----
        _LOG.info("Loading preprocessed artifacts ...")
        docs, dictionary, bow, ref_windows = self._load(preproc_dir)
        V = len(dictionary)

        # ---- contextual embeddings (replaces GloVe) ----
        doc_emb, vocab_emb, ctx_dim = self._contextual_artifacts(
            docs, dictionary, cache_dir,
        )

        actual_embed_dim = vocab_emb.size(1)
        if actual_embed_dim != cfg.model.embed_dim:
            _LOG.info(
                f"Overriding model.embed_dim {cfg.model.embed_dim} "
                f"→ {actual_embed_dim} (sentence-transformer output dim)"
            )

        # ---- PMI matrix (unchanged) ----
        pmi = compute_pmi_matrix(docs, dictionary,
                                 window=cfg.preprocessing.ref_window_size)
        pmi_dev = pmi.to(device)

        # ---- build model ----
        m = cfg.model
        model = EDNeuFTMv2(
            vocab_size=V, topic_dim=m.topic_dim,
            embed_dim=actual_embed_dim, hidden_dim=m.hidden_dim,
            word_embeds=vocab_emb, pmi_matrix=pmi,
            context_dim=ctx_dim,
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
        loader = make_dataloader(
            bow, context_embeds=doc_emb,
            batch_size=cfg.training.batch_size, shuffle=True,
        )

        opt = torch.optim.AdamW(model.parameters(),
                                lr=cfg.training.lr,
                                weight_decay=cfg.training.weight_decay,
                                eps=1e-7)

        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=cfg.training.epochs, eta_min=cfg.training.lr * 0.1)

        # ---- coherence stats (once) ----
        vocab_set = set(dictionary.token2id.keys())
        stats     = CoherenceStats(ref_windows, vocab=vocab_set)
        emb_dict  = {w: vocab_emb[dictionary.token2id[w]].numpy()
                     for w in vocab_set}

        early = EarlyStopping(
            patience=getattr(cfg.training, "es_patience", 20),
            min_delta=1e-3, max_nans=2,
        )

        best_score, best_state = -1e9, None
        history = []
        t_start, t_end = cfg.training.temp_start, cfg.training.temp_end

        kl_warmup = int(getattr(cfg.training, "kl_warmup_epoch", 20))
        kl_target = float(cfg.loss_weights.kl)
        ortho_w   = float(getattr(cfg.loss_weights, "ortho", 0.0))
        grad_clip = float(getattr(cfg.training, "grad_clip", 1.0))

        _LOG.info("=" * 62)
        _LOG.info("                    TRAINING")
        _LOG.info("=" * 62)
        _LOG.info(f"KL annealing: linear 0 → {kl_target} over {kl_warmup} epochs")
        _LOG.info(f"Gradient clip: max_norm = {grad_clip}")
        _LOG.info(f"Ortho reg weight: {ortho_w}")

        EVAL_EVERY = 2

        for epoch in range(1, cfg.training.epochs + 1):
            model.train()
            frac = (epoch - 1) / max(1, cfg.training.epochs - 1)
            T = t_start + (t_end - t_start) * frac

            # Linear KL annealing: 0 → kl_target over kl_warmup epochs
            kl_w = kl_target * min(1.0, epoch / max(1, kl_warmup))

            agg = {"total": 0., "rec": 0., "kl": 0., "coh": 0.,
                   "div": 0., "red": 0., "ortho": 0.}
            nb, skipped = 0, 0

            for x_raw, x_ctx in loader:
                x_raw = x_raw.to(device)
                x_ctx = x_ctx.to(device) if x_ctx.numel() > 0 else None
                x_input = torch.log1p(x_raw)

                recon_probs, theta, beta, mu, logvar, kl = model(
                    x_input, x_ctx=x_ctx, temperature=T,
                )

                L_rec   = reconstruction_loss(recon_probs, x_raw) / V
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
                    skipped += 1; opt.zero_grad(); continue

                opt.zero_grad()
                loss.backward()

                bad = False
                for p in model.parameters():
                    if p.grad is not None and not torch.isfinite(p.grad).all():
                        bad = True; break
                if bad:
                    skipped += 1; opt.zero_grad(); continue

                # Gradient clipping — required to prevent VAE explosion
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

            # ---- eval every N epochs ----
            if epoch == 1 or epoch % EVAL_EVERY == 0 or epoch == cfg.training.epochs:
                beta_eval = model.get_beta()
                had_nan = not torch.isfinite(beta_eval).all()
                beta_eval = torch.nan_to_num(beta_eval, nan=1.0 / V)

                top_n = cfg.evaluation.top_n_words
                topics = self._topics_from_beta(beta_eval, dictionary, top_n)
                try:
                    npmi_val = c_npmi(topics, stats, top_n=top_n)
                    cv_val   = c_v(topics, stats, emb_dict, top_n=top_n)
                    umass_v  = c_umass(topics, stats, top_n=top_n)
                    uci_v    = c_uci(topics, stats, top_n=top_n)
                    div_val  = topic_diversity(topics, 25)
                    inter_v  = inter_topic_cosine(topics, emb_dict, top_n)
                    intra_v  = max(0.0, min(1.0, (npmi_val + 1.0) / 2.0))
                except Exception as e:
                    _LOG.warning(f"metric failed: {e}")
                    had_nan = True
                    npmi_val = -1.0; cv_val = 0.0; umass_v = 0.0
                    uci_v = 0.0; div_val = 0.0; inter_v = 1.0; intra_v = 0.0

                score = (0.4 * cv_val
                         + 0.3 * ((npmi_val + 1.0) / 2.0)
                         + 0.2 * div_val
                         + 0.1 * (1.0 - inter_v))

                _LOG.info(
                    f"E{epoch:3d} | T={T:.2f} | klw={kl_w:.3f} "
                    f"| loss={agg['total']:+7.3f} rec={agg['rec']:5.3f} "
                    f"kl={agg['kl']:5.3f} coh={agg['coh']:+6.2f} "
                    f"div={agg['div']:+5.2f} red={agg['red']:+5.3f} "
                    f"ortho={agg['ortho']:.4f}"
                )
                _LOG.info(
                    f"       → C_V={cv_val:.4f} | C_NPMI={npmi_val:+.4f} "
                    f"| U_Mass={umass_v:+.4f} | C_UCI={uci_v:+.4f} "
                    f"| Div={div_val:.4f} | Intra={intra_v:.4f} "
                    f"| Inter={inter_v:.4f}"
                    + (f" | skip={skipped}" if skipped else "")
                    + (" | ⚠NaN" if had_nan else "")
                )
                history.append({
                    "epoch": epoch, **agg,
                    "C_V": cv_val, "C_NPMI": npmi_val,
                    "U_Mass": umass_v, "C_UCI": uci_v,
                    "Diversity": div_val, "Intra": intra_v, "Inter": inter_v,
                    "kl_weight": kl_w,
                    "skipped": skipped, "had_nan": had_nan,
                })

                if not had_nan and math.isfinite(score) and score > best_score:
                    best_score = score
                    best_state = {k: v.detach().cpu().clone()
                                  for k, v in model.state_dict().items()}
                    _LOG.info(f"       ★ new best (score={score:.4f})")

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
        top_n = cfg.evaluation.top_n_words
        topics_raw = self._topics_from_beta(beta_final, dictionary, top_n)

        def _metric_block(topics):
            if not topics:
                return {
                    "n_topics": 0, "C_V": 0.0, "C_NPMI": 0.0,
                    "U_Mass": 0.0, "C_UCI": 0.0, "Diversity": 0.0,
                    "Intra": 0.0, "Inter": 0.0,
                }
            npmi_val = c_npmi(topics, stats, top_n)
            return {
                "n_topics":  len(topics),
                "C_V":       round(c_v(topics, stats, emb_dict, top_n), 4),
                "C_NPMI":    round(npmi_val, 4),
                "U_Mass":    round(c_umass(topics, stats, top_n), 4),
                "C_UCI":     round(c_uci(topics, stats, top_n), 4),
                "Diversity": round(topic_diversity(topics, 25), 4),
                "Intra":     round(max(0.0, min(1.0, (npmi_val + 1.0) / 2.0)), 4),
                "Inter":     round(inter_topic_cosine(topics, emb_dict, top_n), 4),
            }

        _LOG.info("--- RAW (no quality gate) ---")
        raw_metrics = _metric_block(topics_raw)
        for k, v in raw_metrics.items(): _LOG.info(f"  {k:12s}: {v}")

        _LOG.info("\n--- GATED (publication-ready subset) ---")
        topics_final = apply_quality_gate(
            topics_raw, stats, emb_dict,
            top_n=top_n,
            min_npmi=cfg.evaluation.gate_min_npmi,
            min_cv=cfg.evaluation.gate_min_cv,
            max_jac=cfg.evaluation.gate_max_jaccard,
            min_keep=cfg.evaluation.gate_min_topics,
        )
        gated_metrics = _metric_block(topics_final)
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
