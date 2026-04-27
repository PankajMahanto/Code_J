# EDNeuFTM-v2: Enhanced Deep NeuroFusion Topic Modeling

> Three novel contributions for short-text topic modeling, targeting IEEE Transactions (Q1).

**Novelties**
- **SGP-E** — *Spectral Graph-Infused Hierarchical Poincaré Encoder*: semantic graph convolution + hyperbolic Poincaré latent manifold + Fisher–Rao information-geometry KL.
- **EMGD-CR** — *Entropic Momentum Graph-Diffused Capsule Routing*: momentum on routing logits + annealed entropic temperature + learnable topic-topic graph diffusion + Lorentzian squash.
- **SCAD** — *Sinkhorn Concept-Anchor Decoder*: learnable concept anchors + low-rank Mahalanobis cost + entropic optimal transport (Sinkhorn).

## Directory layout
```
ednftm_v2/
├── configs/                     # dataset-specific hyperparameter configs
│   ├── twitter.yaml
│   ├── bbc.yaml
│   └── twentyng.yaml
├── src/
│   ├── data/                    # preprocessing + dataset classes
│   │   ├── __init__.py
│   │   ├── preprocessing.py
│   │   ├── reference_corpus.py
│   │   └── dataset.py
│   ├── modules/                 # low-level reusable building blocks
│   │   ├── __init__.py
│   │   ├── poincare.py          # PoincareBall geometry
│   │   ├── spectral_gcn.py      # SpectralGraphConv
│   │   └── fisher_rao.py        # Fisher-Rao KL
│   ├── models/                  # the three novelties + full model
│   │   ├── __init__.py
│   │   ├── sgpe_encoder.py      # Novelty 1
│   │   ├── emgdcr_routing.py    # Novelty 3
│   │   ├── scad_decoder.py      # Novelty 2
│   │   ├── ablation_modules.py  # vanilla baselines for ablation
│   │   └── ednftm.py            # full model combining all 3
│   ├── training/
│   │   ├── __init__.py
│   │   ├── losses.py
│   │   ├── trainer.py
│   │   └── ablation.py
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── coherence_stats.py
│   │   ├── coherence_metrics.py
│   │   ├── diversity_metrics.py
│   │   └── quality_gate.py
│   └── utils/
│       ├── __init__.py
│       ├── contextual_loader.py   # sentence-transformer word + doc embeds
│       ├── pmi.py
│       ├── config.py
│       └── logging_utils.py
├── scripts/
│   ├── run_preprocessing.py     # entry point 1
│   ├── run_training.py          # entry point 2
│   └── run_ablation.py          # entry point 3
└── notebooks/
    └── kaggle_full_run.ipynb    # end-to-end Kaggle notebook
```

## Quick start
```bash
# 1. Preprocess raw corpus
python scripts/run_preprocessing.py --config configs/twitter.yaml

# 2. Train the full model
python scripts/run_training.py --config configs/twitter.yaml

# 3. Run ablation study (for paper Table IV)
python scripts/run_ablation.py --config configs/twitter.yaml
```

## Target metrics (for Q1 journal)
| Metric          | Target        |
|-----------------|---------------|
| C_V             | ≥ 0.95        |
| C_NPMI          | ≥ 0.76        |
| U_Mass          | ∈ [-4, -2]    |
| C_UCI           | ≥ -1.2        |
| Topic Diversity | ≥ 0.95        |
| Intra-coherence | ≥ 0.95        |
| Inter-coherence | ≤ 0.30        |
