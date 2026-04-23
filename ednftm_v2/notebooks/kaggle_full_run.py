# ============================================================================
#  notebooks/kaggle_full_run.py
#  ---------------------------------------------------------------------------
#  Kaggle end-to-end runner.  Each "CELL N" comment block is one notebook cell.
#  Copy-paste into a Kaggle notebook in order.
# ============================================================================


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║ CELL 1  —  SETUP  (install + nltk downloads + upload project)            ║
# ╚══════════════════════════════════════════════════════════════════════════╝
"""
# Install deps
!pip install -q gensim==4.3.2 pyyaml sentence-transformers

# Download NLTK resources once
import nltk
for pkg in ['punkt','punkt_tab','stopwords','averaged_perceptron_tagger',
            'averaged_perceptron_tagger_eng','wordnet','omw-1.4']:
    nltk.download(pkg, quiet=True)

# Upload the ednftm_v2/ folder as a Kaggle dataset (or unzip from /kaggle/input)
# Assume it sits at /kaggle/working/ednftm_v2/
import os, sys
PROJECT_ROOT = "/kaggle/working/ednftm_v2"
sys.path.insert(0, PROJECT_ROOT)
print('Project root:', PROJECT_ROOT)
print('Contents    :', os.listdir(PROJECT_ROOT))
"""


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║ CELL 2  —  PICK DATASET  (edit paths here)                               ║
# ╚══════════════════════════════════════════════════════════════════════════╝
"""
# Choose one of: 'twitter', 'bbc', 'twentyng'
DATASET = 'twitter'

# Edit the YAML in place so paths match Kaggle locations
import yaml, os
CFG_PATH = f'{PROJECT_ROOT}/configs/{DATASET}.yaml'
with open(CFG_PATH) as f:
    cfg = yaml.safe_load(f)

# Point to your Kaggle input files (change these to your actual dataset IDs)
cfg['dataset']['input_file'] = '/kaggle/input/twitter-data/tweets_dataset.txt'
cfg['dataset']['work_dir']   = f'/kaggle/working/{DATASET}'

# v4: GloVe replaced with sentence-transformers.  Override if needed.
cfg['model']['sbert_model']  = 'sentence-transformers/all-MiniLM-L6-v2'

with open(CFG_PATH, 'w') as f:
    yaml.dump(cfg, f, default_flow_style=False)

print('Updated config:')
print(yaml.dump(cfg, default_flow_style=False))
"""


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║ CELL 3  —  RUN PREPROCESSING                                              ║
# ╚══════════════════════════════════════════════════════════════════════════╝
"""
from src.utils.config       import load_config
from src.data.preprocessing import PreprocessingPipeline
import os

cfg = load_config(CFG_PATH)
out_dir = os.path.join(cfg.dataset.work_dir, 'preproc')
PreprocessingPipeline(cfg).run(out_dir)
"""


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║ CELL 4  —  RUN TRAINING  (this is the main training cell; ~45–75 min P100)║
# ╚══════════════════════════════════════════════════════════════════════════╝
"""
from src.training.trainer import Trainer
import torch

trainer = Trainer(cfg, device=torch.device('cuda'))
results = trainer.fit()
print('\\n=== FINAL RESULTS ===')
for k, v in results.items():
    print(f'  {k}: {v}')
"""


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║ CELL 5  —  (OPTIONAL) RUN ABLATION                                        ║
# ╚══════════════════════════════════════════════════════════════════════════╝
"""
from src.training.ablation import run_ablation_suite

ablation_results = run_ablation_suite(
    cfg,
    variants=['full', 'no_sgpe', 'no_emgd', 'no_scad'],
    ablation_epochs=50,          # reduce further if time-constrained
)
"""


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║ CELL 6  —  SANITY-CHECK EVERY MODULE (fast, ~10 s)                        ║
# ╚══════════════════════════════════════════════════════════════════════════╝
"""
import torch
from src.models import (
    SGPEncoder, EMGDCapsuleRouting, SCADecoder, EDNeuFTMv2,
)

torch.manual_seed(0)
B, V, K, d, H = 4, 200, 8, 50, 128
dummy_embed = torch.randn(V, d) * 0.1
dummy_pmi   = torch.randn(V, V) * 0.5

model = EDNeuFTMv2(vocab_size=V, topic_dim=K,
                   embed_dim=d, hidden_dim=H,
                   word_embeds=dummy_embed,
                   pmi_matrix=dummy_pmi)

x = torch.rand(B, V)
recon, theta, beta, mu, logvar, kl = model(x, temperature=1.0)
print('recon :', recon.shape)
print('theta :', theta.shape, 'row sums ≈ 1:', theta.sum(-1).mean().item())
print('beta  :', beta.shape,  'row sums ≈ 1:', beta.sum(-1).mean().item())
print('kl    :', kl.item())
print('✓ Sanity check passed')
"""


# ═════════════════════════════════════════════════════════════════════════════
# End of runner.
# ═════════════════════════════════════════════════════════════════════════════
