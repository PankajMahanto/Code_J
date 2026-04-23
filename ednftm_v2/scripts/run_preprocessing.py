#!/usr/bin/env python
"""
============================================================================
 scripts/run_preprocessing.py
 ---------------------------------------------------------------------------
 Entry point 1 — Preprocess the raw corpus.

 Usage
 -----
   python scripts/run_preprocessing.py --config configs/twitter.yaml
   python scripts/run_preprocessing.py --config configs/bbc.yaml
   python scripts/run_preprocessing.py --config configs/twentyng.yaml

 Output
 ------
 <work_dir>/preproc/
     clean_docs.pkl
     dictionary.gensim
     bow_matrix.npz
     reference_corpus.pkl
     preprocessing_report.json
============================================================================
"""
import os
import sys
import argparse

# add project root to PYTHONPATH so `from src...` imports work
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.config      import load_config
from src.utils.logging_utils import get_logger
from src.data.preprocessing import PreprocessingPipeline

_LOG = get_logger("preproc")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True,
                   help="Path to dataset YAML config (e.g. configs/twitter.yaml)")
    args = p.parse_args()

    cfg = load_config(args.config)
    _LOG.info(f"Dataset: {cfg.dataset.name}")
    _LOG.info(f"Input  : {cfg.dataset.input_file}")

    out_dir = os.path.join(cfg.dataset.work_dir, "preproc")
    pipeline = PreprocessingPipeline(cfg)
    pipeline.run(out_dir)
    _LOG.info("✓ preprocessing done")


if __name__ == "__main__":
    main()
