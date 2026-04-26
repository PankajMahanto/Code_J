#!/usr/bin/env python
"""
============================================================================
 scripts/run_training.py
 ---------------------------------------------------------------------------
 Entry point 2 — Train EDNeuFTM-v2 on preprocessed data.

 Usage
 -----
   python scripts/run_training.py --config configs/twitter.yaml

 Prerequisite
 ------------
 Run scripts/run_preprocessing.py first for the same config.

 Output
 ------
 <work_dir>/train/
     model_best.pt
     topics_final.json
     training_history.json
============================================================================
"""
import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.config       import load_config
from src.utils.logging_utils import get_logger
from src.training.trainer    import Trainer

_LOG = get_logger("train")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True,
                   help="Path to dataset YAML config")
    args = p.parse_args()

    cfg = load_config(args.config)
    _LOG.info(f"Dataset: {cfg.dataset.name}")
    _LOG.info(f"Topics : {cfg.model.topic_dim}")

    trainer = Trainer(cfg)
    trainer.fit()
    _LOG.info("✓ training done")


if __name__ == "__main__":
    main()
