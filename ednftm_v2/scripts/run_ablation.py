#!/usr/bin/env python
"""
============================================================================
 scripts/run_ablation.py
 ---------------------------------------------------------------------------
 Entry point 3 — Run the 4-variant ablation study for paper Table IV.

 Usage
 -----
   python scripts/run_ablation.py --config configs/twitter.yaml

 Output
 ------
 <work_dir>/train/ablation_results.json
============================================================================
"""
import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.config          import load_config
from src.utils.logging_utils   import get_logger
from src.training.ablation     import run_ablation_suite

_LOG = get_logger("ablation")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True,
                   help="Path to dataset YAML config")
    p.add_argument("--epochs", type=int, default=50,
                   help="Reduced epochs per variant (default: 50)")
    p.add_argument("--variants", nargs="+",
                   default=["full", "no_sgpe", "no_emgd", "no_scad"],
                   help="Which variants to run")
    args = p.parse_args()

    cfg = load_config(args.config)
    _LOG.info(f"Dataset : {cfg.dataset.name}")
    _LOG.info(f"Variants: {args.variants}")
    _LOG.info(f"Epochs  : {args.epochs}")

    run_ablation_suite(cfg, variants=args.variants,
                       ablation_epochs=args.epochs)
    _LOG.info("✓ ablation done")


if __name__ == "__main__":
    main()
