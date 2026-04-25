"""
============================================================================
 src/utils/config.py
 ---------------------------------------------------------------------------
 Typed, dotted-access configuration loader.
 Reads a YAML file and exposes every field as an attribute, so you can write
     cfg.model.topic_dim
 instead of
     cfg["model"]["topic_dim"].
============================================================================
"""
from __future__ import annotations

import os
import yaml
from typing import Any, Dict


class Config:
    """
    Dotted-access wrapper around a nested dict.

    Example
    -------
    >>> cfg = load_config("configs/twitter.yaml")
    >>> print(cfg.model.topic_dim)
    20
    >>> print(cfg.dataset.name)
    'twitter'
    """

    def __init__(self, d: Dict[str, Any]):
        for k, v in d.items():
            if isinstance(v, dict):
                setattr(self, k, Config(v))
            else:
                setattr(self, k, v)

    # ── dict-like helpers ───────────────────────────────────────────────
    def to_dict(self) -> Dict[str, Any]:
        out = {}
        for k, v in self.__dict__.items():
            out[k] = v.to_dict() if isinstance(v, Config) else v
        return out

    def __repr__(self) -> str:
        return f"Config({self.to_dict()})"


def load_config(path: str) -> Config:
    """
    Load a YAML config file and return a Config object.

    Parameters
    ----------
    path : str
        Path to the .yaml configuration file.

    Raises
    ------
    FileNotFoundError
        If the config file does not exist.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return Config(raw)
