"""
============================================================================
 src/utils/logging_utils.py
 ---------------------------------------------------------------------------
 Colorful, consistent logging for the whole pipeline.
============================================================================
"""
from __future__ import annotations
import logging
import sys


def get_logger(name: str = "ednftm", level: int = logging.INFO) -> logging.Logger:
    """Return a singleton logger with a clean stream handler."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(level)
    handler = logging.StreamHandler(sys.stdout)
    fmt = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    handler.setFormatter(fmt)
    logger.addHandler(handler)
    logger.propagate = False
    return logger
