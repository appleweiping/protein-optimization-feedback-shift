"""Logging helpers for the reproducible execution shell."""

from __future__ import annotations

import logging
from pathlib import Path


def build_logger(
    name: str,
    log_dir: Path,
    experiment_id: str,
    level: str = "INFO",
) -> logging.Logger:
    """Create a logger that writes both to console and to a run-local file."""
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter(
        fmt=f"%(asctime)s | %(levelname)s | {experiment_id} | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(log_dir / "execution.log", encoding="utf-8")
    file_handler.setLevel(logger.level)
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logger.level)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger
