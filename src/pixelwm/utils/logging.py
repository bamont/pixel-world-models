"""Minimal experiment logging (console + JSON lines)."""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path


def get_logger(name: str = "pixelwm") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False  # avoid duplicate lines when Hydra configures the root logger
    return logger


def finite_or_none(value: float) -> float | None:
    """JSON has no NaN/Infinity: map non-finite values to ``None`` (``null``)."""
    return value if math.isfinite(value) else None


class MetricLogger:
    """Append metrics to ``<out_dir>/metrics.jsonl``, one JSON object per call."""

    def __init__(self, out_dir: str | Path) -> None:
        self.path = Path(out_dir) / "metrics.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, step: int, metrics: dict[str, float]) -> None:
        record = {"step": step, **{k: finite_or_none(v) for k, v in metrics.items()}}
        with self.path.open("a") as f:
            f.write(json.dumps(record) + "\n")
