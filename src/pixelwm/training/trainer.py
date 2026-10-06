"""Model-agnostic training loop for any :class:`~pixelwm.models.base.WorldModel`."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

from pixelwm.data.buffer import EpisodeBuffer
from pixelwm.eval.rollout import open_loop_eval
from pixelwm.models.base import WorldModel
from pixelwm.utils.checkpoint import save_checkpoint
from pixelwm.utils.logging import MetricLogger, get_logger


@dataclass
class TrainConfig:
    steps: int = 20_000
    batch_size: int = 16
    seq_len: int = 32
    lr: float = 3e-4
    grad_clip: float = 100.0
    log_every: int = 100
    eval_every: int = 1_000
    save_every: int = 5_000
    eval_context: int = 5
    eval_horizons: tuple[int, ...] = (1, 5, 15, 45)
    eval_sequences: int = 64
    eval_objects: bool = False
    seed: int = 0


class Trainer:
    """Trains a world model with Adam, gradient clipping, periodic evaluation and checkpoints."""

    def __init__(
        self,
        model: WorldModel,
        train_buffer: EpisodeBuffer,
        config: TrainConfig,
        out_dir: str | Path,
        val_buffer: EpisodeBuffer | None = None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        self.model = model
        self.train_buffer = train_buffer
        self.val_buffer = val_buffer
        self.cfg = config
        self.out_dir = Path(out_dir)
        self.extra = extra or {}
        self.optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
        self.logger = get_logger()
        self.metrics = MetricLogger(self.out_dir)
        self.rng = np.random.default_rng(config.seed)
        self.step = 0

    def evaluate(self) -> dict[str, float]:
        """Open-loop evaluation on the validation buffer (empty if none or not decodable)."""
        if self.val_buffer is None or not self.model.supports_decoding:
            return {}
        cfg = self.cfg
        results = open_loop_eval(
            self.model,
            self.val_buffer,
            horizons=tuple(cfg.eval_horizons),
            context=cfg.eval_context,
            num_sequences=cfg.eval_sequences,
            seed=cfg.seed,
            object_metrics=cfg.eval_objects,
        )
        return {f"val/{k}": v for k, v in results.items()}

    def save(self, name: str) -> Path:
        path = self.out_dir / "checkpoints" / name
        save_checkpoint(path, self.model, self.optimizer, self.step, self.extra)
        return path

    def fit(self) -> None:
        cfg = self.cfg
        self.model.train()
        self.logger.info(
            "Training %s (%.2fM parameters) on %s for %d steps",
            self.model.name,
            self.model.num_parameters() / 1e6,
            self.model.device,
            cfg.steps,
        )
        for self.step in tqdm(range(1, cfg.steps + 1), desc="train", dynamic_ncols=True):
            batch = self.train_buffer.sample(cfg.batch_size, cfg.seq_len, self.rng)
            batch = {k: v.to(self.model.device) for k, v in batch.items()}

            loss, metrics = self.model.loss(batch)
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip)
            self.optimizer.step()

            if self.step % cfg.log_every == 0 or self.step == 1:
                metrics["grad_norm"] = float(grad_norm)
                self.metrics.log(self.step, {f"train/{k}": v for k, v in metrics.items()})
            if self.step % cfg.eval_every == 0:
                val = self.evaluate()
                if val:
                    self.metrics.log(self.step, val)
                    self.logger.info(
                        "step %d | %s", self.step, " ".join(f"{k}={v:.3f}" for k, v in val.items())
                    )
            if self.step % cfg.save_every == 0:
                self.save("last.pt")
        self.save("last.pt")
