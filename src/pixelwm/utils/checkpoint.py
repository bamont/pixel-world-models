"""Checkpoint saving / loading that restores the model without extra configuration."""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import torch

from pixelwm.models import WorldModel, build_model


def save_checkpoint(
    path: str | Path,
    model: WorldModel,
    optimizer: torch.optim.Optimizer | None = None,
    step: int = 0,
    extra: dict[str, Any] | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    config = model.config
    torch.save(
        {
            "model_name": model.name,
            "model_config": (
                                dataclasses.asdict(config)
                                if dataclasses.is_dataclass(config) and not isinstance(config, type)
                                else {}
                            ),
            "action_dim": model.action_dim,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict() if optimizer is not None else None,
            "step": step,
            "extra": extra or {},
        },
        path,
    )


def load_checkpoint(
    path: str | Path, device: torch.device | str = "cpu"
) -> tuple[WorldModel, dict[str, Any]]:
    """Rebuild a model from a checkpoint. Returns ``(model, checkpoint_dict)``."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = build_model(ckpt["model_name"], ckpt["action_dim"], ckpt["model_config"])
    model.load_state_dict(ckpt["model"])
    return model.to(device).eval(), ckpt
