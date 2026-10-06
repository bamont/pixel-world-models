"""Model registry. Add new world models here so that scripts can build them by name."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from pixelwm.models.base import State, WorldModel
from pixelwm.models.rssm import RSSM, RSSMConfig

ModelBuilder = Callable[[int, Mapping[str, Any]], WorldModel]

MODEL_REGISTRY: dict[str, ModelBuilder] = {
    "rssm": lambda action_dim, cfg: RSSM(action_dim, RSSMConfig(**cfg)),
}


def build_model(name: str, action_dim: int, config: Mapping[str, Any] | None = None) -> WorldModel:
    """Instantiate a registered world model from a plain config mapping."""
    if name not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model '{name}'. Available: {sorted(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[name](action_dim, config or {})


__all__ = ["MODEL_REGISTRY", "RSSM", "RSSMConfig", "State", "WorldModel", "build_model"]
