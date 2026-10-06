"""Environment factory and action-space helpers."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from pixelwm.envs.point_reach import PointReach
from pixelwm.envs.wrappers import PixelObsWrapper


def make_env(name: str, size: int = 64, max_steps: int | None = None) -> gym.Env:  # type: ignore[type-arg]
    """Create a pixel-observation environment.

    Args:
        name: ``"point_reach"`` for the built-in environment, or ``"gym-<EnvId>"``
            (e.g. ``"gym-CartPole-v1"``) for any Gymnasium environment that supports
            ``render_mode="rgb_array"``.
        size: Side length of the (square) RGB observations.
        max_steps: Optional episode length limit.
    """
    if name == "point_reach":
        return PointReach(size=size, max_steps=max_steps or 50)
    if name.startswith("gym-"):
        env = gym.make(name[len("gym-") :], render_mode="rgb_array", max_episode_steps=max_steps)
        return PixelObsWrapper(env, size=size)
    raise ValueError(f"Unknown environment '{name}'. Use 'point_reach' or 'gym-<EnvId>'.")


def get_action_dim(space: spaces.Space) -> int:  # type: ignore[type-arg]
    """Dimension of the vector used to represent actions (one-hot for discrete spaces)."""
    if isinstance(space, spaces.Discrete):
        return int(space.n)
    if isinstance(space, spaces.Box):
        return int(np.prod(space.shape))
    raise NotImplementedError(f"Unsupported action space: {space}")


def to_action_vector(space: spaces.Space, action: Any) -> np.ndarray:  # type: ignore[type-arg]
    """Convert an environment action into a float32 vector of size ``get_action_dim``."""
    if isinstance(space, spaces.Discrete):
        vector = np.zeros(int(space.n), dtype=np.float32)
        vector[int(action)] = 1.0
        return vector
    return np.asarray(action, dtype=np.float32).reshape(-1)
