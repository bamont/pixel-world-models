"""Wrappers that turn generic Gymnasium environments into pixel-only environments."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from PIL import Image


class PixelObsWrapper(gym.Wrapper):  # type: ignore[type-arg]
    """Replace the observation by the rendered frame, resized to ``size x size``.

    The wrapped environment must have been created with ``render_mode="rgb_array"``.
    """

    def __init__(self, env: gym.Env, size: int = 64) -> None:  # type: ignore[type-arg]
        super().__init__(env)
        self.size = size
        self.observation_space = spaces.Box(0, 255, (size, size, 3), dtype=np.uint8)

    def _frame(self) -> np.ndarray:
        frame = self.env.render()
        if not isinstance(frame, np.ndarray):
            raise RuntimeError("The environment must be created with render_mode='rgb_array'.")
        image = Image.fromarray(frame).resize((self.size, self.size), Image.Resampling.BILINEAR)
        return np.asarray(image, dtype=np.uint8)

    def reset(self, **kwargs: Any) -> tuple[np.ndarray, dict[str, Any]]:
        _, info = self.env.reset(**kwargs)
        return self._frame(), info

    def step(self, action: Any) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        _, reward, terminated, truncated, info = self.env.step(action)
        return self._frame(), float(reward), terminated, truncated, info
