"""A tiny, dependency-free pixel environment used for fast experiments and tests."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class PointReach(gym.Env[np.ndarray, np.ndarray]):
    """Move a red dot onto a green square.

    The agent only observes a ``size x size`` RGB image. Its velocity is *not* visible in a
    single frame (the dot has momentum), so a world model must integrate information over
    time to predict the future accurately.

    * Action: 2D velocity command in ``[-1, 1]^2``.
    * Reward: negative Euclidean distance between the dot and the goal.
    * Episodes are truncated after ``max_steps`` steps.
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 10}

    def __init__(
        self,
        size: int = 64,
        max_steps: int = 50,
        max_speed: float = 0.15,
        momentum: float = 0.5,
        render_mode: str | None = "rgb_array",
    ) -> None:
        super().__init__()
        self.size = size
        self.max_steps = max_steps
        self.max_speed = max_speed
        self.momentum = momentum
        self.render_mode = render_mode
        self.observation_space = spaces.Box(0, 255, (size, size, 3), dtype=np.uint8)
        self.action_space = spaces.Box(-1.0, 1.0, (2,), dtype=np.float32)
        self._pos = np.zeros(2)
        self._vel = np.zeros(2)
        self._goal = np.zeros(2)
        self._t = 0

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        self._pos = self.np_random.uniform(-0.8, 0.8, size=2)
        self._goal = self.np_random.uniform(-0.8, 0.8, size=2)
        self._vel = np.zeros(2)
        self._t = 0
        return self._render(), {}

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        self._vel = self.momentum * self._vel + (1.0 - self.momentum) * self.max_speed * action
        self._pos = np.clip(self._pos + self._vel, -1.0, 1.0)
        self._t += 1
        distance = float(np.linalg.norm(self._pos - self._goal))
        truncated = self._t >= self.max_steps
        return self._render(), -distance, False, truncated, {"distance": distance}

    def render(self) -> np.ndarray:
        return self._render()

    # ------------------------------------------------------------------ helpers
    def _to_pixels(self, point: np.ndarray) -> tuple[int, int]:
        px = np.clip(np.round((point + 1.0) / 2.0 * (self.size - 1)), 0, self.size - 1)
        return int(px[0]), int(px[1])

    def _render(self) -> np.ndarray:
        s = self.size
        img = np.full((s, s, 3), 20, dtype=np.uint8)
        gx, gy = self._to_pixels(self._goal)
        r_goal = max(2, s // 20)
        img[max(gy - r_goal, 0) : gy + r_goal + 1, max(gx - r_goal, 0) : gx + r_goal + 1] = (
            40,
            200,
            80,
        )
        ax, ay = self._to_pixels(self._pos)
        yy, xx = np.ogrid[:s, :s]
        mask = (xx - ax) ** 2 + (yy - ay) ** 2 <= max(2, s // 20) ** 2
        img[mask] = (230, 60, 60)
        return img
