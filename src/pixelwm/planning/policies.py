"""Reference policies used to put planner results in context.

Both baselines follow the same interface as the planners: ``reset()`` at the start of an episode
and ``policy(obs) -> action`` at every step.
"""

from __future__ import annotations

from typing import Any, Protocol

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class Policy(Protocol):
    """Anything that maps a pixel observation to an action."""

    def reset(self) -> None: ...

    def __call__(self, obs: np.ndarray) -> np.ndarray: ...


class RandomPolicy:
    """Uniform random actions in a continuous (Box) action space: the lower reference."""

    def __init__(self, action_space: spaces.Box, seed: int = 0) -> None:
        self.low = action_space.low
        self.high = action_space.high
        self.rng = np.random.default_rng(seed)

    def reset(self) -> None:
        pass

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        return self.rng.uniform(self.low, self.high).astype(np.float32)


class OraclePolicy:
    """Privileged controller for ``point_reach``: the upper reference.

    It reads the simulator state (position, velocity, goal), which a pixel-only agent cannot do,
    and steers the dot to the goal with a proportional controller that accounts for momentum.
    """

    def __init__(self, env: gym.Env[Any, Any], gain: float = 1.0) -> None:
        self.env: Any = env.unwrapped
        if not hasattr(self.env, "_goal"):
            raise TypeError("OraclePolicy only supports the point_reach environment.")
        self.gain = gain

    def reset(self) -> None:
        pass

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        e = self.env
        desired_velocity = self.gain * (e._goal - e._pos)
        action = (desired_velocity - e.momentum * e._vel) / ((1.0 - e.momentum) * e.max_speed)
        return np.clip(action, -1.0, 1.0).astype(np.float32)
