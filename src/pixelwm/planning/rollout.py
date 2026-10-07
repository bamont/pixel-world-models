"""Evaluate a policy in the *real* environment."""

from __future__ import annotations

import time
from typing import Any

import gymnasium as gym
import numpy as np

from pixelwm.planning.policies import Policy


def evaluate_policy(
    env: gym.Env[Any, Any],
    policy: Policy,
    episodes: int = 30,
    seed: int = 1000,
    success_distance: float = 0.1,
    record_first: bool = False,
) -> dict[str, Any]:
    """Run ``policy`` for ``episodes`` episodes and report real-environment performance.

    Episode ``i`` is reset with ``seed + i``, so different policies evaluated with the same
    ``seed`` face exactly the same start and goal positions (paired comparison).

    Returns:
        A dict with per-episode lists ``returns`` and ``final_distances`` (for paired statistics)
        and the summaries ``return_mean``, ``return_std``, ``final_distance_mean``,
        ``success_rate`` (final distance below ``success_distance``) and ``seconds_per_step``.
        If ``record_first`` is true, ``frames`` holds the first episode's observations.
    """
    returns: list[float] = []
    final_distances: list[float] = []
    frames: list[np.ndarray] = []
    steps = 0
    started = time.perf_counter()
    for i in range(episodes):
        obs, _ = env.reset(seed=seed + i)
        policy.reset()
        if record_first and i == 0:
            frames.append(obs)
        total, done = 0.0, False
        info: dict[str, Any] = {}
        while not done:
            obs, reward, terminated, truncated, info = env.step(policy(obs))
            total += float(reward)
            steps += 1
            done = terminated or truncated
            if record_first and i == 0:
                frames.append(obs)
        returns.append(total)
        final_distances.append(float(info.get("distance", np.nan)))
    elapsed = time.perf_counter() - started
    result: dict[str, Any] = {
        "returns": returns,
        "final_distances": final_distances,
        "return_mean": float(np.mean(returns)),
        "return_std": float(np.std(returns, ddof=1)) if episodes > 1 else 0.0,
        "final_distance_mean": float(np.mean(final_distances)),
        "success_rate": float(np.mean(np.asarray(final_distances) < success_distance)),
        "seconds_per_step": elapsed / max(steps, 1),
    }
    if record_first:
        result["frames"] = np.stack(frames)
    return result
