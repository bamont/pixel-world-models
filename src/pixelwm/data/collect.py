"""Data collection."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import gymnasium as gym
import numpy as np

from pixelwm.data.buffer import Episode, EpisodeBuffer
from pixelwm.envs.factory import get_action_dim, to_action_vector

Policy = Callable[[np.ndarray], Any]


def collect_episodes(
    env: gym.Env,  # type: ignore[type-arg]
    num_episodes: int,
    policy: Policy | None = None,
    seed: int = 0,
) -> list[Episode]:
    """Roll out ``policy`` (uniform random if ``None``) and return aligned episodes."""
    space = env.action_space
    space.seed(seed)
    action_dim = get_action_dim(space)
    episodes: list[Episode] = []
    for i in range(num_episodes):
        obs, _ = env.reset(seed=seed + i)
        frames = [obs]
        actions = [np.zeros(action_dim, dtype=np.float32)]
        rewards = [0.0]
        done = False
        while not done:
            action = space.sample() if policy is None else policy(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            frames.append(obs)
            actions.append(to_action_vector(space, action))
            rewards.append(float(reward))
            done = terminated or truncated
        episodes.append(
            Episode(
                obs=np.stack(frames).astype(np.uint8),
                prev_actions=np.stack(actions).astype(np.float32),
                rewards=np.asarray(rewards, dtype=np.float32),
            )
        )
    return episodes


def collect_buffer(
    env: gym.Env,  # type: ignore[type-arg]
    num_episodes: int,
    policy: Policy | None = None,
    seed: int = 0,
) -> EpisodeBuffer:
    """Convenience wrapper returning an :class:`EpisodeBuffer`."""
    buffer = EpisodeBuffer()
    buffer.extend(collect_episodes(env, num_episodes, policy=policy, seed=seed))
    return buffer
