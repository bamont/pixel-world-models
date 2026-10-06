"""Episode storage and sequence sampling.

Alignment convention (shared by every model in this repository)
---------------------------------------------------------------
For an episode with ``T`` frames, index ``t`` holds:

* ``obs[t]``          - the observation at time ``t``;
* ``prev_actions[t]`` - the action that *led to* ``obs[t]`` (zeros at ``t = 0``);
* ``rewards[t]``      - the reward received when *arriving at* ``obs[t]`` (0 at ``t = 0``).

A world model therefore consumes ``(obs[t], prev_actions[t])`` at every step and is
trained to predict ``obs[t]`` and ``rewards[t]``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


@dataclass
class Episode:
    """A single trajectory following the alignment convention above."""

    obs: np.ndarray  # (T, H, W, 3) uint8
    prev_actions: np.ndarray  # (T, A) float32
    rewards: np.ndarray  # (T,) float32

    def __post_init__(self) -> None:
        if not (len(self.obs) == len(self.prev_actions) == len(self.rewards)):
            raise ValueError("obs, prev_actions and rewards must have the same length.")

    def __len__(self) -> int:
        return len(self.obs)


class EpisodeBuffer:
    """In-memory FIFO buffer of episodes with uniform sub-sequence sampling."""

    def __init__(self, capacity: int = 1_000_000) -> None:
        self.capacity = capacity
        self._episodes: list[Episode] = []
        self._steps = 0

    # ------------------------------------------------------------------ basics
    def __len__(self) -> int:
        """Total number of stored frames."""
        return self._steps

    @property
    def num_episodes(self) -> int:
        return len(self._episodes)

    @property
    def episodes(self) -> list[Episode]:
        return self._episodes

    @property
    def action_dim(self) -> int:
        return int(self._episodes[0].prev_actions.shape[-1])

    def add(self, episode: Episode) -> None:
        self._episodes.append(episode)
        self._steps += len(episode)
        while self._steps > self.capacity and len(self._episodes) > 1:
            self._steps -= len(self._episodes.pop(0))

    def extend(self, episodes: list[Episode]) -> None:
        for episode in episodes:
            self.add(episode)

    # ---------------------------------------------------------------- sampling
    def sample(
        self, batch_size: int, seq_len: int, rng: np.random.Generator
    ) -> dict[str, torch.Tensor]:
        """Sample ``batch_size`` sub-sequences of ``seq_len`` consecutive frames.

        Returns a dict with ``obs`` (B, L, 3, H, W) uint8, ``prev_actions`` (B, L, A) float32
        and ``rewards`` (B, L) float32.
        """
        eligible = [e for e in self._episodes if len(e) >= seq_len]
        if not eligible:
            raise ValueError(f"No stored episode has at least {seq_len} frames.")
        obs, actions, rewards = [], [], []
        for _ in range(batch_size):
            episode = eligible[int(rng.integers(len(eligible)))]
            start = int(rng.integers(0, len(episode) - seq_len + 1))
            window = slice(start, start + seq_len)
            obs.append(episode.obs[window])
            actions.append(episode.prev_actions[window])
            rewards.append(episode.rewards[window])
        obs_arr = np.transpose(np.stack(obs), (0, 1, 4, 2, 3))  # (B, L, 3, H, W)
        return {
            "obs": torch.from_numpy(np.ascontiguousarray(obs_arr)),
            "prev_actions": torch.from_numpy(np.stack(actions)),
            "rewards": torch.from_numpy(np.stack(rewards)),
        }

    # ------------------------------------------------------------- persistence
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            obs=np.concatenate([e.obs for e in self._episodes]),
            prev_actions=np.concatenate([e.prev_actions for e in self._episodes]),
            rewards=np.concatenate([e.rewards for e in self._episodes]),
            lengths=np.array([len(e) for e in self._episodes]),
        )

    @classmethod
    def load(cls, path: str | Path, capacity: int = 1_000_000) -> EpisodeBuffer:
        buffer = cls(capacity=capacity)
        with np.load(path) as data:
            # NpzFile decompresses on every access: read each array exactly once.
            obs = data["obs"]
            prev_actions = data["prev_actions"]
            rewards = data["rewards"]
            lengths = data["lengths"]
        bounds = np.concatenate([[0], np.cumsum(lengths)])
        for start, end in zip(bounds[:-1], bounds[1:], strict=True):
            buffer.add(
                Episode(
                    obs=obs[start:end],
                    prev_actions=prev_actions[start:end],
                    rewards=rewards[start:end],
                )
            )
        return buffer