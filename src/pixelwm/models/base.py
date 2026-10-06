"""Common interface implemented by every world model in the benchmark.

A *state* is a ``dict[str, Tensor]`` whose entries share the leading batch dimension (and,
for sequences, a time dimension right after it). Models are free to put whatever they need
inside (deterministic memory, stochastic latents, logits, ...); the rest of the code base only
manipulates states through the methods below.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar

import torch
from torch import nn

State = dict[str, torch.Tensor]


def stack_states(states: list[State], dim: int = 1) -> State:
    """Stack a list of states along a new (time) dimension."""
    return {key: torch.stack([s[key] for s in states], dim=dim) for key in states[0]}


def index_state(state: State, index: int, dim: int = 1) -> State:
    """Select one time step from a sequence of states."""
    return {key: value.select(dim, index) for key, value in state.items()}


class WorldModel(nn.Module, ABC):
    """Abstract pixel-based world model.

    Tensor conventions (``B`` batch, ``L`` time, ``A`` action size):

    * observations: ``(B, L, 3, H, W)``, ``uint8`` in [0, 255] or ``float`` in [0, 1];
    * ``prev_actions``: ``(B, L, A)``, the action that led to each observation;
    * decoded images: ``float`` in [0, 1].
    """

    name: ClassVar[str] = "base"
    #: Whether :meth:`decode` can produce pixels (False for e.g. JEPA-style models).
    supports_decoding: ClassVar[bool] = True
    config: Any
    action_dim: int

    # ------------------------------------------------------------------ state
    @abstractmethod
    def initial_state(self, batch_size: int) -> State:
        """Return the initial state for ``batch_size`` sequences."""

    @abstractmethod
    def features(self, state: State) -> torch.Tensor:
        """Flat feature vector of a state, used by heads and planners."""

    # --------------------------------------------------------------- inference
    @abstractmethod
    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        """Map observations ``(..., 3, H, W)`` to embeddings ``(..., E)``."""

    @abstractmethod
    def observe(
        self,
        obs: torch.Tensor,
        prev_actions: torch.Tensor,
        state: State | None = None,
    ) -> tuple[State, State]:
        """Filter a sequence of observations.

        Returns:
            ``(posterior, prior)`` state sequences of shape ``(B, L, ...)``. The prior is
            the model's prediction *before* seeing the observation of that step.
        """

    @abstractmethod
    def step(self, state: State, action: torch.Tensor) -> State:
        """Predict the next state from ``state`` and ``action`` without any observation."""

    def imagine(self, state: State, actions: torch.Tensor) -> State:
        """Roll the model forward in latent space.

        Args:
            state: starting state with shape ``(B, ...)``.
            actions: ``(B, H, A)``; ``actions[:, k]`` is applied at the ``k``-th step.

        Returns:
            States of shape ``(B, H, ...)``; entry ``k`` is the state *after* ``actions[:, k]``.
        """
        states: list[State] = []
        for k in range(actions.shape[1]):
            state = self.step(state, actions[:, k])
            states.append(state)
        return stack_states(states)

    # ------------------------------------------------------------------- heads
    @abstractmethod
    def decode(self, state: State) -> torch.Tensor:
        """Decode states ``(..., )`` into images ``(..., 3, H, W)`` in [0, 1]."""

    @abstractmethod
    def predict_reward(self, state: State) -> torch.Tensor:
        """Predict the reward associated with each state, shape ``(...)``."""

    # ---------------------------------------------------------------- training
    @abstractmethod
    def loss(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute the training loss and a dict of scalar metrics for logging."""

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device
