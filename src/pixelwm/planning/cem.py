"""Model-predictive control with the cross-entropy method (CEM), planning in imagination.

At every environment step the planner

1. filters the new pixel observation into the model's belief state (``model.observe``);
2. searches for the action sequence that maximises the *predicted* discounted return, by
   repeatedly sampling Gaussian action sequences, imagining them with the world model
   (``model.imagine`` + ``model.predict_reward``) and refitting the Gaussian on the best ones;
3. executes only the first action of the best sequence (receding horizon).

The planner only needs ``observe`` / ``imagine`` / ``predict_reward``, so it works with any
:class:`~pixelwm.models.base.WorldModel`, including future models that cannot decode pixels.

Reference: Hafner et al., "Learning Latent Dynamics for Planning from Pixels" (PlaNet), 2019.
"""

from __future__ import annotations

import numpy as np
import torch

from pixelwm.models.base import State, WorldModel, index_state


class CEMPlanner:
    """Receding-horizon CEM planner over a learned latent world model.

    Args:
        model: any world model with a reward head.
        horizon: number of imagined steps per plan.
        num_samples: action sequences evaluated per CEM iteration.
        num_elites: best sequences used to refit the Gaussian (at least 2).
        iterations: CEM refinement iterations per plan.
        init_std: initial standard deviation of the action sequences.
        min_std: floor on the standard deviation, which keeps exploring until the end.
        discount: discount applied to predicted rewards along the horizon.
        action_low, action_high: action bounds.
        warm_start: reuse the previous plan (shifted by one step) as the initial mean.
    """

    def __init__(
        self,
        model: WorldModel,
        horizon: int = 12,
        num_samples: int = 256,
        num_elites: int = 32,
        iterations: int = 4,
        init_std: float = 0.5,
        min_std: float = 0.05,
        discount: float = 0.99,
        action_low: float = -1.0,
        action_high: float = 1.0,
        warm_start: bool = True,
    ) -> None:
        if not 2 <= num_elites <= num_samples:
            raise ValueError("num_elites must satisfy 2 <= num_elites <= num_samples.")
        self.model = model.eval()
        self.horizon = horizon
        self.num_samples = num_samples
        self.num_elites = num_elites
        self.iterations = iterations
        self.init_std = init_std
        self.min_std = min_std
        self.action_low = action_low
        self.action_high = action_high
        self.warm_start = warm_start
        steps = torch.arange(horizon, dtype=torch.float32, device=model.device)
        self.discounts = discount**steps
        self.state: State | None = None
        self.prev_action = np.zeros(model.action_dim, dtype=np.float32)
        self._mean: torch.Tensor | None = None

    def reset(self) -> None:
        """Forget the belief state and the previous plan (call at the start of an episode)."""
        self.state = None
        self.prev_action = np.zeros(self.model.action_dim, dtype=np.float32)
        self._mean = None

    @torch.no_grad()
    def __call__(self, obs: np.ndarray) -> np.ndarray:
        """Return the action to execute after seeing ``obs`` (uint8 image ``(H, W, 3)``)."""
        device = self.model.device
        frame = torch.from_numpy(np.ascontiguousarray(obs)).permute(2, 0, 1)[None, None]
        prev = torch.from_numpy(self.prev_action)[None, None]
        posterior, _ = self.model.observe(frame.to(device), prev.to(device), self.state)
        self.state = index_state(posterior, 0)
        action = self.plan(self.state).float().cpu().numpy()
        self.prev_action = action
        return action.copy()

    @torch.no_grad()
    def plan(self, state: State) -> torch.Tensor:
        """Run CEM from a belief ``state`` of batch size 1 and return the first action."""
        device = self.model.device
        action_dim = self.model.action_dim
        start = {k: v.expand(self.num_samples, *v.shape[1:]) for k, v in state.items()}
        mean = self._mean
        if mean is None:
            mean = torch.zeros(self.horizon, action_dim, device=device)
        std = torch.full((self.horizon, action_dim), self.init_std, device=device)

        for _ in range(self.iterations):
            noise = torch.randn(self.num_samples, self.horizon, action_dim, device=device)
            actions = (mean + std * noise).clamp(self.action_low, self.action_high)
            returns = self.score(start, actions)
            elites = actions[returns.topk(self.num_elites).indices]
            mean = elites.mean(dim=0)
            std = elites.std(dim=0).clamp_min(self.min_std)

        if self.warm_start:
            self._mean = torch.cat([mean[1:], torch.zeros(1, action_dim, device=device)])
        return mean[0].clamp(self.action_low, self.action_high)

    def score(self, start: State, actions: torch.Tensor) -> torch.Tensor:
        """Predicted discounted return of each action sequence ``(N, H, A)`` -> ``(N,)``."""
        states = self.model.imagine(start, actions)
        rewards = self.model.predict_reward(states)  # (N, H)
        return (rewards * self.discounts).sum(dim=1)
