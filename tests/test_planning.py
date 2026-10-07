import numpy as np
import pytest
import torch
from torch import nn

from pixelwm.envs import PointReach
from pixelwm.models import RSSM, RSSMConfig
from pixelwm.models.base import State, WorldModel, stack_states
from pixelwm.planning import CEMPlanner, evaluate_policy

GOAL = torch.tensor([0.5, -0.5])


class ToyModel(WorldModel):
    """Known dynamics (pos += 0.1 * action), reward -|pos - goal|: nothing is learned."""

    name = "toy"
    supports_decoding = False

    def __init__(self) -> None:
        super().__init__()
        self.action_dim = 2
        self.config = None
        self.dummy = nn.Parameter(torch.zeros(1))  # gives the module a device

    def initial_state(self, batch_size: int) -> State:
        return {"pos": torch.zeros(batch_size, 2)}

    def features(self, state: State) -> torch.Tensor:
        return state["pos"]

    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def step(self, state: State, action: torch.Tensor) -> State:
        return {"pos": state["pos"] + 0.1 * action}

    def observe(self, obs, prev_actions, state=None):
        state = state if state is not None else self.initial_state(prev_actions.shape[0])
        posteriors = []
        for t in range(prev_actions.shape[1]):
            state = self.step(state, prev_actions[:, t])
            posteriors.append(state)
        stacked = stack_states(posteriors)
        return stacked, stacked

    def decode(self, state: State) -> torch.Tensor:
        raise NotImplementedError

    def predict_reward(self, state: State) -> torch.Tensor:
        return -(state["pos"] - GOAL).norm(dim=-1)

    def loss(self, batch):
        raise NotImplementedError


def test_cem_reaches_a_known_goal_with_a_perfect_model():
    torch.manual_seed(0)
    planner = CEMPlanner(ToyModel(), horizon=8, num_samples=256, num_elites=32, iterations=4)
    obs = np.zeros((64, 64, 3), dtype=np.uint8)  # ignored by the toy model
    pos = torch.zeros(2)
    for _ in range(25):
        action = planner(obs)
        assert action.shape == (2,) and np.all(np.abs(action) <= 1.0)
        pos = pos + 0.1 * torch.from_numpy(action)
    assert (pos - GOAL).norm() < 0.15


def test_planner_state_is_tracked_and_reset():
    planner = CEMPlanner(ToyModel(), horizon=4, num_samples=16, num_elites=4, iterations=2)
    obs = np.zeros((64, 64, 3), dtype=np.uint8)
    action = planner(obs)
    assert planner.state is not None and np.array_equal(planner.prev_action, action)
    planner.reset()
    assert planner.state is None and not planner.prev_action.any()


def test_invalid_elite_count_raises():
    with pytest.raises(ValueError):
        CEMPlanner(ToyModel(), num_samples=8, num_elites=9)
    with pytest.raises(ValueError):
        CEMPlanner(ToyModel(), num_samples=8, num_elites=1)


def test_planner_runs_with_an_untrained_rssm():
    model = RSSM(2, RSSMConfig(deter=16, groups=2, classes=4, hidden=16, cnn_depth=8))
    planner = CEMPlanner(model, horizon=3, num_samples=8, num_elites=2, iterations=2)
    env = PointReach(max_steps=4)
    result = evaluate_policy(env, planner, episodes=2, seed=0)
    assert len(result["returns"]) == 2 and np.isfinite(result["return_mean"])
