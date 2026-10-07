import numpy as np

from pixelwm.envs import PointReach
from pixelwm.planning import OraclePolicy, RandomPolicy, evaluate_policy


def test_oracle_reaches_the_goal_and_random_does_not():
    env = PointReach(max_steps=50)
    oracle = evaluate_policy(env, OraclePolicy(env), episodes=10, seed=0)
    random = evaluate_policy(env, RandomPolicy(env.action_space, 0), episodes=10, seed=0)
    assert oracle["success_rate"] == 1.0
    assert oracle["return_mean"] > random["return_mean"] + 20.0


def test_evaluation_is_paired_across_policies():
    env = PointReach(max_steps=5)
    a = evaluate_policy(env, RandomPolicy(env.action_space, 0), episodes=3, seed=7)
    b = evaluate_policy(env, RandomPolicy(env.action_space, 0), episodes=3, seed=7)
    assert a["returns"] == b["returns"]  # same seeds -> identical episodes and actions


def test_random_policy_respects_bounds():
    env = PointReach()
    policy = RandomPolicy(env.action_space, seed=1)
    actions = np.stack([policy(np.zeros((64, 64, 3))) for _ in range(200)])
    assert actions.min() >= -1.0 and actions.max() <= 1.0 and actions.dtype == np.float32


def test_record_first_episode_returns_frames():
    env = PointReach(max_steps=4)
    result = evaluate_policy(env, OraclePolicy(env), episodes=2, seed=0, record_first=True)
    assert result["frames"].shape == (5, 64, 64, 3)
