import numpy as np
import pytest

from pixelwm.envs import PointReach, get_action_dim, make_env, to_action_vector


def test_point_reach_shapes_and_truncation():
    env = PointReach(size=64, max_steps=5)
    obs, _ = env.reset(seed=0)
    assert obs.shape == (64, 64, 3) and obs.dtype == np.uint8
    for i in range(5):
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        assert reward <= 0.0 and not terminated
        assert truncated == (i == 4)


def test_point_reach_is_deterministic_given_seed():
    a, b = PointReach(), PointReach()
    obs_a, _ = a.reset(seed=3)
    obs_b, _ = b.reset(seed=3)
    assert np.array_equal(obs_a, obs_b)
    action = np.array([1.0, -0.5], dtype=np.float32)
    assert np.array_equal(a.step(action)[0], b.step(action)[0])


def test_momentum_makes_velocity_invisible_but_effective():
    env = PointReach(momentum=0.9)
    env.reset(seed=0)
    env.step(np.array([1.0, 0.0]))
    # With no further command the dot keeps moving.
    pos_before = env._pos.copy()
    env.step(np.zeros(2))
    assert not np.allclose(pos_before, env._pos)


def test_action_helpers():
    from gymnasium import spaces

    assert get_action_dim(spaces.Discrete(4)) == 4
    assert get_action_dim(spaces.Box(-1, 1, (3,))) == 3
    assert to_action_vector(spaces.Discrete(3), 1).tolist() == [0.0, 1.0, 0.0]


def test_unknown_env_raises():
    with pytest.raises(ValueError):
        make_env("does_not_exist")
