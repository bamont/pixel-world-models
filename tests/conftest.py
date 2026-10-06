import numpy as np
import pytest

from pixelwm.data import EpisodeBuffer, collect_buffer
from pixelwm.envs import make_env


@pytest.fixture(scope="session")
def small_buffer() -> EpisodeBuffer:
    env = make_env("point_reach", size=64, max_steps=20)
    return collect_buffer(env, num_episodes=6, seed=0)


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(0)
