import numpy as np
import pytest

from pixelwm.data import EpisodeBuffer


def test_episode_alignment(small_buffer):
    episode = small_buffer.episodes[0]
    assert len(episode) == 21  # max_steps + initial frame
    assert np.all(episode.prev_actions[0] == 0) and episode.rewards[0] == 0


def test_sample_shapes(small_buffer, rng):
    batch = small_buffer.sample(batch_size=4, seq_len=10, rng=rng)
    assert batch["obs"].shape == (4, 10, 3, 64, 64)
    assert not batch["obs"].dtype.is_floating_point
    assert batch["prev_actions"].shape == (4, 10, 2)
    assert batch["rewards"].shape == (4, 10)


def test_sample_too_long_raises(small_buffer, rng):
    with pytest.raises(ValueError):
        small_buffer.sample(1, 1000, rng)


def test_capacity_eviction(small_buffer):
    buffer = EpisodeBuffer(capacity=45)
    buffer.extend(small_buffer.episodes)
    assert len(buffer) <= 45 and buffer.num_episodes == 2


def test_save_load_roundtrip(small_buffer, tmp_path):
    path = tmp_path / "data.npz"
    small_buffer.save(path)
    loaded = EpisodeBuffer.load(path)
    assert loaded.num_episodes == small_buffer.num_episodes
    assert np.array_equal(loaded.episodes[2].obs, small_buffer.episodes[2].obs)


def test_load_decompresses_arrays_once(small_buffer, tmp_path):
    path = tmp_path / "data.npz"
    small_buffer.save(path)
    loaded = EpisodeBuffer.load(path)
    # All episodes must be views of one shared array, not one full copy each.
    assert loaded.episodes[0].obs.base is loaded.episodes[1].obs.base