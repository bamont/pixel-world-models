import numpy as np
from PIL import Image

from pixelwm.data import collect_episodes
from pixelwm.envs import make_env
from pixelwm.utils.viz import save_dream_gif


def _episode():
    return collect_episodes(make_env("point_reach", max_steps=12), 1, seed=0)[0]


def test_save_dream_gif_with_trails(tmp_path):
    episode = _episode()
    real = episode.obs[5:]
    dream = np.roll(real, 3, axis=2)  # a deliberately wrong "dream"
    path = tmp_path / "dream.gif"
    save_dream_gif(real, dream, path, last_context=episode.obs[4], scale=2)
    with Image.open(path) as gif:
        assert gif.n_frames == len(real)
        assert gif.size == (4 * 64 * 2 + 3 * 2 * 2, 64 * 2 + 14)


def test_save_dream_gif_without_trails_has_three_panels(tmp_path):
    episode = _episode()
    real = episode.obs[5:]
    path = tmp_path / "dream.gif"
    save_dream_gif(real, real, path, scale=2, trails=False)
    with Image.open(path) as gif:
        assert gif.size == (3 * 64 * 2 + 2 * 2 * 2, 64 * 2 + 14)


def test_save_dream_gif_survives_undetected_objects(tmp_path):
    episode = _episode()
    real = episode.obs[5:]
    blank = np.full_like(real, 20)  # the model dreams an empty world
    save_dream_gif(real, blank, tmp_path / "blank.gif", last_context=episode.obs[4], scale=2)
    assert (tmp_path / "blank.gif").exists()
