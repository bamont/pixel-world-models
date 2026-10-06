import numpy as np
from PIL import Image, ImageFilter

from pixelwm.envs import PointReach
from pixelwm.eval.objects import localization_error, object_centroids


def _render(seed: int) -> tuple[PointReach, np.ndarray]:
    env = PointReach()
    env.reset(seed=seed)
    return env, env.render()


def test_centroids_match_environment_state():
    checked = 0
    for seed in range(10):
        env, image = _render(seed)
        agent_px, goal_px = env._to_pixels(env._pos), env._to_pixels(env._goal)
        if np.hypot(agent_px[0] - goal_px[0], agent_px[1] - goal_px[1]) < 12:
            continue  # the agent would partly hide the goal
        found = object_centroids(image)
        assert np.allclose(found["agent"], agent_px, atol=0.6)
        assert np.allclose(found["goal"], goal_px, atol=0.6)
        checked += 1
    assert checked >= 5


def test_blurred_images_are_still_localised():
    env, image = _render(1)
    blurred = np.asarray(Image.fromarray(image).filter(ImageFilter.GaussianBlur(1.5)))
    sharp, soft = object_centroids(image), object_centroids(blurred)
    for name in ("agent", "goal"):
        assert np.linalg.norm(sharp[name] - soft[name]) < 1.0


def test_batch_shapes_and_float_input():
    _, image = _render(2)
    batch = np.stack([image, image, image])
    assert object_centroids(batch)["agent"].shape == (3, 2)
    as_float = batch.astype(np.float32) / 255.0
    assert np.allclose(object_centroids(as_float)["goal"], object_centroids(batch)["goal"])


def test_faint_noise_is_not_detected():
    rng = np.random.default_rng(0)
    image = 20 / 255 + rng.uniform(0, 0.05, size=(64, 64, 3)).astype(np.float32)
    assert np.isnan(object_centroids(image)["agent"]).all()
    assert np.isnan(object_centroids(image)["goal"]).all()


def test_localization_error_and_found_rate():
    real = np.array([[10.0, 10.0], [20.0, 20.0], [np.nan, np.nan], [30.0, 30.0]])
    pred = np.array([[13.0, 14.0], [np.nan, np.nan], [5.0, 5.0], [30.0, 30.0]])
    error, rate = localization_error(real, pred)
    assert np.isclose(error, (5.0 + 0.0) / 2)  # samples 0 and 3 are detected in both
    assert np.isclose(rate, 2 / 3)  # real object present in 3 samples, found in 2


def test_localization_error_degenerate_cases():
    nan = np.full((2, 2), np.nan)
    assert all(np.isnan(v) for v in localization_error(nan, nan))
    err, rate = localization_error(np.ones((2, 2)), nan)
    assert np.isnan(err) and rate == 0.0
