"""Colour-based object localisation for the built-in ``point_reach`` environment.

Pixel metrics such as PSNR are dominated by the uniform background, so they say little about
whether a model tracks the *objects* that matter. Here we locate the red agent and the green
goal in an image with a soft, colour-dominance centroid, which also works on blurry decoded
images. Positions are expressed in pixels, as ``(x, y)``.

These helpers are specific to environments whose agent is red and goal is green.
"""

from __future__ import annotations

import numpy as np

OBJECT_NAMES = ("agent", "goal")


def object_scores(images: np.ndarray) -> dict[str, np.ndarray]:
    """Per-pixel "redness" (agent) and "greenness" (goal), in roughly [0, 0.7].

    Args:
        images: ``(..., H, W, 3)``, uint8 in [0, 255] or float in [0, 1].
    """
    x = np.asarray(images)
    x = x.astype(np.float32) / 255.0 if x.dtype == np.uint8 else x.astype(np.float32)
    r, g, b = x[..., 0], x[..., 1], x[..., 2]
    return {"agent": r - np.maximum(g, b), "goal": g - np.maximum(r, b)}


def object_centroids(
    images: np.ndarray, threshold: float = 0.1, min_mass: float = 2.0
) -> dict[str, np.ndarray]:
    """Soft centroid of each object, ``NaN`` where the object is not detected.

    Pixels whose score is below ``threshold`` are ignored. An object counts as detected when
    the total remaining score mass is at least ``min_mass``.

    Returns:
        Dict mapping each name in :data:`OBJECT_NAMES` to an array ``(..., 2)`` of ``(x, y)``.
    """
    centroids: dict[str, np.ndarray] = {}
    for name, score in object_scores(images).items():
        weights = np.clip(score - threshold, 0.0, None)
        height, width = weights.shape[-2:]
        mass = weights.sum(axis=(-2, -1))
        xs = np.arange(width, dtype=np.float32)[None, :]
        ys = np.arange(height, dtype=np.float32)[:, None]
        denom = np.maximum(mass, 1e-8)
        cx = (weights * xs).sum(axis=(-2, -1)) / denom
        cy = (weights * ys).sum(axis=(-2, -1)) / denom
        xy = np.stack([cx, cy], axis=-1)
        xy[mass < min_mass] = np.nan
        centroids[name] = xy
    return centroids


def localization_error(real_xy: np.ndarray, pred_xy: np.ndarray) -> tuple[float, float]:
    """Compare predicted and real positions of one object over a set of samples.

    Args:
        real_xy, pred_xy: ``(N, 2)`` arrays with ``NaN`` for "not detected".

    Returns:
        ``(mean_error, found_rate)``: the mean Euclidean error in pixels over samples where
        the object is detected in both images, and the fraction of samples where the object is
        present in the real image *and* detected in the prediction. Both are ``NaN`` when the
        real object is never detected; the error is ``NaN`` when no prediction is detected.
    """
    present = ~np.isnan(real_xy[:, 0])
    if not present.any():
        return float("nan"), float("nan")
    found = present & ~np.isnan(pred_xy[:, 0])
    rate = int(found.sum()) / int(present.sum())
    if not found.any():
        return float("nan"), rate
    error = np.linalg.norm(real_xy[found] - pred_xy[found], axis=1).mean()
    return float(error), rate
