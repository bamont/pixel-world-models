"""Visualisation helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def comparison_frames(real: np.ndarray, dream: np.ndarray, gap: int = 2) -> np.ndarray:
    """Stack ``real | dream | |difference|`` side by side.

    Args:
        real: uint8 frames of shape (T, H, W, 3).
        dream: uint8 frames of the same shape.
    """
    diff = np.abs(real.astype(np.int16) - dream.astype(np.int16)).astype(np.uint8)
    spacer = np.full((real.shape[0], real.shape[1], gap, 3), 255, dtype=np.uint8)
    return np.concatenate([real, spacer, dream, spacer, diff], axis=2)


def save_gif(frames: np.ndarray, path: str | Path, fps: int = 10, scale: int = 4) -> None:
    """Write uint8 frames (T, H, W, 3) to an infinitely looping GIF, upscaled by ``scale``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    images = [Image.fromarray(f) for f in frames]
    if scale > 1:
        images = [im.resize((im.width * scale, im.height * scale), Image.Resampling.NEAREST) for im in images]
    images[0].save(
        path,
        save_all=True,
        append_images=images[1:],
        duration=int(1000 / fps),  # milliseconds per frame
        loop=0,
    )
