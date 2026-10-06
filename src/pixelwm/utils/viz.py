"""Visualisation helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

WHITE = (255, 255, 255)
ORANGE = (255, 160, 0)
GOAL_GREEN = (40, 200, 80)
BACKGROUND = (20, 20, 20)
HEADER_HEIGHT = 14


def comparison_frames(real: np.ndarray, dream: np.ndarray, gap: int = 2) -> np.ndarray:
    """Stack ``real | dream | |difference|`` side by side.

    Args:
        real: uint8 frames of shape (T, H, W, 3).
        dream: uint8 frames of the same shape.
    """
    diff = np.abs(real.astype(np.int16) - dream.astype(np.int16)).astype(np.uint8)
    spacer = np.full((real.shape[0], real.shape[1], gap, 3), 255, dtype=np.uint8)
    return np.concatenate([real, spacer, dream, spacer, diff], axis=2)


def _write_gif(images: list[Image.Image], path: Path, fps: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(
        path,
        save_all=True,
        append_images=images[1:],
        duration=int(1000 / fps),  # milliseconds per frame
        loop=0,
    )


def save_gif(frames: np.ndarray, path: str | Path, fps: int = 10, scale: int = 4) -> None:
    """Write uint8 frames (T, H, W, 3) to an infinitely looping GIF, upscaled by ``scale``."""
    images = [Image.fromarray(f) for f in frames]
    if scale > 1:
        images = [
            im.resize((im.width * scale, im.height * scale), Image.Resampling.NEAREST)
            for im in images
        ]
    _write_gif(images, Path(path), fps)


# ------------------------------------------------------------------ trajectory panel
def _px(point: np.ndarray, scale: int) -> tuple[float, float]:
    return (float(point[0]) + 0.5) * scale, (float(point[1]) + 0.5) * scale


def _draw_path(
    draw: ImageDraw.ImageDraw,
    points: np.ndarray,
    color: tuple[int, int, int],
    scale: int,
    width: int,
) -> None:
    """Draw segments between consecutive points, skipping undetected (NaN) ones."""
    for p, q in zip(points[:-1], points[1:], strict=True):
        if np.isfinite(p).all() and np.isfinite(q).all():
            draw.line([_px(p, scale), _px(q, scale)], fill=color, width=width)


def _draw_dot(
    draw: ImageDraw.ImageDraw,
    point: np.ndarray,
    color: tuple[int, int, int],
    radius: float,
    scale: int,
) -> None:
    if np.isfinite(point).all():
        x, y = _px(point, scale)
        draw.ellipse([x - radius, y - radius, x + radius, y + radius], fill=color)


def trajectory_panels(
    real_xy: np.ndarray,
    dream_xy: np.ndarray,
    start_xy: np.ndarray,
    goal_xy: np.ndarray | None,
    size: int,
    scale: int,
) -> np.ndarray:
    """Render, for every time step, the agent's path so far (real vs dreamed).

    Args:
        real_xy, dream_xy: ``(T, 2)`` agent positions in pixels (NaN when undetected).
        start_xy: ``(2,)`` agent position at the last context frame (the path's origin).
        goal_xy: ``(2,)`` goal position or ``None``.
        size: side of the original (un-scaled) frames; scale: integer upscaling factor.

    Returns:
        uint8 array ``(T, size * scale, size * scale, 3)``.
    """
    side = size * scale
    real_path = np.concatenate([start_xy[None], real_xy])
    dream_path = np.concatenate([start_xy[None], dream_xy])
    real_width, dream_width = max(2, scale), max(1, scale // 2)  # thick real, thin dream on top
    panels = []
    for t in range(len(real_xy)):
        image = Image.new("RGB", (side, side), BACKGROUND)
        draw = ImageDraw.Draw(image)
        if goal_xy is not None and np.isfinite(goal_xy).all():
            gx, gy = _px(goal_xy, scale)
            half = 3.5 * scale
            draw.rectangle([gx - half, gy - half, gx + half, gy + half], outline=GOAL_GREEN)
        if np.isfinite(start_xy).all():
            sx, sy = _px(start_xy, scale)
            ring = 2.5 * scale
            draw.ellipse([sx - ring, sy - ring, sx + ring, sy + ring], outline=WHITE)
        _draw_path(draw, real_path[: t + 2], WHITE, scale, real_width)
        _draw_path(draw, dream_path[: t + 2], ORANGE, scale, dream_width)
        _draw_dot(draw, real_xy[t], WHITE, 2.0 * scale, scale)
        _draw_dot(draw, dream_xy[t], ORANGE, 1.2 * scale, scale)
        panels.append(np.asarray(image))
    return np.stack(panels)


def save_dream_gif(
    real: np.ndarray,
    dream: np.ndarray,
    path: str | Path,
    last_context: np.ndarray | None = None,
    fps: int = 10,
    scale: int = 4,
    gap: int = 2,
    trails: bool = True,
) -> None:
    """Write the annotated "dream" GIF.

    Panels, left to right: the real frame, the model's dream, the absolute difference and
    (if ``trails``) the agent's trajectory since the end of the context: white = real,
    orange = dream, white ring = starting point, green square = goal. Each frame is labelled
    with its prediction horizon ``h`` (number of steps after the context).

    Args:
        real, dream: uint8 frames ``(T, H, W, 3)``; frame ``k`` is ``h = k + 1`` steps ahead.
        last_context: last observed (context) frame ``(H, W, 3)``, used as trajectory origin.
        trails: draw the trajectory panel (only meaningful for ``point_reach``-like images,
            where the agent is red and the goal is green).
    """
    steps, height, _, _ = real.shape
    diff = np.abs(real.astype(np.int16) - dream.astype(np.int16)).astype(np.uint8)

    def upscale(x: np.ndarray) -> np.ndarray:
        return x.repeat(scale, axis=1).repeat(scale, axis=2)

    panels = [upscale(real), upscale(dream), upscale(diff)]
    labels = ["real  h=+{h}", "dream", "|real - dream|"]
    if trails:
        from pixelwm.eval.objects import object_centroids

        real_c, dream_c = object_centroids(real), object_centroids(dream)
        if last_context is not None:
            start = object_centroids(last_context)["agent"]
        else:
            start = real_c["agent"][0]
        goal = real_c["goal"]
        goal_xy = None if np.isnan(goal[:, 0]).all() else np.nanmedian(goal, axis=0)
        panels.append(
            trajectory_panels(real_c["agent"], dream_c["agent"], start, goal_xy, height, scale)
        )
        labels.append("agent: white=real orange=dream")

    spacer = np.full((steps, height * scale, gap * scale, 3), 255, dtype=np.uint8)
    pieces: list[np.ndarray] = []
    for i, panel in enumerate(panels):
        if i:
            pieces.append(spacer)
        pieces.append(panel)
    body = np.concatenate(pieces, axis=2)

    offsets = [i * (height * scale + gap * scale) for i in range(len(panels))]
    images = []
    for t in range(steps):
        canvas = Image.new("RGB", (body.shape[2], HEADER_HEIGHT + body.shape[1]), (0, 0, 0))
        canvas.paste(Image.fromarray(body[t]), (0, HEADER_HEIGHT))
        draw = ImageDraw.Draw(canvas)
        for x, text in zip(offsets, labels, strict=True):
            draw.text((x + 3, 1), text.format(h=t + 1), fill=WHITE)
        images.append(canvas)
    _write_gif(images, Path(path), fps)
