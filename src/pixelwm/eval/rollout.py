"""Open-loop evaluation: how far can a model predict the future from a few context frames?"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch

from pixelwm.data.buffer import Episode, EpisodeBuffer
from pixelwm.eval.metrics import mse, psnr, ssim
from pixelwm.models.base import WorldModel, index_state


@torch.no_grad()
def _dream(
    model: WorldModel,
    obs: torch.Tensor,
    prev_actions: torch.Tensor,
    context: int,
) -> torch.Tensor:
    """Observe ``context`` frames, then imagine the rest using the ground-truth actions.

    Returns the decoded imagined frames ``(B, L - context, 3, H, W)`` in [0, 1].
    """
    posterior, _ = model.observe(obs[:, :context], prev_actions[:, :context])
    start = index_state(posterior, context - 1)
    imagined = model.imagine(start, prev_actions[:, context:])
    return model.decode(imagined)


@torch.no_grad()
def open_loop_eval(
    model: WorldModel,
    buffer: EpisodeBuffer,
    horizons: Sequence[int] = (1, 5, 15, 45),
    context: int = 5,
    num_sequences: int = 64,
    seed: int = 0,
) -> dict[str, float]:
    """Measure multi-step prediction quality (the "drift" of long rollouts).

    For each horizon ``h``, reports ``psnr@h``, ``ssim@h`` and ``mse@h`` between the image
    imagined ``h`` steps after the context and the real one. Only meaningful for models that
    can decode pixels.
    """
    if not model.supports_decoding:
        raise NotImplementedError(f"{model.name} cannot decode pixels; use latent metrics.")
    was_training = model.training
    model.eval()
    rng = np.random.default_rng(seed)
    max_h = max(horizons)
    batch = buffer.sample(num_sequences, context + max_h, rng)
    obs = batch["obs"].to(model.device)
    prev_actions = batch["prev_actions"].to(model.device)

    pred = _dream(model, obs, prev_actions, context)
    target = obs[:, context:].float() / 255.0
    
    # Trivial baselines that any useful model must beat.
    last_context = obs[:, context - 1].float() / 255.0
    background = target.flatten(0, 1).median(dim=0).values  # per-pixel median over all frames

    results: dict[str, float] = {}
    for h in horizons:
        p, t = pred[:, h - 1], target[:, h - 1]
        results[f"psnr@{h}"] = psnr(p, t).mean().item()
        results[f"ssim@{h}"] = ssim(p, t).mean().item()
        results[f"mse@{h}"] = mse(p, t).mean().item()
        results[f"psnr_copy@{h}"] = psnr(last_context, t).mean().item()
        results[f"psnr_bg@{h}"] = psnr(background.expand_as(t), t).mean().item()
    model.train(was_training)
    return results


@torch.no_grad()
def dream_episode(
    model: WorldModel, episode: Episode, context: int = 5, horizon: int | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(real, dream)`` uint8 frames ``(T, H, W, 3)`` for qualitative comparison."""
    was_training = model.training
    model.eval()
    horizon = horizon or (len(episode) - context)
    end = context + horizon
    obs = torch.from_numpy(episode.obs[:end]).permute(0, 3, 1, 2)[None].to(model.device)
    actions = torch.from_numpy(episode.prev_actions[:end])[None].to(model.device)
    pred = _dream(model, obs, actions, context)[0]
    dream = (pred.permute(0, 2, 3, 1).cpu().numpy() * 255).round().astype(np.uint8)
    model.train(was_training)
    return episode.obs[context:end], dream
