"""Image-quality metrics. All functions take images in [0, 1] shaped ``(N, 3, H, W)``."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Per-sample mean squared error, shape ``(N,)``."""
    return (pred - target).pow(2).flatten(1).mean(dim=1)


def psnr(pred: torch.Tensor, target: torch.Tensor, max_val: float = 1.0) -> torch.Tensor:
    """Per-sample peak signal-to-noise ratio in dB, shape ``(N,)``."""
    return 10.0 * torch.log10(max_val**2 / mse(pred, target).clamp_min(1e-10))


def _gaussian_window(size: int, sigma: float, channels: int, like: torch.Tensor) -> torch.Tensor:
    coords = torch.arange(size, dtype=like.dtype, device=like.device) - size // 2
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g = g / g.sum()
    window = torch.outer(g, g)
    return window.expand(channels, 1, size, size).contiguous()


def ssim(
    pred: torch.Tensor, target: torch.Tensor, window_size: int = 11, sigma: float = 1.5
) -> torch.Tensor:
    """Per-sample structural similarity (Gaussian window, no padding), shape ``(N,)``."""
    channels = pred.shape[1]
    window = _gaussian_window(window_size, sigma, channels, pred)

    def blur(x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(x, window, groups=channels)

    mu_x, mu_y = blur(pred), blur(target)
    var_x = blur(pred * pred) - mu_x**2
    var_y = blur(target * target) - mu_y**2
    cov = blur(pred * target) - mu_x * mu_y
    c1, c2 = 0.01**2, 0.03**2
    score = ((2 * mu_x * mu_y + c1) * (2 * cov + c2)) / (
        (mu_x**2 + mu_y**2 + c1) * (var_x + var_y + c2)
    )
    return score.flatten(1).mean(dim=1)


class LPIPS:
    """Lazy wrapper around the optional ``lpips`` package (``pip install pixel-wm[metrics]``)."""

    def __init__(self, device: torch.device | str = "cpu") -> None:
        try:
            import lpips
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError("LPIPS requires `pip install pixel-wm[metrics]`.") from exc
        self._net = lpips.LPIPS(net="alex", verbose=False).to(device).eval()

    @torch.no_grad()
    def __call__(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # lpips expects inputs in [-1, 1]
        return self._net(pred * 2 - 1, target * 2 - 1).flatten()
