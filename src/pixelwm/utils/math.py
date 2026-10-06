"""Small numerical helpers."""

from __future__ import annotations

import torch


def symlog(x: torch.Tensor) -> torch.Tensor:
    """Symmetric logarithm ``sign(x) * log(1 + |x|)`` (DreamerV3)."""
    return torch.sign(x) * torch.log1p(torch.abs(x))


def symexp(x: torch.Tensor) -> torch.Tensor:
    """Inverse of :func:`symlog`."""
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1.0)
