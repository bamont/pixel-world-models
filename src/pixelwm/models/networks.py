"""Reusable neural building blocks."""

from __future__ import annotations

import torch
from torch import nn


class MLP(nn.Module):
    """``layers`` x (Linear -> LayerNorm -> SiLU) followed by a linear output layer."""

    def __init__(self, in_dim: int, hidden: int, out_dim: int, layers: int = 1) -> None:
        super().__init__()
        blocks: list[nn.Module] = []
        dim = in_dim
        for _ in range(layers):
            blocks += [nn.Linear(dim, hidden), nn.LayerNorm(hidden), nn.SiLU()]
            dim = hidden
        blocks.append(nn.Linear(dim, out_dim))
        self.net = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LayerNormGRUCell(nn.Module):
    """GRU cell with layer normalisation, as used in Dreamer's RSSM."""

    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.linear = nn.Linear(input_size + hidden_size, 3 * hidden_size, bias=False)
        self.norm = nn.LayerNorm(3 * hidden_size)

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        parts = self.norm(self.linear(torch.cat([x, h], dim=-1)))
        reset, candidate, update = parts.chunk(3, dim=-1)
        reset = torch.sigmoid(reset)
        candidate = torch.tanh(reset * candidate)
        update = torch.sigmoid(update - 1.0)
        return update * candidate + (1.0 - update) * h


class ConvEncoder(nn.Module):
    """4 stride-2 convolutions mapping ``(N, 3, 64, 64)`` to a flat embedding."""

    def __init__(self, depth: int = 32) -> None:
        super().__init__()
        channels = [3, depth, 2 * depth, 4 * depth, 8 * depth]
        layers: list[nn.Module] = []
        for c_in, c_out in zip(channels[:-1], channels[1:], strict=True):
            layers += [nn.Conv2d(c_in, c_out, 4, 2, 1), nn.GroupNorm(1, c_out), nn.SiLU()]
        self.net = nn.Sequential(*layers)
        self.embed_dim = 8 * depth * 4 * 4

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-2:] != (64, 64):
            raise ValueError(f"ConvEncoder expects 64x64 inputs, got {tuple(x.shape[-2:])}.")
        return self.net(x).flatten(1)


class ConvDecoder(nn.Module):
    """Transposed-convolution decoder mapping flat features to ``(N, 3, 64, 64)``."""

    def __init__(self, in_dim: int, depth: int = 32) -> None:
        super().__init__()
        self.base_channels = 8 * depth
        self.fc = nn.Linear(in_dim, self.base_channels * 4 * 4)
        channels = [8 * depth, 4 * depth, 2 * depth, depth]
        layers: list[nn.Module] = []
        for c_in, c_out in zip(channels[:-1], channels[1:], strict=True):
            layers += [nn.ConvTranspose2d(c_in, c_out, 4, 2, 1), nn.GroupNorm(1, c_out), nn.SiLU()]
        layers.append(nn.ConvTranspose2d(depth, 3, 4, 2, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        x = self.fc(features).view(-1, self.base_channels, 4, 4)
        return self.net(x)
