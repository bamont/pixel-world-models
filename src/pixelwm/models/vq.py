"""Vector-quantised autoencoder used as a frame tokenizer.

A 64x64 RGB frame is mapped to an 8x8 grid of discrete tokens (64 tokens, one id in
``[0, num_codes)`` each) and decoded back from them. The codebook is learned with exponential
moving averages (van den Oord et al., 2017; Razavi et al., 2019) and unused codes are periodically
restarted on the tokens that are currently worst reconstructed, which is what keeps small,
rare objects (here the agent) representable instead of collapsing onto the background code.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class ResBlock(nn.Module):
    """Two 3x3 convolutions with a skip connection."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.GroupNorm(1, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.GroupNorm(1, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, 1, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class VQEncoder(nn.Module):
    """``(N, 3, 64, 64)`` -> ``(N, code_dim, 8, 8)``."""

    def __init__(self, depth: int, code_dim: int) -> None:
        super().__init__()
        channels = [3, depth, 2 * depth, 4 * depth]
        layers: list[nn.Module] = []
        for c_in, c_out in zip(channels[:-1], channels[1:], strict=True):
            layers += [nn.Conv2d(c_in, c_out, 4, 2, 1), nn.GroupNorm(1, c_out), nn.SiLU()]
        layers += [ResBlock(4 * depth), ResBlock(4 * depth), nn.Conv2d(4 * depth, code_dim, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-2:] != (64, 64):
            raise ValueError(f"VQEncoder expects 64x64 inputs, got {tuple(x.shape[-2:])}.")
        return self.net(x)


class VQDecoder(nn.Module):
    """``(N, code_dim, 8, 8)`` -> ``(N, 3, 64, 64)``."""

    def __init__(self, depth: int, code_dim: int) -> None:
        super().__init__()
        channels = [4 * depth, 2 * depth, depth]
        layers: list[nn.Module] = [
            nn.Conv2d(code_dim, 4 * depth, 3, 1, 1),
            ResBlock(4 * depth),
            ResBlock(4 * depth),
        ]
        for c_in, c_out in zip(channels[:-1], channels[1:], strict=True):
            layers += [nn.ConvTranspose2d(c_in, c_out, 4, 2, 1), nn.GroupNorm(1, c_out), nn.SiLU()]
        layers.append(nn.ConvTranspose2d(depth, 3, 4, 2, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class VectorQuantizer(nn.Module):
    """Nearest-neighbour quantiser with an EMA-updated codebook and dead-code restarts.

    The codebook is a buffer, not a parameter: it is moved by moving averages of the encoder
    outputs assigned to each code, never by gradients.

    Args:
        num_codes: codebook size.
        dim: dimension of each code.
        decay: EMA decay of the per-code statistics.
        dead_threshold: a code whose EMA usage falls below this value is considered dead.
        restart_every: dead codes are re-initialised every this many updates (0 disables it).
    """

    codebook: torch.Tensor
    ema_count: torch.Tensor
    ema_sum: torch.Tensor
    initialized: torch.Tensor
    updates: torch.Tensor

    def __init__(
        self,
        num_codes: int,
        dim: int,
        decay: float = 0.99,
        eps: float = 1e-5,
        dead_threshold: float = 0.03,
        restart_every: int = 50,
    ) -> None:
        super().__init__()
        self.num_codes = num_codes
        self.dim = dim
        self.decay = decay
        self.eps = eps
        self.dead_threshold = dead_threshold
        self.restart_every = restart_every
        codebook = torch.randn(num_codes, dim) * 0.1
        self.register_buffer("codebook", codebook)
        self.register_buffer("ema_count", torch.ones(num_codes))
        self.register_buffer("ema_sum", codebook.clone())
        self.register_buffer("initialized", torch.zeros((), dtype=torch.bool))
        self.register_buffer("updates", torch.zeros((), dtype=torch.long))

    def indices(self, z: torch.Tensor) -> torch.Tensor:
        """Index of the nearest code for each row of ``z`` ``(M, dim)``."""
        codes = self.codebook
        distances = z.pow(2).sum(1, keepdim=True) - 2.0 * z @ codes.t() + codes.pow(2).sum(1)[None]
        return distances.argmin(dim=1)

    def forward(
        self, z: torch.Tensor, update: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Quantise ``z`` ``(M, dim)``.

        Returns:
            ``(quantised, indices, commitment_loss)``; ``quantised`` carries straight-through
            gradients to ``z``. With ``update=True`` the codebook statistics are updated.
        """
        if update and not bool(self.initialized):
            self._initialise(z.detach())
        indices = self.indices(z.detach())
        quantised = self.codebook[indices]
        if update:
            self._ema_update(z.detach(), indices)
        commitment = F.mse_loss(z, quantised.detach())
        return z + (quantised - z).detach(), indices, commitment

    @torch.no_grad()
    def _initialise(self, z: torch.Tensor) -> None:
        pick = torch.randint(0, z.shape[0], (self.num_codes,), device=z.device)
        self.codebook.copy_(z[pick])
        self.ema_sum.copy_(self.codebook)
        self.ema_count.fill_(1.0)
        self.initialized.fill_(True)

    @torch.no_grad()
    def _ema_update(self, z: torch.Tensor, indices: torch.Tensor) -> None:
        one_hot = F.one_hot(indices, self.num_codes).to(z.dtype)
        self.ema_count.mul_(self.decay).add_(one_hot.sum(0), alpha=1.0 - self.decay)
        self.ema_sum.mul_(self.decay).add_(one_hot.t() @ z, alpha=1.0 - self.decay)
        total = self.ema_count.sum()
        smoothed = (self.ema_count + self.eps) / (total + self.num_codes * self.eps) * total
        self.codebook.copy_(self.ema_sum / smoothed.unsqueeze(1))
        self.updates.add_(1)
        if self.restart_every and int(self.updates) % self.restart_every == 0:
            self._restart_dead(z, indices)

    @torch.no_grad()
    def _restart_dead(self, z: torch.Tensor, indices: torch.Tensor) -> None:
        """Re-initialise unused codes on the encoder outputs that are worst quantised."""
        dead = self.ema_count < self.dead_threshold
        count = int(dead.sum())
        if count == 0:
            return
        error = (z - self.codebook[indices]).pow(2).sum(1)
        if float(error.sum()) <= 0.0:
            return
        pick = torch.multinomial(error / error.sum(), count, replacement=True)
        self.codebook[dead] = z[pick]
        self.ema_sum[dead] = z[pick]
        self.ema_count[dead] = 1.0


class VQTokenizer(nn.Module):
    """Frame tokenizer: encoder + vector quantiser + decoder.

    Images are expected centred in ``[-0.5, 0.5]`` (the convention of the RSSM) and reconstructions
    are returned in the same space.
    """

    tokens_per_frame = 64  # 8 x 8 grid
    grid = 8

    def __init__(
        self,
        num_codes: int = 512,
        code_dim: int = 64,
        depth: int = 32,
        decay: float = 0.99,
        restart_every: int = 50,
    ) -> None:
        super().__init__()
        self.code_dim = code_dim
        self.encoder = VQEncoder(depth, code_dim)
        self.decoder = VQDecoder(depth, code_dim)
        self.quantizer = VectorQuantizer(
            num_codes, code_dim, decay=decay, restart_every=restart_every
        )

    def _flatten_latents(self, z: torch.Tensor) -> torch.Tensor:
        """``(N, D, 8, 8)`` -> ``(N * 64, D)`` with positions in row-major order."""
        return z.permute(0, 2, 3, 1).reshape(-1, self.code_dim)

    def encode_ids(self, x: torch.Tensor) -> torch.Tensor:
        """Token ids ``(N, 64)`` of images ``(N, 3, 64, 64)``."""
        indices = self.quantizer.indices(self._flatten_latents(self.encoder(x)))
        return indices.view(x.shape[0], self.tokens_per_frame)

    def embed(self, ids: torch.Tensor) -> torch.Tensor:
        """Codebook vectors of token ids ``(N, 64)`` as a latent map ``(N, D, 8, 8)``."""
        codes = self.quantizer.codebook[ids]
        return codes.view(ids.shape[0], self.grid, self.grid, self.code_dim).permute(0, 3, 1, 2)

    def decode_ids(self, ids: torch.Tensor, chunk: int = 2048) -> torch.Tensor:
        """Images ``(N, 3, 64, 64)`` (centred) from token ids ``(N, 64)``, in chunks."""
        return torch.cat([self.decoder(self.embed(part)) for part in ids.split(chunk)])

    def forward(
        self, x: torch.Tensor, update: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Training pass. Returns ``(reconstruction, token ids (N, 64), commitment loss)``."""
        batch = x.shape[0]
        quantised, indices, commitment = self.quantizer(
            self._flatten_latents(self.encoder(x)), update=update
        )
        latents = quantised.view(batch, self.grid, self.grid, self.code_dim).permute(0, 3, 1, 2)
        return self.decoder(latents), indices.view(batch, -1), commitment
