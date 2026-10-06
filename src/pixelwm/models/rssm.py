"""Recurrent State-Space Model (Dreamer-style) with discrete stochastic latents.

Reference: Hafner et al., "Mastering Diverse Domains through World Models" (DreamerV3, 2023).
This is a compact re-implementation that keeps the key ingredients: a GRU deterministic path,
categorical latents with straight-through gradients, uniform mixing ("unimix"), KL balancing
and free bits, and a symlog reward target.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from pixelwm.models.base import State, WorldModel, stack_states
from pixelwm.models.networks import MLP, ConvDecoder, ConvEncoder, LayerNormGRUCell
from pixelwm.utils.math import symexp, symlog


@dataclass
class RSSMConfig:
    deter: int = 256  # size of the deterministic (GRU) state
    groups: int = 16  # number of categorical variables
    classes: int = 16  # classes per categorical variable
    hidden: int = 256  # width of the MLPs
    cnn_depth: int = 32  # base number of channels of the CNN encoder / decoder
    unimix: float = 0.01  # mixing with a uniform distribution, avoids deterministic latents
    free_bits: float = 1.0  # minimum KL (nats) below which the KL is not optimised
    beta_dyn: float = 0.5  # weight of the dynamics loss  KL(sg(post) || prior)
    beta_rep: float = 0.1  # weight of the representation loss  KL(post || sg(prior))
    reward_scale: float = 1.0  # weight of the reward prediction loss
    fg_weight: float = 0.0  # extra weight on pixels that differ from the background (0 = off)


def categorical_kl(logits_p: torch.Tensor, logits_q: torch.Tensor) -> torch.Tensor:
    """``KL(p || q)`` for factorised categoricals, summed over groups.

    Args:
        logits_p, logits_q: ``(..., groups, classes)`` (un-normalised logits are fine).

    Returns:
        Tensor of shape ``(...)``.
    """
    log_p = F.log_softmax(logits_p, dim=-1)
    log_q = F.log_softmax(logits_q, dim=-1)
    return (log_p.exp() * (log_p - log_q)).sum(dim=-1).sum(dim=-1)


class RSSM(WorldModel):
    """Dreamer-style RSSM world model."""

    name = "rssm"
    supports_decoding = True

    def __init__(self, action_dim: int, config: RSSMConfig | None = None) -> None:
        super().__init__()
        cfg = config or RSSMConfig()
        self.config = cfg
        self.action_dim = action_dim
        self.stoch_size = cfg.groups * cfg.classes
        self.feature_size = cfg.deter + self.stoch_size

        self.encoder = ConvEncoder(cfg.cnn_depth)
        self.decoder = ConvDecoder(self.feature_size, cfg.cnn_depth)
        self.img_in = nn.Sequential(
            nn.Linear(self.stoch_size + action_dim, cfg.hidden),
            nn.LayerNorm(cfg.hidden),
            nn.SiLU(),
        )
        self.gru = LayerNormGRUCell(cfg.hidden, cfg.deter)
        self.prior_head = MLP(cfg.deter, cfg.hidden, self.stoch_size)
        self.post_head = MLP(cfg.deter + self.encoder.embed_dim, cfg.hidden, self.stoch_size)
        self.reward_head = MLP(self.feature_size, cfg.hidden, 1, layers=2)

    # ------------------------------------------------------------ latent utils
    def _mix_logits(self, raw: torch.Tensor) -> torch.Tensor:
        """Reshape to ``(..., groups, classes)`` and apply uniform mixing (returns log-probs)."""
        cfg = self.config
        logits = raw.reshape(*raw.shape[:-1], cfg.groups, cfg.classes)
        probs = F.softmax(logits, dim=-1)
        probs = (1.0 - cfg.unimix) * probs + cfg.unimix / cfg.classes
        return torch.log(probs)

    def _sample(self, log_probs: torch.Tensor) -> torch.Tensor:
        """Sample one-hot latents with straight-through gradients, flattened to ``(..., G*C)``."""
        probs = log_probs.exp()
        index = torch.distributions.Categorical(probs=probs).sample()
        one_hot = F.one_hot(index, self.config.classes).to(probs.dtype)
        sample = one_hot + probs - probs.detach()
        return sample.flatten(-2)

    # ------------------------------------------------------------------- state
    def initial_state(self, batch_size: int) -> State:
        cfg = self.config
        device = self.device
        return {
            "deter": torch.zeros(batch_size, cfg.deter, device=device),
            "stoch": torch.zeros(batch_size, self.stoch_size, device=device),
            "logits": torch.full(
                (batch_size, cfg.groups, cfg.classes), -math.log(cfg.classes), device=device
            ),
        }

    def features(self, state: State) -> torch.Tensor:
        return torch.cat([state["deter"], state["stoch"]], dim=-1)

    # --------------------------------------------------------------- transitions
    def step(self, state: State, action: torch.Tensor) -> State:
        """Prior transition ``p(z_t | h_{t-1}, z_{t-1}, a_{t-1})``."""
        x = self.img_in(torch.cat([state["stoch"], action], dim=-1))
        deter = self.gru(x, state["deter"])
        log_probs = self._mix_logits(self.prior_head(deter))
        return {"deter": deter, "stoch": self._sample(log_probs), "logits": log_probs}

    def _posterior(self, prior: State, embed: torch.Tensor) -> State:
        """Posterior ``q(z_t | h_t, x_t)`` given the prior state and the observation embedding."""
        raw = self.post_head(torch.cat([prior["deter"], embed], dim=-1))
        log_probs = self._mix_logits(raw)
        return {"deter": prior["deter"], "stoch": self._sample(log_probs), "logits": log_probs}

    # ---------------------------------------------------------------- inference
    @staticmethod
    def _preprocess(obs: torch.Tensor) -> torch.Tensor:
        """Convert uint8 [0, 255] or float [0, 1] images to float in [-0.5, 0.5]."""
        obs = obs.float() / 255.0 if obs.dtype == torch.uint8 else obs.float()
        return obs - 0.5

    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        lead = obs.shape[:-3]
        x = self._preprocess(obs).reshape(-1, *obs.shape[-3:])
        embed = self.encoder(x)
        return embed.reshape(*lead, -1)

    def observe(
        self,
        obs: torch.Tensor,
        prev_actions: torch.Tensor,
        state: State | None = None,
    ) -> tuple[State, State]:
        embed = self.encode(obs)
        batch, length = embed.shape[:2]
        state = state if state is not None else self.initial_state(batch)
        posteriors: list[State] = []
        priors: list[State] = []
        for t in range(length):
            prior = self.step(state, prev_actions[:, t])
            state = self._posterior(prior, embed[:, t])
            priors.append(prior)
            posteriors.append(state)
        return stack_states(posteriors), stack_states(priors)

    # -------------------------------------------------------------------- heads
    def _decode_raw(self, state: State) -> torch.Tensor:
        """Decoder output in the centred pixel space [-0.5, 0.5] (unclamped)."""
        feat = self.features(state)
        lead = feat.shape[:-1]
        out = self.decoder(feat.reshape(-1, feat.shape[-1]))
        return out.reshape(*lead, *out.shape[1:])

    def decode(self, state: State) -> torch.Tensor:
        return (self._decode_raw(state) + 0.5).clamp(0.0, 1.0)

    def predict_reward(self, state: State) -> torch.Tensor:
        return symexp(self.reward_head(self.features(state)).squeeze(-1))

    # ----------------------------------------------------------------- training
    def loss(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, float]]:
        cfg = self.config
        obs, prev_actions, rewards = batch["obs"], batch["prev_actions"], batch["rewards"]
        post, prior = self.observe(obs, prev_actions)

        # Reconstruction: Gaussian log-likelihood with unit variance (up to a constant).
        recon = self._decode_raw(post)
        target = self._preprocess(obs)
        sq_err = (recon - target).pow(2)
        if cfg.fg_weight > 0:
            # Background estimated as the per-pixel median over the batch (objects are sparse).
            background = target.flatten(0, 1).median(dim=0).values
            foreground = ((target - background).abs().amax(dim=-3, keepdim=True) > 0.05).float()
            sq_err = sq_err * (1.0 + cfg.fg_weight * foreground)
        recon_loss = 0.5 * sq_err.sum(dim=(-3, -2, -1)).mean()

        # Reward prediction in symlog space.
        reward_pred = self.reward_head(self.features(post)).squeeze(-1)
        reward_loss = 0.5 * (reward_pred - symlog(rewards)).pow(2).mean()

        # KL balancing with free bits.
        kl_dyn = categorical_kl(post["logits"].detach(), prior["logits"])
        kl_rep = categorical_kl(post["logits"], prior["logits"].detach())
        dyn_loss = kl_dyn.clamp(min=cfg.free_bits).mean()
        rep_loss = kl_rep.clamp(min=cfg.free_bits).mean()

        total = (
            recon_loss
            + cfg.reward_scale * reward_loss
            + cfg.beta_dyn * dyn_loss
            + cfg.beta_rep * rep_loss
        )
        metrics = {
            "loss": total.item(),
            "recon_loss": recon_loss.item(),
            "reward_loss": reward_loss.item(),
            "kl": kl_dyn.mean().item(),
            "mse": ((recon.detach() - self._preprocess(obs)).pow(2).mean()).item(),
        }
        return total, metrics
