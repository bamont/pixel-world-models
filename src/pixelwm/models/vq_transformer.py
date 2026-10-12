"""Discrete-token world model: VQ tokenizer + transformer (in the spirit of IRIS and Genie).

Each frame is turned into 64 discrete tokens by a VQ-VAE (:mod:`pixelwm.models.vq`). The dynamics
are learned by a transformer that sees the tokens of the last ``window`` frames, each tagged with
the action that led to it, plus a *query frame* made of learned mask embeddings tagged with the
new action. It outputs a categorical distribution over the 64 tokens of the next frame, and the
reward, in a single forward pass.

Difference with IRIS: the tokens of a frame are predicted in parallel (conditionally independent
given the context) instead of one by one. This makes one imagined step a single forward pass,
which is what makes planning affordable, at the price of not modelling dependencies between
tokens of the same frame. ``temperature=0`` (greedy decoding) is often a good inference choice
when the dynamics are close to deterministic.

Training is two-stage inside a single ``loss``: for the first ``tokenizer_steps`` calls only the
tokenizer is trained; afterwards it is frozen and only the transformer is trained on its tokens.
The call counter is a buffer, so it survives checkpointing.

State (batch ``B``, window ``W``, ``N = 64`` tokens, action size ``A``):

* ``tokens``  ``(B, W, N)`` long: token ids of the last ``W`` frames (the last one is current);
* ``actions`` ``(B, W, A)``: the action that led to each of those frames;
* ``reward``  ``(B,)``: the reward predicted on arrival at the current frame.

At the start of an episode the window is filled with copies of the first frame.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from pixelwm.models.base import State, WorldModel, stack_states
from pixelwm.models.networks import MLP
from pixelwm.models.vq import VQTokenizer
from pixelwm.utils.math import symexp, symlog


@dataclass
class VQTransformerConfig:
    num_codes: int = 512  # codebook size
    code_dim: int = 64  # dimension of each code
    cnn_depth: int = 32  # base number of channels of the tokenizer
    window: int = 3  # context frames given to the transformer
    d_model: int = 128
    n_layers: int = 4
    n_heads: int = 4
    dropout: float = 0.0
    beta_commit: float = 0.25  # weight of the VQ commitment loss
    fg_weight: float = 20.0  # extra weight on foreground pixels in the tokenizer reconstruction
    tokenizer_steps: int = 5000  # training calls spent on the tokenizer before freezing it
    tokenizer_batch: int = 128  # frames per tokenizer step
    window_batch: int = 256  # (context, target) windows per transformer step
    pad_prob: float = 0.25  # share of windows turned into "episode start" patterns
    reward_scale: float = 1.0
    temperature: float = 1.0  # sampling temperature at inference (0 = greedy)
    ema_decay: float = 0.99  # EMA decay of the codebook
    restart_every: int = 50  # dead-code restart period (tokenizer updates)


class VQTransformer(WorldModel):
    """VQ tokenizer + frame-level transformer world model."""

    name = "vq_transformer"
    supports_decoding = True
    train_calls: torch.Tensor

    def __init__(self, action_dim: int, config: VQTransformerConfig | None = None) -> None:
        super().__init__()
        cfg = config or VQTransformerConfig()
        self.config = cfg
        self.action_dim = action_dim
        self.temperature = cfg.temperature  # may be overridden at inference time
        self.tokenizer = VQTokenizer(
            cfg.num_codes, cfg.code_dim, cfg.cnn_depth, cfg.ema_decay, cfg.restart_every
        )
        n = self.tokenizer.tokens_per_frame
        d = cfg.d_model
        self.tok_emb = nn.Embedding(cfg.num_codes, d)
        self.pos_emb = nn.Parameter(torch.zeros(n, d))
        self.time_emb = nn.Parameter(torch.zeros(cfg.window + 1, d))
        self.mask_emb = nn.Parameter(torch.zeros(d))
        self.act_proj = nn.Linear(action_dim, d)
        layer = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=cfg.n_heads,
            dim_feedforward=4 * d,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, cfg.n_layers, enable_nested_tensor=False)
        self.out_norm = nn.LayerNorm(d)
        self.token_head = nn.Linear(d, cfg.num_codes)
        # The reward is a function of the *geometry* of the next frame, so it is read from a
        # per-position projection (which keeps "where") rather than from a pooled vector.
        self.reward_proj = nn.Linear(d, 16)
        self.reward_head = MLP(n * 16, 2 * d, 1)
        self.register_buffer("train_calls", torch.zeros((), dtype=torch.long))
        for parameter in (self.pos_emb, self.time_emb, self.mask_emb):
            nn.init.normal_(parameter, std=0.02)

    # ------------------------------------------------------------------ tokens
    @staticmethod
    def _preprocess(obs: torch.Tensor) -> torch.Tensor:
        """Convert uint8 [0, 255] or float [0, 1] images to float in [-0.5, 0.5]."""
        obs = obs.float() / 255.0 if obs.dtype == torch.uint8 else obs.float()
        return obs - 0.5

    @torch.no_grad()
    def tokenize(self, obs: torch.Tensor) -> torch.Tensor:
        """Token ids ``(..., 64)`` of observations ``(..., 3, H, W)``."""
        lead = obs.shape[:-3]
        x = self._preprocess(obs).reshape(-1, *obs.shape[-3:])
        ids = torch.cat([self.tokenizer.encode_ids(part) for part in x.split(2048)])
        return ids.reshape(*lead, -1)

    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        """Codebook vectors of the frame's tokens, flattened to ``(..., 64 * code_dim)``."""
        ids = self.tokenize(obs)
        return self.tokenizer.quantizer.codebook[ids].flatten(-2)

    # ------------------------------------------------------------------- state
    def initial_state(self, batch_size: int) -> State:
        n = self.tokenizer.tokens_per_frame
        w = self.config.window
        device = self.device
        return {
            "tokens": torch.zeros(batch_size, w, n, dtype=torch.long, device=device),
            "actions": torch.zeros(batch_size, w, self.action_dim, device=device),
            "reward": torch.zeros(batch_size, device=device),
        }

    def _episode_start(self, tokens: torch.Tensor) -> State:
        """State whose window is filled with copies of the first frame ``(B, N)``."""
        batch, n = tokens.shape
        w = self.config.window
        return {
            "tokens": tokens[:, None, :].expand(batch, w, n).clone(),
            "actions": torch.zeros(batch, w, self.action_dim, device=tokens.device),
            "reward": torch.zeros(batch, device=tokens.device),
        }

    def features(self, state: State) -> torch.Tensor:
        return self.tokenizer.quantizer.codebook[state["tokens"][..., -1, :]].flatten(-2)

    # ---------------------------------------------------------------- dynamics
    def _predict(
        self, tokens: torch.Tensor, actions: torch.Tensor, action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Next-frame token logits ``(B, N, K)`` and reward (symlog space) ``(B,)``.

        Args:
            tokens: ``(B, W, N)`` ids of the context frames.
            actions: ``(B, W, A)`` actions that led to each context frame.
            action: ``(B, A)`` action leading to the frame to predict.
        """
        batch, window, n = tokens.shape
        d = self.config.d_model
        frames = self.tok_emb(tokens)  # (B, W, N, d)
        query = self.mask_emb.expand(batch, 1, n, d)  # the frame to predict, fully masked
        x = torch.cat([frames, query], dim=1)  # (B, W + 1, N, d)
        act = self.act_proj(torch.cat([actions, action[:, None]], dim=1))  # (B, W + 1, d)
        x = x + self.pos_emb[None, None] + self.time_emb[None, :, None] + act[:, :, None]
        hidden = self.transformer(x.reshape(batch, (window + 1) * n, d))
        hidden = self.out_norm(hidden[:, window * n :])  # (B, N, d) at the query positions
        reward = self.reward_head(self.reward_proj(hidden).flatten(1)).squeeze(-1)
        return self.token_head(hidden), reward

    def _sample(self, logits: torch.Tensor) -> torch.Tensor:
        """Token ids ``(B, N)`` from logits ``(B, N, K)``; greedy when ``temperature <= 0``."""
        if self.temperature <= 0.0:
            return logits.argmax(dim=-1)
        scaled = logits.float() / self.temperature
        return torch.distributions.Categorical(logits=scaled).sample()

    def step(self, state: State, action: torch.Tensor) -> State:
        """Imagine the next frame: one forward pass, all tokens sampled in parallel."""
        logits, reward = self._predict(state["tokens"], state["actions"], action)
        next_tokens = self._sample(logits)
        return {
            "tokens": torch.cat([state["tokens"][:, 1:], next_tokens[:, None]], dim=1),
            "actions": torch.cat([state["actions"][:, 1:], action[:, None]], dim=1),
            "reward": symexp(reward.float()),
        }

    def observe(
        self,
        obs: torch.Tensor,
        prev_actions: torch.Tensor,
        state: State | None = None,
    ) -> tuple[State, State]:
        """Filter observations into token windows.

        The posterior holds the *real* tokens of each frame; its reward is the one predicted
        by the dynamics before seeing the frame. The prior is the model's own sampled prediction
        of the frame (equal to the posterior for the very first frame of an episode).
        """
        tokens = self.tokenize(obs)  # (B, L, N)
        posteriors: list[State] = []
        priors: list[State] = []
        for t in range(tokens.shape[1]):
            action = prev_actions[:, t]
            if state is None:
                state = self._episode_start(tokens[:, t])
                prior = state
            else:
                prior = self.step(state, action)
                state = {
                    "tokens": torch.cat([state["tokens"][:, 1:], tokens[:, t, None]], dim=1),
                    "actions": torch.cat([state["actions"][:, 1:], action[:, None]], dim=1),
                    "reward": prior["reward"],
                }
            priors.append(prior)
            posteriors.append(state)
        return stack_states(posteriors), stack_states(priors)

    # ------------------------------------------------------------------- heads
    def decode(self, state: State) -> torch.Tensor:
        last = state["tokens"][..., -1, :]  # (..., N)
        lead = last.shape[:-1]
        images = self.tokenizer.decode_ids(last.reshape(-1, last.shape[-1]))
        return (images + 0.5).clamp(0.0, 1.0).reshape(*lead, *images.shape[1:])

    def predict_reward(self, state: State) -> torch.Tensor:
        return state["reward"]

    # ---------------------------------------------------------------- training
    @property
    def stage(self) -> str:
        """``"tokenizer"`` for the first ``tokenizer_steps`` loss calls, then ``"transformer"``."""
        return "tokenizer" if int(self.train_calls) < self.config.tokenizer_steps else "transformer"

    def loss(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, float]]:
        stage = self.stage
        self.train_calls.add_(1)
        if stage == "tokenizer":
            return self._tokenizer_loss(batch)
        return self._transformer_loss(batch)

    def _tokenizer_loss(
        self, batch: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, dict[str, float]]:
        cfg = self.config
        frames = batch["obs"].flatten(0, 1)
        if frames.shape[0] > cfg.tokenizer_batch:
            pick = torch.randperm(frames.shape[0], device=frames.device)[: cfg.tokenizer_batch]
            frames = frames[pick]
        x = self._preprocess(frames)
        recon, ids, commitment = self.tokenizer(x, update=self.training)
        squared = (recon - x).pow(2)
        weight: torch.Tensor | float = 1.0
        if cfg.fg_weight > 0:
            background = x.median(dim=0).values
            foreground = ((x - background).abs().amax(dim=1, keepdim=True) > 0.05).float()
            weight = 1.0 + cfg.fg_weight * foreground
        recon_loss = (weight * squared).mean()
        total = recon_loss + cfg.beta_commit * commitment

        with torch.no_grad():
            counts = torch.bincount(ids.flatten(), minlength=cfg.num_codes).float()
            probs = counts / counts.sum()
            perplexity = torch.exp(-(probs * torch.log(probs.clamp_min(1e-10))).sum())
        metrics = {
            "loss": total.item(),
            "stage": 0.0,
            "recon_loss": recon_loss.item(),
            "commitment": commitment.item(),
            "mse": squared.mean().item(),
            "code_perplexity": perplexity.item(),
            "codes_used": float((counts > 0).sum().item()),
        }
        return total, metrics

    def _make_windows(
        self, tokens: torch.Tensor, prev_actions: torch.Tensor, rewards: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Random (context window, query action, target frame, reward) training examples.

        A share ``pad_prob`` of the windows is rewritten into an "episode start" pattern
        (earliest frames replaced by copies of a later one, with zero actions), because that is
        what the model sees during the first steps of every episode at inference time.
        """
        cfg = self.config
        batch, length, _ = tokens.shape
        window = cfg.window
        if length <= window:
            raise ValueError(f"seq_len ({length}) must exceed the context window ({window}).")
        device = tokens.device
        m = cfg.window_batch
        b = torch.randint(0, batch, (m,), device=device)
        t = torch.randint(window, length, (m,), device=device)
        idx = t[:, None] + torch.arange(-window, 0, device=device)[None]  # (m, W)
        ctx_tokens = tokens[b[:, None], idx]  # (m, W, N)
        ctx_actions = prev_actions[b[:, None], idx]  # (m, W, A)
        if cfg.pad_prob > 0:
            augment = torch.rand(m, device=device) < cfg.pad_prob
            k = torch.randint(0, window, (m,), device=device)
            replace = (torch.arange(window, device=device)[None] <= k[:, None]) & augment[:, None]
            anchor = ctx_tokens[torch.arange(m, device=device), k]  # (m, N)
            ctx_tokens = torch.where(replace[..., None], anchor[:, None, :], ctx_tokens)
            zeros = torch.zeros_like(ctx_actions)
            ctx_actions = torch.where(replace[..., None], zeros, ctx_actions)
        return {
            "tokens": ctx_tokens,
            "actions": ctx_actions,
            "action": prev_actions[b, t],
            "target": tokens[b, t],
            "reward": rewards[b, t],
        }

    def _transformer_loss(
        self, batch: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, dict[str, float]]:
        cfg = self.config
        tokens = self.tokenize(batch["obs"])  # frozen tokenizer, no gradients
        windows = self._make_windows(tokens, batch["prev_actions"], batch["rewards"])
        logits, reward_pred = self._predict(
            windows["tokens"], windows["actions"], windows["action"]
        )
        token_loss = F.cross_entropy(
            logits.reshape(-1, cfg.num_codes), windows["target"].reshape(-1)
        )
        reward_loss = 0.5 * (reward_pred - symlog(windows["reward"])).pow(2).mean()
        total = token_loss + cfg.reward_scale * reward_loss
        with torch.no_grad():
            correct = logits.argmax(dim=-1) == windows["target"]
        metrics = {
            "loss": total.item(),
            "stage": 1.0,
            "token_loss": token_loss.item(),
            "reward_loss": reward_loss.item(),
            "token_acc": correct.float().mean().item(),
            "frame_acc": correct.all(dim=-1).float().mean().item(),
        }
        return total, metrics
