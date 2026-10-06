# pixel-wm

**A modular benchmark of pixel-based world models.**
Learn a simulator of an environment from raw pixels and actions, then measure how well each
architecture *predicts the future* and (soon) *acts*, under identical data, environments and
compute budgets.

> **Status: v0.1 (alpha).** The shared interface, data pipeline, training loop, evaluation
> tools and the RSSM baseline are implemented. Token-transformer and JEPA models, and planning,
> are on the roadmap. No benchmark numbers are published yet.

## Why this repository?

Three families of world models are usually studied in isolation:

| Family | Latent space | Reconstructs pixels? | Status |
|---|---|---|---|
| RSSM (Dreamer-style) | GRU + categorical latents | yes | implemented |
| Discrete tokens + Transformer (IRIS / Genie-style) | VQ tokens | yes | planned |
| JEPA (latent prediction) | continuous embeddings | no | planned |

`pixel-wm` puts them behind one `WorldModel` interface to study:

1. Is pixel reconstruction necessary to plan well?
2. How does error accumulate over long rollouts, per architecture?
3. Does image-prediction quality (PSNR / SSIM / LPIPS) correlate with agent performance?

## Installation

```bash
git clone https://github.com/bamont/pixel-wm.git
cd pixel-wm
pip install -e ".[dev]"        # add ,metrics for LPIPS and ,envs for Gymnasium rendering
```

## Quick start

```bash
# 1. Train an RSSM on the built-in point_reach environment (data is collected automatically)
python scripts/train.py

# Quick CPU smoke run
python scripts/train.py device=cpu training.steps=500 data.train_episodes=50 training.eval_every=250

# 2. Evaluate a checkpoint: open-loop drift metrics + real-vs-dream GIF
python scripts/evaluate.py --checkpoint runs/rssm_point_reach_s0/checkpoints/last.pt
```

Configuration is handled by [Hydra](https://hydra.cc): any value in `configs/` can be
overridden from the command line (`model.deter=512 seed=1`).

## The `WorldModel` interface

```python
class WorldModel(nn.Module):
    def initial_state(self, batch_size) -> State: ...
    def encode(self, obs) -> Tensor: ...                       # pixels -> embedding
    def observe(self, obs, prev_actions, state=None): ...      # filtering -> (posterior, prior)
    def step(self, state, action) -> State: ...                # one imagined transition
    def imagine(self, state, actions) -> State: ...            # latent rollout
    def decode(self, state) -> Tensor: ...                     # latent -> pixels in [0, 1]
    def predict_reward(self, state) -> Tensor: ...
    def loss(self, batch) -> tuple[Tensor, dict[str, float]]: ...
```

Adding a model = subclass `WorldModel`, register it in `pixelwm/models/__init__.py`, add a YAML
file in `configs/model/`.

**Data convention.** At index `t` an episode stores the observation `obs[t]`, the action that
*led to* it (`prev_actions[t]`) and the reward received on arrival (`rewards[t]`). See
`pixelwm/data/buffer.py`.

## Evaluation

`open_loop_eval` observes a few context frames, imagines the rest with the ground-truth actions,
and reports `psnr@h`, `ssim@h`, `mse@h` for chosen horizons `h`, which exposes how rollouts drift.
Metrics are implemented in pure PyTorch; LPIPS is available as an optional extra.

## Environments

* `point_reach` (built-in, no dependencies): a dot with hidden momentum that must reach a goal.
  Fast on CPU, useful for debugging and tests.
* Any Gymnasium environment with an `rgb_array` render mode via the `gym-` prefix, e.g.
  `env=cartpole` (`gym-CartPole-v1`). DeepMind Control, Crafter and MiniGrid wrappers are planned.

## Repository layout

```
src/pixelwm/
├── envs/        # PointReach, Gymnasium pixel wrapper, factory
├── data/        # Episode storage, sequence sampling, collection
├── models/      # WorldModel interface, building blocks, RSSM, registry
├── training/    # Model-agnostic trainer
├── eval/        # PSNR / SSIM / LPIPS, open-loop drift evaluation
├── planning/    # (planned) CEM / MPC, actor-critic in imagination
└── utils/       # seeding, logging, checkpoints, GIFs
configs/         # Hydra configs      scripts/  # train / evaluate      tests/  # pytest
```

## Development

```bash
make all      # ruff + mypy + pytest
```

## Roadmap

- [x] v0.1: skeleton, data pipeline, trainer, evaluation, RSSM baseline
- [ ] v0.2: CEM planning and actor-critic in imagination, continue head
- [ ] v0.3: VQ tokenizer + transformer world model
- [ ] v0.4: JEPA with anti-collapse regularisation (EMA target, VICReg)
- [ ] v0.5: full benchmark table, interactive "play in the dream" demo, MkDocs site, Hugging Face checkpoints

## References

* Hafner et al., *Mastering Diverse Domains through World Models* (DreamerV3), 2023.
* Micheli et al., *Transformers are Sample-Efficient World Models* (IRIS), 2023.
* Bruce et al., *Genie: Generative Interactive Environments*, 2024.
* Zhou et al., *DINO-WM: World Models on Pre-trained Visual Features*, 2024.

## License

MIT, see [LICENSE](LICENSE).
