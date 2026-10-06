"""Train a world model.

Usage:
    python scripts/train.py                                  # RSSM on point_reach
    python scripts/train.py training.steps=2000 device=cpu   # Hydra overrides
    python scripts/train.py env=cartpole                     # needs `pip install -e ".[envs]"`
"""

from __future__ import annotations

from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from pixelwm.data import EpisodeBuffer, collect_buffer
from pixelwm.envs import get_action_dim, make_env
from pixelwm.eval import dream_episode
from pixelwm.models import build_model
from pixelwm.training import TrainConfig, Trainer
from pixelwm.utils import get_device, seed_everything
from pixelwm.utils.logging import get_logger
from pixelwm.utils.viz import save_dream_gif

VAL_SEED_OFFSET = 1_000_000


def load_or_collect(cfg: DictConfig, split: str, episodes: int, seed: int) -> EpisodeBuffer:
    """Load a cached dataset or collect it with a random policy."""
    path = Path(cfg.data.cache_dir) / f"{cfg.env.name}_{split}_{episodes}ep_seed{seed}.npz"
    if path.exists():
        return EpisodeBuffer.load(path)
    env = make_env(cfg.env.name, cfg.env.size, cfg.env.max_steps)
    buffer = collect_buffer(env, episodes, seed=seed)
    buffer.save(path)
    return buffer


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    logger = get_logger()
    seed_everything(cfg.seed)
    device = get_device(cfg.device)
    out_dir = Path(HydraConfig.get().runtime.output_dir)

    env = make_env(cfg.env.name, cfg.env.size, cfg.env.max_steps)
    action_dim = get_action_dim(env.action_space)
    train_buffer = load_or_collect(cfg, "train", cfg.data.train_episodes, cfg.seed)
    val_buffer = load_or_collect(cfg, "val", cfg.data.val_episodes, cfg.seed + VAL_SEED_OFFSET)
    logger.info(
        "Data: %d train / %d val episodes", train_buffer.num_episodes, val_buffer.num_episodes
    )

    model_dict = OmegaConf.to_container(cfg.model, resolve=True)
    assert isinstance(model_dict, dict)
    model_cfg = {k: v for k, v in model_dict.items() if k != "name"}
    model = build_model(cfg.model.name, action_dim, model_cfg).to(device)

    train_cfg = OmegaConf.to_container(cfg.training, resolve=True)
    assert isinstance(train_cfg, dict)
    train_cfg["eval_horizons"] = tuple(train_cfg["eval_horizons"])
    trainer = Trainer(
        model,
        train_buffer,
        TrainConfig(
            **train_cfg,
            seed=cfg.seed,
            eval_objects=bool(cfg.env.get("object_metrics", False)),
        ),
        out_dir,
        val_buffer=val_buffer,
        extra={"env": OmegaConf.to_container(cfg.env, resolve=True)},
    )
    trainer.fit()
    
    ctx = cfg.training.eval_context
    episode = val_buffer.episodes[0]
    real, dream = dream_episode(model, episode, context=ctx)
    save_dream_gif(
        real, dream, out_dir / "dream.gif",
        last_context=episode.obs[ctx - 1],
        trails=bool(cfg.env.get("object_metrics", False)),
    )
    logger.info("Done. Outputs written to %s", out_dir)


if __name__ == "__main__":
    main()
