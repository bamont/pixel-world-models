"""Evaluate a trained checkpoint: open-loop drift metrics and a qualitative GIF.

Usage:
    python scripts/evaluate.py --checkpoint runs/rssm_point_reach_s0/checkpoints/last.pt
    python scripts/evaluate.py --checkpoint ... --episodes 100 --sequences 256 --objects
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pixelwm.data import collect_buffer
from pixelwm.envs import make_env
from pixelwm.eval import dream_episode, open_loop_eval
from pixelwm.utils import get_device, seed_everything
from pixelwm.utils.checkpoint import load_checkpoint
from pixelwm.utils.logging import finite_or_none
from pixelwm.utils.viz import save_dream_gif


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=30, help="Fresh evaluation episodes.")
    parser.add_argument("--sequences", type=int, default=256, help="Sequences sampled from them.")
    parser.add_argument("--context", type=int, default=5)
    parser.add_argument("--horizons", type=int, nargs="+", default=[1, 5, 15, 45])
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--objects",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Object-localisation metrics and trajectory panel (point_reach-like images). "
        "Default: taken from the environment config stored in the checkpoint, else off.",
    )
    parser.add_argument("--out", type=Path, default=None, help="Output dir (default: eval/).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = get_device(args.device)
    model, ckpt = load_checkpoint(args.checkpoint, device)
    env_cfg = ckpt["extra"]["env"]
    objects = args.objects if args.objects is not None else bool(env_cfg.get("object_metrics"))
    out_dir = args.out or args.checkpoint.parent.parent / "eval"

    env = make_env(env_cfg["name"], env_cfg["size"], env_cfg["max_steps"])
    buffer = collect_buffer(env, args.episodes, seed=args.seed)

    metrics = open_loop_eval(
        model,
        buffer,
        horizons=args.horizons,
        context=args.context,
        num_sequences=args.sequences,
        seed=args.seed,
        object_metrics=objects,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    clean = {k: finite_or_none(v) for k, v in metrics.items()}
    (out_dir / "metrics.json").write_text(json.dumps(clean, indent=2))
    for key, value in metrics.items():
        print(f"{key:>16s}: {value:.4f}")

    episode = buffer.episodes[0]
    real, dream = dream_episode(model, episode, context=args.context)
    save_dream_gif(
        real,
        dream,
        out_dir / "dream.gif",
        last_context=episode.obs[args.context - 1],
        trails=objects,
    )
    print(f"Saved metrics and GIF to {out_dir}")


if __name__ == "__main__":
    main()
