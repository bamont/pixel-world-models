"""Evaluate a trained checkpoint: open-loop drift metrics and a qualitative GIF.

Usage:
    python scripts/evaluate.py --checkpoint runs/rssm_point_reach_s0/checkpoints/last.pt
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
from pixelwm.utils.viz import comparison_frames, save_gif


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=30, help="Fresh evaluation episodes.")
    parser.add_argument("--context", type=int, default=5)
    parser.add_argument("--horizons", type=int, nargs="+", default=[1, 5, 15, 45])
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out", type=Path, default=None, help="Output dir (default: eval/).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = get_device(args.device)
    model, ckpt = load_checkpoint(args.checkpoint, device)
    env_cfg = ckpt["extra"]["env"]
    out_dir = args.out or args.checkpoint.parent.parent / "eval"

    env = make_env(env_cfg["name"], env_cfg["size"], env_cfg["max_steps"])
    buffer = collect_buffer(env, args.episodes, seed=args.seed)

    metrics = open_loop_eval(
        model, buffer, horizons=args.horizons, context=args.context, seed=args.seed
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    for key, value in metrics.items():
        print(f"{key:>10s}: {value:.4f}")

    real, dream = dream_episode(model, buffer.episodes[0], context=args.context)
    save_gif(comparison_frames(real, dream), out_dir / "dream.gif")
    print(f"Saved metrics and GIF to {out_dir}")


if __name__ == "__main__":
    main()
