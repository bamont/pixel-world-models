"""Plan with CEM in a trained world model and measure the *real* return.

Usage:
    python scripts/plan.py --checkpoint runs/<run>/checkpoints/last.pt
    python scripts/plan.py --checkpoint ... --episodes 50 --horizon 20 --gif

The output ``planning.json`` contains, for the CEM planner and for the random / oracle
reference policies (same episodes, same seeds): per-episode returns and final distances,
their means, the success rate and the wall-clock time per step.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from pixelwm.envs import make_env
from pixelwm.planning import CEMPlanner, OraclePolicy, RandomPolicy, evaluate_policy
from pixelwm.utils import get_device, seed_everything
from pixelwm.utils.checkpoint import load_checkpoint
from pixelwm.utils.viz import save_gif


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--seed", type=int, default=1000, help="First episode seed.")
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--elites", type=int, default=32)
    parser.add_argument("--iterations", type=int, default=4)
    parser.add_argument("--init-std", type=float, default=0.5)
    parser.add_argument("--discount", type=float, default=0.99)
    parser.add_argument("--no-warm-start", action="store_true")
    parser.add_argument("--no-baselines", action="store_true", help="Skip random / oracle runs.")
    parser.add_argument("--gif", action="store_true", help="Save the first controlled episode.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out", type=Path, default=None, help="Output dir (default: planning/).")
    return parser.parse_args()


def summarise(result: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in result.items() if k != "frames"}


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = get_device(args.device)
    model, ckpt = load_checkpoint(args.checkpoint, device)
    env_cfg = ckpt["extra"]["env"]
    out_dir = args.out or args.checkpoint.parent.parent / "planning"
    out_dir.mkdir(parents=True, exist_ok=True)

    env = make_env(env_cfg["name"], env_cfg["size"], env_cfg["max_steps"])
    planner = CEMPlanner(
        model,
        horizon=args.horizon,
        num_samples=args.samples,
        num_elites=args.elites,
        iterations=args.iterations,
        init_std=args.init_std,
        discount=args.discount,
        warm_start=not args.no_warm_start,
    )
    results: dict[str, Any] = {}
    cem = evaluate_policy(env, planner, args.episodes, args.seed, record_first=args.gif)
    if args.gif:
        save_gif(cem["frames"], out_dir / "plan.gif", fps=10, scale=4)
    results["cem"] = summarise(cem)

    if not args.no_baselines:
        results["random"] = evaluate_policy(
            env, RandomPolicy(env.action_space, args.seed), args.episodes, args.seed
        )
        if env_cfg["name"] == "point_reach":
            results["oracle"] = evaluate_policy(env, OraclePolicy(env), args.episodes, args.seed)

    payload = {"settings": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
               "results": results}
    (out_dir / "planning.json").write_text(json.dumps(payload, indent=2))
    for name, res in results.items():
        print(
            f"{name:>7s}: return {res['return_mean']:8.2f} ± {res['return_std']:5.2f} | "
            f"final dist {res['final_distance_mean']:.3f} | success {res['success_rate']:.2f} | "
            f"{1000 * res['seconds_per_step']:.1f} ms/step"
        )
    print(f"Saved results to {out_dir}")


if __name__ == "__main__":
    main()
