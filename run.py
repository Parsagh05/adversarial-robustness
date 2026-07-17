"""Command-line entrypoint for the MVTec adversarial benchmark."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from adversarial_harness import AttackConfig, ExperimentConfig, run_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Adversarial robustness evaluation for zero-shot anomaly detection"
    )
    parser.add_argument("--mvtec-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--anomalyclip-root", required=True)
    parser.add_argument("--anomalyclip-checkpoint", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--categories", nargs="*")
    parser.add_argument("--scopes", nargs="+", default=list(AttackConfig().scopes))
    parser.add_argument("--directions", nargs="+", default=list(AttackConfig().directions))
    parser.add_argument("--loss-modes", nargs="+", default=list(AttackConfig().loss_modes))
    parser.add_argument("--epsilon", type=float, default=8.0 / 255.0)
    parser.add_argument("--step-size", type=float, default=2.0 / 255.0)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--universal-steps", type=int, default=200)
    parser.add_argument("--target-batch-size", type=int, default=2)
    parser.add_argument("--attack-batch-size", type=int, default=1)
    parser.add_argument("--universal-batch-size", type=int, default=2)
    parser.add_argument(
        "--universal-protocol",
        choices=("transductive", "held_out"),
        default="transductive",
    )
    parser.add_argument("--fit-fraction", type=float, default=0.5)
    parser.add_argument("--split-seed", type=int, default=111)
    parser.add_argument("--diagnostic-max-samples", type=int, default=64)
    parser.add_argument(
        "--threshold-mode",
        choices=("normal_train_quantile",),
        default="normal_train_quantile",
    )
    parser.add_argument("--threshold-quantile", type=float, default=0.95)
    parser.add_argument("--thresholds-path")
    parser.add_argument("--max-samples-per-category", type=int)
    parser.add_argument("--save-adversarial-examples", type=int, default=0)
    parser.add_argument("--no-lpips", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    attack = AttackConfig(
        epsilon=args.epsilon,
        step_size=args.step_size,
        steps=args.steps,
        universal_steps=args.universal_steps,
        scopes=tuple(args.scopes),
        directions=tuple(args.directions),
        loss_modes=tuple(args.loss_modes),
        per_image_batch_size=args.attack_batch_size,
        universal_batch_size=args.universal_batch_size,
    )
    config = ExperimentConfig(
        mvtec_root=args.mvtec_root,
        output_root=args.output_root,
        anomalyclip_root=args.anomalyclip_root,
        anomalyclip_checkpoint=args.anomalyclip_checkpoint,
        device=args.device,
        categories=tuple(args.categories) if args.categories else None,
        target_batch_size=args.target_batch_size,
        compute_lpips=not args.no_lpips,
        save_adversarial_examples=args.save_adversarial_examples,
        universal_protocol=args.universal_protocol,
        fit_fraction=args.fit_fraction,
        split_seed=args.split_seed,
        diagnostic_max_samples=args.diagnostic_max_samples,
        threshold_mode=args.threshold_mode,
        threshold_quantile=args.threshold_quantile,
        thresholds_path=args.thresholds_path,
        max_samples_per_category=args.max_samples_per_category,
        resume=not args.no_resume,
        attack=attack,
    )
    run_experiment(config)


if __name__ == "__main__":
    main()
