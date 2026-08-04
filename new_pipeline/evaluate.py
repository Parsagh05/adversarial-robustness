"""Command-line entry point for JSON-configured evaluation runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .universal_eval.runner import EvaluationConfig, run_evaluation


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate canonical fixed perturbations on an anomaly model"
    )
    parser.add_argument(
        "--config",
        required=True,
        help="JSON file containing EvaluationConfig fields",
    )
    args = parser.parse_args()
    config_path = Path(args.config).expanduser().resolve()
    data = json.loads(config_path.read_text(encoding="utf-8"))
    summary = run_evaluation(EvaluationConfig(**data))
    print(summary)


if __name__ == "__main__":
    main()

