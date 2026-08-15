#!/usr/bin/env python3
"""Fail-fast validation for expensive end-to-end runs."""

from __future__ import annotations

import os
from pathlib import Path
import tempfile

import torch


ROOT = Path(__file__).resolve().parent
RESULTS = Path(os.environ["RESULTS_ROOT"]).expanduser().resolve()
RUNTIME = Path(os.environ["RUNTIME_ROOT"]).expanduser().resolve()
DATASETS = tuple(value for value in os.environ["DATASETS"].split(",") if value)


def validate_checkpoint(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Missing AnomalyCLIP checkpoint: {path}")
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict) or "prompt_learner" not in payload:
        raise ValueError(f"Invalid AnomalyCLIP checkpoint payload: {path}")


def main() -> None:
    required_code = (
        ROOT / "generation" / "run_per_dataset.py",
        ROOT / "evaluation" / "universal_eval" / "runner.py",
        ROOT / "run_evaluations.py",
        ROOT / "path_contract.py",
    )
    for path in required_code:
        if not path.is_file():
            raise FileNotFoundError(f"Incomplete pipeline checkout: {path}")

    model_root = RUNTIME / "AnomalyCLIP"
    if not (model_root / "AnomalyCLIP_lib").is_dir():
        raise FileNotFoundError(f"Invalid AnomalyCLIP checkout: {model_root}")
    checkpoint_by_dataset = {
        "mvtec": model_root
        / "checkpoints"
        / "9_12_4_multiscale"
        / "epoch_15.pth",
        "visa": model_root
        / "checkpoints"
        / "9_12_4_multiscale_visa"
        / "epoch_15.pth",
    }
    for dataset in DATASETS:
        validate_checkpoint(checkpoint_by_dataset[dataset])

    RESULTS.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=RESULTS, prefix=".write_test_", delete=True):
        pass

    print("===== PREFLIGHT PASSED =====")
    print("Datasets:", DATASETS)
    print("Results root:", RESULTS)
    print("Runtime root:", RUNTIME)
    for dataset in DATASETS:
        print(f"{dataset} checkpoint:", checkpoint_by_dataset[dataset])


if __name__ == "__main__":
    main()

