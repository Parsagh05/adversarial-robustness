#!/usr/bin/env python3
"""Calibrate clean thresholds and evaluate every selected attack setup."""

from __future__ import annotations

import csv
import math
import os
from pathlib import Path

from evaluation import (
    EvaluationConfig,
    ThresholdCalibrationConfig,
    calibrate_thresholds,
    run_evaluation,
)
from evaluation.universal_eval.artifacts import load_manifest
from path_contract import bundle_path, ensure_bundle_protocol_files


ROOT = Path(__file__).resolve().parent
OUTPUT = Path(
    os.environ.get("RESULTS_ROOT") or os.environ["PIPELINE_OUTPUT"]
).expanduser().resolve()
RUNTIME = Path(os.environ["RUNTIME_ROOT"]).expanduser().resolve()
MVTEC_ROOT = Path(os.environ["MVTEC_ROOT"]).expanduser().resolve()
VISA_ROOT = Path(os.environ["VISA_ROOT"]).expanduser().resolve()
MODEL_ROOT = RUNTIME / "AnomalyCLIP"
RUN_SETUPS = os.environ.get("RUN_SETUPS", "all")
OVERWRITE = os.environ.get("OVERWRITE_EXISTING", "false").lower() in {
    "1", "true", "yes", "on"
}
SMOKE = os.environ.get("SMOKE_TEST", "false").lower() in {
    "1", "true", "yes", "on"
}
MAX_CONDITIONS = int(os.environ.get("SMOKE_MAX_CONDITIONS", "1")) if SMOKE else None
SAVE_PREDICTIONS = os.environ.get("SAVE_PREDICTIONS", "false").lower() in {
    "1", "true", "yes", "on"
}
DATASETS = tuple(
    value.strip() for value in os.environ.get("DATASETS", "mvtec,visa").split(",")
    if value.strip()
)
if not DATASETS or len(set(DATASETS)) != len(DATASETS) or set(DATASETS) - {"mvtec", "visa"}:
    raise ValueError("DATASETS must contain mvtec, visa, or both exactly once")

SETUPS = {
    "steps500_eps2": (500, 2 / 255),
    "steps500_eps4": (500, 4 / 255),
    "steps800_eps2": (800, 2 / 255),
    "steps800_eps4": (800, 4 / 255),
}
PIXEL_THRESHOLD_MODES = ("fixed_0_5", "image_f1", "clean_pixel_f1")


def selected_setup_ids() -> list[str]:
    if RUN_SETUPS == "all":
        return list(SETUPS)
    requested = [value.strip() for value in RUN_SETUPS.split(",") if value.strip()]
    unknown = sorted(set(requested) - set(SETUPS))
    if unknown:
        raise ValueError(f"Unknown RUN_SETUPS values: {unknown}")
    if not requested:
        raise ValueError("RUN_SETUPS selected nothing")
    return requested


def bundle_for(setup_id: str) -> Path:
    return bundle_path(OUTPUT, setup_id)


def validate_bundle(setup_id: str) -> tuple[Path, int, str]:
    expected_steps, expected_epsilon = SETUPS[setup_id]
    if SMOKE:
        expected_steps = int(os.environ.get("SMOKE_STEPS", "2"))
    bundle = bundle_for(setup_id)
    for repaired in ensure_bundle_protocol_files(bundle):
        print(f"[repair] copied protocol CSV into bundle: {repaired}")
    artifacts = load_manifest(
        bundle,
        scopes=("per_dataset",),
        sources=DATASETS,
        targets=DATASETS,
        verify_files=True,
    )
    expected_artifacts = 6 * len(DATASETS) * len(DATASETS)
    if len(artifacts) != expected_artifacts:
        raise RuntimeError(
            f"{setup_id} has {len(artifacts)} manifest rows; expected {expected_artifacts}"
        )
    protocol_hashes = {
        str(artifact.record.get("protocol_split_sha256", ""))
        for artifact in artifacts
    }
    if len(protocol_hashes) != 1 or not next(iter(protocol_hashes)):
        raise RuntimeError(f"{setup_id} does not have one consistent protocol hash")
    for artifact in artifacts:
        if int(artifact.record.get("optimization_steps", -1)) != expected_steps:
            raise RuntimeError(f"Unexpected step count in {artifact.name}")
        if not math.isclose(
            float(artifact.record["epsilon"]), expected_epsilon, abs_tol=1e-12
        ):
            raise RuntimeError(f"Unexpected epsilon in {artifact.name}")
        if artifact.record["scope"] != "per_dataset":
            raise RuntimeError(f"Non-per-dataset artifact found: {artifact.name}")
    return bundle, len(artifacts), next(iter(protocol_hashes))


def row_count(path: Path) -> int:
    if not path.is_file():
        return 0
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def main() -> None:
    setup_ids = selected_setup_ids()
    validated = {setup_id: validate_bundle(setup_id) for setup_id in setup_ids}
    protocol_hashes = {record[2] for record in validated.values()}
    if len(protocol_hashes) != 1:
        raise RuntimeError("Selected setups do not use the same train/evaluation split")
    protocol_hash = next(iter(protocol_hashes))
    all_model_kwargs = {
        "mvtec": {
            "repository_root": str(MODEL_ROOT),
            "checkpoint_path": str(
                MODEL_ROOT / "checkpoints" / "9_12_4_multiscale" / "epoch_15.pth"
            ),
            "clip_download_root": str(RUNTIME / "clip_cache"),
        },
        "visa": {
            "repository_root": str(MODEL_ROOT),
            "checkpoint_path": str(
                MODEL_ROOT
                / "checkpoints"
                / "9_12_4_multiscale_visa"
                / "epoch_15.pth"
            ),
            "clip_download_root": str(RUNTIME / "clip_cache"),
        },
    }
    model_kwargs = {dataset: all_model_kwargs[dataset] for dataset in DATASETS}
    for kwargs in model_kwargs.values():
        checkpoint = Path(kwargs["checkpoint_path"])
        if not checkpoint.is_file():
            raise FileNotFoundError(f"AnomalyCLIP checkpoint missing: {checkpoint}")

    threshold_root = OUTPUT / "calibrated_thresholds" / "anomalyclip"
    threshold_paths = {
        dataset: threshold_root / dataset / "category_thresholds.json"
        for dataset in DATASETS
    }
    threshold_protocol = threshold_root / "protocol_split_sha256.txt"
    thresholds_match_protocol = (
        threshold_protocol.is_file()
        and threshold_protocol.read_text(encoding="utf-8").strip() == protocol_hash
    )
    if (
        OVERWRITE
        or not thresholds_match_protocol
        or not all(path.is_file() for path in threshold_paths.values())
    ):
        first_setup = setup_ids[0]
        evaluation_index = validated[first_setup][0] / "evaluation_test_indices.csv"
        print("===== CALIBRATE CLEAN IMAGE F1-MAX THRESHOLDS =====")
        calibrate_thresholds(
            ThresholdCalibrationConfig(
                output_root=str(threshold_root),
                model_name="anomalyclip",
                model_kwargs_by_target=model_kwargs,
                datasets=DATASETS,
                mvtec_root=str(MVTEC_ROOT),
                visa_root=str(VISA_ROOT),
                device="cuda",
                batch_size=int(os.environ.get("EVALUATION_BATCH_SIZE", "2")),
                image_size=518,
                evaluation_index_path=str(evaluation_index),
                provenance="automatic_clean_evaluation_image_f1_for_ablation",
                run_metadata={"pipeline": ROOT.name, "setup_for_ids": first_setup},
            )
        )
        threshold_protocol.parent.mkdir(parents=True, exist_ok=True)
        threshold_protocol.write_text(protocol_hash + "\n", encoding="utf-8")
    else:
        print("[reuse] Calibrated clean image thresholds")

    threshold_config = {
        dataset: str(path) for dataset, path in threshold_paths.items()
    }
    for setup_id in setup_ids:
        bundle, artifact_count, _ = validated[setup_id]
        selected_count = min(artifact_count, MAX_CONDITIONS) if MAX_CONDITIONS else artifact_count
        mode_roots: dict[str, Path] = {}
        numerical_roots: dict[str, str] = {}
        qualitative_roots: dict[str, str] = {}
        all_complete = True
        for threshold_mode in PIXEL_THRESHOLD_MODES:
            mode_root = OUTPUT / "setups" / setup_id / "evaluation" / threshold_mode
            mode_roots[threshold_mode] = mode_root
            numerical = mode_root / "numerical"
            qualitative = mode_root / "visualizations"
            numerical_roots[threshold_mode] = str(numerical)
            qualitative_roots[threshold_mode] = str(qualitative)
            summary = numerical / "summary.csv"
            visualization_count = len(
                list(qualitative.glob("*/selection_manifest.json"))
            ) if qualitative.is_dir() else 0
            complete = (
                not OVERWRITE
                and row_count(summary) == selected_count
                and visualization_count == selected_count
            )
            all_complete = all_complete and complete
        if all_complete:
            print(f"[reuse] {setup_id}/all_threshold_modes")
            continue
        print(f"===== EVALUATE {setup_id}/ALL THRESHOLDS (SHARED INFERENCE) =====")
        run_evaluation(
            EvaluationConfig(
                artifacts_root=str(bundle),
                output_root=numerical_roots[PIXEL_THRESHOLD_MODES[0]],
                model_name="anomalyclip",
                model_kwargs_by_target=model_kwargs,
                thresholds_by_target=threshold_config,
                mvtec_root=str(MVTEC_ROOT),
                visa_root=str(VISA_ROOT),
                device="cuda",
                batch_size=int(os.environ.get("EVALUATION_BATCH_SIZE", "2")),
                metric_size=518,
                anomaly_map_sigma=4.0,
                aupro_fpr_limit=0.30,
                aupro_max_thresholds=200,
                verify_checksums=True,
                save_predictions=SAVE_PREDICTIONS,
                save_qualitative_samples=True,
                qualitative_output_root=qualitative_roots[PIXEL_THRESHOLD_MODES[0]],
                source_datasets=DATASETS,
                target_datasets=DATASETS,
                attack_scopes=("per_dataset",),
                max_conditions=MAX_CONDITIONS,
                pixel_success_min_flip_fraction=float(
                    os.environ.get("PIXEL_SUCCESS_MIN_FLIP_FRACTION", "0.50")
                ),
                pixel_threshold_mode=PIXEL_THRESHOLD_MODES[0],
                pixel_threshold_modes=PIXEL_THRESHOLD_MODES,
                output_roots_by_pixel_threshold_mode=numerical_roots,
                qualitative_output_roots_by_pixel_threshold_mode=qualitative_roots,
                qualitative_selection_basis="target_region_pixel",
                run_notes=(
                    f"Per-dataset ablation {setup_id}; all pixel threshold modes "
                    "share one inference pass; visualizations are for debugging."
                ),
            )
        )


if __name__ == "__main__":
    main()
