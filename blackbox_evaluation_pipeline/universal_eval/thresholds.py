"""Leakage-free model threshold calibration on normal training images."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from tqdm.auto import tqdm

from .adapters import build_adapter
from .datasets import EvaluationSample, discover_calibration_dataset, load_image


def _normalized_model_name(value: str) -> str:
    return "".join(character for character in value.lower() if character.isalnum())


def load_category_thresholds(
    path: str | Path,
    *,
    expected_dataset: str | None = None,
    expected_model: str | None = None,
) -> dict[str, float]:
    """Load and validate a frozen per-category image-score threshold artifact."""

    threshold_path = Path(path).expanduser().resolve()
    if not threshold_path.is_file():
        raise FileNotFoundError(f"Threshold artifact not found: {threshold_path}")
    payload = json.loads(threshold_path.read_text(encoding="utf-8"))
    dataset = str(payload.get("dataset", ""))
    model = str(payload.get("target_model", ""))
    if expected_dataset is not None and dataset != expected_dataset:
        raise ValueError(
            f"Threshold dataset mismatch: expected {expected_dataset!r}, got {dataset!r}"
        )
    if (
        expected_model is not None
        and _normalized_model_name(model) != _normalized_model_name(expected_model)
    ):
        raise ValueError(
            f"Threshold model mismatch: expected {expected_model!r}, got {model!r}"
        )
    if payload.get("threshold_mode") != "normal_train_quantile":
        raise ValueError("Only frozen normal-training quantile thresholds are supported")
    records = payload.get("categories")
    if not isinstance(records, dict) or not records:
        raise ValueError("Threshold artifact must contain non-empty category records")
    thresholds: dict[str, float] = {}
    for category, record in records.items():
        if not isinstance(record, dict) or "threshold" not in record:
            raise ValueError(f"Invalid threshold record for category {category!r}")
        threshold = float(record["threshold"])
        if not np.isfinite(threshold):
            raise ValueError(f"Non-finite threshold for category {category!r}")
        thresholds[str(category)] = threshold
    return thresholds


@dataclass
class ThresholdCalibrationConfig:
    output_root: str
    model_name: str
    model_kwargs_by_target: dict[str, dict[str, Any]]
    datasets: tuple[str, ...] = ("mvtec", "visa")
    mvtec_root: str | None = None
    visa_root: str | None = None
    device: str = "cuda"
    batch_size: int = 2
    image_size: int = 518
    quantile: float = 0.95
    provenance: str = "custom_normal_train_quantile_no_official_threshold"
    official_model_threshold: bool = False
    run_metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.datasets:
            raise ValueError("At least one calibration dataset is required")
        unknown = set(self.datasets) - {"mvtec", "visa"}
        if unknown:
            raise ValueError(f"Unsupported calibration datasets: {sorted(unknown)}")
        if len(set(self.datasets)) != len(self.datasets):
            raise ValueError("Calibration datasets must not contain duplicates")
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")
        if not 0.0 < self.quantile < 1.0:
            raise ValueError("quantile must be in (0, 1)")


def _chunks(
    samples: list[EvaluationSample], size: int
) -> Iterable[list[EvaluationSample]]:
    for index in range(0, len(samples), size):
        yield samples[index : index + size]


def _predict_scores(
    adapter: Any,
    samples: list[EvaluationSample],
    *,
    image_size: int,
    batch_size: int,
    description: str,
) -> np.ndarray:
    scores: list[float] = []
    batches = list(_chunks(samples, batch_size))
    for batch in tqdm(batches, desc=description, leave=False):
        images = torch.stack([load_image(sample, image_size) for sample in batch])
        batch_scores, _ = adapter.predict(images)
        if len(batch_scores) != len(batch):
            raise RuntimeError("Model adapter returned a different batch length")
        scores.extend(float(score) for score in batch_scores)
    result = np.asarray(scores, dtype=np.float32)
    if result.shape != (len(samples),) or not np.isfinite(result).all():
        raise ValueError("Calibration scores must be finite and one per sample")
    return result


def _json_config(config: ThresholdCalibrationConfig) -> dict[str, Any]:
    return json.loads(json.dumps(asdict(config), default=str))


def calibrate_thresholds(
    config: ThresholdCalibrationConfig,
) -> dict[str, Path]:
    """Calibrate q-thresholds and return one JSON artifact path per dataset."""

    if config.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable; enable a Kaggle GPU")
    output_root = Path(config.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    generated: dict[str, Path] = {}

    for dataset in config.datasets:
        print(f"[calibration] Discovering normal training images for {dataset}")
        samples = discover_calibration_dataset(
            dataset,
            mvtec_root=config.mvtec_root,
            visa_root=config.visa_root,
        )
        kwargs = dict(config.model_kwargs_by_target.get(dataset, {}))
        if not kwargs:
            raise ValueError(f"No model configuration supplied for {dataset!r}")
        kwargs.setdefault("device", config.device)
        kwargs.setdefault("image_size", config.image_size)
        print(f"[model] Loading {config.model_name} for calibration target={dataset}")
        adapter = build_adapter(config.model_name, **kwargs)
        try:
            scores = _predict_scores(
                adapter,
                samples,
                image_size=config.image_size,
                batch_size=config.batch_size,
                description=f"q{config.quantile:g} {dataset}",
            )
        finally:
            adapter.release()

        categories: dict[str, dict[str, Any]] = {}
        for category in sorted({sample.category for sample in samples}):
            indices = [
                index for index, sample in enumerate(samples) if sample.category == category
            ]
            category_scores = scores[indices].astype(np.float64)
            categories[category] = {
                "dataset": dataset,
                "threshold": float(np.quantile(category_scores, config.quantile)),
                "sample_count": len(indices),
                "score_min": float(category_scores.min()),
                "score_mean": float(category_scores.mean()),
                "score_max": float(category_scores.max()),
                "score_std": float(category_scores.std()),
                "calibration_sample_ids": [samples[index].protocol_id for index in indices],
            }

        dataset_output = output_root / dataset
        dataset_output.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": 2,
            "target_model": config.model_name,
            "dataset": dataset,
            "image_size": config.image_size,
            "calibration_split": "normal training split",
            "threshold_mode": "normal_train_quantile",
            "threshold_quantile": config.quantile,
            "provenance": config.provenance,
            "official_model_threshold": config.official_model_threshold,
            "categories": categories,
        }
        threshold_path = dataset_output / "category_thresholds.json"
        threshold_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        np.savez_compressed(
            dataset_output / "normal_train_scores.npz",
            sample_ids=np.asarray([sample.protocol_id for sample in samples]),
            categories=np.asarray([sample.category for sample in samples]),
            scores=scores,
        )
        config_payload = _json_config(config)
        config_payload["calibrated_dataset"] = dataset
        (dataset_output / "threshold_config.json").write_text(
            json.dumps(config_payload, indent=2), encoding="utf-8"
        )
        generated[dataset] = threshold_path
        print(f"[done] {dataset}: {threshold_path}")
    return generated
