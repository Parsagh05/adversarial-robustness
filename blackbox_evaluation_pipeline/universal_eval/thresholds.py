"""Clean F1-optimal operating points for benchmark evaluation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import csv
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from tqdm.auto import tqdm

from .adapters import build_adapter
from .datasets import EvaluationSample, discover_dataset, index_samples, load_image
from .metrics import optimal_f1_operating_point


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
    if payload.get("threshold_mode") != "clean_f1_optimal":
        raise ValueError("Only frozen clean F1-optimal thresholds are supported")
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
    evaluation_index_path: str | None = None
    provenance: str = "clean_evaluation_f1_optimal_following_crane"
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
        if self.evaluation_index_path is not None and not Path(
            self.evaluation_index_path
        ).is_file():
            raise FileNotFoundError(
                f"Evaluation index not found: {self.evaluation_index_path}"
            )


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
    map_mins: list[float] = []
    map_maxs: list[float] = []
    batches = list(_chunks(samples, batch_size))
    for batch in tqdm(batches, desc=description, leave=False):
        images = torch.stack([load_image(sample, image_size) for sample in batch])
        batch_scores, batch_maps = adapter.predict_with_categories(
            images, [sample.category for sample in batch]
        )
        if len(batch_scores) != len(batch) or len(batch_maps) != len(batch):
            raise RuntimeError("Model adapter returned a different batch length")
        scores.extend(float(score) for score in batch_scores)
        map_mins.extend(float(np.asarray(anomaly_map).min()) for anomaly_map in batch_maps)
        map_maxs.extend(float(np.asarray(anomaly_map).max()) for anomaly_map in batch_maps)
    result = adapter.postprocess_image_scores(
        np.asarray(scores, dtype=np.float32),
        np.asarray(map_mins, dtype=np.float32),
        np.asarray(map_maxs, dtype=np.float32),
        [sample.category for sample in samples],
    )
    if result.shape != (len(samples),) or not np.isfinite(result).all():
        raise ValueError("Calibration scores must be finite and one per sample")
    return result


def _json_config(config: ThresholdCalibrationConfig) -> dict[str, Any]:
    return json.loads(json.dumps(asdict(config), default=str))


def _select_evaluation_samples(
    samples: list[EvaluationSample],
    *,
    dataset: str,
    evaluation_index_path: str | None,
) -> list[EvaluationSample]:
    """Select the fixed final-evaluation cohort in CSV order when configured."""

    if evaluation_index_path is None:
        return samples
    path = Path(evaluation_index_path).expanduser().resolve()
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    required = {"protocol_id", "dataset", "category", "label", "partition"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"Evaluation index must contain {sorted(required)}: {path}")
    selected_rows = [
        row
        for row in rows
        if row["dataset"] == dataset and row["partition"] == "evaluation"
    ]
    if not selected_rows:
        raise ValueError(f"No fixed evaluation rows found for {dataset!r} in {path}")
    sample_index = index_samples(samples)
    selected: list[EvaluationSample] = []
    seen: set[str] = set()
    for row in selected_rows:
        sample_id = row["protocol_id"]
        if sample_id in seen:
            raise ValueError(f"Duplicate evaluation protocol ID in {path}: {sample_id}")
        seen.add(sample_id)
        sample = sample_index.get(sample_id)
        if sample is None:
            raise ValueError(f"Evaluation ID is absent from {dataset}: {sample_id}")
        if sample.category != row["category"] or sample.label != int(row["label"]):
            raise ValueError(f"Evaluation metadata mismatch for {sample_id}")
        selected.append(sample)
    return selected


def calibrate_thresholds(
    config: ThresholdCalibrationConfig,
) -> dict[str, Path]:
    """Calibrate clean F1-optimal thresholds, one JSON artifact per dataset."""

    if config.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable; enable a Kaggle GPU")
    output_root = Path(config.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    generated: dict[str, Path] = {}

    for dataset in config.datasets:
        print(f"[calibration] Discovering clean labeled evaluation images for {dataset}")
        samples = discover_dataset(
            dataset,
            mvtec_root=config.mvtec_root,
            visa_root=config.visa_root,
        )
        samples = _select_evaluation_samples(
            samples,
            dataset=dataset,
            evaluation_index_path=config.evaluation_index_path,
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
                description=f"clean F1-optimal {dataset}",
            )
        finally:
            adapter.release()

        categories: dict[str, dict[str, Any]] = {}
        for category in sorted({sample.category for sample in samples}):
            indices = [
                index for index, sample in enumerate(samples) if sample.category == category
            ]
            category_scores = scores[indices].astype(np.float64)
            category_labels = np.asarray(
                [samples[index].label for index in indices], dtype=np.uint8
            )
            operating_point = optimal_f1_operating_point(
                category_labels, category_scores
            )
            categories[category] = {
                "dataset": dataset,
                "threshold": operating_point["threshold"],
                "f1_max": 100.0 * operating_point["f1"],
                "precision_at_threshold": 100.0 * operating_point["precision"],
                "recall_at_threshold": 100.0 * operating_point["recall"],
                "sample_count": len(indices),
                "normal_count": int((category_labels == 0).sum()),
                "anomaly_count": int((category_labels == 1).sum()),
                "score_min": float(category_scores.min()),
                "score_mean": float(category_scores.mean()),
                "score_max": float(category_scores.max()),
                "score_std": float(category_scores.std()),
                "calibration_sample_ids": [
                    samples[index].protocol_id for index in indices
                ],
            }

        dataset_output = output_root / dataset
        dataset_output.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": 2,
            "target_model": config.model_name,
            "dataset": dataset,
            "image_size": config.image_size,
            "calibration_split": "clean labeled evaluation split",
            "threshold_mode": "clean_f1_optimal",
            "provenance": config.provenance,
            "official_model_threshold": config.official_model_threshold,
            "uses_test_labels": True,
            "evaluation_index_path": config.evaluation_index_path,
            "run_metadata": config.run_metadata,
            "categories": categories,
        }
        threshold_path = dataset_output / "category_thresholds.json"
        threshold_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        np.savez_compressed(
            dataset_output / "clean_evaluation_scores.npz",
            sample_ids=np.asarray([sample.protocol_id for sample in samples]),
            categories=np.asarray([sample.category for sample in samples]),
            labels=np.asarray([sample.label for sample in samples], dtype=np.uint8),
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
