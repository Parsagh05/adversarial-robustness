"""Shared, model-agnostic evaluation runner for fixed canonical attacks."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
import csv
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from .adapters import build_adapter
from .artifacts import AttackArtifact, json_safe_record, load_manifest
from .datasets import (
    EvaluationSample,
    discover_dataset,
    index_samples,
    load_image,
    load_mask,
)
from .metrics import (
    binary_classification_metrics,
    image_metrics,
    pixel_metrics,
    resize_anomaly_maps,
    targeted_attack_metrics,
    targeted_region_pixel_metrics,
)
from .qualitative import export_representative_samples
from .thresholds import load_category_thresholds


@dataclass
class EvaluationConfig:
    artifacts_root: str
    output_root: str
    model_name: str
    model_kwargs_by_target: dict[str, dict[str, Any]]
    thresholds_by_target: dict[str, str] = field(default_factory=dict)
    mvtec_root: str | None = None
    visa_root: str | None = None
    device: str = "cuda"
    batch_size: int = 2
    metric_size: int = 518
    anomaly_map_sigma: float = 4.0
    aupro_fpr_limit: float = 0.30
    aupro_max_thresholds: int = 200
    verify_checksums: bool = True
    save_predictions: bool = True
    prediction_map_size: int | None = None
    save_qualitative_samples: bool = False
    qualitative_output_root: str | None = None
    source_datasets: tuple[str, ...] | None = None
    target_datasets: tuple[str, ...] | None = None
    attack_scopes: tuple[str, ...] = ("per_dataset",)
    attack_categories: tuple[str, ...] | None = None
    attack_directions: tuple[str, ...] | None = None
    attack_loss_modes: tuple[str, ...] | None = None
    condition_names: tuple[str, ...] | None = None
    max_conditions: int | None = None
    run_notes: str = ""
    pixel_success_min_flip_fraction: float = 0.5
    pixel_threshold_mode: str = "clean_pixel_f1"
    qualitative_selection_basis: str = "image"
    pixel_threshold_modes: tuple[str, ...] | None = None
    output_roots_by_pixel_threshold_mode: dict[str, str] = field(default_factory=dict)
    qualitative_output_roots_by_pixel_threshold_mode: dict[str, str] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")
        if self.metric_size < 1:
            raise ValueError("metric_size must be positive")
        if self.prediction_map_size is not None and self.prediction_map_size < 1:
            raise ValueError("prediction_map_size must be positive when supplied")
        if not 0.0 < self.aupro_fpr_limit <= 1.0:
            raise ValueError("aupro_fpr_limit must be in (0, 1]")
        if self.max_conditions is not None and self.max_conditions < 1:
            raise ValueError("max_conditions must be positive when supplied")
        if not self.attack_scopes:
            raise ValueError("attack_scopes must select at least one scope")
        if not 0.0 <= self.pixel_success_min_flip_fraction <= 1.0:
            raise ValueError("pixel_success_min_flip_fraction must be in [0, 1]")
        valid_pixel_threshold_modes = {
            "fixed_0_5",
            "image_f1",
            "clean_pixel_f1",
        }
        if self.pixel_threshold_mode not in valid_pixel_threshold_modes:
            raise ValueError(
                "pixel_threshold_mode must be fixed_0_5, image_f1, or clean_pixel_f1"
            )
        if self.pixel_threshold_modes is not None:
            if not self.pixel_threshold_modes:
                raise ValueError("pixel_threshold_modes must not be empty")
            if len(set(self.pixel_threshold_modes)) != len(self.pixel_threshold_modes):
                raise ValueError("pixel_threshold_modes must not contain duplicates")
            unknown = set(self.pixel_threshold_modes) - valid_pixel_threshold_modes
            if unknown:
                raise ValueError(f"Unknown pixel threshold modes: {sorted(unknown)}")
            expected = set(self.pixel_threshold_modes)
            if set(self.output_roots_by_pixel_threshold_mode) != expected:
                raise ValueError(
                    "output_roots_by_pixel_threshold_mode must define every selected mode"
                )
            if self.save_qualitative_samples and set(
                self.qualitative_output_roots_by_pixel_threshold_mode
            ) != expected:
                raise ValueError(
                    "qualitative_output_roots_by_pixel_threshold_mode must define every "
                    "selected mode when saving qualitative samples"
                )
        if self.qualitative_selection_basis not in {"image", "target_region_pixel"}:
            raise ValueError(
                "qualitative_selection_basis must be image or target_region_pixel"
            )
        if (
            self.save_qualitative_samples
            and self.pixel_threshold_modes is None
            and not self.qualitative_output_root
        ):
            raise ValueError(
                "qualitative_output_root is required when saving qualitative samples"
            )
        if self.save_qualitative_samples and not self.thresholds_by_target:
            raise ValueError(
                "Frozen thresholds are required to select qualitative successes/failures"
            )


Prediction = tuple[float, np.ndarray]


def _fixed_normal_target_region(
    artifact: AttackArtifact, height: int, width: int
) -> np.ndarray:
    """Recreate the fixed normal-image region recorded by attack generation."""

    target_mode = str(artifact.record.get("normal_local_target") or "full_image")
    if target_mode == "full_image":
        return np.ones((height, width), dtype=bool)
    if target_mode != "fixed_region":
        raise ValueError(f"Unknown normal local target mode: {target_mode!r}")
    fraction = float(artifact.record.get("normal_target_region_fraction", 0.25))
    center_x = float(artifact.record.get("normal_target_center_x", 0.5))
    center_y = float(artifact.record.get("normal_target_center_y", 0.5))
    if not 0.0 < fraction <= 1.0:
        raise ValueError("normal_target_region_fraction must be in (0, 1]")
    if not 0.0 <= center_x <= 1.0 or not 0.0 <= center_y <= 1.0:
        raise ValueError("normal target center coordinates must be in [0, 1]")
    region_height = max(1, int(round(height * fraction)))
    region_width = max(1, int(round(width * fraction)))
    center_column = int(round(center_x * (width - 1)))
    center_row = int(round(center_y * (height - 1)))
    left = min(max(center_column - region_width // 2, 0), width - region_width)
    top = min(max(center_row - region_height // 2, 0), height - region_height)
    region = np.zeros((height, width), dtype=bool)
    region[top : top + region_height, left : left + region_width] = True
    return region


def _target_region(
    artifact: AttackArtifact, ground_truth_mask: np.ndarray
) -> np.ndarray:
    """Return the evaluation region that the local targeted attack intends to flip."""

    mask = np.asarray(ground_truth_mask) > 0
    if int(artifact.record["target_label"]) == 1:
        return _fixed_normal_target_region(artifact, *mask.shape)
    return mask


def _chunks(items: list[EvaluationSample], size: int) -> Iterable[list[EvaluationSample]]:
    for index in range(0, len(items), size):
        yield items[index : index + size]


def _predict_clean(
    adapter: Any,
    samples: list[EvaluationSample],
    *,
    image_size: int,
    batch_size: int,
    description: str,
) -> dict[str, Prediction]:
    predictions: dict[str, Prediction] = {}
    batches = list(_chunks(samples, batch_size))
    for batch in tqdm(batches, desc=description, leave=False):
        images = torch.stack([load_image(sample, image_size) for sample in batch])
        scores, maps = adapter.predict_with_categories(
            images, [sample.category for sample in batch]
        )
        if len(scores) != len(batch) or len(maps) != len(batch):
            raise RuntimeError("Model adapter returned a different batch length")
        for sample, score, anomaly_map in zip(batch, scores, maps):
            predictions[sample.protocol_id] = (
                float(score), np.asarray(anomaly_map, dtype=np.float32)
            )
    return predictions


def _predict_adversarial(
    adapter: Any,
    samples: list[EvaluationSample],
    delta: torch.Tensor,
    attacked_ids: set[str],
    delta_indices: dict[str, int],
    *,
    image_size: int,
    batch_size: int,
    description: str,
) -> tuple[dict[str, Prediction], dict[str, float]]:
    predictions: dict[str, Prediction] = {}
    actual_linf: dict[str, float] = {}
    batches = list(_chunks(samples, batch_size))
    for batch in tqdm(batches, desc=description, leave=False):
        clean = torch.stack([load_image(sample, image_size) for sample in batch])
        adversarial = clean.clone()
        attacked_indices = [
            index
            for index, sample in enumerate(batch)
            if sample.protocol_id in attacked_ids
        ]
        if attacked_indices:
            perturbations = torch.stack(
                [delta[delta_indices[batch[index].protocol_id]] for index in attacked_indices]
            )
            adversarial[attacked_indices] = (
                clean[attacked_indices] + perturbations
            ).clamp(0.0, 1.0)
        linf = (adversarial - clean).abs().flatten(1).amax(dim=1).numpy()
        scores, maps = adapter.predict_with_categories(
            adversarial, [sample.category for sample in batch]
        )
        if len(scores) != len(batch) or len(maps) != len(batch):
            raise RuntimeError("Model adapter returned a different batch length")
        for sample, score, anomaly_map, distance in zip(batch, scores, maps, linf):
            predictions[sample.protocol_id] = (
                float(score), np.asarray(anomaly_map, dtype=np.float32)
            )
            actual_linf[sample.protocol_id] = float(distance)
    return predictions, actual_linf


def _postprocess_prediction_scores(
    adapter: Any,
    samples: list[EvaluationSample],
    predictions: dict[str, Prediction],
    *,
    reference_samples: list[EvaluationSample] | None = None,
    reference_predictions: dict[str, Prediction] | None = None,
) -> dict[str, Prediction]:
    scores = np.asarray(
        [predictions[sample.protocol_id][0] for sample in samples], dtype=np.float32
    )
    maps = [predictions[sample.protocol_id][1] for sample in samples]
    map_mins = np.asarray([float(anomaly_map.min()) for anomaly_map in maps])
    map_maxs = np.asarray([float(anomaly_map.max()) for anomaly_map in maps])
    categories = [sample.category for sample in samples]
    if (reference_samples is None) != (reference_predictions is None):
        raise ValueError(
            "reference_samples and reference_predictions must be supplied together"
        )
    if reference_samples is None or reference_predictions is None:
        processed = adapter.postprocess_image_scores(
            scores, map_mins, map_maxs, categories
        )
    else:
        reference_maps = [
            reference_predictions[sample.protocol_id][1]
            for sample in reference_samples
        ]
        processed = adapter.postprocess_image_scores_with_reference(
            scores,
            map_mins,
            map_maxs,
            categories,
            reference_scores=np.asarray(
                [
                    reference_predictions[sample.protocol_id][0]
                    for sample in reference_samples
                ],
                dtype=np.float32,
            ),
            reference_map_mins=np.asarray(
                [float(anomaly_map.min()) for anomaly_map in reference_maps]
            ),
            reference_map_maxs=np.asarray(
                [float(anomaly_map.max()) for anomaly_map in reference_maps]
            ),
            reference_categories=[sample.category for sample in reference_samples],
        )
    if processed.shape != scores.shape or not np.isfinite(processed).all():
        raise ValueError("Postprocessed image scores must be finite and one per sample")
    return {
        sample.protocol_id: (float(score), anomaly_map)
        for sample, score, anomaly_map in zip(samples, processed, maps)
    }


def _validate_ids(
    artifact: AttackArtifact,
    sample_index: dict[str, EvaluationSample],
) -> list[EvaluationSample]:
    missing = [sample_id for sample_id in artifact.evaluation_ids if sample_id not in sample_index]
    if missing:
        preview = ", ".join(missing[:5])
        raise ValueError(
            f"{len(missing)} fixed evaluation IDs for {artifact.name} were not found "
            f"in the mounted {artifact.record['target_dataset']} dataset; first: {preview}"
        )
    evaluation = [sample_index[sample_id] for sample_id in artifact.evaluation_ids]
    source_label = int(artifact.record["source_label"])
    expected_attacked = {
        sample.protocol_id for sample in evaluation if sample.label == source_label
    }
    actual_attacked = set(artifact.attacked_ids)
    if actual_attacked != expected_attacked:
        raise ValueError(
            f"Attacked IDs do not exactly match source-label evaluation images for "
            f"{artifact.name}: expected {len(expected_attacked)}, got {len(actual_attacked)}"
        )
    expected_count = artifact.record.get("target_evaluation_all_count")
    if expected_count is not None and int(expected_count) != len(evaluation):
        raise ValueError(f"Evaluation count mismatch in {artifact.name}")
    attacked_count = artifact.record.get("target_attacked_label_count")
    if attacked_count is not None and int(attacked_count) != len(actual_attacked):
        raise ValueError(f"Attacked count mismatch in {artifact.name}")
    return evaluation


def _metric_row(
    artifact: AttackArtifact,
    category: str,
    samples: list[EvaluationSample],
    clean_predictions: dict[str, Prediction],
    adversarial_predictions: dict[str, Prediction],
    actual_linf: dict[str, float],
    config: EvaluationConfig,
    category_thresholds: dict[str, float] | None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    labels = np.asarray([sample.label for sample in samples], dtype=np.uint8)
    clean_scores = np.asarray(
        [clean_predictions[sample.protocol_id][0] for sample in samples],
        dtype=np.float32,
    )
    adversarial_scores = np.asarray(
        [adversarial_predictions[sample.protocol_id][0] for sample in samples],
        dtype=np.float32,
    )
    clean_maps = resize_anomaly_maps(
        [clean_predictions[sample.protocol_id][1] for sample in samples],
        config.metric_size,
        config.anomaly_map_sigma,
    )
    adversarial_maps = resize_anomaly_maps(
        [adversarial_predictions[sample.protocol_id][1] for sample in samples],
        config.metric_size,
        config.anomaly_map_sigma,
    )
    masks = np.stack([load_mask(sample, config.metric_size) for sample in samples])
    clean = {
        **image_metrics(labels, clean_scores),
        **pixel_metrics(
            masks,
            clean_maps,
            fpr_limit=config.aupro_fpr_limit,
            max_thresholds=config.aupro_max_thresholds,
        ),
    }
    adversarial = {
        **image_metrics(labels, adversarial_scores),
        **pixel_metrics(
            masks,
            adversarial_maps,
            fpr_limit=config.aupro_fpr_limit,
            max_thresholds=config.aupro_max_thresholds,
        ),
    }
    clean_pixel_f1_threshold = float(clean["p_f1_threshold"])
    if config.pixel_threshold_mode == "fixed_0_5":
        pixel_threshold = 0.5
    elif config.pixel_threshold_mode == "image_f1":
        if category_thresholds is None or category not in category_thresholds:
            raise ValueError(
                "The image_f1 pixel threshold mode requires frozen image thresholds"
            )
        pixel_threshold = float(category_thresholds[category])
    else:
        pixel_threshold = clean_pixel_f1_threshold
    if not math.isfinite(pixel_threshold):
        raise ValueError(
            f"A clean pixel F1-max threshold could not be calibrated for {category!r}"
        )
    attacked_ids = set(artifact.attacked_ids)
    attacked_mask = np.asarray(
        [sample.protocol_id in attacked_ids for sample in samples], dtype=bool
    )
    target_label = int(artifact.record["target_label"])
    direction_sign = 1.0 if target_label == 1 else -1.0
    threshold: float | None = None
    clean_binary: np.ndarray | None = None
    adversarial_binary: np.ndarray | None = None
    threshold_metrics: dict[str, float | int] = {}
    if category_thresholds is not None:
        if category not in category_thresholds:
            raise ValueError(f"No frozen threshold found for category {category!r}")
        threshold = category_thresholds[category]
        clean_binary = (clean_scores >= threshold).astype(np.uint8)
        adversarial_binary = (adversarial_scores >= threshold).astype(np.uint8)
        clean_classification = binary_classification_metrics(labels, clean_binary)
        adversarial_classification = binary_classification_metrics(
            labels, adversarial_binary
        )
        threshold_metrics = {
            "clean_accuracy": clean_classification["accuracy"],
            "adversarial_accuracy": adversarial_classification["accuracy"],
            "clean_fpr": clean_classification["fpr"],
            "adversarial_fpr": adversarial_classification["fpr"],
            "clean_fnr": clean_classification["fnr"],
            "adversarial_fnr": adversarial_classification["fnr"],
            **targeted_attack_metrics(
                clean_binary,
                adversarial_binary,
                attacked_mask,
                source_label=int(artifact.record["source_label"]),
                target_label=int(artifact.record["target_label"]),
            ),
        }
    per_image: list[dict[str, Any]] = []
    for index, sample in enumerate(samples):
        attacked = sample.protocol_id in attacked_ids
        map_delta = direction_sign * (adversarial_maps[index] - clean_maps[index])
        target_region = _target_region(artifact, masks[index])
        region_delta = map_delta[target_region]
        score_shift = float(adversarial_scores[index] - clean_scores[index])
        detail: dict[str, Any] = {
            "model": config.model_name,
            "condition": artifact.name,
            "source_dataset": artifact.record["source_dataset"],
            "target_dataset": artifact.record["target_dataset"],
            "direction": artifact.record["direction"],
            "source_label": int(artifact.record["source_label"]),
            "target_label": target_label,
            "loss_mode": artifact.record["loss_mode"],
            "scope": artifact.record["scope"],
            "sample_id": sample.protocol_id,
            "category": sample.category,
            "label": sample.label,
            "attacked": int(attacked),
            "clean_score": float(clean_scores[index]),
            "adversarial_score": float(adversarial_scores[index]),
            "score_shift": score_shift,
            "directional_score_shift": direction_sign * score_shift,
            "map_directional_mean_shift": float(map_delta.mean()),
            "map_directional_pixel_fraction": float((map_delta > 0).mean()),
            "target_region_directional_mean_shift": float(region_delta.mean()),
            "target_region_directional_pixel_fraction": float(
                (region_delta > 0).mean()
            ),
            "pixel_decision_threshold": pixel_threshold,
            "clean_pixel_f1_threshold": clean_pixel_f1_threshold,
            "pixel_threshold_mode": config.pixel_threshold_mode,
            "normal_local_target": str(
                artifact.record.get("normal_local_target") or "full_image"
            ),
            "normal_target_region_fraction": float(
                artifact.record.get("normal_target_region_fraction", 1.0)
            ),
            "normal_target_center_x": float(
                artifact.record.get("normal_target_center_x", 0.5)
            ),
            "normal_target_center_y": float(
                artifact.record.get("normal_target_center_y", 0.5)
            ),
            "actual_linf": actual_linf.get(sample.protocol_id, 0.0),
        }
        if attacked:
            detail.update(
                targeted_region_pixel_metrics(
                    clean_maps[index],
                    adversarial_maps[index],
                    target_region,
                    threshold=pixel_threshold,
                    source_label=int(artifact.record["source_label"]),
                    target_label=target_label,
                    minimum_flip_fraction=config.pixel_success_min_flip_fraction,
                )
            )
        else:
            detail.update(
                {
                    "target_region_pixel_count": int(target_region.sum()),
                    "target_region_pixel_eligible_count": 0,
                    "target_region_pixel_flip_count": 0,
                    "target_region_pixel_flip_rate": float("nan"),
                    "target_region_pixel_success_eligible": 0,
                    "target_region_pixel_attack_success": 0,
                }
            )
        if (
            threshold is not None
            and clean_binary is not None
            and adversarial_binary is not None
        ):
            clean_prediction = int(clean_binary[index])
            adversarial_prediction = int(adversarial_binary[index])
            eligible = attacked and clean_prediction == int(
                artifact.record["source_label"]
            )
            detail.update(
                {
                    "threshold": threshold,
                    "clean_binary_prediction": clean_prediction,
                    "adversarial_binary_prediction": adversarial_prediction,
                    "attack_flipped": int(
                        attacked and clean_prediction != adversarial_prediction
                    ),
                    "targeted_success_eligible": int(eligible),
                    "targeted_attack_success": int(
                        eligible
                        and adversarial_prediction == target_label
                        and adversarial_prediction != clean_prediction
                    ),
                    "clean_target_margin": direction_sign
                    * (float(clean_scores[index]) - threshold),
                    "adversarial_target_margin": direction_sign
                    * (float(adversarial_scores[index]) - threshold),
                }
            )
        per_image.append(detail)
    attacked_rows = [row for row in per_image if row["attacked"]]
    pixel_success_eligible_rows = [
        row for row in attacked_rows if row["target_region_pixel_success_eligible"]
    ]
    pixel_eligible_count = sum(
        int(row["target_region_pixel_eligible_count"]) for row in attacked_rows
    )
    pixel_flip_count = sum(
        int(row["target_region_pixel_flip_count"]) for row in attacked_rows
    )
    pixel_success_count = sum(
        int(row["target_region_pixel_attack_success"])
        for row in pixel_success_eligible_rows
    )

    def attacked_mean(field: str) -> float:
        return (
            float(np.mean([float(row[field]) for row in attacked_rows]))
            if attacked_rows
            else float("nan")
        )

    row: dict[str, Any] = {
        "model": config.model_name,
        "condition": artifact.name,
        "source_dataset": artifact.record["source_dataset"],
        "target_dataset": artifact.record["target_dataset"],
        "direction": artifact.record["direction"],
        "loss_mode": artifact.record["loss_mode"],
        "scope": artifact.record["scope"],
        "category": category,
        "sample_count": len(samples),
        "attacked_count": len(attacked_rows),
        "mean_directional_score_shift": attacked_mean("directional_score_shift"),
        "mean_directional_map_shift": attacked_mean("map_directional_mean_shift"),
        "mean_directional_map_pixel_fraction": attacked_mean(
            "map_directional_pixel_fraction"
        ),
        "mean_target_region_directional_map_shift": attacked_mean(
            "target_region_directional_mean_shift"
        ),
        "mean_target_region_directional_pixel_fraction": attacked_mean(
            "target_region_directional_pixel_fraction"
        ),
        "pixel_decision_threshold": pixel_threshold,
        "clean_pixel_f1_threshold": clean_pixel_f1_threshold,
        "pixel_threshold_mode": config.pixel_threshold_mode,
        "pixel_success_min_flip_fraction": config.pixel_success_min_flip_fraction,
        "target_region_pixel_eligible_count": pixel_eligible_count,
        "target_region_pixel_flip_count": pixel_flip_count,
        "target_region_pixel_flip_rate": (
            100.0 * pixel_flip_count / pixel_eligible_count
            if pixel_eligible_count
            else float("nan")
        ),
        "target_region_pixel_success_eligible_count": len(
            pixel_success_eligible_rows
        ),
        "target_region_pixel_success_count": pixel_success_count,
        "target_region_pixel_attack_success_rate": (
            100.0 * pixel_success_count / len(pixel_success_eligible_rows)
            if pixel_success_eligible_rows
            else float("nan")
        ),
        "mean_actual_linf": attacked_mean("actual_linf"),
        **threshold_metrics,
    }
    if threshold is not None:
        row["threshold"] = threshold
    for metric in (
        "i_auroc",
        "i_ap",
        "i_f1_max",
        "p_auroc",
        "p_f1_max",
        "aupro",
    ):
        row[f"clean_{metric}"] = clean[metric]
        row[f"adversarial_{metric}"] = adversarial[metric]
        row[f"delta_{metric}"] = clean[metric] - adversarial[metric]
    return row, per_image


def _finite_mean(values: Iterable[Any]) -> float:
    finite = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return float(np.mean(finite)) if finite else float("nan")


def _macro_row(artifact: AttackArtifact, rows: list[dict[str, Any]], model: str) -> dict[str, Any]:
    base = {
        "model": model,
        "condition": artifact.name,
        "source_dataset": artifact.record["source_dataset"],
        "target_dataset": artifact.record["target_dataset"],
        "direction": artifact.record["direction"],
        "loss_mode": artifact.record["loss_mode"],
        "scope": artifact.record["scope"],
        "pixel_threshold_mode": rows[0]["pixel_threshold_mode"] if rows else "",
        "category": "macro",
        "sample_count": sum(int(row["sample_count"]) for row in rows),
        "attacked_count": sum(int(row["attacked_count"]) for row in rows),
    }
    mean_fields = [
        "mean_directional_score_shift",
        "mean_directional_map_shift",
        "mean_directional_map_pixel_fraction",
        "mean_target_region_directional_map_shift",
        "mean_target_region_directional_pixel_fraction",
        "pixel_decision_threshold",
        "clean_pixel_f1_threshold",
        "target_region_pixel_flip_rate",
        "target_region_pixel_attack_success_rate",
        "mean_actual_linf",
    ] + [
        f"{prefix}_{metric}"
        for metric in (
            "i_auroc",
            "i_ap",
            "i_f1_max",
            "p_auroc",
            "p_f1_max",
            "aupro",
        )
        for prefix in ("clean", "adversarial", "delta")
    ]
    threshold_mean_fields = [
        "clean_accuracy",
        "adversarial_accuracy",
        "clean_fpr",
        "adversarial_fpr",
        "clean_fnr",
        "adversarial_fnr",
        "attack_flip_rate",
        "targeted_attack_success_rate",
    ]
    if rows and all(
        all(field in row for field in threshold_mean_fields) for row in rows
    ):
        mean_fields.extend(threshold_mean_fields)
        base["targeted_success_eligible_count"] = sum(
            int(row["targeted_success_eligible_count"]) for row in rows
        )
    for field_name in mean_fields:
        base[field_name] = _finite_mean(row[field_name] for row in rows)
    if rows:
        base["pixel_success_min_flip_fraction"] = rows[0][
            "pixel_success_min_flip_fraction"
        ]
        for count_field in (
            "target_region_pixel_eligible_count",
            "target_region_pixel_flip_count",
            "target_region_pixel_success_eligible_count",
            "target_region_pixel_success_count",
        ):
            base[count_field] = sum(int(row[count_field]) for row in rows)
    return base


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _save_predictions(
    path: Path,
    samples: list[EvaluationSample],
    clean: dict[str, Prediction],
    adversarial: dict[str, Prediction],
    attacked_ids: set[str],
    category_thresholds: dict[str, float] | None,
    prediction_map_size: int | None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    def maps_for_storage(predictions: dict[str, Prediction]) -> np.ndarray:
        maps = [predictions[sample.protocol_id][1] for sample in samples]
        if prediction_map_size is None or maps[0].shape[-2:] == (
            prediction_map_size,
            prediction_map_size,
        ):
            return np.stack(maps).astype(np.float32, copy=False)
        resized: list[np.ndarray] = []
        # Avoid materializing another full 518x518 stack for AA-CLIP.
        for start in range(0, len(maps), 32):
            tensor = torch.from_numpy(
                np.stack(maps[start : start + 32]).astype(np.float32, copy=False)
            ).unsqueeze(1)
            resized.append(
                F.interpolate(
                    tensor,
                    size=(prediction_map_size, prediction_map_size),
                    mode="bilinear",
                    align_corners=False,
                    antialias=True,
                )
                .squeeze(1)
                .numpy()
            )
        return np.concatenate(resized)

    clean_maps = maps_for_storage(clean)
    adversarial_maps = maps_for_storage(adversarial)
    payload: dict[str, np.ndarray] = {
        "sample_ids": np.asarray([sample.protocol_id for sample in samples]),
        "labels": np.asarray([sample.label for sample in samples], dtype=np.uint8),
        "attacked": np.asarray(
            [sample.protocol_id in attacked_ids for sample in samples], dtype=bool
        ),
        "clean_scores": np.asarray(
            [clean[sample.protocol_id][0] for sample in samples], dtype=np.float32
        ),
        "adversarial_scores": np.asarray(
            [adversarial[sample.protocol_id][0] for sample in samples],
            dtype=np.float32,
        ),
        "clean_lowres_maps": clean_maps,
        "adversarial_lowres_maps": adversarial_maps,
    }
    if category_thresholds is not None:
        thresholds = np.asarray(
            [category_thresholds[sample.category] for sample in samples],
            dtype=np.float32,
        )
        payload["thresholds"] = thresholds
        payload["clean_binary_predictions"] = (
            payload["clean_scores"] >= thresholds
        ).astype(np.uint8)
        payload["adversarial_binary_predictions"] = (
            payload["adversarial_scores"] >= thresholds
        ).astype(np.uint8)
    np.savez_compressed(path, **payload)


def _safe_config_dict(config: EvaluationConfig) -> dict[str, Any]:
    value = asdict(config)
    # Tuple/list differences are immaterial in the persisted run configuration.
    return json.loads(json.dumps(value, default=str))


def run_evaluation(config: EvaluationConfig) -> Path:
    """Evaluate every selected manifest condition and return summary.csv."""

    if config.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable; enable a Kaggle GPU")
    artifacts = load_manifest(
        config.artifacts_root,
        scopes=config.attack_scopes,
        sources=config.source_datasets,
        targets=config.target_datasets,
        categories=config.attack_categories,
        directions=config.attack_directions,
        loss_modes=config.attack_loss_modes,
    )
    if config.condition_names is not None:
        selected = set(config.condition_names)
        artifacts = [artifact for artifact in artifacts if artifact.name in selected]
        missing = selected - {artifact.name for artifact in artifacts}
        if missing:
            raise ValueError(f"Requested conditions not found in manifest: {sorted(missing)}")
    if config.max_conditions is not None:
        artifacts = artifacts[: config.max_conditions]
    modes = config.pixel_threshold_modes or (config.pixel_threshold_mode,)
    mode_configs: dict[str, EvaluationConfig] = {}
    outputs: dict[str, Path] = {}
    category_rows: dict[str, list[dict[str, Any]]] = {}
    summary_rows: dict[str, list[dict[str, Any]]] = {}
    per_image_rows: dict[str, list[dict[str, Any]]] = {}
    manifest_snapshot = json.dumps(
        [json_safe_record(a.record) for a in artifacts], indent=2
    )
    for mode in modes:
        output_root = (
            config.output_roots_by_pixel_threshold_mode[mode]
            if config.pixel_threshold_modes is not None
            else config.output_root
        )
        qualitative_root = (
            config.qualitative_output_roots_by_pixel_threshold_mode[mode]
            if config.pixel_threshold_modes is not None
            and config.save_qualitative_samples
            else config.qualitative_output_root
        )
        mode_config = replace(
            config,
            output_root=output_root,
            qualitative_output_root=qualitative_root,
            pixel_threshold_mode=mode,
            pixel_threshold_modes=None,
            output_roots_by_pixel_threshold_mode={},
            qualitative_output_roots_by_pixel_threshold_mode={},
        )
        mode_configs[mode] = mode_config
        output = Path(output_root).expanduser().resolve()
        outputs[mode] = output
        output.mkdir(parents=True, exist_ok=True)
        (output / "run_config.json").write_text(
            json.dumps(_safe_config_dict(mode_config), indent=2), encoding="utf-8"
        )
        (output / "manifest_snapshot.json").write_text(
            manifest_snapshot, encoding="utf-8"
        )
        category_rows[mode] = []
        summary_rows[mode] = []
        per_image_rows[mode] = []

    groups: dict[tuple[str, int], list[AttackArtifact]] = {}
    for artifact in artifacts:
        groups.setdefault(
            (str(artifact.record["target_dataset"]), int(artifact.record["image_size"])),
            [],
        ).append(artifact)

    for (target_dataset, image_size), target_artifacts in groups.items():
        print(f"[data] Discovering {target_dataset}")
        category_thresholds: dict[str, float] | None = None
        if config.thresholds_by_target:
            threshold_path = config.thresholds_by_target.get(target_dataset)
            if threshold_path is None:
                raise ValueError(
                    f"No threshold artifact configured for target {target_dataset!r}"
                )
            category_thresholds = load_category_thresholds(
                threshold_path,
                expected_dataset=target_dataset,
                expected_model=config.model_name,
            )
        sample_index = index_samples(
            discover_dataset(
                target_dataset,
                mvtec_root=config.mvtec_root,
                visa_root=config.visa_root,
            )
        )
        evaluations = {
            artifact.name: _validate_ids(artifact, sample_index)
            for artifact in target_artifacts
        }
        clean_ids: list[str] = []
        seen_ids: set[str] = set()
        for artifact in target_artifacts:
            for sample_id in artifact.evaluation_ids:
                if sample_id not in seen_ids:
                    clean_ids.append(sample_id)
                    seen_ids.add(sample_id)
        clean_samples = [sample_index[sample_id] for sample_id in clean_ids]

        kwargs = dict(config.model_kwargs_by_target.get(target_dataset, {}))
        if not kwargs:
            raise ValueError(
                f"No model configuration supplied for target dataset {target_dataset!r}"
            )
        kwargs.setdefault("device", config.device)
        kwargs.setdefault("image_size", image_size)
        print(f"[model] Loading {config.model_name} for target={target_dataset}")
        adapter = build_adapter(config.model_name, **kwargs)
        try:
            raw_clean_predictions = _predict_clean(
                adapter,
                clean_samples,
                image_size=image_size,
                batch_size=config.batch_size,
                description=f"clean {target_dataset}",
            )
            clean_predictions = _postprocess_prediction_scores(
                adapter, clean_samples, raw_clean_predictions
            )
            for artifact in target_artifacts:
                print(f"[condition] {artifact.name}")
                delta = artifact.load_delta(verify_checksum=config.verify_checksums)
                evaluation = evaluations[artifact.name]
                attacked_set = set(artifact.attacked_ids)
                delta_indices = artifact.delta_indices()
                adversarial_predictions, actual_linf = _predict_adversarial(
                    adapter,
                    evaluation,
                    delta,
                    attacked_set,
                    delta_indices,
                    image_size=image_size,
                    batch_size=config.batch_size,
                    description=artifact.name,
                )
                # The opposite-label cohort receives no perturbation. Reuse its
                # cached clean output exactly instead of allowing batch-shape or
                # cohort-normalization effects to create false control changes.
                for sample in evaluation:
                    if sample.protocol_id not in attacked_set:
                        adversarial_predictions[sample.protocol_id] = (
                            raw_clean_predictions[sample.protocol_id]
                        )
                adversarial_predictions = _postprocess_prediction_scores(
                    adapter,
                    evaluation,
                    adversarial_predictions,
                    reference_samples=evaluation,
                    reference_predictions=raw_clean_predictions,
                )

                grouped: dict[str, list[EvaluationSample]] = {}
                for sample in evaluation:
                    grouped.setdefault(sample.category, []).append(sample)
                for mode in modes:
                    mode_config = mode_configs[mode]
                    condition_category_rows: list[dict[str, Any]] = []
                    condition_per_image: list[dict[str, Any]] = []
                    for category, category_samples in sorted(grouped.items()):
                        row, details = _metric_row(
                            artifact,
                            category,
                            category_samples,
                            clean_predictions,
                            adversarial_predictions,
                            actual_linf,
                            mode_config,
                            category_thresholds,
                        )
                        condition_category_rows.append(row)
                        condition_per_image.extend(details)
                    macro = _macro_row(
                        artifact, condition_category_rows, config.model_name
                    )
                    category_rows[mode].extend(condition_category_rows)
                    summary_rows[mode].append(macro)
                    per_image_rows[mode].extend(condition_per_image)

                    if mode_config.save_qualitative_samples:
                        export_representative_samples(
                            mode_config.qualitative_output_root,
                            artifact.name,
                            condition_per_image,
                            {sample.protocol_id: sample for sample in evaluation},
                            clean_predictions,
                            adversarial_predictions,
                            delta,
                            delta_indices,
                            image_size=image_size,
                            anomaly_map_sigma=config.anomaly_map_sigma,
                            selection_basis=config.qualitative_selection_basis,
                        )

                if config.save_predictions:
                    _save_predictions(
                        outputs[modes[0]] / "predictions" / f"{artifact.name}.npz",
                        evaluation,
                        clean_predictions,
                        adversarial_predictions,
                        attacked_set,
                        category_thresholds,
                        config.prediction_map_size,
                    )
                # Persist after every condition so long Kaggle runs retain progress.
                for mode in modes:
                    output = outputs[mode]
                    _write_csv(output / "category_metrics.csv", category_rows[mode])
                    _write_csv(output / "summary.csv", summary_rows[mode])
                    _write_csv(output / "per_image.csv", per_image_rows[mode])
        finally:
            adapter.release()

    for mode in modes:
        print(f"[done] {mode} summary: {outputs[mode] / 'summary.csv'}")
    return outputs[modes[0]] / "summary.csv"
