"""Shared, model-agnostic evaluation runner for fixed universal attacks."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import csv
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
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
from .metrics import image_metrics, pixel_metrics, resize_anomaly_maps


@dataclass
class EvaluationConfig:
    artifacts_root: str
    output_root: str
    model_name: str
    model_kwargs_by_target: dict[str, dict[str, Any]]
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
    source_datasets: tuple[str, ...] | None = None
    target_datasets: tuple[str, ...] | None = None
    condition_names: tuple[str, ...] | None = None
    max_conditions: int | None = None
    run_notes: str = ""

    def __post_init__(self) -> None:
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")
        if self.metric_size < 1:
            raise ValueError("metric_size must be positive")
        if not 0.0 < self.aupro_fpr_limit <= 1.0:
            raise ValueError("aupro_fpr_limit must be in (0, 1]")
        if self.max_conditions is not None and self.max_conditions < 1:
            raise ValueError("max_conditions must be positive when supplied")


Prediction = tuple[float, np.ndarray]


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
        scores, maps = adapter.predict(images)
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
    *,
    image_size: int,
    batch_size: int,
    description: str,
) -> tuple[dict[str, Prediction], dict[str, float]]:
    predictions: dict[str, Prediction] = {}
    actual_linf: dict[str, float] = {}
    perturbation = delta[0]
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
            adversarial[attacked_indices] = (
                clean[attacked_indices] + perturbation
            ).clamp(0.0, 1.0)
        linf = (adversarial - clean).abs().flatten(1).amax(dim=1).numpy()
        scores, maps = adapter.predict(adversarial)
        if len(scores) != len(batch) or len(maps) != len(batch):
            raise RuntimeError("Model adapter returned a different batch length")
        for sample, score, anomaly_map, distance in zip(batch, scores, maps, linf):
            predictions[sample.protocol_id] = (
                float(score), np.asarray(anomaly_map, dtype=np.float32)
            )
            actual_linf[sample.protocol_id] = float(distance)
    return predictions, actual_linf


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
    attacked_ids = set(artifact.attacked_ids)
    target_label = int(artifact.record["target_label"])
    direction_sign = 1.0 if target_label == 1 else -1.0
    per_image: list[dict[str, Any]] = []
    for index, sample in enumerate(samples):
        attacked = sample.protocol_id in attacked_ids
        map_delta = direction_sign * (adversarial_maps[index] - clean_maps[index])
        score_shift = float(adversarial_scores[index] - clean_scores[index])
        per_image.append(
            {
                "model": config.model_name,
                "condition": artifact.name,
                "source_dataset": artifact.record["source_dataset"],
                "target_dataset": artifact.record["target_dataset"],
                "direction": artifact.record["direction"],
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
                "actual_linf": actual_linf.get(sample.protocol_id, 0.0),
            }
        )
    attacked_rows = [row for row in per_image if row["attacked"]]

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
        "mean_actual_linf": attacked_mean("actual_linf"),
    }
    for metric in ("i_auroc", "i_ap", "p_auroc", "aupro"):
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
        "category": "macro",
        "sample_count": sum(int(row["sample_count"]) for row in rows),
        "attacked_count": sum(int(row["attacked_count"]) for row in rows),
    }
    mean_fields = [
        "mean_directional_score_shift",
        "mean_directional_map_shift",
        "mean_directional_map_pixel_fraction",
        "mean_actual_linf",
    ] + [
        f"{prefix}_{metric}"
        for metric in ("i_auroc", "i_ap", "p_auroc", "aupro")
        for prefix in ("clean", "adversarial", "delta")
    ]
    for field_name in mean_fields:
        base[field_name] = _finite_mean(row[field_name] for row in rows)
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
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        sample_ids=np.asarray([sample.protocol_id for sample in samples]),
        labels=np.asarray([sample.label for sample in samples], dtype=np.uint8),
        attacked=np.asarray(
            [sample.protocol_id in attacked_ids for sample in samples], dtype=bool
        ),
        clean_scores=np.asarray(
            [clean[sample.protocol_id][0] for sample in samples], dtype=np.float32
        ),
        adversarial_scores=np.asarray(
            [adversarial[sample.protocol_id][0] for sample in samples],
            dtype=np.float32,
        ),
        clean_lowres_maps=np.stack([clean[sample.protocol_id][1] for sample in samples]),
        adversarial_lowres_maps=np.stack(
            [adversarial[sample.protocol_id][1] for sample in samples]
        ),
    )


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
        sources=config.source_datasets,
        targets=config.target_datasets,
    )
    if config.condition_names is not None:
        selected = set(config.condition_names)
        artifacts = [artifact for artifact in artifacts if artifact.name in selected]
        missing = selected - {artifact.name for artifact in artifacts}
        if missing:
            raise ValueError(f"Requested conditions not found in manifest: {sorted(missing)}")
    if config.max_conditions is not None:
        artifacts = artifacts[: config.max_conditions]
    unsupported = [artifact.name for artifact in artifacts if artifact.record["scope"] != "dataset"]
    if unsupported:
        raise NotImplementedError(
            "This runner currently accepts fixed dataset-universal tensors only; "
            f"unsupported manifest records: {unsupported}"
        )

    output = Path(config.output_root).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "run_config.json").write_text(
        json.dumps(_safe_config_dict(config), indent=2), encoding="utf-8"
    )
    (output / "manifest_snapshot.json").write_text(
        json.dumps([json_safe_record(a.record) for a in artifacts], indent=2),
        encoding="utf-8",
    )

    category_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    per_image_rows: list[dict[str, Any]] = []

    groups: dict[tuple[str, int], list[AttackArtifact]] = {}
    for artifact in artifacts:
        groups.setdefault(
            (str(artifact.record["target_dataset"]), int(artifact.record["image_size"])),
            [],
        ).append(artifact)

    for (target_dataset, image_size), target_artifacts in groups.items():
        print(f"[data] Discovering {target_dataset}")
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
            clean_predictions = _predict_clean(
                adapter,
                clean_samples,
                image_size=image_size,
                batch_size=config.batch_size,
                description=f"clean {target_dataset}",
            )
            for artifact in target_artifacts:
                print(f"[condition] {artifact.name}")
                delta = artifact.load_delta(verify_checksum=config.verify_checksums)
                evaluation = evaluations[artifact.name]
                attacked_set = set(artifact.attacked_ids)
                adversarial_predictions, actual_linf = _predict_adversarial(
                    adapter,
                    evaluation,
                    delta,
                    attacked_set,
                    image_size=image_size,
                    batch_size=config.batch_size,
                    description=artifact.name,
                )

                grouped: dict[str, list[EvaluationSample]] = {}
                for sample in evaluation:
                    grouped.setdefault(sample.category, []).append(sample)
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
                        config,
                    )
                    condition_category_rows.append(row)
                    condition_per_image.extend(details)
                macro = _macro_row(
                    artifact, condition_category_rows, config.model_name
                )
                category_rows.extend(condition_category_rows)
                summary_rows.append(macro)
                per_image_rows.extend(condition_per_image)

                if config.save_predictions:
                    _save_predictions(
                        output / "predictions" / f"{artifact.name}.npz",
                        evaluation,
                        clean_predictions,
                        adversarial_predictions,
                        attacked_set,
                    )
                # Persist after every condition so long Kaggle runs retain progress.
                _write_csv(output / "category_metrics.csv", category_rows)
                _write_csv(output / "summary.csv", summary_rows)
                _write_csv(output / "per_image.csv", per_image_rows)
        finally:
            adapter.release()

    summary_path = output / "summary.csv"
    print(f"[done] Summary: {summary_path}")
    return summary_path
