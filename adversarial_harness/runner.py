"""End-to-end orchestration for the adversarial robustness benchmark."""

from __future__ import annotations

import csv
from dataclasses import replace
import gc
import json
from pathlib import Path
import random
import shutil
import time
from typing import Dict, Iterable, List, Mapping, Sequence
import zlib

import numpy as np
import torch
from PIL import Image, ImageOps
from tqdm.auto import tqdm

from .attacks import TargetedPGD, direction_labels
from .config import ExperimentConfig
from .dataset import (
    MVTecSample,
    discover_anomaly_datasets,
    group_by_category,
    load_image_tensor,
    load_mask,
)
from .metrics import (
    LPIPSMetric,
    image_metrics,
    perceptual_metrics,
    pixel_metrics,
    resize_anomaly_maps,
)
from .models import CLIPSurrogate, build_target
from .split_manifest import (
    MATCHED_SPLIT_PROTOCOL,
    LoadedSplitManifest,
    load_matched_split_manifest,
)


SUMMARY_FIELDS = (
    "condition",
    "target_model",
    "scope",
    "universal_protocol",
    "direction",
    "loss_mode",
    "dataset",
    "category",
    "decision_threshold",
    "source_count",
    "eligible_clean_correct_count",
    "classification_flip_rate",
    "targeted_success_rate_all",
    "clean_i_auroc",
    "adversarial_i_auroc",
    "delta_i_auroc",
    "clean_i_ap",
    "adversarial_i_ap",
    "delta_i_ap",
    "clean_p_auroc",
    "adversarial_p_auroc",
    "delta_p_auroc",
    "clean_aupro",
    "adversarial_aupro",
    "delta_aupro",
    "mean_linf",
    "max_linf",
    "epsilon",
    "mean_ssim",
    "mean_lpips",
)

DETAIL_FIELDS = (
    "condition",
    "target_model",
    "scope",
    "universal_protocol",
    "direction",
    "loss_mode",
    "sample_id",
    "dataset",
    "category",
    "defect_type",
    "image_path",
    "mask_path",
    "label",
    "source_label",
    "target_label",
    "decision_threshold",
    "clean_score",
    "adversarial_score",
    "clean_prediction",
    "adversarial_prediction",
    "clean_correct_for_source",
    "targeted_success",
    "score_shift",
    "directional_score_shift",
    "directional_success",
    "linf",
    "ssim",
    "lpips",
)

LOSS_CURVE_FIELDS = (
    "group",
    "step",
    "pre_update_total_loss",
    "total_loss",
    "global_loss",
    "local_loss",
    "global_gradient_l2",
    "local_gradient_l2",
    "combined_gradient_l2",
)

SURROGATE_FIELDS = (
    "condition",
    "group",
    "sample_id",
    "dataset",
    "category",
    "defect_type",
    "target_label",
    "clean_global_score",
    "adversarial_global_score",
    "clean_local_score",
    "adversarial_local_score",
    "clean_mode_score",
    "adversarial_mode_score",
    "clean_mode_prediction",
    "adversarial_mode_prediction",
    "surrogate_targeted_success",
)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _batches(items: Sequence[object], size: int) -> Iterable[Sequence[object]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def _safe_name(value: str) -> str:
    return value.replace("/", "__").replace("\\", "__").replace(" ", "_")


def _write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _condition(scope: str, direction: str, mode: str) -> str:
    return f"{scope}__{direction}__{mode}"


def _counts_by_category(samples: Sequence[MVTecSample]) -> Dict[str, int]:
    return {
        category: len(category_samples)
        for category, category_samples in sorted(group_by_category(samples).items())
    }


def _counts_by_category_and_label(
    samples: Sequence[MVTecSample],
) -> Dict[str, Dict[str, int]]:
    counts: Dict[str, Dict[str, int]] = {}
    for sample in samples:
        label_name = "normal" if sample.label == 0 else "anomaly"
        counts.setdefault(sample.category, {"normal": 0, "anomaly": 0})
        counts[sample.category][label_name] += 1
    return {category: counts[category] for category in sorted(counts)}


def _reindex_samples(samples: Sequence[MVTecSample]) -> List[MVTecSample]:
    return [replace(sample, index=index) for index, sample in enumerate(samples)]


def _samples_by_id(
    samples: Sequence[MVTecSample],
) -> Dict[str, MVTecSample]:
    """Index samples by both artifact IDs and split-qualified protocol IDs."""

    return {
        identifier: sample
        for sample in samples
        for identifier in (sample.sample_id, sample.protocol_id)
    }


def _save_tensor_image(tensor: torch.Tensor, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    array = (
        tensor.detach().cpu().clamp(0.0, 1.0).permute(1, 2, 0).numpy() * 255.0
    ).round().astype(np.uint8)
    Image.fromarray(array).save(path)


def _tensor_uint8(tensor: torch.Tensor) -> np.ndarray:
    return (
        tensor.detach().cpu().clamp(0.0, 1.0).permute(1, 2, 0).numpy() * 255.0
    ).round().astype(np.uint8)


def _normalized_pair(first: np.ndarray, second: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    minimum = float(min(np.min(first), np.min(second)))
    maximum = float(max(np.max(first), np.max(second)))
    scale = maximum - minimum
    if scale <= 1e-12:
        return np.zeros_like(first, dtype=np.float32), np.zeros_like(second, dtype=np.float32)
    return (
        ((first - minimum) / scale).astype(np.float32),
        ((second - minimum) / scale).astype(np.float32),
    )


def _heatmap_image(values_01: np.ndarray) -> Image.Image:
    gray = Image.fromarray(
        (np.clip(values_01, 0.0, 1.0) * 255.0).round().astype(np.uint8)
    )
    return ImageOps.colorize(gray, black="#0b1f6d", mid="#f7f7f7", white="#b40426")


def _signed_heatmap(values: np.ndarray) -> Image.Image:
    maximum = float(np.max(np.abs(values)))
    if maximum <= 1e-12:
        normalized = np.full_like(values, 0.5, dtype=np.float32)
    else:
        normalized = (values / maximum + 1.0) / 2.0
    return _heatmap_image(normalized)


class AdversarialExperiment:
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.output = config.output_path.resolve()
        self.output.mkdir(parents=True, exist_ok=True)
        legacy_completion = self.output / "completed_conditions.json"
        if legacy_completion.is_file():
            legacy_completion.unlink()
        legacy_partial = self.output / "partial"
        if legacy_partial.is_dir():
            shutil.rmtree(legacy_partial)
        self.summary_path = self.output / "summary.csv"
        self.detail_path = self.output / "per_image.csv"
        self.summary_rows: List[Dict[str, object]] = []
        self.detail_rows: List[Dict[str, object]] = []
        self.target = None
        self.surrogate = None
        self.category_thresholds: Dict[str, float] = {}
        self.split_assignments: Dict[str, str] = {}
        self.split_manifest_metadata: Dict[str, object] = {}
        self.split_manifest_sha256: str | None = None

    def image_loader(self, sample: MVTecSample) -> torch.Tensor:
        return load_image_tensor(sample, self.config.attack.image_size)

    def _protocol_samples(
        self,
        test_samples: Sequence[MVTecSample],
        train_good_samples: Sequence[MVTecSample],
        scope: str,
        direction: str,
        source_test_samples: Sequence[MVTecSample] = (),
        source_train_normal_samples: Sequence[MVTecSample] = (),
    ) -> tuple[List[MVTecSample], List[MVTecSample], List[MVTecSample]]:
        """Return fit sources, evaluation sources, and all evaluation samples."""

        source_label, _ = direction_labels(direction)
        test_sources = [sample for sample in test_samples if sample.label == source_label]
        if self.config.is_cross_dataset:
            cross_fit_pool = (
                source_train_normal_samples
                if source_label == 0 and not self.config.use_split_manifest
                else source_test_samples
            )
            fit_sources = [
                sample
                for sample in cross_fit_pool
                if sample.label == source_label
                and (
                    not self.config.use_split_manifest
                    or self.split_assignments.get(sample.protocol_id) == "fit"
                )
            ]
            evaluation_sources = [
                sample
                for sample in test_sources
                if (
                    not self.config.use_split_manifest
                    or self.split_assignments.get(sample.protocol_id) == "evaluation"
                )
            ]
            evaluation_samples = [
                sample
                for sample in test_samples
                if (
                    not self.config.use_split_manifest
                    or self.split_assignments.get(sample.protocol_id) == "evaluation"
                )
            ]
            if not fit_sources or not evaluation_sources or not evaluation_samples:
                raise RuntimeError(
                    "Cross-dataset protocol produced an empty source fit or target "
                    "evaluation set"
                )
            return list(fit_sources), evaluation_sources, evaluation_samples
        if scope == "per_image" or self.config.universal_protocol == "transductive":
            return list(test_sources), list(test_sources), list(test_samples)

        if self.config.use_split_manifest:
            fit_sources = [
                sample
                for sample in test_sources
                if self.split_assignments.get(sample.protocol_id) == "fit"
            ]
            evaluation_sources = [
                sample
                for sample in test_sources
                if self.split_assignments.get(sample.protocol_id) == "evaluation"
            ]
            evaluation_samples = [
                sample
                for sample in test_samples
                if self.split_assignments.get(sample.protocol_id) == "evaluation"
            ]
            if not fit_sources or not evaluation_sources or not evaluation_samples:
                raise RuntimeError(
                    "The matched split manifest produced an empty fit or evaluation set"
                )
            return fit_sources, evaluation_sources, evaluation_samples

        if direction == "normal_to_abnormal":
            fit_sources = [sample for sample in train_good_samples if sample.label == 0]
            if not fit_sources:
                raise RuntimeError(
                    "held_out normal_to_abnormal requires normal training images"
                )
            return fit_sources, list(test_sources), list(test_samples)

        # Neither benchmark has an abnormal training split. Split anomalies within every
        # category, fit on one deterministic subset, and exclude those exact
        # images from every held-out metric and prediction artifact.
        rng = np.random.default_rng(self.config.split_seed)
        fit_ids: set[str] = set()
        evaluation_ids: set[str] = set()
        anomalies_by_category = group_by_category(test_sources)
        for category in sorted(anomalies_by_category):
            category_samples = list(anomalies_by_category[category])
            if len(category_samples) < 2:
                raise RuntimeError(
                    "held_out abnormal_to_normal needs at least two test anomalies "
                    f"per category; {category!r} has {len(category_samples)}"
                )
            order = rng.permutation(len(category_samples))
            fit_count = int(np.floor(len(category_samples) * self.config.fit_fraction))
            fit_count = min(max(fit_count, 1), len(category_samples) - 1)
            for position, sample_index in enumerate(order):
                sample_id = category_samples[int(sample_index)].protocol_id
                if position < fit_count:
                    fit_ids.add(sample_id)
                else:
                    evaluation_ids.add(sample_id)

        fit_sources = [sample for sample in test_sources if sample.protocol_id in fit_ids]
        evaluation_sources = [
            sample for sample in test_sources if sample.protocol_id in evaluation_ids
        ]
        evaluation_samples = [
            sample
            for sample in test_samples
            if sample.label == 0 or sample.protocol_id in evaluation_ids
        ]
        return fit_sources, evaluation_sources, evaluation_samples

    def _diagnostic_subset(
        self, samples: Sequence[MVTecSample]
    ) -> List[MVTecSample]:
        """Choose a deterministic category-balanced diagnostic subset."""

        grouped = group_by_category(samples)
        queues = {category: list(values) for category, values in grouped.items()}
        selected: List[MVTecSample] = []
        while len(selected) < self.config.diagnostic_max_samples:
            added = False
            for category in sorted(queues):
                if queues[category]:
                    selected.append(queues[category].pop(0))
                    added = True
                    if len(selected) >= self.config.diagnostic_max_samples:
                        break
            if not added:
                break
        return selected

    def _load_target(self) -> None:
        target_arguments = {
            "anomalyclip_root": self.config.anomalyclip_root,
            "checkpoint_path": self.config.anomalyclip_checkpoint,
            "device": self.config.device,
            "image_size": self.config.attack.image_size,
        }
        target_arguments.update(self.config.target_kwargs)
        print(f"[model] Loading black-box target: {self.config.target_model}")
        self.target = build_target(self.config.target_model, **target_arguments)

    def _load_surrogate(self, categories: Sequence[str]) -> None:
        print("[model] Loading frozen public CLIP surrogate")
        self.surrogate = CLIPSurrogate(
            anomalyclip_root=self.config.anomalyclip_root,
            categories=categories,
            device=self.config.device,
            feature_layers=self.config.attack.feature_layers,
            clip_download_root=str(self.config.target_kwargs.get("clip_download_root", "")),
        )

    def _checkpoint_fingerprint(self) -> Dict[str, object]:
        checkpoint = Path(self.config.anomalyclip_checkpoint).expanduser()
        return {
            "name": checkpoint.name,
            "size_bytes": checkpoint.stat().st_size if checkpoint.is_file() else None,
        }

    def _validate_threshold_payload(
        self,
        payload: Mapping[str, object],
        categories: Sequence[str],
        path: Path,
    ) -> Dict[str, float]:
        expected = {
            "threshold_mode": self.config.threshold_mode,
            "target_model": self.config.target_model,
            "image_size": self.config.attack.image_size,
        }
        for key, value in expected.items():
            if payload.get(key) != value:
                raise ValueError(
                    f"Threshold artifact {path} has {key}={payload.get(key)!r}; "
                    f"expected {value!r}. Recalibrate for this target setup."
                )
        stored_dataset = payload.get("dataset")
        if (
            stored_dataset is not None
            and stored_dataset != self.config.evaluation_dataset
        ):
            raise ValueError(
                f"Threshold artifact {path} was calibrated for dataset "
                f"{stored_dataset!r}; expected "
                f"{self.config.evaluation_dataset!r}."
            )
        stored_quantile = float(payload.get("threshold_quantile", float("nan")))
        if not np.isclose(stored_quantile, self.config.threshold_quantile):
            raise ValueError(
                f"Threshold artifact {path} uses quantile {stored_quantile}; "
                f"requested {self.config.threshold_quantile}."
            )
        stored_fingerprint = payload.get("checkpoint")
        current_fingerprint = self._checkpoint_fingerprint()
        if isinstance(stored_fingerprint, Mapping):
            for key in ("name", "size_bytes"):
                stored_value = stored_fingerprint.get(key)
                current_value = current_fingerprint.get(key)
                if (
                    stored_value is not None
                    and current_value is not None
                    and stored_value != current_value
                ):
                    raise ValueError(
                        f"Threshold artifact {path} was calibrated with a different "
                        f"checkpoint ({key}: {stored_value!r} != {current_value!r})."
                    )
        category_payload = payload.get("categories")
        if not isinstance(category_payload, Mapping):
            raise ValueError(f"Threshold artifact {path} has no category mapping")
        missing = sorted(set(categories) - set(category_payload))
        if missing:
            raise ValueError(
                f"Threshold artifact {path} is missing categories: {missing}"
            )
        thresholds: Dict[str, float] = {}
        for category in categories:
            record = category_payload[category]
            if not isinstance(record, Mapping) or "threshold" not in record:
                raise ValueError(
                    f"Threshold artifact {path} has no threshold for {category!r}"
                )
            thresholds[category] = float(record["threshold"])
        return thresholds

    def _calibrate_category_thresholds(
        self,
        train_good_samples: Sequence[MVTecSample],
        destination: Path,
    ) -> Dict[str, float]:
        if self.target is None:
            raise RuntimeError("Target model must be loaded before threshold calibration")
        grouped = group_by_category(train_good_samples)
        scores_by_category: Dict[str, List[float]] = {
            category: [] for category in grouped
        }
        all_ids: List[str] = []
        all_datasets: List[str] = []
        all_categories: List[str] = []
        all_scores: List[float] = []
        batches = list(_batches(train_good_samples, self.config.target_batch_size))
        for batch in tqdm(
            batches,
            desc="Calibrating target thresholds on normal training images",
            unit="batch",
            dynamic_ncols=True,
        ):
            images = torch.stack([self.image_loader(sample) for sample in batch])
            batch_scores, _ = self.target.predict(images)
            for sample, score in zip(batch, batch_scores):
                value = float(score)
                scores_by_category[sample.category].append(value)
                all_ids.append(sample.protocol_id)
                all_datasets.append(sample.dataset)
                all_categories.append(sample.category)
                all_scores.append(value)

        category_records: Dict[str, Dict[str, object]] = {}
        thresholds: Dict[str, float] = {}
        for category in sorted(scores_by_category):
            values = np.asarray(scores_by_category[category], dtype=np.float64)
            if values.size == 0:
                raise RuntimeError(f"No normal-training calibration scores for {category}")
            threshold = float(np.quantile(values, self.config.threshold_quantile))
            thresholds[category] = threshold
            category_records[category] = {
                "dataset": grouped[category][0].dataset,
                "threshold": threshold,
                "sample_count": int(values.size),
                "score_min": float(np.min(values)),
                "score_max": float(np.max(values)),
                "score_mean": float(np.mean(values)),
                "score_std": float(np.std(values)),
                "calibration_sample_ids": [
                    sample.protocol_id for sample in grouped[category]
                ],
            }
        payload = {
            "schema_version": 1,
            "threshold_mode": self.config.threshold_mode,
            "threshold_quantile": self.config.threshold_quantile,
            "target_model": self.config.target_model,
            "image_size": self.config.attack.image_size,
            "checkpoint": self._checkpoint_fingerprint(),
            "dataset": self.config.evaluation_dataset,
            "calibration_split": "normal training split",
            "categories": category_records,
        }
        _write_json(destination, payload)
        np.savez_compressed(
            destination.parent / "normal_train_scores.npz",
            protocol_ids=np.asarray(all_ids),
            datasets=np.asarray(all_datasets),
            categories=np.asarray(all_categories),
            scores=np.asarray(all_scores, dtype=np.float32),
        )
        return thresholds

    def _prepare_category_thresholds(
        self,
        train_good_samples: Sequence[MVTecSample],
        categories: Sequence[str],
    ) -> Path:
        output_path = self.output / "category_thresholds.json"
        if self.config.thresholds_path:
            source_path = Path(self.config.thresholds_path).expanduser().resolve()
            if not source_path.is_file():
                raise FileNotFoundError(f"Threshold artifact not found: {source_path}")
            payload = json.loads(source_path.read_text(encoding="utf-8"))
            self.category_thresholds = self._validate_threshold_payload(
                payload, categories, source_path
            )
            copied_payload = dict(payload)
            copied_payload["loaded_from"] = str(source_path)
            _write_json(output_path, copied_payload)
            source_scores = source_path.parent / "normal_train_scores.npz"
            destination_scores = self.output / "normal_train_scores.npz"
            if source_scores.is_file() and source_scores != destination_scores:
                shutil.copy2(source_scores, destination_scores)
            print(f"[threshold] Loaded frozen category thresholds from {source_path}")
            return output_path
        self.category_thresholds = self._calibrate_category_thresholds(
            train_good_samples, output_path
        )
        print(f"[threshold] Saved category thresholds to {output_path}")
        return output_path

    def _clean_predictions(
        self, samples: Sequence[MVTecSample]
    ) -> tuple[np.ndarray, np.ndarray]:
        cache = self.output / "clean_predictions.npz"
        scores = np.zeros(len(samples), dtype=np.float32)
        maps: List[np.ndarray] = [None] * len(samples)  # type: ignore[list-item]
        batches = list(_batches(samples, self.config.target_batch_size))
        for batch in tqdm(
            batches,
            desc="Clean target inference",
            unit="batch",
            dynamic_ncols=True,
        ):
            images = torch.stack([self.image_loader(sample) for sample in batch])
            batch_scores, batch_maps = self.target.predict(images)
            for sample, score, anomaly_map in zip(batch, batch_scores, batch_maps):
                scores[sample.index] = score
                maps[sample.index] = anomaly_map
        map_array = np.stack(maps).astype(np.float32)
        np.savez_compressed(
            cache,
            sample_ids=np.asarray([sample.sample_id for sample in samples]),
            datasets=np.asarray([sample.dataset for sample in samples]),
            scores=scores,
            lowres_maps=map_array,
        )
        return scores, map_array

    def _category_metrics(
        self,
        category_samples: Sequence[MVTecSample],
        scores: np.ndarray,
        lowres_maps: np.ndarray,
    ) -> Dict[str, float]:
        indices = [sample.index for sample in category_samples]
        labels = [sample.label for sample in category_samples]
        masks = np.stack(
            [load_mask(sample, self.config.metric_size) for sample in category_samples]
        )
        maps = resize_anomaly_maps(
            lowres_maps[indices],
            self.config.metric_size,
            self.config.anomaly_map_sigma,
        )
        return {
            **image_metrics(labels, scores[indices]),
            **pixel_metrics(
                masks,
                maps,
                fpr_limit=self.config.aupro_fpr_limit,
                max_thresholds=self.config.aupro_max_thresholds,
            ),
        }

    def _clean_metric_cache(
        self,
        grouped: Mapping[str, Sequence[MVTecSample]],
        scores: np.ndarray,
        maps: np.ndarray,
        cache_key: str = "full_test",
    ) -> Dict[str, Dict[str, float]]:
        metrics_path = self.output / "clean_metrics" / f"{_safe_name(cache_key)}.json"
        result = {}
        for category, category_samples in tqdm(
            grouped.items(),
            total=len(grouped),
            desc="Clean metrics",
            unit="category",
            dynamic_ncols=True,
        ):
            result[category] = self._category_metrics(category_samples, scores, maps)
        _write_json(metrics_path, result)
        return result

    def _surrogate_transfer_rows(
        self,
        attacker: TargetedPGD,
        samples: Sequence[MVTecSample],
        delta: torch.Tensor,
        target_label: int,
        mode: str,
        condition: str,
        group_name: str,
    ) -> List[Dict[str, object]]:
        rows: List[Dict[str, object]] = []
        for batch in _batches(samples, self.config.target_batch_size):
            clean = torch.stack([self.image_loader(sample) for sample in batch])
            attacked = attacker.apply_universal(clean, delta).detach()
            categories = [sample.category for sample in batch]
            clean_values = attacker.surrogate_scores(clean, categories, mode)
            attacked_values = attacker.surrogate_scores(attacked, categories, mode)
            for position, sample in enumerate(batch):
                clean_mode = float(clean_values["mode_score"][position])
                attacked_mode = float(attacked_values["mode_score"][position])
                rows.append(
                    {
                        "condition": condition,
                        "group": group_name,
                        "sample_id": sample.protocol_id,
                        "dataset": sample.dataset,
                        "category": sample.category,
                        "defect_type": sample.defect_type,
                        "target_label": target_label,
                        "clean_global_score": float(
                            clean_values["global_score"][position]
                        ),
                        "adversarial_global_score": float(
                            attacked_values["global_score"][position]
                        ),
                        "clean_local_score": float(clean_values["local_score"][position]),
                        "adversarial_local_score": float(
                            attacked_values["local_score"][position]
                        ),
                        "clean_mode_score": clean_mode,
                        "adversarial_mode_score": attacked_mode,
                        "clean_mode_prediction": int(clean_mode >= 0.5),
                        "adversarial_mode_prediction": int(attacked_mode >= 0.5),
                        "surrogate_targeted_success": int(
                            int(attacked_mode >= 0.5) == target_label
                        ),
                    }
                )
        return rows

    def _save_representative_examples(
        self,
        evaluation_sources: Sequence[MVTecSample],
        details: Sequence[Mapping[str, object]],
        clean_maps: np.ndarray,
        adversarial_maps: np.ndarray,
        universal_deltas: Mapping[str, torch.Tensor],
        scope: str,
        condition: str,
    ) -> None:
        if self.config.save_adversarial_examples <= 0:
            return
        if scope not in {"dataset", "per_category"}:
            _write_json(
                self.output / "adversarial_examples" / condition / "manifest.json",
                {
                    "status": "not_saved",
                    "reason": (
                        "Representative regeneration is only available for saved "
                        "universal perturbations; per-image deltas are not retained."
                    ),
                },
            )
            return

        sample_lookup = _samples_by_id(evaluation_sources)
        rows_by_category: Dict[str, List[Mapping[str, object]]] = {}
        for row in details:
            rows_by_category.setdefault(str(row["category"]), []).append(row)

        root = self.output / "adversarial_examples" / condition
        manifest: Dict[str, object] = {
            "threshold_mode": self.config.threshold_mode,
            "threshold_quantile": self.config.threshold_quantile,
            "category_thresholds": self.category_thresholds,
            "heatmap_normalization": "joint_min_max_per_clean_adversarial_pair",
            "categories": {},
        }
        role_limit = min(self.config.save_adversarial_examples, 5)
        for category in sorted(rows_by_category):
            rows = rows_by_category[category]
            successful = [row for row in rows if int(row["targeted_success"]) == 1]
            failed = [row for row in rows if int(row["targeted_success"]) == 0]
            finite_lpips = [row for row in rows if np.isfinite(float(row["lpips"]))]
            finite_ssim = [row for row in rows if np.isfinite(float(row["ssim"]))]
            rng = np.random.default_rng(
                self.config.split_seed
                + zlib.crc32(f"{condition}/{category}".encode("utf-8"))
            )
            selections: List[tuple[str, Mapping[str, object] | None]] = [
                (
                    "successful_attack",
                    max(successful, key=lambda row: float(row["directional_score_shift"]))
                    if successful
                    else None,
                ),
                (
                    "failed_attack",
                    min(failed, key=lambda row: float(row["directional_score_shift"]))
                    if failed
                    else None,
                ),
                (
                    "highest_lpips",
                    max(finite_lpips, key=lambda row: float(row["lpips"]))
                    if finite_lpips
                    else None,
                ),
                (
                    "lowest_ssim",
                    min(finite_ssim, key=lambda row: float(row["ssim"]))
                    if finite_ssim
                    else None,
                ),
                (
                    "deterministic_random",
                    rows[int(rng.integers(0, len(rows)))] if rows else None,
                ),
            ][:role_limit]
            category_manifest: Dict[str, object] = {}
            for role, row in selections:
                if row is None:
                    category_manifest[role] = {"status": "unavailable"}
                    continue
                sample = sample_lookup[str(row["sample_id"])]
                group_name = "all_categories" if scope == "dataset" else category
                delta = universal_deltas[group_name]
                clean = self.image_loader(sample)
                attacked = TargetedPGD.apply_universal(
                    clean.unsqueeze(0), delta
                )[0].detach().cpu()
                clean_cpu = clean.detach().cpu()
                role_dir = root / _safe_name(category) / role
                role_dir.mkdir(parents=True, exist_ok=True)
                _save_tensor_image(clean_cpu, role_dir / "clean.png")
                _save_tensor_image(attacked, role_dir / "adversarial.png")

                clean_array = clean_cpu.permute(1, 2, 0).numpy()
                attacked_array = attacked.permute(1, 2, 0).numpy()
                difference = np.clip(
                    0.5 + 10.0 * (attacked_array - clean_array), 0.0, 1.0
                )
                Image.fromarray(
                    (difference * 255.0).round().astype(np.uint8)
                ).save(role_dir / "difference_x10.png")

                mask = load_mask(sample, self.config.metric_size)
                Image.fromarray((mask * 255).astype(np.uint8)).save(
                    role_dir / "ground_truth_mask.png"
                )
                clean_map = resize_anomaly_maps(
                    clean_maps[[sample.index]],
                    self.config.metric_size,
                    self.config.anomaly_map_sigma,
                )[0]
                adversarial_map = resize_anomaly_maps(
                    adversarial_maps[[sample.index]],
                    self.config.metric_size,
                    self.config.anomaly_map_sigma,
                )[0]
                clean_normalized, adversarial_normalized = _normalized_pair(
                    clean_map, adversarial_map
                )
                clean_heatmap = _heatmap_image(clean_normalized)
                adversarial_heatmap = _heatmap_image(adversarial_normalized)
                clean_heatmap.save(role_dir / "clean_heatmap.png")
                adversarial_heatmap.save(role_dir / "adversarial_heatmap.png")
                _signed_heatmap(adversarial_map - clean_map).save(
                    role_dir / "heatmap_difference.png"
                )

                image_size = (self.config.metric_size, self.config.metric_size)
                clean_image = Image.fromarray(_tensor_uint8(clean_cpu)).resize(
                    image_size, Image.Resampling.BICUBIC
                )
                adversarial_image = Image.fromarray(_tensor_uint8(attacked)).resize(
                    image_size, Image.Resampling.BICUBIC
                )
                Image.blend(clean_image, clean_heatmap, 0.45).save(
                    role_dir / "clean_overlay.png"
                )
                Image.blend(adversarial_image, adversarial_heatmap, 0.45).save(
                    role_dir / "adversarial_overlay.png"
                )
                category_manifest[role] = {
                    "status": "saved",
                    "sample_id": sample.protocol_id,
                    "targeted_success": int(row["targeted_success"]),
                    "clean_score": float(row["clean_score"]),
                    "adversarial_score": float(row["adversarial_score"]),
                    "lpips": float(row["lpips"]),
                    "ssim": float(row["ssim"]),
                }
            manifest["categories"][category] = category_manifest  # type: ignore[index]
        _write_json(root / "manifest.json", manifest)

    def _attack_and_predict(
        self,
        fit_source_samples: Sequence[MVTecSample],
        evaluation_source_samples: Sequence[MVTecSample],
        evaluation_samples: Sequence[MVTecSample],
        clean_scores: np.ndarray,
        clean_maps: np.ndarray,
        scope: str,
        direction: str,
        mode: str,
        lpips_metric: LPIPSMetric,
    ) -> tuple[np.ndarray, np.ndarray, List[Dict[str, object]]]:
        source_label, target_label = direction_labels(direction)
        if not fit_source_samples:
            raise RuntimeError(f"No fit source-label samples for {direction}")
        if not evaluation_source_samples:
            raise RuntimeError(f"No source-label samples for {direction}")
        attacker = TargetedPGD(self.surrogate, self.config.attack)
        adversarial_scores = clean_scores.copy()
        adversarial_maps = clean_maps.copy()
        details: List[Dict[str, object]] = []
        condition = _condition(scope, direction, mode)
        diagnostics_dir = self.output / "diagnostics" / condition
        universal_deltas: Dict[str, torch.Tensor] = {}
        optimization_groups: List[Dict[str, object]] = []
        loss_rows: List[Dict[str, object]] = []
        surrogate_rows: List[Dict[str, object]] = []

        _write_json(
            diagnostics_dir / "data_split.json",
            {
                "dataset": self.config.dataset,
                "source_dataset": self.config.source_dataset,
                "evaluation_dataset": self.config.evaluation_dataset,
                "cross_dataset_transfer": self.config.is_cross_dataset,
                "data_protocol_revision": (
                    MATCHED_SPLIT_PROTOCOL
                    if self.config.use_split_manifest
                    else (
                        "cross_dataset_transfer_v1"
                        if self.config.is_cross_dataset
                        else "legacy_asymmetric_v1"
                    )
                ),
                "universal_protocol": (
                    self.config.universal_protocol if scope != "per_image" else "per_image"
                ),
                "scope": scope,
                "direction": direction,
                "fit_fraction": self.config.fit_fraction,
                "split_seed": self.config.split_seed,
                "use_split_manifest": self.config.use_split_manifest,
                "split_manifest_csv": self.config.split_manifest_csv,
                "split_manifest_json": self.config.split_manifest_json,
                "split_manifest_sha256": self.split_manifest_sha256,
                "split_manifest_runtime_counts": self.split_manifest_metadata.get(
                    "runtime_counts", {}
                ),
                "sampling_strategy": (
                    "saved_category_label_balanced_50_50_manifest"
                    if self.config.use_split_manifest
                    else (
                        "source_dataset_fit_destination_dataset_evaluation"
                        if self.config.is_cross_dataset
                        else "legacy_runtime_split"
                    )
                ),
                "fit_source_ids": [sample.protocol_id for sample in fit_source_samples],
                "evaluation_source_ids": [
                    sample.protocol_id for sample in evaluation_source_samples
                ],
                "evaluation_all_ids": [sample.protocol_id for sample in evaluation_samples],
                "fit_source_count": len(fit_source_samples),
                "evaluation_source_count": len(evaluation_source_samples),
                "evaluation_all_count": len(evaluation_samples),
                "fit_source_counts_by_category": _counts_by_category(
                    fit_source_samples
                ),
                "evaluation_source_counts_by_category": _counts_by_category(
                    evaluation_source_samples
                ),
                "evaluation_counts_by_category_and_label": (
                    _counts_by_category_and_label(evaluation_samples)
                ),
            },
        )

        def evaluate_batch(
            batch_samples: Sequence[MVTecSample],
            clean: torch.Tensor,
            attacked: torch.Tensor,
        ) -> None:
            scores, maps = self.target.predict(attacked.detach())
            linf, ssim, lpips_values = perceptual_metrics(
                clean.detach(), attacked.detach(), lpips_metric
            )
            for position, sample in enumerate(batch_samples):
                adversarial_scores[sample.index] = scores[position]
                adversarial_maps[sample.index] = maps[position]
                decision_threshold = self.category_thresholds[sample.category]
                clean_prediction = int(
                    clean_scores[sample.index] >= decision_threshold
                )
                adversarial_prediction = int(scores[position] >= decision_threshold)
                clean_correct = clean_prediction == source_label
                score_shift = float(scores[position] - clean_scores[sample.index])
                directional_score_shift = (
                    score_shift if target_label == 1 else -score_shift
                )
                details.append(
                    {
                        "condition": condition,
                        "target_model": self.config.target_model,
                        "scope": scope,
                        "universal_protocol": (
                            self.config.universal_protocol
                            if scope != "per_image"
                            else "per_image"
                        ),
                        "direction": direction,
                        "loss_mode": mode,
                        "sample_id": sample.sample_id,
                        "dataset": sample.dataset,
                        "category": sample.category,
                        "defect_type": sample.defect_type,
                        "image_path": str(sample.image_path),
                        "mask_path": str(sample.mask_path) if sample.mask_path else "",
                        "label": sample.label,
                        "source_label": source_label,
                        "target_label": target_label,
                        "decision_threshold": decision_threshold,
                        "clean_score": float(clean_scores[sample.index]),
                        "adversarial_score": float(scores[position]),
                        "clean_prediction": clean_prediction,
                        "adversarial_prediction": adversarial_prediction,
                        "clean_correct_for_source": int(clean_correct),
                        "targeted_success": int(adversarial_prediction == target_label),
                        "score_shift": score_shift,
                        "directional_score_shift": directional_score_shift,
                        "directional_success": int(directional_score_shift > 0.0),
                        "linf": float(linf[position]),
                        "ssim": float(ssim[position]),
                        "lpips": float(lpips_values[position]),
                    }
                )
        if scope == "per_image":
            batches = list(
                _batches(
                    evaluation_source_samples,
                    self.config.attack.per_image_batch_size,
                )
            )
            for batch in tqdm(
                batches,
                desc=f"PGD + target [{condition}]",
                unit="batch",
                dynamic_ncols=True,
                leave=False,
            ):
                clean = torch.stack([self.image_loader(sample) for sample in batch])
                attacked, _ = attacker.perturb_batch(
                    clean,
                    [sample.category for sample in batch],
                    target_label,
                    mode,
                )
                evaluate_batch(batch, clean, attacked)
        else:
            if scope == "dataset":
                fit_groups = {"all_categories": list(fit_source_samples)}
                evaluation_groups = {
                    "all_categories": list(evaluation_source_samples)
                }
            elif scope == "per_category":
                fit_groups = group_by_category(fit_source_samples)
                evaluation_groups = group_by_category(evaluation_source_samples)
            else:
                raise ValueError(f"Unknown scope: {scope}")

            if set(fit_groups) != set(evaluation_groups):
                raise RuntimeError(
                    "Universal fit/evaluation category groups do not match: "
                    f"fit={sorted(fit_groups)}, evaluation={sorted(evaluation_groups)}"
                )

            group_progress = tqdm(
                fit_groups.items(),
                total=len(fit_groups),
                desc=f"Universal groups [{condition}]",
                unit="group",
                dynamic_ncols=True,
                leave=False,
            )
            for group_name, group_fit_samples in group_progress:
                group_evaluation_samples = evaluation_groups[group_name]
                group_progress.set_postfix_str(
                    (
                        f"{group_name}: fit={len(group_fit_samples)}, "
                        f"eval={len(group_evaluation_samples)}"
                    ),
                    refresh=False,
                )
                with tqdm(
                    total=self.config.attack.universal_steps,
                    desc=f"Universal PGD [{group_name}]",
                    unit="step",
                    dynamic_ncols=True,
                    leave=False,
                ) as optimization_progress:

                    def report(step: int, total: int, loss: float) -> None:
                        optimization_progress.update(step - optimization_progress.n)
                        optimization_progress.set_postfix(
                            targeted_loss=f"{loss:.6f}", refresh=False
                        )

                    attack_result = attacker.optimize_universal(
                        group_fit_samples,
                        self.image_loader,
                        target_label,
                        mode,
                        diagnostic_samples=self._diagnostic_subset(group_fit_samples),
                        progress=report,
                    )
                delta = attack_result.delta
                group_surrogate_rows = self._surrogate_transfer_rows(
                    attacker,
                    group_evaluation_samples,
                    delta,
                    target_label,
                    mode,
                    condition,
                    group_name,
                )
                surrogate_rows.extend(group_surrogate_rows)
                optimization_groups.append(
                    {
                        "group": group_name,
                        "fit_count": len(group_fit_samples),
                        "evaluation_count": len(group_evaluation_samples),
                        "diagnostic_sample_ids": attack_result.diagnostic_sample_ids,
                        "initial_losses": attack_result.initial_losses,
                        "final_losses": attack_result.final_losses,
                        "loss_reduction": {
                            key: attack_result.initial_losses[key]
                            - attack_result.final_losses[key]
                            for key in attack_result.initial_losses
                            if key in attack_result.final_losses
                        },
                        "surrogate_targeted_success_rate": (
                            float(
                                np.mean(
                                    [
                                        int(row["surrogate_targeted_success"])
                                        for row in group_surrogate_rows
                                    ]
                                )
                            )
                            if group_surrogate_rows
                            else float("nan")
                        ),
                    }
                )
                loss_rows.extend(
                    {"group": group_name, **history_row}
                    for history_row in attack_result.history
                )
                universal_deltas[group_name] = delta.detach().cpu().float()
                if self.config.save_universal_perturbations:
                    perturbation_path = (
                        self.output
                        / "perturbations"
                        / condition
                        / f"{_safe_name(group_name)}.pt"
                    )
                    perturbation_path.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(
                        {
                            "delta": delta.detach().cpu().half(),
                            "epsilon": self.config.attack.epsilon,
                            "scope": scope,
                            "direction": direction,
                            "loss_mode": mode,
                            "group": group_name,
                            "universal_protocol": self.config.universal_protocol,
                            "fit_sample_ids": [
                                sample.protocol_id for sample in group_fit_samples
                            ],
                            "evaluation_sample_ids": [
                                sample.protocol_id
                                for sample in group_evaluation_samples
                            ],
                        },
                        perturbation_path,
                    )
                batches = list(
                    _batches(group_evaluation_samples, self.config.target_batch_size)
                )
                for batch in tqdm(
                    batches,
                    desc=f"Universal target [{group_name}]",
                    unit="batch",
                    dynamic_ncols=True,
                    leave=False,
                ):
                    clean = torch.stack([self.image_loader(sample) for sample in batch])
                    attacked = attacker.apply_universal(clean, delta)
                    evaluate_batch(batch, clean, attacked)
                del delta, attack_result
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            _write_json(
                diagnostics_dir / "optimization.json",
                {
                    "condition": condition,
                    "target_label": target_label,
                    "global_weight": self.config.attack.global_weight,
                    "local_weight": self.config.attack.local_weight,
                    "groups": optimization_groups,
                },
            )
            _write_csv(
                diagnostics_dir / "loss_curve.csv",
                loss_rows,
                LOSS_CURVE_FIELDS,
            )
            _write_csv(
                diagnostics_dir / "surrogate_predictions.csv",
                surrogate_rows,
                SURROGATE_FIELDS,
            )
            self._save_representative_examples(
                evaluation_source_samples,
                details,
                clean_maps,
                adversarial_maps,
                universal_deltas,
                scope,
                condition,
            )
        return adversarial_scores, adversarial_maps, details

    def _summaries(
        self,
        grouped: Mapping[str, Sequence[MVTecSample]],
        clean_metrics: Mapping[str, Mapping[str, float]],
        adversarial_scores: np.ndarray,
        adversarial_maps: np.ndarray,
        details: Sequence[Mapping[str, object]],
        scope: str,
        direction: str,
        mode: str,
    ) -> List[Dict[str, object]]:
        condition = _condition(scope, direction, mode)
        rows: List[Dict[str, object]] = []
        for category, category_samples in tqdm(
            grouped.items(),
            total=len(grouped),
            desc=f"Metrics [{condition}]",
            unit="category",
            dynamic_ncols=True,
            leave=False,
        ):
            attacked_metrics = self._category_metrics(
                category_samples, adversarial_scores, adversarial_maps
            )
            clean = clean_metrics[category]
            category_details = [row for row in details if row["category"] == category]
            eligible = [row for row in category_details if int(row["clean_correct_for_source"]) == 1]
            eligible_success = [
                row for row in eligible if int(row["targeted_success"]) == 1
            ]
            all_success = [
                row for row in category_details if int(row["targeted_success"]) == 1
            ]
            linf = np.asarray([float(row["linf"]) for row in category_details])
            ssim = np.asarray([float(row["ssim"]) for row in category_details])
            lpips = np.asarray([float(row["lpips"]) for row in category_details])
            row = {
                "condition": condition,
                "target_model": self.config.target_model,
                "scope": scope,
                "universal_protocol": (
                    self.config.universal_protocol
                    if scope != "per_image"
                    else "per_image"
                ),
                "direction": direction,
                "loss_mode": mode,
                "dataset": category_samples[0].dataset,
                "category": category,
                "decision_threshold": self.category_thresholds[category],
                "source_count": len(category_details),
                "eligible_clean_correct_count": len(eligible),
                "classification_flip_rate": (
                    100.0 * len(eligible_success) / len(eligible) if eligible else float("nan")
                ),
                "targeted_success_rate_all": (
                    100.0 * len(all_success) / len(category_details)
                    if category_details else float("nan")
                ),
                "clean_i_auroc": clean["i_auroc"],
                "adversarial_i_auroc": attacked_metrics["i_auroc"],
                "delta_i_auroc": clean["i_auroc"] - attacked_metrics["i_auroc"],
                "clean_i_ap": clean["i_ap"],
                "adversarial_i_ap": attacked_metrics["i_ap"],
                "delta_i_ap": clean["i_ap"] - attacked_metrics["i_ap"],
                "clean_p_auroc": clean["p_auroc"],
                "adversarial_p_auroc": attacked_metrics["p_auroc"],
                "delta_p_auroc": clean["p_auroc"] - attacked_metrics["p_auroc"],
                "clean_aupro": clean["aupro"],
                "adversarial_aupro": attacked_metrics["aupro"],
                "delta_aupro": clean["aupro"] - attacked_metrics["aupro"],
                "mean_linf": float(np.mean(linf)) if linf.size else float("nan"),
                "max_linf": float(np.max(linf)) if linf.size else float("nan"),
                "epsilon": self.config.attack.epsilon,
                "mean_ssim": float(np.mean(ssim)) if ssim.size else float("nan"),
                "mean_lpips": float(np.nanmean(lpips)) if np.isfinite(lpips).any() else float("nan"),
            }
            rows.append(row)

        macro: Dict[str, object] = {
            "condition": condition,
            "target_model": self.config.target_model,
            "scope": scope,
            "universal_protocol": (
                self.config.universal_protocol if scope != "per_image" else "per_image"
            ),
            "direction": direction,
            "loss_mode": mode,
            "dataset": self.config.dataset,
            "category": "__macro__",
            "decision_threshold": float("nan"),
            "source_count": sum(int(row["source_count"]) for row in rows),
            "eligible_clean_correct_count": sum(
                int(row["eligible_clean_correct_count"]) for row in rows
            ),
            "epsilon": self.config.attack.epsilon,
        }
        for field in SUMMARY_FIELDS:
            if field in macro or field in {
                "condition", "target_model", "scope", "universal_protocol",
                "direction", "loss_mode", "dataset", "category"
            }:
                continue
            values = np.asarray([float(row[field]) for row in rows], dtype=np.float64)
            macro[field] = float(np.nanmean(values)) if np.isfinite(values).any() else float("nan")
        rows.append(macro)
        return rows

    def run(self) -> Path:
        _seed_everything(self.config.attack.seed)
        sample_cap = (
            None
            if self.config.use_split_manifest
            else self.config.max_samples_per_category
        )
        samples = discover_anomaly_datasets(
            self.config.evaluation_dataset,
            mvtec_root=self.config.mvtec_root,
            visa_root=self.config.visa_root,
            categories=self.config.categories,
            max_samples_per_category=sample_cap,
        )
        source_test_samples: List[MVTecSample] = []
        if self.config.is_cross_dataset:
            source_test_samples = discover_anomaly_datasets(
                self.config.source_dataset,
                mvtec_root=self.config.mvtec_root,
                visa_root=self.config.visa_root,
                categories=self.config.source_categories,
                max_samples_per_category=sample_cap,
            )
        if self.config.use_split_manifest:
            manifest_samples = (
                _reindex_samples([*source_test_samples, *samples])
                if self.config.is_cross_dataset
                else samples
            )
            loaded_manifest: LoadedSplitManifest = load_matched_split_manifest(
                manifest_samples,
                csv_path=str(self.config.split_manifest_csv),
                json_path=str(self.config.split_manifest_json),
                split_seed=self.config.split_seed,
                fit_fraction=self.config.fit_fraction,
                max_samples_per_category=self.config.max_samples_per_category,
                expected_dataset=self.config.manifest_dataset,
            )
            if self.config.is_cross_dataset:
                source_test_samples = _reindex_samples(
                    [
                        sample
                        for sample in loaded_manifest.samples
                        if sample.dataset == self.config.source_dataset
                    ]
                )
                samples = _reindex_samples(
                    [
                        sample
                        for sample in loaded_manifest.samples
                        if sample.dataset == self.config.evaluation_dataset
                    ]
                )
            else:
                samples = loaded_manifest.samples
            self.split_assignments = loaded_manifest.assignments
            self.split_manifest_metadata = loaded_manifest.metadata
            self.split_manifest_sha256 = loaded_manifest.csv_sha256
            print(
                f"[data] Loaded matched split manifest {MATCHED_SPLIT_PROTOCOL} "
                f"(sha256={self.split_manifest_sha256[:12]}...)"
            )

        config_path = self.output / "config.json"
        # JSON-normalize tuples before comparison with the persisted JSON.
        requested_config = json.loads(json.dumps(self.config.to_dict()))
        if self.config.use_split_manifest:
            requested_config["data_protocol_revision"] = MATCHED_SPLIT_PROTOCOL
            requested_config["split_manifest_sha256"] = self.split_manifest_sha256
            requested_config["split_manifest_runtime_counts"] = (
                self.split_manifest_metadata.get("runtime_counts", {})
            )
        _write_json(config_path, requested_config)
        if self.config.use_split_manifest:
            snapshot_dir = self.output / "split_manifest"
            snapshot_dir.mkdir(parents=True, exist_ok=True)
            for source_value in (
                self.config.split_manifest_csv,
                self.config.split_manifest_json,
            ):
                source = Path(str(source_value)).expanduser().resolve()
                destination = snapshot_dir / source.name
                if source != destination.resolve():
                    shutil.copy2(source, destination)
        grouped = group_by_category(samples)
        print(
            f"[data] {len(samples)} {self.config.evaluation_dataset} evaluation "
            "test images across "
            f"{len(grouped)} categories: {', '.join(grouped)}"
        )
        train_good_samples = discover_anomaly_datasets(
            self.config.evaluation_dataset,
            mvtec_root=self.config.mvtec_root,
            visa_root=self.config.visa_root,
            categories=self.config.categories,
            train_normal=True,
        )
        source_train_normal_samples: List[MVTecSample] = []
        if self.config.is_cross_dataset:
            source_train_normal_samples = discover_anomaly_datasets(
                self.config.source_dataset,
                mvtec_root=self.config.mvtec_root,
                visa_root=self.config.visa_root,
                categories=self.config.source_categories,
                train_normal=True,
            )
            print(
                f"[data] Cross-dataset attack fit: {self.config.source_dataset} -> "
                f"{self.config.evaluation_dataset}; {len(source_test_samples)} source "
                "test images available"
            )
        print(
            f"[data] {len(train_good_samples)} normal training images available "
            + (
                "for threshold calibration only"
                if self.config.use_split_manifest
                else "for threshold calibration and held-out universal fitting"
            )
        )
        self._load_target()
        try:
            self._prepare_category_thresholds(
                train_good_samples, list(grouped)
            )
            prompt_categories = sorted(
                set(grouped)
                | {sample.category for sample in source_test_samples}
                | {sample.category for sample in source_train_normal_samples}
            )
            self._load_surrogate(prompt_categories)
            lpips_metric = LPIPSMetric(
                self.config.device,
                backbone=self.config.lpips_backbone,
                enabled=self.config.compute_lpips,
            )
            if lpips_metric.error:
                print(
                    f"[warning] LPIPS unavailable; values will be NaN: "
                    f"{lpips_metric.error}"
                )
            clean_scores, clean_maps = self._clean_predictions(samples)
            conditions = [
                (direction, mode, scope)
                for direction in self.config.attack.directions
                for mode in self.config.attack.loss_modes
                for scope in self.config.attack.scopes
            ]
            condition_progress = tqdm(
                conditions,
                desc="Experiment conditions",
                unit="condition",
                dynamic_ncols=True,
            )
            for direction, mode, scope in condition_progress:
                condition = _condition(scope, direction, mode)
                condition_progress.set_postfix_str(condition, refresh=True)
                tqdm.write("=" * 80)
                tqdm.write(f"[condition] {condition}")
                started = time.time()
                fit_sources, evaluation_sources, evaluation_samples = (
                    self._protocol_samples(
                        samples,
                        train_good_samples,
                        scope,
                        direction,
                        source_test_samples=source_test_samples,
                        source_train_normal_samples=source_train_normal_samples,
                    )
                )
                evaluation_grouped = group_by_category(evaluation_samples)
                if (
                    scope == "per_image"
                    or self.config.universal_protocol == "transductive"
                ):
                    clean_cache_key = "full_test"
                elif self.config.use_split_manifest:
                    clean_cache_key = (
                        f"{MATCHED_SPLIT_PROTOCOL}__{self.split_manifest_sha256}"
                        f"__max_{self.config.max_samples_per_category}"
                    )
                elif self.config.is_cross_dataset:
                    clean_cache_key = "cross_dataset_full_destination_test"
                elif direction == "normal_to_abnormal":
                    clean_cache_key = "full_test"
                else:
                    clean_cache_key = (
                        f"held_out__{direction}__fraction_{self.config.fit_fraction}"
                        f"__seed_{self.config.split_seed}"
                    )
                clean_metrics = self._clean_metric_cache(
                    evaluation_grouped,
                    clean_scores,
                    clean_maps,
                    cache_key=clean_cache_key,
                )
                adversarial_scores, adversarial_maps, details = self._attack_and_predict(
                    fit_sources,
                    evaluation_sources,
                    evaluation_samples,
                    clean_scores,
                    clean_maps,
                    scope,
                    direction,
                    mode,
                    lpips_metric,
                )
                summaries = self._summaries(
                    evaluation_grouped,
                    clean_metrics,
                    adversarial_scores,
                    adversarial_maps,
                    details,
                    scope,
                    direction,
                    mode,
                )
                condition_dir = self.output / "predictions" / condition
                condition_dir.mkdir(parents=True, exist_ok=True)
                evaluation_indices = np.asarray(
                    [sample.index for sample in evaluation_samples], dtype=np.int64
                )
                np.savez_compressed(
                    condition_dir / "target_outputs.npz",
                    labels=np.asarray(
                        [sample.label for sample in evaluation_samples], dtype=np.uint8
                    ),
                    sample_ids=np.asarray(
                        [sample.sample_id for sample in evaluation_samples]
                    ),
                    datasets=np.asarray(
                        [sample.dataset for sample in evaluation_samples]
                    ),
                    protocol_ids=np.asarray(
                        [sample.protocol_id for sample in evaluation_samples]
                    ),
                    original_test_indices=evaluation_indices,
                    decision_thresholds=np.asarray(
                        [
                            self.category_thresholds[sample.category]
                            for sample in evaluation_samples
                        ],
                        dtype=np.float32,
                    ),
                    clean_scores=clean_scores[evaluation_indices],
                    adversarial_scores=adversarial_scores[evaluation_indices],
                    clean_lowres_maps=clean_maps[evaluation_indices],
                    adversarial_lowres_maps=adversarial_maps[evaluation_indices],
                )
                self.summary_rows = [
                    row for row in self.summary_rows if row.get("condition") != condition
                ] + summaries
                self.detail_rows = [
                    row for row in self.detail_rows if row.get("condition") != condition
                ] + details
                _write_csv(self.summary_path, self.summary_rows, SUMMARY_FIELDS)
                _write_csv(self.detail_path, self.detail_rows, DETAIL_FIELDS)
                tqdm.write(
                    f"[condition] completed {condition} in "
                    f"{(time.time() - started) / 60.0:.1f} minutes"
                )
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        finally:
            if self.surrogate is not None:
                self.surrogate.release()
            if self.target is not None:
                self.target.release()
        print(f"[done] Summary: {self.summary_path}")
        return self.summary_path


def run_experiment(config: ExperimentConfig) -> Path:
    """Public entrypoint used by local scripts and the Kaggle notebook."""

    return AdversarialExperiment(config).run()


def calibrate_thresholds(config: ExperimentConfig) -> Path:
    """Calibrate reusable category thresholds without loading the surrogate."""

    if config.thresholds_path:
        raise ValueError(
            "calibrate_thresholds creates a threshold artifact; thresholds_path "
            "must be None"
        )
    _seed_everything(config.attack.seed)
    experiment = AdversarialExperiment(config)
    _write_json(
        experiment.output / "threshold_config.json",
        json.loads(json.dumps(config.to_dict())),
    )
    train_good_samples = discover_anomaly_datasets(
        config.evaluation_dataset,
        mvtec_root=config.mvtec_root,
        visa_root=config.visa_root,
        categories=config.categories,
        train_normal=True,
    )
    categories = sorted(group_by_category(train_good_samples))
    print(
        f"[data] Calibrating {len(categories)} categories from "
        f"{len(train_good_samples)} normal training images from "
        f"{config.evaluation_dataset}"
    )
    experiment._load_target()
    try:
        return experiment._prepare_category_thresholds(train_good_samples, categories)
    finally:
        if experiment.target is not None:
            experiment.target.release()
