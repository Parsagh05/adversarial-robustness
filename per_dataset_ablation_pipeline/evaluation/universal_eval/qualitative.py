"""Representative qualitative exports for fixed adversarial attacks."""

from __future__ import annotations

import json
import math
from pathlib import Path
import re
import shutil
from typing import Any

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
import torch

from .datasets import EvaluationSample, load_image, load_mask
from .metrics import location_free_topk_region, resize_anomaly_maps


Prediction = tuple[float, np.ndarray]


def _resolved_selection_basis(
    rows: list[dict[str, Any]], selection_basis: str
) -> str:
    if selection_basis != "direction_aware_pixel":
        return selection_basis
    target_labels = {int(row["target_label"]) for row in rows}
    if len(target_labels) != 1:
        raise ValueError(
            "direction_aware_pixel selection requires one attack direction"
        )
    return (
        "location_free_topk_pixel"
        if target_labels == {1}
        else "target_region_pixel"
    )


def select_representative_rows(
    rows: list[dict[str, Any]],
    *,
    selection_basis: str = "image",
) -> list[tuple[str, dict[str, Any]]]:
    """Select strongest/median successes and the worst eligible failure."""

    selection_basis = _resolved_selection_basis(rows, selection_basis)
    if selection_basis == "image":
        eligible_field = "targeted_success_eligible"
        success_field = "targeted_attack_success"
        strength_field = "adversarial_target_margin"
    elif selection_basis == "target_region_pixel":
        eligible_field = "target_region_pixel_success_eligible"
        success_field = "target_region_pixel_attack_success"
        strength_field = "target_region_pixel_flip_rate"
    elif selection_basis == "location_free_topk_pixel":
        eligible_field = "location_free_topk_pixel_success_eligible"
        success_field = "location_free_topk_pixel_attack_success"
        strength_field = "location_free_topk_pixel_flip_rate"
    else:
        raise ValueError(
            "selection_basis must be image, target_region_pixel, "
            "location_free_topk_pixel, or direction_aware_pixel"
        )
    eligible = [row for row in rows if int(row.get(eligible_field, 0))]
    successes = sorted(
        (row for row in eligible if int(row[success_field])),
        key=lambda row: float(row[strength_field]),
    )
    failures = sorted(
        (row for row in eligible if not int(row[success_field])),
        key=lambda row: float(row[strength_field]),
    )
    selected: list[tuple[str, dict[str, Any]]] = []
    if successes:
        selected.append(("strongest_success", successes[-1]))
        if len(successes) > 1:
            median = successes[(len(successes) - 1) // 2]
            if median["sample_id"] != successes[-1]["sample_id"]:
                selected.append(("median_success", median))
    if failures:
        selected.append(("worst_failure", failures[0]))
    return selected


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "__", value).strip("._")


def _rgb_uint8(tensor: torch.Tensor) -> np.ndarray:
    return (
        tensor.detach()
        .cpu()
        .permute(1, 2, 0)
        .clamp(0.0, 1.0)
        .mul(255.0)
        .round()
        .byte()
        .numpy()
    )


def _anomaly_colormap(values: np.ndarray) -> np.ndarray:
    stops = np.asarray([0.0, 0.25, 0.50, 0.75, 1.0])
    colors = np.asarray(
        [
            [0, 0, 128],
            [0, 128, 255],
            [255, 255, 0],
            [255, 0, 0],
            [255, 255, 255],
        ],
        dtype=np.float32,
    )
    clipped = np.clip(values, 0.0, 1.0)
    channels = [np.interp(clipped, stops, colors[:, index]) for index in range(3)]
    return np.stack(channels, axis=-1).round().astype(np.uint8)


def _normalize_pair(clean: np.ndarray, adversarial: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    minimum = float(min(clean.min(), adversarial.min()))
    maximum = float(max(clean.max(), adversarial.max()))
    scale = maximum - minimum
    if scale <= np.finfo(np.float32).eps:
        return np.zeros_like(clean), np.zeros_like(adversarial)
    return (clean - minimum) / scale, (adversarial - minimum) / scale


def _difference_colormap(difference: np.ndarray) -> np.ndarray:
    bound = float(np.max(np.abs(difference)))
    if bound <= np.finfo(np.float32).eps:
        return np.zeros((*difference.shape, 3), dtype=np.uint8)
    normalized = np.clip(difference / bound, -1.0, 1.0)
    output = np.zeros((*difference.shape, 3), dtype=np.float32)
    positive = np.clip(normalized, 0.0, 1.0)
    negative = np.clip(-normalized, 0.0, 1.0)
    output[..., 0] = 255.0 * positive
    output[..., 1] = 255.0 * np.maximum(0.0, 2.0 * positive - 1.0)
    output[..., 2] = 255.0 * negative
    return output.round().astype(np.uint8)


def _target_region_from_row(
    row: dict[str, Any], ground_truth_mask: np.ndarray
) -> np.ndarray:
    """Recreate the region used by the thresholded pixel-success calculation."""

    mask = np.asarray(ground_truth_mask) > 0
    if int(row["target_label"]) == 0:
        return mask
    if str(row.get("normal_local_target", "full_image")) == "full_image":
        return np.ones_like(mask, dtype=bool)
    height, width = mask.shape
    fraction = float(row["normal_target_region_fraction"])
    region_height = max(1, int(round(height * fraction)))
    region_width = max(1, int(round(width * fraction)))
    center_column = int(round(float(row["normal_target_center_x"]) * (width - 1)))
    center_row = int(round(float(row["normal_target_center_y"]) * (height - 1)))
    left = min(max(center_column - region_width // 2, 0), width - region_width)
    top = min(max(center_row - region_height // 2, 0), height - region_height)
    region = np.zeros_like(mask, dtype=bool)
    region[top : top + region_height, left : left + region_width] = True
    return region


def _ssim(clean: np.ndarray, adversarial: np.ndarray) -> float:
    c1 = 0.01**2
    c2 = 0.03**2
    values: list[float] = []
    for channel in range(3):
        x = clean[..., channel].astype(np.float64)
        y = adversarial[..., channel].astype(np.float64)
        mu_x = gaussian_filter(x, sigma=1.5)
        mu_y = gaussian_filter(y, sigma=1.5)
        sigma_x = gaussian_filter(x * x, sigma=1.5) - mu_x * mu_x
        sigma_y = gaussian_filter(y * y, sigma=1.5) - mu_y * mu_y
        sigma_xy = gaussian_filter(x * y, sigma=1.5) - mu_x * mu_y
        numerator = (2.0 * mu_x * mu_y + c1) * (2.0 * sigma_xy + c2)
        denominator = (mu_x * mu_x + mu_y * mu_y + c1) * (
            sigma_x + sigma_y + c2
        )
        values.append(float(np.mean(numerator / denominator)))
    return float(np.mean(values))


def _sample_metrics(
    row: dict[str, Any],
    clean: np.ndarray,
    adversarial: np.ndarray,
    clean_map: np.ndarray,
    adversarial_map: np.ndarray,
    *,
    selection: str,
) -> dict[str, Any]:
    difference = adversarial - clean
    map_difference = adversarial_map - clean_map
    mse = float(np.mean(difference.astype(np.float64) ** 2))
    metrics = dict(row)
    metrics.update(
        {
            "selection": selection,
            "perturbation_l2": float(np.linalg.norm(difference)),
            "mean_absolute_pixel_difference": float(np.mean(np.abs(difference))),
            "psnr": None if mse == 0.0 else -10.0 * math.log10(mse),
            "ssim": _ssim(clean, adversarial),
            "clean_heatmap_mean": float(clean_map.mean()),
            "clean_heatmap_max": float(clean_map.max()),
            "adversarial_heatmap_mean": float(adversarial_map.mean()),
            "adversarial_heatmap_max": float(adversarial_map.max()),
            "heatmap_mean_change": float(map_difference.mean()),
            "heatmap_max_increase": float(map_difference.max()),
            "increased_anomaly_pixel_fraction": float((map_difference > 0).mean()),
        }
    )
    return metrics


def _finite_json(value: Any) -> Any:
    """Replace non-finite diagnostic values with JSON null."""

    if isinstance(value, dict):
        return {key: _finite_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite_json(item) for item in value]
    if isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
        return None
    return value


def export_representative_samples(
    output_root: str | Path,
    condition: str,
    rows: list[dict[str, Any]],
    samples_by_id: dict[str, EvaluationSample],
    clean_predictions: dict[str, Prediction],
    adversarial_predictions: dict[str, Prediction],
    delta: torch.Tensor,
    delta_indices: dict[str, int],
    *,
    image_size: int,
    anomaly_map_sigma: float,
    selection_basis: str = "image",
) -> list[dict[str, Any]]:
    """Write one self-contained folder for every selected representative sample."""

    root = Path(output_root).expanduser().resolve()
    condition_root = root / _safe_name(condition)
    if condition_root.parent != root:
        raise ValueError(f"Unsafe qualitative condition path: {condition_root}")
    if condition_root.exists():
        shutil.rmtree(condition_root)
    condition_root.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, Any]] = []
    resolved_basis = _resolved_selection_basis(rows, selection_basis)
    for selection, row in select_representative_rows(
        rows, selection_basis=selection_basis
    ):
        sample_id = str(row["sample_id"])
        sample = samples_by_id[sample_id]
        clean_tensor = load_image(sample, image_size)
        adversarial_tensor = (
            clean_tensor + delta[delta_indices[sample_id]]
        ).clamp(0.0, 1.0)
        clean = clean_tensor.permute(1, 2, 0).numpy()
        adversarial = adversarial_tensor.permute(1, 2, 0).numpy()
        clean_map = resize_anomaly_maps(
            [clean_predictions[sample_id][1]], image_size, anomaly_map_sigma
        )[0]
        adversarial_map = resize_anomaly_maps(
            [adversarial_predictions[sample_id][1]], image_size, anomaly_map_sigma
        )[0]
        clean_normalized, adversarial_normalized = _normalize_pair(
            clean_map, adversarial_map
        )
        clean_heatmap = _anomaly_colormap(clean_normalized)
        adversarial_heatmap = _anomaly_colormap(adversarial_normalized)
        clean_rgb = _rgb_uint8(clean_tensor)
        adversarial_rgb = _rgb_uint8(adversarial_tensor)
        folder = condition_root / f"{selection}__{_safe_name(sample_id)}"
        folder.mkdir(parents=True, exist_ok=True)
        Image.fromarray(clean_rgb).save(folder / "clean.png")
        Image.fromarray(adversarial_rgb).save(folder / "adversarial.png")
        amplified = np.clip(np.abs(adversarial - clean) * 10.0, 0.0, 1.0)
        Image.fromarray((amplified * 255.0).round().astype(np.uint8)).save(
            folder / "difference_x10.png"
        )
        Image.fromarray(clean_heatmap).save(folder / "clean_heatmap.png")
        Image.fromarray(adversarial_heatmap).save(folder / "adversarial_heatmap.png")
        Image.blend(
            Image.fromarray(clean_rgb), Image.fromarray(clean_heatmap), 0.45
        ).save(folder / "clean_overlay.png")
        Image.blend(
            Image.fromarray(adversarial_rgb),
            Image.fromarray(adversarial_heatmap),
            0.45,
        ).save(folder / "adversarial_overlay.png")
        ground_truth = load_mask(sample, image_size)
        Image.fromarray((ground_truth * 255).astype(np.uint8)).save(
            folder / "ground_truth_mask.png"
        )
        pixel_threshold = float(row["pixel_decision_threshold"])
        clean_pixel_prediction = clean_map >= pixel_threshold
        adversarial_pixel_prediction = adversarial_map >= pixel_threshold
        target_region = _target_region_from_row(row, ground_truth)
        successful_target_flips = (
            target_region
            & (clean_pixel_prediction == int(row["source_label"]))
            & (adversarial_pixel_prediction == int(row["target_label"]))
            & (clean_pixel_prediction != adversarial_pixel_prediction)
        )
        Image.fromarray((clean_pixel_prediction * 255).astype(np.uint8)).save(
            folder / "clean_pixel_prediction.png"
        )
        Image.fromarray((adversarial_pixel_prediction * 255).astype(np.uint8)).save(
            folder / "adversarial_pixel_prediction.png"
        )
        Image.fromarray((target_region * 255).astype(np.uint8)).save(
            folder / "target_region_mask.png"
        )
        Image.fromarray((successful_target_flips * 255).astype(np.uint8)).save(
            folder / "successful_target_pixel_flips.png"
        )
        if int(row["target_label"]) == 1:
            location_free_region = location_free_topk_region(
                adversarial_map,
                topk_fraction=float(row["location_free_topk_fraction"]),
            )
            successful_location_free_flips = (
                location_free_region
                & (clean_pixel_prediction == int(row["source_label"]))
                & (adversarial_pixel_prediction == int(row["target_label"]))
                & (clean_pixel_prediction != adversarial_pixel_prediction)
            )
            Image.fromarray((location_free_region * 255).astype(np.uint8)).save(
                folder / "location_free_topk_region_mask.png"
            )
            Image.fromarray(
                (successful_location_free_flips * 255).astype(np.uint8)
            ).save(folder / "successful_location_free_topk_pixel_flips.png")
        Image.fromarray(_difference_colormap(adversarial_map - clean_map)).save(
            folder / "heatmap_difference.png"
        )
        metrics = _sample_metrics(
            row,
            clean,
            adversarial,
            clean_map,
            adversarial_map,
            selection=selection,
        )
        (folder / "metrics.json").write_text(
            json.dumps(_finite_json(metrics), indent=2, allow_nan=False),
            encoding="utf-8",
        )
        manifest.append(
            {
                "selection": selection,
                "sample_id": sample_id,
                "folder": str(folder.relative_to(condition_root.parent)),
                "target_margin": float(row["adversarial_target_margin"]),
                "target_region_pixel_flip_rate": float(
                    row["target_region_pixel_flip_rate"]
                ),
                "location_free_topk_pixel_flip_rate": (
                    float(row["location_free_topk_pixel_flip_rate"])
                    if int(row["target_label"]) == 1
                    else None
                ),
            }
        )
    if resolved_basis == "target_region_pixel":
        eligible_field = "target_region_pixel_success_eligible"
        success_field = "target_region_pixel_attack_success"
        ranking = (
            "Strongest and median successes use target-region pixel flip rate; "
            "worst failure uses the smallest target-region pixel flip rate."
        )
    elif resolved_basis == "location_free_topk_pixel":
        eligible_field = "location_free_topk_pixel_success_eligible"
        success_field = "location_free_topk_pixel_attack_success"
        ranking = (
            "Strongest and median successes use location-free Top-K pixel flip "
            "rate; worst failure uses the smallest location-free rate."
        )
    else:
        eligible_field = "targeted_success_eligible"
        success_field = "targeted_attack_success"
        ranking = (
            "Target margin is positive inside the target class. Strongest success "
            "uses the largest margin; worst failure uses the smallest margin."
        )
    eligible_count = sum(int(row.get(eligible_field, 0)) for row in rows)
    success_count = sum(int(row.get(success_field, 0)) for row in rows)
    selected_names = {record["selection"] for record in manifest}
    selection_manifest = {
        "condition": condition,
        "selection_basis": resolved_basis,
        "ranking": ranking,
        "eligible_count": eligible_count,
        "success_count": success_count,
        "failure_count": eligible_count - success_count,
        "missing_selections": [
            name
            for name in ("strongest_success", "median_success", "worst_failure")
            if name not in selected_names
        ],
        "selected": manifest,
    }
    (condition_root / "selection_manifest.json").write_text(
        json.dumps(selection_manifest, indent=2), encoding="utf-8"
    )
    return manifest
