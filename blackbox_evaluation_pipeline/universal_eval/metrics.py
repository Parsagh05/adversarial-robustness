"""Continuous image- and pixel-level anomaly-detection metrics."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy.ndimage import gaussian_filter, label as connected_components
import torch
import torch.nn.functional as F


def _binary_curve(labels: Sequence[int], scores: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    labels_array = np.asarray(labels, dtype=np.uint8)
    scores_array = np.asarray(scores, dtype=np.float64)
    if labels_array.ndim != 1 or labels_array.shape != scores_array.shape:
        raise ValueError("Labels and scores must be matching one-dimensional arrays")
    if labels_array.size == 0 or not np.isfinite(scores_array).all():
        raise ValueError("Metric inputs must be non-empty and finite")
    order = np.argsort(scores_array, kind="mergesort")[::-1]
    sorted_scores = scores_array[order]
    sorted_labels = labels_array[order]
    indices = np.r_[np.where(np.diff(sorted_scores))[0], labels_array.size - 1]
    true_positives = np.cumsum(sorted_labels, dtype=np.float64)[indices]
    false_positives = 1.0 + indices - true_positives
    return false_positives, true_positives


def _auroc(labels: Sequence[int], scores: Sequence[float]) -> float:
    fp, tp = _binary_curve(labels, scores)
    fpr = np.r_[0.0, fp / fp[-1]]
    tpr = np.r_[0.0, tp / tp[-1]]
    trapezoid = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    return float(trapezoid(tpr, fpr))


def _average_precision(labels: Sequence[int], scores: Sequence[float]) -> float:
    fp, tp = _binary_curve(labels, scores)
    precision = tp / (tp + fp)
    recall = tp / tp[-1]
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


def image_metrics(labels: Sequence[int], scores: Sequence[float]) -> dict[str, float]:
    labels_array = np.asarray(labels, dtype=np.uint8)
    if np.unique(labels_array).size < 2:
        return {"i_auroc": float("nan"), "i_ap": float("nan")}
    return {
        "i_auroc": 100.0 * _auroc(labels_array, scores),
        "i_ap": 100.0 * _average_precision(labels_array, scores),
    }


def resize_anomaly_maps(lowres_maps: Sequence[np.ndarray], size: int, sigma: float) -> np.ndarray:
    tensor = torch.as_tensor(np.stack(lowres_maps), dtype=torch.float32)[:, None]
    resized = F.interpolate(tensor, size=(size, size), mode="bilinear", align_corners=False)[:, 0].numpy()
    if sigma > 0:
        resized = np.stack([gaussian_filter(item, sigma=sigma) for item in resized])
    return resized.astype(np.float32)


def compute_aupro(
    masks: np.ndarray,
    maps: np.ndarray,
    *,
    fpr_limit: float = 0.30,
    max_thresholds: int = 200,
) -> float:
    masks = (np.asarray(masks) > 0).astype(np.uint8)
    maps = np.asarray(maps, dtype=np.float32)
    negatives = masks == 0
    negative_count = int(negatives.sum())
    regions: list[list[np.ndarray]] = []
    for mask in masks:
        component_map, count = connected_components(mask, structure=np.ones((3, 3)))
        regions.append([component_map == index for index in range(1, count + 1)])
    if negative_count == 0 or sum(map(len, regions)) == 0:
        return float("nan")

    flat = maps.reshape(-1)
    stride = max(1, int(np.ceil(flat.size / 1_000_000)))
    sampled = flat[::stride]
    quantiles = np.linspace(1.0, 0.0, min(max_thresholds, sampled.size))
    thresholds = np.unique(np.quantile(sampled, quantiles))[::-1]
    fprs = [0.0]
    pros = [0.0]
    for threshold in thresholds:
        prediction = maps >= float(threshold)
        fprs.append(float(np.logical_and(prediction, negatives).sum()) / negative_count)
        overlaps = [
            float(prediction[image_index][region].mean())
            for image_index, image_regions in enumerate(regions)
            for region in image_regions
        ]
        pros.append(float(np.mean(overlaps)))
    order = np.argsort(fprs)
    fprs_array = np.asarray(fprs)[order]
    pros_array = np.asarray(pros)[order]
    unique_fprs = np.unique(fprs_array)
    unique_pros = np.asarray([pros_array[fprs_array == value].max() for value in unique_fprs])
    boundary = float(np.interp(fpr_limit, unique_fprs, unique_pros))
    keep = unique_fprs < fpr_limit
    x = np.concatenate([unique_fprs[keep], [fpr_limit]])
    y = np.concatenate([unique_pros[keep], [boundary]])
    if x[0] > 0:
        x, y = np.r_[0.0, x], np.r_[0.0, y]
    trapezoid = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    return 100.0 * float(trapezoid(y, x) / fpr_limit)


def pixel_metrics(
    masks: np.ndarray,
    maps: np.ndarray,
    *,
    fpr_limit: float,
    max_thresholds: int,
) -> dict[str, float]:
    flat_masks = (np.asarray(masks) > 0).reshape(-1).astype(np.uint8)
    flat_maps = np.asarray(maps, dtype=np.float32).reshape(-1)
    if np.unique(flat_masks).size < 2:
        return {"p_auroc": float("nan"), "aupro": float("nan")}
    return {
        "p_auroc": 100.0 * _auroc(flat_masks, flat_maps),
        "aupro": compute_aupro(
            masks, maps, fpr_limit=fpr_limit, max_thresholds=max_thresholds
        ),
    }

