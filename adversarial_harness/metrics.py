"""Effectiveness and perceptual-budget metrics."""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter
from skimage.measure import label as connected_components
from skimage.metrics import structural_similarity


def _binary_curve(
    labels: Sequence[int], scores: Sequence[float]
) -> Tuple[np.ndarray, np.ndarray]:
    """Return cumulative false/true positives at distinct score thresholds."""

    labels_array = np.asarray(labels, dtype=np.uint8)
    scores_array = np.asarray(scores, dtype=np.float64)
    if labels_array.ndim != 1 or scores_array.ndim != 1:
        raise ValueError("Binary labels and scores must be one-dimensional")
    if labels_array.shape != scores_array.shape:
        raise ValueError("Binary labels and scores must have matching shapes")
    if not np.isfinite(scores_array).all():
        raise ValueError("Binary metric scores must be finite")
    if not np.isin(labels_array, (0, 1)).all():
        raise ValueError("Binary metric labels must contain only 0 and 1")

    order = np.argsort(scores_array, kind="mergesort")[::-1]
    sorted_scores = scores_array[order]
    sorted_labels = labels_array[order]
    distinct_indices = np.where(np.diff(sorted_scores))[0]
    threshold_indices = np.r_[distinct_indices, sorted_labels.size - 1]
    true_positives = np.cumsum(sorted_labels, dtype=np.float64)[threshold_indices]
    false_positives = 1.0 + threshold_indices - true_positives
    return false_positives, true_positives


def _binary_auroc(labels: Sequence[int], scores: Sequence[float]) -> float:
    false_positives, true_positives = _binary_curve(labels, scores)
    negative_count = false_positives[-1]
    positive_count = true_positives[-1]
    fpr = np.r_[0.0, false_positives / negative_count]
    tpr = np.r_[0.0, true_positives / positive_count]
    trapezoid = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    return float(trapezoid(tpr, fpr))


def _binary_average_precision(
    labels: Sequence[int], scores: Sequence[float]
) -> float:
    false_positives, true_positives = _binary_curve(labels, scores)
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / true_positives[-1]
    recall_increments = np.diff(np.r_[0.0, recall])
    return float(np.sum(recall_increments * precision))


def image_metrics(labels: Sequence[int], scores: Sequence[float]) -> Dict[str, float]:
    labels_array = np.asarray(labels, dtype=np.uint8)
    scores_array = np.asarray(scores, dtype=np.float64)
    if np.unique(labels_array).size < 2:
        return {"i_auroc": float("nan"), "i_ap": float("nan")}
    return {
        "i_auroc": 100.0 * _binary_auroc(labels_array, scores_array),
        "i_ap": 100.0 * _binary_average_precision(labels_array, scores_array),
    }


def resize_anomaly_maps(
    lowres_maps: Sequence[np.ndarray],
    size: int,
    sigma: float,
) -> np.ndarray:
    tensor = torch.as_tensor(np.stack(lowres_maps), dtype=torch.float32)[:, None]
    resized = F.interpolate(
        tensor, size=(size, size), mode="bilinear", align_corners=False
    )[:, 0].numpy()
    if sigma > 0:
        resized = np.stack([gaussian_filter(item, sigma=sigma) for item in resized])
    return resized.astype(np.float32)


def compute_aupro(
    masks: np.ndarray,
    maps: np.ndarray,
    fpr_limit: float = 0.30,
    max_thresholds: int = 200,
) -> float:
    """Region-overlap AUC normalized over ``[0, fpr_limit]``.

    Thresholds are sampled from score quantiles rather than a fixed raw-score
    interval, making the computation invariant to monotonic score scaling.
    """

    masks = (np.asarray(masks) > 0).astype(np.uint8)
    maps = np.asarray(maps, dtype=np.float32)
    negatives = masks == 0
    negative_count = int(negatives.sum())
    regions = []
    for mask in masks:
        component_map = connected_components(mask, connectivity=2)
        regions.append(
            [component_map == index for index in range(1, int(component_map.max()) + 1)]
        )
    region_count = sum(len(items) for items in regions)
    if negative_count == 0 or region_count == 0:
        return 0.0

    flat = maps.reshape(-1)
    # Quantiles from a deterministic, evenly spaced score sample avoid sorting
    # hundreds of millions of full-resolution pixels only to choose 200 points.
    stride = max(1, int(np.ceil(flat.size / 1_000_000)))
    sampled_scores = flat[::stride]
    quantiles = np.linspace(1.0, 0.0, min(max_thresholds, sampled_scores.size))
    thresholds = np.unique(np.quantile(sampled_scores, quantiles))[::-1]
    fprs = [0.0]
    pros = [0.0]
    for threshold in thresholds:
        prediction = maps >= float(threshold)
        fpr = float(np.logical_and(prediction, negatives).sum()) / negative_count
        overlaps = []
        for image_index, image_regions in enumerate(regions):
            for region in image_regions:
                overlaps.append(float(prediction[image_index][region].mean()))
        fprs.append(fpr)
        pros.append(float(np.mean(overlaps)))

    order = np.argsort(fprs)
    fprs_array = np.asarray(fprs)[order]
    pros_array = np.asarray(pros)[order]
    unique_fprs = np.unique(fprs_array)
    unique_pros = np.asarray(
        [pros_array[fprs_array == value].max() for value in unique_fprs]
    )
    boundary_pro = float(np.interp(fpr_limit, unique_fprs, unique_pros))
    keep = unique_fprs < fpr_limit
    x = np.concatenate([unique_fprs[keep], [fpr_limit]])
    y = np.concatenate([unique_pros[keep], [boundary_pro]])
    if x[0] > 0:
        x = np.concatenate([[0.0], x])
        y = np.concatenate([[0.0], y])
    trapezoid = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    return 100.0 * float(trapezoid(y, x) / fpr_limit)


def pixel_metrics(
    masks: np.ndarray,
    maps: np.ndarray,
    fpr_limit: float,
    max_thresholds: int,
) -> Dict[str, float]:
    flat_masks = (np.asarray(masks) > 0).reshape(-1).astype(np.uint8)
    flat_maps = np.asarray(maps, dtype=np.float32).reshape(-1)
    if np.unique(flat_masks).size < 2:
        return {"p_auroc": float("nan"), "aupro": float("nan")}
    return {
        "p_auroc": 100.0 * _binary_auroc(flat_masks, flat_maps),
        "aupro": compute_aupro(
            masks, maps, fpr_limit=fpr_limit, max_thresholds=max_thresholds
        ),
    }


class LPIPSMetric:
    """Small optional wrapper so the benchmark can report missing LPIPS clearly."""

    def __init__(self, device: str, backbone: str = "alex", enabled: bool = True):
        self.device = torch.device(device)
        self.model = None
        self.error: Optional[str] = None
        if not enabled:
            return
        try:
            import lpips

            self.model = lpips.LPIPS(net=backbone).to(self.device).eval()
        except Exception as exc:  # dependency/download failures remain explicit in output
            self.error = repr(exc)

    def __call__(self, clean: torch.Tensor, adversarial: torch.Tensor) -> np.ndarray:
        if self.model is None:
            return np.full(clean.shape[0], np.nan, dtype=np.float32)
        with torch.inference_mode():
            values = self.model(
                clean.to(self.device) * 2.0 - 1.0,
                adversarial.to(self.device) * 2.0 - 1.0,
            )
        return values.reshape(-1).detach().cpu().numpy().astype(np.float32)


def perceptual_metrics(
    clean: torch.Tensor,
    adversarial: torch.Tensor,
    lpips_metric: LPIPSMetric,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Run LPIPS before copying tensors to CPU. In the attack pipeline the
    # adversarial tensor is already on the model device, so this avoids an
    # unnecessary GPU -> CPU -> GPU round trip. L-infinity and SSIM remain on
    # CPU and retain their existing numerical behavior.
    lpips_values = lpips_metric(clean.detach(), adversarial.detach())
    clean_cpu = clean.detach().cpu().float()
    adversarial_cpu = adversarial.detach().cpu().float()
    linf = (adversarial_cpu - clean_cpu).abs().flatten(1).amax(dim=1).numpy()
    ssim_values = []
    for original, attacked in zip(clean_cpu, adversarial_cpu):
        original_np = original.permute(1, 2, 0).numpy()
        attacked_np = attacked.permute(1, 2, 0).numpy()
        ssim_values.append(
            structural_similarity(
                original_np, attacked_np, data_range=1.0, channel_axis=-1
            )
        )
    return (
        linf.astype(np.float32),
        np.asarray(ssim_values, dtype=np.float32),
        lpips_values,
    )
