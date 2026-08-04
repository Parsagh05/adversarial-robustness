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


def _binary_average_precision(labels: Sequence[int], scores: Sequence[float]) -> float:
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


def per_image_classification_metrics(
    clean_score: float,
    adversarial_score: float,
    decision_threshold: float,
    source_label: int,
    target_label: int,
) -> Dict[str, float]:
    """Return explicit per-image classification outcomes at a frozen threshold."""

    if source_label not in (0, 1) or target_label not in (0, 1):
        raise ValueError("source_label and target_label must be 0 or 1")
    if source_label == target_label:
        raise ValueError("source_label and target_label must differ")
    values = np.asarray(
        [clean_score, adversarial_score, decision_threshold], dtype=np.float64
    )
    if not np.isfinite(values).all():
        raise ValueError("Classification scores and threshold must be finite")

    clean_prediction = int(clean_score >= decision_threshold)
    adversarial_prediction = int(adversarial_score >= decision_threshold)
    clean_correct = clean_prediction == source_label
    image_targeted_success = adversarial_prediction == target_label
    score_shift = float(adversarial_score - clean_score)
    directional_score_shift = score_shift if target_label == 1 else -score_shift
    return {
        "clean_prediction": clean_prediction,
        "adversarial_prediction": adversarial_prediction,
        "clean_correct_for_source": int(clean_correct),
        "image_targeted_success": int(image_targeted_success),
        "classification_flip": int(clean_correct and image_targeted_success),
        "score_shift": score_shift,
        "directional_score_shift": directional_score_shift,
        "score_directional_success": int(directional_score_shift > 0.0),
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


def _single_image_aupro(
    mask: np.ndarray,
    anomaly_map: np.ndarray,
    fpr_limit: float,
) -> float:
    """Return exact per-image AUPRO without materializing threshold masks.

    Each foreground pixel contributes its region-normalized share of PRO while
    each background pixel contributes its share of FPR. Sorting once by score
    therefore traces the same threshold curve much more cheaply than repeatedly
    thresholding a full-resolution map.
    """

    binary_mask = np.asarray(mask, dtype=bool)
    scores = np.asarray(anomaly_map, dtype=np.float64)
    component_map = connected_components(binary_mask, connectivity=2)
    region_count = int(component_map.max())
    negative_count = int((~binary_mask).sum())
    if negative_count == 0 or region_count == 0:
        return float("nan")

    pro_weights = np.zeros(binary_mask.shape, dtype=np.float64)
    for region_index in range(1, region_count + 1):
        region = component_map == region_index
        pro_weights[region] = 1.0 / (region_count * int(region.sum()))

    flat_scores = scores.reshape(-1)
    order = np.argsort(flat_scores, kind="mergesort")[::-1]
    sorted_scores = flat_scores[order]
    sorted_background = (~binary_mask).reshape(-1)[order]
    sorted_pro_weights = pro_weights.reshape(-1)[order]
    distinct_indices = np.r_[
        np.where(np.diff(sorted_scores))[0], sorted_scores.size - 1
    ]
    fprs = np.cumsum(sorted_background, dtype=np.float64)[distinct_indices]
    fprs /= negative_count
    pros = np.cumsum(sorted_pro_weights, dtype=np.float64)[distinct_indices]
    fprs = np.r_[0.0, fprs]
    pros = np.r_[0.0, pros]

    # FPR is already non-decreasing and PRO is non-decreasing within an FPR
    # plateau. Keep the last point of each plateau in linear time.
    plateau_ends = np.r_[np.where(np.diff(fprs) > 0.0)[0], fprs.size - 1]
    unique_fprs = fprs[plateau_ends]
    unique_pros = pros[plateau_ends]
    boundary_pro = float(np.interp(fpr_limit, unique_fprs, unique_pros))
    keep = unique_fprs < fpr_limit
    x = np.concatenate([unique_fprs[keep], [fpr_limit]])
    y = np.concatenate([unique_pros[keep], [boundary_pro]])
    trapezoid = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    return 100.0 * float(trapezoid(y, x) / fpr_limit)


def per_image_map_metrics(
    mask: np.ndarray,
    clean_map: np.ndarray,
    adversarial_map: np.ndarray,
    target_label: int,
    *,
    aupro_fpr_limit: float = 0.30,
    map_success_min_mean_shift: Optional[float] = None,
    map_success_min_pixel_fraction: float = 0.5,
    map_false_positive_threshold: float = 2.0,
    localization_success_min_p_ap_drop: float = 0.0,
) -> Dict[str, float]:
    """Measure target-direction map change and per-image localization damage.

    Map-direction metrics are defined for normal and anomalous source images.
    Mask-dependent localization metrics are NaN for normal images because an
    all-zero mask has no positive region to localize.
    """

    clean = np.asarray(clean_map, dtype=np.float64)
    adversarial = np.asarray(adversarial_map, dtype=np.float64)
    binary_mask = np.asarray(mask) > 0
    if clean.shape != adversarial.shape or clean.shape != binary_mask.shape:
        raise ValueError("Mask and clean/adversarial maps must have matching shapes")
    if clean.ndim != 2:
        raise ValueError("Per-image map metrics require two-dimensional maps")
    if not np.isfinite(clean).all() or not np.isfinite(adversarial).all():
        raise ValueError("Per-image anomaly maps must be finite")
    if target_label not in (0, 1):
        raise ValueError("target_label must be 0 or 1")
    if not 0.0 <= map_success_min_pixel_fraction <= 1.0:
        raise ValueError("map_success_min_pixel_fraction must be in [0, 1]")
    if not np.isfinite(map_false_positive_threshold):
        raise ValueError("map_false_positive_threshold must be finite")

    direction = 1.0 if target_label == 1 else -1.0
    directional_delta = direction * (adversarial - clean)
    directional_mean_shift = float(directional_delta.mean())
    directional_pixel_fraction = float((directional_delta > 0.0).mean())
    result = {
        "map_directional_mean_shift": directional_mean_shift,
        "map_directional_pixel_fraction": directional_pixel_fraction,
        "map_absolute_shift": float(np.abs(adversarial - clean).mean()),
        "map_directional_success": (
            int(
                directional_mean_shift > map_success_min_mean_shift
                and directional_pixel_fraction > map_success_min_pixel_fraction
            )
            if map_success_min_mean_shift is not None
            else float("nan")
        ),
        "defect_directional_mean_shift": float("nan"),
        "background_directional_mean_shift": (
            float(directional_delta[~binary_mask].mean())
            if (~binary_mask).any()
            else float("nan")
        ),
        "clean_false_positive_map_area": float("nan"),
        "adversarial_false_positive_map_area": float("nan"),
        "false_positive_map_area_increase": float("nan"),
        "clean_image_p_auroc": float("nan"),
        "adversarial_image_p_auroc": float("nan"),
        "image_p_auroc_drop": float("nan"),
        "clean_image_p_ap": float("nan"),
        "adversarial_image_p_ap": float("nan"),
        "image_p_ap_drop": float("nan"),
        "clean_image_aupro": float("nan"),
        "adversarial_image_aupro": float("nan"),
        "image_aupro_drop": float("nan"),
        "clean_localization_contrast": float("nan"),
        "adversarial_localization_contrast": float("nan"),
        "localization_contrast_drop": float("nan"),
        "localization_degradation_success": float("nan"),
    }

    if not binary_mask.any():
        clean_false_positive_area = float(
            (clean > map_false_positive_threshold).mean()
        )
        adversarial_false_positive_area = float(
            (adversarial > map_false_positive_threshold).mean()
        )
        result.update(
            {
                "clean_false_positive_map_area": clean_false_positive_area,
                "adversarial_false_positive_map_area": (
                    adversarial_false_positive_area
                ),
                "false_positive_map_area_increase": (
                    adversarial_false_positive_area - clean_false_positive_area
                ),
            }
        )
        return result
    if binary_mask.all():
        return result

    result["defect_directional_mean_shift"] = float(
        directional_delta[binary_mask].mean()
    )
    flat_mask = binary_mask.reshape(-1).astype(np.uint8)
    clean_flat = clean.reshape(-1)
    adversarial_flat = adversarial.reshape(-1)
    clean_p_auroc = 100.0 * _binary_auroc(flat_mask, clean_flat)
    adversarial_p_auroc = 100.0 * _binary_auroc(flat_mask, adversarial_flat)
    clean_p_ap = 100.0 * _binary_average_precision(flat_mask, clean_flat)
    adversarial_p_ap = 100.0 * _binary_average_precision(flat_mask, adversarial_flat)
    clean_aupro = _single_image_aupro(binary_mask, clean, aupro_fpr_limit)
    adversarial_aupro = _single_image_aupro(binary_mask, adversarial, aupro_fpr_limit)
    clean_contrast = float(clean[binary_mask].mean() - clean[~binary_mask].mean())
    adversarial_contrast = float(
        adversarial[binary_mask].mean() - adversarial[~binary_mask].mean()
    )
    p_ap_drop = clean_p_ap - adversarial_p_ap
    result.update(
        {
            "clean_image_p_auroc": clean_p_auroc,
            "adversarial_image_p_auroc": adversarial_p_auroc,
            "image_p_auroc_drop": clean_p_auroc - adversarial_p_auroc,
            "clean_image_p_ap": clean_p_ap,
            "adversarial_image_p_ap": adversarial_p_ap,
            "image_p_ap_drop": p_ap_drop,
            "clean_image_aupro": clean_aupro,
            "adversarial_image_aupro": adversarial_aupro,
            "image_aupro_drop": clean_aupro - adversarial_aupro,
            "clean_localization_contrast": clean_contrast,
            "adversarial_localization_contrast": adversarial_contrast,
            "localization_contrast_drop": clean_contrast - adversarial_contrast,
            "localization_degradation_success": int(
                p_ap_drop > localization_success_min_p_ap_drop
            ),
        }
    )
    return result


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
        except (
            Exception
        ) as exc:  # dependency/download failures remain explicit in output
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
