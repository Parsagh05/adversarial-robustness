# AnomalyCLIP decision-threshold review

Reviewed on 2026-08-05.

The official AnomalyCLIP implementation and paper do not publish a fixed
image-level normal/abnormal decision threshold:

- The official [`test.py`](https://github.com/zqhang/AnomalyCLIP/blob/main/test.py)
  collects continuous anomaly probabilities and reports image AUROC, image AP,
  pixel AUROC, and pixel AUPRO. It does not binarize image predictions.
- The official [`metrics.py`](https://github.com/zqhang/AnomalyCLIP/blob/main/metrics.py)
  implements those ranking/curve metrics and does not expose a reusable image
  decision threshold.
- The [AnomalyCLIP paper](https://arxiv.org/abs/2310.18961) defines the
  abnormal-class probability as the anomaly score and reports the same
  threshold-independent metrics, but does not specify an operating threshold.

Consequently, there is no official fixed numeric threshold to export for MVTec
AD or VisA. `kaggle_new_thresholds.ipynb` follows CRANE's F1-max benchmark
convention: it selects the threshold maximizing image F1 from the fixed clean
evaluation scores and labels, separately for every model, target dataset, and
category. The generated artifacts record that this is not an official model
threshold and that clean evaluation labels were used.

This calibration must not be described as an official AnomalyCLIP threshold or
as leakage-free deployment calibration. It is an oracle benchmark operating
point. The value is frozen before applying attacks and reused for accuracy,
flip rate, targeted success, FPR, and FNR. AUROC/AP/AUPRO remain independent of
the frozen threshold.
