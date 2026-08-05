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

Consequently, there is no official numeric threshold to export for MVTec AD or
VisA. `kaggle_new_anomalyclip_thresholds.ipynb` creates a custom fallback for
the team's secondary threshold-dependent attack metrics. It uses the 95th
percentile of scores from normal training images separately for every target
dataset and category. The generated artifacts explicitly record that this is
not an official AnomalyCLIP threshold.

This calibration must not be described as part of AnomalyCLIP's official clean
benchmark. It is a leakage-free operating-point policy chosen for the
adversarial comparison. Official AUROC/AP/AUPRO results remain unchanged and do
not consume the generated thresholds.
