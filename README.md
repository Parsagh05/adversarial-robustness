# Adversarial Robustness of AnomalyCLIP

This project evaluates whether adversarial perturbations optimized on a public CLIP surrogate transfer to **AnomalyCLIP** for zero-shot anomaly detection on **MVTec AD**. It measures both image-level detection and pixel-level localization under two targeted attack directions, three loss objectives, and three perturbation scopes.

The attack is black-box with respect to the anomaly detector: gradients come only from the public CLIP surrogate. AnomalyCLIP is queried after the adversarial image is generated and never contributes scores, maps, weights, or gradients to attack optimization.

## Pipeline

```text
MVTec image
    |
    +--> AnomalyCLIP on clean image -----------------------+
    |                                                      |
    +--> public CLIP surrogate + prompt ensemble           |
             |                                             |
             +--> select perturbation scope                |
             |       |                                     |
             |       +--> per_image: one delta per image   |
             |       +--> per_category: one delta per      |
             |       |                  category           |
             |       +--> dataset: one delta shared across |
             |                          all categories     |
             |                                             |
             +--> targeted PGD at 518 x 518                |
                        |                                  |
                        +--> apply the optimized delta      |
                                   |                       |
                                   +--> AnomalyCLIP --------+
                                               |
                                               +--> clean/adversarial scores and maps
                                               +--> detection, localization, attack, and perceptual metrics
```

The surrogate uses WinCLIP-style prompt ensembles and keeps every prompt embedding separate. The attack can optimize:

- `global`: the projected image/CLS feature, mainly targeting image-level detection.
- `local`: spatial patch features from layers 6, 12, 18, and 24, mainly targeting localization.
- `combined`: `0.2 * global_loss + 0.8 * local_loss` by default. The local term is weighted more heavily to compensate for the larger global-gradient magnitude observed with equal weights.

The perturbation scope controls how widely the optimized `delta` is shared:

- `per_image` optimizes a new perturbation for each attacked image. It is the most sample-specific scope and acts as the transfer-attack upper bound.
- `per_category` optimizes one universal perturbation from source-label images in a single MVTec category, then applies that same perturbation to the category's evaluation images.
- `dataset` optimizes one universal perturbation across source-label images from all selected categories. During optimization, each image still uses the prompt bank for its own category, but the final perturbation is category-agnostic and shared across the dataset.

In all three scopes, PGD uses only surrogate gradients. The resulting image is then evaluated by AnomalyCLIP through the same target-only inference path.

The two attack directions are reported separately:

- `normal_to_abnormal`: attacks normal images to create false alarms.
- `abnormal_to_normal`: attacks anomalous images to hide defects.

Only the source class is perturbed in each direction. Opposite-class images remain clean so AUROC, AP, and localization metrics are still computed over a population containing both labels.

## Attack scopes

| Scope | Perturbation |
|---|---|
| `per_image` | A separate perturbation is optimized for each source image. |
| `per_category` | One universal perturbation is optimized and evaluated within each MVTec category. |
| `dataset` | One category-agnostic universal perturbation is optimized across all selected categories. |

Universal attacks support two protocols:

- `held_out` with `use_split_manifest=True` is the notebook default. A saved manifest matches normal/anomalous counts within every category and assigns each label exactly 50/50 to fit and evaluation. Normal-to-abnormal fits only on manifest normal-fit rows; abnormal-to-normal fits only on anomaly-fit rows. Both directions use the same balanced held-out evaluation cohort.
- `held_out` without a manifest retains the legacy asymmetric behavior for backward compatibility. It should not be used for a direction-fair comparison.
- `transductive` fits and evaluates on the same selected source population. Results from this protocol must be labeled as transductive and should not be compared directly with held-out results.

The saved 50/50 fit/evaluation assignment is used differently by each scope:

| Scope | Optimization samples | Reported evaluation samples |
|---|---|---|
| `per_image` | Every selected source image is optimized independently. There is no reusable universal fit set. | The combined manifest-selected fit and evaluation rows. |
| `per_category` | The source-label `fit` rows from that category. | The shared normal/anomaly `evaluation` rows from that category. |
| `dataset` | The source-label `fit` rows pooled across all selected categories. | The same shared normal/anomaly `evaluation` rows pooled across all selected categories. |

Therefore, the held-out 50/50 generalization protocol applies directly to the two universal scopes, `per_category` and `dataset`; it is not a fit/evaluation protocol for `per_image`. The per-image result is inherently image-specific because its perturbation is optimized on the same image it attacks. It should be compared with the universal results with that distinction stated explicitly. Both universal directions still use identical per-category fit counts and the identical held-out evaluation cohort.

## Main settings

The full Kaggle experiment uses the following settings:

| Setting | Value |
|---|---|
| Dataset | Saved category- and label-balanced subset of the MVTec AD official test split; `train/good` is calibration-only |
| Target | AnomalyCLIP with the official VisA-trained `9_12_4_multiscale/epoch_15.pth` checkpoint |
| Surrogate | Public CLIP `ViT-L/14@336px` loaded through the AnomalyCLIP CLIP implementation |
| Attack/evaluation resolution | `518 x 518` |
| Perturbation bound | `epsilon = 8/255` in the unnormalized `[0, 1]` image domain |
| PGD step size | `2/255` |
| Per-image PGD steps | `20` |
| Universal update steps | `200` |
| Random start | Enabled |
| Prompt temperature | `0.07` |
| Feature layers | `6, 12, 18, 24` |
| Global/local weights | `0.5 / 0.5` |
| Per-image / universal batch size | `1 / 2` |
| Target batch size | `2` |
| Universal protocol | `held_out` |
| Matched held-out fit fraction for both labels | `0.5` |
| Attack and split seed | `111` |
| Threshold calibration | Per-category 95th percentile of AnomalyCLIP scores on `train/good` |
| Anomaly-map processing | Resize to `518 x 518`, then Gaussian smoothing with sigma `4.0` |
| AUPRO | FPR limit `0.30`, at most `200` thresholds |

The complete run evaluates `2 directions x 3 loss modes x 3 scopes = 18` conditions across all MVTec categories. A smoke run is also provided: category `bottle`, direction `abnormal_to_normal`, loss `combined`, scope `per_image`, two PGD steps, at most four samples, and LPIPS disabled.

Perturbations are applied before normalization:

```text
||delta||_infinity <= epsilon
x_adv = clamp(x + delta, 0, 1)
```

CLIP normalization happens inside the differentiable surrogate. The target receives the perturbed 518-pixel tensor directly, so the perturbation is not weakened by later upscaling.

## Metrics

All detection, localization, and success-rate metrics are reported on a `0-100` scale. `L-infinity`, SSIM, LPIPS, raw scores, and decision thresholds are not percentages.

| Metric | Meaning | Preferred direction |
|---|---|---|
| I-AUROC | Image-level area under the ROC curve using continuous anomaly scores. | Higher detector performance |
| Image AP | Image-level average precision; useful when normal/anomalous counts are imbalanced. | Higher detector performance |
| P-AUROC | Pixel AUROC after flattening all predicted map pixels and ground-truth mask pixels in a category. | Higher detector performance |
| AUPRO | Region-overlap score integrated up to FPR `0.30`; weights connected defect regions rather than only individual pixels. | Higher detector performance |
| Classification flip rate | Among attacked source images classified correctly when clean, the percentage moved across the frozen threshold into the targeted wrong class. | Higher attack effectiveness |
| Targeted success rate (all) | Percentage of all attacked source images predicted as the target class, including images already wrong when clean. | Higher attack effectiveness |
| Directional score shift | Anomaly-score movement toward the intended target class; it does not require a threshold crossing. | Positive means intended movement |
| L-infinity | Maximum absolute pixel change between clean and adversarial images in `[0, 1]`. | Lower distortion |
| SSIM | Structural similarity between clean and adversarial images. | Closer to `1` means more similar |
| LPIPS | Learned perceptual distance between clean and adversarial images. | Lower means more similar |

For the threshold-dependent success metrics, category `c` uses a frozen threshold `t_c`:

```text
prediction = 1[anomaly_score >= t_c]
```

Each `t_c` is the configured quantile—`0.95` by default—of AnomalyCLIP scores on that category's MVTec `train/good` images. Labeled test anomalies are never used to select thresholds. I-AUROC, image AP, P-AUROC, and AUPRO use continuous scores and do not depend on this threshold.

For every detector metric, the output contains clean, adversarial, and delta values:

```text
delta_metric = clean_metric - adversarial_metric
```

A positive delta therefore means that the attack degraded detector performance. Macro rows are unweighted means across categories; invalid category values such as a one-class AUROC are excluded rather than pooled globally.

## Kaggle notebooks

The notebooks are orchestration-only. They clone/update the public [`Parsagh05/adversarial-robustness`](https://github.com/Parsagh05/adversarial-robustness) repository and the official AnomalyCLIP repository, install `requirements.txt`, resolve Kaggle paths, call the shared harness, and package the output. Attack and metric code is not duplicated in notebook cells. No GitHub token or Kaggle secret is required to clone either repository.

### `kaggle_generate_mvtec_matched_splits.ipynb`

Run this notebook once before the adversarial benchmark.

- Reads the official MVTec test split and never loads a model.
- Matches each category to its smaller normal/anomaly count, drops one from each label when needed to make the count even, and creates exact 50/50 fit/evaluation assignments.
- Writes `splits/mvtec_matched_test_per_category_v1_seed111.csv` and its validated JSON metadata.
- Keeps excluded majority-label rows in the CSV as `excluded_balance` for auditing.
- Package and upload the `splits/` output as a Kaggle Dataset so every experiment uses the identical sample IDs.

### `kaggle_calibrate_anomalyclip_thresholds.ipynb`

Run this notebook once before the main benchmark when you want reusable decision thresholds.

- Reads only MVTec `train/good`; it does not use test images or labeled anomalies.
- Uses image size `518`, AnomalyCLIP, and threshold quantile `0.95`.
- Saves `category_thresholds.json`, `normal_train_scores.npz`, and `anomalyclip_thresholds_q95.zip` under `/kaggle/working`.
- Prints the threshold and score statistics for every category.

The threshold artifact is valid only for the same AnomalyCLIP checkpoint, target configuration, preprocessing/image size, categories, and quantile. The harness validates this metadata and rejects incompatible artifacts.

### `kaggle_adversarial_anomalyclip.ipynb`

This notebook runs the actual adversarial benchmark.

- The public pipeline repository is cloned to `/kaggle/working/adversarial-robustness`; no GitHub token is needed.
- Mount MVTec at `/kaggle/input/datasets/alirezasalehy/mvtec-ad/mvtec_anomaly_detection` or update `MVTEC_ROOT`.
- Set `THRESHOLDS_PATH` to the mounted `category_thresholds.json`. Leave it as `None` to calibrate automatically inside the benchmark output directory.
- Keep `USE_SPLIT_MANIFEST = True` and set `SPLIT_MANIFEST_ROOT` to the mounted output of `kaggle_generate_mvtec_matched_splits.ipynb`.
- Use `FULL_RUN = False` for the small end-to-end smoke test and `FULL_RUN = True` for all 18 conditions.
- The full run writes to `/kaggle/working/anomalyclip_adversarial_held_out_full`; the smoke run uses a separate output directory.
- The final cell packages the complete output directory as a ZIP.

Both notebooks expect a Kaggle GPU and use the official AnomalyCLIP checkpoint at `checkpoints/9_12_4_multiscale/epoch_15.pth`. The checkpoint is trained on VisA and is used for zero-shot evaluation on MVTec.

## How to run on Kaggle

1. Add the MVTec AD dataset and run `kaggle_generate_mvtec_matched_splits.ipynb` once.
2. Upload its `splits/` output as a Kaggle Dataset.
3. Run `kaggle_calibrate_anomalyclip_thresholds.ipynb` once to generate `category_thresholds.json`.
4. Mount both artifacts in the adversarial notebook and set `SPLIT_MANIFEST_ROOT` and `THRESHOLDS_PATH`.
5. Set `FULL_RUN = False` for a smoke test or `True` for the complete benchmark.
6. Run all cells in `kaggle_adversarial_anomalyclip.ipynb`; the final cell creates the results ZIP.

With a manifest, `max_samples_per_category` is a strict upper bound. The loader selects equal counts from normal-fit, anomaly-fit, normal-evaluation, and anomaly-evaluation. Consequently, values that are not divisible by four are rounded down to the largest valid balanced subset; values below four are rejected.

## Output artifacts

```text
output_root/
|-- config.json
|-- completed_conditions.json
|-- category_thresholds.json
|-- normal_train_scores.npz
|-- clean_predictions.npz
|-- split_manifest/
|   |-- mvtec_matched_test_per_category_v1_seed111.csv
|   `-- mvtec_matched_test_per_category_v1_seed111.json
|-- clean_metrics/
|-- summary.csv
|-- per_image.csv
|-- predictions/<condition>/target_outputs.npz
|-- perturbations/<condition>/*.pt
|-- diagnostics/<condition>/
|   |-- data_split.json
|   |-- optimization.json
|   |-- loss_curve.csv
|   `-- surrogate_predictions.csv
|-- adversarial_examples/<condition>/
`-- partial/<condition>/
```

- `summary.csv` is the main result table. It contains one row per category plus a `__macro__` row for each condition.
- `per_image.csv` contains attacked source images and their clean/adversarial scores, predictions, success flags, and distortion metrics.
- `target_outputs.npz` stores the evaluated sample IDs, labels, clean/adversarial scores, and low-resolution anomaly maps for later metric auditing.
- `optimization.json`, `loss_curve.csv`, and `surrogate_predictions.csv` diagnose whether optimization succeeded on the surrogate. They are not AnomalyCLIP target results.
- Universal perturbations and representative adversarial examples are saved when enabled.

Runs are resumable. Completed conditions are skipped, while active-condition partial predictions are refreshed periodically. Resume is rejected if the requested configuration differs from the existing `config.json`, preventing results from incompatible settings from being mixed.

## Project structure

```text
.
|-- README.md
|-- requirements.txt
|-- run.py
|-- kaggle_generate_mvtec_matched_splits.ipynb
|-- kaggle_calibrate_anomalyclip_thresholds.ipynb
|-- kaggle_adversarial_anomalyclip.ipynb
`-- adversarial_harness/
    |-- config.py
    |-- dataset.py
    |-- split_manifest.py
    |-- prompts.py
    |-- models.py
    |-- attacks.py
    |-- metrics.py
    `-- runner.py
```
