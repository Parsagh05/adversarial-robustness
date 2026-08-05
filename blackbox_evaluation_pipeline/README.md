# Fixed-perturbation evaluation pipeline

This pipeline compares anomaly-detection models using perturbations that have
already been generated. It never regenerates attacks or creates a random data
split. The canonical JSON manifest is the authority for:

- the exact evaluation image IDs;
- the subset of evaluation images that receives each perturbation;
- source/target datasets, attack direction, objective, and scope;
- perturbation paths, image size, epsilon, and file checksums.

The model adapter receives clean/adversarial RGB tensors in `[0, 1]` and owns
all model-specific preprocessing. This keeps the clean/adversarial construction
identical across models.

## Current coverage

The supplied archive currently contains 12 dataset-universal attacks:

| Source | Target | Directions | Objectives | Count |
|---|---|---|---|---:|
| MVTec AD | VisA | normal→abnormal, abnormal→normal | global, local, combined | 6 |
| VisA | MVTec AD | normal→abnormal, abnormal→normal | global, local, combined | 6 |

Standalone MVTec→MVTec and VisA→VisA records are not yet in this archive.
When dataset-universal records for those protocols are appended to the same
canonical manifest and their tensors are added to the archive, the runner will
discover and evaluate them automatically.

The current runner deliberately accepts `scope="dataset"` tensors only. It
fails explicitly if a future manifest introduces per-image or per-category
payload formats; support for those formats should be added as separate artifact
providers rather than inferred silently.

## Canonical artifacts

Exact Google Drive source:

<https://drive.google.com/file/d/10ZiaDs6u5G_WFVbrd9tt-Ug6Dy_aaFS5/view?usp=sharing>

After extraction, the expected local path is:

```text
blackbox_evaluation_pipeline/perturbations/canonical_clip_universal_attacks_full/
```

The archive is intentionally ignored by Git because it contains binary `.pt`
files. `kaggle_new_anomalyclip.ipynb` first searches attached Kaggle datasets
for the extracted directory and otherwise downloads this exact Drive file.

For every record the runner verifies the serialized `.pt` SHA-256 checksum,
tensor shape, finite values, and epsilon bound. It also checks that every fixed
manifest ID exists in the mounted target dataset and that the attacked IDs are
exactly the source-label subset of the evaluation cohort.

## AnomalyCLIP Kaggle notebook

Open [`kaggle_new_anomalyclip.ipynb`](kaggle_new_anomalyclip.ipynb) in Kaggle,
enable a GPU and Internet, confirm the MVTec/VisA mount paths, and run all cells.
The notebook independently:

1. clones this experiment repository and the official AnomalyCLIP repository;
2. installs `blackbox_evaluation_pipeline/requirements.txt`;
3. locates or downloads the exact canonical archive;
4. validates the manifest and dataset IDs;
5. loads the appropriate AnomalyCLIP checkpoint for each target dataset;
6. evaluates all currently available conditions and packages the results.

`FULL_RUN=True` evaluates all manifest records. Setting it to `False` evaluates
one complete condition as an integration check; it does not subsample the fixed
evaluation cohort.

## AnomalyCLIP decision thresholds

An audit of the official paper and repository found no published image-level
normal/abnormal decision threshold. See
[`OFFICIAL_THRESHOLD_REVIEW.md`](OFFICIAL_THRESHOLD_REVIEW.md) for the evidence.
The official clean benchmark reports continuous AUROC, AP, pixel AUROC, and
AUPRO, so there is no official MVTec or VisA numeric threshold to export.

For the team's secondary classification-flip and targeted-success metrics, run
[`kaggle_new_anomalyclip_thresholds.ipynb`](kaggle_new_anomalyclip_thresholds.ipynb).
It independently clones the same repositories, loads the same target-specific
checkpoints, and computes a custom per-dataset, per-category q95 threshold from
normal training images only. It never uses test images, labeled anomalies, or
adversarial images.

The notebook generates and packages:

```text
/kaggle/working/anomalyclip_thresholds_q95/
├── mvtec/{category_thresholds.json,normal_train_scores.npz,threshold_config.json}
└── visa/{category_thresholds.json,normal_train_scores.npz,threshold_config.json}
```

These thresholds are a benchmark operating-point policy, not an official
AnomalyCLIP result. Freeze the generated values and use the identical threshold
for clean and adversarial scores. Recalibrate if the model checkpoint,
preprocessing, image size, or anomaly-score implementation changes.

The committed q95 artifacts under `attack_generation_pipeline/thresholds/`
have been checked against their saved normal-training scores. The AnomalyCLIP
evaluation notebook loads the frozen `mvtec` or `visa` artifact according to
the target dataset; it never recalibrates during evaluation.

## Outputs

Each model run produces:

- `summary.csv`: macro-average results, one row per attack condition;
- `category_metrics.csv`: per-category results;
- `per_image.csv`: exact IDs, attacked flags, scores, clean/adversarial binary
  predictions, flip/success flags, directional shifts, and realized L-infinity
  distances;
- `predictions/*.npz`: auditable clean/adversarial scores and low-resolution
  maps;
- `run_config.json` and `manifest_snapshot.json`: reproducibility metadata.

The AnomalyCLIP notebook also creates a separate qualitative-samples archive;
it is not placed inside the numerical-results ZIP. For each attack condition,
eligible attacked images are ranked by their adversarial target margin:

- `strongest_success`: successful sample farthest inside the target class;
- `median_success`: middle successful sample after sorting by target margin;
- `worst_failure`: eligible failure farthest from reaching the target class.

An image is eligible only when its clean prediction is the intended source
class. This avoids selecting a pre-existing clean error as an attack success.
Selections that do not exist for a condition are listed in
`selection_manifest.json` rather than being fabricated.

Each selected folder contains:

```text
sample_id/
├── clean.png
├── adversarial.png
├── difference_x10.png
├── clean_heatmap.png
├── adversarial_heatmap.png
├── clean_overlay.png
├── adversarial_overlay.png
├── ground_truth_mask.png
├── heatmap_difference.png
└── metrics.json
```

The two heatmaps use a shared scale within the sample, making their colors
directly comparable. `heatmap_difference.png` uses red for increased anomaly
response and blue for decreased response. `metrics.json` includes identifiers,
scores, thresholds, binary decisions, target margins, flip/success flags,
perturbation norms, PSNR, SSIM, and heatmap-change statistics.

Continuous performance metrics are reported on a `0–100` scale:

- image AUROC (`i_auroc`);
- image average precision (`i_ap`);
- pixel AUROC (`p_auroc`);
- AUPRO integrated to FPR 0.30 (`aupro`).

For each metric, `delta = clean - adversarial`, so a positive delta means the
attack degraded the detector. The evaluator also reports directional image
score shift, directional anomaly-map shift, directional map-pixel fraction,
and realized L-infinity.

When `thresholds_by_target` is configured, threshold-based metrics are reported
on a `0–100` scale using `anomaly = score >= category threshold`:

- clean and adversarial classification accuracy;
- clean and adversarial false-positive rate (normal images predicted abnormal);
- clean and adversarial false-negative rate (abnormal images predicted normal);
- attack flip rate among images that actually receive the perturbation;
- targeted attack success rate among attacked images whose clean prediction is
  the attack's source class and whose adversarial prediction flips to the
  opposite target class.

Restricting targeted success to that eligible clean-source subset prevents an
image already predicted as the target class before the attack from counting as
a success. Reaching the target without a clean-to-adversarial classification
change cannot count as success. `targeted_success_eligible_count` is stored next
to the rate.
Category metrics are computed directly; `summary.csv` reports macro-averages.

## Adding another model

Keep the shared protocol code unchanged:

1. Add `universal_eval/adapters/<model>.py` implementing `ModelAdapter`.
2. Register it with `@register_adapter("<model>")` and import the module in
   `universal_eval/adapters/__init__.py`.
3. Put all resizing, normalization, prompting, and checkpoint logic inside the
   adapter's `predict()` method.
4. Create `kaggle_new_<model>.ipynb`. It should clone the model's official
   repository, configure per-target checkpoints, and call the same
   `EvaluationConfig`/`run_evaluation` API.

The adapter contract is intentionally small:

```python
scores, low_resolution_maps = adapter.predict(images_01)
```

`images_01` has shape `[B, 3, H, W]`. `scores` must have shape `[B]`, and maps
must have shape `[B, h, w]`. The shared runner handles artifact validation,
fixed sample selection, perturbation application, metrics, and output files.

## Local CLI

The notebook uses the Python API directly. A JSON-configured CLI is also
available:

```bash
python -m blackbox_evaluation_pipeline.evaluate --config /path/to/config.json
```

The JSON keys are the fields of `EvaluationConfig` in
`universal_eval/runner.py`, including `model_kwargs_by_target` for target-
dataset-specific checkpoints and `thresholds_by_target` for frozen per-target
threshold JSON files. If thresholds are omitted, continuous metrics still run
and threshold-dependent columns are omitted.

For local validation:

```bash
pip install -r blackbox_evaluation_pipeline/requirements-dev.txt
python -m pytest
```

## Layout

```text
blackbox_evaluation_pipeline/
├── kaggle_new_anomalyclip.ipynb
├── kaggle_new_anomalyclip_thresholds.ipynb
├── calculate_dataset_perturbations.ipynb
├── OFFICIAL_THRESHOLD_REVIEW.md
├── requirements.txt
├── evaluate.py
├── universal_eval/
│   ├── adapters/
│   ├── artifacts.py
│   ├── datasets.py
│   ├── metrics.py
│   └── runner.py
└── perturbations/                 # local extracted archive; Git-ignored
```
