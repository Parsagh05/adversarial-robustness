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
new_pipeline/perturbations/canonical_clip_universal_attacks_full/
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
2. installs `new_pipeline/requirements.txt`;
3. locates or downloads the exact canonical archive;
4. validates the manifest and dataset IDs;
5. loads the appropriate AnomalyCLIP checkpoint for each target dataset;
6. evaluates all currently available conditions and packages the results.

`FULL_RUN=True` evaluates all manifest records. Setting it to `False` evaluates
one complete condition as an integration check; it does not subsample the fixed
evaluation cohort.

## Outputs

Each model run produces:

- `summary.csv`: macro-average results, one row per attack condition;
- `category_metrics.csv`: per-category results;
- `per_image.csv`: exact IDs, attacked flags, scores, directional shifts, and
  realized L-infinity distances;
- `predictions/*.npz`: auditable clean/adversarial scores and low-resolution
  maps;
- `run_config.json` and `manifest_snapshot.json`: reproducibility metadata.

Continuous performance metrics are reported on a `0–100` scale:

- image AUROC (`i_auroc`);
- image average precision (`i_ap`);
- pixel AUROC (`p_auroc`);
- AUPRO integrated to FPR 0.30 (`aupro`).

For each metric, `delta = clean - adversarial`, so a positive delta means the
attack degraded the detector. The evaluator also reports directional image
score shift, directional anomaly-map shift, directional map-pixel fraction,
and realized L-infinity. Threshold-dependent classification flips are not
reported because the canonical archive does not provide a fair, frozen
model-specific calibration artifact.

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
python -m new_pipeline.evaluate --config /path/to/config.json
```

The JSON keys are the fields of `EvaluationConfig` in
`universal_eval/runner.py`, including `model_kwargs_by_target` for target-
dataset-specific checkpoints.

For local validation:

```bash
pip install -r new_pipeline/requirements-dev.txt
python -m pytest
```

## Layout

```text
new_pipeline/
├── kaggle_new_anomalyclip.ipynb
├── calculate_dataset_perturbations.ipynb
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
