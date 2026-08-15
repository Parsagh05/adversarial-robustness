# Per-dataset attack ablation pipeline

This folder is an independent end-to-end handoff for generating universal
per-dataset perturbations and evaluating them on AnomalyCLIP. It supports both
MVTec AD and VisA and does not run per-category or per-image attacks.

Set `DATASETS=mvtec`, `DATASETS=visa`, or `DATASETS=mvtec,visa`. Single-dataset
runs require only the selected dataset to be mounted. The two-dataset option
also evaluates cross-dataset transfer and is therefore substantially longer.

## Experiment matrix

Four perturbation sets are generated:

| Setup | PGD steps | Linf epsilon |
|---|---:|---:|
| `steps500_eps2` | 500 | 2/255 |
| `steps500_eps4` | 500 | 4/255 |
| `steps800_eps2` | 800 | 2/255 |
| `steps800_eps4` | 800 | 4/255 |

Every perturbation set contains both attack directions and the global, local,
and combined losses. The initial step size is 0.25/255 with cosine decay to
10%, and the full balanced attack-training split is used for optimization and
checkpoint selection.

Each perturbation set uses one shared AnomalyCLIP inference pass to produce
three threshold-specific reports and visualization folders:

1. `fixed_0_5`: use 0.5 as the anomaly-map pixel decision threshold.
2. `image_f1`: apply the automatically calibrated clean image-level F1-max
   threshold to pixels. This is included only as the requested ablation.
3. `clean_pixel_f1`: calculate a separate clean pixel-level F1-max threshold
   from anomaly maps and masks, then freeze it for adversarial evaluation.

Image-level F1-max thresholds are always calculated automatically for both
datasets. They are frozen before adversarial evaluation.

## Pixel success definition

Pixel success is separate from image-level targeted success. For
normal-to-abnormal, only the fixed central target region is evaluated. For
abnormal-to-normal, only the ground-truth defect mask is evaluated. A pixel is
eligible only if its clean prediction is the source class. An image is a pixel
success when at least 50% of eligible target-region pixels flip to the target
class. This fraction is configurable with
`PIXEL_SUCCESS_MIN_FLIP_FRACTION`.

Visualizations are generated independently under every pixel-threshold folder.
They are selected by target-region pixel flip rate and are intended for
debugging, not as formal benchmark evidence.

## Run on a server

Edit the three paths in `config.sh`, then run:

```bash
bash train.sh
```

To run only one setup:

```bash
RUN_SETUPS=steps500_eps2 bash train.sh
```

To select one dataset:

```bash
DATASETS=mvtec RUN_SETUPS=steps500_eps2 bash train.sh
```

To generate and evaluate in separate jobs:

```bash
RUN_PHASE=generate RUN_SETUPS=steps500_eps2 bash train.sh
RUN_PHASE=evaluate RUN_SETUPS=steps500_eps2 bash train.sh
```

Completed artifacts and evaluations are reused unless
`OVERWRITE_EXISTING=true` is set.

Before any optimization, `preflight.py` validates the selected dataset roots,
the pinned AnomalyCLIP checkout and checkpoints, the required pipeline files,
and output-directory writability. Generated directory and ZIP bundles are also
audited against their manifests before evaluation begins.

Local CPU unit checks (they do not generate attacks) can be run with:

```bash
PYTHONPATH=generation python -m unittest discover -s generation/tests
PYTHONPATH=. python -m unittest discover -s tests
```

## Kaggle warning

The complete four-setup matrix is substantially longer than one Kaggle T4
session. Use the included notebook, but select one setup per saved Kaggle run.
`RUN_SETUPS=all` is intended for a server or a multi-session workflow. The
`SMOKE_TEST=true` option validates plumbing only and must never be reported as a
benchmark result.

The notebook exposes separate generation and evaluation batch sizes. Both are
set to 2 by default for a 16 GB Kaggle T4.

## Output layout

```text
outputs/
  datasets_mvtec/                 # or datasets_visa / datasets_mvtec_visa
    calibrated_thresholds/anomalyclip/{mvtec,visa}/
    setups/
      steps500_eps2/
        attack_generation/
        evaluation/
          fixed_0_5/{numerical,visualizations}/
          image_f1/{numerical,visualizations}/
          clean_pixel_f1/{numerical,visualizations}/
      steps500_eps4/
      steps800_eps2/
      steps800_eps4/
    ablation_high_level_summary.csv
```

The high-level CSV contains the four main threshold-free benchmark metrics
(I-AUROC, image AP, P-AUROC, and AUPRO) plus image-level and target-region
pixel-level success diagnostics.
