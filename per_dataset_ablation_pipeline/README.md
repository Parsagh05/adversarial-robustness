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

Every perturbation set contains both attack directions crossed with the global,
local, and combined losses and with both loss formulations, so each source
dataset yields 12 perturbations per setup. The initial step size is 0.25/255
with cosine decay to 10%, and the full balanced attack-training split is used
for optimization and checkpoint selection.

## Loss formulations

`LOSS_FORMULATIONS` selects the objective families to ablate. Both run by
default, which doubles generation and evaluation time; pass a single value to
run one.

| | `ce_focal_dice` (default) | `margin_topk` |
|---|---|---|
| Image level | cross-entropy towards the target class | maximize or minimize `s(x+δ) = z_a(x+δ) − z_n(x+δ)` |
| Pixel level | mask-weighted focal + soft Dice over patch tokens | maximize or minimize `TopK(H(x+δ))` |
| Fake defect location | a fixed central square on normal images | unconstrained |
| Ground-truth masks | read for `local` and `combined` | never read |
| Extra knobs | region fraction, focal/Dice weights | `K` per direction |

`H` is the per-token margin map averaged over the captured surrogate layers, and
`K` is set per direction as a fraction of the patch tokens. At `IMAGE_SIZE=518`
the ViT-L/14 surrogate has a 518/14 = 37 grid per side, so 1369 patch tokens:
`MARGIN_TOPK_FRACTION_NORMAL_TO_ABNORMAL` is 0.20 (274 tokens) and
`MARGIN_TOPK_FRACTION_ABNORMAL_TO_NORMAL` is 0.40 (548 tokens). Planting a fake
defect only needs a small region to read as abnormal, while suppressing a real
one has to push down everything the map already ranks highly, so the
anomalous-source attack uses the larger `K`. Both formulations use the same
protocol, split, epsilon budget, and metrics, so their rows are directly
comparable. The formulation appears as a `loss_formulation` column in
`attack_manifest.csv`, in every result CSV, and in
`ablation_high_level_summary.csv`, with the `K` actually used recorded next to it
as `margin_topk_fraction`.

Because `margin_topk` never constrains where the fake abnormal region appears,
its `normal_to_abnormal` pixel-success numbers are a conservative lower bound:
the evaluator still scores only the fixed central region, so a fake defect
planted off-center counts as a pixel failure even when the attack worked. That
region is `NORMAL_TARGET_REGION_FRACTION` of each side, so the default 0.25 is a
130x130 box, about 6.3% of the image area, while `K=0.20` spreads the fake
anomaly over about 20% of the area. A broad fake defect therefore tends to
overlap the scored box anyway; a smaller `K` concentrates the attack but makes
its placement matter more. The same box is what `ce_focal_dice` optimizes
against, so that one metric is tilted in its favour by construction. Image-level
metrics and the threshold-free pixel metrics (P-AUROC, AUPRO) carry no such
caveat, and `abnormal_to_normal` is unaffected in every metric because it is
scored inside the ground-truth defect mask.

`K` is a fixed hyperparameter per direction, not an ablation axis: two runs with
different `K` produce the same delta filenames and condition names, so sweep it
into a separate `PIPELINE_OUTPUT` (or with `OVERWRITE_EXISTING=true`) rather than
side by side. Changing `K` does invalidate reuse of existing `margin_topk`
deltas, so a re-run regenerates them automatically.


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

To run only one loss formulation:

```bash
LOSS_FORMULATIONS=margin_topk RUN_SETUPS=steps500_eps2 bash train.sh
```

Each formulation is generated and evaluated under its own
`setups/<setup>/<formulation>/` folder, so this can be run once per formulation
in separate sessions and the results accumulate.

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
`RUN_SETUPS=all` is intended for a server or a multi-session workflow. Running
both loss formulations doubles the work inside a single setup, so
`LOSS_FORMULATIONS=ce_focal_dice` or `LOSS_FORMULATIONS=margin_topk` splits one
setup across two sessions; each writes its own folder and the second session's
audit picks up both. The `SMOKE_TEST=true` option validates plumbing only
and must never be reported as a benchmark result.

The notebook exposes separate generation and evaluation batch sizes. Both are
set to 2 by default for a 16 GB Kaggle T4.

## Output layout

```text
outputs/
  datasets_mvtec/                 # or datasets_visa / datasets_mvtec_visa
    calibrated_thresholds/anomalyclip/{mvtec,visa}/
    setups/
      steps500_eps2/
        ce_focal_dice/
          attack_generation/
          evaluation/
            fixed_0_5/{numerical,visualizations}/
            image_f1/{numerical,visualizations}/
            clean_pixel_f1/{numerical,visualizations}/
        margin_topk/
          attack_generation/
          evaluation/
      steps500_eps4/
      steps800_eps2/
      steps800_eps4/
    ablation_high_level_summary.csv
```

Each loss formulation gets its own bundle and its own evaluation tree, because
generation rewrites `attack_manifest.csv` for the whole bundle it writes into and
evaluation decides completeness from the row count of one `summary.csv`. Sharing
a folder would let two single-formulation runs discard each other's results.
Clean thresholds do not depend on the attack, so `calibrated_thresholds/` stays
shared.

`audit_results.py` discovers the formulation folders present on disk rather than
reading `LOSS_FORMULATIONS`, so the comparison CSV aggregates every formulation
that has been evaluated so far, not only the one from the latest run. Result
folders written before this layout (`setups/<id>/evaluation/...`, with no
formulation level) are still audited and reported as `ce_focal_dice`. To reuse
their generated deltas as well, move them down one level:

```bash
cd outputs/datasets_mvtec/setups/steps500_eps2
mkdir -p ce_focal_dice
mv attack_generation evaluation ce_focal_dice/
```

The high-level CSV contains the four main threshold-free benchmark metrics
(I-AUROC, image AP, P-AUROC, and AUPRO) plus image-level and target-region
pixel-level success diagnostics.
