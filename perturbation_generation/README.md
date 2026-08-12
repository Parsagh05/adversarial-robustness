# Corrected adversarial perturbation generator (v2)

This directory regenerates the fixed CLIP-surrogate perturbations used by the
black-box evaluation pipeline. Version 2 replaces patch-wise mean cross entropy
with a segmentation-aware local objective and writes to new artifact names, so
it cannot silently reuse or overwrite the original Kaggle dataset.

This project is self-contained: its local `adversarial_harness/` package owns
the attack, configuration, dataset, CLIP adapter, and prompt code required by
the generator. It does not import from `full_attack_generation_pipeline` or
`blackbox_evaluation_pipeline`.

## Local objective

For every selected CLIP layer and spatial token, the generator forms normal and
abnormal logits from the frozen public-CLIP prompt ensemble. The requested
target class is optimized with:

```
local = LOCAL_FOCAL_WEIGHT * target_class_focal
      + LOCAL_DICE_WEIGHT  * target_class_soft_dice
```

- `normal_to_abnormal`: the zero-mask fallback targets the full patch grid as
  anomalous. This explicitly models a dense false-positive attack.
- `abnormal_to_normal`: the ground-truth defect mask receives weight `1.0` and
  background receives `LOCAL_BACKGROUND_WEIGHT`. The target-class probability
  is normal, so the objective suppresses the known defect region.
- `combined`: the segmentation-aware local objective is combined with global
  targeted cross entropy using the existing global/local weights.

The attack still uses only the frozen public CLIP surrogate. It never loads or
differentiates through an evaluated anomaly detector.

## Optimization safeguards

- Dataset-level attacks use an effective batch of 8 instead of one image.
- PGD uses cosine step-size decay and smaller initial steps.
- A fixed diagnostic subset is evaluated periodically. Random batch loss is
  labelled separately and is never presented as a convergence curve.
- Gradient norms, Linf-bound saturation, initial/final focal and Dice losses,
  and the full universal-optimization history are stored in artifact metadata.
- The best fixed-diagnostic checkpoint is saved, rather than blindly saving the
  last stochastic iterate.
- Every bundle includes `optimization_diagnostics.csv`.
- Artifact reuse checks include all loss/schedule settings, generator hashes,
  repository commit, and the pinned AnomalyCLIP commit.

## Run locally

Edit paths in `config.sh`, or override them as environment variables:

```bash
export MVTEC_ROOT=/data/mvtec_anomaly_detection
export VISA_ROOT=/data/VisA_20220922
export OUTPUT_BASE=/data/canonical_clip_v2
bash train.sh
```

The three modes can be selected independently with `RUN_PER_DATASET`,
`RUN_PER_CATEGORY`, and `RUN_PER_IMAGE`. Long Kaggle runs should normally run
one scope per session. Existing `.pt` files are safely resumed only when every
reproducibility field matches.

## Outputs

- `canonical_clip_per_dataset_segmentation_loss_v2.zip`
- `canonical_clip_per_category_segmentation_loss_v2.zip`
- `canonical_clip_per_image_segmentation_loss_v2.zip`

Do not merge these archives with the old `canonical_clip_*` bundles under the
same dataset version. Publish them as a new Kaggle dataset version and rerun the
black-box evaluations before drawing conclusions from local or combined losses.

The ready-to-run Kaggle notebook is
`kaggle_generate_corrected_perturbations.ipynb` in this directory.
