#!/usr/bin/env bash
# Edit the three paths, then run: bash train.sh

MVTEC_ROOT="${MVTEC_ROOT:-/ABSOLUTE/PATH/TO/mvtec_anomaly_detection}"
VISA_ROOT="${VISA_ROOT:-/ABSOLUTE/PATH/TO/VisA_20220922}"
OUTPUT_BASE="${OUTPUT_BASE:-/ABSOLUTE/PATH/TO/canonical_clip_outputs}"

# Pin the external feature-loader implementation used by every run.
ANOMALYCLIP_COMMIT="${ANOMALYCLIP_COMMIT:-3911738c0867544f545a076ad78f3f11d9ecbfdf}"

# Fixed CSV split: 50% attack_train and 50% evaluation/test.
SPLIT_SEED="${SPLIT_SEED:-111}"
EVALUATION_FRACTION="${EVALUATION_FRACTION:-0.50}"

# Fraction used from the attack_train half. Use one value per run.
# Later change it to 0.05, 0.10, 0.25, 0.50, or 1.00 for data-efficiency.
ATTACK_TRAIN_FRACTION="${ATTACK_TRAIN_FRACTION:-1.00}"

RUN_PER_DATASET="${RUN_PER_DATASET:-true}"
RUN_PER_CATEGORY="${RUN_PER_CATEGORY:-true}"
RUN_PER_IMAGE="${RUN_PER_IMAGE:-true}"

GPU="${GPU:-0}"
IMAGE_SIZE="${IMAGE_SIZE:-518}"
EPSILON="${EPSILON:-8/255}"
ATTACK_SEED="${ATTACK_SEED:-111}"

PER_DATASET_STEPS="${PER_DATASET_STEPS:-500}"
PER_DATASET_STEP_SIZE="${PER_DATASET_STEP_SIZE:-0.25/255}"
PER_DATASET_BATCH_SIZE="${PER_DATASET_BATCH_SIZE:-8}"

PER_CATEGORY_STEPS="${PER_CATEGORY_STEPS:-128}"
PER_CATEGORY_STEP_SIZE="${PER_CATEGORY_STEP_SIZE:-0.25/255}"
PER_CATEGORY_EFFECTIVE_BATCH_SIZE="${PER_CATEGORY_EFFECTIVE_BATCH_SIZE:-8}"
PER_CATEGORY_MICRO_BATCH_SIZE="${PER_CATEGORY_MICRO_BATCH_SIZE:-2}"

PER_IMAGE_STEPS="${PER_IMAGE_STEPS:-20}"
PER_IMAGE_STEP_SIZE="${PER_IMAGE_STEP_SIZE:-0.5/255}"
PER_IMAGE_BATCH_SIZE="${PER_IMAGE_BATCH_SIZE:-2}"

# Segmentation-aware local objective. The target-class focal and soft-Dice
# terms are evaluated on patch tokens; defect masks focus abnormal->normal
# attacks, while zero-mask normal images intentionally target the full grid.
LOCAL_FOCAL_WEIGHT="${LOCAL_FOCAL_WEIGHT:-0.5}"
LOCAL_DICE_WEIGHT="${LOCAL_DICE_WEIGHT:-0.5}"
LOCAL_FOCAL_GAMMA="${LOCAL_FOCAL_GAMMA:-2.0}"
LOCAL_DICE_SMOOTH="${LOCAL_DICE_SMOOTH:-1.0}"
LOCAL_BACKGROUND_WEIGHT="${LOCAL_BACKGROUND_WEIGHT:-0.1}"

# Cosine decay prevents a sign-PGD iterate from bouncing indefinitely on the
# Linf boundary. Fixed-subset diagnostics are recorded at this interval.
STEP_SIZE_SCHEDULE="${STEP_SIZE_SCHEDULE:-cosine}"
STEP_SIZE_MIN_RATIO="${STEP_SIZE_MIN_RATIO:-0.1}"
DIAGNOSTIC_INTERVAL="${DIAGNOSTIC_INTERVAL:-8}"
DIAGNOSTIC_MAX_SAMPLES="${DIAGNOSTIC_MAX_SAMPLES:-16}"
