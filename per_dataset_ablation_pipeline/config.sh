#!/usr/bin/env bash
# Paths may be supplied by the environment or edited here for a local server.
MVTEC_ROOT="${MVTEC_ROOT:-/ABSOLUTE/PATH/TO/mvtec_anomaly_detection}"
VISA_ROOT="${VISA_ROOT:-/ABSOLUTE/PATH/TO/VisA_20220922}"
PIPELINE_OUTPUT="${PIPELINE_OUTPUT:-/ABSOLUTE/PATH/TO/per_dataset_ablation_outputs}"
DATASETS="${DATASETS:-mvtec,visa}"  # mvtec, visa, or mvtec,visa

# all, or a comma-separated subset of:
# steps500_eps2, steps500_eps4, steps800_eps2, steps800_eps4
RUN_SETUPS="${RUN_SETUPS:-all}"
RUN_PHASE="${RUN_PHASE:-all}"  # all, generate, or evaluate
GPU="${GPU:-0}"

ANOMALYCLIP_COMMIT="${ANOMALYCLIP_COMMIT:-3911738c0867544f545a076ad78f3f11d9ecbfdf}"
SPLIT_SEED="${SPLIT_SEED:-111}"
EVALUATION_FRACTION="${EVALUATION_FRACTION:-0.50}"
ATTACK_TRAIN_FRACTION="${ATTACK_TRAIN_FRACTION:-1.00}"
IMAGE_SIZE="${IMAGE_SIZE:-518}"
PER_DATASET_STEP_SIZE="${PER_DATASET_STEP_SIZE:-0.25/255}"
PER_DATASET_BATCH_SIZE="${PER_DATASET_BATCH_SIZE:-8}"
DIAGNOSTIC_INTERVAL="${DIAGNOSTIC_INTERVAL:-50}"
STEP_SIZE_SCHEDULE="${STEP_SIZE_SCHEDULE:-cosine}"
STEP_SIZE_MIN_RATIO="${STEP_SIZE_MIN_RATIO:-0.1}"

NORMAL_LOCAL_TARGET="${NORMAL_LOCAL_TARGET:-fixed_region}"
NORMAL_TARGET_REGION_FRACTION="${NORMAL_TARGET_REGION_FRACTION:-0.25}"
NORMAL_TARGET_CENTER_X="${NORMAL_TARGET_CENTER_X:-0.5}"
NORMAL_TARGET_CENTER_Y="${NORMAL_TARGET_CENTER_Y:-0.5}"
LOCAL_FOCAL_WEIGHT="${LOCAL_FOCAL_WEIGHT:-0.5}"
LOCAL_DICE_WEIGHT="${LOCAL_DICE_WEIGHT:-0.5}"
LOCAL_FOCAL_GAMMA="${LOCAL_FOCAL_GAMMA:-2.0}"
LOCAL_DICE_SMOOTH="${LOCAL_DICE_SMOOTH:-1.0}"
LOCAL_BACKGROUND_WEIGHT="${LOCAL_BACKGROUND_WEIGHT:-0.1}"

# Objective families to ablate: ce_focal_dice, margin_topk, or both. Both
# doubles generation and evaluation time. ce_focal_dice is the mask-aware
# cross-entropy/focal/soft-Dice loss; margin_topk maximizes or minimizes the
# abnormal-minus-normal logit margin and its TopK anomaly map, without a target
# region and without ground-truth masks.
LOSS_FORMULATIONS="${LOSS_FORMULATIONS:-ce_focal_dice,margin_topk}"
# K for TopK(H), as a fraction of the patch tokens (37x37 = 1369 at 518px with
# the ViT-L/14 surrogate). Planting a fake defect needs a smaller region than
# suppressing a real one, so K is set per direction: 20% (274 tokens) when
# attacking normal images and 40% (548 tokens) when attacking anomalous ones.
MARGIN_TOPK_FRACTION_NORMAL_TO_ABNORMAL="${MARGIN_TOPK_FRACTION_NORMAL_TO_ABNORMAL:-0.20}"
MARGIN_TOPK_FRACTION_ABNORMAL_TO_NORMAL="${MARGIN_TOPK_FRACTION_ABNORMAL_TO_NORMAL:-0.40}"

# A target-region image is a pixel success when this fraction of eligible
# region pixels crosses the selected pixel threshold.
PIXEL_SUCCESS_MIN_FLIP_FRACTION="${PIXEL_SUCCESS_MIN_FLIP_FRACTION:-0.50}"
EVALUATION_BATCH_SIZE="${EVALUATION_BATCH_SIZE:-2}"
SAVE_PREDICTIONS="${SAVE_PREDICTIONS:-false}"
OVERWRITE_EXISTING="${OVERWRITE_EXISTING:-false}"

# Fast plumbing check. It intentionally does not produce benchmark results.
SMOKE_TEST="${SMOKE_TEST:-false}"
SMOKE_STEPS="${SMOKE_STEPS:-2}"
SMOKE_MAX_CONDITIONS="${SMOKE_MAX_CONDITIONS:-1}"
