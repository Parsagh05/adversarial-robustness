#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${MVTEC_ROOT:?MVTEC_ROOT is required}"
: "${VISA_ROOT:?VISA_ROOT is required}"
: "${OUTPUT_BASE:?OUTPUT_BASE is required}"
: "${WORK_DIR:?WORK_DIR is required}"
: "${ANOMALYCLIP_COMMIT:?ANOMALYCLIP_COMMIT is required}"

export PROJECT_ROOT="$ROOT"
export PROTOCOL_DIR="$OUTPUT_BASE/protocol"
export ATTACK_TRAIN_CSV="$PROTOCOL_DIR/attack_train_indices.csv"
export EVALUATION_CSV="$PROTOCOL_DIR/evaluation_test_indices.csv"
export ATTACK_TRAIN_FRACTION="${ATTACK_TRAIN_FRACTION:-1.0}"
export PER_DATASET_ATTACK_TRAIN_FRACTIONS="$ATTACK_TRAIN_FRACTION"
export GENERATION_DATASETS="${GENERATION_DATASETS:-mvtec,visa}"
export DIRECTIONS="${DIRECTIONS:-normal_to_abnormal,abnormal_to_normal}"
export LOSS_MODES="${LOSS_MODES:-global,local,combined}"
export LOSS_FORMULATIONS="${LOSS_FORMULATIONS:-ce_focal_dice,margin_topk}"
export MARGIN_TOPK_FRACTION_NORMAL_TO_ABNORMAL="${MARGIN_TOPK_FRACTION_NORMAL_TO_ABNORMAL:-0.20}"
export MARGIN_TOPK_FRACTION_ABNORMAL_TO_NORMAL="${MARGIN_TOPK_FRACTION_ABNORMAL_TO_NORMAL:-0.40}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export PYTHONUNBUFFERED=1
mkdir -p "$WORK_DIR" "$OUTPUT_BASE/logs"

PYTHON="${PYTHON_BIN:-python3}"
"$PYTHON" -m pip --version >/dev/null
echo "Python runtime: $PYTHON"

MODEL_ROOT="$WORK_DIR/AnomalyCLIP"
if [[ ! -d "$MODEL_ROOT/.git" ]]; then
  git clone --filter=blob:none --no-checkout \
    https://github.com/zqhang/AnomalyCLIP.git "$MODEL_ROOT"
fi
git -C "$MODEL_ROOT" fetch --depth 1 origin "$ANOMALYCLIP_COMMIT"
git -C "$MODEL_ROOT" checkout --detach --force FETCH_HEAD
[[ "$(git -C "$MODEL_ROOT" rev-parse HEAD)" == "$ANOMALYCLIP_COMMIT" ]] || {
  echo "Pinned AnomalyCLIP checkout mismatch" >&2
  exit 2
}

SETUP_STAMP="$WORK_DIR/.ablation_dependencies_ready"
if [[ ! -f "$SETUP_STAMP" ]]; then
  "$PYTHON" -m pip install -q -r "$ROOT/requirements.txt"
  touch "$SETUP_STAMP"
fi

"$PYTHON" "$ROOT/common.py" split | tee "$OUTPUT_BASE/logs/00_split.log"
"$PYTHON" "$ROOT/run_per_dataset.py" 2>&1 | tee "$OUTPUT_BASE/logs/per_dataset.log"

