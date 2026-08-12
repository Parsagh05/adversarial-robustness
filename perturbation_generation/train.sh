#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
set -a
source "$ROOT/config.sh"
set +a

export CUDA_VISIBLE_DEVICES="$GPU"
export PROJECT_ROOT="$ROOT"
export WORK_DIR="$ROOT/runtime"
export PROTOCOL_DIR="$OUTPUT_BASE/protocol"
export ATTACK_TRAIN_CSV="$PROTOCOL_DIR/attack_train_indices.csv"
export EVALUATION_CSV="$PROTOCOL_DIR/evaluation_test_indices.csv"
export ATTACK_TRAIN_FRACTION
export PER_DATASET_ATTACK_TRAIN_FRACTIONS="$ATTACK_TRAIN_FRACTION"
export PER_CATEGORY_ATTACK_TRAIN_FRACTIONS="$ATTACK_TRAIN_FRACTION"
export PER_IMAGE_EFFECTIVE_BATCH_SIZE="$PER_IMAGE_BATCH_SIZE"
export PER_IMAGE_MICRO_BATCH_SIZE="$PER_IMAGE_BATCH_SIZE"
export DIRECTIONS="${DIRECTIONS:-normal_to_abnormal,abnormal_to_normal}"
export LOSS_MODES="${LOSS_MODES:-global,local,combined}"
export USE_AMP="${USE_AMP:-true}"
export CACHE_INPUTS_IN_RAM="${CACHE_INPUTS_IN_RAM:-true}"
export OVERWRITE_EXISTING="${OVERWRITE_EXISTING:-false}"
export PER_IMAGE_EVALUATION_FRACTION="${PER_IMAGE_EVALUATION_FRACTION:-1.0}"
export PER_DATASET_DIAGNOSTIC_MAX_SAMPLES="$DIAGNOSTIC_MAX_SAMPLES"
export PER_CATEGORY_DIAGNOSTIC_MAX_SAMPLES="$DIAGNOSTIC_MAX_SAMPLES"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export PYTHONUNBUFFERED=1

for name in OUTPUT_BASE; do
  value="${!name}"
  [[ "$value" != /ABSOLUTE/PATH/TO/* ]] || { echo "Edit $name in config.sh" >&2; exit 2; }
done
case ",$GENERATION_DATASETS," in
  *,mvtec,*) [[ -d "$MVTEC_ROOT" ]] || { echo "Missing MVTec directory: $MVTEC_ROOT" >&2; exit 2; } ;;
esac
case ",$GENERATION_DATASETS," in
  *,visa,*) [[ -d "$VISA_ROOT" ]] || { echo "Missing VisA directory: $VISA_ROOT" >&2; exit 2; } ;;
esac
mkdir -p "$WORK_DIR" "$OUTPUT_BASE/logs"

PYTHON="${PYTHON_BIN:-python3}"
USE_VENV="${USE_VENV:-false}"
if [[ "${USE_VENV,,}" == "true" ]]; then
  VENV_DIR="$ROOT/.venv"
  if [[ ! -x "$VENV_DIR/bin/python" ]]; then
    if ! "$PYTHON" -m venv --system-site-packages "$VENV_DIR"; then
      echo "WARNING: virtualenv creation failed; using $PYTHON directly." >&2
    fi
  fi
  if [[ -x "$VENV_DIR/bin/python" ]] && "$VENV_DIR/bin/python" -m pip --version >/dev/null 2>&1; then
    PYTHON="$VENV_DIR/bin/python"
  else
    echo "WARNING: virtualenv has no working pip; using $PYTHON directly." >&2
  fi
fi
"$PYTHON" -m pip --version >/dev/null
echo "Python runtime: $PYTHON"

clone_pinned() {
  local url="$1" dest="$2" commit="$3"
  if [[ ! -d "$dest/.git" ]]; then
    git clone --filter=blob:none --no-checkout "$url" "$dest"
  fi
  git -C "$dest" fetch --depth 1 origin "$commit"
  git -C "$dest" checkout --detach --force FETCH_HEAD
  [[ "$(git -C "$dest" rev-parse HEAD)" == "$commit" ]] || {
    echo "Pinned checkout mismatch for $dest" >&2
    exit 2
  }
}
clone_pinned \
  "https://github.com/zqhang/AnomalyCLIP.git" \
  "$WORK_DIR/AnomalyCLIP" \
  "$ANOMALYCLIP_COMMIT"

SETUP_STAMP="$WORK_DIR/.canonical_clip_segmentation_loss_v2_ready"
if [[ ! -f "$SETUP_STAMP" ]]; then
  "$PYTHON" -m pip install -q -r "$ROOT/requirements.txt"
  touch "$SETUP_STAMP"
fi

# The two CSV files are created once, before all attack modes.
"$PYTHON" "$ROOT/common.py" split | tee "$OUTPUT_BASE/logs/00_split.log"

run_mode() {
  local enabled="$1" name="$2" script="$3"
  if [[ "${enabled,,}" == "true" ]]; then
    echo "===== $name ====="
    "$PYTHON" "$ROOT/$script" 2>&1 | tee "$OUTPUT_BASE/logs/$name.log"
  fi
}

run_mode "$RUN_PER_DATASET"  per_dataset  run_per_dataset.py
run_mode "$RUN_PER_CATEGORY" per_category run_per_category.py
run_mode "$RUN_PER_IMAGE"    per_image    run_per_image.py

echo "Done. Results are in: $OUTPUT_BASE"
