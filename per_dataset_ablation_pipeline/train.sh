#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
set -a
source "$ROOT/config.sh"
set +a

for name in PIPELINE_OUTPUT; do
  value="${!name}"
  [[ "$value" != /ABSOLUTE/PATH/TO/* ]] || {
    echo "Set $name in config.sh or the environment" >&2
    exit 2
  }
done
case "$DATASETS" in
  mvtec)
    [[ "$MVTEC_ROOT" != /ABSOLUTE/PATH/TO/* && -d "$MVTEC_ROOT" ]] || { echo "Missing MVTec: $MVTEC_ROOT" >&2; exit 2; }
    ;;
  visa)
    [[ "$VISA_ROOT" != /ABSOLUTE/PATH/TO/* && -d "$VISA_ROOT" ]] || { echo "Missing VisA: $VISA_ROOT" >&2; exit 2; }
    ;;
  mvtec,visa)
    [[ "$MVTEC_ROOT" != /ABSOLUTE/PATH/TO/* && -d "$MVTEC_ROOT" ]] || { echo "Missing MVTec: $MVTEC_ROOT" >&2; exit 2; }
    [[ "$VISA_ROOT" != /ABSOLUTE/PATH/TO/* && -d "$VISA_ROOT" ]] || { echo "Missing VisA: $VISA_ROOT" >&2; exit 2; }
    ;;
  *) echo "DATASETS must be mvtec, visa, or mvtec,visa" >&2; exit 2 ;;
esac
case "$RUN_PHASE" in all|generate|evaluate) ;; *) echo "RUN_PHASE must be all, generate, or evaluate" >&2; exit 2 ;; esac

export CUDA_VISIBLE_DEVICES="$GPU"
export PYTHON_BIN="${PYTHON_BIN:-python3}"
export PIPELINE_ROOT="$ROOT"
export RUNTIME_ROOT="${RUNTIME_ROOT:-$PIPELINE_OUTPUT/runtime}"
DATASET_TAG="${DATASETS//,/_}"
export RESULTS_ROOT="$PIPELINE_OUTPUT/datasets_$DATASET_TAG"
export WORK_DIR="$RUNTIME_ROOT"
mkdir -p "$RESULTS_ROOT/setups" "$RUNTIME_ROOT"

"$PYTHON_BIN" -m pip --version >/dev/null
MODEL_ROOT="$RUNTIME_ROOT/AnomalyCLIP"
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

STAMP="$RUNTIME_ROOT/.evaluation_dependencies_ready"
if [[ ! -f "$STAMP" ]]; then
  "$PYTHON_BIN" -m pip install -q -r "$ROOT/evaluation/requirements.txt"
  touch "$STAMP"
fi

SETUP_IDS=(steps500_eps2 steps500_eps4 steps800_eps2 steps800_eps4)
SETUP_STEPS=(500 500 800 800)
SETUP_EPS=(2/255 4/255 2/255 4/255)

selected() {
  local id="$1"
  [[ "$RUN_SETUPS" == "all" || ",$RUN_SETUPS," == *",$id,"* ]]
}

selected_count=0
for id in "${SETUP_IDS[@]}"; do
  if selected "$id"; then
    selected_count=$((selected_count + 1))
  fi
done
[[ "$selected_count" -gt 0 ]] || {
  echo "RUN_SETUPS did not select a known setup" >&2
  exit 2
}

if [[ "$RUN_PHASE" == "all" || "$RUN_PHASE" == "generate" ]]; then
  for index in "${!SETUP_IDS[@]}"; do
    id="${SETUP_IDS[$index]}"
    selected "$id" || continue
    steps="${SETUP_STEPS[$index]}"
    epsilon="${SETUP_EPS[$index]}"
    if [[ "${SMOKE_TEST,,}" == "true" ]]; then
      steps="$SMOKE_STEPS"
    fi
    setup_root="$RESULTS_ROOT/setups/$id"
    echo "===== GENERATE datasets=$DATASETS $id: steps=$steps epsilon=$epsilon ====="
    OUTPUT_BASE="$setup_root/attack_generation" \
    GENERATION_DATASETS="$DATASETS" \
    PER_DATASET_STEPS="$steps" \
    EPSILON="$epsilon" \
    RUN_PER_DATASET=true RUN_PER_CATEGORY=false RUN_PER_IMAGE=false \
      bash "$ROOT/generation/train.sh"
  done
fi

if [[ "$RUN_PHASE" == "all" || "$RUN_PHASE" == "evaluate" ]]; then
  "$PYTHON_BIN" "$ROOT/run_evaluations.py"
fi

"$PYTHON_BIN" "$ROOT/audit_results.py"
echo "Complete: $RESULTS_ROOT"
