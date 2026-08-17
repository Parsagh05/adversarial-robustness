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
    [[ "$MVTEC_ROOT" != /ABSOLUTE/PATH/TO/* && -d "$MVTEC_ROOT/bottle/test" ]] || { echo "Invalid MVTec root: $MVTEC_ROOT" >&2; exit 2; }
    ;;
  visa)
    [[ "$VISA_ROOT" != /ABSOLUTE/PATH/TO/* && -f "$VISA_ROOT/split_csv/1cls.csv" ]] || { echo "Invalid VisA root: $VISA_ROOT" >&2; exit 2; }
    ;;
  mvtec,visa)
    [[ "$MVTEC_ROOT" != /ABSOLUTE/PATH/TO/* && -d "$MVTEC_ROOT/bottle/test" ]] || { echo "Invalid MVTec root: $MVTEC_ROOT" >&2; exit 2; }
    [[ "$VISA_ROOT" != /ABSOLUTE/PATH/TO/* && -f "$VISA_ROOT/split_csv/1cls.csv" ]] || { echo "Invalid VisA root: $VISA_ROOT" >&2; exit 2; }
    ;;
  *) echo "DATASETS must be mvtec, visa, or mvtec,visa" >&2; exit 2 ;;
esac
case "$RUN_PHASE" in all|generate|evaluate) ;; *) echo "RUN_PHASE must be all, generate, or evaluate" >&2; exit 2 ;; esac
case "$LOSS_FORMULATIONS" in
  ce_focal_dice|margin_topk|ce_focal_dice,margin_topk|margin_topk,ce_focal_dice) ;;
  *) echo "LOSS_FORMULATIONS must be ce_focal_dice, margin_topk, or both" >&2; exit 2 ;;
esac
IFS=',' read -r -a FORMULATIONS <<< "$LOSS_FORMULATIONS"

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
case "$DATASETS" in
  mvtec|mvtec,visa)
    [[ -f "$MODEL_ROOT/checkpoints/9_12_4_multiscale/epoch_15.pth" ]] || {
      echo "Missing AnomalyCLIP MVTec-target checkpoint" >&2
      exit 2
    }
    ;;
esac
case "$DATASETS" in
  visa|mvtec,visa)
    [[ -f "$MODEL_ROOT/checkpoints/9_12_4_multiscale_visa/epoch_15.pth" ]] || {
      echo "Missing AnomalyCLIP VisA-target checkpoint" >&2
      exit 2
    }
    ;;
esac

STAMP="$RUNTIME_ROOT/.evaluation_dependencies_ready"
if [[ ! -f "$STAMP" ]]; then
  "$PYTHON_BIN" -m pip install -q -r "$ROOT/evaluation/requirements.txt"
  touch "$STAMP"
fi
"$PYTHON_BIN" "$ROOT/preflight.py"

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
    # One bundle per loss formulation. Generation rewrites attack_manifest.csv
    # for the whole bundle it writes into, so a run restricted to one
    # formulation must not share a folder with the other one.
    for formulation in "${FORMULATIONS[@]}"; do
      echo "===== GENERATE datasets=$DATASETS $id/$formulation: steps=$steps epsilon=$epsilon ====="
      OUTPUT_BASE="$RESULTS_ROOT/setups/$id/$formulation/attack_generation" \
      GENERATION_DATASETS="$DATASETS" \
      PER_DATASET_STEPS="$steps" \
      EPSILON="$epsilon" \
      LOSS_FORMULATIONS="$formulation" \
      RUN_PER_DATASET=true RUN_PER_CATEGORY=false RUN_PER_IMAGE=false \
        bash "$ROOT/generation/train.sh"
    done
  done
fi

if [[ "$RUN_PHASE" == "all" || "$RUN_PHASE" == "evaluate" ]]; then
  "$PYTHON_BIN" "$ROOT/run_evaluations.py"
fi

"$PYTHON_BIN" "$ROOT/audit_results.py"
echo "Complete: $RESULTS_ROOT"
