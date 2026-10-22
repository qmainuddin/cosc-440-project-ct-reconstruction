#!/usr/bin/env bash
# Repeat the submitted paired experiment using a separately supplied course dataset.
set -euo pipefail
if [[ $# != 2 ]]; then
  echo "Usage: bash run_experiments.sh /absolute/path/chest_50.pickle /absolute/path/NEW-output-folder" >&2
  exit 2
fi
DATASET="$1"
RUN_ROOT="$2"
case "$DATASET" in /*) ;; *) echo 'Dataset path must be absolute.' >&2; exit 2;; esac
case "$RUN_ROOT" in /*) ;; *) echo 'Output path must be absolute.' >&2; exit 2;; esac
[[ -f "$DATASET" ]] || { echo 'Dataset file not found.' >&2; exit 2; }
[[ ! -e "$RUN_ROOT" ]] || { echo 'Use a new output directory to preserve earlier results.' >&2; exit 2; }
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
mkdir -p "$RUN_ROOT"
cd "$SCRIPT_DIR"
"$PYTHON_BIN" --version > "$RUN_ROOT/test-environment.txt" 2>&1
"$PYTHON_BIN" -m unittest test_project 2>&1 | tee "$RUN_ROOT/automated-tests.txt"
"$PYTHON_BIN" project_unit_test.py 2>&1 | tee "$RUN_ROOT/supplied-unit-test.txt"
for model in baseline attention; do
  EPOCHS=100
  [[ "$model" != attention ]] || EPOCHS=300
  "$PYTHON_BIN" gpu_python.py project_main.py \
    --dataset "$DATASET" --model "$model" --output "$RUN_ROOT/$model-s43" \
    --epochs "$EPOCHS" --n-points 192 --n-rays 2048 --depth 8 --embedding-dim 1 \
    --token-dim 8 --heads 2 --learning-rate 0.001 --seed 43 \
    --query-batch 8192 --save-every 10 --max-projections 0 --device gpu \
    2>&1 | tee "$RUN_ROOT/$model-s43.log"
done
"$PYTHON_BIN" compare_runs.py "$RUN_ROOT/baseline-s43" "$RUN_ROOT/attention-s43" \
  --epoch 100 --output "$RUN_ROOT/comparison"
