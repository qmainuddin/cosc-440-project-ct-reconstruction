#!/usr/bin/env bash
# Run on either supported UC lab machine; use the fresh seed-50, LR-0.0001 experiment settings.
set -euo pipefail
ACTION="${1:-}"
case "$ACTION" in smoke|train|resume|compare) ;; *) echo 'Usage: bash run_lab.sh smoke|train|resume|compare' >&2; exit 2;; esac
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
LAB_HOST="$(hostname -s)"
case "$LAB_HOST" in
 cs23002lv) SEED=50; TAG=3090; BASELINE="runs/3090-baseline-s50-lr1e-4";;
 cs25049kx) SEED=50; TAG=4070; BASELINE="runs/4070-baseline-s50-lr1e-4";;
 *) echo "Unrecognised host: $LAB_HOST. Use the documented manual commands." >&2; exit 2;;
esac
PYTHON_BIN="${COSC440_PYTHON:-$HOME/.venvs/cosc440-$LAB_HOST/bin/python}"
DATASET_PATH="${COSC440_DATASET:-$SCRIPT_DIR/data/ct_data/chest_50.pickle}"
ATTENTION="runs/$TAG-attention-s$SEED-lr1e-4"
[[ -x "$PYTHON_BIN" ]] || { echo "Missing Python: $PYTHON_BIN" >&2; exit 2; }
[[ -f "$DATASET_PATH" ]] || { echo "Missing dataset: $DATASET_PATH" >&2; exit 2; }
mkdir -p logs
COMMON=(--dataset "$DATASET_PATH" --n-points 192 --n-rays 2048 --depth 8 --embedding-dim 1 --learning-rate 0.0001 --seed "$SEED" --query-batch 8192 --save-every 10 --device gpu)
if [[ "$ACTION" == smoke ]]; then
 "$PYTHON_BIN" -m unittest test_project
 SMOKE="runs/$TAG-attention-smoke-$(date +%Y%m%d-%H%M%S)"
 "$PYTHON_BIN" gpu_python.py project_main.py "${COMMON[@]}" --model attention --output "$SMOKE" --epochs 1 --max-projections 2 2>&1 | tee "logs/$(basename "$SMOKE").log"
 exit 0
fi
run_model() {
 local model="$1" folder="$2"
 local resume_args=()
 if [[ -d "$folder" ]]; then
  # Validate a reused/resumed run before skipping it or loading its checkpoint.
  local completed
  completed="$("$PYTHON_BIN" - "$folder" "$model" "$SEED" "$DATASET_PATH" <<'PY'
import csv,hashlib,json,sys
from pathlib import Path
p=Path(sys.argv[1]);c=json.loads((p/'config.json').read_text())
expected={'model':sys.argv[2],'seed':int(sys.argv[3]),'learning_rate':.0001,'n_points':192,'n_rays':2048,'depth':8,'embedding_dim':1,'max_projections':0,'dataset_sha256':hashlib.sha256(Path(sys.argv[4]).read_bytes()).hexdigest()}
for key,value in expected.items():
 if c.get(key)!=value:raise SystemExit(f'Cannot reuse {p}: {key} differs')
rows=list(csv.DictReader((p/'metrics.csv').open())) if (p/'metrics.csv').exists() else []
epochs=[int(r['epoch']) for r in rows]
if epochs!=list(range(1,len(rows)+1)):raise SystemExit('Incomplete or duplicate metric history')
print(len(rows))
PY
)"
  if (( completed >= 250 )); then
   [[ -f "$folder/epoch_0250.tiff" ]] || { echo 'Missing epoch-250 image' >&2; exit 2; }
   echo "Reusing completed matching $model run: $folder"
   return
  fi
  [[ "$ACTION" == resume ]] || { echo "Incomplete run $folder; use resume after confirming no job is active." >&2; exit 2; }
  resume_args=(--resume)
 fi
 "$PYTHON_BIN" gpu_python.py project_main.py "${COMMON[@]}" --model "$model" --output "$folder" --epochs 250 --max-projections 0 "${resume_args[@]}" 2>&1 | tee -a "logs/$(basename "$folder").log"
}
if [[ "$ACTION" != compare ]]; then
 run_model baseline "$BASELINE"
 run_model attention "$ATTENTION"
fi
"$PYTHON_BIN" compare_runs.py "$BASELINE" "$ATTENTION" --epoch 250 --output "runs/$TAG-attention-comparison-s$SEED"
