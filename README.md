# COSC440: fresh baseline and attention experiments

Active branch: main. Fresh experiment settings:

- Seed: **50** for every model on both machines.
- Learning rate: **0.0001** (1e-4) for every model.
- Epochs: **250** for both baseline and attention.
- Depth 8, embedding width 1, 192 points per ray, 2,048 rays per projection.
- Adam optimiser; all training projections; reconstruction evaluated every epoch.

The previous runs/ folder has been removed. The remote main-backup branch preserves
all prior tracked runs and the local analysis artifacts. archive/ retains the old
submission, logs, screenshots, original starter and documentation with their actual
historical settings. Those are NOT results for this fresh experiment. No fresh
seed-50 result or replacement final report has yet been produced.

The main branch contains baseline and attention. Trilinear development is still on
multiscale-trilinear-interpolation; this reset does not alter that separate branch.

## Get the updated code on a UC machine

Wait for any existing training jobs to finish or stop them deliberately before
switching branches. Both UC machines share the same home-directory checkout.
Update it once, before starting jobs on either machine:

```bash
cd "$HOME/Desktop/cosc-440-project/cosc-440-project-ct-reconstruction"
git status --short
git fetch origin
git switch main
git pull --ff-only origin main
```

Do not discard local modifications if Git reports a conflict. Git removes tracked
run files deleted by this update, but it does not delete ignored checkpoints or
untracked local outputs. Therefore a lab checkout may still have leftover runs/.
Keep them out of the new experiment; the new names below never reuse old folders.
Local Mac deletion does not automatically remove every untracked file on a lab PC.

## Environment and supplied dataset

The helper recognises cs23002lv (3090) and cs25049kx (4070), and uses the existing
host-specific Python environment. Keep course data at data/ct_data/chest_50.pickle.

```bash
source "$HOME/.venvs/cosc440-$(hostname -s)/bin/activate"
python -m pip check
nvidia-smi
```

For a new environment, install requirements-linux.txt in that environment. The
requirements.txt file specifies shared dependencies. requirements-installed-local.txt
records the earlier lab environment, not fresh experiment outcomes. No new Python
packages are needed for the settings change.

## Verify before full training

```bash
bash run_lab.sh smoke
```

This runs the five assertion tests, then one full-size attention epoch on only two
projections to check GPU availability/memory. The diagnostic folder is timestamped.
It uses seed 50 and learning rate 0.0001, but is NOT a full experimental run.
Stop on errors and retain the output before changing settings.

## Full experiment on either machine

```bash
mkdir -p logs
nohup bash run_lab.sh train > "logs/fresh-$(hostname -s)-launch.log" 2>&1 &
echo "Training process ID: $!"
tail -f "logs/fresh-$(hostname -s)-launch.log"
```

The helper trains baseline first, then attention, each for 250 epochs, and generates
comparison figures. Ctrl+C stops tail, not training. nohup survives SSH disconnect;
it does not survive shutdown. Do not start a second job against the same folders.

| Host | Baseline output | Attention output |
|---|---|---|
| cs23002lv | runs/3090-baseline-s50-lr1e-4 | runs/3090-attention-s50-lr1e-4 |
| cs25049kx | runs/4070-baseline-s50-lr1e-4 | runs/4070-attention-s50-lr1e-4 |

Both machines now use seed 50. Runs on different GPUs are not independent-seed
replicates, and identical numeric results across hardware are not guaranteed.
Prefer comparing the two models on the same machine with the same budget.

If interrupted, confirm the earlier job has stopped, then:

```bash
nohup bash run_lab.sh resume > "logs/fresh-$(hostname -s)-resume.log" 2>&1 &
```

Existing runs must match seed, learning rate, dataset and sampling settings.
Incomplete runs require resume; completed matching runs are reused. Checkpoints
are required for resume and remain local. A fresh run must not use --resume.
A stale .running lock must only be removed after confirming no job is active.

## Manual commands (activated environment, project root)

Use unique output folders. Example for the 3090:

```bash
python gpu_python.py project_main.py --model baseline \
  --output runs/3090-baseline-s50-lr1e-4 \
  --seed 50 --learning-rate 0.0001 --epochs 250 --device gpu
python gpu_python.py project_main.py --model attention \
  --output runs/3090-attention-s50-lr1e-4 \
  --seed 50 --learning-rate 0.0001 --epochs 250 --device gpu
python compare_runs.py runs/3090-baseline-s50-lr1e-4 \
  runs/3090-attention-s50-lr1e-4 --epoch 250 --output runs/3090-attention-comparison-s50
```

For the 4070 change the folder prefix to 4070, keeping seed 50 and learning rate
0.0001. Alternatively, `bash run_lab.sh compare` chooses the host-specific folders.
The helper accepts COSC440_DATASET and COSC440_PYTHON environment overrides for
an alternative absolute dataset path or Python executable.

## What to collect

Each full run records config.json, runtime.json, metrics.csv, raw/display TIFFs,
reference TIFF and local checkpoints. Preserve terminal logs and screenshots.
The comparison writes PNG/PDF learning curves and three-slice images, plus JSON
metric differences. A lower learning rate may learn more slowly; it does not
promise higher SSIM after the 250-epoch budget.

Commit the new run folders and logs only after completion; exclude course datasets,
virtual environments and checkpoints. Rebuild the report and submission from the
actual fresh measurements. The archived PDF must not be relabelled as seed-50 work.

## Code overview and checks

- project_main.py: baseline/attention models, sampling, integration, training,
  checkpoints, evaluation and command-line settings.
- geometry.py: supplied scan geometry and dataset loader.
- gpu_python.py: makes installed NVIDIA libraries discoverable before Python starts.
- project_unit_test.py: supplied print-out test; leave its numerical examples intact.
- test_project.py: assertion tests, using seed 50 and optimiser rate 0.0001.
- compare_runs.py: validates matching settings and compares a shared saved epoch.
- run_lab.sh: lab smoke/train/resume/compare launcher for these fresh settings.

```bash
python -m unittest test_project
python project_unit_test.py
```

Original test coordinates and package version numbers are not training seeds or
learning rates; they are deliberately unchanged. See explanation.md for exact edits
and VALIDATION.md for the checks actually performed.
