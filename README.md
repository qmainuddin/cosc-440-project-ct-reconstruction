# COSC440: baseline and trilinear reconstruction

The Python training defaults remain seed **50**, learning rate **0.0001** and
**250 epochs**. The lab launcher requires you to state these values explicitly.
Both models use depth 8, embedding width 1, 192 points per ray, 2,048 rays per
projection, Adam, and all training projections. Reconstruction metrics are saved
every epoch, and TIFF snapshots every 10 epochs and at the final epoch.

## Get the code and environment

Wait until existing jobs finish before pulling or switching branches: both UC
machines share the same home-directory checkout. On the lab machine:

```bash
cd "$HOME/Desktop/cosc-440-project/cosc-440-project-ct-reconstruction"
git status --short
git fetch origin
git switch main-skalaton-multiscale-trilinear-interpolation-no-training
git pull --ff-only origin main-skalaton-multiscale-trilinear-interpolation-no-training
source "$HOME/.venvs/cosc440-$(hostname -s)/bin/activate"
python -m pip check
nvidia-smi
python -m unittest test_project test_run_lab
```

Keep the supplied dataset at `data/ct_data/chest_50.pickle`. Existing environments
need no new packages. New environments use `requirements-linux.txt`.
`COSC440_DATASET` and `COSC440_PYTHON` can override the dataset and interpreter.
`gpu_python.py` makes the installed NVIDIA libraries discoverable.

## Required run arguments

```text
bash run_lab.sh ENVIRONMENT MODEL SEED LEARNING_RATE EPOCHS [COMMENTS] [--resume]
```

All five experiment arguments are mandatory; only COMMENTS may be omitted.
`--resume` is an optional operating mode, not an experiment setting.

| Argument | Meaning and allowed values |
|---|---|
| ENVIRONMENT | `3090` on cs23002lv or `4070` on cs25049kx; a mismatch stops the launcher |
| MODEL | `baseline` or `trilinear`; each invocation trains exactly one model |
| SEED | Integer 0 through 4294967295; use `50` for the planned experiment |
| LEARNING_RATE | Finite positive number; use `0.0001` |
| EPOCHS | Positive total epoch count; use `250` |
| COMMENTS | Optional description, e.g. `changed-learning-rate`; quote text containing spaces |

Comments are lowercased, spaces/punctuation become hyphens, and path separators
cannot create directories. Use at most 80 characters after sanitising.
The name is `ENV-MODEL-sSEED-lrRATE-eEPOCHS[-COMMENT]` for both the run folder
under `runs/` and the `.log` file under `logs/`. Equivalent rates such as `.0001`
and `1e-4` produce the same name. Comments are labels; they do not change training.

For example, on the 3090:

```bash
bash run_lab.sh 3090 baseline 50 0.0001 250
bash run_lab.sh 3090 trilinear 50 0.0001 250 changed-learning-rate
```

The second command writes:
- `logs/3090-trilinear-s50-lr0.0001-e250-changed-learning-rate.log`
- `runs/3090-trilinear-s50-lr0.0001-e250-changed-learning-rate/`

On the 4070, replace `3090` with `4070`. No SSH connection is made by the script.
Use matching seed, learning rate, epochs and sampling settings for a fair model
comparison. Different GPUs with the same seed are not independent-seed replicates.

## Overnight training and resume

The launcher creates and streams the detailed experiment log itself:

```bash
nohup bash run_lab.sh 3090 baseline 50 0.0001 250 > /tmp/cosc440-baseline-launch.log 2>&1 &
echo "Process ID: $!"
tail -f logs/3090-baseline-s50-lr0.0001-e250.log
```

Run the second model after the first finishes, or on the other available machine.
Ctrl+C stops `tail`, not the background job. `nohup` survives SSH disconnection,
but not shutdown. Do not run two jobs against the same output directory.
For a short GPU check, use `1` epoch and a comment such as `smoke`; this still
uses all projections and must not be treated as the full experiment.

If interrupted, confirm the old process has stopped and repeat the exact original
arguments (including comment), adding `--resume`:

```bash
bash run_lab.sh 3090 baseline 50 0.0001 250 --resume
```

This targets 250 total epochs, not 250 additional epochs. Checkpoints must exist.
The training code checks the saved configuration and rejects incompatible resume.
An existing log is appended only with `--resume`; fresh runs cannot overwrite it.
A stale `.running` lock must only be removed after confirming no job is active.
To extend a completed run to a different total budget, use `project_main.py`
directly with its existing `--output`, matching settings, `--epochs` and `--resume`;
the launcher includes the budget in the folder name and will not guess an old path.

## Graphs and measured comparison

Use the actual folders, including comments if supplied:

```bash
python compare_runs.py runs/3090-baseline-s50-lr0.0001-e250 \
  runs/3090-trilinear-s50-lr0.0001-e250-changed-learning-rate \
  --epoch 250 --output runs/3090-comparison-s50-e250
```

This writes learning curves and reconstruction comparisons as PNG/PDF and measured
metric differences as JSON. Each training run contains `config.json`,
`runtime.json`, `metrics.csv`, ground-truth and reconstruction TIFFs, and local
checkpoints. Preserve logs and screenshots for the report. These are training-fit
metrics, not held-out generalisation. No improved accuracy is promised.

## Project files and submission status

- `project_main.py`: models, sampling, integration, training, evaluation and checkpoints.
- `geometry.py`: supplied scan geometry and dataset loader.
- `project_unit_test.py`: supplied numerical examples.
- `test_project.py`: numerical, gradient, model-update and checkpoint assertions.
- `run_lab.sh` / `run_lab.py`: argument validation, model launch and named logging.
- `test_run_lab.py`: required arguments, comment handling and command-forwarding tests.
- `compare_runs.py`: matching-settings checks, graphs and image comparison.
- `explanation.md` / `VALIDATION.md`: change explanations and executed checks.

No new full training results or completed submission are included in this branch.
The remaining archive/ files are historical baseline records and shared utilities,
not a complete submission and not evidence for the new method. Rebuild the report
from actual new measurements. Exclude datasets, environments and checkpoints from
Git. Pulling changes does not remove ignored or untracked lab outputs.

## How multiscale trilinear interpolation works

The baseline looks up one grid cell per scale. The modified model blends the eight
surrounding grid vertices at each of the same scales. Each corner gets a weight
based on the point's distance along x, y and z. For example, at the centre of a
cube, all eight weights are 1/8: corner values 0 through 7 give (0+1+2+3+4+5+6+7)/8
= 3.5. On a corner, that corner's value is returned exactly.

For resolution r, vertex coordinates span 0 through r-1. Inputs outside the volume
are clamped to its boundary. Features are concatenated and passed through the same
64-unit prediction head as the baseline. This adds no trainable parameters; it
adds eight-corner lookup and weighting work and gives continuous features across
grid faces. It is a learned representation change, not smoothing the final TIFF.
Smoother features may reduce block boundaries, but sharper structures and accuracy
must be assessed from measured MSE, SSIM and images after matched training.
