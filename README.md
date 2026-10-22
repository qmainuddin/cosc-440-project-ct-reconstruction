# Multiscale trilinear interpolation branch

Branch: `multiscale-trilinear-interpolation`.

The **submit/** folder is a frozen, ready-to-review submission based on the five
completed baseline/attention experiments. Its own source matches those experiments.
Root-level code adds the new `trilinear` model, which has not yet been trained for
100 epochs on a lab GPU. Do not replace the report's measured results with predictions
about this method. Its full results can be incorporated after training.

## What changes

`--model trilinear` retains the baseline's eight grid tables, scalar features and
64-unit dense head, but replaces single-cell lookup with the weighted sum of eight
surrounding grid vertices at every scale. For resolution r, vertices span the unit
cube, so the grid coordinate is `scaled * (r - 1)`. Coordinates clamp to [0,1].
The eight weights are products of distances along x, y and z and sum to one.
The model is continuous across internal grid faces. It has the same trainable
parameter count as baseline (19,174,601 at depth 8, embedding width 1).

This is a change inside the trainable spatial representation, not smoothing a
finished TIFF. Its hypothesis is that continuous multiscale features reduce grid
artefacts and improve SSIM at a matched budget. Smoothing can also lose detail;
there is no guarantee of improvement. Eight lookups per scale cost more computation
and temporary memory than one. Baseline and attention behaviour are unchanged.

## UC machines: setup once before starting either job

From a lab terminal with access to the repository (not necessarily from your Mac):

```bash
cd "$HOME/Desktop/cosc-440-project/cosc-440-project-ct-reconstruction"
git status --short
git fetch origin
git switch multiscale-trilinear-interpolation
git pull --ff-only origin multiscale-trilinear-interpolation
```

If Git reports local modifications, preserve them; do not use reset/clean to discard
work. Both machines share your home directory, so perform the branch switch once
and avoid pulling/editing code while either job runs. The two jobs use distinct
host-specific output folders. Never run two jobs against the same output folder.
Keep the supplied dataset at `data/ct_data/chest_50.pickle` (it is not in Git).

On each machine, confirm its hostname and environment:

```bash
hostname
source "$HOME/.venvs/cosc440-$(hostname -s)/bin/activate"
python -m pip check
nvidia-smi
bash run_trilinear_lab.sh smoke
```

No new dependencies are introduced. If a dependency is missing, install it in the
virtual environment with `python -m pip install -r requirements-linux.txt`.
The smoke action runs the eight assertion tests and a full-depth, full-ray-batch
trilinear epoch using two projections. This checks real GPU memory capacity before
the longer run. Its timestamped output is diagnostic, not a completed experiment.
If the smoke test fails, stop and retain the full error; do not silently lower the
ray budget for just one model. If fewer rays are necessary, restart a matched pair
with identical changed settings using the manual CLI below.

## Start the matched experiment (same command on either machine)

```bash
mkdir -p logs
nohup bash run_trilinear_lab.sh train > "logs/trilinear-$(hostname -s)-launch.log" 2>&1 &
echo "Training process ID: $!"
tail -f "logs/trilinear-$(hostname -s)-launch.log"
```

Ctrl+C stops `tail`, not the background job. The launcher uses the correct existing
host-specific virtual environment automatically. It needs no tmux or new system
software. Keep the computer powered on; nohup protects against SSH disconnection,
not machine shutdown. Do not start another job if this one is still running.

| Host | Seed | Baseline | New model |
|---|---:|---|---|
| cs23002lv (3090) | 42 | Train fresh at LR .001, 100 epochs in `runs/3090-baseline-s42-lr0001` | `runs/3090-trilinear-s42-lr0001`, 100 epochs |
| cs25049kx (4070) | 43 | Reuse matching `runs/4070-baseline-s43`, already 100 epochs at LR .001 | `runs/4070-trilinear-s43-lr0001`, 100 epochs |

The launcher checks the reused baseline's model, seed, dataset hash, learning rate,
depth, embedding width and ray/point budgets. It refuses an incomplete existing run
unless `resume` is requested. It never uses the LR .1 baseline as the matched control
for LR .001 trilinear. Comparison figures and JSON are generated automatically.

## Monitor, resume and inspect results

```bash
nvidia-smi
# Choose the matching host's file:
tail -n 3 runs/3090-trilinear-s42-lr0001/metrics.csv
tail -n 3 runs/4070-trilinear-s43-lr0001/metrics.csv
```

After confirming the earlier process has stopped, resume interrupted training:

```bash
nohup bash run_trilinear_lab.sh resume > "logs/trilinear-$(hostname -s)-resume.log" 2>&1 &
```

Resume requires the local checkpoints; they are intentionally excluded from Git.
The launcher skips already completed matching runs, resumes incomplete runs, and
starts missing runs. It appends training output to logs instead of overwriting it.
If a stale `.running` lock remains after a killed process, verify no training job
is alive before removing that lock; do not remove it from an active run.

Regenerate figures after completion:

```bash
bash run_trilinear_lab.sh compare
```

Outputs: `runs/3090-trilinear-comparison-s42/` or
`runs/4070-trilinear-comparison-s43/`, containing learning curves, three-slice
comparisons (PNG/PDF) and numerical differences (JSON). Every run records epoch
MSE/SSIM/PSNR, configuration, runtime, source hashes, raw/display TIFFs and checkpoints.
Judge improved SSIM and anatomical detail as well as projection MSE. A single seed
or lower training loss alone is not proof of a generally better method.

## Manual equivalent (activated environment, new output folder)

```bash
python gpu_python.py project_main.py --model trilinear   --dataset data/ct_data/chest_50.pickle --output runs/NEW-trilinear   --epochs 100 --learning-rate 0.001 --seed 43 --depth 8 --embedding-dim 1   --n-rays 2048 --n-points 192 --query-batch 8192 --save-every 10 --device gpu
python compare_runs.py runs/4070-baseline-s43 runs/NEW-trilinear   --epoch 100 --output runs/NEW-comparison
```

To use another location or interpreter, set `COSC440_DATASET` to the absolute
dataset path and/or `COSC440_PYTHON` to the desired Python executable before running
the helper. The helper only auto-configures the two known lab hostnames.

## Return results for report updates

Keep the new run directories (CSV, configuration/runtime and TIFF outputs), their
logs, comparison figures, and a screenshot showing the completed job. Do not add
datasets or checkpoints. Example on the 4070, after its job finishes:

```bash
git add runs/4070-trilinear-s43-lr0001 runs/4070-trilinear-comparison-s43
git add logs/4070-trilinear-s43-lr0001.log
git commit -m "Record 4070 trilinear experiment"
git push origin multiscale-trilinear-interpolation
```

On the 3090 also add `runs/3090-baseline-s42-lr0001` and its log. Because home is
shared, coordinate commits/pushes sequentially after the jobs finish; do not switch
branches while either job runs. The report in submit/ is not automatically altered.

## Validation

Eight tests cover the original operations, both existing models and trilinear,
affine-field interpolation, exact corner values, eight-corner gradient weights,
clamping, continuity, equal parameter count and checkpoint round trips. Local
CPU smoke training also checks the full output/evaluation/resume/plot pipeline.
These checks are not evidence of full GPU reconstruction quality.

---

# Earlier baseline and attention workflow

The following sections describe the original project and historical experiments.
Use the branch-specific instructions above for the new trilinear study.

# COSC440 CT reconstruction project

This implementation completes the supplied baseline and adds **scale attention** as Model 2.
It includes real training, per-epoch measurements, TIFF export, checkpoints, evaluation and figures.
It does not contain invented experiment results or a completed assessed report. Run the full
experiments in the lab and write your interpretation of the measured results. Follow the course's
rules for assistance and attribution. The assessment brief controls marking; complexity marks
and a score above 90% cannot be guaranteed.

## Files

- `project_main.py`: both models, mathematical operations, training/evaluation CLI.
- `geometry.py`: unchanged supplied geometry/dataset loader.
- `project_unit_test.py`: unchanged supplied illustrative test (prints expected examples).
- `test_project.py`: numerical, gradient, model-update and checkpoint assertions.
- `gpu_python.py`: the library-search workaround verified on your two lab GPUs.
- `compare_runs.py`: comparison figures and numerical summary from saved measurements.
- `requirements.txt`, `requirements-linux.txt`: project dependencies.
- `starter_original/`: unmodified source backup, for reviewing changes.
- `explanation.md`: annotated change guide with exact line ranges and counts.
- `VALIDATION.md`: what was actually tested locally and what remains for the lab.

## Scientific design

A location is mapped to learned features in grids of resolutions 2, 4, 8, ... . The baseline
concatenates the scale features and uses the supplied 64-unit dense layer and scalar output.
Model 2 projects each scale feature to a token, adds a learned scale identity, performs
multi-head self-attention **across scales at that same spatial point**, uses a residual connection
and layer normalization, then decodes to attenuation. It does not attend across different points
or adjacent rays. It is a new layer type and changes how scales interact; it is not simply one
extra dense layer. All added attention parameters are trainable and covered by a gradient test.

Suggested hypothesis to evaluate (not a result): scale interaction may improve reconstruction
SSIM at a fixed number of epochs by combining coarse structure with fine-scale information.
It may also fail, overfit or converge more slowly. Both results are valid evidence to discuss.
The attention model has more parameters/operations. The experiment compares complete designs;
it does not isolate attention from parameter count. An additional parameter-matched control
would be an optional future experiment, not a substitute for the required significant change.

## Setup on your Mac

Use Python 3.11 or 3.12. From this directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest -v test_project
python project_unit_test.py
```

The Mac CPU can run small tests. A full default experiment is intended for your lab GPUs.
Do not install NVIDIA packages on the Mac. Data remains in `data/ct_data/` and is ignored by Git.
Only load the trusted course pickle files (pickle is not a safe format for unknown downloads).

## GitHub workflow

Make a private repository for this release directory. Replace the URL placeholder below with
your real repository URL. No remote has been created or pushed by this implementation.

```bash
git init
git add .
git status --short
```

Verify no datasets, run volumes, checkpoints or environments are staged, then:

```bash
git commit -m "Implement CT reconstruction baseline and scale attention experiment"
git branch -M main
git remote add origin https://github.com/YOUR_ACCOUNT/YOUR_PRIVATE_REPOSITORY.git
git push -u origin main
```

On the lab machine, clone into a new project directory of your choice. For a private repository,
use your university-approved GitHub authentication method; an SSH login password is not a GitHub
HTTPS password. Do not put access tokens into committed files or URLs.

```bash
git clone https://github.com/YOUR_ACCOUNT/YOUR_PRIVATE_REPOSITORY.git cosc440-project
cd cosc440-project
git rev-parse HEAD
```

Transfer the course datasets separately using approved university storage, scp or rsync. If your
home is shared, one dataset copy is sufficient. Keep it at a stable path and supply `--dataset`.
Do not pull updates into a working copy while it is training. Use separate clones if needed;
run folders must always be unique even when home storage is shared. For later code updates,
run `git pull --ff-only` when that working copy has no running experiments.

## Linux environments and commands

The user has already verified TensorFlow 2.21.0 with actual GPU calculations on:

| Host | GPU | GPU memory | RAM | Python / OS |
|---|---|---:|---:|---|
| cs23002lv | RTX 3090 | 24 GB | 62 GiB | 3.12.3 / Mint 22 |
| cs25049kx | RTX 4070 | 12 GB | 31 GiB | 3.12.3 / Mint 22 |

Direct SSH/overnight jobs are permitted according to the user. No remote connection was made by
Codex. Both machines share the user's network home; the previously displayed 48 GB free is not
48 GB per machine. Keep only useful volume snapshots and monitor storage.

On the 3090:

```bash
source "$HOME/.venvs/cosc440-cs23002lv/bin/activate"
```

On the 4070:

```bash
source "$HOME/.venvs/cosc440-cs25049kx/bin/activate"
```

From the project directory in the activated environment:

```bash
python -m pip install -r requirements.txt
python -m pip check
python gpu_python.py -m unittest -v test_project
python gpu_python.py project_unit_test.py
python -m pip freeze > requirements-installed-local.txt
```

Keep that final file with experiment records; do not replace the supplied requirements with a
Mac-specific freeze. For a NEW Linux environment only, use `requirements-linux.txt`, which
includes the CUDA extra. Existing environments already have the NVIDIA packages.

## First real-data checks (diagnostics, not report experiments)

Commands assume the Chest data is in `data/ct_data/chest_50.pickle`; otherwise add `--dataset`
with the actual path to every command. Run from this project directory:

```bash
python gpu_python.py project_main.py --model baseline --output runs/smoke-baseline --epochs 1 --depth 3 --n-rays 16 --n-points 8 --max-projections 2 --device gpu
python gpu_python.py project_main.py --model attention --output runs/smoke-attention --epochs 1 --depth 3 --n-rays 16 --n-points 8 --max-projections 2 --device gpu
```

A smoke test is a small end-to-end check. These runs use only two projections per epoch and
small models: NEVER present their metrics as the final 50-epoch experiment.

## Full paired experiment

Start a persistent terminal **on the machine doing training**:

```bash
tmux new -s cosc440
```

Activate the appropriate environment again inside tmux, enter the project directory, then:

```bash
set -o pipefail
python gpu_python.py project_main.py --model baseline --output runs/3090-baseline-s42 --seed 42 --learning-rate 0.1 --device gpu 2>&1 | tee baseline-s42.log
python gpu_python.py project_main.py --model attention --output runs/3090-attention-s42 --seed 42 --learning-rate 0.1 --device gpu 2>&1 | tee attention-s42.log
```

These are historical reproduction commands; use new output folders if the originals already exist.
They explicitly use learning rate 0.1. The current CLI default is 0.001. Depth 8, embedding width 1,
192 sample points, 2048 rays per projection and 50 epochs remain defaults. Attention costs
more memory than the baseline. First try a one-epoch run of EACH full-sized model in separate
trial folders to establish memory use. If a model runs out of memory, choose a smaller common
`--n-rays` (e.g. 512) and restart BOTH comparison runs in NEW folders with that setting.
This changes the training budget and must be reported. Lower `--query-batch` separately if
volume reconstruction runs out of memory; it only changes inference batching.
Do not assume the original brief's runtime estimate applies to the attention model or per-epoch evaluation.

Detach with Ctrl+B then D. Reconnect with `tmux attach -t cosc440` on the same host.
In another terminal, use `nvidia-smi`, `tail -n 5 runs/3090-baseline-s42/metrics.csv`, and
`df -h "$HOME"` to monitor GPU, progress and space. Do not edit run files during training.

On the 4070, repeat BOTH models with seed 43 in folders `runs/4070-baseline-s43` and
`runs/4070-attention-s43`. Keep the same common settings. Do not claim algorithmic speedups
by comparing elapsed time on different GPU models. Same seeds do not guarantee bitwise
agreement across hardware. Seeds vary initial weights and the sampled training rays.

## Resume, saved-model evaluation, and outputs

Each epoch saves model weights, optimizer state, optimizer iteration and completed epoch.
Projection/ray choices are derived from seed and epoch, so they do not reset on resume.
If interrupted mid-epoch, that unfinished epoch is repeated from the last completed checkpoint.
Use the SAME original settings (including any non-default flags):

```bash
python gpu_python.py project_main.py --model baseline --output runs/3090-baseline-s42 --seed 42 --learning-rate 0.1 --device gpu --resume
```

Use `--epochs 100 --resume` to extend a run to 100 total epochs, not 100 extra epochs. The
original `config.json` remains the initial configuration; `runtime.json` records the latest target.
A `.running` lock prevents two processes using the same run folder. After a hard kill or reboot,
remove a stale lock ONLY after confirming no process is still using that folder.

Reconstruct and evaluate a saved checkpoint without further training:

```bash
python gpu_python.py project_main.py --model baseline --output runs/3090-baseline-s42 --seed 42 --learning-rate 0.1 --device gpu --evaluate-only
```

Each run contains:

- `config.json`: original settings and dataset SHA-256 identity.
- `runtime.json`: host, versions, source hashes, parameter count, device and target epoch.
- `metrics.csv`: sampled projection MSE and NMSE, mean slice SSIM/PSNR, display-volume MSE,
  separate training/evaluation wall times, EVERY epoch.
- `ground_truth.tiff`: supplied normalized reference volume in [Z,X,Y] order.
- `epoch_NNNN.tiff`: displayed reconstruction every ten epochs and at the final epoch.
- `epoch_NNNN_raw.tiff`: raw float32 attenuation prediction at those epochs.
- `checkpoints/`: latest two recoverable checkpoints.
- `evaluation.json` and `evaluation_raw.tiff` after `--evaluate-only`.

The original train loop optimized normalized MSE. This implementation logs both raw MSE and
that normalized objective so they cannot be confused. The SSIM/PSNR convention follows the
starter: nonnegative prediction, divide by its own maximum, convert to uint8; reference also
converted to uint8. SSIM is averaged over top-down slices, not a 3D SSIM calculation.
These are scale-normalized image comparisons, NOT absolute physical attenuation accuracy.
Reconstruction uses ground-truth positions for evaluation but ground-truth voxel values do NOT
enter the training loss. No held-out split is claimed: the model fits the supplied training
projections. An evaluation command re-evaluates a checkpoint; it does not imply unseen data.

## Generate the report figures

After a paired run has finished at the same epoch:

```bash
python compare_runs.py runs/3090-baseline-s42 runs/3090-attention-s42 --output runs/comparison-s42
```

This generates `learning_curves.png/.pdf`, `reconstruction_comparison.png/.pdf`, and
`comparison.json`. Matching seed, dataset, sampling and optimizer settings are checked.
Figures show projection MSE and SSIM by epoch, and three matching reconstruction slices.
They satisfy the two figure TYPES requested by the brief; write captions and interpretation
using the actual results. The comparison script does not manufacture a successful hypothesis.
Copy selected reports/images/logs back to the Mac through the approved network route. No lab
hostname accessibility from the Mac is assumed. Avoid copying all checkpoints unnecessarily.

## Submission checklist

1. Complete full baseline and modified experiments; retain their precise settings and code.
2. Write Experimental Design and Methods: model, change, reasons, hypothesis, settings/ranges,
   evaluation conditions and comparison fairness (4 marks).
3. Write Results with at least two figures, numerical differences and positive/negative/neutral
   trends (4 marks).
4. Write Conclusions: answer the hypothesis, explain why, limitations, implications, future
   questions (4 marks).
5. Include readable script/screenshots of both models actually running with output (4 marks).
6. Verify baseline correctness (5), modified-model correctness (5), and explain the significant
   change/complexity (4). Total 30 marks. A negative result is not automatically a failed project.
7. The brief estimates 3-4 pages including graphs at a minimum; check Learn for format, deadline,
   later instructions, and assistance/citation requirements. Cite any external ideas/code used.
8. Upload report, ACTUAL code and downloaded notebooks if used, plus execution evidence to
   Learn. A GitHub link alone is not accepted. Do NOT submit the supplied datasets or environments.
9. Recommended supporting files: README, dependencies, metrics CSV, configurations and source
   explanations. Check permitted upload formats and file-size limits on Learn.
10. Download/reopen the submitted files, confirm they match the experiments, and keep submission
    confirmation. Do not claim the local smoke test is a completed lab experiment.

## References for implementation

- Supplied COSC440 Project 2026 brief, research_ct_only slides and research slides.
- TensorFlow checkpoint API: https://www.tensorflow.org/api_docs/python/tf/train/Checkpoint
- Keras attention API: https://www.tensorflow.org/api_docs/python/tf/keras/layers/MultiHeadAttention
- TensorFlow GPU installation: https://www.tensorflow.org/install/pip
