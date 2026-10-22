# Explicit experiment launcher

Python defaults remain seed 50, learning rate 0.0001 and 250 epochs.
`run_lab.sh` is a five-line entry point forwarding arguments to `run_lab.py`.
The launcher requires environment, model, seed, learning rate and epoch count;
comments are optional. A single command launches a single model.

`run_lab.py` parses and validates settings, sanitises comments, constructs unique
run/log names, verifies the machine, interpreter and dataset, then streams merged
stdout/stderr to the terminal and log. The child process exit code is preserved.
Fresh output/log collisions fail; explicit resume appends the same log and delegates
checkpoint/configuration validation to the training code. Filenames include the
epoch budget, so extending a run uses the direct Python command and original output.

`test_run_lab.py` checks missing/invalid arguments, equivalent learning-rate names,
safe comments, both environment/model choices, and exact forwarding of custom
settings and resume. `README.md` documents mandatory settings and runnable examples.
Remaining baseline records in archive/ retain their original settings.
This partial archive is not a complete submission.

## Multiscale trilinear implementation

`TrilinearVoxelModel.interpolate_scale` maps positions to grid-vertex coordinates,
gathers eight neighbouring embedding vectors, and returns their weighted sum.
Input shape [N,3] becomes features [N,E] per scale; weights are nonnegative and sum
to one. `embeddings` applies this at each scale, while the inherited prediction
head combines the features. `build_model` and the CLI support baseline/trilinear.
The default seed, learning rate and epoch budget remain 50, 0.0001 and 250.

`test_project.py` checks affine-field interpolation, exact corners, boundary
clamping, gradients to all eight corners, continuity and equal parameter count,
as well as training updates/checkpoint restoration for both models.
`compare_runs.py` compares matching baseline/trilinear runs, using appropriate
plot labels and JSON keys. `run_lab.py` accepts either model with explicit settings.

## Code locations

| File and block | Lines | Purpose |
|---|---|---|
| project_main.py: class TrilinearVoxelModel | 87-115 (29 lines) | Interpolate eight corners at every scale |
| project_main.py: def build_model | 116-126 (11 lines) | Select and initialise either supported model |
| run_lab.py: def parse_args | 16-40 (25 lines) | Validate the required settings and optional comments |
| run_lab.py: def run_name | 41-46 (6 lines) | Create the run and log name |
| run_lab.py: def command | 47-57 (11 lines) | Forward settings to GPU training |
