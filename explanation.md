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
Historical source and measurements in archive/ retain their original settings.
