COSC440 - Mainuddin Talukdar - 81255475 - updated 22 October 2026

UPLOAD CONTENTS
The six-page report PDF, Execution_Evidence.pdf, code/, evidence/, results/,
figures/, editable report_source.txt and MANIFEST_SHA256.txt form the submission.
The companion ZIP contains all these files. If Learn has separate upload fields,
upload the report PDF separately and the ZIP as the code/evidence attachment.
Upload actual source files, not only a repository URL. Review the report and
retain the Learn submission receipt. Deadline supplied by the student: 25 October.

EXPERIMENTS
Original 3090 seed-42 baseline and attention: LR 0.1, 50 epochs each.
3090 seed-42 attention follow-up: LR 0.001, 100 epochs from a fresh start.
4070 seed-43 baseline: LR 0.001, 100 epochs.
4070 seed-43 attention: LR 0.001, 300 epochs; latest resume 200 to 300.
The matched lower-rate architecture comparison uses epoch 100 for both models.
Longer attention training is analysed separately. No trilinear results are claimed.

EVIDENCE
The original 3090 attention log was recovered from Git commit 71fc796.
A later overwritten log records a failed non-empty-output-folder attempt and is
preserved separately. 4070 attention terminal output covers epochs 201-300 only;
earlier measurements remain in its CSV and the original TIFF run folders.
analysis.json records all CSV rows, log coverage, source/config metadata and
checked snapshots. Tests are fresh Mac CPU checks, not simulated GPU runs.

REPRODUCE ON A LINUX NVIDIA MACHINE
From the submit folder, create/activate a Python 3.12 environment if needed:
  python3 -m venv .venv
  source .venv/bin/activate
  python -m pip install -r code/requirements-linux.txt
  python -m pip check
  bash code/run_experiments.sh /absolute/path/chest_50.pickle /absolute/path/NEW-results
This runs the seed-43 baseline for 100 epochs and attention for 300 epochs, then
compares the shared epoch 100. It creates new outputs and preserves old runs.

To run a single experiment, activate your environment and run from code/:
  python gpu_python.py project_main.py --dataset /absolute/path/chest_50.pickle --model attention --output /absolute/path/NEW-run --seed 42 --learning-rate 0.001 --epochs 100 --device gpu
Use --learning-rate 0.1 --epochs 50 for the original experiment settings.
All reproduction commands specify the learning rate explicitly. The historical
source version is included for provenance; the only difference is its LR default.

VERIFY WITHOUT GPU
  cd code
  python -m unittest test_project
  python project_unit_test.py
The supplied test prints expected examples; the additional tests use assertions.

PLOTTING
compare_runs.py needs full original run folders including reference and snapshot
TIFFs, which are not duplicated in this package. For the updated paired runs:
  python code/compare_runs.py /path/4070-baseline-s43 /path/4070-attention-s43 --epoch 100 --output /path/NEW-comparison
Existing report-ready graphs and original CSVs are included in this archive.

EXCLUSIONS
No supplied datasets, full TIFF volumes, checkpoints, environments, credentials
or caches are uploaded. Course data is supplied separately. Checkpoints are
only needed to resume an old run, not to train from scratch.
