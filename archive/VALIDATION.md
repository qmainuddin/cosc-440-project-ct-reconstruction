# Validation record

Validation date: 20 October 2026 (Pacific/Auckland).

## Environment

Mac ARM64 CPU; Python 3.11; TensorFlow 2.21.0; Keras 3.15.1; NumPy 2.4.6.
Testing used a temporary isolated environment, not either lab machine. Linux Python 3.12 GPU
execution of the complete project remains to be verified by the user. The lab GPU launcher
pattern had previously passed the user's basic calculations on RTX 3090 and RTX 4070.

## Executed checks

- Python compilation of all delivered Python files.
- Five assertion-based test methods in `test_project.py`, covering ray endpoints/float64,
  analytical integrals and derivatives, all supplied indexing examples, boundary clipping,
  baseline and attention model updates, attention-weight updates, checkpoint predictions and
  optimizer iteration restoration, zero-volume display, and identical-volume metrics.
- Unmodified supplied `project_unit_test.py`: executed successfully; printed values match
  the supplied expected examples, including integral 0.12000003 and all index arrays.
- Real Chest dataset, baseline, one epoch followed by checkpoint resume to epoch two.
- Real Chest dataset, attention model, two epochs.
- Both diagnostic runs use depth 3, 16 rays/projection, 8 sample points, two projections/epoch,
  seed 42 and query batch 8192. They reconstruct the full 128 x 128 x 128 Chest volume and
  produce finite MSE/SSIM, TIFFs and checkpoints. They are NOT final experiments.
- A separate uninterrupted two-epoch baseline run produced exactly equal numerical metrics
  (excluding time) and a byte-equivalent array of raw voxel predictions to the resumed run.
- Saved-checkpoint evaluate-only mode matched the final epoch's SSIM/PSNR/display-MSE.
- Comparison script produced both PNG/PDF figure types and comparison JSON. PNGs were opened
  and visually checked for legibility, matching slices and labels.

## Remaining verification

- Full depth-8/192-point/2048-ray runs on Linux GPUs, memory use and training duration.
- Full 50-epoch convergence and whether Model 2 improves any metric.
- Repeat seeds, optional additional datasets, and interpretation of scientific results.
- Final report, real execution screenshots, Learn deadline/format and final submission.

The very coarse diagnostic reconstructions are expected after four tiny updates and are not
submission-quality results. Do not use their metric differences as evidence for the hypothesis.

Final restart guards were also exercised: a valid no-op resume succeeds; an attempted fresh
run into a nonempty folder is rejected; resume with a changed seed is rejected.
