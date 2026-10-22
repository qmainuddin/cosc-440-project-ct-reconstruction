# Validation - 22 October 2026

- All 12 tests passed on the Mac CPU: eight numerical/model tests and four launcher tests.
- Interpolation checks cover affine fields, corners, clamping, gradients, continuity,
  unchanged parameter count, model updates and checkpoint restoration.
- Both Python model choices default to seed 50, learning rate 0.0001 and 250 epochs.
- Launcher tests verify mandatory settings, safe optional comments, parameter forwarding,
  resume log append, fresh-log collision protection and propagation of failure exit codes.
- bash -n run_lab.sh and git diff --check passed.
- A synthetic baseline/trilinear comparison generated all five expected figure/JSON
  outputs and rejected a mismatched seed. These temporary fixtures are not research results.
- Removed-method names/content were checked across remaining tracked files and paths;
  affected PDF content and screenshots were inspected before selective archive removal.
- Remaining archive files are historical baseline records/shared utilities, not a
  complete submission. No new full GPU training or reconstruction quality is claimed.
