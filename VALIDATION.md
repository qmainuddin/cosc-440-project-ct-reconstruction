# Fresh experiment validation - 22 October 2026

- CLI defaults verified: seed 50, learning rate 0.0001, epochs 250.
- python -m unittest test_project: all five tests passed on the Mac CPU.
- Supplied project_unit_test.py executed successfully; its numerical fixtures are unchanged.
- bash -n run_lab.sh passed. The launcher requires explicit settings for both hosts.
- Four launcher tests passed: argument validation, comment naming, command forwarding, and process/log handling (including resume, collision protection and nonzero exit status).
- No fresh full GPU training has been performed. No new reconstruction quality is claimed.
- No new local runs/ folder was created during validation; diagnostic output went to /tmp.
- Historical validation and completed submission materials are in archive/.
- The complete previous tracked runs and local analysis files are on remote main-backup.

The tests check sampling, integration/gradients, grid indexing, model updates,
checkpoint restoration and metric/display edge cases. Passing them does not
predict final reconstruction quality for the new learning rate.
