# Fresh experiment reset: seed 50, learning rate 0.0001

The previous main was preserved and pushed as main-backup before deleting runs/.
Historical code, reports and measurements keep their true original settings in
archive/ and the backup branch. They are not fresh experiment results.

| File / block | Lines | Change and purpose | Input -> output |
|---|---|---|---|
| project_main.py: parse_args | 348-349 | Learning-rate default 0.0001; seed default 50 | CLI without overrides -> fresh settings |
| project_main.py: parse_args | 341 (1 line) | Epoch default increased from 50 to 250 for baseline and attention | CLI without an epoch override -> 250 total training epochs |
| test_project.py: model/checkpoint test | 42, 46, 65 | Seed 50 and Adam learning rate 0.0001 in both original and restored optimiser | Small synthetic training -> finite updates and matching restored predictions |
| run_lab.sh | Entire file | Smoke, matched training, resume and comparison for both known hosts; fixed seed 50 and rate 0.0001 | Course dataset and action -> new host-specific run folders, logs and figures |
| README.md | Entire file | Fresh setup, commands, outputs and archival status | Instructor/student -> reproducible experiment instructions |
| VALIDATION.md | Entire file | Records reset verification; older validation moved into archive | Executed checks -> honest validation record |
| .gitattributes | Archive rule | Preserve historical file bytes and CSV line endings | Git checkout -> unchanged historical checksums |

project_main.py remains 367 lines. Its learning-rate, seed and epoch default argument lines changed;
model architecture, objective and sampling budget are unchanged.
The root test_project.py remains 80 lines. Its numerical geometry fixtures and the
supplied project_unit_test.py are unchanged because those numbers are not seeds or
learning-rate settings.

run_lab.sh validates any existing configuration before reuse/resume. Both hosts
use seed 50; only the output prefix differs. Fresh baseline/attention folders use
the suffix s50-lr1e-4, preventing accidental reuse of earlier experiments. Logs
append when training resumes. Training and resume target 250 total epochs;
completed-run checks require at least 250 epochs and epoch_0250.tiff.
The smoke check remains one epoch. The launcher generates a comparison at epoch 250.

The previous submission was moved to archive/submit and its ZIP to archive/.
No active submit folder claims results for these unrun settings. The backup branch
retains original complete runs, including their TIFF reconstructions. The local
main runs folder was removed; a lab pull cannot remove ignored/untracked files on
another machine, so such leftovers must not be mistaken for fresh results.
