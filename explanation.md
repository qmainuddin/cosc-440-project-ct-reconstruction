# Explanation of the COSC440 implementation

All paths below are relative to `final-codes/`. Source files are inside `COSC440Project_StudentRelease_2026_v4/`.
Line numbers describe the delivered version. Counts are inclusive physical source lines, including comments, docstrings and blank lines inside each block; nested blocks overlap their parent and must not be added twice.

## What is complete and what remains

The code implements baseline reconstruction, a significant scale-attention modification, tests, training, checkpoints, evaluation, graphs, GPU launching and lab/submission instructions. Full 50-epoch GPU experiments, the final assessed report, and submission are still tasks for the student. No improvement, full-mark outcome, or unrun result is claimed.

## Plain-language pipeline

1. A ray is a straight X-ray path. Sample several positions along it, like placing markers along a road.
2. At each position, look up learned features on coarse and fine grids: like city/street/house levels of an address.
3. The baseline combines these features; Model 2 lets scales exchange information through attention.
4. Predict attenuation (how strongly material weakens the beam). Average adjacent samples and multiply by distance to estimate a whole-ray total.
5. Compare totals with supplied log-transformed projections. Backpropagation changes learned values to reduce the normalized squared error.
6. Query a full grid of positions to produce CT slices. Compare them with the reference without using voxel ground truth in the training objective.

## Significant change and hypothesis

Model 2 uses multi-head self-attention across scale tokens at each spatial position, with learned scale identity, residual connection and layer normalization. It can learn input-dependent interactions between coarse and fine information. This is a new layer type, not only an extra dense layer or learning-rate adjustment. It also adds parameters, so observed effects cannot be attributed to attention alone without further controls.

Testable hypothesis: at matched dataset, epoch/ray budget and optimizer settings, scale attention may yield higher reconstruction SSIM. This remains a hypothesis; failed or neutral outcomes must be reported honestly. Check the brief/tutor if uncertain about complexity credit.

## File-level changes and counts

| File | Total lines | Added lines | Removed lines | Status / reason |
|---|---:|---:|---:|---|
| `project_main.py` | 367 | 350 | 220 | Modified starter file |
| `geometry.py` | 176 | 0 | 0 | Unchanged supplied file |
| `project_unit_test.py` | 71 | 0 | 0 | Unchanged supplied file |
| `test_project.py` | 80 | 80 | 0 | New supporting file |
| `gpu_python.py` | 26 | 26 | 0 | New supporting file |
| `compare_runs.py` | 76 | 76 | 0 | New supporting file |
| `.gitignore` | 10 | 10 | 0 | New supporting file |
| `requirements.txt` | 6 | 6 | 0 | New supporting file |
| `requirements-linux.txt` | 2 | 2 | 0 | New supporting file |
| `README.md` | 263 | 263 | 0 | New supporting file |
| `VALIDATION.md` | 43 | 43 | 0 | New supporting file |

Added/removed counts use a line-based diff against the preserved `starter_original/` files. New files count all lines as additions. The original geometry and illustrative tests are byte-for-byte preserved. Dataset files are not edited.

## Code block guide: input, output, behavior and reason

### `project_main.py`

| Block | Lines | Count | Input | Output / behavior | Reason |
|---|---|---:|---|---|---|
| Imports | 2-18 | 17 | Installed packages | Available APIs | Numerical operations, runtime recording or plotting needed by this file |
| `rays_to_points` | 21-30 | 10 | Ray origins/directions [R,6], sample count S, near/far limits. | Positions [R,S,3] and parameters [S]; preserves float32/float64 and includes endpoints. | Implements required TODO; vectorization avoids Python loops over rays. |
| `ray_attenuation` | 33-48 | 16 | Attenuations [R,S], ordered parameters [S], direction magnitudes [R], near/far. | One line integral [R]; gradients flow back to attenuation predictions. | Uses neighboring averages times spacing and ray magnitude. Constant edge extension handles interior-only samples; uniform endpoint samples have zero edge contribution. |
| `MultiScaleVoxelModel.__init__` | 53-67 | 15 | Physical cube size, number of grid scales, feature width. | Learned grids plus the original 64-unit SiLU dense layer and scalar decoder. | Preserves baseline architecture. Explicit named grid attributes ensure checkpoint tracking; validates shape/memory-related inputs. |
| `MultiScaleVoxelModel.get_flattened_position` | 69-74 | 6 | Normalized positions [N,3] and zero-based scale depth. | Integer lookup indices [N], using x + r*y + r^2*z. | Completes required TODO. Clipping protects boundary indices. Depth 0 uses resolution 2, not 1. |
| `MultiScaleVoxelModel.embeddings` | 76-80 | 5 | Physical positions [N,3]. | One [N,E] feature tensor per scale. | Factors the shared normalization and lookup into one method so both models use identical coordinates. |
| `MultiScaleVoxelModel.call` | 82-84 | 3 | Positions [N,3], optional training flag. | Attenuation [N,1]. | Concatenates scales exactly as the starter baseline before the dense decoder. |
| `ScaleAttentionModel.__init__` | 89-100 | 12 | Baseline settings, token width, number of attention heads. | Trainable projection, scale identities, attention and normalization layers. | Creates a structural Model 2 change. Token width must divide evenly into heads. Default width 8 and two heads keep the added component small. |
| `ScaleAttentionModel.call` | 102-108 | 7 | Positions [N,3]. | Attenuation [N,1], same interface as baseline. | Features become [N,D,E], project to [N,D,T], interact across D scales, pass through residual/normalization, then flatten/decode. No mixing between different points. |
| `build_model` | 111-119 | 9 | Saved/current configuration and physical cube size. | Built baseline or attention model. | Creates weights before optimizer construction/restoration; model choice is explicit and reproducible. |
| `configure_device` | 122-131 | 10 | cpu, auto, or gpu string. | TensorFlow execution device string. | Enables GPU memory growth; explicit gpu mode fails instead of silently running full training on CPU. |
| `make_train_step` | 134-152 | 19 | Model, optimizer and sample count. | Compiled function taking rays/targets/near/far and returning MSE/NMSE. | One update per sampled projection. Finite-loss and disconnected-gradient checks catch failures; the whole epoch is no longer captured in one giant graph. |
| `make_train_step.step` | 136-151 | 16 | Rays [R,6], target projections [R], limits. | Two scalar losses and updated model/optimizer state. | Uses differentiable integration and optimizes normalized MSE as in the starter; also returns unnormalized MSE for unambiguous reporting. |
| `train_epoch` | 155-172 | 18 | Compiled update, dataset, settings and epoch. | Mean sampled MSE/NMSE across selected projections. | NumPy RNG seeded by seed+epoch gives model-independent projection/ray selection and reproducible restart at epoch boundaries. |
| `reconstruct` | 175-185 | 11 | Model, dataset voxel coordinates and query batch size. | Raw float32 volume [Z,X,Y]. | Queries all supplied voxel positions in bounded chunks. Training batch size and reconstruction query batch are independent. |
| `display_volume` | 188-192 | 5 | Raw float volume. | Nonnegative maximum-normalized uint8 volume, or all zeros. | Matches starter visualization and avoids division by zero for an all-zero/negative prediction. |
| `volume_metrics` | 195-209 | 15 | Raw [Z,X,Y] prediction and supplied reference [X,Y,Z]. | SSIM, PSNR dB and display-scaled MSE. | Uses matching starter-style display scale. Computes per-slice SSIM/PSNR then averages; does not claim absolute attenuation accuracy or 3D SSIM. |
| `write_json` | 212-216 | 5 | Destination Path and JSON-serializable mapping. | JSON file atomically replaced. | A temporary sibling file prevents partial JSON if writing is interrupted. |
| `dataset_digest` | 219-225 | 7 | Dataset file path. | SHA-256 hexadecimal identity. | Confirms dataset content is the same even when absolute paths differ across Mac/Linux. |
| `runtime_info` | 228-240 | 13 | Current runtime/source directory. | Host, Python/framework versions, Git commit when available, source SHA-256 hashes. | Makes experiment provenance traceable even outside a Git repository. |
| `run` | 243-332 | 90 | Parsed CLI options. | CSV, config/runtime JSON, TIFFs and recoverable checkpoints in a unique run folder. | Coordinates CPU dataset placement, model updates, evaluation every epoch, resume and evaluate-only modes. Rejects mismatched resume settings and active-folder collisions. Preserves baseline geometry and objective. |
| `parse_args` | 335-363 | 29 | Command-line options. | Validated Namespace. | Exposes settings previously hardcoded in main. Defaults preserve baseline settings; --max-projections is explicitly diagnostic. |
| Script entry point | 366-367 | 2 | CLI invocation | Starts the appropriate runner | Importing the file does not start training |

### `test_project.py`

| Block | Lines | Count | Input | Output / behavior | Reason |
|---|---|---:|---|---|---|
| Imports | 2-8 | 7 | Installed packages | Available APIs | Numerical operations, runtime recording or plotting needed by this file |
| `ProjectTests.test_ray_endpoints_and_dtype` | 12-19 | 8 | A known float64 ray. | Assertions of endpoint positions, shape, spacing and dtype. | Checks geometry against analytical expected values. |
| `ProjectTests.test_integral_analytic_and_gradients` | 21-29 | 9 | Linear and constant attenuation fields. | Assertions of integrals and known derivatives. | Checks physical distance scaling, endpoint handling and differentiability. |
| `ProjectTests.test_supplied_index_examples_and_boundaries` | 31-37 | 7 | Supplied coordinate example plus extreme coordinates. | Assertions at all eight scales and boundary indices. | Checks the specific indexing convention required by the supplied reference outputs. |
| `ProjectTests.test_models_update_and_checkpoint_roundtrip` | 39-69 | 31 | Both model classes, tiny rays/targets and a temporary checkpoint. | Finite updates, changed attention parameters, restored predictions and optimizer iteration. | Proves Model 2 is used in training and that saved models can be recovered. |
| `ProjectTests.test_zero_volume_and_identical_metrics` | 71-76 | 6 | All-zero volume and identical synthetic volumes. | Zero-safe display and perfect-image metric assertions. | Checks normalization edge cases and metric conventions. |
| Script entry point | 79-80 | 2 | CLI invocation | Starts the appropriate runner | Importing the file does not start training |

### `gpu_python.py`

| Block | Lines | Count | Input | Output / behavior | Reason |
|---|---|---:|---|---|---|
| Imports | 2-5 | 4 | Installed packages | Available APIs | Numerical operations, runtime recording or plotting needed by this file |
| `main` | 8-22 | 15 | Python command-line arguments and installed NVIDIA package records. | Replaces launcher process with Python using the selected arguments. | Adds actual library directories before process startup: the workaround verified by the user on both lab GPUs. On Mac with no NVIDIA packages, behaves like a normal Python launcher. |
| Script entry point | 25-26 | 2 | CLI invocation | Starts the appropriate runner | Importing the file does not start training |

### `compare_runs.py`

| Block | Lines | Count | Input | Output / behavior | Reason |
|---|---|---:|---|---|---|
| Imports | 2-10 | 9 | Installed packages | Available APIs | Numerical operations, runtime recording or plotting needed by this file |
| `load_run` | 13-19 | 7 | Run folder containing config and CSV. | Configuration and numerical epoch rows. | Rejects empty runs before figure generation. |
| `compare` | 22-67 | 46 | Baseline and attention folders and figure destination. | Two figure types in PNG/PDF and comparison JSON. | Checks paired settings and completed epochs, plots measured curves and matching slices. Does not infer significance or create report conclusions. |
| Script entry point | 70-76 | 7 | CLI invocation | Starts the appropriate runner | Importing the file does not start training |

## Important differences from the starter

- `train` and the old hardcoded `main` are replaced by a configurable one-step update, seeded epoch loop and `run` orchestration. Legacy direct imports of `train`/`main` are not retained; the supplied tests only need the three functions/model and still run unchanged.
- `get_sample_slices` is split into raw reconstruction and display normalization, preserving raw values separately for future analysis.
- The starter reports normalized projection MSE as loss. Both raw projection MSE and normalized MSE are now explicitly labelled.
- SSIM/PSNR are evaluated EVERY epoch, as requested by the brief, rather than only every ten. This increases runtime; training and evaluation time are recorded separately.
- Epoch checkpoints retain optimizer state and all embedding weights. Sampling uses explicit epoch seeds; checkpoints do not rely on restoring Python global random state.
- Output directories reject accidental overwrites. A lock prevents concurrent writers. A crash after CSV write but before checkpoint save is handled by discarding rows newer than the restored epoch.
- The dataset loader is unchanged; its `device` argument is unused in the supplied implementation. An explicit TensorFlow CPU device context keeps data preparation off GPU memory.
- Physical cube size remains `far - near` as in the starter; this is not a redesigned scanner geometry. All metrics use the provided reference geometry.
- Full datasets are not automatically trained one after another. Chest is the required initial dataset; extra datasets are optional experiments, selected with `--dataset`.

## Supporting files

`gpu_python.py` is the portable version of the launcher proven in the lab conversation. `compare_runs.py` creates measured figures after experiments. `.gitignore` excludes large/private runtime artifacts. Dependency files distinguish Mac CPU from Linux CUDA setup. `README.md` contains Mac, GitHub, SSH/tmux, resume, evaluation, comparison and submission commands. `VALIDATION.md` records exactly which checks ran.

## Reading the measurements honestly

MSE: average squared prediction error, lower is better. NMSE: MSE divided by the sampled target energy; this is the optimized loss. SSIM: image-structure similarity, higher is generally better. PSNR: image error expressed in decibels, higher is better. Image metrics use per-volume display normalization, so they do not certify physical attenuation values. Epoch metrics measure training reconstruction, not an unseen-data test.

## Validation and remaining work

See `VALIDATION.md` for executed checks. Run the supplied tests and assertions in each lab environment before full training. Use matching settings for both models, preserve logs/screenshots, generate figures from real runs, and write the final report in your own understanding. Submit actual code/report/execution evidence to Learn; do not upload the supplied datasets. The precise deadline and permitted report upload format must be checked on Learn.

## Source references

- Course brief: `COSC440 Project 2026.pdf`, particularly TODO steps and grading breakdown.
- Course slides: `research_ct_only.pdf` for baseline positional lookup and attenuation synthesis; `research.pdf` for reconstruction background.
- TensorFlow checkpoint API: https://www.tensorflow.org/api_docs/python/tf/train/Checkpoint
- TensorFlow multi-head attention API: https://www.tensorflow.org/api_docs/python/tf/keras/layers/MultiHeadAttention
- TensorFlow installation: https://www.tensorflow.org/install/pip

## Smaller blocks inside the training runner

All entries below refer to `project_main.py` and overlap the `run` entry above.

| Lines | Count | Responsibility | Input -> expected output / behavior |
|---|---:|---|---|
| 245-251 | 7 | Device, paths and dataset identity | CLI options -> verified dataset path and SHA-256 configuration |
| 252-260 | 9 | Protect original results | Saved settings/current flags -> reject changed experiment settings or accidental overwrite |
| 261-267 | 7 | Exclude concurrent writers | Output directory -> exclusive `.running` lock |
| 269-273 | 5 | Reproducible setup and CPU data | Seed/file -> seeded initialization and loaded CPU dataset |
| 274-284 | 11 | Model and training state | Configuration/checkpoint -> model, optimizer slots and restored epoch |
| 285-290 | 6 | Record provenance | Configuration/runtime -> original config and latest runtime JSON |
| 291-297 | 7 | Reference and evaluation-only mode | Supplied ground truth/model -> reference TIFF and optional checkpoint metrics/raw reconstruction |
| 298-311 | 14 | Maintain measurement history | Prior CSV and restored epoch -> validated, deduplicated CSV with header |
| 312-322 | 11 | Epoch training and measurement | Model/dataset/epoch -> updates, full reconstruction, metrics and separate elapsed times |
| 323-325 | 3 | Preserve image evidence | Epoch and raw volume -> scheduled display/raw TIFF snapshots |
| 326-330 | 5 | Commit an epoch | Metrics/model/optimizer -> appended CSV, checkpoint and printed JSON progress |
| 331-332 | 2 | Release lock | Success or ordinary exception -> `.running` removed |

Symbol glossary: R = sampled rays, S = points per ray, N = queried spatial positions,
D = number of grid scales, E = embedding features per scale, T = attention-token width.
A checkpoint is a recoverable training save-point. An epoch is a pass through projection
images (or the explicitly labelled diagnostic subset). A residual connection adds a layer's
input back to its output; layer normalization rescales token features. These help the
attention block train, but do not guarantee better reconstruction.
