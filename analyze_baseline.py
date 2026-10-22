"""Validate a baseline run and create reproducible report figures (no training)."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
import tifffile
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def analyze(run, log, output, source):
    config = json.loads((run / 'config.json').read_text())
    runtime = json.loads((run / 'runtime.json').read_text())
    with (run / 'metrics.csv').open() as stream:
        rows = [{k: float(v) for k, v in row.items()} for row in csv.DictReader(stream)]
    assert config['model'] == 'baseline', 'Expected baseline run'
    assert [r['epoch'] for r in rows] == list(range(1, runtime['target_epochs'] + 1)), 'Missing epochs'
    assert all(np.isfinite(list(r.values())).all() for r in rows), 'Nonfinite metrics'
    logged = [json.loads(line) for line in log.read_text().splitlines() if line.startswith('{"epoch":')]
    assert logged == rows, 'Log and CSV disagree'
    for name, digest in runtime['code_sha256'].items():
        assert hashlib.sha256((source / name).read_bytes()).hexdigest() == digest, f'Changed source: {name}'
    truth = tifffile.imread(run / 'ground_truth.tiff')
    volumes = {}
    errors = {}
    for file in sorted(run.glob('epoch_*.tiff')):
        if file.stem.endswith('_raw'):
            continue
        epoch = int(file.stem.split('_')[1])
        display = tifffile.imread(file)
        raw = tifffile.imread(file.with_name(file.stem + '_raw.tiff'))
        assert raw.shape == display.shape == truth.shape, 'Volume shape mismatch'
        assert np.isfinite(raw).all(), 'Nonfinite reconstruction'
        clipped = np.maximum(raw, 0.)
        expected = (clipped / clipped.max() * 255).astype(np.uint8) if clipped.max() > 0 else np.zeros_like(display)
        assert np.array_equal(display, expected), 'Raw/display mismatch'
        mse = float(np.mean((display.astype(np.float64)/255 - truth.astype(np.float64)/255)**2))
        assert np.isclose(mse, rows[epoch-1]['display_mse'], rtol=1e-10), 'Image/CSV MSE mismatch'
        errors[str(epoch)] = mse
        volumes[epoch] = display
    assert int(rows[-1]['epoch']) in volumes, 'Missing final image'
    output.mkdir(parents=True, exist_ok=True)
    def save(fig, name):
        fig.savefig(output / (name + '.png'), dpi=200)
        fig.savefig(output / (name + '.pdf'))
        plt.close(fig)
    epochs = [r['epoch'] for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout='constrained')
    axes[0].semilogy(epochs, [r['train_mse'] for r in rows], color='#2166ac')
    axes[0].set(xlabel='Epoch', ylabel='Sampled projection MSE (log scale)', title='Baseline training error — lower is better')
    axes[1].plot(epochs, [r['ssim'] for r in rows], color='#00866a')
    axes[1].set(xlabel='Epoch', ylabel='Mean slice SSIM', ylim=(0, 1), title='Baseline image similarity — higher is better')
    for ax in axes:
        ax.grid(alpha=.2)
    save(fig, 'baseline_learning_curves')
    indices = [truth.shape[0]//4, truth.shape[0]//2, 3*truth.shape[0]//4]
    panels = [('Ground truth', truth)] + [(f'Epoch {e}', v) for e, v in sorted(volumes.items())]
    fig, axes = plt.subplots(len(indices), len(panels), figsize=(3*len(panels), 9), layout='constrained', squeeze=False)
    for i, z in enumerate(indices):
        for j, (title, volume) in enumerate(panels):
            axes[i,j].imshow(volume[z], cmap='gray', vmin=0, vmax=255)
            axes[i,j].set_title(f'{title}\nSlice {z}')
            axes[i,j].axis('off')
    save(fig, 'baseline_reconstruction_progress')
    summary = {
        'validation': 'PASS: contiguous finite epoch records; CSV matches log; source hashes match; saved raw/display volumes and image MSE agree',
        'volume_shape': list(truth.shape), 'first_epoch': rows[0], 'final_epoch': rows[-1],
        'best_ssim_epoch': max(rows, key=lambda r:r['ssim']),
        'saved_epoch_display_mse': errors,
        'total_recorded_training_seconds': sum(r['training_seconds'] for r in rows),
        'total_recorded_evaluation_seconds': sum(r['evaluation_seconds'] for r in rows),
        'limitations': ['One baseline seed; full attention run still required for comparison.',
                       'Training-fit metrics do not measure unseen-data generalization.',
                       'SSIM is averaged over 2D slices; images are independently scaled per volume.',
                       'Recorded timings exclude startup and some file/checkpoint writing.']}
    (output / 'baseline_summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--log', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source', type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    analyze(args.run, args.log, args.output, args.source)
