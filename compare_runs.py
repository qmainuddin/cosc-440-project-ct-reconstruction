"""Create report figures and a measured comparison from two completed run folders."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tifffile


def load_run(folder):
    config = json.loads((folder / 'config.json').read_text())
    with (folder / 'metrics.csv').open() as source:
        rows = [{k: float(v) for k, v in row.items()} for row in csv.DictReader(source)]
    if not rows:
        raise ValueError(f'No epoch measurements in {folder}')
    return config, rows


def compare(first, second, output, epoch=None):
    a, rows_a = load_run(first)
    b, rows_b = load_run(second)
    # Reject misleading comparisons of different datasets or sampling budgets.
    for key in ('dataset_sha256', 'seed', 'n_points', 'n_rays', 'depth',
                'embedding_dim', 'learning_rate', 'max_projections'):
        if a[key] != b[key]:
            raise ValueError(f'Unmatched experiment setting: {key}')
    if a['model'] != 'baseline' or b['model'] != 'attention':
        raise ValueError('Supply baseline first, attention second')
    if epoch is None:
        epoch = int(rows_a[-1]['epoch'])
        if epoch != int(rows_b[-1]['epoch']):
            raise ValueError('Use --epoch to select a shared epoch for unequal run lengths')
    rows_a = [r for r in rows_a if r['epoch'] <= epoch]
    rows_b = [r for r in rows_b if r['epoch'] <= epoch]
    if not rows_a or not rows_b or rows_a[-1]['epoch'] != epoch or rows_b[-1]['epoch'] != epoch:
        raise ValueError('Requested epoch must exist in both histories')
    output.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 2, figsize=(11, 4), layout='constrained')
    for rows, label in ((rows_a, 'Baseline'), (rows_b, 'Scale attention')):
        axes[0].plot([r['epoch'] for r in rows], [r['train_mse'] for r in rows], label=label)
        axes[1].plot([r['epoch'] for r in rows], [r['ssim'] for r in rows], label=label)
    axes[0].set(ylabel='Sampled projection MSE (log scale)', xlabel='Epoch', yscale='log')
    axes[1].set(ylabel='Mean slice SSIM (higher is better)', xlabel='Epoch')
    for axis in axes:
        axis.legend()
        axis.grid(alpha=.2)
    figure.savefig(output / 'learning_curves.png', dpi=180)
    figure.savefig(output / 'learning_curves.pdf')
    plt.close(figure)
    truth = tifffile.imread(first / 'ground_truth.tiff')
    volumes = [truth, tifffile.imread(first / f'epoch_{epoch:04d}.tiff'),
               tifffile.imread(second / f'epoch_{epoch:04d}.tiff')]
    indices = sorted(set([truth.shape[0] // 4, truth.shape[0] // 2, 3 * truth.shape[0] // 4]))
    figure, axes = plt.subplots(len(indices), 3, figsize=(9, 3 * len(indices)), squeeze=False,
                                layout='constrained')
    for row, z in enumerate(indices):
        for column, (label, volume) in enumerate(zip(['Ground truth', 'Baseline', 'Scale attention'], volumes)):
            axes[row, column].imshow(volume[z], cmap='gray', vmin=0, vmax=255)
            axes[row, column].set_title(f'{label} | slice {z}')
            axes[row, column].axis('off')
    figure.savefig(output / 'reconstruction_comparison.png', dpi=180)
    figure.savefig(output / 'reconstruction_comparison.pdf')
    plt.close(figure)
    summary = {'epoch': epoch, 'baseline': rows_a[-1], 'attention': rows_b[-1],
               'attention_minus_baseline': {key: rows_b[-1][key] - rows_a[-1][key]
                                            for key in ('train_mse', 'ssim', 'display_mse')},
               'caution': 'Training-fit metrics, not held-out generalization. Images use per-volume scaling.'}
    (output / 'comparison.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('baseline', type=Path)
    parser.add_argument('attention', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--epoch', type=int, help='Common epoch to compare; must have TIFF snapshots')
    args = parser.parse_args()
    compare(args.baseline, args.attention, args.output, args.epoch)
