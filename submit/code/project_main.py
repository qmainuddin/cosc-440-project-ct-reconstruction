"""COSC440 CT reconstruction: baseline, scale-attention experiment and lab runner."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import socket
import subprocess
import sys
import time

import numpy as np
import tensorflow as tf
from tensorflow import keras
import tifffile
from geometry import TIGREDataset


def rays_to_points(rays, n_points, near, far):
    """[R,6] rays -> ([R,S,3] positions, [S] ray parameters), endpoints included."""
    rays = tf.convert_to_tensor(rays)
    if n_points < 2:
        raise ValueError('n_points must be at least 2')
    near, far = tf.cast(near, rays.dtype), tf.cast(far, rays.dtype)
    tf.debugging.assert_greater(far, near)
    distances = tf.linspace(near, far, n_points)
    points = rays[:, None, :3] + rays[:, None, 3:] * distances[None, :, None]
    return points, distances


def ray_attenuation(attenuations, distances, magnitudes, near, far):
    """Trapezoidal line integral; constant extension if samples exclude endpoints."""
    attenuations = tf.convert_to_tensor(attenuations)
    dtype = attenuations.dtype
    distances = tf.cast(distances, dtype)
    magnitudes = tf.cast(magnitudes, dtype)
    near, far = tf.cast(near, dtype), tf.cast(far, dtype)
    tf.debugging.assert_greater_equal(distances[1:], distances[:-1])
    tf.debugging.assert_greater_equal(distances[0], near)
    tf.debugging.assert_less_equal(distances[-1], far)
    middle = tf.reduce_sum(
        (attenuations[:, 1:] + attenuations[:, :-1]) * 0.5
        * (distances[1:] - distances[:-1])[None, :], axis=1)
    edges = (attenuations[:, 0] * (distances[0] - near)
             + attenuations[:, -1] * (far - distances[-1]))
    return (middle + edges) * magnitudes


class MultiScaleVoxelModel(keras.Model):
    """Original multiresolution embedding model, with completed x-fastest indexing."""
    def __init__(self, size, n_depth, embedding_dim=1):
        super().__init__()
        self.size = float(size)
        self.n_depth = int(n_depth)
        self.embedding_dim = int(embedding_dim)
        if self.size <= 0 or not 1 <= n_depth <= 9 or embedding_dim < 1:
            raise ValueError('Require size > 0, 1 <= depth <= 9, embedding_dim >= 1')
        self.embedding_layers = []
        for depth in range(n_depth):
            layer = keras.layers.Embedding((2 ** (depth + 1)) ** 3, embedding_dim)
            # Named attributes also make all tables explicit checkpoint dependencies.
            setattr(self, f'grid_{depth}', layer)
            self.embedding_layers.append(layer)
        self.hidden_dense = keras.layers.Dense(64, activation='silu')
        self.output_layer = keras.layers.Dense(1)

    def get_flattened_position(self, scaled_values, depth):
        """[N,3] coordinates in [0,1) -> [N] int32 indices x + r*y + r*r*z."""
        resolution = 2 ** (depth + 1)
        cells = tf.cast(tf.floor(scaled_values * resolution), tf.int32)
        cells = tf.clip_by_value(cells, 0, resolution - 1)
        return cells[:, 0] + resolution * cells[:, 1] + resolution ** 2 * cells[:, 2]

    def embeddings(self, inputs):
        """Physical [N,3] positions -> a list of D feature tensors [N,E]."""
        scaled = tf.clip_by_value(0.5 + inputs / self.size, 0., 1. - 1e-7)
        return [layer(self.get_flattened_position(scaled, depth))
                for depth, layer in enumerate(self.embedding_layers)]

    def call(self, inputs, training=False):
        features = tf.concat(self.embeddings(inputs), axis=-1)
        return self.output_layer(self.hidden_dense(features))


class ScaleAttentionModel(MultiScaleVoxelModel):
    """Model 2: learned interaction between coarse/fine scale tokens at each point."""
    def __init__(self, size, n_depth, embedding_dim=1, token_dim=8, heads=2):
        super().__init__(size, n_depth, embedding_dim)
        if token_dim < 2 or heads < 1 or token_dim % heads:
            raise ValueError('token_dim must be >=2 and divisible by heads')
        self.token_dim = token_dim
        self.token_projection = keras.layers.Dense(token_dim)
        self.scale_identity = self.add_weight(
            name='scale_identity', shape=(n_depth, token_dim),
            initializer=keras.initializers.RandomNormal(stddev=0.02))
        self.attention = keras.layers.MultiHeadAttention(
            num_heads=heads, key_dim=token_dim // heads, dropout=0.0)
        self.normalization = keras.layers.LayerNormalization()

    def call(self, inputs, training=False):
        tokens = tf.stack(self.embeddings(inputs), axis=1)
        tokens = self.token_projection(tokens) + self.scale_identity[None, :, :]
        attended = self.attention(tokens, tokens, training=training)
        features = self.normalization(tokens + attended)
        features = tf.reshape(features, [-1, self.n_depth * self.token_dim])
        return self.output_layer(self.hidden_dense(features))


def build_model(config, size):
    """Build variables before optimizer/checkpoint restoration."""
    common = (size, config['depth'], config['embedding_dim'])
    if config['model'] == 'attention':
        model = ScaleAttentionModel(*common, config['token_dim'], config['heads'])
    else:
        model = MultiScaleVoxelModel(*common)
    model(tf.zeros((1, 3), tf.float32))
    return model


def configure_device(device):
    """Select CPU, automatic GPU, or fail explicitly when a GPU was requested."""
    if device == 'cpu':
        tf.config.set_visible_devices([], 'GPU')
    gpus = tf.config.list_physical_devices('GPU') if device != 'cpu' else []
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    if device == 'gpu' and not gpus:
        raise RuntimeError('No TensorFlow GPU: run through gpu_python.py and verify setup')
    return '/GPU:0' if gpus else '/CPU:0'


def make_train_step(model, optimizer, n_points):
    """Compile one update, avoiding a graph containing all projection iterations."""
    @tf.function(reduce_retracing=True)
    def step(rays, targets, near, far):
        points, distances = rays_to_points(rays, n_points, near, far)
        with tf.GradientTape() as tape:
            values = model(tf.reshape(points, [-1, 3]), training=True)
            values = tf.reshape(values, [tf.shape(rays)[0], n_points])
            prediction = ray_attenuation(
                values, distances, tf.norm(rays[:, 3:], axis=1), near, far)
            mse = tf.reduce_mean(tf.square(prediction - targets))
            nmse = mse / tf.maximum(tf.reduce_mean(tf.square(targets)), 1e-12)
        gradients = tape.gradient(nmse, model.trainable_variables)
        if any(g is None for g in gradients):
            raise RuntimeError('A trainable variable is disconnected from the loss')
        tf.debugging.assert_all_finite(nmse, 'Non-finite training loss')
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return mse, nmse
    return step


def train_epoch(step, dataset, config, epoch):
    """Same seeded projection/ray selection in both models and after epoch resume."""
    rng = np.random.default_rng(np.random.SeedSequence([config['seed'], epoch]))
    count = int(dataset.projections.shape[0])
    order = rng.permutation(count)
    if config['max_projections']:
        order = order[:config['max_projections']]
    totals = np.zeros(2, dtype=np.float64)
    pixels = int(np.prod(dataset.projections.shape[1:]))
    n_rays = min(config['n_rays'], pixels)
    for idx in order:
        indices = rng.choice(pixels, n_rays, replace=False)
        rays = tf.gather(tf.reshape(dataset.rays[..., int(idx)], [-1, 6]), indices)
        targets = tf.gather(tf.reshape(dataset.projections[int(idx)], [-1]), indices)
        values = step(rays, targets, tf.constant(dataset.near, tf.float32),
                      tf.constant(dataset.far, tf.float32))
        totals += [float(v.numpy()) for v in values]
    return totals / len(order)


def reconstruct(model, dataset, query_batch):
    """Return raw float32 [Z,X,Y] attenuation volume using bounded query batches."""
    output = []
    for z in range(dataset.voxels.shape[2]):
        positions = dataset.voxels[:, :, z, :]
        flat = positions.reshape(-1, 3)
        chunks = [model(tf.convert_to_tensor(flat[i:i + query_batch], tf.float32),
                        training=False).numpy().ravel()
                  for i in range(0, len(flat), query_batch)]
        output.append(np.concatenate(chunks).reshape(positions.shape[:2]))
    return np.asarray(output, dtype=np.float32)


def display_volume(raw):
    """Match starter display scaling: nonnegative, per-volume maximum -> uint8."""
    clipped = np.maximum(raw, 0.)
    maximum = float(clipped.max())
    return ((clipped / maximum) * 255).astype(np.uint8) if maximum > 0 else np.zeros_like(raw, dtype=np.uint8)


def volume_metrics(raw, ground_truth):
    """Mean per-slice SSIM/PSNR on starter-style uint8 volumes; not physical units."""
    predicted = display_volume(raw)
    reference = (np.clip(ground_truth.transpose(2, 0, 1), 0, 1) * 255).astype(np.uint8)
    if predicted.shape != reference.shape or min(reference.shape[1:]) < 11:
        raise ValueError('Matching volumes with slice dimensions >= 11 required for SSIM')
    # Slice-wise evaluation bounds GPU/CPU memory for larger datasets.
    ssim, psnr = [], []
    for p, g in zip(predicted, reference):
        p, g = tf.convert_to_tensor(p[..., None]), tf.convert_to_tensor(g[..., None])
        ssim.append(float(tf.image.ssim(p, g, max_val=255).numpy()))
        psnr.append(float(tf.image.psnr(p, g, max_val=255).numpy()))
    return {'ssim': float(np.mean(ssim)), 'psnr_db': float(np.mean(psnr)),
            'display_mse': float(np.mean((predicted.astype(np.float64) / 255
                                         - reference.astype(np.float64) / 255) ** 2))}


def write_json(path, data):
    """Atomic JSON replacement, so interruption does not leave a partial file."""
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, indent=2) + '\n')
    temporary.replace(path)


def dataset_digest(path):
    """Content hash makes comparisons independent of a machine's dataset path."""
    digest = hashlib.sha256()
    with open(path, 'rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def runtime_info():
    """Record code hashes, host and installed software for reproducibility."""
    root = Path(__file__).resolve().parent
    try:
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root,
                                         stderr=subprocess.DEVNULL, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    return {'hostname': socket.gethostname(), 'platform': platform.platform(),
            'python': sys.version, 'tensorflow': tf.__version__, 'numpy': np.__version__,
            'git_commit': commit, 'code_sha256': {
                name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                for name in ('project_main.py', 'geometry.py')}}


def run(args):
    """Train/resume, reconstruct, evaluate and save recoverable per-epoch results."""
    device = configure_device(args.device)
    out = Path(args.output).resolve()
    dataset_path = Path(args.dataset).resolve()
    if not dataset_path.is_file():
        raise FileNotFoundError(dataset_path)
    config = vars(args).copy()
    config.update(dataset=str(dataset_path), dataset_sha256=dataset_digest(dataset_path))
    previous = None
    if args.resume or args.evaluate_only:
        previous = json.loads((out / 'config.json').read_text())
        mutable = {'epochs', 'dataset', 'output', 'resume', 'evaluate_only', 'device', 'query_batch'}
        for key, value in config.items():
            if key not in mutable and previous.get(key) != value:
                raise ValueError(f'Resume/evaluation configuration differs: {key}')
    elif out.exists() and any(out.iterdir()):
        raise ValueError('Output folder is not empty; use a new run name or --resume')
    out.mkdir(parents=True, exist_ok=True)
    lock = out / '.running'
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as error:
        raise RuntimeError('Run is locked. Check for an active process before removing .running') from error
    os.close(descriptor)
    try:
        tf.keras.utils.set_random_seed(args.seed)
        tf.config.experimental.enable_op_determinism()
        # Keep the large source arrays on CPU; only selected batches go to the GPU.
        with tf.device('/CPU:0'):
            dataset = TIGREDataset(str(dataset_path), device='cpu', n_rays=args.n_rays)
        with tf.device(device):
            model = build_model(config, dataset.far - dataset.near)
            optimizer = keras.optimizers.Adam(learning_rate=args.learning_rate)
            optimizer.build(model.trainable_variables)
            completed = tf.Variable(0, dtype=tf.int64, trainable=False)
            checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer, epoch=completed)
            manager = tf.train.CheckpointManager(checkpoint, str(out / 'checkpoints'), max_to_keep=2)
            if previous:
                if not manager.latest_checkpoint:
                    raise FileNotFoundError('No saved checkpoint to restore')
                checkpoint.restore(manager.latest_checkpoint).assert_consumed()
            else:
                write_json(out / 'config.json', config)
            info = runtime_info()
            info.update(device=device, parameters=model.count_params(),
                        resumed_from=int(completed.numpy()), target_epochs=args.epochs)
            write_json(out / ('runtime-evaluation.json' if args.evaluate_only else 'runtime.json'), info)
            reference = (np.clip(dataset.ground_truth.transpose(2, 0, 1), 0, 1) * 255).astype(np.uint8)
            tifffile.imwrite(out / 'ground_truth.tiff', reference, photometric='minisblack')
            if args.evaluate_only:
                raw = reconstruct(model, dataset, args.query_batch)
                write_json(out / 'evaluation.json', volume_metrics(raw, dataset.ground_truth))
                tifffile.imwrite(out / 'evaluation_raw.tiff', raw, photometric='minisblack')
                return
            fields = ['epoch', 'train_mse', 'train_nmse', 'ssim', 'psnr_db',
                      'display_mse', 'training_seconds', 'evaluation_seconds']
            metrics_path = out / 'metrics.csv'
            # A crash between CSV append and checkpoint save must not duplicate rows.
            rows = []
            if previous and metrics_path.exists():
                with metrics_path.open() as stream:
                    rows = [r for r in csv.DictReader(stream) if int(r['epoch']) <= int(completed.numpy())]
            if previous and [int(r['epoch']) for r in rows] != list(range(1, int(completed.numpy()) + 1)):
                raise ValueError('Metrics history is incomplete; recover metrics.csv before resuming')
            with metrics_path.open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
            step = make_train_step(model, optimizer, args.n_points)
            for epoch in range(int(completed.numpy()) + 1, args.epochs + 1):
                start = time.perf_counter()
                mse, nmse = train_epoch(step, dataset, config, epoch)
                training_seconds = time.perf_counter() - start
                start = time.perf_counter()
                raw = reconstruct(model, dataset, args.query_batch)
                scores = volume_metrics(raw, dataset.ground_truth)
                row = dict(epoch=epoch, train_mse=mse, train_nmse=nmse, **scores,
                           training_seconds=training_seconds,
                           evaluation_seconds=time.perf_counter() - start)
                if epoch % args.save_every == 0 or epoch == args.epochs:
                    tifffile.imwrite(out / f'epoch_{epoch:04d}.tiff', display_volume(raw), photometric='minisblack')
                    tifffile.imwrite(out / f'epoch_{epoch:04d}_raw.tiff', raw, photometric='minisblack')
                with metrics_path.open('a', newline='') as stream:
                    csv.DictWriter(stream, fieldnames=fields).writerow(row)
                completed.assign(epoch)
                manager.save(checkpoint_number=epoch)
                print(json.dumps(row), flush=True)
    finally:
        lock.unlink(missing_ok=True)


def parse_args():
    """Explicit CLI defaults preserve starter settings; smoke runs must be labelled."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', default=str(Path(__file__).parent / 'data/ct_data/chest_50.pickle'))
    parser.add_argument('--output', required=True)
    parser.add_argument('--model', choices=['baseline', 'attention'], default='baseline')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--n-points', type=int, default=192)
    parser.add_argument('--n-rays', type=int, default=2048)
    parser.add_argument('--depth', type=int, default=8)
    parser.add_argument('--embedding-dim', type=int, default=1)
    parser.add_argument('--token-dim', type=int, default=8)
    parser.add_argument('--heads', type=int, default=2)
    parser.add_argument('--learning-rate', type=float, default=0.001)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--query-batch', type=int, default=8192)
    parser.add_argument('--save-every', type=int, default=10)
    parser.add_argument('--max-projections', type=int, default=0,
                        help='0=all projections; nonzero is a diagnostic subset per epoch')
    parser.add_argument('--device', choices=['auto', 'cpu', 'gpu'], default='auto')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--evaluate-only', action='store_true')
    args = parser.parse_args()
    for key in ('epochs', 'n_rays', 'embedding_dim', 'query_batch', 'save_every'):
        if getattr(args, key) < 1:
            parser.error(f'{key} must be positive')
    if args.n_points < 2 or not 1 <= args.depth <= 9 or args.max_projections < 0 or args.learning_rate <= 0:
        parser.error('Invalid points, depth, projection limit or learning rate')
    return args


if __name__ == '__main__':
    run(parse_args())
