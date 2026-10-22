"""Launch one reproducibly named UC experiment with explicit settings."""
import argparse
import math
import os
from pathlib import Path
import re
import shlex
import socket
import subprocess
import sys

MODELS = ('baseline', 'attention')
HOSTS = {'3090': 'cs23002lv', '4070': 'cs25049kx'}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('environment', choices=HOSTS)
    parser.add_argument('model', choices=MODELS)
    parser.add_argument('seed', type=int)
    parser.add_argument('learning_rate', type=float)
    parser.add_argument('epochs', type=int)
    parser.add_argument('comments', nargs='?', default='')
    parser.add_argument('--resume', action='store_true', help='Continue this same named run')
    args = parser.parse_args(argv)
    if not 0 <= args.seed < 2**32:
        parser.error('seed must be an integer from 0 through 4294967295')
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error('learning rate must be a finite positive number')
    if args.epochs < 1:
        parser.error('epochs must be a positive integer')
    comment = re.sub(r'[^a-zA-Z0-9_-]+', '-', args.comments.strip()).strip('-_').lower()
    if args.comments and not comment:
        parser.error('comments must contain at least one letter or number')
    if len(comment) > 80:
        parser.error('comments must be at most 80 characters after filename sanitising')
    args.comments = comment
    return args


def run_name(args):
    rate = str(args.learning_rate)
    name = f'{args.environment}-{args.model}-s{args.seed}-lr{rate}-e{args.epochs}'
    return name + (f'-{args.comments}' if args.comments else '')


def command(args, root, python, dataset):
    cmd = [str(python), '-u', str(root / 'gpu_python.py'), str(root / 'project_main.py'),
           '--dataset', str(dataset), '--output', str(root / 'runs' / run_name(args)),
           '--model', args.model, '--seed', str(args.seed),
           '--learning-rate', str(args.learning_rate), '--epochs', str(args.epochs),
           '--n-points', '192', '--n-rays', '2048', '--depth', '8',
           '--embedding-dim', '1', '--query-batch', '8192', '--save-every', '10',
           '--max-projections', '0', '--device', 'gpu']
    return cmd + (['--resume'] if args.resume else [])


def main():
    args = parse_args()
    root = Path(__file__).resolve().parent
    host = socket.gethostname().split('.')[0]
    if host != HOSTS[args.environment]:
        raise SystemExit(f'Environment {args.environment} requires {HOSTS[args.environment]}; current host is {host}')
    python = Path(os.environ.get('COSC440_PYTHON', str(Path.home() / '.venvs' / f'cosc440-{host}' / 'bin/python')))
    dataset = Path(os.environ.get('COSC440_DATASET', str(root / 'data/ct_data/chest_50.pickle'))).resolve()
    if not python.is_file() or not os.access(python, os.X_OK):
        raise SystemExit(f'Missing executable Python: {python}')
    if not dataset.is_file():
        raise SystemExit(f'Missing dataset: {dataset}')
    output = root / 'runs' / run_name(args)
    if args.resume:
        if not (output / 'config.json').is_file():
            raise SystemExit(f'Cannot resume: missing config.json in {output}')
    elif output.exists():
        raise SystemExit(f'Output already exists: {output}; use --resume or a new comment')
    (root / 'logs').mkdir(exist_ok=True)
    log = root / 'logs' / f'{run_name(args)}.log'
    cmd = command(args, root, python, dataset)
    # Exclusive creation prevents accidentally replacing an earlier experiment log.
    with log.open('a' if args.resume else 'x', buffering=1) as stream:
        header = f'Host: {host}\nCommand: {shlex.join(cmd)}\nLog: {log}\n'
        stream.write(header)
        print(header, end='', flush=True)
        with subprocess.Popen(cmd, cwd=root, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              text=True, bufsize=1) as process:
            try:
                for line in process.stdout:
                    stream.write(line)
                    print(line, end='', flush=True)
                status = process.wait()
            except KeyboardInterrupt:
                process.terminate()
                process.wait()
                raise
        stream.write(f'Process exit status: {status}\n')
    raise SystemExit(status)


if __name__ == '__main__':
    main()
