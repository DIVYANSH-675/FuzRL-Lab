#!/usr/bin/env python3
"""Launch supported algorithms from any working directory without exposing secrets."""
import argparse
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def build_command(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-a', '--algorithm', choices=['ppolag', 'fuzppolag', 'cvarppolag'], required=True)
    parser.add_argument('-e', '--env', required=True)
    device = parser.add_mutually_exclusive_group()
    device.add_argument('-g', '--gpu', type=int)
    device.add_argument('--cpu', action='store_true')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('-s', '--seed', type=int, default=42)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--log-dir', default=str(ROOT / 'logs'))
    parser.add_argument('--swanlab-api-key', help=argparse.SUPPRESS)
    args, extra = parser.parse_known_args(argv)
    if args.epochs < 1 or (args.gpu is not None and args.gpu < 0):
        parser.error('epochs must be positive and gpu must be nonnegative')
    env = os.environ.copy()
    if args.gpu is not None:
        env['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    if args.cpu:
        env['CUDA_VISIBLE_DEVICES'] = ''
    if args.swanlab_api_key:
        env['SWANLAB_API_KEY'] = args.swanlab_api_key
    selected = 'cpu' if args.cpu else ('cuda:0' if args.gpu is not None else 'auto')
    command = [sys.executable, str(ROOT / 'scripts/train.py'),
               '--algorithm', args.algorithm, '--env', args.env, '--device', selected,
               '--epochs', str(args.epochs), '--seed', str(args.seed), '--log_dir', args.log_dir]
    # The direct training entry point uses underscore options; accept either spelling here.
    command.extend(('--' + arg[2:].replace('-', '_') if arg.startswith('--') else arg)
                   for arg in extra)
    return command, env, args.dry_run


def main():
    command, env, dry_run = build_command()
    print(shlex.join(command), flush=True)
    if not dry_run:
        try:
            return subprocess.run(command, env=env, cwd=ROOT).returncode
        except KeyboardInterrupt:
            return 130
    return 0


if __name__ == '__main__':
    sys.exit(main())
