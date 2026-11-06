from __future__ import annotations

import argparse
from pathlib import Path

from .mmcad import prepare_mmcad
from .text2cad import prepare_text2cad


def text2cad_command(argv=None):
    parser = argparse.ArgumentParser(prog="cadret prepare-text2cad")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args(argv)
    result = prepare_text2cad(args.data_root, args.output, args.workers)
    print(result, flush=True)


def mmcad_command(argv=None):
    parser = argparse.ArgumentParser(prog="cadret prepare-mmcad")
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--step-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--test-size", type=int, default=1000)
    parser.add_argument("--split-seed", type=int, default=20261025)
    args = parser.parse_args(argv)
    result = prepare_mmcad(
        args.metadata,
        args.step_root,
        args.output,
        args.workers,
        args.test_size,
        args.split_seed,
    )
    print(result, flush=True)
