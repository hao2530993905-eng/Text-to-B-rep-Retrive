import argparse
from pathlib import Path

from ..config import DATA
from .descriptors import prepare_cache


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="cadret prepare", description="Build CAD descriptors for training and evaluation"
    )
    parser.add_argument("--data_dir", default=DATA)
    parser.add_argument("--output_dir", default="runs")
    parser.add_argument("--cache", help="Override the descriptor cache file")
    parser.add_argument("--workers", type=int, default=48)
    args = parser.parse_args(argv)
    if args.workers < 1:
        parser.error("workers must be positive")
    cache_path = Path(args.cache) if args.cache else Path(args.output_dir) / "spec_desc_cache.pt"
    prepare_cache(args.data_dir, cache_path, workers=args.workers)
