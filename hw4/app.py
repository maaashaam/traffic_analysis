from __future__ import annotations

import argparse
import logging
from pathlib import Path

from pipeline.pipeline import run

def parse_args():
    parser = argparse.ArgumentParser(
        prog="app",
        description="CSV preprocessing pipeline",
    )
    parser.add_argument("csv_path", type=Path, help="path to csv file")
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO)
    ctx = run(args.csv_path.resolve())

if __name__ == "__main__":
    main()
