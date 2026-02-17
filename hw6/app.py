from __future__ import annotations

import argparse
from pathlib import Path

from it_level.settings import Settings
from it_level.usecases.run_poc import run_poc


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="app",
        description="PoC: classify IT developer level (junior/middle/senior) from hh.ru resumes",
    )
    parser.add_argument("csv_path", type=str, help="Path to hh.csv")
    args = parser.parse_args()

    csv_path = Path(args.csv_path).expanduser().resolve()
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    settings = Settings()
    run_poc(csv_path=csv_path, settings=settings)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
