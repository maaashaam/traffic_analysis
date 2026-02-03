from __future__ import annotations

import argparse
import json
from pathlib import Path

from salaryreg.predict import predict_from_file
from salaryreg.errors import AppError


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="app",
        description="Predict salaries (RUB) from x_data.npy using weights in ./resources/model.npz",
    )
    parser.add_argument("x_path", type=str, help="Path to x_data.npy")
    args = parser.parse_args()

    try:
        preds = predict_from_file(Path(args.x_path))
    except AppError as e:
        print(str(e))
        return 1

    print(json.dumps(preds, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
