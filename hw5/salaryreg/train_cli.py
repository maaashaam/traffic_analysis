from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np

from salaryreg.config import ModelConfig
from salaryreg.errors import AppError
from salaryreg.io_npy import load_x, load_y
from salaryreg.model import fit_ridge
from salaryreg.persistence import save_model


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="salaryreg.train_cli",
        description="Train ridge regression and save weights to ./resources/model.npz",
    )
    parser.add_argument("x_path", type=str, help="Path to x_data.npy")
    parser.add_argument("y_path", type=str, help="Path to y_data.npy")
    parser.add_argument("--alpha", type=float, default=ModelConfig.alpha)
    parser.add_argument("--no-log", action="store_true", help="Disable log1p target")
    args = parser.parse_args()

    cfg = ModelConfig(alpha=float(args.alpha), use_log_target=not args.no_log)

    try:
        x = load_x(Path(args.x_path))
        y = load_y(Path(args.y_path))

        row_ok = np.isfinite(x).all(axis=1) & np.isfinite(y)
        x = x[row_ok]
        y = y[row_ok]

        y_ok = (y >= 10_000) & (y <= 1_000_000)
        x = x[y_ok]
        y = y[y_ok]

    except AppError as e:
        print(str(e))
        return 1

    if x.shape[0] != y.shape[0]:
        print(f"Shape mismatch: X rows {x.shape[0]} != y {y.shape[0]}")
        return 1

    model = fit_ridge(x, y, alpha=cfg.alpha, use_log_target=cfg.use_log_target)

    project_dir = Path(__file__).resolve().parents[1]
    out_path = project_dir / "resources" / "model.npz"

    meta = {
        "alpha": cfg.alpha,
        "use_log_target": cfg.use_log_target,
        "n_rows": int(x.shape[0]),
        "n_features": int(x.shape[1]),
    }
    save_model(out_path, model, meta=meta)

    print(f"Saved model: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
