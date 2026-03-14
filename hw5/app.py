from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from salaryreg.nn_out import load_model, load_scaler


def main() -> int:
    parser = argparse.ArgumentParser(prog="app", description="Predict salaries from x_data.npy")
    parser.add_argument("x_path", type=str, help="Path to x_data.npy")
    args = parser.parse_args()

    x_path = Path(args.x_path).expanduser().resolve()
    if not x_path.exists():
        raise FileNotFoundError(f"x_data.npy not found: {x_path}")

    project_dir = Path(__file__).resolve().parent
    resources = project_dir / "resources"
    model_path = resources / "model.pt"
    scaler_path = resources / "scaler.npz"

    if not model_path.exists() or not scaler_path.exists():
        raise FileNotFoundError("NN weights not found in resources/. Train first.")

    x = np.load(x_path).astype(np.float32, copy=False)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    scaler = load_scaler(scaler_path)
    x_n = scaler.transform(x).astype(np.float32, copy=False)

    hidden_dims = [512, 256, 128]
    dropout = 0.05
    model = load_model(model_path, input_dim=x.shape[1], hidden_dims=hidden_dims, dropout=dropout)

    with torch.no_grad():
        pred_t = model(torch.from_numpy(x_n)).numpy().reshape(-1)

    pred = np.expm1(pred_t) if scaler.use_log_target else pred_t
    pred = np.clip(pred, 0, None)

    print(json.dumps(pred.astype(float).tolist(), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())