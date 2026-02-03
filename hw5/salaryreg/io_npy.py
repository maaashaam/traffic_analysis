from __future__ import annotations

from pathlib import Path
import numpy as np

from salaryreg.errors import AppError


def load_x(path: Path) -> np.ndarray:
    if not path.exists():
        raise AppError(f"File not found: {path}")

    try:
        x = np.load(path)
    except Exception as e:
        raise AppError(f"Cannot read npy file: {path}. Error: {e}")

    if x.ndim != 2:
        raise AppError(f"x_data must be 2D array, got ndim={x.ndim} for {path}")

    return x.astype(np.float32, copy=False)


def load_y(path: Path) -> np.ndarray:
    if not path.exists():
        raise AppError(f"File not found: {path}")

    try:
        y = np.load(path)
    except Exception as e:
        raise AppError(f"Cannot read npy file: {path}. Error: {e}")

    if y.ndim != 1:
        raise AppError(f"y_data must be 1D array, got ndim={y.ndim} for {path}")

    return y.astype(np.float32, copy=False)
