from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import tempfile
import zipfile
import numpy as np

from salaryreg.errors import AppError
from salaryreg.model import LinearModel


def save_model(path: Path, model: LinearModel, meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    tmp_dir = path.parent
    with tempfile.NamedTemporaryFile(dir=tmp_dir, delete=False, suffix=".npz") as tmp:
        tmp_path = Path(tmp.name)

    try:
        np.savez_compressed(
            tmp_path,
            w=model.w,
            b=np.array([model.b], dtype=np.float32),
            mean=model.mean,
            std=model.std,
            use_log_target=np.array([1 if model.use_log_target else 0], dtype=np.int8),
            meta=np.array([repr(meta)], dtype=object),
            version=np.array([1], dtype=np.int32),
        )
        tmp_path.replace(path)
    except Exception as e:
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass
        raise AppError(f"Failed to save model to {path}: {e}")


def load_model(path: Path) -> LinearModel:
    if not path.exists():
        raise AppError(
            f"Model not found: {path}\n"
            f"Train it first: python -m salaryreg.train_cli path/to/x_data.npy path/to/y_data.npy"
        )

    if path.stat().st_size == 0:
        raise AppError(f"Model file is empty: {path}. Retrain the model.")

    if not zipfile.is_zipfile(path):
        head = path.read_bytes()[:16]
        raise AppError(f"Model file is not a valid .npz (zip). Head: {head!r}. Path: {path}")

    try:
        data = np.load(path, allow_pickle=True)
        w = data["w"].astype(np.float32, copy=False)
        b = float(data["b"].reshape(-1)[0])
        mean = data["mean"].astype(np.float32, copy=False)
        std = data["std"].astype(np.float32, copy=False)
        use_log_target = bool(int(data["use_log_target"].reshape(-1)[0]))
    except Exception as e:
        raise AppError(f"Cannot load model from {path}: {e}")

    return LinearModel(w=w, b=b, mean=mean, std=std, use_log_target=use_log_target)
