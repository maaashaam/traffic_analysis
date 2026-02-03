from __future__ import annotations

from pathlib import Path
import numpy as np

from salaryreg.config import ModelConfig
from salaryreg.errors import AppError
from salaryreg.io_npy import load_x
from salaryreg.persistence import load_model


def predict_from_file(x_path: Path, cfg: ModelConfig = ModelConfig()) -> list[float]:
    x = load_x(x_path)

    project_dir = Path(__file__).resolve().parents[1]
    model_path = project_dir / "resources" / "model.npz"

    model = load_model(model_path)
    pred = model.predict(x)
    pred = np.clip(pred, 0, None)

    pred = np.clip(pred, cfg.clip_min_pred, None)

    return pred.astype(float).tolist()
