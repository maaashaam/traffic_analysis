from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from salaryreg.nn import FCNRegressor


@dataclass(frozen=True)
class Scaler:
    mean: np.ndarray
    std: np.ndarray
    use_log_target: bool

    def transform(self, x: np.ndarray) -> np.ndarray:
        std = np.where(self.std > 1e-8, self.std, 1.0)
        return (x - self.mean) / std


def save_scaler(path: Path, scaler: Scaler) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        mean=scaler.mean.astype(np.float32),
        std=scaler.std.astype(np.float32),
        use_log_target=np.array([1 if scaler.use_log_target else 0], dtype=np.int8),
        version=np.array([1], dtype=np.int32),
    )


def load_scaler(path: Path) -> Scaler:
    d = np.load(path)
    mean = d["mean"].astype(np.float32, copy=False)
    std = d["std"].astype(np.float32, copy=False)
    use_log_target = bool(int(d["use_log_target"].reshape(-1)[0]))
    return Scaler(mean=mean, std=std, use_log_target=use_log_target)


def save_model(path: Path, model: FCNRegressor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


def load_model(path: Path, input_dim: int, hidden_dims: list[int], dropout: float) -> FCNRegressor:
    model = FCNRegressor(input_dim=input_dim, hidden_dims=hidden_dims, dropout=dropout)
    state = torch.load(path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model