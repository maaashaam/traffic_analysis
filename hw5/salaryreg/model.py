from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class LinearModel:
    """
    Linear Regression: y = X_norm @ w + b

    X_norm = (X - mean) / std
    """
    w: np.ndarray          # (D,)
    b: float
    mean: np.ndarray       # (D,)
    std: np.ndarray        # (D,)
    use_log_target: bool

    def predict(self, x: np.ndarray) -> np.ndarray:
        x = _as_2d(x).astype(np.float32, copy=False)
        x_norm = (x - self.mean) / self.std
        pred = x_norm @ self.w + self.b

        if self.use_log_target:
            pred = np.expm1(pred)

        return pred.astype(np.float32, copy=False)


def fit_ridge(
    x: np.ndarray,
    y: np.ndarray,
    alpha: float,
    use_log_target: bool,
) -> LinearModel:
    """
    Ridge closed-form:
      w = (X^T X + alpha I)^-1 X^T y
    """
    x = _as_2d(x).astype(np.float64, copy=False)
    y = y.reshape(-1).astype(np.float64, copy=False)

    if x.shape[0] != y.shape[0]:
        raise ValueError(f"X rows != y size: {x.shape[0]} != {y.shape[0]}")

    if use_log_target:
        y = np.log1p(y)

    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std[std < 1e-8] = 1.0
    x_norm = (x - mean) / std

    y_mean = float(y.mean())
    y_center = y - y_mean

    d = x_norm.shape[1]
    sqrt_a = np.sqrt(alpha)

    x_aug = np.vstack([x_norm, sqrt_a * np.eye(d, dtype=np.float64)])
    y_aug = np.concatenate([y_center, np.zeros(d, dtype=np.float64)])

    w, *_ = np.linalg.lstsq(x_aug, y_aug, rcond=None)
    b = y_mean

    return LinearModel(
        w=w.astype(np.float32),
        b=float(b),
        mean=mean.astype(np.float32),
        std=std.astype(np.float32),
        use_log_target=bool(use_log_target),
    )


def _as_2d(x: np.ndarray) -> np.ndarray:
    if x.ndim == 1:
        return x.reshape(1, -1)
    if x.ndim != 2:
        raise ValueError(f"X must be 2D, got ndim={x.ndim}")
    return x
