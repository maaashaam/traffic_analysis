from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConfig:
    alpha: float = 30.0
    use_log_target: bool = True
    clip_min_pred: float = 0.0
