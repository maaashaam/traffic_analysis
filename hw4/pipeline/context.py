from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

@dataclass
class PipelineContext:
    """Mutable context passed through the chain."""

    csv_path: Path
    df: Optional[pd.DataFrame] = None
    x: Optional[np.ndarray] = None
    y: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def output_dir(self) -> Path:
        return self.csv_path.parent

    @property
    def x_path(self) -> Path:
        return self.output_dir / "x_data.npy"

    @property
    def y_path(self) -> Path:
        return self.output_dir / "y_data.npy"