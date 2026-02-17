from __future__ import annotations

from pathlib import Path
import pandas as pd


def read_hh_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(
        path,
        engine="python",
        sep=",",
        quotechar='"',
        encoding="utf-8",
        on_bad_lines="skip",
    )
