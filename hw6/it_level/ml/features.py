from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

from it_level.settings import Settings


def split_xy(df: pd.DataFrame, s: Settings):
    X = df.drop(columns=["level"])
    y = df["level"].astype(str)

    return train_test_split(
        X, y,
        test_size=s.test_size,
        random_state=s.random_state,
        stratify=y,
    )
