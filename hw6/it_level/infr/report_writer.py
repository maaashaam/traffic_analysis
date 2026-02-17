from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd


def save_class_balance_plot(y: pd.Series, out_path: Path) -> None:
    counts = y.value_counts().reindex(["junior", "middle", "senior"]).fillna(0).astype(int)
    plt.figure()
    counts.plot(kind="bar")
    plt.title("Class balance: junior / middle / senior")
    plt.xlabel("Level")
    plt.ylabel("Count")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close()

    print(f"Saved plot: {out_path}")
    print("Class counts:\n", counts.to_string())
