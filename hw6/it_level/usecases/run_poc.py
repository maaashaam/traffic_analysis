from __future__ import annotations

from pathlib import Path

import pandas as pd

from it_level.domain.schema import HHColumns
from it_level.domain.rules import infer_level, is_it_resume
from it_level.infr.hh_csv import read_hh_csv
from it_level.infr.parsing import (
    clean_city,
    parse_experience_months,
    parse_gender_age,
    parse_salary_rub,
)
from it_level.infr.report_writer import save_class_balance_plot
from it_level.ml.features import split_xy
from it_level.ml.metrics import print_report
from it_level.ml.pipeline import make_sklearn_pipeline
from it_level.settings import Settings


def run_poc(csv_path: Path, settings: Settings) -> None:
    cols = HHColumns()
    df = read_hh_csv(csv_path)

    out = df.copy()

    out["title"] = out[cols.title].fillna("").astype(str)

    ga = out[cols.gender_age].map(parse_gender_age)
    out["gender"] = ga.map(lambda t: t[0])
    out["age"] = ga.map(lambda t: t[1])

    out["salary_rub"] = out[cols.salary].map(parse_salary_rub)
    out["city"] = out[cols.city].map(clean_city)
    out["exp_months"] = out[cols.exp].map(parse_experience_months)

    if cols.skills in out.columns:
        out["skills"] = out[cols.skills].fillna("").astype(str)
    else:
        out["skills"] = ""

    out["blob"] = (
        out["title"].fillna("").astype(str)
        + " "
        + out["skills"].fillna("").astype(str)
        + " "
        + out[cols.exp].fillna("").astype(str)
    )

    out = out[out.apply(lambda r: is_it_resume(r["title"], r["blob"]), axis=1)].copy()

    out = out.dropna(subset=["age", "gender", "exp_months", "salary_rub"]).copy()

    out = out[(out["salary_rub"] >= 10_000) & (out["salary_rub"] <= 1_000_000)].copy()

    out["exp_months"] = out["exp_months"].clip(0, 720)

    out["text"] = (out["title"].fillna("").astype(str) + " " + out["skills"].fillna("").astype(str)).str.strip()

    out = out[out["text"].str.len() > 0].copy()

    out["level"] = out.apply(
        lambda r: infer_level(r["title"], int(r["exp_months"]), settings),
        axis=1,
    )

    dataset = out[
        ["salary_rub", "city", "age", "gender", "exp_months", "text", "level"]
    ].reset_index(drop=True)

    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    save_class_balance_plot(dataset["level"], reports_dir / "class_balance.png")

    print("Dataset size:", dataset.shape)
    print("Empty text:", int((dataset["text"].str.strip() == "").sum()))

    X_train, X_test, y_train, y_test = split_xy(dataset, settings)

    X_train = X_train.copy()
    X_test = X_test.copy()
    X_train["text"] = X_train["text"].fillna("").astype(str)
    X_test["text"] = X_test["text"].fillna("").astype(str)

    pipe = make_sklearn_pipeline(settings)
    pipe.fit(X_train, y_train)

    pred = pipe.predict(X_test)
    print_report(y_test, pred)

    print("\nPredicted class counts:\n", pd.Series(pred).value_counts().to_string())
