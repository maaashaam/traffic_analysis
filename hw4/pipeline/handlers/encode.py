from __future__ import annotations

import logging

import pandas as pd

from pipeline.context import PipelineContext
from pipeline.handlers.base import Handler

logger = logging.getLogger(__name__)


class EncodeHandler(Handler):
    """One-hot encode categoricals and produce a numeric dataframe."""

    def __init__(self, max_categories: int = 200) -> None:
        super().__init__()
        self.max_categories = max_categories

    def handle(self, ctx: PipelineContext) -> PipelineContext:
        if ctx.df is None:
            raise ValueError("DataFrame is missing.")
        df = ctx.df.copy()

        feature_cols = [
            "gender",
            "age",
            "city",
            "experience_months",
            "has_car",
            "education_level",
            "employment",
            "schedule",
            "desired_position",
        ]
        df = df[feature_cols + ["target_salary_rub"]]

        for col in ["city", "desired_position"]:
            top = df[col].value_counts(dropna=True).head(self.max_categories).index
            df[col] = df[col].where(df[col].isin(top), "__other__")

        cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
        df_num = pd.get_dummies(df, columns=cat_cols, dummy_na=True)

        df_num = df_num.reindex(sorted(df_num.columns), axis=1)
        ctx.df = df_num
        logger.info("Encoded feature matrix shape (with target): %s", df_num.shape)
        return ctx
