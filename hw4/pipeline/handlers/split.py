from __future__ import annotations

import logging

import numpy as np

from pipeline.context import PipelineContext
from pipeline.handlers.base import Handler

logger = logging.getLogger(__name__)


class SplitXYHandler(Handler):
    """Split numeric dataframe into X and y numpy arrays."""

    def handle(self, ctx: PipelineContext) -> PipelineContext:
        if ctx.df is None:
            raise ValueError("DataFrame is missing.")
        if "target_salary_rub" not in ctx.df.columns:
            raise ValueError("Target column 'target_salary_rub' missing.")

        df = ctx.df
        y = df["target_salary_rub"].to_numpy(dtype=np.float32)
        x = df.drop(columns=["target_salary_rub"]).to_numpy(dtype=np.float32)

        ctx.x = x
        ctx.y = y
        ctx.metadata["n_samples"] = int(x.shape[0])
        ctx.metadata["n_features"] = int(x.shape[1])

        logger.info("X shape: %s, y shape: %s", x.shape, y.shape)
        return ctx
