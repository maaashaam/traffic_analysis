from __future__ import annotations

import logging

from pipeline.context import PipelineContext
from pipeline.handlers.base import Handler
from pipeline.utils.parsers import (
    parse_age,
    parse_city,
    parse_education_level,
    parse_experience_months,
    parse_gender,
    parse_has_car,
    parse_salary,
)

logger = logging.getLogger(__name__)


class FeatureEngineeringHandler(Handler):
    """Create target and cleaned feature columns."""

    def handle(self, ctx: PipelineContext) -> PipelineContext:
        if ctx.df is None:
            raise ValueError("DataFrame is missing.")
        df = ctx.df.copy()

        salary_parsed = df["ЗП"].map(parse_salary)
        df["target_salary_rub"] = salary_parsed.map(lambda s: s.amount if s else None)

        df["gender"] = df["Пол, возраст"].map(parse_gender)
        df["age"] = df["Пол, возраст"].map(parse_age)
        df["city"] = df["Город"].map(parse_city)
        df["experience_months"] = df["Опыт (двойное нажатие для полной версии)"].map(
            parse_experience_months
        )
        df["has_car"] = df["Авто"].map(parse_has_car)
        df["education_level"] = df["Образование и ВУЗ"].map(parse_education_level)

        df["employment"] = df["Занятость"]
        df["schedule"] = df["График"]
        df["desired_position"] = df["Ищет работу на должность:"]

        before = len(df)
        df = df.dropna(subset=["target_salary_rub"])
        logger.info("Dropped rows without target: %s", before - len(df))

        ctx.df = df
        return ctx
