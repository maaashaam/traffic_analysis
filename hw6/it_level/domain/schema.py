from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class HHColumns:
    title: str = "Ищет работу на должность:"
    gender_age: str = "Пол, возраст"
    salary: str = "ЗП"
    city: str = "Город"
    exp: str = "Опыт (двойное нажатие для полной версии)"
    skills: str = "Ключевые навыки"
