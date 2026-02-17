from __future__ import annotations

import re
from typing import Optional

from it_level.settings import Settings


IT_KEYWORDS = (
    "разработчик", "developer", "programmer", "software",
    "backend", "frontend", "fullstack",
    "golang", "java", "python", "javascript", "typescript", "node",
    "c#", "c++", ".net", "php", "kotlin", "swift", "android", "ios",
    "django", "flask", "spring", "react", "vue", "angular",
)

JUN_PAT = re.compile(r"\b(junior|джуниор|стаж(е|ё)р|intern)\b", re.IGNORECASE)
MID_PAT = re.compile(r"\b(middle|миддл|mid)\b", re.IGNORECASE)
SEN_PAT = re.compile(r"\b(senior|сеньор|lead|team\s*lead|тимлид|ведущ(ий|ая)|главн(ый|ая))\b", re.IGNORECASE)


def is_it_resume(title: str, text_blob: str) -> bool:
    t = (title or "").lower()
    b = (text_blob or "").lower()
    return any(k in t for k in IT_KEYWORDS) or any(k in b for k in IT_KEYWORDS)


def infer_level(title: str, exp_months: Optional[int], s: Settings) -> str:
    if JUN_PAT.search(title or ""):
        return "junior"
    if MID_PAT.search(title or ""):
        return "middle"
    if SEN_PAT.search(title or ""):
        return "senior"

    if exp_months is None:
        return "middle"
    if exp_months <= s.junior_max_months:
        return "junior"
    if exp_months >= s.senior_min_months:
        return "senior"
    return "middle"
