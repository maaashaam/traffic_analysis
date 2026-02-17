from __future__ import annotations

import re
from typing import Optional

_SPACE_RE = re.compile(r"[\s\u00A0]+")


def _ns(s: str) -> str:
    return _SPACE_RE.sub(" ", s).strip()


def parse_salary_rub(val: object) -> Optional[float]:
    if val is None:
        return None
    txt = _ns(str(val)).lower()
    digits = re.findall(r"\d+", txt)
    return float("".join(digits)) if digits else None


def parse_gender_age(val: object) -> tuple[Optional[int], Optional[int]]:
    if val is None:
        return None, None
    txt = _ns(str(val)).lower()
    gender = 1 if "муж" in txt else 0 if "жен" in txt else None
    m = re.search(r"(\d{1,3})\s*год", txt)
    age = int(m.group(1)) if m else None
    return gender, age


def parse_experience_months(val: object) -> Optional[int]:
    if val is None:
        return None
    txt = _ns(str(val)).lower()
    y = 0
    mth = 0
    my = re.search(r"(\d+)\s*(лет|года|год)", txt)
    if my:
        y = int(my.group(1))
    mm = re.search(r"(\d+)\s*(месяц|месяца|месяцев)", txt)
    if mm:
        mth = int(mm.group(1))
    total = y * 12 + mth
    return total if total > 0 else None


def clean_city(val: object) -> str:
    if val is None:
        return ""
    txt = _ns(str(val))
    return txt.split(",")[0].strip() if txt else ""
