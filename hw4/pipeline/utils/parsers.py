from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional


_RE_AGE = re.compile(r"(\d{1,3})\s*год")
_RE_GENDER = re.compile(r"^(Мужчина|Женщина)")
_RE_SALARY = re.compile(r"([\d\s]+)\s*(руб\.|USD|EUR)?", re.IGNORECASE)
_RE_EXP = re.compile(
    r"Опыт работы\s+(?:(\d+)\s*лет?)?\s*(?:(\d+)\s*месяц\w*)?",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Salary:
    amount: int
    currency: str = "RUB"


def parse_gender(text: Optional[str]) -> Optional[str]:
    if not text:
        return None
    m = _RE_GENDER.search(text)
    return m.group(1) if m else None


def parse_age(text: Optional[str]) -> Optional[int]:
    if not text:
        return None
    m = _RE_AGE.search(text)
    return int(m.group(1)) if m else None


def parse_salary(text: Optional[str]) -> Optional[Salary]:
    """Parse salary string like '27 000 руб.'.

    Returns None if can't parse or currency is not RUB.
    """
    if not text:
        return None
    m = _RE_SALARY.search(text)
    if not m:
        return None
    digits, cur = m.group(1), (m.group(2) or "руб.").lower()
    amount = int(re.sub(r"\s+", "", digits))
    if "руб" in cur:
        return Salary(amount=amount, currency="RUB")
    return None


def parse_city(text: Optional[str]) -> Optional[str]:
    if not text:
        return None
    return text.split(",")[0].strip()


def parse_experience_months(text: Optional[str]) -> Optional[int]:
    if not text:
        return None
    m = _RE_EXP.search(text)
    if not m:
        return None
    years = int(m.group(1)) if m.group(1) else 0
    months = int(m.group(2)) if m.group(2) else 0
    total = years * 12 + months
    return total if total > 0 else None


def parse_has_car(text: Optional[str]) -> Optional[int]:
    if not text:
        return None
    t = text.lower()
    if "имеется" in t or "есть" in t:
        return 1
    if "нет" in t:
        return 0
    return None


def parse_education_level(text: Optional[str]) -> Optional[str]:
    if not text:
        return None
    first_line = text.splitlines()[0].strip()
    tokens = first_line.split()
    return " ".join(tokens[:3]) if tokens else None
