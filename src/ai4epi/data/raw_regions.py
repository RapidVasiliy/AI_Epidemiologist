"""
Конвертер «сырых» xlsx-выгрузок по городам в формат ``ai4epi.data.local_excel``.

Исходные файлы — из ``RSCF_Uncertainty/generateIncidence/input`` (``Заб <город>.xlsx``,
``ПЦР <город>.xlsx``). Особенности формата:

* в ``Заб`` двухуровневый заголовок: строка с блоками ``ОРЗ`` / ``Грипп`` / ``Население`` и
  строка с возрастными группами; в каждом блоке 8 столбцов
  ``Всего, 0 - 2, 3 - 6, 7 - 14, 15 и ст., 15 - 64, 65 и ст., 3 - 4``;
* Excel превратил часть заголовков групп в даты (``3 - 6`` → 2021-06-03, ``7 - 14`` →
  2014-07-01, ``3 - 4`` → 2021-04-03) или в ``3-Jun`` / ``Jul-14``;
* в метке недели номер и даты разделены неразрывным пробелом (``2021.1\\xa0(04.01.2021 - ...)``);
* в ``ПЦР`` в первой ячейке заголовка — название региона, а не ``Дата``.

Блок ``Грипп`` (клинический диагноз) не используется: оценка гриппа в проекте строится по
ОРВИ и доле положительных ПЦР, как в ``generateIncidence.py``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd

from ai4epi.data.local_excel import PCR_TESTED_COLUMN, week_start_from_label

GROUP_LABELS: list[str] = ["Всего", "0 - 2", "3 - 6", "7 - 14", "15 и ст.", "15 - 64", "65 и ст.", "3 - 4"]

# Заголовки, испорченные Excel, -> нормальные метки групп.
_DATE_LABELS: dict[tuple[int, int], str] = {(6, 3): "3 - 6", (7, 1): "7 - 14", (4, 3): "3 - 4"}
_TEXT_LABELS: dict[str, str] = {"3-Jun": "3 - 6", "Jul-14": "7 - 14", "3-Apr": "3 - 4", "Apr-03": "3 - 4"}

PCR_COLUMNS: list[str] = [
    PCR_TESTED_COLUMN,
    "A (субтип не определен)",
    "A(H1)pdm09",
    "A(H3)",
    "B",
]

_WEEK_LABEL_RE = re.compile(r"^\s*\d{4}\.\d{1,2}\s*\(\d{2}\.\d{2}\.\d{4}\s*-\s*\d{2}\.\d{2}\.\d{4}\)\s*$")


@dataclass(frozen=True)
class RawCity:
    code: str
    name_ru: str
    cases_file: str
    pcr_file: str


RAW_CITIES: dict[str, RawCity] = {
    "moscow": RawCity("moscow", "Москва", "Заб Мск.xlsx", "ПЦР Мск.xlsx"),
    "novosibirsk": RawCity("novosibirsk", "Новосибирск", "Заб Новосиб.xlsx", "ПЦР Новосиб.xlsx"),
    "omsk": RawCity("omsk", "Омск", "Заб Омск.xlsx", "ПЦР Омск.xlsx"),
    "voronezh": RawCity("voronezh", "Воронеж", "Заб Воронеж.xlsx", "ПЦР Воронеж.xlsx"),
    "krasnodar": RawCity("krasnodar", "Краснодар", "Заб Краснодар.xlsx", "ПЦР Краснодар.xlsx"),
    # СПб уже есть в data/regions; сырой файл используется для сверки конвертера.
    "spb_raw": RawCity("spb_raw", "Санкт-Петербург", "Заб СПб 1990.01 - 2020.43_red.xlsx", "ПЦР СПб 2010.01 - 2020.43.xlsx"),
}


class RawFormatError(ValueError):
    """Сырой файл не соответствует ожидаемому формату."""


def normalize_group_label(value: object) -> str:
    """Привести заголовок возрастной группы к виду ``3 - 6``."""

    if isinstance(value, (datetime, pd.Timestamp)):
        key = (value.month, value.day)
        if key in _DATE_LABELS:
            return _DATE_LABELS[key]
        raise RawFormatError(f"Неизвестный заголовок-дата {value!r}.")
    text = str(value).replace("\xa0", " ").strip()
    return _TEXT_LABELS.get(text, text)


def normalize_week_label(value: object) -> str:
    return re.sub(r"\s+", " ", str(value).replace("\xa0", " ")).strip()


def _is_week_label(value: object) -> bool:
    return isinstance(value, str) and bool(_WEEK_LABEL_RE.match(value.replace("\xa0", " ")))


def read_raw_cases(path: str | Path) -> pd.DataFrame:
    """Прочитать ``Заб <город>.xlsx`` → таблица в формате ``cases.xlsx`` (ОРВИ и население)."""

    raw = pd.read_excel(path, header=None)
    block_row = next((i for i in range(min(10, len(raw))) if "ОРЗ" in raw.iloc[i].astype(str).tolist()), None)
    if block_row is None:
        raise RawFormatError(f"{path}: не найдена строка заголовка с блоком 'ОРЗ'.")
    blocks = {str(v).strip(): j for j, v in enumerate(raw.iloc[block_row]) if isinstance(v, str)}
    for name in ("ОРЗ", "Население"):
        if name not in blocks:
            raise RawFormatError(f"{path}: нет блока {name!r}.")
    group_row = raw.iloc[block_row + 1]

    out = pd.DataFrame()
    data = raw.iloc[block_row + 2 :]
    data = data[data.iloc[:, 0].map(_is_week_label)]
    out["Дата"] = data.iloc[:, 0].map(normalize_week_label).to_numpy()
    for block, suffix in (("ОРЗ", "_орви"), ("Население", "")):
        start = blocks[block]
        labels = [normalize_group_label(group_row.iloc[start + k]) for k in range(len(GROUP_LABELS))]
        if labels != GROUP_LABELS:
            raise RawFormatError(f"{path}: блок {block!r} — группы {labels}, ожидались {GROUP_LABELS}.")
        for k, label in enumerate(labels):
            out[f"{label}{suffix}"] = pd.to_numeric(data.iloc[:, start + k], errors="coerce").to_numpy()
    return out


def read_raw_pcr(path: str | Path) -> pd.DataFrame:
    """Прочитать ``ПЦР <город>.xlsx`` → таблица в формате ``pcr.xlsx``."""

    raw = pd.read_excel(path, header=None)
    header = [normalize_group_label(v) if isinstance(v, str) else v for v in raw.iloc[0]]
    missing = [c for c in PCR_COLUMNS if c not in header]
    if missing:
        raise RawFormatError(f"{path}: нет столбцов {missing}.")
    data = raw.iloc[1:]
    data = data[data.iloc[:, 0].map(_is_week_label)]
    out = pd.DataFrame({"Дата": data.iloc[:, 0].map(normalize_week_label).to_numpy()})
    for j, name in enumerate(header):
        if isinstance(name, str) and j >= 2:
            out[name] = pd.to_numeric(data.iloc[:, j], errors="coerce").to_numpy()
    return out


@dataclass(frozen=True)
class ConversionReport:
    city: str
    cases_rows: int
    pcr_rows: int
    first_week: str
    last_week: str
    pcr_missing_weeks: int
    duplicate_weeks_dropped: int


def _dedupe(frame: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    starts = frame["Дата"].map(week_start_from_label)
    keep = ~starts.duplicated(keep="last")
    return frame[keep.to_numpy()].reset_index(drop=True), int((~keep).sum())


def convert_raw_city(
    city: str,
    *,
    raw_dir: str | Path,
    out_dir: str | Path,
    start_year: int = 2010,
) -> ConversionReport:
    """Сконвертировать сырые xlsx города в ``<out_dir>/<city>/{cases,pcr}.xlsx``.

    Ряд ОРВИ обрезается с ``start_year`` (ПЦР в выгрузках начинается с 2010 г.).
    При повторе недели остаётся последняя строка.
    """

    spec = RAW_CITIES[city]
    raw_dir = Path(raw_dir)
    cases = read_raw_cases(raw_dir / spec.cases_file)
    pcr = read_raw_pcr(raw_dir / spec.pcr_file)
    cases, dup_c = _dedupe(cases)
    pcr, dup_p = _dedupe(pcr)
    starts = cases["Дата"].map(week_start_from_label)
    cases = cases[(starts >= pd.Timestamp(start_year, 1, 1)).to_numpy()].reset_index(drop=True)
    target = Path(out_dir) / city
    target.mkdir(parents=True, exist_ok=True)
    cases.to_excel(target / "cases.xlsx", index=False)
    pcr.to_excel(target / "pcr.xlsx", index=False)
    pcr_starts = set(pcr["Дата"].map(week_start_from_label))
    tested = pcr.set_index(pcr["Дата"].map(week_start_from_label))[PCR_TESTED_COLUMN]
    missing = sum(
        1 for s in cases["Дата"].map(week_start_from_label)
        if s not in pcr_starts or not (tested.get(s, 0) > 0)
    )
    return ConversionReport(
        city=city,
        cases_rows=len(cases),
        pcr_rows=len(pcr),
        first_week=str(cases["Дата"].iloc[0]),
        last_week=str(cases["Дата"].iloc[-1]),
        pcr_missing_weeks=missing,
        duplicate_weeks_dropped=dup_c + dup_p,
    )
