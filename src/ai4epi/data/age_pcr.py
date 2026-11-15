"""
Результаты ПЦР на грипп по возрастным группам (общероссийская недельная выгрузка).

Источник — ``export_LABD_PCR_weekly_201740_201839.xlsx`` из RSCF_Uncertainty
(сезон 2017/18, 40-я неделя 2017 г. — 39-я неделя 2018 г., 55 городов). В выгрузке по
каждому учреждению и неделе — число протестированных (``SWB_*``) и положительных на грипп
A без подтипа (``A_*``), A(H1N1)pdm09 (``PDM_*``), A(H3N2) (``H3_*``), B (``B_*``) и сезонный
A(H1) (``H1_*``) в группах 0–2, 3–6, 7–14 и 15+ лет.

В проект выгрузка сведена в компактный CSV (``data/national/age_pcr_2017w40_2018w39.csv``,
функция :func:`extract_age_pcr`): суммы по учреждениям города за неделю.

Зачем: доля положительных сильно зависит от возраста (за сезон по всем городам 0–2 года —
0.09, 15+ — 0.21), а возрастные ряды по умолчанию строятся в допущении одинаковой доли
(``ai4epi.data.age_groups``). Там, где возрастная ПЦР есть, она заменяет это допущение.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

DEFAULT_AGE_PCR_PATH = Path("data") / "national" / "age_pcr_2017w40_2018w39.csv"

# Возрастные группы выгрузки -> индексы ``sars_cases_age_group_k`` таблицы cases.
AGE_PCR_GROUPS: dict[int, str] = {0: "0_2", 1: "3_6", 2: "7_14", 3: "15"}
FLU_PREFIXES: tuple[str, ...] = ("A", "PDM", "H3", "B", "H1")

# Код города проекта -> DISTRICT_NAME выгрузки.
CITY_TO_DISTRICT: dict[str, str] = {
    "spb": "Санкт-Петербург",
    "samara": "Самара",
    "chelyabinsk": "Челябинск",
    "moscow": "Москва",
    "novosibirsk": "Новосибирск",
    "omsk": "Омск",
    "voronezh": "Воронеж",
    "krasnodar": "Краснодар",
}


def extract_age_pcr(xlsx_path: str | Path) -> pd.DataFrame:
    """Свести выгрузку ``export_LABD_PCR_weekly_*.xlsx`` к таблице город × неделя.

    Столбцы: ``district``, ``iso_year``, ``iso_week``, ``tested_<g>``, ``positive_<g>`` для
    g ∈ {0_2, 3_6, 7_14, 15, total}; ``positive`` — сумма положительных на все типы гриппа.
    """

    raw = pd.read_excel(xlsx_path)
    groups = [*AGE_PCR_GROUPS.values(), "TOTAL"]
    frame = pd.DataFrame({"district": raw["DISTRICT_NAME"], "iso_year": raw["YEAR"], "iso_week": raw["WEEK"]})
    for g in groups:
        name = g.lower()
        frame[f"tested_{name}"] = pd.to_numeric(raw[f"SWB_{g}"], errors="coerce").fillna(0)
        frame[f"positive_{name}"] = sum(
            pd.to_numeric(raw[f"{p}_{g}"], errors="coerce").fillna(0) for p in FLU_PREFIXES
        )
    out = frame.groupby(["district", "iso_year", "iso_week"], as_index=False).sum()
    return out.sort_values(["district", "iso_year", "iso_week"]).reset_index(drop=True)


def load_age_pcr(
    city: str,
    *,
    path: str | Path = DEFAULT_AGE_PCR_PATH,
) -> pd.DataFrame | None:
    """Возрастная ПЦР города, индексированная датой начала ISO-недели; ``None``, если её нет."""

    district = CITY_TO_DISTRICT.get(city)
    path = Path(path)
    if district is None or not path.is_file():
        return None
    table = pd.read_csv(path)
    table = table[table["district"] == district]
    if table.empty:
        return None
    starts = [
        pd.Timestamp(date.fromisocalendar(int(y), int(w), 1))
        for y, w in zip(table["iso_year"], table["iso_week"], strict=True)
    ]
    table = table.drop(columns=["district", "iso_year", "iso_week"]).set_index(pd.DatetimeIndex(starts, name="datetime"))
    return table.sort_index()


def attach_age_pcr(cases: pd.DataFrame, age_pcr: pd.DataFrame | None) -> pd.DataFrame:
    """Добавить к таблице ``cases`` столбцы ``age_pcr_tested_k`` / ``age_pcr_positive_k``.

    k — индекс группы ``sars_cases_age_group_k`` (0: 0–2, 1: 3–6, 2: 7–14, 3: 15+). На неделях
    без возрастной ПЦР значения — NaN.
    """

    if age_pcr is None:
        return cases
    out = cases.copy()
    when = pd.to_datetime(out["datetime"])
    for k, g in AGE_PCR_GROUPS.items():
        out[f"age_pcr_tested_{k}"] = when.map(age_pcr[f"tested_{g}"]).astype(float).to_numpy()
        out[f"age_pcr_positive_{k}"] = when.map(age_pcr[f"positive_{g}"]).astype(float).to_numpy()
    return out


def season_positivity(age_pcr: pd.DataFrame) -> dict[str, float]:
    """Доля положительных по возрастным группам за весь период выгрузки."""

    return {
        g: float(age_pcr[f"positive_{g}"].sum() / age_pcr[f"tested_{g}"].sum())
        for g in [*AGE_PCR_GROUPS.values(), "total"]
        if age_pcr[f"tested_{g}"].sum() > 0
    }


__all__ = [
    "AGE_PCR_GROUPS",
    "CITY_TO_DISTRICT",
    "DEFAULT_AGE_PCR_PATH",
    "attach_age_pcr",
    "extract_age_pcr",
    "load_age_pcr",
    "season_positivity",
]
