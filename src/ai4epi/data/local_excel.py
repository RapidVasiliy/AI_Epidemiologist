"""
Локальный источник эпидемиологических данных: xlsx-выгрузки по городам.

Формат файлов перенесён из ``Influenza_bulletin`` (``model_complex.EpidData``):
каталог ``<data_dir>/<city>/`` содержит

* ``cases.xlsx`` — недельные обращения с ОРВИ по возрастным группам и численность
  населения (столбцы ``Дата``, ``Всего_орви``, ``0 - 2_орви`` ..., ``Всего``,
  ``0 - 2`` ...);
* ``pcr.xlsx`` — число образцов, тестированных на грипп, и число положительных
  по штаммам (``A (субтип не определен)``, ``A(H1)pdm09``, ``A(H3)``, ``B``).

Модуль приводит эти таблицы к тому же контракту ``cases``, который
``ai4epi.data.influenza`` строит из ответа db.influenza.spb.ru, и далее
использует общую функцию ``weekly_incidence_from_cases``. Поэтому весь
численный pipeline работает одинаково для локальных и сетевых данных.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Literal

import pandas as pd

from ai4epi.data.age_pcr import CITY_TO_DISTRICT, attach_age_pcr, load_age_pcr
from ai4epi.data.influenza import (
    INFLUENZA_CASES_SCHEMA,
    STRAIN_NAMES,
    weekly_incidence_from_cases,
)

DEFAULT_LOCAL_DATA_DIR = Path("data") / "regions"

LOCAL_CITY_NAMES_RU: dict[str, str] = {
    "spb": "Санкт-Петербург",
    "samara": "Самара",
    "chelyabinsk": "Челябинск",
    "moscow": "Москва",
    "novosibirsk": "Новосибирск",
    "omsk": "Омск",
    "voronezh": "Воронеж",
    "krasnodar": "Краснодар",
}

# Excel-заголовок -> имя столбца контракта ``cases``.
CASES_COLUMNS: dict[str, str] = {
    "Всего_орви": "sars_total_cases",
    "0 - 2_орви": "sars_cases_age_group_0",
    "3 - 6_орви": "sars_cases_age_group_1",
    "7 - 14_орви": "sars_cases_age_group_2",
    "15 и ст._орви": "sars_cases_age_group_3",
    "15 - 64_орви": "sars_cases_age_group_4",
    "65 и ст._орви": "sars_cases_age_group_5",
    "Всего": "total_population",
    "0 - 2": "population_age_group_0",
    "3 - 6": "population_age_group_1",
    "7 - 14": "population_age_group_2",
    "15 и ст.": "population_age_group_3",
    "15 - 64": "population_age_group_4",
    "65 и ст.": "population_age_group_5",
}

PCR_TESTED_COLUMN = "Число образцов тестированных на грипп"

_DATE_RE = re.compile(r"(\d{2}\.\d{2}\.\d{4})")

PcrGapPolicy = Literal["interpolate", "zero"]


class LocalDataError(ValueError):
    """Локальные xlsx-файлы отсутствуют или не соответствуют формату."""


@dataclass(frozen=True)
class LocalInfluenzaBundle:
    """Результат загрузки локальных данных по одному городу."""

    city: str
    city_name_ru: str
    cases: pd.DataFrame
    weekly: pd.DataFrame
    source_dir: Path
    dropped_tail_weeks: int
    has_15_64_65_split: bool
    pcr_imputed_weeks: int
    pcr_gap_policy: str


def city_meta(city: str, data_dir: str | Path = DEFAULT_LOCAL_DATA_DIR) -> dict[str, str]:
    """Описание города: ``<data_dir>/<city>/meta.json`` (``name_ru``, ``district``) поверх встроенных значений.

    ``district`` — название города в общероссийских выгрузках (``DISTRICT_NAME``), по нему
    подключается возрастная ПЦР. Для нового города достаточно положить meta.json рядом с xlsx.
    """

    meta = {"name_ru": LOCAL_CITY_NAMES_RU.get(city, city)}
    if city in CITY_TO_DISTRICT:
        meta["district"] = CITY_TO_DISTRICT[city]
    path = Path(data_dir) / city / "meta.json"
    if path.is_file():
        meta.update({k: str(v) for k, v in json.loads(path.read_text(encoding="utf-8")).items()})
    meta.setdefault("district", meta["name_ru"])
    return meta


def list_local_cities(data_dir: str | Path = DEFAULT_LOCAL_DATA_DIR) -> list[str]:
    """Вернуть города, для которых в ``data_dir`` есть пара cases/pcr."""

    root = Path(data_dir)
    if not root.is_dir():
        return []
    return sorted(
        p.name for p in root.iterdir() if (p / "cases.xlsx").is_file() and (p / "pcr.xlsx").is_file()
    )


def week_start_from_label(value: Any) -> datetime:
    """Дата начала недели из ячейки ``Дата``.

    Строки вида ``2010.1 (04.01.2010 - 10.01.2010)`` дают первую дату;
    значения, уже прочитанные как дата, возвращаются как есть.
    """

    if isinstance(value, (datetime, pd.Timestamp)):
        return pd.Timestamp(value).to_pydatetime()
    match = _DATE_RE.search(str(value))
    if not match:
        raise LocalDataError(f"Не удалось извлечь дату из значения {value!r}.")
    return datetime.strptime(match.group(1), "%d.%m.%Y")


def load_local_cases(
    city: str,
    *,
    data_dir: str | Path = DEFAULT_LOCAL_DATA_DIR,
    pcr_gap_policy: PcrGapPolicy = "interpolate",
    end_date: date | datetime | str | None = None,
    age_pcr_path: str | Path | Literal["auto"] | None = "auto",
) -> tuple[pd.DataFrame, int]:
    """Прочитать cases.xlsx + pcr.xlsx и привести к контракту ``cases``.

    Недели без данных ПЦР (число тестированных образцов отсутствует или равно
    нулю) обрабатываются по ``pcr_gap_policy``:

    * ``"interpolate"`` (по умолчанию) — доля положительных проб по каждому
      штамму линейно интерполируется между соседними неделями с данными;
      такие недели помечаются ``pcr_imputed=True``; недели без ПЦР в начале
      ряда отбрасываются;
    * ``"zero"`` — как в исходном коде Influenza_bulletin: оценка случаев гриппа
      на такой неделе равна нулю.

    В обоих режимах отбрасываются последние недели без ПЦР (неполная выгрузка).
    Если для города есть ПЦР по возрастам (``age_pcr_path``, см. ``ai4epi.data.age_pcr``),
    к таблице добавляются столбцы ``age_pcr_tested_k`` / ``age_pcr_positive_k``; ``"auto"`` —
    все ``age_pcr_*.csv`` каталога ``<data_dir>/../national``, ``None`` — не добавлять.
    ``end_date`` обрезает ряд *до* обработки пропусков: интерполяция не
    использует данные ПЦР недель после ``end_date`` (состояние «как было
    известно на дату», без заглядывания вперёд).
    Возвращает таблицу и число отброшенных последних недель.
    """

    folder = Path(data_dir) / city
    cases_path, pcr_path = folder / "cases.xlsx", folder / "pcr.xlsx"
    for path in (cases_path, pcr_path):
        if not path.is_file():
            available = ", ".join(list_local_cities(data_dir)) or "нет"
            raise LocalDataError(f"Файл {path} не найден. Доступные города: {available}.")

    raw_cases = pd.read_excel(cases_path)
    raw_pcr = pd.read_excel(pcr_path)
    for frame, name in ((raw_cases, "cases.xlsx"), (raw_pcr, "pcr.xlsx")):
        if "Дата" not in frame.columns:
            raise LocalDataError(f"В {folder / name} нет столбца 'Дата'.")

    cases = pd.DataFrame({"datetime": raw_cases["Дата"].map(week_start_from_label)})
    for source, target in CASES_COLUMNS.items():
        cases[target] = pd.to_numeric(raw_cases[source], errors="coerce") if source in raw_cases else float("nan")
    meta = city_meta(city, data_dir)
    city_name = meta["name_ru"]
    cases.insert(1, "region_name", city_name)
    cases.insert(2, "district_name", city_name)

    pcr = pd.DataFrame({"datetime": raw_pcr["Дата"].map(week_start_from_label)})
    pcr["tested_total"] = pd.to_numeric(raw_pcr[PCR_TESTED_COLUMN], errors="coerce")
    for index, strain in STRAIN_NAMES.items():
        pcr[f"tested_strain_{index}"] = pd.to_numeric(raw_pcr[strain], errors="coerce")

    merged = cases.merge(pcr, on="datetime", how="left", validate="one_to_one")
    merged = merged.sort_values("datetime").reset_index(drop=True)
    if end_date is not None:
        merged = merged[merged["datetime"] <= pd.Timestamp(end_date)].reset_index(drop=True)

    dropped = 0
    while len(merged) and pd.isna(merged["tested_total"].iloc[-1]):
        merged = merged.iloc[:-1]
        dropped += 1

    no_pcr = merged["tested_total"].isna() | (merged["tested_total"] <= 0)
    tested = merged["tested_total"].where(~no_pcr)
    for index in STRAIN_NAMES:
        merged[f"rel_strain_{index}"] = merged[f"tested_strain_{index}"] / tested
    merged["pcr_imputed"] = False
    if pcr_gap_policy == "interpolate":
        rel_cols = [f"rel_strain_{index}" for index in STRAIN_NAMES]
        merged[rel_cols] = merged[rel_cols].interpolate(method="linear", limit_area="inside")
        merged["pcr_imputed"] = no_pcr & merged[rel_cols].notna().all(axis=1)
        leading = int(merged[rel_cols].notna().all(axis=1).to_numpy().argmax())
        merged = merged.iloc[leading:]
    elif pcr_gap_policy != "zero":
        raise LocalDataError(f"Неизвестная политика пропусков ПЦР: {pcr_gap_policy!r}.")
    for index in STRAIN_NAMES:
        merged[f"real_cases_strain_{index}"] = (
            merged[f"rel_strain_{index}"] * merged["sars_total_cases"]
        ).round()
    merged = merged.drop(columns=["tested_total", *[f"tested_strain_{i}" for i in STRAIN_NAMES]])
    merged = merged.reset_index(drop=True)
    if age_pcr_path == "auto":
        age_pcr_path = Path(data_dir).parent / "national"
    if age_pcr_path is not None:
        merged = attach_age_pcr(merged, load_age_pcr(city, path=age_pcr_path, district=meta["district"]))
    INFLUENZA_CASES_SCHEMA.validate_dataframe(merged, name="influenza_cases")
    return merged, dropped


def load_local_influenza(
    city: str,
    *,
    data_dir: str | Path = DEFAULT_LOCAL_DATA_DIR,
    start_date: date | datetime | str | None = None,
    end_date: date | datetime | str | None = None,
    pcr_gap_policy: PcrGapPolicy = "interpolate",
) -> LocalInfluenzaBundle:
    """Загрузить город из локальных xlsx и построить недельный ряд ``inc_per_10k``.

    ``start_date``/``end_date`` обрезают ряд по дате начала недели (включительно);
    ``end_date`` позволяет воспроизвести состояние системы на прошлую неделю.
    """

    cases, dropped = load_local_cases(city, data_dir=data_dir, pcr_gap_policy=pcr_gap_policy, end_date=end_date)
    if start_date is not None:
        cases = cases[cases["datetime"] >= pd.Timestamp(start_date)]
    cases = cases.reset_index(drop=True)
    if cases.empty:
        raise LocalDataError(f"Для города {city!r} нет данных в заданном интервале дат.")
    weekly = weekly_incidence_from_cases(cases)
    imputed = cases.set_index("datetime")["pcr_imputed"]
    weekly["pcr_imputed"] = weekly["datetime"].map(imputed).fillna(False).astype(bool)
    has_split = bool(cases[["sars_cases_age_group_4", "sars_cases_age_group_5"]].notna().any().all())
    return LocalInfluenzaBundle(
        city=city,
        city_name_ru=city_meta(city, data_dir)["name_ru"],
        cases=cases,
        weekly=weekly,
        source_dir=Path(data_dir) / city,
        dropped_tail_weeks=dropped,
        has_15_64_65_split=has_split,
        pcr_imputed_weeks=int(cases["pcr_imputed"].sum()),
        pcr_gap_policy=pcr_gap_policy,
    )


__all__ = [
    "DEFAULT_LOCAL_DATA_DIR",
    "LOCAL_CITY_NAMES_RU",
    "LocalDataError",
    "LocalInfluenzaBundle",
    "PcrGapPolicy",
    "city_meta",
    "list_local_cities",
    "load_local_cases",
    "load_local_influenza",
    "week_start_from_label",
]
