"""
Многолетняя история заболеваемости ОРВИ по городам (выгрузка «data1990_2019.xlsx») и признаки ОРВИ для ML.

Формат выгрузки: один лист, строки «город × год × неделя»; колонки ``town_name``, ``cld_year``,
``week_number``, ``wd_<g>`` — число случаев ОРВИ, ``qty_<g>`` — население, ``wda_<g>`` — отметка
эпидемической недели («Э»), где ``<g>`` — ``total``, ``0_2``, ``3_6``, ``7_14``, ``15``.

Неделя в выгрузке считается от недели, содержащей 1 января (неделя 1 начинается в понедельник этой
недели), а не по ISO: в годы, где 1 января — пятница–воскресенье, номер больше ISO-номера на 1. С этим
выравниванием ряды совпадают с ``cases.xlsx`` (СПб, Челябинск, Самара — точно; остальные города — в
пределах 1 %, поздние уточнения).

Признаки ОРВИ (``FeatureEngineeringConfig.ari_features``): прогнозируемый ряд — грипп (ОРВИ × доля
положительных ПЦР), а ОРВИ известна не позже гриппа и даёт ранний сигнал подъёма. В недельную таблицу
добавляются ``ari_per_10k`` (ОРВИ на 10 тыс.) и ``ari_norm`` — многолетняя норма log(1 + ОРВИ на 10 тыс.)
для этой ISO-недели по прошлым годам (только годы до года строки — без заглядывания вперёд). Длинная
история нужна прежде всего для нормы: с 1990 года это 15 лет вместо 3–9 по ``cases.xlsx``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ARI_GROUPS: tuple[str, ...] = ("total", "0_2", "3_6", "7_14", "15")
ARI_HISTORY_FILE = "ari_history.csv"

#: названия городов выгрузки → коды каталогов data/regions
TOWN_CODES: dict[str, str] = {
    "Москва": "moscow",
    "Санкт-Петербург": "spb",
    "Новосибирск": "novosibirsk",
    "Челябинск": "chelyabinsk",
    "Омск": "omsk",
    "Самара": "samara",
    "Воронеж": "voronezh",
    "Краснодар": "krasnodar",
}


def week_start_from_jan1_week(year: pd.Series, week: pd.Series) -> pd.Series:
    """Понедельник недели ``week`` года ``year`` при нумерации от недели, содержащей 1 января."""

    jan1 = pd.to_datetime(year.astype(int).astype(str) + "-01-01")
    return jan1 - pd.to_timedelta(jan1.dt.day_of_week, unit="D") + pd.to_timedelta((week.astype(int) - 1) * 7, unit="D")


def read_ari_history_export(path: str | Path) -> pd.DataFrame:
    """Выгрузка → длинная таблица: ``town``, ``week_start``, ``ari_<g>``, ``pop_<g>``, ``epidemic_<g>``."""

    raw = pd.read_excel(path, sheet_name=0)
    missing = {"town_name", "cld_year", "week_number", "wd_total", "qty_total"} - set(raw.columns)
    if missing:
        raise ValueError(f"{path}: нет колонок {sorted(missing)}.")
    out = pd.DataFrame(
        {
            "town": raw["town_name"].astype(str).str.strip(),
            "week_start": week_start_from_jan1_week(raw["cld_year"], raw["week_number"]),
        }
    )
    for g in ARI_GROUPS:
        if f"wd_{g}" not in raw:
            continue
        out[f"ari_{g}"] = pd.to_numeric(raw[f"wd_{g}"], errors="coerce")
        out[f"pop_{g}"] = pd.to_numeric(raw[f"qty_{g}"], errors="coerce")
        out[f"epidemic_{g}"] = (
            raw.get(f"wda_{g}", pd.Series(index=raw.index, dtype=object)).astype(str).str.strip() == "Э"
        )
    # нулевая неделя при ненулевых соседних — пропуск выгрузки, а не отсутствие заболевших
    out.loc[out["ari_total"] <= 0, [c for c in out.columns if c.startswith("ari_")]] = np.nan
    out = out.dropna(subset=["ari_total", "pop_total"])
    return out.sort_values(["town", "week_start"]).drop_duplicates(["town", "week_start"]).reset_index(drop=True)


def convert_ari_history(
    path: str | Path,
    out_dir: str | Path,
    *,
    towns: dict[str, str] | None = None,
    all_towns_file: str | Path | None = None,
) -> dict[str, Path]:
    """Разложить выгрузку по городам: ``<out_dir>/<код>/ari_history.csv``.

    Города без каталога в ``out_dir`` пропускаются; при ``all_towns_file`` вся таблица (все города)
    дополнительно сохраняется одним CSV.
    """

    table = read_ari_history_export(path)
    if all_towns_file is not None:
        Path(all_towns_file).parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(all_towns_file, index=False)
    written: dict[str, Path] = {}
    for town, code in (towns or TOWN_CODES).items():
        city_dir = Path(out_dir) / code
        part = table[table["town"] == town]
        if part.empty or not city_dir.is_dir():
            continue
        target = city_dir / ARI_HISTORY_FILE
        part.drop(columns="town").to_csv(target, index=False)
        written[code] = target
    return written


def load_ari_history(city_dir: str | Path) -> pd.DataFrame | None:
    path = Path(city_dir) / ARI_HISTORY_FILE
    if not path.exists():
        return None
    return pd.read_csv(path, parse_dates=["week_start"])


def ari_series(cases: pd.DataFrame, history: pd.DataFrame | None = None) -> pd.Series:
    """ОРВИ на 10 тыс. по неделям: ``cases.xlsx`` (``sars_total_cases``), до их начала — история."""

    own = cases.dropna(subset=["sars_total_cases", "total_population"])
    own = own[own["total_population"] > 0]
    series = pd.Series(
        (own["sars_total_cases"] / own["total_population"] * 10_000).to_numpy(float),
        index=pd.DatetimeIndex(own["datetime"]),
    )
    if history is not None and not history.empty:
        hist = history.dropna(subset=["ari_total", "pop_total"])
        hist = pd.Series(
            (hist["ari_total"] / hist["pop_total"] * 10_000).to_numpy(float), index=pd.DatetimeIndex(hist["week_start"])
        )
        series = pd.concat([hist[hist.index < series.index.min()], series]) if len(series) else hist
    return series[~series.index.duplicated(keep="last")].sort_index()


def causal_weekly_norm(
    series: pd.Series, *, max_years: int = 15, min_years: int = 3, smooth_weeks: int = 1
) -> pd.Series:
    """Норма log(1 + x) для ISO-недели строки по прошлым годам (``min_years``..``max_years``).

    Для недели w года Y берутся значения недель w ± ``smooth_weeks`` годов Y − max_years .. Y − 1;
    при меньше ``min_years`` годах — NaN.
    """

    z = np.log1p(series.clip(lower=0))
    iso = z.index.isocalendar()
    frame = pd.DataFrame({"z": z.to_numpy(), "year": iso.year.to_numpy(int), "week": iso.week.to_numpy(int)})
    frame.loc[frame["week"] == 53, "week"] = 52
    by_year_week = frame.groupby(["year", "week"])["z"].mean().unstack()  # годы × недели 1..52
    by_year_week = by_year_week.reindex(columns=range(1, 53))
    out = np.full(len(frame), np.nan)
    years = by_year_week.index.to_numpy()
    for i, (y, w) in enumerate(zip(frame["year"], frame["week"], strict=True)):
        past = by_year_week[(years < y) & (years >= y - max_years)]
        if past.empty:
            continue
        cols = [((w - 1 + d) % 52) + 1 for d in range(-smooth_weeks, smooth_weeks + 1)]
        per_year = past[cols].mean(axis=1).dropna()
        if len(per_year) >= min_years:
            out[i] = per_year.mean()
    return pd.Series(out, index=series.index)


def add_ari_columns(weekly: pd.DataFrame, cases: pd.DataFrame, history: pd.DataFrame | None = None) -> pd.DataFrame:
    """Добавить в недельную таблицу ``ari_per_10k`` и ``ari_norm`` (см. описание модуля)."""

    series = ari_series(cases, history)
    norm = causal_weekly_norm(series)
    out = weekly.copy()
    dates = pd.DatetimeIndex(out["datetime"])
    out["ari_per_10k"] = series.reindex(dates).to_numpy(float)
    out["ari_norm"] = norm.reindex(dates).to_numpy(float)
    out["ari_per_10k"] = out["ari_per_10k"].interpolate(limit=3, limit_area="inside")
    return out


__all__ = [
    "ARI_GROUPS",
    "ARI_HISTORY_FILE",
    "TOWN_CODES",
    "add_ari_columns",
    "ari_series",
    "causal_weekly_norm",
    "convert_ari_history",
    "load_ari_history",
    "read_ari_history_export",
    "week_start_from_jan1_week",
]
