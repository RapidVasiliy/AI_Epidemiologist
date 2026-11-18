"""
Импорт выгрузок погоды (rp5.ru или произвольная таблица) в недельную таблицу для ``--weather-weekly``.

Поддерживаемые входы:

* **архив rp5.ru** («Расписание погоды»: «Архив погоды» → «Скачать», формат xls/xlsx или csv): строки
  с ``#`` в начале — комментарии, первая колонка «Местное время в <город>» (``дд.мм.гггг чч:мм``),
  колонки ``T`` — температура, °C, ``U`` — относительная влажность, %; срочные наблюдения раз в 3 часа;
* **месячные архивы meteo9.ru** (``archive_<id>_<месяц>_<год>.csv``): без заголовка, ``;``-разделитель,
  колонки «дата дд/мм/гггг; день недели; время ч:мм; температура; облачность; давление; влажность; ...»,
  почасовые наблюдения; файлы за разные месяцы передаются вместе;
* **произвольная таблица** CSV / Excel: столбец даты или времени (``date``, ``datetime``, ``time``,
  ``дата``, ``время``, ...), температура (``temp``, ``t``, ``tmean``, ``температура`` ...), по желанию
  влажность (``rh``, ``u``, ``humidity``, ``влажность``) и суточные минимум / максимум (``tmin`` / ``tmax``).
  Подходят и срочные, и суточные данные.

Выход — недельная таблица контракта ``ai4epi`` (README, «Weekly weather table»): ``week_start``
(понедельник), ``temp_mean``, ``temp_max``, ``temp_min``, ``rh_mean``, ``rh_max``, ``rh_min``, ``n_hours``
(число наблюдений за неделю). Неделя — ISO-неделя, как в рядах заболеваемости.
"""

from __future__ import annotations

import gzip
import io
import re
from pathlib import Path

import numpy as np
import pandas as pd

WEEKLY_WEATHER_COLUMNS = ["week_start", "temp_mean", "temp_max", "temp_min", "rh_mean", "rh_max", "rh_min", "n_hours"]

_TIME_NAMES = ("местное время", "local time", "datetime", "date", "time", "дата", "время", "date_time")
_TEMP_NAMES = ("t", "temp", "temperature", "tmean", "t_mean", "temp_mean", "температура", "средняя температура")
_TMIN_NAMES = ("tmin", "t_min", "temp_min", "tn", "минимальная температура", "мин. температура")
_TMAX_NAMES = ("tmax", "t_max", "temp_max", "tx", "максимальная температура", "макс. температура")
_RH_NAMES = ("u", "rh", "humidity", "влажность", "относительная влажность", "rh_mean")


class WeatherImportError(ValueError):
    """Не удалось распознать выгрузку погоды."""


def _norm(name: object) -> str:
    return re.sub(r"\s+", " ", str(name).replace("\xa0", " ")).strip().strip('"').lower()


def _find(columns: list[str], names: tuple[str, ...], *, prefix: bool = False) -> str | None:
    normed = {_norm(c): c for c in columns}
    for name in names:
        if name in normed:
            return normed[name]
    if prefix:
        for key, original in normed.items():
            if any(key.startswith(name) for name in names):
                return original
    return None


def _decode(data: bytes) -> str:
    """rp5 отдаёт UTF-8, ANSI (cp1251) и «Unicode» (UTF-16 с BOM)."""

    if data[:2] in (b"\xff\xfe", b"\xfe\xff"):
        return data.decode("utf-16")
    for encoding in ("utf-8-sig", "cp1251"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise WeatherImportError("неизвестная кодировка.")  # pragma: no cover


def _peek_lines(path: Path, n: int = 5) -> list[str]:
    """Первые строки текстового файла (для распознавания формата); для Excel — пусто."""

    data = path.read_bytes()
    if path.suffix.lower() == ".gz":
        data = gzip.decompress(data)
    if data[:4] == b"\xd0\xcf\x11\xe0" or data[:2] == b"PK":  # xls / xlsx
        return []
    try:
        return [ln for ln in _decode(data[:20000]).splitlines() if ln.strip()][:n]
    except (WeatherImportError, UnicodeDecodeError):
        return []


def _read_table(path: Path) -> pd.DataFrame:
    """Таблица с заголовком; строки-комментарии rp5 (``#``) и строки до заголовка пропускаются.

    Файлы ``.gz`` (как их отдаёт rp5) читаются без распаковки.
    """

    data = path.read_bytes()
    suffixes = [s.lower() for s in path.suffixes]
    if suffixes and suffixes[-1] == ".gz":
        data = gzip.decompress(data)
        suffixes = suffixes[:-1]
    kind = suffixes[-1] if suffixes else ""
    if kind in {".xls", ".xlsx"}:
        raw = pd.read_excel(io.BytesIO(data), header=None, dtype=object)
        header_row = next(
            (i for i in range(min(len(raw), 40)) if _find([str(v) for v in raw.iloc[i]], _TIME_NAMES, prefix=True)),
            None,
        )
        if header_row is None:
            raise WeatherImportError(f"{path}: не найдена строка заголовка со столбцом даты/времени.")
        table = raw.iloc[header_row + 1 :].copy()
        table.columns = [str(v) for v in raw.iloc[header_row]]
        return table.reset_index(drop=True)
    decoded = _decode(data)
    lines = [ln for ln in decoded.splitlines() if ln.strip() and not ln.lstrip().startswith("#")]
    if not lines:
        raise WeatherImportError(f"{path}: файл пуст.")
    sep = ";" if lines[0].count(";") >= lines[0].count(",") else ","
    # в поле облачности METAR-архивов rp5 изредка попадает HTML (<span style="...;">) — его «;» и кавычки
    # ломают разбор; теги вычищаются, а строки с лишними полями обрезаются до длины заголовка
    lines = [_HTML_TAG.sub("", ln) if "<" in ln else ln for ln in lines]
    return pd.read_csv(
        io.StringIO("\n".join(lines)),
        sep=sep,
        dtype=str,
        index_col=False,
        engine="python",
        on_bad_lines=lambda bad: bad[: lines[0].count(sep) + 1],
    )


_HTML_TAG = re.compile(r"<[^<>]*>")


def _to_number(series: pd.Series) -> pd.Series:
    cleaned = series.astype(str).str.replace("\xa0", "").str.replace(",", ".").str.strip()
    return pd.to_numeric(cleaned, errors="coerce")


_METEO9_ROW = re.compile(r'^"?\d{2}/\d{2}/\d{4}"?;"?[^;]*"?;"?\d{1,2}:\d{2}"?;')


def _read_meteo9(lines: list[str]) -> pd.DataFrame:
    rows = [ln.split(";") for ln in lines if _METEO9_ROW.match(ln)]
    frame = pd.DataFrame([[c.strip().strip('"') for c in r] for r in rows])
    out = pd.DataFrame({"time": pd.to_datetime(frame[0] + " " + frame[2], format="%d/%m/%Y %H:%M", errors="coerce")})
    out["temp"] = _to_number(frame[3])
    out["rh"] = _to_number(frame[6])
    return out


def read_weather_export(path: str | Path) -> pd.DataFrame:
    """Наблюдения с колонками ``time``, ``temp`` и, если есть, ``rh``, ``tmin``, ``tmax``."""

    path = Path(path)
    head = _peek_lines(path)
    if head and all(_METEO9_ROW.match(ln) for ln in head[:3]):
        data = path.read_bytes()
        if path.suffix.lower() == ".gz":
            data = gzip.decompress(data)
        out = _read_meteo9(_decode(data).splitlines())
        out = out.dropna(subset=["time", "temp"]).sort_values("time").drop_duplicates("time").reset_index(drop=True)
        if out.empty:
            raise WeatherImportError(f"{path}: не удалось разобрать архив meteo9.")
        return out
    table = _read_table(path)
    columns = list(table.columns)
    time_col = _find(columns, _TIME_NAMES, prefix=True)
    temp_col = _find(columns, _TEMP_NAMES)
    if time_col is None or temp_col is None:
        raise WeatherImportError(f"{path}: нужны столбцы даты/времени и температуры; найдены: {columns[:12]}.")
    out = pd.DataFrame(
        {"time": pd.to_datetime(table[time_col].astype(str).str.strip(), dayfirst=True, errors="coerce")}
    )
    out["temp"] = _to_number(table[temp_col])
    for key, names in (("rh", _RH_NAMES), ("tmin", _TMIN_NAMES), ("tmax", _TMAX_NAMES)):
        col = _find(columns, names)
        if col is not None and col != temp_col:
            out[key] = _to_number(table[col])
    out = out.dropna(subset=["time", "temp"]).sort_values("time").drop_duplicates("time").reset_index(drop=True)
    if out.empty:
        raise WeatherImportError(f"{path}: после разбора не осталось наблюдений с датой и температурой.")
    return out


def weekly_weather(observations: pd.DataFrame, *, min_observations: int = 4) -> pd.DataFrame:
    """Недельная таблица (неделя с понедельника) из наблюдений ``read_weather_export``."""

    obs = observations.copy()
    obs["week_start"] = obs["time"].dt.to_period("W-SUN").dt.start_time
    groups = obs.groupby("week_start")
    out = pd.DataFrame(
        {
            "temp_mean": groups["temp"].mean(),
            "temp_max": groups["tmax"].max() if "tmax" in obs else groups["temp"].max(),
            "temp_min": groups["tmin"].min() if "tmin" in obs else groups["temp"].min(),
            "rh_mean": groups["rh"].mean() if "rh" in obs else np.nan,
            "rh_max": groups["rh"].max() if "rh" in obs else np.nan,
            "rh_min": groups["rh"].min() if "rh" in obs else np.nan,
            "n_hours": groups["temp"].count(),
        }
    ).reset_index()
    out = out[out["n_hours"] >= min_observations].reset_index(drop=True)
    return out[WEEKLY_WEATHER_COLUMNS]


def fill_short_gaps(weekly: pd.DataFrame, *, max_gap_weeks: int = 3) -> tuple[pd.DataFrame, list[tuple[str, str]]]:
    """Регулярная недельная сетка; разрывы до ``max_gap_weeks`` недель заполняются линейной интерполяцией.

    Анализ не допускает пропусков в погодных признаках, поэтому колонки влажности без единого значения
    удаляются (модель тогда строится только на температуре). Возвращает таблицу и список оставшихся
    длинных разрывов (первая и последняя пропущенная неделя).
    """

    out = weekly.set_index("week_start").sort_index()
    out = out.reindex(pd.date_range(out.index.min(), out.index.max(), freq="7D"))
    value_cols = [c for c in out.columns if c != "n_hours" and out[c].notna().any()]
    out = out[[*value_cols, "n_hours"]]
    missing = out[value_cols[0]].isna()
    runs = (missing != missing.shift()).cumsum()
    long_gaps = []
    for _, block in out[missing].groupby(runs[missing]):
        if len(block) > max_gap_weeks:
            long_gaps.append((block.index[0].date().isoformat(), block.index[-1].date().isoformat()))
    out[value_cols] = out[value_cols].interpolate(limit=max_gap_weeks, limit_area="inside")
    out["n_hours"] = out["n_hours"].fillna(0).astype(int)
    return out.rename_axis("week_start").reset_index(), long_gaps


def convert_weather_exports(paths: list[str | Path], *, min_observations: int = 4) -> pd.DataFrame:
    """Несколько файлов (например, годовые выгрузки rp5) → одна недельная таблица."""

    observations = pd.concat([read_weather_export(p) for p in paths], ignore_index=True)
    observations = observations.sort_values("time").drop_duplicates("time").reset_index(drop=True)
    return weekly_weather(observations, min_observations=min_observations)


__all__ = [
    "WEEKLY_WEATHER_COLUMNS",
    "WeatherImportError",
    "convert_weather_exports",
    "fill_short_gaps",
    "read_weather_export",
    "weekly_weather",
]
