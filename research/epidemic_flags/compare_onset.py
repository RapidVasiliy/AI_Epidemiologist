"""Сверка определения волны ai4epi с официальными эпидемическими неделями («Э») выгрузки 1990–2019.

Для каждого города и сезона (с 35-й ISO-недели, как MechanisticWaveConfig) по ряду гриппа inc_per_10k:
* наше начало волны — detect_wave (порог, 3 нед. роста подряд; при неудаче — по скользящему среднему,
  как в run_mechanistic_wave_analysis) и дата, когда оно было бы обнаружено в реальном времени;
* официальный эпидемический период — первая и последняя неделя с отметкой «Э» (ОРВИ, все возрасты);
* пик гриппа и попадание его в период «Э»;
* порог гриппа на 10 тыс., лучше всего воспроизводящий недели «Э» (F1 по неделям всех сезонов).

    python research/epidemic_flags/compare_onset.py --out research/epidemic_flags/out
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from ai4epi.analysis.mechanistic_wave import MechanisticWaveConfig, _holiday_interpolated, _season_start, detect_wave
from ai4epi.data.ari_history import load_ari_history
from ai4epi.data.local_excel import load_local_influenza

CITIES = ("moscow", "spb", "novosibirsk", "chelyabinsk", "omsk", "samara", "voronezh", "krasnodar")


def detect_wave_level(
    values: np.ndarray, dates: pd.Series, cfg: MechanisticWaveConfig, level: float
) -> tuple[bool, pd.Timestamp | None]:
    """Вариант правила: рост ``wave_min_growth_weeks`` недель подряд, на последней неделе которого показатель
    не ниже ``level`` на 10 тыс. (рост с 0.1 до 0.3 подъёмом не считается)."""

    run = 0
    for i in range(1, len(values)):
        run = run + 1 if values[i] > values[i - 1] else 0
        if run >= cfg.wave_min_growth_weeks and values[i] >= level:
            j = i - run
            return True, pd.Timestamp(dates.iloc[j])
    return False, None


LEVEL: float | None = None  # None — текущее правило ai4epi


def onset(values: np.ndarray, dates: pd.Series, cfg: MechanisticWaveConfig) -> tuple[pd.Timestamp | None, bool]:
    if LEVEL is not None:
        found, start = detect_wave_level(values, dates, cfg, LEVEL)
        if found:
            return start, False
        smoothed = pd.Series(values).rolling(cfg.wave_smoothing_fallback_weeks, min_periods=1).mean().to_numpy()
        found, start = detect_wave_level(smoothed, dates, cfg, LEVEL)
        return (start, True) if found else (None, False)
    found, start = detect_wave(values, dates, cfg)
    if found:
        return start, False
    smoothed = pd.Series(values).rolling(cfg.wave_smoothing_fallback_weeks, min_periods=1).mean().to_numpy()
    found, start = detect_wave(smoothed, dates, cfg)
    return (start, True) if found else (None, False)


def realtime_detection(values: np.ndarray, dates: pd.Series, cfg: MechanisticWaveConfig) -> pd.Timestamp | None:
    """Первая неделя t, на которой detect_wave по данным до t включительно находит подъём."""

    for t in range(cfg.wave_min_growth_weeks, len(values)):
        if onset(values[: t + 1], dates.iloc[: t + 1], cfg)[0] is not None:
            return pd.Timestamp(dates.iloc[t])
    return None


def season_rows(city: str, cfg: MechanisticWaveConfig) -> tuple[list[dict], pd.DataFrame]:
    bundle = load_local_influenza(city)
    weekly = bundle.weekly[["datetime", "inc_per_10k"]].copy()
    flags = load_ari_history(bundle.source_dir)
    flags = flags[["week_start", "epidemic_total"]].rename(columns={"week_start": "datetime"})
    flag_end = flags["datetime"].max()
    weekly = weekly.merge(flags, on="datetime", how="left")
    weekly = weekly[weekly["datetime"] <= flag_end]
    rows, weeks = [], []
    for year in range(2010, 2019):
        start = pd.Timestamp(_season_start(year, cfg.season_start_week))
        end = pd.Timestamp(_season_start(year + 1, cfg.season_start_week))
        s = weekly[(weekly["datetime"] >= start) & (weekly["datetime"] < end)].reset_index(drop=True)
        if len(s) < 40 or s["epidemic_total"].isna().mean() > 0.2:
            continue
        s["epidemic_total"] = s["epidemic_total"].fillna(False).astype(bool)
        values = _holiday_interpolated(s["inc_per_10k"].to_numpy(float), s["datetime"], cfg)
        ons, smoothed = onset(values, s["datetime"], cfg)
        det = realtime_detection(values, s["datetime"], cfg)
        peak_i = int(np.nanargmax(values))
        epi = s.loc[s["epidemic_total"], "datetime"]
        e0, e1 = (epi.min(), epi.max()) if len(epi) else (None, None)
        wk = lambda a, b: None if a is None or b is None else int((a - b).days // 7)  # noqa: E731
        rows.append(
            {
                "city": city,
                "season": f"{year}/{(year + 1) % 100:02d}",
                "peak_per_10k": round(float(values[peak_i]), 1),
                "peak_date": s["datetime"][peak_i].date(),
                "onset": ons.date() if ons is not None else None,
                "onset_smoothed": smoothed,
                "detected": det.date() if det is not None else None,
                "epi_first": e0.date() if e0 is not None else None,
                "epi_last": e1.date() if e1 is not None else None,
                "epi_weeks": len(epi),
                "onset_minus_epi_first_wk": wk(ons, e0),
                "detected_minus_epi_first_wk": wk(det, e0),
                "peak_in_epi": bool(e0 is not None and e0 <= s["datetime"][peak_i] <= e1),
            }
        )
        weeks.append(s.assign(city=city, season=year, value=values))
    return rows, pd.concat(weeks) if weeks else pd.DataFrame()


def best_threshold(weeks: pd.DataFrame) -> pd.DataFrame:
    out = []
    for thr in np.arange(2, 60.1, 1.0):
        pred = weeks["value"] >= thr
        tp = int((pred & weeks["epidemic_total"]).sum())
        fp = int((pred & ~weeks["epidemic_total"]).sum())
        fn = int((~pred & weeks["epidemic_total"]).sum())
        f1 = 2 * tp / (2 * tp + fp + fn) if tp else 0.0
        out.append(
            {
                "threshold": thr,
                "f1": f1,
                "precision": tp / (tp + fp) if tp + fp else np.nan,
                "recall": tp / (tp + fn) if tp + fn else np.nan,
            }
        )
    return pd.DataFrame(out)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("research/epidemic_flags/out"))
    parser.add_argument("--level", type=float, default=None, help="Вариант правила: уровень на конце роста, на 10 тыс.")
    args = parser.parse_args()
    global LEVEL
    LEVEL = args.level
    args.out.mkdir(parents=True, exist_ok=True)
    cfg = MechanisticWaveConfig()
    rows, weeks = [], []
    for city in CITIES:
        r, w = season_rows(city, cfg)
        rows += r
        weeks.append(w)
    table = pd.DataFrame(rows)
    table.to_csv(args.out / "seasons.csv", index=False)
    weeks = pd.concat(weeks)
    thr = best_threshold(weeks)
    thr.to_csv(args.out / "thresholds.csv", index=False)
    per_city = []
    for city, w in weeks.groupby("city"):
        t = best_threshold(w).sort_values("f1").iloc[-1]
        per_city.append({"city": city, **t.to_dict()})
    pd.DataFrame(per_city).to_csv(args.out / "thresholds_by_city.csv", index=False)

    pd.set_option("display.width", 250)
    print(table.to_string(index=False))
    epi = table[table["epi_weeks"] > 0]
    print(
        f"\nсезонов: {len(table)}, с эпидемией по «Э»: {len(epi)}, волна найдена ai4epi: {table['onset'].notna().sum()}"
    )
    print(
        "начало волны − первая неделя «Э», нед.:",
        epi["onset_minus_epi_first_wk"].describe()[["mean", "50%", "min", "max"]].round(1).to_dict(),
    )
    print(
        "обнаружение − первая неделя «Э», нед.:",
        epi["detected_minus_epi_first_wk"].describe()[["mean", "50%", "min", "max"]].round(1).to_dict(),
    )
    print("пик гриппа внутри периода «Э»:", f"{epi['peak_in_epi'].mean():.2f}")
    no_epi = table[table["epi_weeks"] == 0]
    print(
        "сезоны без «Э»: пик гриппа, медиана", no_epi["peak_per_10k"].median(), "; с «Э»:", epi["peak_per_10k"].median()
    )
    print("\nлучший единый порог:", thr.sort_values("f1").iloc[-1].round(3).to_dict())
    print(pd.DataFrame(per_city).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
