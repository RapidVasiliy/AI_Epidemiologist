from __future__ import annotations

import numpy as np
import pandas as pd

from ai4epi.data.weather_import import WEEKLY_WEATHER_COLUMNS, convert_weather_exports, read_weather_export


def _rp5_csv(path, start="2019-01-07", days=14):
    times = pd.date_range(start, periods=days * 8, freq="3h")[::-1]  # rp5 — от новых к старым
    lines = ["# Метеостанция Санкт-Петербург (WMO_ID=26063)", "# Все данные в местном времени", "#",
             '"Местное время в Санкт-Петербурге";"T";"Po";"U";"DD"']
    for i, t in enumerate(times):
        lines.append(f'"{t:%d.%m.%Y %H:%M}";"{-5 + (i % 8) * 0.5:.1f}";"760.1";"{80 + i % 5}";"Ветер, дующий с юга"')
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_rp5_export_to_weekly(tmp_path):
    obs = read_weather_export(_rp5_csv(tmp_path / "rp5.csv"))
    assert {"time", "temp", "rh"} <= set(obs.columns) and obs["time"].is_monotonic_increasing
    weekly = convert_weather_exports([tmp_path / "rp5.csv"])
    assert list(weekly.columns) == WEEKLY_WEATHER_COLUMNS
    assert len(weekly) == 2 and (weekly["week_start"].dt.day_of_week == 0).all()
    assert weekly["n_hours"].tolist() == [56, 56]
    assert (weekly["temp_min"] <= weekly["temp_mean"]).all() and (weekly["temp_mean"] <= weekly["temp_max"]).all()


def test_generic_daily_table(tmp_path):
    dates = pd.date_range("2020-01-06", periods=21, freq="D")
    table = pd.DataFrame({"Дата": dates.strftime("%d.%m.%Y"), "Tmean": np.linspace(-10, 0, 21),
                          "Tmin": np.linspace(-15, -5, 21), "Tmax": np.linspace(-5, 5, 21)})
    table.to_excel(tmp_path / "daily.xlsx", index=False)
    weekly = convert_weather_exports([tmp_path / "daily.xlsx"])
    assert len(weekly) == 3 and weekly["rh_mean"].isna().all()
    assert weekly["temp_min"].iloc[0] == -15 and weekly["temp_max"].iloc[-1] == 5


def test_fill_short_gaps_and_drop_empty_humidity(tmp_path):
    from ai4epi.data.weather_import import fill_short_gaps

    weeks = pd.date_range("2020-01-06", periods=12, freq="7D")
    weekly = pd.DataFrame({"week_start": weeks, "temp_mean": np.arange(12.0), "temp_max": np.arange(12.0) + 5,
                           "temp_min": np.arange(12.0) - 5, "rh_mean": np.nan, "rh_max": np.nan, "rh_min": np.nan,
                           "n_hours": 56})
    weekly = weekly.drop(index=[3, 4, 7, 8, 9, 10]).reset_index(drop=True)  # разрывы 2 нед. и 4 нед.
    out, long_gaps = fill_short_gaps(weekly, max_gap_weeks=3)
    assert "rh_mean" not in out.columns
    assert out.loc[out.week_start == weeks[3], "temp_mean"].item() == 3.0  # короткий разрыв заполнен
    assert long_gaps == [(weeks[7].date().isoformat(), weeks[10].date().isoformat())]
