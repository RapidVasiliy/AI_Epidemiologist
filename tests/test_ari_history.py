from __future__ import annotations

import numpy as np
import pandas as pd

from ai4epi.analysis.preprocessing import FeatureEngineeringConfig, build_supervised
from ai4epi.data.ari_history import (
    add_ari_columns,
    causal_weekly_norm,
    convert_ari_history,
    week_start_from_jan1_week,
)


def test_week_numbering_from_week_containing_jan1():
    # 2016-01-01 — пятница: неделя 1 выгрузки = ISO-неделя 53 2015 года, неделя 2 = ISO-неделя 1
    starts = week_start_from_jan1_week(pd.Series([2016, 2016, 2018]), pd.Series([1, 2, 1]))
    assert starts.dt.date.astype(str).tolist() == ["2015-12-28", "2016-01-04", "2018-01-01"]


def _export(path, town="Москва", years=(2006, 2007, 2008, 2009)):
    rows = []
    for y in years:
        for w in range(1, 53):
            n = 1000 + 500 * np.cos(2 * np.pi * (w - 5) / 52)
            rows.append(
                {
                    "town_name": town,
                    "cld_year": y,
                    "week_number": w,
                    "wd_total": n,
                    "wda_total": "Э" if w == 5 else None,
                    "qty_total": 1_000_000,
                }
            )
    pd.DataFrame(rows).to_excel(path, index=False)
    return path


def test_convert_splits_by_city(tmp_path):
    (tmp_path / "regions" / "moscow").mkdir(parents=True)
    written = convert_ari_history(_export(tmp_path / "x.xlsx"), tmp_path / "regions")
    t = pd.read_csv(written["moscow"], parse_dates=["week_start"])
    assert len(t) == 4 * 52 and int(t["epidemic_total"].sum()) == 4
    assert (t["week_start"].dt.day_of_week == 0).all()


def test_causal_norm_uses_only_past_years():
    idx = pd.date_range("2010-01-04", periods=52 * 6, freq="7D")
    year = idx.isocalendar().year.to_numpy()
    s = pd.Series(10.0, index=idx)
    s[year == 2015] = 1000.0  # аномальный последний год не должен влиять на собственную норму
    norm = causal_weekly_norm(s, min_years=3)
    assert norm[year < 2013].isna().all()
    assert np.allclose(norm[year == 2015].dropna(), np.log1p(10.0))


def test_ari_features_only_with_flag():
    idx = pd.date_range("2012-01-02", periods=52 * 4, freq="7D")
    cases = pd.DataFrame(
        {"datetime": idx, "sars_total_cases": 1000 + 100 * np.sin(np.arange(len(idx))), "total_population": 1e6}
    )
    weekly = pd.DataFrame({"datetime": idx, "inc_per_10k": np.abs(np.sin(np.arange(len(idx)) / 8)) * 10})
    iso = idx.isocalendar()
    weekly["iso_year"], weekly["iso_week"] = iso.year.to_numpy(int), iso.week.to_numpy(int)
    with_ari = add_ari_columns(weekly, cases)
    plain = build_supervised(with_ari, config=FeatureEngineeringConfig(temp_cols=()))
    assert not any("ari" in c for c in plain.feature_cols)
    feat = build_supervised(with_ari, config=FeatureEngineeringConfig(temp_cols=(), ari_features=True))
    assert {"ari_lag_0", "ari_growth_1", "ari_anom_lag_0"} <= set(feat.feature_cols)
    assert "ari_per_10k" not in feat.feature_cols and feat.valid_row_count == plain.valid_row_count
