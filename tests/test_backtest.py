from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ai4epi.analysis.backtest import BacktestConfig, _add_conformal_intervals, run_backtest
from ai4epi.analysis.forecasting import HistGradientBoostingPointConfig
from ai4epi.data.local_excel import load_local_influenza


def test_conformal_intervals_use_only_known_errors():
    origins = pd.date_range("2020-01-06", periods=30, freq="7D")
    pred = pd.DataFrame(
        {
            "origin": origins,
            "horizon": 2,
            "target_date": origins + pd.Timedelta(weeks=2),
            "ml_hgb": 10.0,
            "y_origin": 10.0,
            "y_true": [10.0] * 15 + [1000.0] * 15,  # огромные ошибки только у поздних целей
        }
    )
    cfg = BacktestConfig(start="2020-01-06", end="2020-08-31", min_conformal_errors=5, n_jobs=1)
    out = _add_conformal_intervals(pred.copy(), cfg)
    # для недели 16 (индекс 15) цели с ошибкой 1000 ещё не наблюдались -> интервал нулевой ширины
    row = out.iloc[15]
    assert row["ml_hi"] - row["ml_lo"] == pytest.approx(0.0)
    # к последней неделе часть больших ошибок уже известна -> интервал расширяется
    assert out.iloc[-1]["ml_hi"] > 100
    assert out.iloc[-1]["ml_hi_level"] > 100


def _truncate(full: pd.DataFrame):
    def data_at(origin):
        return full[full["datetime"] <= origin].reset_index(drop=True)

    return data_at


def test_backtest_runs_without_lookahead_and_reports(tmp_path, synthetic_weekly):
    rng = np.random.default_rng(0)
    season = synthetic_weekly(lead_weeks=10, wave_weeks=42)["inc_per_10k"].to_numpy()
    dates = pd.date_range("2012-09-03", periods=len(season) * 7, freq="7D")
    full = pd.DataFrame({"datetime": dates, "inc_per_10k": np.tile(season, 7) * rng.uniform(0.9, 1.1, len(dates))})
    iso = full["datetime"].dt.isocalendar()
    full["iso_year"], full["iso_week"] = iso.year.astype(int), iso.week.astype(int)
    full["total_population"] = 5e6
    full["total_cases_formula"] = full["inc_per_10k"] * 500
    cfg = BacktestConfig(
        start="2018-10-01",
        end="2018-12-31",
        model_variants={"hgb": HistGradientBoostingPointConfig(learning_rate=0.05, max_iter=100)},
        run_mechanistic=False,
        interval_warmup_weeks=30,
        min_conformal_errors=10,
        n_jobs=1,
    )
    res = run_backtest(_truncate(full), full, config=cfg, output_dir=tmp_path, label="synthetic")
    pred = res.predictions
    assert pred["origin"].min() == pd.Timestamp("2018-10-01")
    assert (pred["target_date"] > pred["origin"]).all()
    assert pred["ml_lo"].notna().any()
    m = res.metrics
    assert {"ml_hgb", "persistence", "climatology"} <= set(m["overall"])
    assert (tmp_path / "report.md").read_text(encoding="utf-8").startswith("# Бэктест")


def test_local_state_at_date_does_not_use_future_pcr(data_dir):
    # 2022-12-12 в Челябинске — неделя без ПЦР. На эту дату она последняя и
    # должна быть отброшена, а не интерполирована по данным следующей недели.
    state = load_local_influenza("chelyabinsk", data_dir=data_dir, end_date="2022-12-12")
    assert state.weekly["datetime"].max() == pd.Timestamp("2022-12-05")
    assert state.dropped_tail_weeks == 1
