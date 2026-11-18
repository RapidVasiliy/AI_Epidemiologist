from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("statsmodels")

from ai4epi.analysis.baselines import forecast_ets, forecast_harmonic_arima, statistical_baselines


def _seasonal(n_weeks: int = 400, noise: float = 0.05) -> pd.Series:
    dates = pd.date_range("2012-01-02", periods=n_weeks, freq="7D")
    t = np.arange(n_weeks)
    level = np.exp(1.5 + 1.5 * np.cos(2 * np.pi * (t - 5) / (365.25 / 7)))
    rng = np.random.default_rng(0)
    return pd.Series(level * np.exp(rng.normal(0, noise, n_weeks)), index=dates)


def test_baselines_follow_seasonal_cycle():
    s = _seasonal()
    history, future = s.iloc[:-4], s.iloc[-4:].to_numpy()
    out = statistical_baselines(history, 4)
    assert set(out) == {"harmonic_arima", "ets"}
    for name, pred in out.items():
        assert pred.shape == (4,) and np.all(pred >= 0), name
        assert np.mean(np.abs(pred - future)) / future.mean() < 0.25, name


def test_baselines_handle_short_and_gappy_series():
    s = _seasonal(60)
    assert np.isnan(forecast_ets(s, 4)).all()  # меньше двух сезонов — ETS с годовой сезонностью не строится
    gappy = _seasonal().drop(_seasonal().index[100:103])  # пропущенные недели заполняются интерполяцией
    assert np.isfinite(forecast_harmonic_arima(gappy, 4)).all()
