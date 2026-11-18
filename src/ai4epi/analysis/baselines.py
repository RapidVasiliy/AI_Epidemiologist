"""
Статистические базовые прогнозы для сравнения с ML-моделью.

* ``harmonic_arima`` — ARIMA(p, d, q) на log(1 + y) с регрессорами — гармониками годового цикла
  (sin/cos 2πk·t/52.18, k = 1..K), то есть «динамическая гармоническая регрессия». Классическая
  сезонная SARIMA с периодом 52 недели на таких рядах медленна и часто не сходится; гармоники
  описывают тот же годовой цикл устойчиво.
* ``ets`` — экспоненциальное сглаживание (Холт–Уинтерс) на log(1 + y): аддитивный затухающий тренд и
  аддитивная годовая сезонность с периодом 52 недели.

Обе модели обучаются только на данных до даты прогноза (последние ``train_years`` лет), прогноз
возвращается в исходной шкале (обратное преобразование, отрицательные значения обнуляются). Если
модель не сходится, возвращается NaN — бэктест и блок качества данных это учитывают.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

WEEKS_PER_YEAR = 365.25 / 7
BASELINE_NAMES: tuple[str, ...] = ("harmonic_arima", "ets")
BASELINE_LABELS_RU: dict[str, str] = {
    "harmonic_arima": "ARIMA с гармониками годового цикла",
    "ets": "экспоненциальное сглаживание (ETS) с годовой сезонностью",
}


@dataclass(frozen=True)
class BaselineConfig:
    arima_order: tuple[int, int, int] = (2, 0, 1)
    harmonics: int = 3
    train_years: int = 6
    ets_seasonal_periods: int = 52


def _fourier(t: np.ndarray, k: int) -> np.ndarray:
    cols = []
    for j in range(1, k + 1):
        cols += [np.sin(2 * np.pi * j * t / WEEKS_PER_YEAR), np.cos(2 * np.pi * j * t / WEEKS_PER_YEAR)]
    return np.column_stack(cols)


def _week_index(dates: pd.DatetimeIndex) -> np.ndarray:
    """Номер недели от фиксированной даты: гармоники согласованы между запусками."""

    return ((dates - pd.Timestamp("2000-01-03")).days / 7.0).to_numpy(float)


def _prepare(series: pd.Series, cfg: BaselineConfig) -> pd.Series:
    s = series.astype(float).sort_index()
    s = s[s.index >= s.index[-1] - pd.Timedelta(weeks=int(cfg.train_years * 52.18))]
    # регулярная недельная сетка: пропуски заполняются интерполяцией (статистическим моделям нужен ряд без дыр)
    grid = pd.date_range(s.index[0], s.index[-1], freq="7D")
    return s.reindex(grid).interpolate(limit_direction="both")


def forecast_harmonic_arima(series: pd.Series, horizons: int, config: BaselineConfig | None = None) -> np.ndarray:
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    cfg = config or BaselineConfig()
    s = _prepare(series, cfg)
    t = _week_index(s.index)
    t_future = t[-1] + np.arange(1, horizons + 1)
    y = np.log1p(np.clip(s.to_numpy(), 0, None))
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = SARIMAX(y, exog=_fourier(t, cfg.harmonics), order=cfg.arima_order, trend="c")
            fit = model.fit(disp=False, maxiter=200)
            z = fit.forecast(steps=horizons, exog=_fourier(t_future, cfg.harmonics))
        return np.clip(np.expm1(np.asarray(z, dtype=float)), 0.0, None)
    except Exception:
        return np.full(horizons, np.nan)


def forecast_ets(series: pd.Series, horizons: int, config: BaselineConfig | None = None) -> np.ndarray:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing

    cfg = config or BaselineConfig()
    s = _prepare(series, cfg)
    y = np.log1p(np.clip(s.to_numpy(), 0, None))
    if len(y) < 2 * cfg.ets_seasonal_periods + 4:
        return np.full(horizons, np.nan)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = ExponentialSmoothing(
                y, trend="add", damped_trend=True, seasonal="add", seasonal_periods=cfg.ets_seasonal_periods
            )
            z = model.fit(optimized=True).forecast(horizons)
        return np.clip(np.expm1(np.asarray(z, dtype=float)), 0.0, None)
    except Exception:
        return np.full(horizons, np.nan)


def harmonic_arima_feature_matrix(
    dates: pd.Series | pd.DatetimeIndex,
    values: np.ndarray,
    horizons: int,
    config: BaselineConfig | None = None,
    *,
    warmup_weeks: int = 26,
) -> np.ndarray:
    """Прогнозы ARIMA с гармониками на 1..horizons недель от каждой недели ряда — признаки ML-модели.

    Параметры модели «регрессия на гармоники + ARMA-ошибки» на log(1 + y) подбираются один раз по
    последним ``train_years`` годам ряда (данные до даты прогноза), затем для каждой недели t прогноз
    на t + h считается ARMA-рекурсией по остаткам u_t = z_t − x_t·b и инновациям до t включительно —
    без данных после t. Возвращается массив (n, horizons) в исходной шкале; первые ``warmup_weeks``
    строк — NaN. Если модель не сходится — весь массив NaN.
    """

    from statsmodels.tsa.statespace.sarimax import SARIMAX

    cfg = config or BaselineConfig()
    p_ord, d_ord, q_ord = cfg.arima_order
    if d_ord != 0:
        raise ValueError("Признаки ARIMA поддерживают только d = 0.")
    idx = pd.DatetimeIndex(pd.to_datetime(dates))
    y = pd.Series(np.asarray(values, dtype=float), index=idx).interpolate(limit_direction="both").to_numpy()
    z = np.log1p(np.clip(y, 0, None))
    t = _week_index(idx)
    x = np.column_stack([np.ones(len(t)), _fourier(t, cfg.harmonics)])
    n = len(z)
    out = np.full((n, horizons), np.nan)
    fit_from = max(0, n - int(cfg.train_years * 52.18))
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = SARIMAX(z[fit_from:], exog=x[fit_from:], order=cfg.arima_order, trend="n").fit(disp=False, maxiter=200)
    except Exception:
        return out
    params = np.asarray(res.params, dtype=float)
    k = x.shape[1]
    b, phi, theta = params[:k], params[k : k + p_ord], params[k + p_ord : k + p_ord + q_ord]
    u = z - x @ b
    eps = np.zeros(n)
    for i in range(n):
        pred = sum(phi[j] * u[i - 1 - j] for j in range(p_ord) if i - 1 - j >= 0)
        pred += sum(theta[j] * eps[i - 1 - j] for j in range(q_ord) if i - 1 - j >= 0)
        eps[i] = u[i] - pred
    for i in range(n):
        hist_u = [u[i - j] for j in range(p_ord)]  # u_t, u_{t-1}, ...
        for h in range(1, horizons + 1):
            u_hat = sum(phi[j] * hist_u[j] for j in range(p_ord))
            # инновации известны только до t: вклад MA есть лишь у шагов h ≤ q
            u_hat += sum(theta[j] * eps[i + h - 1 - j] for j in range(q_ord) if h - 1 - j <= 0 and i + h - 1 - j >= 0)
            hist_u = [u_hat, *hist_u[:-1]] if p_ord else hist_u
            tt = t[i] + h
            xf = np.concatenate([[1.0], _fourier(np.array([tt]), cfg.harmonics)[0]])
            out[i, h - 1] = np.expm1(xf @ b + u_hat)
    out[:warmup_weeks] = np.nan
    return np.clip(out, 0.0, None)


def statistical_baselines(
    series: pd.Series,
    horizons: int,
    *,
    names: Sequence[str] = BASELINE_NAMES,
    config: BaselineConfig | None = None,
) -> dict[str, np.ndarray]:
    """Прогнозы базовых моделей на 1..horizons недель вперёд от последней точки ряда."""

    out: dict[str, np.ndarray] = {}
    for name in names:
        if name == "harmonic_arima":
            out[name] = forecast_harmonic_arima(series, horizons, config)
        elif name == "ets":
            out[name] = forecast_ets(series, horizons, config)
        else:
            raise ValueError(f"Неизвестный базовый прогноз {name!r}.")
    return out


__all__ = [
    "BASELINE_LABELS_RU",
    "BASELINE_NAMES",
    "BaselineConfig",
    "forecast_ets",
    "forecast_harmonic_arima",
    "harmonic_arima_feature_matrix",
    "statistical_baselines",
]
