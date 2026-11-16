from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from ai4epi.analysis.br_conformal import conformal_band, conformal_quantile, evaluate_leave_one_out, fit_br_conformal
from ai4epi.analysis.mechanistic_wave import MechanisticWaveConfig, holiday_weights, run_mechanistic_wave_analysis
from ai4epi.analysis.preprocessing import (
    FeatureEngineeringConfig,
    add_holiday_features,
    build_supervised,
    is_holiday_week,
)
from ai4epi.mechanistic import ModelParams, calibrate, get_model, model_observable


def test_holiday_features_follow_calendar():
    dates = pd.date_range("2018-12-17", periods=6, freq="7D")  # 51, 52, 1, 2, 3, 4-я ISO-недели
    assert is_holiday_week(dates).tolist() == [False, False, True, False, False, False]
    out = add_holiday_features(pd.DataFrame({"datetime": dates}), max_lag=2, horizons=2)
    assert out["holiday_lag_pos"].tolist() == [-1, -1, 0, 1, 2, -1]
    assert out["holiday_ahead_pos"].tolist() == [2, 1, 0, 0, 0, 0]


def test_holiday_features_in_supervised_dataset(synthetic_weekly):
    weekly = synthetic_weekly(lead_weeks=60, wave_weeks=40)
    ds = build_supervised(weekly, config=FeatureEngineeringConfig(temp_cols=()))
    assert {"holiday_lag_pos", "holiday_ahead_pos"} <= set(ds.feature_cols)
    off = build_supervised(weekly, config=FeatureEngineeringConfig(temp_cols=(), holiday_features=False))
    assert "holiday_lag_pos" not in off.feature_cols


def test_calibration_ignores_zero_weight_weeks():
    model = get_model("total")
    template = ModelParams([0.0], [0.0], 500_000, [100.0])
    truth = model_observable(model, template.with_values([0.55], [0.7]), 20)
    observed = truth.copy()
    observed[12] *= 0.2  # «праздничный провал»
    w = np.ones(20)
    w[12] = 0.0
    weighted = calibrate(model, observed, template, method="annealing", observation_scale=1e-3, observation_weights=w, random_seed=0)
    fit = model_observable(model, weighted.best_params, 20)
    assert np.abs(fit - truth).max() / truth.max() < 0.02
    with pytest.raises(Exception, match="observation_weights"):
        calibrate(model, observed, template, method="annealing", observation_weights=np.zeros(20))


def test_holiday_weights_and_wave_with_holiday_dip(synthetic_weekly):
    weekly = synthetic_weekly(start="2018-10-08", lead_weeks=4, wave_weeks=26)
    holiday = is_holiday_week(weekly["datetime"])
    assert holiday.sum() == 1
    weekly.loc[holiday, ["total_cases_formula", "inc_per_10k"]] *= 0.3
    w = holiday_weights(weekly["datetime"], MechanisticWaveConfig())
    assert w[holiday].tolist() == [0.0] and (w[~holiday] == 1.0).all()
    origin = weekly.loc[holiday, "datetime"].iloc[0]  # текущая неделя — праздничная
    res = run_mechanistic_wave_analysis(weekly, origin_date=origin, config=MechanisticWaveConfig(method="annealing", random_seed=0, br_band="model"))
    assert res.status == "ok"
    assert any("праздничная" in m for m in res.warnings)
    with_dip = run_mechanistic_wave_analysis(
        weekly, origin_date=origin,
        config=MechanisticWaveConfig(method="annealing", random_seed=0, br_band="model", holiday_weight=1.0),
    )
    # без учёта праздника модель тянется к провалу; сравниваем с неискажённой волной
    truth = synthetic_weekly(start="2018-10-08", lead_weeks=4, wave_weeks=26).set_index("datetime")["inc_per_10k"]

    def max_error(result):
        fit = result.trajectories[~result.trajectories["is_projection"]].set_index("datetime")["best"]
        return float((fit - truth.reindex(fit.index)).abs().max())

    err, err_dip = max_error(res), max_error(with_dip)
    assert err < err_dip


def _predictions(n_seasons: int = 4) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    for season in range(n_seasons):
        for origin in range(20):
            y0 = 10 + 30 * np.sin(origin / 19 * np.pi)
            phase = "growth" if origin < 6 else "peak" if origin < 10 else "decline"
            for h in range(1, 5):
                br = y0 * (1 + 0.05 * h)
                rows.append({"season": season, "horizon": h, "phase": phase, "y_origin": y0, "br": br,
                             "y_true": br + rng.normal(0, 0.3 * (y0 + 1)), "br_lo": br - 1, "br_hi": br + 1})
    return pd.DataFrame(rows)


def test_conformal_quantile_and_table():
    assert conformal_quantile(np.arange(1, 10), 0.9) == 9  # ceil(10·0.9) = 9-я статистика
    table = fit_br_conformal(_predictions(), level=0.9, min_points=10)
    assert set(table["quantiles"]) == {"growth", "other", "all"}
    assert 0.3 < table["quantiles"]["all"]["1"] < 0.8
    lo, hi = conformal_band(np.array([5.0, 20.0, 30.0, 40.0]), 2, table, "growth", current_value=20.0)
    assert np.isnan(lo[:2]).all() and (lo[2:] <= np.array([30.0, 40.0])).all() and (hi[2:] > [30.0, 40.0]).all()
    assert (lo[2:] >= 0).all()


def test_conformal_leave_one_out_coverage_near_nominal():
    loo = evaluate_leave_one_out(_predictions(6), group_col="season", level=0.9, min_points=10)
    assert 0.8 <= loo["inside"].mean() <= 0.98
    assert loo["inside_model"].mean() < 0.5  # узкая «модельная» полоса недопокрывает


def test_conformal_band_used_in_context(tmp_path, synthetic_weekly):
    table = fit_br_conformal(_predictions(), min_points=10)
    path = tmp_path / "br_conformal.json"
    path.write_text(json.dumps(table), encoding="utf-8")
    weekly = synthetic_weekly()
    origin = weekly["datetime"].iloc[14]
    cfg = MechanisticWaveConfig(method="annealing", random_seed=0, br_conformal_path=path)
    block = run_mechanistic_wave_analysis(weekly, origin_date=origin, config=cfg).context_block
    assert block["band_type"] == "conformal" and "ошибкам" in block["band_note"]
    assert all(r["br_low"] <= r["br_value"] <= r["br_high"] for r in block["horizons"])
    missing = run_mechanistic_wave_analysis(
        weekly, origin_date=origin, config=cfg.model_copy(update={"br_conformal_path": tmp_path / "none.json"})
    )
    assert missing.context_block["band_type"] is None
    assert any("конформной полосы" in w for w in missing.warnings)
