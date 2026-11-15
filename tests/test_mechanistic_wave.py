from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from ai4epi.analysis.mechanistic_wave import (
    MechanisticWaveConfig,
    MechanisticWaveOutputConfig,
    detect_wave,
    run_mechanistic_wave_analysis,
)
from ai4epi.core.sections import default_section_registry

FAST = MechanisticWaveConfig(method="annealing", random_seed=0)


def _ml_from_block(block, factor=1.0):
    return {
        "point": [r["br_value"] * factor for r in block["horizons"]],
        "lo": [r["br_value"] * factor * 0.8 for r in block["horizons"]],
        "hi": [r["br_value"] * factor * 1.2 for r in block["horizons"]],
        "dates": [r["target_date"] for r in block["horizons"]],
    }


def test_detect_wave_requires_threshold_and_growth():
    cfg = MechanisticWaveConfig()
    dates = pd.Series(pd.date_range("2020-09-07", periods=6, freq="7D"))
    assert detect_wave(np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]), dates, cfg) == (False, None)  # ниже порога
    assert detect_wave(np.array([1, 5, 2, 6, 3, 7.0]), dates, cfg)[0] is False  # нет 3 недель роста подряд
    ok, onset = detect_wave(np.array([0.1, 0.5, 2, 4, 8, 6.0]), dates, cfg)
    assert ok and onset == dates.iloc[0]


def test_no_wave_on_flat_series(synthetic_weekly):
    weekly = synthetic_weekly().iloc[:6]
    weekly["total_cases_formula"] = 10.0
    weekly["inc_per_10k"] = 10.0 / 5e6 * 1e4
    result = run_mechanistic_wave_analysis(weekly, config=FAST)
    assert result.status == "no_wave" and result.context_block is None


@pytest.mark.parametrize(
    "n_weeks,expected_phase", [(14, "growth"), (19, "peak"), (23, "decline"), (27, "post_wave")]
)
def test_phase_and_peak_on_synthetic_wave(n_weeks, expected_phase, synthetic_weekly):
    full = synthetic_weekly()
    true_peak = full.loc[full["inc_per_10k"].idxmax(), "datetime"]
    # калибровка с той недели, где синтетическая волна стартовала со 100 заражённых
    cfg = FAST.model_copy(update={"calibration_start_date": full["datetime"].iloc[6].date()})
    result = run_mechanistic_wave_analysis(full.iloc[:n_weeks], config=cfg)
    assert result.status == "ok"
    block = result.context_block
    assert block["phase_code"] == expected_phase
    assert block["fit_r2_in_sample"] > 0.99
    assert abs((pd.Timestamp(block["peak"]["date"]) - true_peak).days) <= 7
    assert block["early_phase_caution"] is (expected_phase == "growth")
    assert len(block["horizons"]) == 4
    json.dumps(block)  # блок сериализуем для GlobalContext


def test_agreement_codes(synthetic_weekly):
    weekly = synthetic_weekly().iloc[:14]
    base = run_mechanistic_wave_analysis(weekly, config=FAST).context_block
    agree = run_mechanistic_wave_analysis(weekly, ml_forecast=_ml_from_block(base), config=FAST).context_block
    assert agree["comparison_with_ml"]["agreement_code"] == "agree"
    falling = {"point": [0.1] * 4, "lo": [0.0] * 4, "hi": [0.2] * 4, "dates": []}
    disagree = run_mechanistic_wave_analysis(weekly, ml_forecast=falling, config=FAST).context_block
    cmp_ = disagree["comparison_with_ml"]
    assert cmp_["agreement_code"] == "disagree"
    assert "direction" in cmp_["disagreement_reasons"]
    assert disagree["semantic"]["agreement_code"] == "disagree"


def test_calibration_start_options(synthetic_weekly):
    weekly = synthetic_weekly().iloc[:16]
    onset = run_mechanistic_wave_analysis(weekly, config=FAST).context_block
    season = run_mechanistic_wave_analysis(
        weekly, config=FAST.model_copy(update={"calibration_start": "season_start"})
    ).context_block
    assert onset["calibration_start_date"] > season["calibration_start_date"]
    assert season["calibration_start_date"] == "2018-08-27"  # ISO-неделя 35 сезона 2018–2019


def test_artifacts_written(tmp_path, synthetic_weekly):
    result = run_mechanistic_wave_analysis(
        synthetic_weekly().iloc[:16], config=FAST, output=MechanisticWaveOutputConfig(output_dir=tmp_path)
    )
    traj = pd.read_csv(result.artifacts["trajectories_csv"])
    assert {"datetime", "observed", "best", "is_projection"} <= set(traj.columns)
    assert json.loads(result.artifacts["summary_json"].read_text(encoding="utf-8"))["status"] == "ok"


def test_section_registered_and_optional():
    registry = default_section_registry()
    section = registry.get("mechanistic_wave")
    assert section.required is False and section.activation_path == "mechanistic_wave"
    ids = registry.section_ids()
    assert ids.index("forecast_risks") < ids.index("mechanistic_wave") < ids.index("shap_interpretation")


def test_agreement_not_assessed_after_wave(synthetic_weekly):
    full = synthetic_weekly()
    cfg = FAST.model_copy(update={"calibration_start_date": full["datetime"].iloc[6].date()})
    weekly = full.iloc[:27]  # фаза post_wave
    ml = {"point": [5.0] * 4, "lo": [0.0] * 4, "hi": [10.0] * 4, "dates": []}
    block = run_mechanistic_wave_analysis(weekly, ml_forecast=ml, config=cfg).context_block
    assert block["phase_code"] == "post_wave"
    assert block["comparison_with_ml"]["agreement_code"] is None
    assert block["comparison_with_ml"]["agreement_note"]


def test_small_absolute_gaps_are_not_disagreement():
    from ai4epi.analysis.mechanistic_wave import compare_with_ml

    dates = pd.Series(pd.date_range("2020-01-06", periods=6, freq="7D"))
    best = np.array([1.0, 1.0, 1.0, 1.2, 1.4, 1.6])  # БР: рост с 1 до 1.6 на 10 тыс.
    ml = {"point": [3.0, 3.5, 4.0, 4.5], "lo": [0.0] * 4, "hi": [9.0] * 4, "dates": []}  # ML примерно втрое выше
    _, cmp_floor, _ = compare_with_ml(1.0, best, None, None, dates, 1, ml, MechanisticWaveConfig())
    assert cmp_floor["agreement_code"] == "agree"  # разница < 5 на 10 тыс. не считается расхождением
    strict = MechanisticWaveConfig(agreement_gap_floor_per_10k=0.0)
    _, cmp_strict, _ = compare_with_ml(1.0, best, None, None, dates, 1, ml, strict)
    assert "scale" in cmp_strict["disagreement_reasons"]


def test_bootstrap_band_and_reproduction_number(synthetic_weekly):
    weekly = synthetic_weekly()
    noisy = weekly.copy()
    rng = np.random.default_rng(0)
    noisy["total_cases_formula"] = np.maximum(noisy["total_cases_formula"] * rng.normal(1, 0.1, len(noisy)), 0).round()
    noisy["inc_per_10k"] = noisy["total_cases_formula"] / noisy["total_population"] * 10_000
    origin = noisy["datetime"].iloc[14]
    cfg = MechanisticWaveConfig(method="bootstrap", bootstrap_replicates=15, random_seed=0)
    res = run_mechanistic_wave_analysis(noisy, origin_date=origin, config=cfg)
    assert res.status == "ok"
    block = res.context_block
    assert block["band_type"] == "predictive" and block["n_trajectories"] == 15
    rt = block["reproduction_number"]
    assert rt["current"] > 0 and rt["max"] >= rt["current"]
    assert rt["current_q05"] is not None and rt["current_q05"] <= rt["current_q95"]
    traj = res.trajectories
    assert {"r_t", "r_t_q05", "r_t_q95", "q05", "q95"} <= set(traj.columns)
    assert (traj["q95"] >= traj["q05"]).all()
    param = run_mechanistic_wave_analysis(
        noisy, origin_date=origin, config=cfg.model_copy(update={"bootstrap_band": "parameter"})
    )
    width = lambda t: float((t["q95"] - t["q05"]).mean())  # noqa: E731
    assert width(param.trajectories) < width(traj)  # шум наблюдений расширяет полосу
