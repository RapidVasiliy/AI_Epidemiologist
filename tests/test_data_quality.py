from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd

from ai4epi.analysis.data_quality import build_data_quality_block
from ai4epi.quality.runtime_checks import RuntimeChecker


def _weekly(n_years: int = 6, imputed_tail: int = 0) -> pd.DataFrame:
    dates = pd.date_range("2015-01-05", periods=52 * n_years, freq="7D")
    week = dates.isocalendar().week.to_numpy(int)
    y = 20 * np.exp(-((week - 6) % 52) ** 2 / 20.0) + 1.0  # пик на 6-й неделе каждый год
    df = pd.DataFrame({"datetime": dates, "inc_per_10k": y})
    df["pcr_imputed"] = False
    if imputed_tail:
        df.loc[df.index[-imputed_tail:], "pcr_imputed"] = True
    return df


def _fake_forecast(weekly: pd.DataFrame, pred_fn, *, origin_offset: int = 0) -> SimpleNamespace:
    s = weekly.set_index("datetime")["inc_per_10k"]
    origins = s.index[-60 - origin_offset : -8 - origin_offset]
    y_true = np.array([[s[o + pd.Timedelta(weeks=h)] for h in range(1, 5)] for o in origins])
    y_pred = np.array([[pred_fn(s, o, h) for h in range(1, 5)] for o in origins])
    origin = s.index[-1 - origin_offset]
    return SimpleNamespace(
        y_test_mat=y_true,
        y_pred_mat=y_pred,
        dates_test=list(origins),
        future_dates=[origin + pd.Timedelta(weeks=h) for h in range(1, 5)],
        origin_date=origin,
        config=SimpleNamespace(target_col="inc_per_10k"),
    )


def _perfect(s, o, h):
    return s[o + pd.Timedelta(weeks=h)]


def _bad(_s, _o, _h):
    return 0.0


def test_good_model_and_clean_data_is_ok():
    weekly = _weekly()
    # origin в мае — горизонты не попадают на новогодние недели
    block = build_data_quality_block(weekly, _fake_forecast(weekly, _perfect, origin_offset=30))
    assert block["reliability_code"] == "ok", block["semantic"]
    assert block["baselines"]["persistence"]["mae_skill"] == 1.0
    assert block["baselines"]["climatology"]["mae"] is not None


def test_model_worse_than_persistence_is_low():
    weekly = _weekly()
    block = build_data_quality_block(weekly, _fake_forecast(weekly, _bad, origin_offset=30))
    assert block["reliability_code"] == "low"
    assert "not_better_than_persistence" in block["semantic"]["reason_codes"]


def test_recent_imputation_lowers_reliability():
    weekly = _weekly(imputed_tail=5)
    block = build_data_quality_block(weekly, _fake_forecast(weekly, _perfect))
    assert block["pcr_imputation"]["recent_imputed_weeks"] == 5
    assert "recent_pcr_imputed_high" in block["semantic"]["reason_codes"]
    assert block["reliability_code"] == "low"


def test_holiday_horizons_flagged():
    weekly = _weekly()
    s = weekly.set_index("datetime")["inc_per_10k"]
    # origin — последняя неделя декабря: горизонты захватывают 1-ю неделю
    december = [d for d in s.index if d.isocalendar().week == 50][-1]
    offset = len(s) - 1 - list(s.index).index(december)
    block = build_data_quality_block(weekly, _fake_forecast(weekly, _perfect, origin_offset=offset))
    assert block["holiday"]["holiday_horizons"]
    assert "holiday_weeks" in block["semantic"]["reason_codes"]


def test_runtime_rule_requires_disclosure():
    ctx = {"data_quality": {"semantic": {"reliability_code": "low"}}}
    silent = {"forecast_risks": {"risk_assessment": "Риск умеренный."}, "model_quality": {"quality_summary": "R² 0.5."}}
    disclosed = {
        "forecast_risks": {"risk_assessment": "Надёжность прогноза на этой неделе низкая."},
        "model_quality": {"limitations_text": "Модель не точнее наивного прогноза."},
    }
    checker = RuntimeChecker()

    def rule_issues(sections):
        report = checker.check_bulletin(context=ctx, bulletin=sections, section_ids=list(sections))
        return [i for i in report.issues if i.check_name == "low_reliability_disclosed"]

    issues = rule_issues(silent)
    assert issues and issues[0].severity.value == "error"
    assert not rule_issues(disclosed)
