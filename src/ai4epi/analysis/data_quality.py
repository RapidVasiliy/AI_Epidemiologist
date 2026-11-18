"""
Контроль качества данных и сравнение основной ML-модели с наивными прогнозами.

Блок ``data_quality`` строится при каждом запуске анализа и отвечает на
вопрос: насколько можно доверять количественному прогнозу этой недели.

* **Данные.** Доля недель, на которых данные ПЦР восстановлены интерполяцией
  (``pcr_imputed`` из локального загрузчика), — в обучающей части ряда, в
  проверочном периоде и в последних неделях; новогодние недели (ISO 52, 53, 1),
  на которых регистрация систематически занижена, — для текущей недели и
  горизонтов прогноза.
* **Базовые прогнозы.** На тех же проверочных точках, что и метрики модели:
  наивный прогноз «как на текущей неделе» и климатический прогноз (среднее
  значение той же ISO-недели в предыдущих сезонах). Skill = 1 − MAE модели /
  MAE базового прогноза; skill ≤ 0 — модель не лучше базового прогноза.
* **Итог.** ``reliability_code``: ``ok`` / ``reduced`` / ``low`` со списком
  причин. Секции «Прогноз и оценка рисков» и «Качество модели» обязаны
  отражать пониженную надёжность (runtime-правило
  ``low_reliability_disclosed``).
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from ai4epi.analysis.baselines import BASELINE_LABELS_RU, BASELINE_NAMES, statistical_baselines

JsonObject = dict[str, Any]
ReliabilityCode = Literal["ok", "reduced", "low"]

RELIABILITY_LABELS_RU: dict[str, str] = {
    "ok": "обычная",
    "reduced": "пониженная",
    "low": "низкая",
}

REASON_LABELS_RU: dict[str, str] = {
    "not_better_than_persistence": "на проверочном периоде модель не точнее наивного прогноза «как на текущей неделе»",
    "weak_skill_vs_persistence": "выигрыш модели относительно наивного прогноза меньше 10 % по MAE",
    "not_better_than_climatology": "на проверочном периоде модель не точнее среднего многолетнего значения для той же недели",
    "not_better_than_statistical": (
        "на проверочном периоде модель не точнее простой статистической модели временного ряда "
        "(ARIMA с гармониками годового цикла или экспоненциального сглаживания)"
    ),
    "recent_pcr_imputed_high": "в последних неделях велика доля недель без данных ПЦР (значения восстановлены интерполяцией)",
    "recent_pcr_imputed": "в последних неделях есть недели без данных ПЦР (значения восстановлены интерполяцией)",
    "test_pcr_imputed_high": "в проверочном периоде велика доля недель без данных ПЦР",
    "holiday_weeks": "текущая неделя или горизонты прогноза приходятся на новогодние недели, когда регистрация систематически занижена",
    "models_disagree": "прогнозы модели машинного обучения и механистической модели расходятся; по бэктесту в такие недели ошибка ML выше, а прогнозные интервалы чаще не покрывают фактические значения",
}

HOLIDAY_ISO_WEEKS: tuple[int, ...] = (52, 53, 1)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class DataQualityConfig(StrictModel):
    recent_weeks: int = Field(default=8, ge=1)
    imputed_share_warn: float = Field(default=0.2, ge=0.0, le=1.0)
    imputed_share_critical: float = Field(default=0.4, ge=0.0, le=1.0)
    weak_skill_threshold: float = Field(default=0.1)
    min_climatology_seasons: int = Field(default=3, ge=1)
    # ARIMA с гармониками и ETS (ai4epi.analysis.baselines) переобучаются на каждой дате проверочного
    # периода (~0.2 с на дату); False — только наивный и климатический прогнозы
    statistical_baselines: bool = True
    holiday_iso_weeks: tuple[int, ...] = HOLIDAY_ISO_WEEKS
    round_digits: int = Field(default=3, ge=0, le=8)


def build_data_quality_block(
    influenza_weekly: pd.DataFrame,
    forecast_result: Any,
    *,
    config: DataQualityConfig | None = None,
) -> JsonObject:
    """Собрать блок ``data_quality`` по недельному ряду и результату ``run_forecast``."""

    cfg = config or DataQualityConfig()
    rd = cfg.round_digits
    weekly = influenza_weekly.copy()
    weekly["datetime"] = pd.to_datetime(weekly["datetime"])
    weekly = weekly.sort_values("datetime").drop_duplicates("datetime").reset_index(drop=True)
    target_col = forecast_result.config.target_col
    origin = pd.Timestamp(forecast_result.origin_date)
    known = weekly[weekly["datetime"] <= origin]

    imputed = known["pcr_imputed"].astype(bool) if "pcr_imputed" in known else pd.Series(False, index=known.index)
    test_origins = pd.to_datetime(pd.Series(forecast_result.dates_test)).reset_index(drop=True)
    in_test = known["datetime"].between(test_origins.min(), test_origins.max()) if len(test_origins) else known["datetime"].isna()
    recent = imputed.tail(cfg.recent_weeks)
    imputation = {
        "available": "pcr_imputed" in known,
        "train_share": _share(imputed),
        "test_share": _share(imputed[in_test]),
        "recent_share": _share(recent),
        "recent_weeks": len(recent),
        "recent_imputed_weeks": int(recent.sum()),
        "total_imputed_weeks": int(imputed.sum()),
    }

    future_dates = pd.to_datetime(pd.Series(forecast_result.future_dates)).reset_index(drop=True)
    holiday_horizons = [
        k + 1 for k, d in enumerate(future_dates) if int(d.isocalendar().week) in cfg.holiday_iso_weeks
    ]
    holiday = {
        "iso_weeks": list(cfg.holiday_iso_weeks),
        "current_week_is_holiday": int(origin.isocalendar().week) in cfg.holiday_iso_weeks,
        "holiday_horizons": holiday_horizons,
        "note": "на 1-й ISO-неделе года регистрация систематически занижена относительно соседних недель",
    }

    baselines = compare_with_baselines(weekly, forecast_result, target_col=target_col, config=cfg)

    reasons: list[str] = []
    severity = 0
    pers = baselines["persistence"]["mae_skill"]
    clim = baselines["climatology"]["mae_skill"]
    if pers is not None and pers <= 0:
        reasons.append("not_better_than_persistence")
        severity = max(severity, 2)
    elif pers is not None and pers < cfg.weak_skill_threshold:
        reasons.append("weak_skill_vs_persistence")
        severity = max(severity, 1)
    if clim is not None and clim <= 0:
        reasons.append("not_better_than_climatology")
        severity = max(severity, 1)
    stat_skills = [baselines[n]["mae_skill"] for n in BASELINE_NAMES if n in baselines]
    stat_skill = min((v for v in stat_skills if v is not None), default=None)
    if stat_skill is not None and stat_skill <= 0:
        reasons.append("not_better_than_statistical")
        severity = max(severity, 1)
    if imputation["recent_share"] is not None and imputation["recent_share"] >= cfg.imputed_share_critical:
        reasons.append("recent_pcr_imputed_high")
        severity = max(severity, 2)
    elif imputation["recent_imputed_weeks"]:
        reasons.append("recent_pcr_imputed")
        severity = max(severity, 1)
    if imputation["test_share"] is not None and imputation["test_share"] >= cfg.imputed_share_warn:
        reasons.append("test_pcr_imputed_high")
        severity = max(severity, 1)
    if holiday["current_week_is_holiday"] or holiday_horizons:
        reasons.append("holiday_weeks")
        severity = max(severity, 1)
    code: ReliabilityCode = ("ok", "reduced", "low")[severity]

    return {
        "reliability_code": code,
        "reliability": RELIABILITY_LABELS_RU[code],
        "reasons": [REASON_LABELS_RU[r] for r in reasons],
        "pcr_imputation": imputation,
        "holiday": holiday,
        "baselines": baselines,
        "semantic": {
            "reliability_code": code,
            "reason_codes": reasons,
            "skill_vs_persistence": pers,
            "skill_vs_climatology": clim,
            "skill_vs_harmonic_arima": baselines.get("harmonic_arima", {}).get("mae_skill"),
            "skill_vs_ets": baselines.get("ets", {}).get("mae_skill"),
        },
        "round_digits": rd,
    }


def add_reliability_reason(block: JsonObject, reason: str, *, severity: int) -> JsonObject:
    """Добавить причину и при необходимости понизить ``reliability_code`` (1 — reduced, 2 — low)."""

    codes = block["semantic"].setdefault("reason_codes", [])
    if reason not in codes:
        codes.append(reason)
        block["reasons"].append(REASON_LABELS_RU[reason])
    order = ("ok", "reduced", "low")
    new = order[max(order.index(block["reliability_code"]), severity)]
    block["reliability_code"] = new
    block["reliability"] = RELIABILITY_LABELS_RU[new]
    block["semantic"]["reliability_code"] = new
    return block


def compare_with_baselines(
    weekly: pd.DataFrame,
    forecast_result: Any,
    *,
    target_col: str = "inc_per_10k",
    config: DataQualityConfig | None = None,
) -> JsonObject:
    """Метрики модели и базовых прогнозов на проверочных точках ``run_forecast``."""

    cfg = config or DataQualityConfig()
    rd = cfg.round_digits
    y_true = np.asarray(forecast_result.y_test_mat, dtype=float)
    y_pred = np.asarray(forecast_result.y_pred_mat, dtype=float)
    origins = pd.to_datetime(pd.Series(forecast_result.dates_test)).reset_index(drop=True)
    series = weekly.set_index("datetime")[target_col].astype(float)
    horizons = y_true.shape[1] if y_true.ndim == 2 else 0

    persistence = np.repeat(series.reindex(origins).to_numpy(float)[:, None], horizons, axis=1)
    iso_weeks = pd.Series(series.index.isocalendar().week.to_numpy(int), index=series.index)
    climatology = np.full((len(origins), horizons), np.nan)
    for i, origin in enumerate(origins):
        for h in range(horizons):
            climatology[i, h] = _climatology_value(series, iso_weeks, origin, origin + pd.Timedelta(weeks=h + 1), cfg.min_climatology_seasons)

    model_mae = _mae(y_true, y_pred)
    out: JsonObject = {
        "test_period": {
            "from": origins.min().date().isoformat() if len(origins) else None,
            "to": origins.max().date().isoformat() if len(origins) else None,
            "n_origins": len(origins),
        },
        "model": {"r2": _r2(y_true, y_pred, rd), "mae": _round(model_mae, rd)},
        "definition_ru": {
            "persistence": "наивный прогноз: значение текущей недели на все горизонты",
            "climatology": "среднее значение той же ISO-недели в предыдущих сезонах",
            "mae_skill": "1 − MAE модели / MAE базового прогноза; ≤ 0 — модель не лучше базового прогноза",
            **BASELINE_LABELS_RU,
        },
    }
    candidates = [("persistence", persistence), ("climatology", climatology)]
    if cfg.statistical_baselines and horizons:
        stat = {name: np.full((len(origins), horizons), np.nan) for name in BASELINE_NAMES}
        for i, origin in enumerate(origins):
            history = series[series.index <= origin].dropna()
            if len(history) < 104:
                continue
            for name, values in statistical_baselines(history, horizons).items():
                stat[name][i] = values
        candidates += list(stat.items())
    for name, pred in candidates:
        mask = np.isfinite(pred)
        base_mae = _mae(y_true, pred, mask)
        model_mae_masked = _mae(y_true, y_pred, mask)
        skill = None
        if base_mae is not None and base_mae > 0 and model_mae_masked is not None:
            skill = round(1.0 - model_mae_masked / base_mae, rd)
        out[name] = {
            "r2": _r2(y_true, pred, rd, mask),
            "mae": _round(base_mae, rd),
            "mae_skill": skill,
            "coverage": round(float(mask.mean()), 3) if mask.size else None,
        }
    return out


def _climatology_value(
    series: pd.Series, iso_weeks: pd.Series, origin: pd.Timestamp, target: pd.Timestamp, min_seasons: int
) -> float:
    """Среднее той же ISO-недели в прошлых сезонах (известных к ``origin``, не ближе полугода к цели)."""

    week = int(target.isocalendar().week)
    mask = (iso_weeks.to_numpy() == week) & (series.index <= origin) & (series.index < target - pd.Timedelta(weeks=26))
    values = series.to_numpy(float)[mask]
    values = values[np.isfinite(values)]
    return float(values.mean()) if len(values) >= min_seasons else float("nan")


def _share(values: pd.Series) -> float | None:
    return round(float(values.mean()), 3) if len(values) else None


def _mae(y: np.ndarray, p: np.ndarray, mask: np.ndarray | None = None) -> float | None:
    if y.size == 0:
        return None
    m = np.isfinite(p) if mask is None else mask
    return float(np.mean(np.abs(y[m] - p[m]))) if m.any() else None


def _r2(y: np.ndarray, p: np.ndarray, rd: int, mask: np.ndarray | None = None) -> float | None:
    m = np.isfinite(p) if mask is None else mask
    if not m.any():
        return None
    yt, yp = y[m], p[m]
    ss_tot = float(np.sum((yt - yt.mean()) ** 2))
    return round(1.0 - float(np.sum((yt - yp) ** 2)) / ss_tot, rd) if ss_tot > 0 else None


def _round(value: float | None, rd: int) -> float | None:
    return None if value is None else round(float(value), rd)


__all__ = [
    "HOLIDAY_ISO_WEEKS",
    "RELIABILITY_LABELS_RU",
    "DataQualityConfig",
    "add_reliability_reason",
    "build_data_quality_block",
    "compare_with_baselines",
]
