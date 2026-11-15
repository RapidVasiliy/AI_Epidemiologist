"""
ML-прогноз расчётной заболеваемости гриппом по возрастным группам.

Статистическая (ML) часть схемы Лазутова и др. применялась к агрегированному
ряду; в ограничениях статьи отмечено отсутствие разделения по возрасту. Этот
этап переносит тот же подход на возрастные группы 0–14 и 15+ лет — те же, что
в возрастной модели Барояна–Рвачёва, — чтобы для каждой группы можно было
сопоставить ML-прогноз и механистическую модель по схеме статьи.

Для каждой группы строится тот же прямой многошаговый прогноз, что и для
общего ряда (``build_supervised`` + ``run_forecast``: HGB на горизонты 1–4
недели, конформные интервалы, проверка на последних ``test_weeks`` неделях).
Дополнительно (``cross_group_features=True``) в признаки группы входят лаги
заболеваемости другой группы: дети обычно опережают взрослых, и рост в детской
группе — ранний сигнал для взрослой.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from ai4epi.analysis.forecasting import ForecastingConfig, ForecastOutputConfig, ForecastRunResult, run_forecast
from ai4epi.analysis.mechanistic_wave import DIRECTION_LABELS_RU, ml_forecast_from_result
from ai4epi.analysis.preprocessing import FeatureEngineeringConfig, build_supervised, merge_weekly_influenza_weather
from ai4epi.core.io import write_json

JsonObject = dict[str, Any]

ASSUMPTION_NOTE_RU = (
    "Заболеваемость гриппом по возрастным группам рассчитана в предположении, что доля гриппа "
    "среди обращений с ОРВИ одинакова во всех возрастах (лабораторные данные по возрастам есть только за "
    "сезон 2017/18 и для него использована возрастная доля положительных проб); "
    "показатель — случаи на 10 тыс. населения соответствующей группы."
)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True, arbitrary_types_allowed=True)


class AgeGroupForecastConfig(StrictModel):
    cross_group_features: bool = True
    cross_group_lags: tuple[int, ...] = (0, 1, 2, 3)
    direction_change_pct_threshold: float = Field(default=10.0, gt=0.0)
    round_digits: int = Field(default=3, ge=0, le=8)


class AgeGroupForecastOutputConfig(StrictModel):
    output_dir: Path = Path("age_group_forecast")
    summary_filename: str = "age_group_forecast.json"


class AgeGroupForecastRunResult(StrictModel):
    context_block: JsonObject
    results: dict[str, Any]
    ml_forecasts: dict[str, JsonObject]
    artifacts: dict[str, Path] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)


def add_cross_group_features(
    frame: pd.DataFrame,
    others: Mapping[str, pd.DataFrame],
    *,
    lags: tuple[int, ...],
) -> pd.DataFrame:
    """Добавить лаги ``inc_per_10k`` других групп как признаки ``xg_<code>_lag<k>``."""

    out = frame.copy()
    out["datetime"] = pd.to_datetime(out["datetime"])
    for code, other in others.items():
        series = other.set_index(pd.to_datetime(other["datetime"]))["inc_per_10k"]
        aligned = series.reindex(out["datetime"]).to_numpy(float)
        for k in lags:
            out[f"xg_{code}_lag{k}"] = pd.Series(aligned).shift(k).to_numpy()
    return out


def run_age_group_forecast(
    age_group_weekly: Mapping[str, pd.DataFrame],
    *,
    labels: Mapping[str, str] | None = None,
    weather_weekly: pd.DataFrame | None = None,
    feature_config: FeatureEngineeringConfig | None = None,
    forecasting_config: ForecastingConfig | None = None,
    config: AgeGroupForecastConfig | None = None,
    output: AgeGroupForecastOutputConfig | None = None,
) -> AgeGroupForecastRunResult:
    """Построить ML-прогноз по каждой возрастной группе и блок контекста."""

    cfg = config or AgeGroupForecastConfig()
    fcfg = feature_config or FeatureEngineeringConfig()
    labels = dict(labels or {})
    rd = cfg.round_digits
    results: dict[str, ForecastRunResult] = {}
    ml_forecasts: dict[str, JsonObject] = {}
    groups_out: list[JsonObject] = []
    artifacts: dict[str, Path] = {}
    warnings: list[str] = []

    for code, weekly in age_group_weekly.items():
        frame = weekly.copy()
        if weather_weekly is not None:
            frame = merge_weekly_influenza_weather(frame, weather_weekly, weekly_config=fcfg.weekly_frame_config)
        if cfg.cross_group_features:
            others = {c: w for c, w in age_group_weekly.items() if c != code}
            frame = add_cross_group_features(frame, others, lags=cfg.cross_group_lags)
        supervised = build_supervised(frame, config=fcfg)
        fr = run_forecast(
            supervised,
            config=forecasting_config,
            output=ForecastOutputConfig(output_dir=Path(output.output_dir) / code) if output is not None else None,
        )
        results[code] = fr
        ml_forecasts[code] = ml_forecast_from_result(fr)
        for name, path in fr.artifacts.items():
            artifacts[f"{code}.{name}"] = Path(path)

        history = supervised.data.loc[supervised.data["datetime"] <= fr.origin_date, fcfg.target_col].dropna()
        current, previous = float(history.iloc[-1]), float(history.iloc[-2])
        point = np.asarray(fr.y_hat, dtype=float)
        lo = np.minimum(np.asarray(fr.y_lo, dtype=float), np.asarray(fr.y_hi, dtype=float))
        hi = np.maximum(np.asarray(fr.y_lo, dtype=float), np.asarray(fr.y_hi, dtype=float))
        dates = pd.to_datetime(pd.Series(fr.future_dates)).reset_index(drop=True)
        horizons = [
            {
                "horizon_weeks": k + 1,
                "target_date": dates.iloc[k].date().isoformat(),
                "target_iso_week": int(dates.iloc[k].isocalendar().week),
                "point_forecast": round(float(point[k]), rd),
                "q_lo": round(float(lo[k]), rd),
                "q_hi": round(float(hi[k]), rd),
            }
            for k in range(len(point))
        ]
        change = (point[-1] - current) / max(abs(current), 1e-9) * 100.0
        thr = cfg.direction_change_pct_threshold
        direction = "increase" if change >= thr else "decrease" if change <= -thr else "stable"
        overall = fr.metrics.overall
        persistence_r2, persistence_mae = _persistence_metrics(fr, supervised.data, fcfg.target_col)
        skill = None
        if persistence_mae and persistence_mae > 0:
            skill = round(1.0 - float(overall["mae_overall"]) / persistence_mae, 3)
        if skill is not None and skill <= 0:
            warnings.append(
                f"{code}: на проверочном периоде ML-прогноз не лучше наивного (значение текущей недели); "
                f"MAE {overall['mae_overall']:.2f} против {persistence_mae:.2f}."
            )
        coverage = None
        if fr.interval_metrics is not None and "coverage" in fr.interval_metrics:
            coverage = round(float(fr.interval_metrics["coverage"].mean()), 3)
        groups_out.append(
            {
                "age_group_code": code,
                "age_group_label": labels.get(code, code),
                "current_value": round(current, rd),
                "previous_value": round(previous, rd),
                "change_pct": round((current - previous) / max(abs(previous), 1e-9) * 100.0, 1),
                "horizons": horizons,
                "direction_code": direction,
                "direction": DIRECTION_LABELS_RU[direction],
                "change_to_h4_pct": round(float(change), 1),
                "test_metrics": {
                    "r2": round(float(overall["r2_overall"]), 3),
                    "rmse": round(float(overall["rmse_overall"]), 3),
                    "mae": round(float(overall["mae_overall"]), 3),
                    "interval_coverage": coverage,
                    "n_test_weeks": int(np.asarray(fr.y_test_mat).shape[0]),
                    "persistence_r2": persistence_r2,
                    "persistence_mae": round(persistence_mae, 3) if persistence_mae is not None else None,
                    "mae_skill_vs_persistence": skill,
                    "skill_note": "доля снижения MAE относительно наивного прогноза «как на текущей неделе»; ≤ 0 — модель не лучше наивной",
                },
            }
        )

    highest = max(groups_out, key=lambda g: g["current_value"])["age_group_code"] if groups_out else None
    block: JsonObject = {
        "model_name_ru": "градиентный бустинг (HGB), отдельная модель для каждой возрастной группы",
        "metric_label_ru": "расчётная заболеваемость гриппом на 10 тыс. населения группы",
        "assumption_note": ASSUMPTION_NOTE_RU,
        "cross_group_features": cfg.cross_group_features,
        "cross_group_note": (
            "в признаки каждой группы включены лаги заболеваемости другой группы"
            if cfg.cross_group_features
            else "модели групп независимы"
        ),
        "groups": groups_out,
        "semantic": {
            "highest_current_group": highest,
            "directions": {g["age_group_code"]: g["direction_code"] for g in groups_out},
            "groups_not_better_than_persistence": [
                g["age_group_code"]
                for g in groups_out
                if g["test_metrics"]["mae_skill_vs_persistence"] is not None and g["test_metrics"]["mae_skill_vs_persistence"] <= 0
            ],
        },
    }
    if output is not None:
        out = Path(output.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        artifacts["summary_json"] = write_json(block, out / output.summary_filename)
    return AgeGroupForecastRunResult(
        context_block=block,
        results=dict(results),
        ml_forecasts=ml_forecasts,
        artifacts=artifacts,
        warnings=warnings,
    )


def _persistence_metrics(fr: ForecastRunResult, data: pd.DataFrame, target_col: str) -> tuple[float | None, float | None]:
    """R² и MAE наивного прогноза y(t+h) = y(t) на тех же проверочных точках."""

    y_true = np.asarray(fr.y_test_mat, dtype=float)
    origins = pd.to_datetime(pd.Series(fr.dates_test)).reset_index(drop=True)
    series = data.set_index(pd.to_datetime(data["datetime"]))[target_col]
    base = series.reindex(origins).to_numpy(float)
    if y_true.ndim != 2 or len(base) != len(y_true) or np.isnan(base).any():
        return None, None
    pred = np.repeat(base[:, None], y_true.shape[1], axis=1)
    resid = y_true.ravel() - pred.ravel()
    ss_tot = float(np.sum((y_true.ravel() - y_true.mean()) ** 2))
    r2 = round(1.0 - float(np.sum(resid**2)) / ss_tot, 3) if ss_tot > 0 else None
    return r2, float(np.mean(np.abs(resid)))


__all__ = [
    "AgeGroupForecastConfig",
    "AgeGroupForecastOutputConfig",
    "AgeGroupForecastRunResult",
    "add_cross_group_features",
    "run_age_group_forecast",
]
