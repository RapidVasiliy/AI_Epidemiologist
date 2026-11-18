"""
Скользящий бэктест схемы «ML + модель Барояна–Рвачёва» (еженедельное переобучение).

Воспроизводит процедуру статьи Лазутова и др.: для каждой недели ``t`` из
периода проверки

1. берётся ряд в состоянии «как было известно на неделю t» (для локальных
   данных — с обработкой пропусков ПЦР только по данным до t);
2. для каждого горизонта h = 1..4 модель HGB обучается заново на строках, у
   которых целевое значение уже известно к t, и даёт прогноз на t + h;
3. прогнозный интервал — конформный, по абсолютным ошибкам предыдущих недель
   бэктеста, цели которых известны к t (последние ``conformal_window`` ошибок);
4. наивный (значение недели t) и климатический (среднее той же ISO-недели в
   прошлых сезонах) прогнозы — для сравнения;
5. модель Барояна–Рвачёва (если включена) перекалибровывается по данным до t;
   для недели фиксируются фаза, значения на горизонтах и код согласия с ML.

Затем считаются метрики по горизонтам, по фазам волны и — главный вопрос
схемы — ошибка ML-прогноза при согласии и при расхождении моделей.

Фактические значения берутся из полного (последнего) ряда.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from datetime import date
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from ai4epi.analysis.baselines import BASELINE_NAMES, statistical_baselines
from ai4epi.analysis.forecasting import (
    HistGradientBoostingPointConfig,
    conformal_radius_from_abs_errors,
    fit_models_hist_gbdt,
)
from ai4epi.analysis.mechanistic_wave import MechanisticWaveConfig, run_mechanistic_wave_analysis
from ai4epi.analysis.preprocessing import FeatureEngineeringConfig, apply_holiday_ratio, build_supervised
from ai4epi.core.io import TableWriteOptions, write_json, write_table

JsonObject = dict[str, Any]
DataAtFn = Callable[[pd.Timestamp], pd.DataFrame]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True, arbitrary_types_allowed=True)


class BacktestConfig(StrictModel):
    start: date
    end: date
    horizons: int = Field(default=4, ge=1, le=12)
    min_train_rows: int = Field(default=104, ge=10)
    model_variants: dict[str, HistGradientBoostingPointConfig] = Field(
        default_factory=lambda: {"hgb": HistGradientBoostingPointConfig()}
    )
    primary_variant: str = "hgb"
    interval_alpha: float = Field(default=0.20, gt=0.0, lt=1.0)
    conformal_window: int = Field(default=104, ge=5)
    min_conformal_errors: int = Field(default=20, ge=5)
    # Недели перед периодом проверки: считаются только ML-прогнозы, их ошибки
    # используются для конформных интервалов, в метрики они не входят.
    interval_warmup_weeks: int = Field(default=52, ge=0)
    # Масштабированные по уровню интервалы: квантиль |e| / (y_t + c) умножается на
    # (y_t + c), где y_t — значение недели прогноза. Сравниваются с абсолютными.
    interval_level_offset: float = Field(default=1.0, gt=0.0)
    run_mechanistic: bool = True
    # ARIMA с гармониками и ETS (ai4epi.analysis.baselines) на каждой дате прогноза
    statistical_baselines: bool = True
    mechanistic: MechanisticWaveConfig = Field(default_factory=lambda: MechanisticWaveConfig(cores=1))
    random_state: int = 42
    n_jobs: int = Field(default_factory=lambda: max(1, (os.cpu_count() or 1)))
    min_climatology_seasons: int = Field(default=3, ge=1)


class BacktestResult(StrictModel):
    predictions: Any
    metrics: JsonObject
    artifacts: dict[str, Path] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_backtest(
    data_at: DataAtFn,
    truth: pd.DataFrame,
    *,
    config: BacktestConfig,
    feature_config: FeatureEngineeringConfig | None = None,
    output_dir: str | Path | None = None,
    label: str = "",
) -> BacktestResult:
    """Выполнить бэктест.

    :param data_at: функция ``t -> недельный ряд, известный к неделе t``
        (``datetime``, ``iso_year``, ``iso_week``, ``total_population``,
        ``total_cases_formula``, ``inc_per_10k``, опционально ``pcr_imputed``)
    :param truth: полный ряд для фактических значений
    """

    cfg = config
    fcfg = feature_config or FeatureEngineeringConfig(temp_cols=())
    if cfg.primary_variant not in cfg.model_variants:
        raise ValueError(f"primary_variant {cfg.primary_variant!r} отсутствует в model_variants.")
    truth = truth.copy()
    truth["datetime"] = pd.to_datetime(truth["datetime"])
    truth_series = truth.set_index("datetime")["inc_per_10k"].astype(float)
    start, end = pd.Timestamp(cfg.start), pd.Timestamp(cfg.end)
    warm_start = start - pd.Timedelta(weeks=cfg.interval_warmup_weeks)
    origins = [d for d in truth_series.index if warm_start <= d <= end]
    if not any(d >= start for d in origins):
        raise ValueError("В заданном периоде нет недель.")

    warm_cfg = cfg.model_copy(update={"run_mechanistic": False})
    tasks = [(t, data_at, cfg if t >= start else warm_cfg, fcfg) for t in origins]
    if cfg.n_jobs > 1:
        from joblib import Parallel, delayed

        rows_nested = Parallel(n_jobs=cfg.n_jobs, backend="loky")(delayed(_origin_task)(*task) for task in tasks)
    else:
        rows_nested = [_origin_task(*task) for task in tasks]
    rows = [row for chunk in rows_nested for row in chunk]
    pred = pd.DataFrame(rows)
    pred["y_true"] = pred["target_date"].map(truth_series)
    pred = pred.sort_values(["origin", "horizon"]).reset_index(drop=True)
    pred = _add_conformal_intervals(pred, cfg)
    pred["warmup"] = pred["origin"] < start
    pred = pred[~pred["warmup"]].drop(columns="warmup").reset_index(drop=True)
    metrics = evaluate_backtest(pred, cfg, label=label)

    artifacts: dict[str, Path] = {}
    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        artifacts["predictions_csv"] = write_table(pred, out / "predictions.csv", options=TableWriteOptions(index=False))
        artifacts["metrics_json"] = write_json(metrics, out / "metrics.json")
        report = out / "report.md"
        report.write_text(render_backtest_report(metrics), encoding="utf-8")
        artifacts["report_md"] = report
    return BacktestResult(predictions=pred, metrics=metrics, artifacts=artifacts)


# ---------------------------------------------------------------------------
# One origin
# ---------------------------------------------------------------------------


def _origin_task(
    origin: pd.Timestamp,
    data_at: DataAtFn,
    cfg: BacktestConfig,
    fcfg: FeatureEngineeringConfig,
) -> list[JsonObject]:
    from threadpoolctl import threadpool_limits

    with threadpool_limits(limits=1):
        return _origin_task_impl(origin, data_at, cfg, fcfg)


def _origin_task_impl(
    origin: pd.Timestamp,
    data_at: DataAtFn,
    cfg: BacktestConfig,
    fcfg: FeatureEngineeringConfig,
) -> list[JsonObject]:
    weekly = data_at(origin)
    weekly = weekly[pd.to_datetime(weekly["datetime"]) <= origin].reset_index(drop=True)
    base: JsonObject = {"origin": origin, "status": "ok"}
    if weekly.empty or pd.Timestamp(weekly["datetime"].iloc[-1]) != origin:
        base["status"] = "no_data_at_origin"
        return [{**base, "horizon": h, "target_date": origin + pd.Timedelta(weeks=h)} for h in range(1, cfg.horizons + 1)]

    sup = build_supervised(weekly, config=fcfg.model_copy(update={"horizons": cfg.horizons}))
    data = sup.data
    valid = sup.valid_mask.to_numpy(bool)
    origin_row = data.index[pd.to_datetime(data[fcfg.datetime_col]) == origin]
    feature_ready = bool(len(origin_row)) and bool(data.loc[origin_row[0], list(sup.feature_cols)].notna().all())

    series = weekly.set_index(pd.to_datetime(weekly["datetime"]))["inc_per_10k"].astype(float)
    current = float(series.iloc[-1])
    imputed_recent = int(weekly["pcr_imputed"].tail(4).sum()) if "pcr_imputed" in weekly else 0
    preds: dict[str, np.ndarray] = {}
    if feature_ready and valid.sum() >= cfg.min_train_rows:
        X = data.loc[valid, list(sup.feature_cols)].to_numpy(float)
        x0 = data.loc[[origin_row[0]], list(sup.feature_cols)].to_numpy(float)
        ys = [data.loc[valid, c].to_numpy(float) for c in sup.target_cols]
        for name, point_cfg in cfg.model_variants.items():
            models = fit_models_hist_gbdt(X, ys, config=point_cfg, random_state=cfg.random_state)
            point = np.clip(np.array([m.predict(x0)[0] for m in models], dtype=float), 0.0, None)
            targets = [origin + pd.Timedelta(weeks=h) for h in range(1, cfg.horizons + 1)]
            preds[name] = apply_holiday_ratio(point, targets, sup.holiday_ratio, fcfg.holiday_iso_weeks)
    else:
        base["status"] = "insufficient_training_data"

    br: JsonObject = {"br_status": None}
    if cfg.run_mechanistic and cfg.primary_variant in preds:
        ml = preds[cfg.primary_variant]
        future = [(origin + pd.Timedelta(weeks=h)).date().isoformat() for h in range(1, cfg.horizons + 1)]
        mw_cfg = cfg.mechanistic.model_copy(update={"forecast_horizon_weeks": cfg.horizons})
        try:
            res = run_mechanistic_wave_analysis(
                weekly,
                origin_date=origin,
                ml_forecast={"point": list(ml), "lo": list(ml), "hi": list(ml), "dates": future},
                config=mw_cfg,
            )
            br["br_status"] = res.status
            if res.context_block is not None:
                block = res.context_block
                br.update(
                    {
                        "phase": block["phase_code"],
                        "agreement": block["comparison_with_ml"]["agreement_code"],
                        "disagreement_reasons": ",".join(block["comparison_with_ml"]["disagreement_reasons"]),
                        "br_fit_r2": block["fit_r2_in_sample"],
                        "br_rhat": block["diagnostics"]["r_hat_max"],
                        "br_values": [h["br_value"] for h in block["horizons"]],
                        "br_lows": [h["br_low"] for h in block["horizons"]],
                        "br_highs": [h["br_high"] for h in block["horizons"]],
                        "br_r_t": block.get("reproduction_number", {}).get("current"),
                    }
                )
        except Exception as exc:  # калибровка не должна останавливать бэктест
            br["br_status"] = f"error: {exc}"

    iso_weeks = pd.Series(series.index.isocalendar().week.to_numpy(int), index=series.index)
    stat: dict[str, np.ndarray] = {}
    if cfg.statistical_baselines and base["status"] == "ok" and len(series.dropna()) >= 104:
        stat = statistical_baselines(series.dropna(), cfg.horizons)
    rows = []
    for h in range(1, cfg.horizons + 1):
        target = origin + pd.Timedelta(weeks=h)
        row: JsonObject = {
            **base,
            "horizon": h,
            "target_date": target,
            "y_origin": current,
            "pcr_imputed_last4": imputed_recent,
            "persistence": current,
            "climatology": _climatology(series, iso_weeks, origin, target, cfg.min_climatology_seasons),
            "br_status": br["br_status"],
            "phase": br.get("phase") or (br["br_status"] if br["br_status"] in {"no_wave", "insufficient_data"} else None),
            "agreement": br.get("agreement"),
            "disagreement_reasons": br.get("disagreement_reasons"),
            "br_fit_r2": br.get("br_fit_r2"),
            "br_rhat": br.get("br_rhat"),
            "br": br["br_values"][h - 1] if "br_values" in br else np.nan,
            "br_lo": _num(br["br_lows"][h - 1]) if "br_lows" in br else np.nan,
            "br_hi": _num(br["br_highs"][h - 1]) if "br_highs" in br else np.nan,
            "br_r_t": _num(br.get("br_r_t")),
        }
        for name, values in preds.items():
            row[f"ml_{name}"] = float(values[h - 1])
        for name in BASELINE_NAMES:
            row[name] = float(stat[name][h - 1]) if name in stat else np.nan
        rows.append(row)
    return rows


def _climatology(series: pd.Series, iso_weeks: pd.Series, origin: pd.Timestamp, target: pd.Timestamp, min_seasons: int) -> float:
    week = int(target.isocalendar().week)
    mask = (iso_weeks.to_numpy() == week) & (series.index <= origin) & (series.index < target - pd.Timedelta(weeks=26))
    values = series.to_numpy(float)[mask]
    values = values[np.isfinite(values)]
    return float(values.mean()) if len(values) >= min_seasons else float("nan")


def _add_conformal_intervals(pred: pd.DataFrame, cfg: BacktestConfig) -> pd.DataFrame:
    """Интервал недели t по ошибкам прошлых недель, цели которых известны к t."""

    col = f"ml_{cfg.primary_variant}"
    for name in ("ml_lo", "ml_hi", "ml_lo_level", "ml_hi_level"):
        pred[name] = np.nan
    if col not in pred:
        return pred
    c = cfg.interval_level_offset
    for _h, part in pred.groupby("horizon"):
        part = part.sort_values("origin")
        errors = (part["y_true"] - part[col]).abs()
        scale = part["y_origin"].abs() + c
        rel_errors = errors / scale
        for idx, row in part.iterrows():
            known = part[(part["target_date"] <= row["origin"]) & errors.notna()]
            errs = errors.loc[known.index].tail(cfg.conformal_window).to_numpy(float)
            if len(errs) < cfg.min_conformal_errors or not np.isfinite(row[col]):
                continue
            radius = conformal_radius_from_abs_errors(errs, alpha=cfg.interval_alpha)
            pred.loc[idx, "ml_lo"] = max(0.0, row[col] - radius)
            pred.loc[idx, "ml_hi"] = row[col] + radius
            rel = rel_errors.loc[known.index].tail(cfg.conformal_window).to_numpy(float)
            radius_level = conformal_radius_from_abs_errors(rel, alpha=cfg.interval_alpha) * (abs(row["y_origin"]) + c)
            pred.loc[idx, "ml_lo_level"] = max(0.0, row[col] - radius_level)
            pred.loc[idx, "ml_hi_level"] = row[col] + radius_level
    return pred


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def _metrics(y: np.ndarray, p: np.ndarray) -> JsonObject:
    m = np.isfinite(y) & np.isfinite(p)
    if m.sum() < 2:
        return {"n": int(m.sum()), "r2": None, "rmse": None, "mae": None}
    yt, yp = y[m], p[m]
    ss_tot = float(np.sum((yt - yt.mean()) ** 2))
    return {
        "n": int(m.sum()),
        "r2": round(1.0 - float(np.sum((yt - yp) ** 2)) / ss_tot, 3) if ss_tot > 0 else None,
        "rmse": round(float(np.sqrt(np.mean((yt - yp) ** 2))), 3),
        "mae": round(float(np.mean(np.abs(yt - yp))), 3),
    }


def _num(value: Any) -> float:
    return float(value) if value is not None else np.nan


def evaluate_backtest(pred: pd.DataFrame, cfg: BacktestConfig, *, label: str = "") -> JsonObject:
    model_cols = [c for c in pred.columns if c.startswith("ml_") and not c.startswith(("ml_lo", "ml_hi"))]
    primary = f"ml_{cfg.primary_variant}"
    compare_cols = [*model_cols, "persistence", "climatology", *[c for c in BASELINE_NAMES if c in pred and pred[c].notna().any()]]
    scored = pred[pred[primary].notna() & pred["y_true"].notna()] if primary in pred else pred.iloc[0:0]
    y = scored["y_true"].to_numpy(float)

    overall = {c: _metrics(y, scored[c].to_numpy(float)) for c in compare_cols}
    common = scored[scored[compare_cols].notna().all(axis=1)]
    overall_common = {c: _metrics(common["y_true"].to_numpy(float), common[c].to_numpy(float)) for c in compare_cols}
    by_horizon = {
        int(h): {c: _metrics(part["y_true"].to_numpy(float), part[c].to_numpy(float)) for c in compare_cols}
        for h, part in scored.groupby("horizon")
    }

    intervals: JsonObject = {"nominal_coverage": round(1 - cfg.interval_alpha, 3)}
    wave_phases = {"growth", "peak", "decline"}
    for kind, lo_col, hi_col in (("absolute", "ml_lo", "ml_hi"), ("level_scaled", "ml_lo_level", "ml_hi_level")):
        iv = scored[scored[lo_col].notna()]
        if not len(iv):
            continue
        inside = (iv["y_true"] >= iv[lo_col]) & (iv["y_true"] <= iv[hi_col])
        width = iv[hi_col] - iv[lo_col]
        in_wave = iv["phase"].isin(wave_phases)
        intervals[kind] = {
            "n": len(iv),
            "coverage": round(float(inside.mean()), 3),
            "mean_width": round(float(width.mean()), 3),
            "coverage_wave": round(float(inside[in_wave].mean()), 3) if in_wave.any() else None,
            "coverage_off_wave": round(float(inside[~in_wave].mean()), 3) if (~in_wave).any() else None,
            "width_wave": round(float(width[in_wave].mean()), 3) if in_wave.any() else None,
            "width_off_wave": round(float(width[~in_wave].mean()), 3) if (~in_wave).any() else None,
            "by_horizon": {int(h): round(float(g.mean()), 3) for h, g in inside.groupby(iv["horizon"])},
        }
    # полоса модели БР (q05–q95 пучка): покрытие на точках, где БР откалибрована
    if "br_lo" in scored and scored["br_lo"].notna().any():
        iv = scored[scored["br_lo"].notna() & scored["br"].notna()]
        inside = (iv["y_true"] >= iv["br_lo"]) & (iv["y_true"] <= iv["br_hi"])
        width = iv["br_hi"] - iv["br_lo"]
        in_wave = iv["phase"].isin(wave_phases)
        intervals["br_band"] = {
            "n": len(iv),
            "nominal_coverage": 0.9,
            "coverage": round(float(inside.mean()), 3),
            "mean_width": round(float(width.mean()), 3),
            "coverage_wave": round(float(inside[in_wave].mean()), 3) if in_wave.any() else None,
            "by_horizon": {int(h): round(float(g.mean()), 3) for h, g in inside.groupby(iv["horizon"])},
        }
    # обратная совместимость: верхний уровень — абсолютные интервалы
    if "absolute" in intervals:
        intervals.update({k: intervals["absolute"][k] for k in ("n", "coverage", "mean_width", "by_horizon")})

    by_phase: JsonObject = {}
    for phase, part in scored.groupby(scored["phase"].fillna("not_run")):
        by_phase[str(phase)] = {
            c: _metrics(part["y_true"].to_numpy(float), part[c].to_numpy(float))
            for c in [primary, "persistence", "br", *[b for b in BASELINE_NAMES if b in compare_cols]]
        }
        by_phase[str(phase)]["n_origins"] = int(part["origin"].nunique())

    agreement: JsonObject = {}
    has_agreement = scored[scored["agreement"].notna()]
    for code, part in has_agreement.groupby("agreement"):
        yt = part["y_true"].to_numpy(float)
        ml = part[primary].to_numpy(float)
        brv = part["br"].to_numpy(float)
        rel = np.abs(yt - ml) / np.maximum(np.abs(yt), 1.0)
        closer_ml = np.abs(yt - ml) < np.abs(yt - brv)
        agreement[str(code)] = {
            "n_origins": int(part["origin"].nunique()),
            "ml": _metrics(yt, ml),
            "br": _metrics(yt, brv),
            "ml_mean_relative_error": round(float(np.nanmean(rel)), 3),
            "share_ml_closer_than_br": round(float(np.mean(closer_ml)), 3),
        }
        for kind, lo_col, hi_col in (("", "ml_lo", "ml_hi"), ("_level", "ml_lo_level", "ml_hi_level")):
            if lo_col in part and part[lo_col].notna().any():
                q = part[part[lo_col].notna()]
                agreement[str(code)][f"ml_interval_coverage{kind}"] = round(
                    float(((q["y_true"] >= q[lo_col]) & (q["y_true"] <= q[hi_col])).mean()), 3
                )

    status_counts = pred.groupby("status")["origin"].nunique().to_dict()
    br_counts = pred.groupby(pred["br_status"].fillna("not_run"))["origin"].nunique().to_dict()
    return {
        "label": label,
        "period": {"start": str(cfg.start), "end": str(cfg.end), "n_origins": int(pred["origin"].nunique())},
        "primary_variant": cfg.primary_variant,
        "origin_status": {str(k): int(v) for k, v in status_counts.items()},
        "mechanistic_status": {str(k): int(v) for k, v in br_counts.items()},
        "overall": overall,
        "overall_common_points": overall_common,
        "by_horizon": by_horizon,
        "intervals": intervals,
        "by_phase": by_phase,
        "agreement": agreement,
    }


def render_backtest_report(metrics: JsonObject) -> str:
    """Краткий отчёт в Markdown (русский)."""

    def fmt(v: Any) -> str:
        return "—" if v is None else f"{v}"

    lines = [f"# Бэктест {metrics.get('label', '')}".rstrip(), ""]
    p = metrics["period"]
    lines += [f"Период: {p['start']} – {p['end']}, недель: {p['n_origins']}.", ""]
    lines += ["## Все горизонты (общие точки)", "", "| Модель | n | R² | RMSE | MAE |", "|---|---|---|---|---|"]
    for name, m in metrics["overall_common_points"].items():
        lines.append(f"| {name} | {m['n']} | {fmt(m['r2'])} | {fmt(m['rmse'])} | {fmt(m['mae'])} |")
    lines += ["", "## По горизонтам (R²)", ""]
    names = list(next(iter(metrics["by_horizon"].values()), {}).keys())
    lines += ["| h | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
    for h, row in metrics["by_horizon"].items():
        lines.append(f"| {h} | " + " | ".join(fmt(row[n]["r2"]) for n in names) + " |")
    iv = metrics["intervals"]
    lines += ["", f"## Прогнозные интервалы (номинальное покрытие {iv['nominal_coverage']})", "",
              "| Вид | n | покрытие | на волне | вне волны | ширина на волне | ширина вне волны |", "|---|---|---|---|---|---|---|"]
    for kind, label_ru in (("absolute", "абсолютные"), ("level_scaled", "по уровню")):
        k = iv.get(kind)
        if k:
            lines.append(
                f"| {label_ru} | {k['n']} | {k['coverage']} | {fmt(k['coverage_wave'])} | {fmt(k['coverage_off_wave'])} | "
                f"{fmt(k['width_wave'])} | {fmt(k['width_off_wave'])} |"
            )
    if iv.get("br_band"):
        k = iv["br_band"]
        lines += ["", f"Полоса модели БР (q05–q95, номинально 0.9): n = {k['n']}, покрытие {k['coverage']} "
                  f"(на волне {fmt(k['coverage_wave'])}), средняя ширина {k['mean_width']}; по горизонтам: "
                  + ", ".join(f"h{h}: {v}" for h, v in k["by_horizon"].items()) + "."]
    lines.append("")
    lines += ["## По фазам волны (MAE)", "", "| Фаза | недель | ML | наивный | БР |", "|---|---|---|---|---|"]
    primary = f"ml_{metrics['primary_variant']}"
    for phase, row in metrics["by_phase"].items():
        lines.append(
            f"| {phase} | {row['n_origins']} | {fmt(row[primary]['mae'])} | {fmt(row['persistence']['mae'])} | {fmt(row['br']['mae'])} |"
        )
    lines += ["", "## Согласие моделей (недели волны)", "",
              "| Код | недель | MAE ML | MAE БР | отн. ошибка ML | ML ближе | покрытие ML (абс.) | покрытие ML (по уровню) |",
              "|---|---|---|---|---|---|---|---|"]
    for code, row in metrics["agreement"].items():
        lines.append(
            f"| {code} | {row['n_origins']} | {fmt(row['ml']['mae'])} | {fmt(row['br']['mae'])} | "
            f"{fmt(row['ml_mean_relative_error'])} | {fmt(row['share_ml_closer_than_br'])} | "
            f"{fmt(row.get('ml_interval_coverage'))} | {fmt(row.get('ml_interval_coverage_level'))} |"
        )
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Local data helper
# ---------------------------------------------------------------------------


class LocalDataAt:
    """``data_at`` для локальных xlsx: ряд «как известно на неделю t» (picklable)."""

    def __init__(self, city: str, data_dir: str | Path, pcr_gap_policy: Literal["interpolate", "zero"] = "interpolate") -> None:
        self.city = city
        self.data_dir = str(data_dir)
        self.pcr_gap_policy = pcr_gap_policy

    def __call__(self, origin: pd.Timestamp) -> pd.DataFrame:
        from ai4epi.data.local_excel import load_local_influenza

        return load_local_influenza(
            self.city, data_dir=self.data_dir, end_date=origin, pcr_gap_policy=self.pcr_gap_policy
        ).weekly


__all__ = [
    "BacktestConfig",
    "BacktestResult",
    "LocalDataAt",
    "evaluate_backtest",
    "render_backtest_report",
    "run_backtest",
]
