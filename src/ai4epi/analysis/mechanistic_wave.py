"""
Механистическое описание текущей эпидемической волны (модель Барояна–Рвачёва).

Этап реализует шаги 3–6 схемы комплементарного применения моделей из статьи
Лазутова и др. («Комплементарное применение моделей машинного обучения и
механистической модели для краткосрочного прогнозирования заболеваемости
гриппом»):

3. проверка фазы: идёт ли в текущем сезоне эпидемическая волна;
4. калибровка модели Барояна–Рвачёва по данным текущей волны → семейство
   допустимых траекторий, фаза волны, срок и величина максимума, оценка
   завершения основной волны, эффективные параметры alpha и beta;
5. сопоставление направления и масштаба прогнозов модели БР и ML-модели
   (HGB) на горизонте 1–4 недели;
6. код согласия, который narrator использует для итогового вывода: при
   согласии значения берутся из ML, фаза/пик — из модели БР; при расхождении —
   сообщение о повышенной неопределённости без усреднения.

Результат — блок ``mechanistic_wave`` для ``GlobalContext`` (передаётся через
``extra_context_blocks``) и артефакты (траектории, апостериорная выборка).
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator

from ai4epi.core.io import TableWriteOptions, write_json, write_table
from ai4epi.mechanistic import (
    CalibrationResult,
    ModelParams,
    calibrate,
    get_model,
    model_observable,
    r2_score_np,
)

JsonObject = dict[str, Any]
WaveStageStatus = Literal["ok", "no_wave", "insufficient_data"]
PhaseCode = Literal["growth", "peak", "decline", "post_wave"]
DirectionCode = Literal["increase", "stable", "decrease"]

PHASE_LABELS_RU: dict[str, str] = {
    "growth": "фаза роста",
    "peak": "область максимума",
    "decline": "фаза спада",
    "post_wave": "основная волна завершилась",
}
DIRECTION_LABELS_RU: dict[str, str] = {
    "increase": "рост",
    "stable": "стабилизация",
    "decrease": "снижение",
}
AGREEMENT_LABELS_RU: dict[str, str] = {
    "agree": "прогнозы согласуются",
    "disagree": "прогнозы расходятся",
}

MODEL_DESCRIPTION_RU = (
    "Дискретная модель Барояна–Рвачёва с суточным шагом: число новых заражений "
    "определяется произведением числа восприимчивых и числа инфицированных, "
    "заразность которых меняется по дням болезни. Параметры alpha (доля "
    "восприимчивых в эффективной популяции) и beta (интенсивность передачи) "
    "оцениваются по данным текущей волны; эффективная популяция учитывает "
    "недоучёт заболеваемости."
)

ALLOWED_CLAIMS: list[str] = [
    "фаза текущей эпидемической волны по модели Барояна–Рвачёва",
    "ожидаемые неделя и величина максимума по центральной траектории и их диапазон по пучку траекторий",
    "оценка недели завершения основной волны, если она лежит в пределах горизонта моделирования",
    "согласие или расхождение направления прогнозов механистической модели и модели машинного обучения",
    "качество описания наблюдаемой части волны (R² на калибровочном участке)",
]
FORBIDDEN_CLAIMS: list[str] = [
    "усреднять прогнозы двух моделей или выдавать значения модели Барояна–Рвачёва за основной количественный прогноз",
    "утверждать точную дату окончания эпидемии или эпидемического периода",
    "интерпретировать alpha как долю восприимчивых во всём населении",
    "делать причинные выводы о смене штамма, вакцинации или мерах контроля",
    "называть R² на калибровочном участке точностью прогноза",
]


class MechanisticWaveError(RuntimeError):
    """Ошибка механистического этапа."""


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True, arbitrary_types_allowed=True)


class MechanisticWaveConfig(StrictModel):
    """Настройки этапа. Значения по умолчанию соответствуют статье.

    ``observation_scale="per_10k"`` — расхождение модели и данных в ядре ABC
    считается в случаях на 10 тыс. населения (формула (abc) статьи, ``epsilon=1``);
    ``"cases"`` воспроизводит исходный код Influenza_bulletin (абсолютные числа).

    Начало калибровочного интервала (модель стартует с ``initial_infectious``
    заражённых в эту неделю): явная ``calibration_start_date``; иначе при
    ``calibration_start="wave_onset"`` — обнаруженное начало роста минус
    ``onset_lead_weeks``; при ``"season_start"`` — ISO-неделя
    ``season_start_week`` (так считались рисунки статьи для РФ, 35-я неделя).
    """

    method: Literal["mcmc", "abc", "optuna", "annealing"] = "mcmc"
    season_start_week: int = Field(default=35, ge=1, le=53)
    calibration_start: Literal["wave_onset", "season_start"] = "wave_onset"
    onset_lead_weeks: int = Field(default=2, ge=0, le=12)
    calibration_start_date: date | None = None
    wave_threshold_per_10k: float = Field(default=1.0, ge=0.0)
    wave_min_growth_weeks: int = Field(default=3, ge=1)
    min_calibration_weeks: int = Field(default=5, ge=3)
    underreporting_kappa: float = Field(default=0.1, gt=0.0, le=1.0)
    initial_infectious: float = Field(default=100.0, gt=0.0)
    observation_scale: Literal["per_10k", "cases"] = "per_10k"
    epsilon: float = Field(default=1.0, gt=0.0)
    n_trajectories: int = Field(default=200, ge=1)
    tune: int = Field(default=2500, ge=0)
    draws: int = Field(default=500, ge=10)
    chains: int = Field(default=4, ge=1)
    cores: int | None = Field(default=None, ge=1)
    n_trials: int = Field(default=1000, ge=10)
    random_seed: int | None = 42
    forecast_horizon_weeks: int = Field(default=4, ge=1, le=12)
    projection_weeks: int = Field(default=30, ge=4, le=60)
    wave_end_fraction: float = Field(default=0.1, gt=0.0, lt=1.0)
    peak_tolerance_weeks: int = Field(default=1, ge=0)
    direction_change_pct_threshold: float = Field(default=10.0, gt=0.0)
    agreement_relative_gap_pct: float = Field(default=50.0, gt=0.0)
    rhat_threshold: float = Field(default=1.05, gt=1.0)
    round_digits: int = Field(default=3, ge=0, le=8)


class MechanisticWaveOutputConfig(StrictModel):
    output_dir: Path = Path("mechanistic_wave")
    trajectories_filename: str = "trajectories.csv"
    samples_filename: str = "posterior_samples.csv"
    summary_filename: str = "mechanistic_wave.json"


class MechanisticWaveRunResult(StrictModel):
    status: WaveStageStatus
    message: str | None = None
    context_block: JsonObject | None = None
    trajectories: Any | None = None
    calibration: Any | None = None
    artifacts: dict[str, Path] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)

    @field_validator("trajectories")
    @classmethod
    def _df(cls, value: Any) -> Any:
        if value is not None and not isinstance(value, pd.DataFrame):
            raise TypeError("trajectories должен быть pandas.DataFrame.")
        return value

    def to_public_dict(self) -> JsonObject:
        return {
            "status": self.status,
            "message": self.message,
            "context_block": self.context_block,
            "artifacts": {k: str(v) for k, v in self.artifacts.items()},
            "warnings": list(self.warnings),
        }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_mechanistic_wave_analysis(
    influenza_weekly: pd.DataFrame,
    *,
    origin_date: pd.Timestamp | date | str | None = None,
    ml_forecast: JsonObject | None = None,
    config: MechanisticWaveConfig | None = None,
    output: MechanisticWaveOutputConfig | None = None,
) -> MechanisticWaveRunResult:
    """Выполнить механистический этап.

    :param influenza_weekly: таблица с ``datetime``, ``total_cases_formula``,
        ``total_population``, ``inc_per_10k``
    :param origin_date: последняя неделя, доступная моделям (по умолчанию —
        последняя неделя таблицы); совпадает с ``origin`` ML-прогноза
    :param ml_forecast: ``{"point": [...], "lo": [...], "hi": [...], "dates": [...]}``
        для сопоставления с ML-прогнозом; без него блок сопоставления пуст
    """

    cfg = config or MechanisticWaveConfig()
    frame = _prepare_frame(influenza_weekly, origin_date)
    origin = pd.Timestamp(frame["datetime"].iloc[-1])

    season_year = origin.isocalendar().year if origin.isocalendar().week >= cfg.season_start_week else origin.isocalendar().year - 1
    season_start = pd.Timestamp(_season_start(season_year, cfg.season_start_week))
    season_label = f"{season_year}–{season_year + 1}"
    season = frame[frame["datetime"] >= season_start].reset_index(drop=True)

    wave_detected, onset_date = detect_wave(season["inc_per_10k"].to_numpy(float), season["datetime"], cfg)
    if not wave_detected:
        return MechanisticWaveRunResult(
            status="no_wave",
            message=(
                f"В сезоне {season_label} эпидемический подъём не выявлен (порог {cfg.wave_threshold_per_10k} на 10 тыс., "
                f"{cfg.wave_min_growth_weeks} нед. роста подряд); вывод формируется только по ML-прогнозу."
            ),
        )

    if cfg.calibration_start_date is not None:
        calib_start = pd.Timestamp(cfg.calibration_start_date)
    elif cfg.calibration_start == "wave_onset" and onset_date is not None:
        calib_start = max(season_start, onset_date - pd.Timedelta(weeks=cfg.onset_lead_weeks))
    else:
        calib_start = season_start
    window = frame[frame["datetime"] >= calib_start].reset_index(drop=True)

    if len(window) < cfg.min_calibration_weeks:
        return MechanisticWaveRunResult(
            status="insufficient_data",
            message=f"С начала калибровочного интервала ({calib_start.date()}) доступно {len(window)} нед. "
            f"(нужно не менее {cfg.min_calibration_weeks}).",
        )
    gaps = window["datetime"].diff().dropna().dt.days
    if not (gaps == 7).all():
        return MechanisticWaveRunResult(
            status="insufficient_data",
            message="В калибровочном интервале есть пропущенные недели; модель Барояна–Рвачёва требует регулярного ряда.",
        )

    imputed_weeks = int(window["pcr_imputed"].sum()) if "pcr_imputed" in window else 0

    population = float(window["total_population"].iloc[0])
    per_10k = 10_000.0 / population
    observed_cases = window["total_cases_formula"].to_numpy(float)
    model = get_model("total")
    template = ModelParams(
        alpha=[0.0],
        beta=[0.0],
        population_size=int(population * cfg.underreporting_kappa),
        initial_infectious=[cfg.initial_infectious],
    )
    calibration = calibrate(
        model,
        observed_cases,
        template,
        method=cfg.method,
        observation_scale=per_10k if cfg.observation_scale == "per_10k" else 1.0,
        epsilon=cfg.epsilon,
        n_samples=cfg.n_trajectories,
        tune=cfg.tune,
        draws=cfg.draws,
        chains=cfg.chains,
        cores=cfg.cores,
        n_trials=cfg.n_trials,
        random_seed=cfg.random_seed,
    )

    trajectories, bundle = build_trajectories(
        model, calibration, window, per_10k=per_10k, projection_weeks=cfg.projection_weeks
    )
    block, warnings = build_mechanistic_wave_context(
        trajectories,
        calibration,
        bundle=bundle,
        window=window,
        season_label=season_label,
        calibration_start=calib_start,
        onset_date=onset_date,
        ml_forecast=ml_forecast,
        config=cfg,
    )
    if imputed_weeks:
        warnings.append(
            f"В калибровочном интервале {imputed_weeks} нед. без данных ПЦР; доля положительных проб на них интерполирована."
        )
    block["pcr_imputed_weeks"] = imputed_weeks
    result = MechanisticWaveRunResult(
        status="ok",
        context_block=block,
        trajectories=trajectories,
        calibration=calibration,
        warnings=warnings,
    )
    if output is not None:
        result.artifacts = save_mechanistic_wave_artifacts(result, output)
    return result


def ml_forecast_from_result(forecast_result: Any) -> JsonObject:
    """Достать точечный прогноз и интервалы из ``ForecastRunResult``."""

    return {
        "point": [float(v) for v in np.asarray(forecast_result.y_hat, dtype=float)],
        "lo": [float(v) for v in np.asarray(forecast_result.y_lo, dtype=float)],
        "hi": [float(v) for v in np.asarray(forecast_result.y_hi, dtype=float)],
        "dates": [pd.Timestamp(d).date().isoformat() for d in pd.Series(forecast_result.future_dates)],
    }


# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------


def detect_wave(values: np.ndarray, dates: pd.Series, cfg: MechanisticWaveConfig) -> tuple[bool, pd.Timestamp | None]:
    """Шаг 3 схемы: подъём есть, если показатель превысил порог и рос
    ``wave_min_growth_weeks`` недель подряд. Возвращает (флаг, начало роста)."""

    if len(values) == 0 or float(np.nanmax(values)) < cfg.wave_threshold_per_10k:
        return False, None
    run = 0
    for i in range(1, len(values)):
        run = run + 1 if values[i] > values[i - 1] else 0
        if run >= cfg.wave_min_growth_weeks:
            return True, pd.Timestamp(dates.iloc[i - run])
    return False, None


def build_trajectories(
    model: Any,
    calibration: CalibrationResult,
    window: pd.DataFrame,
    *,
    per_10k: float,
    projection_weeks: int,
) -> tuple[pd.DataFrame, np.ndarray | None]:
    """Центральная траектория и квантили пучка на 10 тыс. населения.

    Возвращает таблицу и матрицу пучка ``(n_trajectories, n_weeks)`` (или None
    для точечных методов калибровки).
    """

    n_obs = len(window)
    n_total = n_obs + projection_weeks
    best = model_observable(model, calibration.best_params, n_total) * per_10k
    dates = pd.date_range(window["datetime"].iloc[0], periods=n_total, freq="7D")
    df = pd.DataFrame(
        {
            "datetime": dates,
            "iso_week": [int(d.isocalendar().week) for d in dates],
            "is_projection": np.arange(n_total) >= n_obs,
            "observed": np.concatenate([window["inc_per_10k"].to_numpy(float), np.full(projection_weeks, np.nan)]),
            "best": best,
        }
    )
    bundle: np.ndarray | None = None
    if calibration.has_bundle:
        bundle = np.vstack([model_observable(model, p, n_total) * per_10k for p in calibration.sample_params()])
        for q in (5, 50, 95):
            df[f"q{q:02d}"] = np.percentile(bundle, q, axis=0)
    return df, bundle


def build_mechanistic_wave_context(
    trajectories: pd.DataFrame,
    calibration: CalibrationResult,
    *,
    bundle: np.ndarray | None,
    window: pd.DataFrame,
    season_label: str,
    calibration_start: pd.Timestamp,
    onset_date: pd.Timestamp | None,
    ml_forecast: JsonObject | None,
    config: MechanisticWaveConfig,
) -> tuple[JsonObject, list[str]]:
    """Собрать блок ``mechanistic_wave`` для GlobalContext."""

    cfg = config
    rd = cfg.round_digits
    warnings: list[str] = []
    n_obs = int((~trajectories["is_projection"]).sum())
    last_idx = n_obs - 1
    best = trajectories["best"].to_numpy(float)
    dates = pd.to_datetime(trajectories["datetime"])
    observed = window["inc_per_10k"].to_numpy(float)

    fit_r2 = r2_score_np(observed, best[:n_obs])

    peak_idx = int(np.argmax(best))
    end_idx = _wave_end_index(best, peak_idx, cfg.wave_end_fraction)
    phase = _phase(last_idx, peak_idx, end_idx, cfg.peak_tolerance_weeks)

    peak_block: JsonObject = {
        "date": dates.iloc[peak_idx].date().isoformat(),
        "iso_week": int(dates.iloc[peak_idx].isocalendar().week),
        "value": round(float(best[peak_idx]), rd),
        "weeks_from_current": int(peak_idx - last_idx),
        "already_passed": bool(peak_idx < last_idx - cfg.peak_tolerance_weeks),
        "range": None,
    }
    if bundle is not None:
        pk_idx = np.argmax(bundle, axis=1)
        pk_val = bundle[np.arange(len(bundle)), pk_idx]
        lo_i, hi_i = np.percentile(pk_idx, [5, 95]).round().astype(int)
        peak_block["range"] = {
            "date_low": dates.iloc[int(lo_i)].date().isoformat(),
            "date_high": dates.iloc[int(hi_i)].date().isoformat(),
            "iso_week_low": int(dates.iloc[int(lo_i)].isocalendar().week),
            "iso_week_high": int(dates.iloc[int(hi_i)].isocalendar().week),
            "value_low": round(float(np.percentile(pk_val, 5)), rd),
            "value_high": round(float(np.percentile(pk_val, 95)), rd),
            "coverage": "5–95% пучка траекторий",
        }

    wave_end: JsonObject = {
        "definition": f"первая неделя после максимума, на которой центральная траектория ниже {int(cfg.wave_end_fraction * 100)}% от максимума",
        "within_projection": end_idx is not None,
        "date": dates.iloc[end_idx].date().isoformat() if end_idx is not None else None,
        "iso_week": int(dates.iloc[end_idx].isocalendar().week) if end_idx is not None else None,
        "weeks_from_current": int(end_idx - last_idx) if end_idx is not None else None,
    }

    current_value = float(observed[-1])
    horizons: list[JsonObject] = []
    h = cfg.forecast_horizon_weeks
    for k in range(1, h + 1):
        idx = last_idx + k
        row: JsonObject = {
            "horizon_weeks": k,
            "target_date": dates.iloc[idx].date().isoformat(),
            "target_iso_week": int(dates.iloc[idx].isocalendar().week),
            "br_value": round(float(best[idx]), rd),
            "br_low": round(float(trajectories["q05"].iloc[idx]), rd) if "q05" in trajectories else None,
            "br_high": round(float(trajectories["q95"].iloc[idx]), rd) if "q95" in trajectories else None,
            "ml_value": None,
            "ml_low": None,
            "ml_high": None,
        }
        if ml_forecast and k <= len(ml_forecast.get("point", [])):
            row["ml_value"] = round(float(ml_forecast["point"][k - 1]), rd)
            row["ml_low"] = round(float(min(ml_forecast["lo"][k - 1], ml_forecast["hi"][k - 1])), rd)
            row["ml_high"] = round(float(max(ml_forecast["lo"][k - 1], ml_forecast["hi"][k - 1])), rd)
            ml_dates = list(ml_forecast.get("dates") or [])
            ml_date = ml_dates[k - 1] if k <= len(ml_dates) else None
            if ml_date and ml_date != row["target_date"]:
                warnings.append(f"Даты горизонта h={k} у ML ({ml_date}) и модели БР ({row['target_date']}) не совпадают.")
        horizons.append(row)

    br_h = float(best[last_idx + h])
    br_dir = _direction(current_value, br_h, cfg.direction_change_pct_threshold)
    comparison: JsonObject = {
        "current_value": round(current_value, rd),
        "br_direction_code": br_dir,
        "br_direction": DIRECTION_LABELS_RU[br_dir],
        "br_change_pct": round(_pct(current_value, br_h), 1),
        "ml_direction_code": None,
        "ml_direction": None,
        "ml_change_pct": None,
        "mean_relative_gap_pct": None,
        "agreement_code": None,
        "agreement": None,
        "disagreement_reasons": [],
    }
    if all(r["ml_value"] is not None for r in horizons):
        ml_h = float(horizons[-1]["ml_value"])
        ml_dir = _direction(current_value, ml_h, cfg.direction_change_pct_threshold)
        gaps = [
            abs(r["br_value"] - r["ml_value"]) / max(abs(r["ml_value"]), 1e-9) * 100.0 for r in horizons
        ]
        gap = float(np.mean(gaps))
        reasons = []
        if ml_dir != br_dir:
            reasons.append("direction")
        if gap > cfg.agreement_relative_gap_pct:
            reasons.append("scale")
        code = "agree" if not reasons else "disagree"
        comparison.update(
            {
                "ml_direction_code": ml_dir,
                "ml_direction": DIRECTION_LABELS_RU[ml_dir],
                "ml_change_pct": round(_pct(current_value, ml_h), 1),
                "mean_relative_gap_pct": round(gap, 1),
                "agreement_code": code,
                "agreement": AGREEMENT_LABELS_RU[code],
                "disagreement_reasons": reasons,
            }
        )
    else:
        warnings.append("ML-прогноз не передан; сопоставление моделей не выполнено.")

    diag = calibration.diagnostics
    rhat = diag.get("r_hat_max")
    converged = None if rhat is None else bool(rhat <= cfg.rhat_threshold)
    if converged is False:
        warnings.append(f"MCMC: R-hat = {rhat:.2f} > {cfg.rhat_threshold}; апостериорная выборка может быть ненадёжной.")
    a_s, b_s = calibration.alpha_samples[:, 0], calibration.beta_samples[:, 0]
    alpha_best, beta_best = float(calibration.best_params.alpha[0]), float(calibration.best_params.beta[0])
    parameters: JsonObject = {
        "alpha": _param_summary(alpha_best, a_s, rd),
        "beta": _param_summary(beta_best, b_s, rd),
        "alpha_beta_product": _param_summary(alpha_best * beta_best, a_s * b_s if len(a_s) else a_s, rd),
        "alpha_at_upper_bound": bool(alpha_best > 0.98),
        "underreporting_kappa": cfg.underreporting_kappa,
        "effective_population": int(calibration.template.population_size),
        "interpretation_note": (
            "alpha — доля восприимчивых в эффективной популяции kappa·N; доля восприимчивых "
            "относительно всего населения равна kappa·alpha."
        ),
    }
    if parameters["alpha_at_upper_bound"]:
        warnings.append("Оценка alpha близка к верхней границе априорного распределения (1).")

    early_caution = phase == "growth"
    block: JsonObject = {
        "model_name_ru": "модель Барояна–Рвачёва",
        "model_description_ru": MODEL_DESCRIPTION_RU,
        "season_label": season_label,
        "calibration_start_date": calibration_start.date().isoformat(),
        "calibration_end_date": dates.iloc[last_idx].date().isoformat(),
        "calibration_weeks": n_obs,
        "wave_onset_date": onset_date.date().isoformat() if onset_date is not None else None,
        "method": calibration.method,
        "method_ru": _method_ru(calibration.method),
        "n_trajectories": len(calibration.alpha_samples),
        "fit_r2_in_sample": round(float(fit_r2), 3),
        "fit_r2_note": "R² рассчитан на данных калибровочного участка и характеризует описание уже наблюдаемой части волны, а не точность прогноза.",
        "phase_code": phase,
        "phase_label": PHASE_LABELS_RU[phase],
        "early_phase_caution": early_caution,
        "peak": peak_block,
        "wave_end": wave_end,
        "horizons": horizons,
        "comparison_with_ml": comparison,
        "parameters": parameters,
        "diagnostics": {
            "r_hat_max": rhat,
            "ess_bulk_min": diag.get("ess_bulk_min"),
            "converged": converged,
            "sampler": diag.get("sampler"),
        },
        "allowed_claims": list(ALLOWED_CLAIMS),
        "forbidden_claims": list(FORBIDDEN_CLAIMS),
        "semantic": {
            "phase_code": phase,
            "agreement_code": comparison["agreement_code"],
            "br_direction_code": br_dir,
            "ml_direction_code": comparison["ml_direction_code"],
            "early_phase_caution": early_caution,
            "peak_already_passed": peak_block["already_passed"],
            "wave_end_within_projection": wave_end["within_projection"],
            "converged": converged,
        },
    }
    return block, warnings


def save_mechanistic_wave_artifacts(result: MechanisticWaveRunResult, output: MechanisticWaveOutputConfig) -> dict[str, Path]:
    out = Path(output.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    artifacts: dict[str, Path] = {}
    if result.trajectories is not None:
        artifacts["trajectories_csv"] = write_table(
            result.trajectories, out / output.trajectories_filename, options=TableWriteOptions(index=False)
        )
    cal = result.calibration
    if isinstance(cal, CalibrationResult) and cal.has_bundle:
        samples = pd.DataFrame({"alpha": cal.alpha_samples[:, 0], "beta": cal.beta_samples[:, 0]})
        artifacts["posterior_samples_csv"] = write_table(
            samples, out / output.samples_filename, options=TableWriteOptions(index=False)
        )
    summary = {**result.to_public_dict(), "calibration_diagnostics": cal.diagnostics if isinstance(cal, CalibrationResult) else None}
    summary.pop("artifacts", None)
    artifacts["summary_json"] = write_json(summary, out / output.summary_filename)
    return artifacts


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _prepare_frame(influenza_weekly: pd.DataFrame, origin_date: Any) -> pd.DataFrame:
    required = ["datetime", "total_cases_formula", "total_population", "inc_per_10k"]
    missing = [c for c in required if c not in influenza_weekly.columns]
    if missing:
        raise MechanisticWaveError(f"В influenza_weekly отсутствуют столбцы {missing} (нужны для модели Барояна–Рвачёва).")
    columns = required + (["pcr_imputed"] if "pcr_imputed" in influenza_weekly.columns else [])
    df = influenza_weekly.loc[:, columns].copy()
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.dropna().sort_values("datetime").drop_duplicates("datetime").reset_index(drop=True)
    if origin_date is not None:
        df = df[df["datetime"] <= pd.Timestamp(origin_date)].reset_index(drop=True)
    if df.empty:
        raise MechanisticWaveError("Нет наблюдений до origin_date.")
    return df


def _season_start(season_year: int, week: int) -> date:
    weeks_in_year = date(season_year, 12, 28).isocalendar().week
    return date.fromisocalendar(season_year, min(week, weeks_in_year), 1)


def _wave_end_index(best: np.ndarray, peak_idx: int, fraction: float) -> int | None:
    threshold = fraction * float(best[peak_idx])
    after = np.where(best[peak_idx + 1 :] < threshold)[0]
    return int(peak_idx + 1 + after[0]) if len(after) else None


def _phase(last_idx: int, peak_idx: int, end_idx: int | None, tol: int) -> PhaseCode:
    if end_idx is not None and last_idx >= end_idx:
        return "post_wave"
    if last_idx < peak_idx - tol:
        return "growth"
    if last_idx <= peak_idx + tol:
        return "peak"
    return "decline"


def _pct(current: float, future: float) -> float:
    return (future - current) / max(abs(current), 1e-9) * 100.0


def _direction(current: float, future: float, threshold_pct: float) -> DirectionCode:
    change = _pct(current, future)
    if change >= threshold_pct:
        return "increase"
    if change <= -threshold_pct:
        return "decrease"
    return "stable"


def _param_summary(best: float, samples: np.ndarray, rd: int) -> JsonObject:
    out: JsonObject = {"estimate": round(float(best), rd + 1), "q05": None, "q95": None}
    if len(samples):
        out["q05"] = round(float(np.percentile(samples, 5)), rd + 1)
        out["q95"] = round(float(np.percentile(samples, 95)), rd + 1)
    return out


def _method_ru(method: str) -> str:
    return {
        "mcmc": "байесовская оценка (ABC-ядро, MCMC DEMetropolisZ)",
        "abc": "байесовская оценка (ABC-ядро, SMC)",
        "optuna": "оптимизация R² (Optuna)",
        "annealing": "оптимизация R² (двойной отжиг)",
    }[method]


__all__ = [
    "AGREEMENT_LABELS_RU",
    "PHASE_LABELS_RU",
    "MechanisticWaveConfig",
    "MechanisticWaveError",
    "MechanisticWaveOutputConfig",
    "MechanisticWaveRunResult",
    "build_mechanistic_wave_context",
    "build_trajectories",
    "detect_wave",
    "ml_forecast_from_result",
    "run_mechanistic_wave_analysis",
]
