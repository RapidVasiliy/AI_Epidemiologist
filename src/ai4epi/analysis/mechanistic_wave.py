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

from collections.abc import Mapping
from datetime import date
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator

from ai4epi.analysis.br_conformal import DEFAULT_BR_CONFORMAL_PATH, conformal_band, load_br_conformal
from ai4epi.analysis.preprocessing import is_holiday_week
from ai4epi.core.io import TableWriteOptions, write_json, write_table
from ai4epi.mechanistic import (
    AJELLI_LITVINOVA_3_GROUPS,
    AJELLI_LITVINOVA_ALL_CONTACTS,
    RSCF_CONTACT_MATRIX_2,
    CalibrationResult,
    FineAgeContactBRModel,
    ModelParams,
    calibrate,
    fine_age_populations,
    get_model,
    model_observable,
    negative_binomial_noise,
    r2_score_np,
)
from ai4epi.mechanistic.models import DEFAULT_INFECTIVITY_PROFILE, seir_kernel

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
    "эффективное репродуктивное число R_t по модели и его диапазон по пучку траекторий",
]
FORBIDDEN_CLAIMS: list[str] = [
    "усреднять прогнозы двух моделей или выдавать значения модели Барояна–Рвачёва за основной количественный прогноз",
    "утверждать точную дату окончания эпидемии или эпидемического периода",
    "интерпретировать alpha как долю восприимчивых во всём населении",
    "делать причинные выводы о смене штамма, вакцинации или мерах контроля",
    "называть R² на калибровочном участке точностью прогноза",
    "называть R_t модели измеренной величиной или сравнивать его с R_0 других болезней",
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

    method: Literal["mcmc", "abc", "optuna", "annealing", "bootstrap", "nuts"] = "mcmc"
    # --- ядро модели обновления ----------------------------------------------
    # "br" — эмпирический профиль заразности Барояна–Рвачёва; "seir" — профиль SEIR
    # (Кермак–Маккендрик: стадии E и I с распределением Эрланга, заразность постоянна в I).
    # По умолчанию средний интервал между поколениями SEIR (≈ 3.6 сут) равен БР, различается
    # только форма профиля. incidence: "auto" — исходная форма БР (min/floor), для NUTS —
    # гладкая S·(1 − exp(−λ)).
    model_kernel: Literal["br", "seir"] = "br"
    incidence: Literal["auto", "linear_capped", "exponential"] = "auto"
    seir_latent_mean: float = Field(default=1.6, gt=0.0)
    seir_infectious_mean: float = Field(default=2.0, gt=0.0)
    seir_latent_shape: int = Field(default=1, ge=1)
    seir_infectious_shape: int = Field(default=1, ge=1)
    nuts_tune: int = Field(default=500, ge=0)
    nuts_draws: int = Field(default=300, ge=10)
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
    # Знаменатель относительного разрыва не меньше этого уровня (на 10 тыс.):
    # при малых значениях разница в доли единицы не считается расхождением.
    agreement_gap_floor_per_10k: float = Field(default=5.0, ge=0.0)
    # После завершения основной волны модель БР не используется для вывода
    # (шаг 7 схемы статьи), поэтому согласие не оценивается.
    agreement_skip_post_wave: bool = True
    rhat_threshold: float = Field(default=1.05, gt=1.0)
    # Праздничные ISO-недели: регистрация систематически занижена (СПб, 1-я неделя — в
    # среднем до 0.35 от соседних). Их вес в невязке калибровки (0 — исключить); при
    # обнаружении начала волны значение такой недели заменяется интерполяцией соседних;
    # если праздничная — последняя неделя, сравнение с ML идёт от значения модели.
    holiday_iso_weeks: tuple[int, ...] = (1,)
    holiday_weight: float = Field(default=0.0, ge=0.0, le=1.0)
    # Полоса прогноза общей модели: "conformal" — по ошибкам прошлых прогнозов БР на бэктесте
    # (таблица ``br_conformal_path``, см. ai4epi.analysis.br_conformal); "model" — 5–95 % пучка.
    # Без таблицы используется пучок и выдаётся предупреждение.
    br_band: Literal["conformal", "model"] = "conformal"
    br_conformal_path: Path | None = DEFAULT_BR_CONFORMAL_PATH
    # --- бутстреп (method="bootstrap", как в RSCF_Uncertainty) ---------------
    # Число синтетических рядов (= число траекторий пучка). Полоса q05–q95:
    # "predictive" — траектории перекалиброванных параметров + шум наблюдений
    # с дисперсией остатков (интервал для будущих наблюдений); "parameter" —
    # только неопределённость параметров (кривые модели).
    bootstrap_replicates: int = Field(default=100, ge=5)
    bootstrap_band: Literal["predictive", "parameter"] = "predictive"
    round_digits: int = Field(default=3, ge=0, le=8)
    # --- возрастная модель (0–14 / 15+), см. AgeGroupBRModel ---------------
    # Недоучёт различается по возрастам: дети регистрируются почти полностью
    # (сезонная регистрируемая заболеваемость гриппом детей 0–14 в СПб — 20–30 %
    # группы, что невозможно при kappa = 0.1), взрослые — с недоучётом порядка
    # десяти раз (Leonenko, Bobashev, 2019).
    age_groups: bool = False
    age_underreporting_kappa: tuple[float, float] = (1.0, 0.1)
    age_initial_infectious: tuple[float, float] = (50.0, 50.0)
    # "group_peak": невязка каждой группы нормируется на её наблюдаемый максимум
    # (группы весят одинаково, epsilon — относительная ошибка);
    # "per_10k": на 10 тыс. населения группы (детская группа доминирует).
    age_residual_scale: Literal["group_peak", "per_10k"] = "group_peak"
    age_epsilon: float = Field(default=0.05, gt=0.0)
    age_beta_upper: float = Field(default=10.0, gt=0.0)
    age_tune: int = Field(default=5000, ge=0)
    age_draws: int = Field(default=2000, ge=10)
    age_best_estimate: Literal["mean", "min_distance"] = "min_distance"
    # "matrix": структура контактов фиксирована внешней матрицей (RSCF_Uncertainty,
    # 0–14 / 15+), калибруется один масштабный множитель (3 параметра вместо 6);
    # "free": четыре свободных коэффициента матрицы.
    # "fine_5y": сила инфекции по 5-летним группам и матрице Ajelli & Litvinova (2017, S5),
    # выход суммируется до 0–14 / 15+ (рекомендация авторов матрицы); калибруются alpha
    # двух групп и один множитель.
    # "authors_3g": то же, но матрица 0–6 / 7–14 / 15+, агрегированная авторами.
    age_contact_structure: Literal["matrix", "free", "fine_5y", "authors_3g"] = "fine_5y"
    age_contact_matrix: tuple[tuple[float, float], tuple[float, float]] = RSCF_CONTACT_MATRIX_2
    # Внешняя матрица описывает контакты между *всеми* инфицированными, а в модели
    # группа i содержит лишь регистрируемую долю kappa_i заражённых. Поэтому столбец
    # источника i делится на kappa_i (истинное число заразных = I_i / kappa_i).
    age_contact_kappa_correction: bool = True
    age_matrix_scale_upper: float = Field(default=10.0, gt=0.0)
    age_fine_beta_upper: float = Field(default=1.0, gt=0.0)

    def kernel_profile(self) -> np.ndarray:
        if self.model_kernel == "seir":
            return seir_kernel(
                self.seir_latent_mean,
                self.seir_infectious_mean,
                latent_shape=self.seir_latent_shape,
                infectious_shape=self.seir_infectious_shape,
            )
        return np.asarray(DEFAULT_INFECTIVITY_PROFILE, dtype=float)

    def resolved_incidence(self) -> str:
        if self.incidence != "auto":
            return self.incidence
        return "exponential" if self.method == "nuts" else "linear_capped"

    def beta_scale(self) -> float:
        """Множитель верхней границы β: тот же диапазон R_0 = β·α·Σg, что у профиля БР."""

        return float(np.sum(DEFAULT_INFECTIVITY_PROFILE) / np.sum(self.kernel_profile()))

    def sampler_settings(self, tune: int, draws: int) -> tuple[int, int]:
        return (self.nuts_tune, self.nuts_draws) if self.method == "nuts" else (tune, draws)


class MechanisticWaveOutputConfig(StrictModel):
    output_dir: Path = Path("mechanistic_wave")
    trajectories_filename: str = "trajectories.csv"
    age_trajectories_filename: str = "trajectories_age.csv"
    samples_filename: str = "posterior_samples.csv"
    summary_filename: str = "mechanistic_wave.json"


class MechanisticWaveRunResult(StrictModel):
    status: WaveStageStatus
    message: str | None = None
    context_block: JsonObject | None = None
    trajectories: Any | None = None
    calibration: Any | None = None
    age_trajectories: Any | None = None
    age_calibration: Any | None = None
    artifacts: dict[str, Path] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)

    @field_validator("trajectories", "age_trajectories")
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
    age_group_weekly: Mapping[str, pd.DataFrame] | None = None,
    age_group_labels: Mapping[str, str] | None = None,
    age_ml_forecasts: Mapping[str, JsonObject] | None = None,
) -> MechanisticWaveRunResult:
    """Выполнить механистический этап.

    :param influenza_weekly: таблица с ``datetime``, ``total_cases_formula``,
        ``total_population``, ``inc_per_10k``
    :param origin_date: последняя неделя, доступная моделям (по умолчанию —
        последняя неделя таблицы); совпадает с ``origin`` ML-прогноза
    :param ml_forecast: ``{"point": [...], "lo": [...], "hi": [...], "dates": [...]}``
        для сопоставления с ML-прогнозом; без него блок сопоставления пуст
    :param age_group_weekly: ряды двух возрастных групп (``ai4epi.data.age_groups``)
        для возрастной модели (при ``config.age_groups=True``)
    :param age_ml_forecasts: ML-прогнозы по группам в том же формате, что ``ml_forecast``
    """

    cfg = config or MechanisticWaveConfig()
    frame = _prepare_frame(influenza_weekly, origin_date)
    origin = pd.Timestamp(frame["datetime"].iloc[-1])

    season_year = origin.isocalendar().year if origin.isocalendar().week >= cfg.season_start_week else origin.isocalendar().year - 1
    season_start = pd.Timestamp(_season_start(season_year, cfg.season_start_week))
    season_label = f"{season_year}–{season_year + 1}"
    season = frame[frame["datetime"] >= season_start].reset_index(drop=True)

    wave_detected, onset_date = detect_wave(
        _holiday_interpolated(season["inc_per_10k"].to_numpy(float), season["datetime"], cfg), season["datetime"], cfg
    )
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
    model = get_model("total", infectivity_profile=cfg.kernel_profile(), incidence=cfg.resolved_incidence())
    template = ModelParams(
        alpha=[0.0],
        beta=[0.0],
        population_size=int(population * cfg.underreporting_kappa),
        initial_infectious=[cfg.initial_infectious],
    )
    tune, draws = cfg.sampler_settings(cfg.tune, cfg.draws)
    calibration = calibrate(
        model,
        observed_cases,
        template,
        method=cfg.method,
        observation_scale=per_10k if cfg.observation_scale == "per_10k" else 1.0,
        epsilon=cfg.epsilon,
        beta_upper=1.0 * cfg.beta_scale(),
        n_samples=cfg.bootstrap_replicates if cfg.method == "bootstrap" else cfg.n_trajectories,
        tune=tune,
        draws=draws,
        chains=cfg.chains,
        cores=cfg.cores,
        n_trials=cfg.n_trials,
        random_seed=cfg.random_seed,
        observation_weights=holiday_weights(window["datetime"], cfg),
    )

    trajectories, bundle = build_trajectories(
        model,
        calibration,
        window,
        per_10k=per_10k,
        projection_weeks=cfg.projection_weeks,
        band=cfg.bootstrap_band,
        random_seed=cfg.random_seed,
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

    block["age_groups"] = None
    age_traj: pd.DataFrame | None = None
    age_cal: CalibrationResult | None = None
    if cfg.age_groups:
        if not age_group_weekly or len(age_group_weekly) != 2:
            warnings.append("Возрастная модель БР не запущена: не переданы ряды двух возрастных групп.")
        else:
            age_block, age_traj, age_cal, age_warnings = run_age_group_mechanistic(
                age_group_weekly,
                calibration_start=calib_start,
                origin=origin,
                labels=age_group_labels or {},
                ml_forecasts=age_ml_forecasts or {},
                config=cfg,
            )
            block["age_groups"] = age_block
            warnings.extend(f"возрастная модель: {w}" for w in age_warnings)

    result = MechanisticWaveRunResult(
        status="ok",
        context_block=block,
        trajectories=trajectories,
        calibration=calibration,
        age_trajectories=age_traj,
        age_calibration=age_cal,
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
    band: Literal["predictive", "parameter"] = "predictive",
    random_seed: int | None = None,
) -> tuple[pd.DataFrame, np.ndarray | None]:
    """Центральная траектория, квантили пучка (на 10 тыс.) и R_t.

    Возвращает таблицу и матрицу пучка кривых модели ``(n_trajectories, n_weeks)``
    (или None для точечных методов калибровки). Для бутстрепа при
    ``band="predictive"`` квантили ``q05``/``q50``/``q95`` строятся по кривым с
    добавленным шумом наблюдений (дисперсия остатков калибровки); пучок кривых модели
    (для оценки пика) возвращается без шума. ``r_t`` — эффективное репродуктивное
    число центральной траектории на начало недели, ``r_t_q05``/``r_t_q95`` — по пучку.
    """

    n_obs = len(window)
    n_total = n_obs + projection_weeks
    best = model_observable(model, calibration.best_params, n_total) * per_10k
    r_best = weekly_reproduction_number(model, calibration.best_params, n_total)
    dates = pd.date_range(window["datetime"].iloc[0], periods=n_total, freq="7D")
    df = pd.DataFrame(
        {
            "datetime": dates,
            "iso_week": [int(d.isocalendar().week) for d in dates],
            "is_projection": np.arange(n_total) >= n_obs,
            "observed": np.concatenate([window["inc_per_10k"].to_numpy(float), np.full(projection_weeks, np.nan)]),
            "best": best,
            "r_t": r_best,
        }
    )
    bundle: np.ndarray | None = None
    if calibration.has_bundle:
        params = calibration.sample_params()
        bundle = np.vstack([model_observable(model, p, n_total) * per_10k for p in params])
        banded = bundle
        variance = calibration.diagnostics.get("residual_variance")
        if calibration.method == "bootstrap" and band == "predictive" and variance:
            rng = np.random.default_rng(random_seed)
            banded = negative_binomial_noise(bundle / per_10k, float(variance[0]), rng) * per_10k
        for q in (5, 50, 95):
            df[f"q{q:02d}"] = np.percentile(banded, q, axis=0)
        r_bundle = np.vstack([weekly_reproduction_number(model, p, n_total) for p in params])
        df["r_t_q05"] = np.percentile(r_bundle, 5, axis=0)
        df["r_t_q95"] = np.percentile(r_bundle, 95, axis=0)
    return df, bundle


def weekly_reproduction_number(model: Any, params: ModelParams, n_weeks: int) -> np.ndarray:
    """R_t модели на первый день каждой недели (см. ``BaroyanRvachevModel.reproduction_number``)."""

    sim = model.simulate(params, n_weeks * 7)
    return model.reproduction_number(params, sim)[::7][:n_weeks]


def reproduction_number_block(
    trajectories: pd.DataFrame,
    last_idx: int,
    round_digits: int,
) -> JsonObject:
    """Сводка R_t: на последней наблюдаемой неделе, максимум и неделя перехода через 1."""

    r = trajectories["r_t"].to_numpy(float)
    dates = pd.to_datetime(trajectories["datetime"])
    below = np.nonzero(r < 1.0)[0]
    first_below = int(below[0]) if len(below) else None
    current = float(r[last_idx])
    out: JsonObject = {
        "current": round(current, round_digits),
        "current_date": dates.iloc[last_idx].date().isoformat(),
        "current_q05": None,
        "current_q95": None,
        "max": round(float(np.max(r)), round_digits),
        "below_one_from_date": dates.iloc[first_below].date().isoformat() if first_below is not None else None,
        "below_one_already": bool(first_below is not None and first_below <= last_idx),
        "note": (
            "R_t — эффективное репродуктивное число модели (среднее число заражений от одного больного при "
            "текущем запасе восприимчивых); R_t < 1 означает спад заболеваемости по модели. Это величина модели, "
            "а не прямое измерение."
        ),
    }
    if "r_t_q05" in trajectories:
        out["current_q05"] = round(float(trajectories["r_t_q05"].iloc[last_idx]), round_digits)
        out["current_q95"] = round(float(trajectories["r_t_q95"].iloc[last_idx]), round_digits)
    return out


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

    used = holiday_weights(window["datetime"], cfg) > 0
    fit_r2 = r2_score_np(observed[used], best[:n_obs][used])
    q05 = trajectories["q05"].to_numpy(float) if "q05" in trajectories else None
    q95 = trajectories["q95"].to_numpy(float) if "q95" in trajectories else None
    phase, peak_block, wave_end = summarize_wave(best, bundle, dates, last_idx, cfg)
    current_value = float(observed[-1]) if used[-1] else float(best[last_idx])
    if not used[-1]:
        warnings.append(
            "Последняя неделя — праздничная (регистрация занижена); направление прогнозов сравнивается "
            "от значения модели на этой неделе."
        )
    band_type = _band_type(calibration, cfg)
    conformal_table = load_br_conformal(cfg.br_conformal_path) if cfg.br_band == "conformal" else None
    if cfg.br_band == "conformal" and conformal_table is None:
        warnings.append(
            f"Таблица конформной полосы модели БР не найдена ({cfg.br_conformal_path}); показан 5–95 % пучка траекторий, "
            "он сильно недооценивает неопределённость."
        )
    if conformal_table is not None:
        lo, hi = conformal_band(best, n_obs, conformal_table, phase, current_value)
        q05 = np.where(np.isfinite(lo), lo, q05 if q05 is not None else np.nan)
        q95 = np.where(np.isfinite(hi), hi, q95 if q95 is not None else np.nan)
        trajectories["q05"], trajectories["q95"] = q05, q95
        band_type = "conformal"
    horizons, comparison, cmp_warnings = compare_with_ml(
        current_value, best, q05, q95, dates, last_idx, ml_forecast, cfg, phase=phase
    )
    warnings.extend(cmp_warnings)
    br_dir = comparison["br_direction_code"]

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
        "model_name_ru": "модель Барояна–Рвачёва" if cfg.model_kernel == "br" else "модель SEIR (Кермака–Маккендрика)",
        "model_kernel": cfg.model_kernel,
        "model_description_ru": MODEL_DESCRIPTION_RU,
        "season_label": season_label,
        "calibration_start_date": calibration_start.date().isoformat(),
        "calibration_end_date": dates.iloc[last_idx].date().isoformat(),
        "calibration_weeks": n_obs,
        "wave_onset_date": onset_date.date().isoformat() if onset_date is not None else None,
        "method": calibration.method,
        "method_ru": _method_ru(calibration.method),
        "n_trajectories": len(calibration.alpha_samples),
        "band_type": band_type,
        "band_note": BAND_NOTES_RU.get(band_type),
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
        "reproduction_number": reproduction_number_block(trajectories, last_idx, rd),
        "diagnostics": {
            "r_hat_max": rhat,
            "ess_bulk_min": diag.get("ess_bulk_min"),
            "converged": converged,
            "sampler": diag.get("sampler"),
            "residual_sd_cases": diag.get("residual_sd"),
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
            "r_t_below_one": block_r_t_below_one(trajectories, last_idx),
        },
    }
    return block, warnings


BAND_NOTES_RU: dict[str | None, str] = {
    "conformal": (
        "диапазон модели Барояна–Рвачёва построен по ошибкам её прошлых прогнозов на исторических данных "
        "(номинально 90 % будущих наблюдений)"
    ),
    "predictive": "диапазон — бутстреп параметров модели с шумом наблюдений",
    "parameter": "диапазон — бутстреп параметров модели, без шума наблюдений",
    "posterior": (
        "диапазон — 5–95 % пучка траекторий модели; отражает только неопределённость параметров и обычно "
        "заметно уже реального разброса"
    ),
}


def block_r_t_below_one(trajectories: pd.DataFrame, last_idx: int) -> bool:
    return bool(float(trajectories["r_t"].iloc[last_idx]) < 1.0)


def _band_type(calibration: CalibrationResult, cfg: MechanisticWaveConfig) -> str | None:
    if not calibration.has_bundle:
        return None
    if calibration.method == "bootstrap":
        return "predictive" if cfg.bootstrap_band == "predictive" else "parameter"
    return "posterior"


def summarize_wave(
    best: np.ndarray,
    bundle: np.ndarray | None,
    dates: pd.Series,
    last_idx: int,
    cfg: MechanisticWaveConfig,
) -> tuple[PhaseCode, JsonObject, JsonObject]:
    """Фаза волны, максимум (с диапазоном по пучку) и завершение основной волны."""

    rd = cfg.round_digits
    dates = pd.to_datetime(pd.Series(dates)).reset_index(drop=True)
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
        "definition": (
            "первая неделя после максимума, на которой центральная траектория ниже "
            f"{int(cfg.wave_end_fraction * 100)}% от максимума"
        ),
        "within_projection": end_idx is not None,
        "date": dates.iloc[end_idx].date().isoformat() if end_idx is not None else None,
        "iso_week": int(dates.iloc[end_idx].isocalendar().week) if end_idx is not None else None,
        "weeks_from_current": int(end_idx - last_idx) if end_idx is not None else None,
    }
    return phase, peak_block, wave_end


def compare_with_ml(
    current_value: float,
    best: np.ndarray,
    q05: np.ndarray | None,
    q95: np.ndarray | None,
    dates: pd.Series,
    last_idx: int,
    ml_forecast: JsonObject | None,
    cfg: MechanisticWaveConfig,
    phase: str | None = None,
) -> tuple[list[JsonObject], JsonObject, list[str]]:
    """Значения модели БР и ML на горизонтах 1..h и код согласия (шаг 5 схемы).

    Согласие: одинаковое направление (изменение от текущей недели к h, порог
    ``direction_change_pct_threshold``) и средний разрыв значений
    ``|БР − ML| / max(|ML|, agreement_gap_floor_per_10k)`` не больше
    ``agreement_relative_gap_pct``. В фазе ``post_wave`` (при
    ``agreement_skip_post_wave``) согласие не оценивается.

    Пороги проверены скользящим бэктестом (docs/backtest.md): на неделях волны
    при «расхождении» ошибка ML примерно вдвое выше, а покрытие его интервалов
    заметно ниже номинального.
    """

    rd = cfg.round_digits
    dates = pd.to_datetime(pd.Series(dates)).reset_index(drop=True)
    warnings: list[str] = []
    horizons: list[JsonObject] = []
    h = cfg.forecast_horizon_weeks
    for k in range(1, h + 1):
        idx = last_idx + k
        row: JsonObject = {
            "horizon_weeks": k,
            "target_date": dates.iloc[idx].date().isoformat(),
            "target_iso_week": int(dates.iloc[idx].isocalendar().week),
            "br_value": round(float(best[idx]), rd),
            "br_low": round(float(q05[idx]), rd) if q05 is not None else None,
            "br_high": round(float(q95[idx]), rd) if q95 is not None else None,
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
        "agreement_note": None,
    }
    if phase == "post_wave" and cfg.agreement_skip_post_wave:
        comparison["agreement_note"] = (
            "основная волна по модели Барояна–Рвачёва завершилась; для вывода используется только ML-прогноз"
        )
    elif all(r["ml_value"] is not None for r in horizons):
        ml_h = float(horizons[-1]["ml_value"])
        ml_dir = _direction(current_value, ml_h, cfg.direction_change_pct_threshold)
        floor = max(cfg.agreement_gap_floor_per_10k, 1e-9)
        gap = float(np.mean([abs(r["br_value"] - r["ml_value"]) / max(abs(r["ml_value"]), floor) * 100.0 for r in horizons]))
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
    return horizons, comparison, warnings


def run_age_group_mechanistic(
    age_group_weekly: Mapping[str, pd.DataFrame],
    *,
    calibration_start: pd.Timestamp,
    origin: pd.Timestamp,
    labels: Mapping[str, str],
    ml_forecasts: Mapping[str, JsonObject],
    config: MechanisticWaveConfig,
) -> tuple[JsonObject | None, pd.DataFrame | None, CalibrationResult | None, list[str]]:
    """Калибровка возрастной модели БР (0–14 / 15+) и сводка по каждой группе.

    Группы берутся в порядке словаря: первая — дети, вторая — взрослые.
    """

    cfg = config
    rd = cfg.round_digits
    warnings: list[str] = []
    codes = list(age_group_weekly)
    frames = []
    for code in codes:
        f = _prepare_frame(age_group_weekly[code], origin)
        frames.append(f[f["datetime"] >= calibration_start].reset_index(drop=True))
    if len({len(f) for f in frames}) != 1 or not all((frames[0]["datetime"] == f["datetime"]).all() for f in frames):
        warnings.append("ряды возрастных групп не совпадают по неделям; калибровка пропущена.")
        return None, None, None, warnings
    if len(frames[0]) < cfg.min_calibration_weeks:
        warnings.append("недостаточно недель для калибровки возрастной модели.")
        return None, None, None, warnings

    observed = np.concatenate([f["total_cases_formula"].to_numpy(float) for f in frames])
    populations = [float(f["total_population"].iloc[0]) for f in frames]
    rho_g = [k * n for k, n in zip(cfg.age_underreporting_kappa, populations, strict=True)]
    if cfg.age_residual_scale == "group_peak":
        scale = [1.0 / max(float(f["total_cases_formula"].max()), 1.0) for f in frames]
    else:
        scale = [10_000.0 / n for n in populations]
    use_matrix = cfg.age_contact_structure == "matrix"
    use_fine = cfg.age_contact_structure in ("fine_5y", "authors_3g")
    contact_matrix = None
    if use_matrix:
        contact_matrix = np.asarray(cfg.age_contact_matrix, dtype=float)
        if cfg.age_contact_kappa_correction:
            contact_matrix = contact_matrix / np.asarray(cfg.age_underreporting_kappa, dtype=float)[None, :]
    structure = cfg.age_contact_structure
    if use_fine:
        def _pops(ks: tuple[int, ...]) -> list[float] | None:
            cols = [f"source_population_age_group_{k}" for k in ks]
            if not set(cols) <= set(frames[0].columns):
                return None
            row = frames[0][cols].iloc[0]
            return None if row.isna().any() or (row <= 0).any() else row.astype(float).tolist()

        pops5 = _pops((0, 1, 2, 4, 5))
        pops4 = _pops((0, 1, 2, 3))
        if structure == "fine_5y" and pops5 is None and pops4 is not None:
            warnings.append("нет численностей 15–64 и 65+; сила инфекции считается по матрице авторов 0–6 / 7–14 / 15+.")
            structure = "authors_3g"
        if (structure == "fine_5y" and pops5 is None) or (structure == "authors_3g" and pops4 is None):
            warnings.append("нет численностей подгрупп 0–2, 3–6, 7–14; используется матрица 2×2 (age_contact_structure='matrix').")
            structure, use_fine, use_matrix = "matrix", False, True
            contact_matrix = np.asarray(cfg.age_contact_matrix, dtype=float)
            if cfg.age_contact_kappa_correction:
                contact_matrix = contact_matrix / np.asarray(cfg.age_underreporting_kappa, dtype=float)[None, :]
    if use_fine:
        kappa = np.asarray(cfg.age_underreporting_kappa, dtype=float)
        if structure == "fine_5y":
            pops = pops5
            n_fine = fine_age_populations(*pops)
            group_index = np.array([0, 0, 0] + [1] * 9)
            matrix = AJELLI_LITVINOVA_ALL_CONTACTS
        else:
            pops = pops4
            n_fine = np.array([pops[0] + pops[1], pops[2], pops[3]])
            group_index = np.array([0, 0, 1])
            matrix = AJELLI_LITVINOVA_3_GROUPS
        model = FineAgeContactBRModel(
            n_fine * kappa[group_index],
            group_index,
            contact_matrix=matrix,
            infectivity_profile=cfg.kernel_profile(),
            incidence=cfg.resolved_incidence(),
        )
    else:
        if cfg.method == "nuts":
            warnings.append("NUTS поддерживает только возрастные модели fine_5y / authors_3g; калибровка пропущена.")
            return None, None, None, warnings
        model = get_model(
            "age",
            contact_matrix=contact_matrix,
            infectivity_profile=cfg.kernel_profile(),
            incidence=cfg.resolved_incidence(),
        )
    age_tune, age_draws = cfg.sampler_settings(cfg.age_tune, cfg.age_draws)
    template = ModelParams(
        alpha=[0.0, 0.0],
        beta=[0.0] * model.params_dims()[1],
        population_size=int(sum(rho_g)),
        initial_infectious=list(cfg.age_initial_infectious),
        group_population_sizes=rho_g,
    )
    calibration = calibrate(
        model,
        observed,
        template,
        method=cfg.method,
        observation_scale=scale,
        epsilon=cfg.age_epsilon,
        beta_upper=cfg.beta_scale()
        * (cfg.age_fine_beta_upper if use_fine else cfg.age_matrix_scale_upper if use_matrix else cfg.age_beta_upper),
        n_samples=cfg.bootstrap_replicates if cfg.method == "bootstrap" else cfg.n_trajectories,
        tune=age_tune,
        draws=age_draws,
        chains=cfg.chains,
        cores=cfg.cores,
        n_trials=cfg.n_trials,
        random_seed=cfg.random_seed,
        best_estimate=cfg.age_best_estimate,
        observation_weights=holiday_weights(frames[0]["datetime"], cfg),
    )
    used = holiday_weights(frames[0]["datetime"], cfg) > 0

    n_obs = len(frames[0])
    n_total = n_obs + cfg.projection_weeks
    last_idx = n_obs - 1
    per_10k = np.array([10_000.0 / n for n in populations])[:, None]
    best = model_observable(model, calibration.best_params, n_total).reshape(2, -1) * per_10k
    bundle = None
    banded = None
    if calibration.has_bundle:
        bundle = np.stack(
            [model_observable(model, p, n_total).reshape(2, -1) * per_10k for p in calibration.sample_params()]
        )  # (n, groups, weeks)
        banded = bundle
        variance = calibration.diagnostics.get("residual_variance")
        if calibration.method == "bootstrap" and cfg.bootstrap_band == "predictive" and variance:
            rng = np.random.default_rng(cfg.random_seed)
            var = np.asarray(variance, dtype=float)[None, :, None]
            banded = negative_binomial_noise(bundle / per_10k[None], var * np.ones_like(bundle), rng) * per_10k[None]
    r_t = weekly_reproduction_number(model, calibration.best_params, n_total)
    dates = pd.Series(pd.date_range(frames[0]["datetime"].iloc[0], periods=n_total, freq="7D"))

    rows = []
    groups_out: list[JsonObject] = []
    for g, code in enumerate(codes):
        obs_g = frames[g]["inc_per_10k"].to_numpy(float)
        bundle_g = bundle[:, g, :] if bundle is not None else None
        band_g = banded[:, g, :] if banded is not None else None
        q05 = np.percentile(band_g, 5, axis=0) if band_g is not None else None
        q95 = np.percentile(band_g, 95, axis=0) if band_g is not None else None
        phase, peak, wave_end = summarize_wave(best[g], bundle_g, dates, last_idx, cfg)
        horizons, comparison, cmp_w = compare_with_ml(
            float(obs_g[-1]) if used[-1] else float(best[g][last_idx]),
            best[g], q05, q95, dates, last_idx, ml_forecasts.get(code), cfg, phase=phase
        )
        warnings.extend(f"{code}: {w}" for w in cmp_w if "не передан" not in w)
        groups_out.append(
            {
                "age_group_code": code,
                "age_group_label": labels.get(code, code),
                "fit_r2_in_sample": round(float(r2_score_np(obs_g[used], best[g][:n_obs][used])), 3),
                "phase_code": phase,
                "phase_label": PHASE_LABELS_RU[phase],
                "peak": peak,
                "wave_end": wave_end,
                "horizons": horizons,
                "comparison_with_ml": comparison,
            }
        )
        rows.append(
            pd.DataFrame(
                {
                    "datetime": dates,
                    "age_group_code": code,
                    "is_projection": np.arange(n_total) >= n_obs,
                    "observed": np.concatenate([obs_g, np.full(cfg.projection_weeks, np.nan)]),
                    "best": best[g],
                    "q05": q05 if q05 is not None else np.nan,
                    "q95": q95 if q95 is not None else np.nan,
                }
            )
        )

    peak_dates = [pd.Timestamp(gr["peak"]["date"]) for gr in groups_out]
    lead_weeks = int((peak_dates[1] - peak_dates[0]).days // 7)
    diag = calibration.diagnostics
    rhat = diag.get("r_hat_max")
    converged = None if rhat is None else bool(rhat <= cfg.rhat_threshold)
    if converged is False:
        warnings.append(
            f"MCMC: R-hat = {rhat:.2f} > {cfg.rhat_threshold}; "
            + ("оценки параметров возрастной модели ненадёжны." if (use_matrix or use_fine) else "оценки матрицы контактов ненадёжны.")
        )
    contact = model.contact(calibration.best_params) if not use_fine else np.full((2, 2), np.nan)
    block: JsonObject = {
        "model_name_ru": "возрастная модель Барояна–Рвачёва (0–14 и 15+ лет)",
        "groups": groups_out,
        "children_peak_lead_weeks": lead_weeks,
        "children_peak_lead_note": "положительное значение — максимум у детей наступает раньше, чем у взрослых",
        "underreporting_kappa": list(cfg.age_underreporting_kappa),
        "alpha": [round(float(a), rd + 1) for a in calibration.best_params.alpha],
        "contact_matrix": None if use_fine else [[round(float(v), rd) for v in row] for row in contact],
        "contact_scale": round(float(calibration.best_params.beta[0]), rd + 2) if use_fine else None,
        "contact_structure": structure,
        "contact_matrix_note": (
            "строка — заражаемая группа, столбец — источник; структура матрицы задана внешней матрицей контактов "
            "(RSCF_Uncertainty) с поправкой на недоучёт по группам, по данным оценён только общий масштаб"
            if use_matrix
            else "сила инфекции рассчитана по возрастным группам матрицы контактов Ajelli & Litvinova (2017) "
            "и суммирована до 0–14 / 15+; по данным оценён только общий масштаб передачи"
            if use_fine
            else "строка — заражаемая группа, столбец — источник; матрица плохо идентифицируема по двум недельным "
            "рядам, интерпретировать стоит фазы и максимумы групп, а не отдельные коэффициенты"
        ),
        "reproduction_number_current": round(float(r_t[last_idx]), rd),
        "band_type": _band_type(calibration, cfg),
        "residual_scale": cfg.age_residual_scale,
        "diagnostics": {"r_hat_max": rhat, "ess_bulk_min": diag.get("ess_bulk_min"), "converged": converged},
        "semantic": {
            "phase_codes": {gr["age_group_code"]: gr["phase_code"] for gr in groups_out},
            "agreement_codes": {gr["age_group_code"]: gr["comparison_with_ml"]["agreement_code"] for gr in groups_out},
            "children_peak_lead_weeks": lead_weeks,
            "converged": converged,
        },
    }
    return block, pd.concat(rows, ignore_index=True), calibration, warnings


def save_mechanistic_wave_artifacts(result: MechanisticWaveRunResult, output: MechanisticWaveOutputConfig) -> dict[str, Path]:
    out = Path(output.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    artifacts: dict[str, Path] = {}
    if result.trajectories is not None:
        artifacts["trajectories_csv"] = write_table(
            result.trajectories, out / output.trajectories_filename, options=TableWriteOptions(index=False)
        )
    if result.age_trajectories is not None:
        artifacts["age_trajectories_csv"] = write_table(
            result.age_trajectories, out / output.age_trajectories_filename, options=TableWriteOptions(index=False)
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


def holiday_weights(dates: pd.Series, cfg: MechanisticWaveConfig) -> np.ndarray:
    """Вес каждой недели в невязке калибровки: ``holiday_weight`` для праздничных, 1 для остальных."""

    holiday = is_holiday_week(dates, cfg.holiday_iso_weeks)
    weights = np.where(holiday, cfg.holiday_weight, 1.0)
    return weights if weights.any() else np.ones(len(weights))


def _holiday_interpolated(values: np.ndarray, dates: pd.Series, cfg: MechanisticWaveConfig) -> np.ndarray:
    """Значения праздничных недель заменяются линейной интерполяцией соседних (для поиска начала волны)."""

    holiday = is_holiday_week(dates, cfg.holiday_iso_weeks)
    if not holiday.any():
        return values
    series = pd.Series(np.where(holiday, np.nan, values))
    return series.interpolate(limit_direction="both").to_numpy(float)


def _prepare_frame(influenza_weekly: pd.DataFrame, origin_date: Any) -> pd.DataFrame:
    required = ["datetime", "total_cases_formula", "total_population", "inc_per_10k"]
    missing = [c for c in required if c not in influenza_weekly.columns]
    if missing:
        raise MechanisticWaveError(f"В influenza_weekly отсутствуют столбцы {missing} (нужны для модели Барояна–Рвачёва).")
    extra = [c for c in influenza_weekly.columns if c == "pcr_imputed" or c.startswith("source_population_")]
    df = influenza_weekly.loc[:, required + extra].copy()
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.dropna(subset=required).sort_values("datetime").drop_duplicates("datetime").reset_index(drop=True)
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
        "bootstrap": "параметрический бутстреп (отрицательно-биномиальный шум, перекалибровка)",
        "nuts": "байесовская оценка (явное правдоподобие, NUTS)",
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
    "compare_with_ml",
    "detect_wave",
    "ml_forecast_from_result",
    "run_age_group_mechanistic",
    "run_mechanistic_wave_analysis",
    "summarize_wave",
]
