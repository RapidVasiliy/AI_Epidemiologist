"""
Калибровка параметров alpha и beta модели Барояна–Рвачёва.

Методы перенесены из ``Influenza_bulletin`` (``model_complex.calibration``):

* ``mcmc`` — ``pm.Simulator`` с гауссовым ядром масштаба ``epsilon`` и
  сэмплер ``DEMetropolisZ`` (основной метод статьи Лазутова и др.);
* ``abc`` — тот же симулятор, sequential Monte Carlo (``pm.sample_smc``);
* ``optuna`` — максимизация R² (TPE);
* ``annealing`` — максимизация R² (``scipy.optimize.dual_annealing``).

Априорные распределения: U[0, 1] для всех компонент alpha и beta.

Отличия от исходного кода (осознанные):

1. ``observation_scale``: наблюдения и модель сравниваются после умножения на
   этот множитель. При ``observation_scale = 10_000 / N`` расхождение считается в
   случаях на 10 тыс. населения, как в формуле (abc) статьи, и ``epsilon = 1``
   имеет тот же смысл, что в статье. ``observation_scale = 1`` воспроизводит
   исходный код (абсолютные числа случаев).
2. Для пучка траекторий выбираются *пары* (alpha, beta) из совместной
   апостериорной выборки. Исходный код выбирал alpha и beta независимо из
   маргинальных распределений и тем самым разрушал их сильную корреляцию
   (на ранней фазе идентифицируемо только произведение alpha·beta).
3. Добавлены ``random_seed`` и диагностика сходимости (R-hat, ESS).
4. Не выполняется ``sample_posterior_predictive`` (результат не использовался).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from ai4epi.mechanistic.models import BaroyanRvachevModel, ModelParams

CalibrationMethod = Literal["mcmc", "abc", "optuna", "annealing"]
TimeStep = Literal["week", "day"]
CALIBRATION_METHODS: tuple[str, ...] = ("mcmc", "abc", "optuna", "annealing")


class CalibrationError(RuntimeError):
    """Калибровка не может быть выполнена."""


@dataclass
class CalibrationResult:
    """Результат калибровки.

    ``alpha_samples``/``beta_samples`` — пары из апостериорной выборки
    (по строке на пару); для точечных методов массивы пусты.
    """

    method: str
    best_params: ModelParams
    alpha_samples: np.ndarray
    beta_samples: np.ndarray
    template: ModelParams
    observation_scale: float
    epsilon: float | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)

    @property
    def has_bundle(self) -> bool:
        return len(self.alpha_samples) > 0

    def sample_params(self) -> list[ModelParams]:
        return [
            self.template.with_values(a, b) for a, b in zip(self.alpha_samples, self.beta_samples, strict=True)
        ]


def model_observable(
    model: BaroyanRvachevModel,
    params: ModelParams,
    n_steps: int,
    *,
    time_step: TimeStep = "week",
) -> np.ndarray:
    """Наблюдаемый ряд модели (группы подряд), как в исходном коде."""

    days = n_steps * 7 if time_step == "week" else n_steps
    sim = model.simulate(params, days)
    series = sim.weekly_newly_infected() if time_step == "week" else sim.newly_infected
    return series.reshape(-1)


def r2_score_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def calibrate(
    model: BaroyanRvachevModel,
    observed: Sequence[float] | np.ndarray,
    template: ModelParams,
    *,
    method: CalibrationMethod = "mcmc",
    time_step: TimeStep = "week",
    observation_scale: float = 1.0,
    epsilon: float = 1.0,
    n_samples: int = 200,
    tune: int = 2500,
    draws: int = 500,
    chains: int = 4,
    cores: int | None = None,
    n_trials: int = 1000,
    random_seed: int | None = 42,
) -> CalibrationResult:
    """Откалибровать ``model`` по наблюдаемому ряду ``observed``.

    ``observed`` — недельные (или суточные) новые случаи в абсолютных числах;
    для модели с группами — ряды групп подряд, как в исходном коде.
    """

    if method not in CALIBRATION_METHODS:
        raise CalibrationError(f"Неизвестный метод калибровки {method!r}; доступны: {CALIBRATION_METHODS}.")
    data = np.asarray(observed, dtype=float).reshape(-1)
    if not np.all(np.isfinite(data)):
        raise CalibrationError("Наблюдаемый ряд содержит пропуски или бесконечности.")
    alpha_dim, beta_dim = model.params_dims()
    if len(data) % alpha_dim:
        raise CalibrationError("Длина ряда не делится на число групп модели.")
    n_steps = len(data) // alpha_dim
    scaled = data * observation_scale

    def observable(alpha: Sequence[float], beta: Sequence[float]) -> np.ndarray:
        return model_observable(model, template.with_values(alpha, beta), n_steps, time_step=time_step) * observation_scale

    if method in ("mcmc", "abc"):
        alpha_s, beta_s, diagnostics = _bayesian(
            observable,
            scaled,
            alpha_dim=alpha_dim,
            beta_dim=beta_dim,
            method=method,
            epsilon=epsilon,
            n_samples=n_samples,
            tune=tune,
            draws=draws,
            chains=chains,
            cores=cores,
            random_seed=random_seed,
        )
        best = template.with_values(alpha_s.mean(axis=0), beta_s.mean(axis=0))
        return CalibrationResult(
            method=method,
            best_params=best,
            alpha_samples=alpha_s,
            beta_samples=beta_s,
            template=template,
            observation_scale=observation_scale,
            epsilon=epsilon,
            diagnostics=diagnostics,
        )

    x, score = (_optuna if method == "optuna" else _annealing)(
        observable, scaled, alpha_dim=alpha_dim, beta_dim=beta_dim, n_trials=n_trials, random_seed=random_seed
    )
    return CalibrationResult(
        method=method,
        best_params=template.with_values(x[:alpha_dim], x[alpha_dim:]),
        alpha_samples=np.empty((0, alpha_dim)),
        beta_samples=np.empty((0, beta_dim)),
        template=template,
        observation_scale=observation_scale,
        epsilon=None,
        diagnostics={"objective": "r2", "best_r2": score},
    )


def _bayesian(
    observable,
    data: np.ndarray,
    *,
    alpha_dim: int,
    beta_dim: int,
    method: str,
    epsilon: float,
    n_samples: int,
    tune: int,
    draws: int,
    chains: int,
    cores: int | None,
    random_seed: int | None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    try:
        import arviz as az
        import pymc as pm
        import pytensor.tensor as pt
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise CalibrationError(
            "Методы mcmc/abc требуют PyMC 5: pip install -e '.[mechanistic]'."
        ) from exc
    if int(pm.__version__.split(".")[0]) >= 6:  # pragma: no cover - depends on environment
        raise CalibrationError(f"Нужен PyMC 5.x, установлен {pm.__version__} (API выборки изменился в 6.x).")

    width = max(alpha_dim, beta_dim)

    def simulator(rng, alpha, beta, size=None):  # noqa: ARG001 - сигнатура pm.Simulator
        return observable(np.asarray(alpha)[:alpha_dim], np.asarray(beta)[:beta_dim])

    with pm.Model():
        alpha = pm.Uniform("alpha", lower=0, upper=1, shape=(alpha_dim,))
        beta = pm.Uniform("beta", lower=0, upper=1, shape=(beta_dim,))
        # pm.Simulator требует параметров одинаковой формы: дополняем нулями,
        # как в исходном коде; лишние компоненты отбрасываются в simulator().
        alpha_in = pt.concatenate([alpha, pt.zeros(width - alpha_dim)]) if width > alpha_dim else alpha
        beta_in = pt.concatenate([beta, pt.zeros(width - beta_dim)]) if width > beta_dim else beta
        pm.Simulator("sim", simulator, alpha_in, beta_in, epsilon=epsilon, observed=data)
        if method == "mcmc":
            idata = pm.sample(
                tune=tune,
                draws=draws,
                chains=chains,
                cores=cores,
                step=pm.DEMetropolisZ(),
                progressbar=False,
                random_seed=random_seed,
                compute_convergence_checks=False,
            )
        else:
            idata = pm.sample_smc(draws=draws, chains=chains, cores=cores, progressbar=False, random_seed=random_seed)

    posterior = idata.posterior.stack(samples=("chain", "draw"))
    a_all = np.asarray(posterior["alpha"].transpose("samples", ...).values, dtype=float).reshape(-1, alpha_dim)
    b_all = np.asarray(posterior["beta"].transpose("samples", ...).values, dtype=float).reshape(-1, beta_dim)
    rng = np.random.default_rng(random_seed)
    idx = rng.choice(len(a_all), size=n_samples, replace=len(a_all) < n_samples)

    summary = az.summary(idata, var_names=["alpha", "beta"], kind="diagnostics")
    diagnostics = {
        "sampler": "DEMetropolisZ" if method == "mcmc" else "SMC",
        "chains": chains,
        "draws": draws,
        "tune": tune if method == "mcmc" else None,
        "posterior_size": len(a_all),
        "r_hat_max": _finite_or_none(summary["r_hat"].max()),
        "ess_bulk_min": _finite_or_none(summary["ess_bulk"].min()),
        "per_parameter": {
            str(name): {"r_hat": _finite_or_none(row["r_hat"]), "ess_bulk": _finite_or_none(row["ess_bulk"])}
            for name, row in summary.iterrows()
        },
    }
    return a_all[idx], b_all[idx], diagnostics


def _optuna(observable, data, *, alpha_dim, beta_dim, n_trials, random_seed):
    try:
        import optuna
    except ImportError as exc:  # pragma: no cover
        raise CalibrationError("Метод optuna требует пакет optuna: pip install -e '.[mechanistic]'.") from exc
    optuna.logging.set_verbosity(optuna.logging.ERROR)

    def objective(trial):
        a = [trial.suggest_float(f"alpha_{i}", 0, 1) for i in range(alpha_dim)]
        b = [trial.suggest_float(f"beta_{i}", 0, 1) for i in range(beta_dim)]
        return r2_score_np(data, observable(a, b))

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=random_seed))
    study.optimize(objective, n_trials=n_trials)
    x = np.array(
        [study.best_params[f"alpha_{i}"] for i in range(alpha_dim)]
        + [study.best_params[f"beta_{i}"] for i in range(beta_dim)]
    )
    return x, float(study.best_value)


def _annealing(observable, data, *, alpha_dim, beta_dim, n_trials, random_seed):  # noqa: ARG001
    from scipy.optimize import dual_annealing

    def loss(x):
        return -r2_score_np(data, observable(x[:alpha_dim], x[alpha_dim:]))

    res = dual_annealing(loss, bounds=[(0.0, 1.0)] * (alpha_dim + beta_dim), seed=random_seed)
    return np.asarray(res.x, dtype=float), float(-res.fun)


def _finite_or_none(value: Any) -> float | None:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) else None


__all__ = [
    "CALIBRATION_METHODS",
    "CalibrationError",
    "CalibrationResult",
    "calibrate",
    "model_observable",
    "r2_score_np",
]
