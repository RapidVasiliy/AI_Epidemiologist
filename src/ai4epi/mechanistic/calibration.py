"""
Калибровка параметров alpha и beta модели Барояна–Рвачёва.

Методы перенесены из ``Influenza_bulletin`` (``model_complex.calibration``):

* ``mcmc`` — ``pm.Simulator`` с гауссовым ядром масштаба ``epsilon`` и
  сэмплер ``DEMetropolisZ`` (основной метод статьи Лазутова и др.);
* ``abc`` — тот же симулятор, sequential Monte Carlo (``pm.sample_smc``);
* ``optuna`` — максимизация R² (TPE);
* ``annealing`` — максимизация R² (``scipy.optimize.dual_annealing``);
* ``nuts`` — явное правдоподобие (нормальное в шкале ``observation_scale``, масштаб ошибки σ
  оценивается, σ ~ HalfNormal) и градиентный сэмплер NUTS по дифференцируемой реализации модели
  (``ai4epi.mechanistic.differentiable``); только для гладкой формы ``incidence="exponential"``;
* ``bootstrap`` — параметрический бутстреп, как в RSCF_Uncertainty
  (``BaroyanAgeMultistrain_v2/bootstrapping``): точечная оценка (минимум невязки,
  ``dual_annealing``), затем ``n_samples`` синтетических рядов «модель +
  отрицательно-биномиальный шум» с дисперсией, оценённой по остаткам каждой группы,
  и перекалибровка на каждом (Нелдер–Мид от точечной оценки). Разброс
  перекалиброванных параметров даёт пучок траекторий; дисперсия шума
  (``diagnostics["residual_variance"]``) позволяет построить предсказательный интервал.

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

CalibrationMethod = Literal["mcmc", "abc", "optuna", "annealing", "bootstrap", "nuts"]
TimeStep = Literal["week", "day"]
CALIBRATION_METHODS: tuple[str, ...] = ("mcmc", "abc", "optuna", "annealing", "bootstrap", "nuts")


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
    observation_scale: float | list[float]
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
    observation_scale: float | Sequence[float] = 1.0,
    epsilon: float = 1.0,
    alpha_upper: float = 1.0,
    beta_upper: float = 1.0,
    n_samples: int = 200,
    tune: int = 2500,
    draws: int = 500,
    chains: int = 4,
    cores: int | None = None,
    n_trials: int = 1000,
    random_seed: int | None = 42,
    best_estimate: Literal["mean", "min_distance"] = "mean",
    observation_weights: Sequence[float] | np.ndarray | None = None,
) -> CalibrationResult:
    """Откалибровать ``model`` по наблюдаемому ряду ``observed``.

    ``observed`` — недельные (или суточные) новые случаи в абсолютных числах;
    для модели с группами — ряды групп подряд, как в исходном коде.
    ``observation_scale`` — число или по числу на группу (например,
    ``10_000 / N_j``: невязка в случаях на 10 тыс. населения своей группы).
    Априорные распределения: alpha ~ U[0, alpha_upper], beta ~ U[0, beta_upper].

    ``best_estimate`` (mcmc/abc): ``"mean"`` — средние по выборке, как в статье и
    исходном коде; ``"min_distance"`` — пара из выборки с наименьшей невязкой
    (устойчивее при многомодальном или несошедшемся апостериорном распределении,
    например для возрастной модели с шестью параметрами).

    ``observation_weights`` — вес каждого шага ряда (длина ``n_steps``, общая для всех групп,
    или полная длина ряда): невязка шага умножается на ``sqrt(w)``; вес 0 исключает шаг
    (например, праздничную неделю с систематически заниженной регистрацией).
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
    scale_per_group = np.broadcast_to(np.asarray(observation_scale, dtype=float), (alpha_dim,))
    base_scale = np.repeat(scale_per_group, n_steps)
    weights = np.ones(len(data))
    if observation_weights is not None:
        w = np.asarray(observation_weights, dtype=float).reshape(-1)
        if len(w) == n_steps:
            w = np.tile(w, alpha_dim)
        if len(w) != len(data) or np.any(w < 0) or not np.any(w > 0):
            raise CalibrationError("observation_weights: неверная длина, отрицательные веса или все веса нулевые.")
        weights = w
    scale = base_scale * np.sqrt(weights)
    scaled = data * scale

    def raw_observable(alpha: Sequence[float], beta: Sequence[float]) -> np.ndarray:
        return model_observable(model, template.with_values(alpha, beta), n_steps, time_step=time_step)

    def observable(alpha: Sequence[float], beta: Sequence[float]) -> np.ndarray:
        return raw_observable(alpha, beta) * scale

    if method in ("mcmc", "abc"):
        alpha_s, beta_s, diagnostics = _bayesian(
            observable,
            scaled,
            alpha_dim=alpha_dim,
            beta_dim=beta_dim,
            method=method,
            epsilon=epsilon,
            alpha_upper=alpha_upper,
            beta_upper=beta_upper,
            n_samples=n_samples,
            tune=tune,
            draws=draws,
            chains=chains,
            cores=cores,
            random_seed=random_seed,
        )
        if best_estimate == "min_distance":
            distances = [float(np.sum((observable(a, b) - scaled) ** 2)) for a, b in zip(alpha_s, beta_s, strict=True)]
            k = int(np.argmin(distances))
            best = template.with_values(alpha_s[k], beta_s[k])
        else:
            best = template.with_values(alpha_s.mean(axis=0), beta_s.mean(axis=0))
        diagnostics["best_estimate"] = best_estimate
        return CalibrationResult(
            method=method,
            best_params=best,
            alpha_samples=alpha_s,
            beta_samples=beta_s,
            template=template,
            observation_scale=float(scale_per_group[0]) if alpha_dim == 1 else scale_per_group.tolist(),
            epsilon=epsilon,
            diagnostics=diagnostics,
        )

    if method == "nuts":
        if time_step != "week":
            raise CalibrationError("Метод nuts поддерживает только недельный шаг.")
        if getattr(model, "incidence", None) != "exponential":
            raise CalibrationError("Метод nuts требует гладкую форму модели (incidence='exponential').")
        alpha_s, beta_s, diagnostics = _nuts(
            model,
            template,
            data,
            base_scale,
            weights,
            n_steps=n_steps,
            alpha_dim=alpha_dim,
            beta_dim=beta_dim,
            alpha_upper=alpha_upper,
            beta_upper=beta_upper,
            n_samples=n_samples,
            tune=tune,
            draws=draws,
            chains=chains,
            cores=cores,
            random_seed=random_seed,
        )
        if best_estimate == "min_distance":
            distances = [float(np.sum((observable(a, b) - scaled) ** 2)) for a, b in zip(alpha_s, beta_s, strict=True)]
            k = int(np.argmin(distances))
            best = template.with_values(alpha_s[k], beta_s[k])
        else:
            best = template.with_values(alpha_s.mean(axis=0), beta_s.mean(axis=0))
        diagnostics["best_estimate"] = best_estimate
        return CalibrationResult(
            method=method,
            best_params=best,
            alpha_samples=alpha_s,
            beta_samples=beta_s,
            template=template,
            observation_scale=float(scale_per_group[0]) if alpha_dim == 1 else scale_per_group.tolist(),
            epsilon=None,
            diagnostics=diagnostics,
        )

    if method == "bootstrap":
        x, alpha_s, beta_s, diagnostics = _bootstrap(
            observable,
            raw_observable,
            data,
            scale,
            weights > 0,
            alpha_dim=alpha_dim,
            beta_dim=beta_dim,
            n_replicates=n_samples,
            random_seed=random_seed,
            alpha_upper=alpha_upper,
            beta_upper=beta_upper,
        )
        return CalibrationResult(
            method=method,
            best_params=template.with_values(x[:alpha_dim], x[alpha_dim:]),
            alpha_samples=alpha_s,
            beta_samples=beta_s,
            template=template,
            observation_scale=float(scale_per_group[0]) if alpha_dim == 1 else scale_per_group.tolist(),
            epsilon=None,
            diagnostics=diagnostics,
        )

    x, score = (_optuna if method == "optuna" else _annealing)(
        observable,
        scaled,
        alpha_dim=alpha_dim,
        beta_dim=beta_dim,
        n_trials=n_trials,
        random_seed=random_seed,
        alpha_upper=alpha_upper,
        beta_upper=beta_upper,
    )
    return CalibrationResult(
        method=method,
        best_params=template.with_values(x[:alpha_dim], x[alpha_dim:]),
        alpha_samples=np.empty((0, alpha_dim)),
        beta_samples=np.empty((0, beta_dim)),
        template=template,
        observation_scale=float(scale_per_group[0]) if alpha_dim == 1 else scale_per_group.tolist(),
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
    alpha_upper: float,
    beta_upper: float,
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

    def simulator(rng, alpha, beta, size=None):
        return observable(np.asarray(alpha)[:alpha_dim], np.asarray(beta)[:beta_dim])

    with pm.Model():
        alpha = pm.Uniform("alpha", lower=0, upper=alpha_upper, shape=(alpha_dim,))
        beta = pm.Uniform("beta", lower=0, upper=beta_upper, shape=(beta_dim,))
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


def _optuna(observable, data, *, alpha_dim, beta_dim, n_trials, random_seed, alpha_upper, beta_upper):
    try:
        import optuna
    except ImportError as exc:  # pragma: no cover
        raise CalibrationError("Метод optuna требует пакет optuna: pip install -e '.[mechanistic]'.") from exc
    optuna.logging.set_verbosity(optuna.logging.ERROR)

    def objective(trial):
        a = [trial.suggest_float(f"alpha_{i}", 0, alpha_upper) for i in range(alpha_dim)]
        b = [trial.suggest_float(f"beta_{i}", 0, beta_upper) for i in range(beta_dim)]
        return r2_score_np(data, observable(a, b))

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=random_seed))
    study.optimize(objective, n_trials=n_trials)
    x = np.array(
        [study.best_params[f"alpha_{i}"] for i in range(alpha_dim)]
        + [study.best_params[f"beta_{i}"] for i in range(beta_dim)]
    )
    return x, float(study.best_value)


def _annealing(observable, data, *, alpha_dim, beta_dim, n_trials, random_seed, alpha_upper, beta_upper):
    from scipy.optimize import dual_annealing

    def loss(x):
        return -r2_score_np(data, observable(x[:alpha_dim], x[alpha_dim:]))

    bounds = [(0.0, alpha_upper)] * alpha_dim + [(0.0, beta_upper)] * beta_dim
    res = dual_annealing(loss, bounds=bounds, seed=random_seed)
    return np.asarray(res.x, dtype=float), float(-res.fun)


def _nuts(
    model: BaroyanRvachevModel,
    template: ModelParams,
    data: np.ndarray,
    base_scale: np.ndarray,
    weights: np.ndarray,
    *,
    n_steps: int,
    alpha_dim: int,
    beta_dim: int,
    alpha_upper: float,
    beta_upper: float,
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
    except ImportError as exc:  # pragma: no cover
        raise CalibrationError("Метод nuts требует PyMC 5: pip install -e '.[mechanistic]'.") from exc
    from ai4epi.mechanistic.differentiable import weekly_observable_tensor

    if beta_dim != 1:
        raise CalibrationError("Метод nuts поддерживает модели с одним параметром интенсивности передачи.")
    used = weights > 0
    y = (data * base_scale)[used]
    w = weights[used]
    spec = model.symbolic_spec(template)
    # Параметризация (α, r): r = β·ᾱ, ᾱ — средняя доля восприимчивых, взвешенная по численностям
    # групп. На ранней фазе волны данные определяют в основном произведение α·β, и в исходной
    # параметризации апостериорное распределение — узкий изогнутый «хребет», по которому NUTS идёт
    # шагами по 100+ точек. В (α, r) хребет выпрямляется (глубина дерева ~3 вместо ~7, в 5 раз
    # быстрее). Априорные: α ~ U[0, alpha_upper], r ~ U[0, beta_upper] (тот же диапазон, что у β
    # при α = 1); β = r / ᾱ — производная величина.
    rho = np.asarray(spec["rho"], dtype=float)
    idx = np.asarray(spec["group_index"], dtype=int)
    group_weight = np.bincount(idx, weights=rho, minlength=alpha_dim)
    group_weight = group_weight / group_weight.sum()
    with pm.Model():
        alpha = pm.Uniform("alpha", lower=0, upper=alpha_upper, shape=(alpha_dim,))
        r = pm.Uniform("r", lower=0, upper=beta_upper, shape=(beta_dim,))
        alpha_bar = pm.math.dot(alpha, group_weight)
        beta = pm.Deterministic("beta", r / alpha_bar)
        sigma = pm.HalfNormal("sigma", sigma=max(float(np.std(y)), 1e-6))
        mu = weekly_observable_tensor(alpha, beta, n_weeks=n_steps, **spec) * base_scale
        pm.Normal("obs", mu=mu[used], sigma=sigma / np.sqrt(w), observed=y)
        idata = pm.sample(
            tune=tune,
            draws=draws,
            chains=chains,
            cores=cores,
            target_accept=0.9,
            progressbar=False,
            random_seed=random_seed,
            compute_convergence_checks=False,
        )
    posterior = idata.posterior.stack(samples=("chain", "draw"))
    a_all = np.asarray(posterior["alpha"].transpose("samples", ...).values, dtype=float).reshape(-1, alpha_dim)
    b_all = np.asarray(posterior["beta"].transpose("samples", ...).values, dtype=float).reshape(-1, beta_dim)
    rng = np.random.default_rng(random_seed)
    idx = rng.choice(len(a_all), size=n_samples, replace=len(a_all) < n_samples)
    summary = az.summary(idata, var_names=["alpha", "r", "sigma"], kind="diagnostics")
    divergences = int(idata.sample_stats["diverging"].sum()) if "diverging" in idata.sample_stats else None
    diagnostics = {
        "sampler": "NUTS",
        "chains": chains,
        "draws": draws,
        "tune": tune,
        "posterior_size": len(a_all),
        "r_hat_max": _finite_or_none(summary["r_hat"].max()),
        "ess_bulk_min": _finite_or_none(summary["ess_bulk"].min()),
        "divergences": divergences,
        "sigma_mean": _finite_or_none(posterior["sigma"].mean()),
        "per_parameter": {
            str(name): {"r_hat": _finite_or_none(row["r_hat"]), "ess_bulk": _finite_or_none(row["ess_bulk"])}
            for name, row in summary.iterrows()
        },
    }
    return a_all[idx], b_all[idx], diagnostics


def negative_binomial_noise(
    mean: np.ndarray,
    variance: np.ndarray | float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Отрицательно-биномиальные отсчёты с заданными средним и дисперсией.

    Как ``TruncatedNegativeBinomial.get_random_value_mv`` из RSCF_Uncertainty, но без
    усечения нуля; если дисперсия не больше среднего — пуассоновские отсчёты.
    """

    mean = np.clip(np.asarray(mean, dtype=float), 0.0, None)
    var = np.broadcast_to(np.asarray(variance, dtype=float), mean.shape)
    out = np.zeros_like(mean)
    over = (mean > 0) & (var > mean * (1 + 1e-9))
    p = mean[over] / var[over]
    n = mean[over] * p / (1.0 - p)
    out[over] = rng.negative_binomial(n, p)
    pois = (mean > 0) & ~over
    out[pois] = rng.poisson(mean[pois])
    return out


def _bootstrap(
    observable,
    raw_observable,
    data: np.ndarray,
    scale: np.ndarray,
    used: np.ndarray,
    *,
    alpha_dim: int,
    beta_dim: int,
    n_replicates: int,
    random_seed: int | None,
    alpha_upper: float,
    beta_upper: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    from scipy.optimize import dual_annealing, minimize

    bounds = [(0.0, alpha_upper)] * alpha_dim + [(0.0, beta_upper)] * beta_dim
    scaled = data * scale

    def sse(x, target):
        return float(np.sum((observable(x[:alpha_dim], x[alpha_dim:]) - target) ** 2))

    point = dual_annealing(sse, bounds=bounds, args=(scaled,), seed=random_seed, maxiter=300)
    x0 = np.asarray(point.x, dtype=float)
    fitted = raw_observable(x0[:alpha_dim], x0[alpha_dim:])  # в отсчётах
    n_steps = len(data) // alpha_dim
    resid = (data - fitted).reshape(alpha_dim, n_steps)
    mask = used.reshape(alpha_dim, n_steps)
    variance_g = np.array(
        [r[m].var(ddof=min(1, int(m.sum()) - 1)) if m.sum() > 1 else 0.0 for r, m in zip(resid, mask, strict=True)]
    )
    variance = np.repeat(variance_g, n_steps)

    rng = np.random.default_rng(random_seed)
    samples: list[np.ndarray] = []
    failed = 0
    for _ in range(n_replicates):
        synthetic = negative_binomial_noise(fitted, variance, rng) * scale
        res = minimize(sse, x0, args=(synthetic,), method="Nelder-Mead", bounds=bounds,
                       options={"maxfev": 400 * len(x0), "xatol": 1e-4, "fatol": 1e-8})
        if not np.all(np.isfinite(res.x)):
            failed += 1
            continue
        samples.append(np.asarray(res.x, dtype=float))
    arr = np.asarray(samples, dtype=float).reshape(-1, alpha_dim + beta_dim)
    ss_tot = float(np.sum((scaled - scaled.mean()) ** 2))
    diagnostics = {
        "sampler": "parametric bootstrap (negative binomial noise)",
        "replicates": n_replicates,
        "failed_refits": failed,
        "point_fit_r2": 1.0 - float(point.fun) / ss_tot if ss_tot > 0 else None,
        "residual_variance": [float(v) for v in variance_g],
        "residual_sd": [float(np.sqrt(v)) for v in variance_g],
        "parameter_sd": [float(v) for v in arr.std(axis=0)] if len(arr) else None,
        "r_hat_max": None,
        "ess_bulk_min": None,
    }
    return x0, arr[:, :alpha_dim], arr[:, alpha_dim:], diagnostics


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
    "negative_binomial_noise",
    "r2_score_np",
]
