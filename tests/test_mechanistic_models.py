from __future__ import annotations

import numpy as np
import pytest

from ai4epi.mechanistic import (
    DEFAULT_INFECTIVITY_PROFILE,
    ModelParams,
    calibrate,
    get_model,
    model_observable,
    r2_score_np,
)


def legacy_total_simulate(alpha, beta, rho, i0, duration):
    """Дословный цикл TotalBRModel.simulate из Influenza_bulletin@6f406cc."""

    g = list(DEFAULT_INFECTIVITY_PROFILE)

    def br(day):
        return g[day] if day < len(g) else 0

    s0 = int(alpha * rho)
    total_infected = np.zeros(duration)
    newly = np.zeros(duration)
    susceptible = np.zeros(duration)
    total_infected[0] = i0
    newly[0] = i0
    susceptible[0] = s0
    for day in range(duration - 1):
        total_infected[day] = min(
            sum([newly[day - tau] * br(tau) for tau in range(len(g)) if (day - tau) >= 0]), rho
        )
        newly[day + 1] = min(beta * susceptible[day] * total_infected[day] / rho, susceptible[day])
        susceptible[day + 1] = susceptible[day] - newly[day + 1]
    return newly


@pytest.mark.parametrize("alpha,beta", [(0.3, 0.9), (0.9, 0.35), (1.0, 1.0), (0.05, 0.2)])
def test_total_model_matches_legacy_loop(alpha, beta):
    rho, i0, days = 400_000, 100.0, 210
    new = get_model("total").simulate(ModelParams([alpha], [beta], rho, [i0]), days)
    np.testing.assert_allclose(new.newly_infected[0], legacy_total_simulate(alpha, beta, rho, i0, days))


def test_simulation_invariants():
    sim = get_model("total").simulate(ModelParams([0.5], [0.8], 1_000_000, [100.0]), 280)
    s = sim.susceptible[0]
    assert np.all(sim.newly_infected >= 0)
    assert np.all(np.diff(s) <= 1e-9)
    # всё, что ушло из S, — это новые заражения начиная со второго дня
    assert s[0] - s[-1] == pytest.approx(sim.newly_infected[0, 1:].sum())
    weekly = sim.weekly_newly_infected()
    assert weekly.shape == (1, 40)
    assert weekly.sum() == pytest.approx(sim.newly_infected.sum())


def test_weekly_requires_whole_weeks():
    sim = get_model("total").simulate(ModelParams([0.5], [0.8], 10_000, [10.0]), 10)
    with pytest.raises(ValueError):
        sim.weekly_newly_infected()


def _age_params(alpha, beta, rho_g=(100_000.0, 400_000.0), i0=(60.0, 40.0)):
    return ModelParams(list(alpha), list(beta), int(sum(rho_g)), list(i0), group_population_sizes=list(rho_g))


def test_age_model_shapes_and_requires_group_sizes():
    sim = get_model("age").simulate(_age_params([0.5, 0.4], [0.3, 0.2, 0.1, 0.4]), 70)
    assert sim.weekly_newly_infected().shape == (2, 10)
    with pytest.raises(ValueError, match="group_population_sizes"):
        get_model("age").simulate(ModelParams([0.5, 0.4], [0.3] * 4, 500_000, [10.0, 10.0]), 70)


def test_age_model_reduces_to_total_model():
    """Одинаковые alpha и все элементы матрицы контактов -> сумма групп = общая модель."""

    age = get_model("age").simulate(_age_params([0.6, 0.6], [0.7] * 4), 210)
    total = get_model("total").simulate(ModelParams([0.6], [0.7], 500_000, [100.0]), 210)
    np.testing.assert_allclose(age.newly_infected.sum(axis=0), total.newly_infected[0], rtol=1e-6)


def test_age_model_has_cross_group_transmission():
    """Взрослые без собственной передачи (B[1,1]=0) заражаются только от детей."""

    no_cross = get_model("age").simulate(_age_params([0.6, 0.6], [5.0, 0.0, 0.0, 0.0]), 140)
    with_cross = get_model("age").simulate(_age_params([0.6, 0.6], [5.0, 0.0, 0.8, 0.0]), 140)
    assert no_cross.newly_infected[1, 1:].sum() == 0.0
    assert with_cross.newly_infected[1, 1:].sum() > 1000.0


def test_age_model_respects_group_susceptibles():
    sim = get_model("age").simulate(_age_params([0.3, 0.5], [3.0, 1.0, 1.0, 3.0]), 280)
    infected_total = sim.newly_infected[:, 1:].sum(axis=1)
    assert infected_total[0] <= 0.3 * 100_000 + 1e-6
    assert infected_total[1] <= 0.5 * 400_000 + 1e-6


def test_age_calibration_per_group_scale_recovers_fit():
    model = get_model("age")
    template = _age_params([0.0, 0.0], [0.0] * 4)
    truth = template.with_values([0.4, 0.3], [2.0, 0.5, 0.8, 1.2])
    observed = model_observable(model, truth, 16)
    groups = observed.reshape(2, -1)
    scale = [1.0 / groups[0].max(), 1.0 / groups[1].max()]
    result = calibrate(model, observed, template, method="annealing", observation_scale=scale, beta_upper=5.0, random_seed=0)
    fit = model_observable(model, result.best_params, 16).reshape(2, -1)
    assert r2_score_np(groups[0], fit[0]) > 0.98
    assert r2_score_np(groups[1], fit[1]) > 0.98


def test_unknown_model_kind():
    with pytest.raises(ValueError):
        get_model("seir")


@pytest.mark.parametrize("method", ["annealing", "optuna"])
def test_point_calibration_recovers_synthetic_wave(method):
    model = get_model("total")
    template = ModelParams([0.0], [0.0], 500_000, [100.0])
    observed = model_observable(model, template.with_values([0.55], [0.7]), 20)
    result = calibrate(model, observed, template, method=method, observation_scale=1e4 / 5e6, n_trials=400, random_seed=1)
    fit = model_observable(model, result.best_params, 20)
    assert r2_score_np(observed, fit) > 0.99
    assert not result.has_bundle
    assert result.diagnostics["objective"] == "r2"


@pytest.mark.slow
def test_mcmc_calibration_returns_joint_pairs_and_diagnostics():
    model = get_model("total")
    template = ModelParams([0.0], [0.0], 500_000, [100.0])
    observed = model_observable(model, template.with_values([0.55], [0.7]), 18)
    result = calibrate(
        model, observed, template, method="mcmc", observation_scale=1e4 / 5e6, epsilon=1.0,
        n_samples=50, tune=2500, draws=500, chains=4, cores=1, random_seed=3,
    )
    assert result.alpha_samples.shape == (50, 1) and result.beta_samples.shape == (50, 1)
    assert {"r_hat_max", "ess_bulk_min", "per_parameter"} <= set(result.diagnostics)
    fit = model_observable(model, result.best_params, 18)
    assert r2_score_np(observed, fit) > 0.9
    # пары берутся из совместной выборки: каждая пара — реальная точка апостериорной выборки
    params = result.sample_params()
    assert len(params) == 50 and params[0].population_size == 500_000


def test_unknown_calibration_method():
    from ai4epi.mechanistic import CalibrationError

    with pytest.raises(CalibrationError):
        calibrate(get_model("total"), np.ones(5), ModelParams([0], [0], 1000, [1.0]), method="grid")  # type: ignore[arg-type]
