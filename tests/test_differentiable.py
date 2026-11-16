from __future__ import annotations

import numpy as np
import pytest

from ai4epi.mechanistic import AJELLI_LITVINOVA_ALL_CONTACTS, FineAgeContactBRModel, ModelParams, model_observable
from ai4epi.mechanistic.models import TotalBRModel, mean_generation_time, seir_kernel

pytensor = pytest.importorskip("pytensor")


def _evaluate(model, params, n_weeks):
    from ai4epi.mechanistic.differentiable import weekly_observable_tensor

    out = weekly_observable_tensor(np.asarray(params.alpha), np.asarray(params.beta), n_weeks=n_weeks, **model.symbolic_spec(params))
    return out.eval()


@pytest.mark.parametrize("profile", [None, "seir"])
def test_symbolic_total_model_matches_numpy(profile):
    g = seir_kernel() if profile else None
    model = TotalBRModel(g, incidence="exponential") if g is not None else TotalBRModel(incidence="exponential")
    params = ModelParams([0.6], [0.8], 500_000, [100.0])
    np.testing.assert_allclose(_evaluate(model, params, 16), model_observable(model, params, 16), rtol=1e-8, atol=1e-6)


def test_symbolic_fine_age_model_matches_numpy():
    rho = np.array([1.0, 2.0, 1.5] + [4.0] * 9) * 1e4
    model = FineAgeContactBRModel(rho, [0, 0, 0] + [1] * 9, AJELLI_LITVINOVA_ALL_CONTACTS, incidence="exponential")
    params = ModelParams([0.5, 0.4], [0.05], int(rho.sum()), [50.0, 50.0])
    np.testing.assert_allclose(_evaluate(model, params, 12), model_observable(model, params, 12), rtol=1e-8, atol=1e-6)


def test_exponential_incidence_close_to_linear_for_small_force():
    params = ModelParams([0.5], [0.6], 1_000_000, [100.0])
    lin = model_observable(TotalBRModel(), params, 20)
    exp = model_observable(TotalBRModel(incidence="exponential"), params, 20)
    assert abs(exp.max() / lin.max() - 1) < 0.05


def test_seir_kernel_properties():
    g = seir_kernel(latent_mean=1.2, infectious_mean=2.0)
    assert g.sum() == pytest.approx(2.0, rel=0.02)  # Σg = средняя длительность заразного периода
    assert mean_generation_time(g) == pytest.approx(1.2 + 2.0, abs=0.15)
