"""Многоштаммовая модель: порт v2 и эквивалентность v2 ↔ v4b (research/multistrain_calibration)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("numba")
RESEARCH = Path(__file__).resolve().parents[1] / "research" / "multistrain_calibration"
sys.path.insert(0, str(RESEARCH))

from brmodel import lam_v4b_to_v2, simulate_v2  # noqa: E402
from msfast import Problem, normalize_exposed, simulate  # noqa: E402


def test_v2_and_v4b_are_equivalent_after_reparametrization():
    N, mu = 12_263_854.0, 0.93
    k = np.r_[0.3, 0.2, 0.25, 0.085, 0.08, 0.09, 0.2]
    ex4, _ = normalize_exposed(k)
    y4 = simulate(ex4, k[3:6], k[6], N * (1 - mu) + 3, 1.0, 1800, True)
    y2 = simulate_v2(ex4.reshape(1, 4), lam_v4b_to_v2(k[3:6], mu, N), k[6], np.array([[6.528]]), np.array([N]), mu,
                     np.zeros(3, np.int64), 1800, True)[0]
    np.testing.assert_allclose(y4.max(axis=1), y2.max(axis=1), rtol=1e-4)


def test_saved_moscow_parameters_reproduce_paper_r2():
    v = np.loadtxt(RESEARCH / "legacy_v4b/out/K_out_Moscow_2017.txt")
    ex4, _ = normalize_exposed(np.r_[v[5:8], v[8:11], v[4]])
    p = Problem(RESEARCH / "data/incidence_strains_msk.csv", 2017)
    np.testing.assert_allclose(p.r2(p.fit_values(ex4, v[8:11], v[4])), v[1:4], atol=2e-3)


def test_b_can_have_own_susceptible_pool():
    ex4 = np.array([0.2, 0.2, 0.2, 0.4])
    lam = np.array([0.09, 0.09, 0.09])
    shared = simulate(ex4, lam, 0.2, 1e6, 1.0, 600, True)
    separate = simulate(ex4, lam, 0.2, 1e6, 1.0, 600, False)
    # без перекрёстной защиты A–B волна B крупнее
    assert separate[2].sum() > shared[2].sum()
