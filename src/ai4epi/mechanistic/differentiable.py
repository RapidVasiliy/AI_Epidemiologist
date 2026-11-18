"""
Дифференцируемая (PyTensor) реализация моделей обновления для градиентной калибровки (NUTS).

Повторяет ``simulate`` моделей ``TotalBRModel`` и ``FineAgeContactBRModel`` в гладкой форме
(``incidence="exponential"``): без ``min``/``floor``, поэтому у наблюдаемого ряда есть градиент по
параметрам. Для внутренних групп k::

    I^k_t     = Σ_τ g_τ · j^k_{t−τ}
    λ^k_t     = β · Σ_m M[k, m] · I^m_t / ρ_m
    j^k_{t+1} = S^k_t · (1 − exp(−λ^k_t)),   S^k_{t+1} = S^k_t − j^k_{t+1},   S^k_0 = α_{g(k)} · ρ_k

Общая модель — частный случай с одной группой и M = [[1]]. Сверка с numpy-версией —
``tests/test_differentiable.py``.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def weekly_observable_tensor(
    alpha,
    beta,
    *,
    profile: Sequence[float],
    rho: Sequence[float],
    group_index: Sequence[int],
    matrix: Sequence[Sequence[float]],
    initial_infectious: Sequence[float],
    n_weeks: int,
):
    """Недельные новые случаи по наблюдаемым группам, группы подряд (как ``model_observable``)."""

    import pytensor
    import pytensor.tensor as pt

    g = np.asarray(profile, dtype=float)
    rho_k = np.asarray(rho, dtype=float)
    idx = np.asarray(group_index, dtype=int)
    m = np.asarray(matrix, dtype=float)
    n, n_groups, lags = len(rho_k), int(idx.max()) + 1, len(g)
    agg = np.zeros((n_groups, n))
    agg[idx, np.arange(n)] = 1.0
    share = rho_k / (agg @ rho_k)[idx]
    i0 = np.asarray(initial_infectious, dtype=float)[idx] * share
    b_norm = pt.as_tensor_variable(m / rho_k[None, :])

    alpha_k = pt.as_tensor_variable(alpha)[idx]
    s0 = alpha_k * rho_k
    hist0 = pt.as_tensor_variable(np.vstack([i0[None, :], np.zeros((lags - 1, n))]))
    g_t = pt.as_tensor_variable(g)
    beta_s = pt.as_tensor_variable(beta).reshape((-1,))[0]

    # параметры передаются явно (non_sequences): PyMC должен видеть их в графе, чтобы заменить
    # случайные величины на значения при вычислении logp
    def step(hist, s, beta_v, g_v, b_v):
        infected = pt.dot(g_v, hist)
        force = beta_v * pt.dot(b_v, infected)
        new = s * (1.0 - pt.exp(-force))
        return pt.concatenate([new[None, :], hist[:-1]], axis=0), s - new, new

    n_days = n_weeks * 7
    (_, _, news), _ = pytensor.scan(
        step, outputs_info=[hist0, s0, None], non_sequences=[beta_s, g_t, b_norm], n_steps=n_days - 1
    )
    daily = pt.concatenate([pt.as_tensor_variable(i0)[None, :], news], axis=0)  # (days, n)
    weekly = daily.reshape((n_weeks, 7, n)).sum(axis=1)  # (weeks, n)
    by_group = pt.dot(weekly, agg.T)  # (weeks, groups)
    return by_group.T.reshape((-1,))


__all__ = ["weekly_observable_tensor"]
