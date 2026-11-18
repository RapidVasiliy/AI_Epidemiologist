"""Быстрая реализация модели Барояна–Рвачёва из RSCF_Uncertainty/BaroyanAgeMultistrain_v2 (BR_model_new.BRModel).

Возрастные группы × штаммы × история заражения, матрица контактов M, доля защищённых mu,
сдвиги заражения штаммов (strain_shift, в днях). Совпадение с исходным кодом проверяется
check_port.py для режимов 'total', 'age-group', 'strain', 'strain_age-group'.

Нормировка как в v2: x0 = exposed · (1 − mu) · rho_i, сила инфекции делится на всё население
sum(rho_i) (защищённые участвуют в контактах, но не заражаются). Модель v4b (RJNAMM 2021) в режиме
'strain' эквивалентна этой при lam_v2 = lam_v4b · sum(rho) / ((1 − mu)·N + n_strains).
"""

from __future__ import annotations

import numpy as np
from numba import njit

Q = np.array([0.0, 0.0, 1, 0.9, 0.55, 0.3, 0.15, 0.05])


@njit(cache=True)
def simulate_v2(exposed, lam, a, M, pop, mu, strain_shift, n_days, seed_by_shift):
    """exposed: (G, H) доли по истории заражения; lam: (S,); M: (G, G); pop: (G,).

    Возвращает y (G, S, n_days + 1) — новые заражения по дням.
    История h < S — переболевшие штаммом h в прошлом сезоне (восприимчивость к нему a),
    h = H − 1 — не болевшие. Для режимов без штаммов (S = 1, H = 2) h = 0 — «переболевшие».
    """
    G, H = exposed.shape
    S = lam.shape[0]
    T = Q.shape[0]
    y = np.zeros((G, S, n_days + 1))
    for i in range(G):
        for s in range(S):
            if seed_by_shift:
                y[i, s, strain_shift[s]] = 1.0
            else:
                y[i, s, 0] = 1.0
    rho = np.zeros(G)
    for i in range(G):
        rho[i] = pop[i] - S  # I0 = 1 на каждый штамм
    total = rho.sum()
    x = np.zeros((G, H))
    for i in range(G):
        for h in range(H):
            x[i, h] = exposed[i, h] * (1.0 - mu) * rho[i]
    cum = np.zeros((G, S))
    fm = np.zeros(S)
    for t in range(n_days):
        for j in range(G):
            for s in range(S):
                acc = 0.0
                for tau in range(T):
                    if t - tau >= 0:
                        acc += y[j, s, t - tau] * Q[tau]
                cum[j, s] = acc
        for i in range(G):
            for h in range(H):
                ftot = 0.0
                for m in range(S):
                    f = a if h == m else 1.0
                    force = 0.0
                    for j in range(G):
                        force += lam[m] * M[i, j] * cum[j, m] * f / total
                    fm[m] = force
                    ftot += force
                real = min(ftot, 1.0) * x[i, h]
                x[i, h] -= real
                if ftot > 0:
                    for m in range(S):
                        y[i, m, t + 1] += fm[m] / ftot * real
    return y


def lam_v4b_to_v2(lam_v4b, mu, pop_total, n_strains=3):
    """Пересчёт интенсивности из параметризации v4b (RJNAMM 2021) в v2."""
    return np.asarray(lam_v4b) * (pop_total - n_strains) / ((1 - mu) * pop_total + n_strains)
