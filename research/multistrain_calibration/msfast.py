"""Быстрый порт BaroyanMultistrain_v4b (BR_model_strains + FindModelFit) для экспериментов.

Исходная модель — legacy_v4b/BR_model_strains.py (В. Н. Леоненко); совпадение порта проверяется
скриптом check_port.py (суточные ряды до 1e-11, R² из legacy_v4b/out/K_out_Moscow_*.txt).

Модель и целевая функция воспроизводят исходный код один в один (проверяется в check_port.py);
варианты включаются флагами:
  share_ab=False  — у гриппа B свой запас восприимчивых (нет перекрёстной защиты A↔B в сезоне);
  align='per_strain' — у каждого штамма свой сдвиг по времени (свой момент старта);
  calendar=True   — ряд по календарным неделям, недели с пропусками не удаляются, а исключаются из невязки.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numba import njit

Q = np.array([0.0, 0.0, 1, 0.9, 0.55, 0.3, 0.15, 0.05])
DELTA_CONT = 6.528
STRAINS = ["A(H1N1)pdm09", "A(H3N2)", "B"]
EPID_THRESHOLD = 100


@njit(cache=True)
def simulate(exposed, lam, a, rho, i0, n_days, share_ab):
    """exposed: 4 доли (H1, H3, B, наивные); возвращает y[3, n_days+1]."""
    y = np.zeros((3, n_days + 1))
    T = len(Q)
    # два набора восприимчивых: для A-штаммов и для B (если share_ab, используется только первый)
    xa = np.zeros(4)
    xb = np.zeros(4)
    for h in range(4):
        xa[h] = exposed[h] * rho
        xb[h] = exposed[h] * rho
    for m in range(3):
        y[m, 0] = i0
    inf = np.zeros(3)
    quiet = 0
    for t in range(n_days):
        for m in range(3):
            s = 0.0
            for tau in range(T):
                if t - tau >= 0:
                    s += y[m, t - tau] * Q[tau]
            inf[m] = s
        for h in range(4):
            if share_ab:
                ftot = 0.0
                fm = np.zeros(3)
                for m in range(3):
                    f = a if h == m else 1.0
                    fm[m] = lam[m] * DELTA_CONT * inf[m] * f / rho
                    ftot += fm[m]
                real = min(ftot, 1.0) * xa[h]
                xa[h] -= real
                if ftot > 0:
                    for m in range(3):
                        y[m, t + 1] += real * fm[m] / ftot
            else:
                # A-штаммы делят xa, B — свой xb
                ftot = 0.0
                fm = np.zeros(2)
                for m in range(2):
                    f = a if h == m else 1.0
                    fm[m] = lam[m] * DELTA_CONT * inf[m] * f / rho
                    ftot += fm[m]
                real = min(ftot, 1.0) * xa[h]
                xa[h] -= real
                if ftot > 0:
                    for m in range(2):
                        y[m, t + 1] += real * fm[m] / ftot
                fb = lam[2] * DELTA_CONT * inf[2] * (a if h == 2 else 1.0) / rho
                realb = min(fb, 1.0) * xb[h]
                xb[h] -= realb
                y[2, t + 1] += realb
        # ранняя остановка: всё затухло (дальнейшие значения ~0)
        if t > 60 and inf.max() < 1e-9 and y[:, t + 1].max() < 1e-9:
            quiet += 1
            if quiet > 10:
                break
    return y


def weekly(y_daily: np.ndarray) -> np.ndarray:
    """Как в setWeeklyInc: к суточному ряду (N+1) добавляется 1800 нулей, суммы по 7 дней."""
    pad = np.zeros((y_daily.shape[0], y_daily.shape[1] + 1800))
    pad[:, : y_daily.shape[1]] = y_daily
    wks = pad.shape[1] // 7
    return pad[:, : wks * 7].reshape(pad.shape[0], wks, 7).sum(axis=2)


def normalize_exposed(k):
    k = np.array(k, dtype=float)
    s = k[0] + k[1] + k[2]
    if s <= 1:
        naive = 1 - s
    else:
        naive = 0.0
        k[:3] = k[:3] / s
    return np.array([k[0], k[1], k[2], naive]), k


class Problem:
    """Данные сезона + целевая функция FindModelFit."""

    def __init__(self, csv_path, year, mu=0.9, sigma=1.5, calendar=False, share_ab=True,
                 align="dominant", n_days=1800, i0=1.0, drop_weeks=()):
        d = pd.read_csv(csv_path, index_col=0)
        self.pop = float(d.dropna()[d.dropna()["Год"] == year]["Население"].iloc[0])
        season = d[((d["Год"] == year) & (d["Неделя"] >= 40)) | ((d["Год"] == year + 1) & (d["Неделя"] <= 24))]
        for (yy, ww) in drop_weeks:
            season = season[~((season["Год"] == yy) & (season["Неделя"] == ww))]
        if not calendar:
            season = season.dropna()
        self.season = season.reset_index(drop=True)
        self.x = self.season[STRAINS].to_numpy(float).T  # (3, n), NaN — исключается из невязки
        self.mask = np.isfinite(self.x)
        self.mu, self.sigma, self.share_ab, self.align = mu, sigma, share_ab, align
        self.n_days, self.i0 = n_days, i0
        n = self.x.shape[1]
        xz = np.where(self.mask, self.x, -np.inf)
        self.peak_real = xz.argmax(axis=1)
        self.main = int(np.nanmax(self.x, axis=1).argmax())
        idx = np.arange(n)
        self.w = np.array([sigma ** (-np.abs(p - idx)) for p in self.peak_real])
        self.w = np.where(self.mask, self.w, 0.0)
        xm = np.where(self.mask, self.x, 0.0)
        mean = np.array([self.x[s][self.mask[s]].mean() for s in range(3)])
        self.res2 = (self.w * (xm - mean[:, None]) ** 2).sum(axis=1)

    def rho(self, mu=None):
        mu = self.mu if mu is None else mu
        return self.pop * (1 - mu) + 3 * self.i0

    def simulate_weekly(self, exposed4, lam, a, mu=None):
        y = simulate(np.asarray(exposed4, float), np.asarray(lam, float), float(a), self.rho(mu), self.i0,
                     self.n_days, self.share_ab)
        return y, weekly(y)

    def fit_values(self, exposed4, lam, a, mu=None, return_curves=False):
        y, yw = self.simulate_weekly(exposed4, lam, a, mu)
        if y.max() == self.i0:  # "No epi dynamics"
            return (np.full(3, 1e12), None) if return_curves else np.full(3, 1e12)
        peak_model = yw.argmax(axis=1)
        peak_val = yw.max(axis=1)
        if self.align == "per_strain":
            deltas = peak_model - self.peak_real
        else:
            if peak_val[self.main] > EPID_THRESHOLD:
                dlt = peak_model[self.main] - self.peak_real[self.main]
            else:
                big = int(peak_val.argmax())
                dlt = peak_model[big] - self.peak_real[big]
            deltas = np.full(3, dlt)
        n = self.x.shape[1]
        dist = np.zeros(3)
        aligned = np.zeros((3, n))
        for s in range(3):
            idx = np.arange(n) + deltas[s]  # как в исходном коде: отрицательные индексы «заворачиваются»
            yy = yw[s][idx]
            aligned[s] = yy
            r = np.where(self.mask[s], self.x[s] - yy, 0.0)
            dist[s] = (self.w[s] * r ** 2).sum()
        if return_curves:
            return dist, (aligned, deltas, yw)
        return dist

    def r2(self, dist):
        return 1 - dist / self.res2

    def unweighted_r2(self, aligned):
        out = []
        for s in range(3):
            m = self.mask[s]
            x, yy = self.x[s][m], aligned[s][m]
            out.append(1 - ((x - yy) ** 2).sum() / ((x - x.mean()) ** 2).sum())
        return np.array(out)
