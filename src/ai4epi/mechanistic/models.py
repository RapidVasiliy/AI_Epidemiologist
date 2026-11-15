"""
Дискретная модель Барояна–Рвачёва (БР).

Перенесено из ``Influenza_bulletin`` (``model_complex.models``) без изменения
разностной схемы. Шаг модели — сутки:

    I_t     = min( sum_{tau=0..T} j_{t-tau} g_tau , rho )
    j_{t+1} = min( beta * S_t * I_t / rho , S_t )
    S_{t+1} = S_t - j_{t+1},      S_0 = floor(alpha * rho),  j_0 = I_0 = I0

где ``g`` — профиль относительной заразности по дням болезни, ``rho`` —
эффективная численность популяции (N с поправкой на недоучёт, см.
``ai4epi.analysis.mechanistic_wave``), ``alpha`` — доля восприимчивых в ней,
``beta`` — эффективная интенсивность передачи.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field, replace

import numpy as np

# Профиль заразности (g_0, ..., g_7) из Influenza_bulletin и статьи Лазутова и др.
DEFAULT_INFECTIVITY_PROFILE: tuple[float, ...] = (0.1, 0.1, 1.0, 0.9, 0.55, 0.3, 0.15, 0.05)


@dataclass
class ModelParams:
    """Параметры одного прогона модели.

    :param alpha: доли восприимчивых (по одной на группу)
    :param beta: интенсивности передачи (1 для total, 4 = матрица 2×2 для age)
    :param population_size: эффективная численность популяции rho
    :param initial_infectious: начальное число инфицированных (по группам)
    :param group_population_sizes: эффективные численности групп rho_j
        (только для возрастной модели; их сумма должна равняться rho)
    """

    alpha: Sequence[float]
    beta: Sequence[float]
    population_size: int
    initial_infectious: Sequence[float] = field(default_factory=lambda: [100.0])
    group_population_sizes: Sequence[float] | None = None

    def with_values(self, alpha: Sequence[float], beta: Sequence[float]) -> ModelParams:
        return replace(self, alpha=list(map(float, alpha)), beta=list(map(float, beta)))


@dataclass
class SimulationResult:
    """Суточные ряды модели; для нескольких групп — по строке на группу."""

    newly_infected: np.ndarray  # (groups, days)
    prevalence: np.ndarray  # (groups, days)
    susceptible: np.ndarray  # (groups, days)

    @property
    def groups(self) -> int:
        return int(self.newly_infected.shape[0])

    def weekly_newly_infected(self) -> np.ndarray:
        """Сумма новых заражений по 7-дневным блокам: (groups, weeks)."""

        days = self.newly_infected.shape[1]
        if days % 7:
            raise ValueError(f"Длительность моделирования {days} дней не кратна 7.")
        return self.newly_infected.reshape(self.groups, -1, 7).sum(axis=2)

    def total_weekly_newly_infected(self) -> np.ndarray:
        return self.weekly_newly_infected().sum(axis=0)


class BaroyanRvachevModel:
    """Базовый класс: профиль заразности и размерности параметров."""

    groups: int = 0
    alpha_dim: int = 0
    beta_dim: int = 0
    name: str = "base"

    def __init__(self, infectivity_profile: Sequence[float] = DEFAULT_INFECTIVITY_PROFILE) -> None:
        self.g = np.asarray(infectivity_profile, dtype=float)

    def params_dims(self) -> tuple[int, int]:
        return self.alpha_dim, self.beta_dim

    def simulate(self, params: ModelParams, duration_days: int) -> SimulationResult:  # pragma: no cover
        raise NotImplementedError

    def _infected(self, newly: np.ndarray, day: int) -> float:
        lo = max(0, day - len(self.g) + 1)
        window = newly[lo : day + 1][::-1]
        return float(np.dot(window, self.g[: len(window)]))


class TotalBRModel(BaroyanRvachevModel):
    """Модель для всей популяции: alpha и beta — по одному параметру."""

    groups = 1
    alpha_dim = 1
    beta_dim = 1
    name = "total"

    def simulate(self, params: ModelParams, duration_days: int) -> SimulationResult:
        alpha, beta = float(params.alpha[0]), float(params.beta[0])
        rho = float(params.population_size)
        i0 = float(params.initial_infectious[0])

        s0 = int(alpha * rho)
        newly = np.zeros(duration_days)
        infected = np.zeros(duration_days)
        susceptible = np.zeros(duration_days)
        newly[0] = i0
        infected[0] = i0
        susceptible[0] = s0

        for day in range(duration_days - 1):
            infected[day] = min(self._infected(newly, day), rho)
            newly[day + 1] = min(beta * susceptible[day] * infected[day] / rho, susceptible[day])
            susceptible[day + 1] = susceptible[day] - newly[day + 1]

        return SimulationResult(newly[None, :], infected[None, :], susceptible[None, :])


class AgeGroupBRModel(BaroyanRvachevModel):
    """Модель Барояна–Рвачёва для двух возрастных групп (0–14 и 15+ лет).

    Для группы j (0 — дети, 1 — взрослые)::

        rho_j   = kappa * N_j,   rho = rho_0 + rho_1,   S^j_0 = floor(alpha_j * rho_j)
        I^j_t   = min( sum_tau j^j_{t-tau} g_tau , rho_j )
        j^j_{t+1} = min( S^j_t * sum_i B[j, i] * I^i_t / rho , S^j_t )

    ``beta`` задаёт матрицу контактов B построчно: ``beta[2*j + i]`` —
    интенсивность заражения группы j от инфицированных группы i.

    Если ``alpha_0 = alpha_1 = alpha`` и все элементы B равны ``beta``, сумма
    групп совпадает с ``TotalBRModel`` (с точностью до округления S_0 вниз).

    Исправляет реализацию Influenza_bulletin, где обе группы стартовали от
    полной популяции, а сила инфекции зависела только от инфицированных той же
    группы (межгрупповой передачи не было).
    """

    groups = 2
    alpha_dim = 2
    beta_dim = 4
    name = "age"

    def simulate(self, params: ModelParams, duration_days: int) -> SimulationResult:
        if params.group_population_sizes is None or len(params.group_population_sizes) != self.groups:
            raise ValueError("AgeGroupBRModel требует group_population_sizes для двух групп.")
        rho_g = np.asarray(params.group_population_sizes, dtype=float)
        rho = float(rho_g.sum())
        alpha = np.asarray(params.alpha, dtype=float)
        contact = np.asarray(params.beta, dtype=float).reshape(self.groups, self.groups)
        i0 = np.asarray(params.initial_infectious, dtype=float)

        newly = np.zeros((self.groups, duration_days))
        infected = np.zeros((self.groups, duration_days))
        susceptible = np.zeros((self.groups, duration_days))
        newly[:, 0] = i0
        infected[:, 0] = i0
        susceptible[:, 0] = np.floor(alpha * rho_g)

        for day in range(duration_days - 1):
            for j in range(self.groups):
                infected[j, day] = min(self._infected(newly[j], day), rho_g[j])
            force = contact @ infected[:, day] / rho
            newly[:, day + 1] = np.minimum(susceptible[:, day] * force, susceptible[:, day])
            susceptible[:, day + 1] = susceptible[:, day] - newly[:, day + 1]
        return SimulationResult(newly, infected, susceptible)


def get_model(kind: str) -> BaroyanRvachevModel:
    """``"total"`` или ``"age"`` -> экземпляр модели."""

    match kind:
        case "total":
            return TotalBRModel()
        case "age":
            return AgeGroupBRModel()
    raise ValueError(f"Неизвестный тип модели Барояна–Рвачёва: {kind!r} (ожидается 'total' или 'age').")


__all__ = [
    "DEFAULT_INFECTIVITY_PROFILE",
    "AgeGroupBRModel",
    "BaroyanRvachevModel",
    "ModelParams",
    "SimulationResult",
    "TotalBRModel",
    "get_model",
]
