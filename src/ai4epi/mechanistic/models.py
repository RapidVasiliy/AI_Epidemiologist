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
    """

    alpha: Sequence[float]
    beta: Sequence[float]
    population_size: int
    initial_infectious: Sequence[float] = field(default_factory=lambda: [100.0])

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
    """Две возрастные группы (0–14, 15+), перенос из Influenza_bulletin как есть.

    Внимание (известные ограничения исходной реализации, сохранены намеренно,
    чтобы не менять ранее полученные результаты):

    * обе группы стартуют с ``S_0 = alpha_j * rho`` от полной популяции;
    * в слагаемом ``sum_i beta[2i+j] S^j I / rho`` используется ``I`` своей же
      группы ``j``, поэтому межгрупповой передачи фактически нет и
      ``beta[j]`` и ``beta[2+j]`` неразличимы.

    В бюллетене используется только ``TotalBRModel``.
    """

    groups = 2
    alpha_dim = 2
    beta_dim = 4
    name = "age"

    def simulate(self, params: ModelParams, duration_days: int) -> SimulationResult:
        rho = float(params.population_size)
        out_newly, out_inf, out_s = [], [], []
        for j in range(self.groups):
            s0 = int(float(params.alpha[j]) * rho)
            i0 = float(params.initial_infectious[j])
            newly = np.zeros(duration_days)
            infected = np.zeros(duration_days)
            susceptible = np.zeros(duration_days)
            newly[0] = i0
            infected[0] = i0
            susceptible[0] = s0
            for day in range(duration_days - 1):
                infected[day] = min(self._infected(newly, day), rho)
                force = sum(
                    float(params.beta[2 * i + j]) * susceptible[day] * infected[day] / rho
                    for i in range(self.groups)
                )
                newly[day + 1] = min(force, susceptible[day])
                susceptible[day + 1] = susceptible[day] - newly[day + 1]
            out_newly.append(newly)
            out_inf.append(infected)
            out_s.append(susceptible)
        return SimulationResult(np.vstack(out_newly), np.vstack(out_inf), np.vstack(out_s))


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
