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

# Матрица контактов 0–14 / 15+ из RSCF_Uncertainty (BaroyanAgeMultistrain_v2,
# ``contact_matrix_2_age_groups.p``): строка — заражаемая группа, столбец — источник,
# как в ``BR_model_new.BRModel`` (сила инфекции группы i: Σ_j λ·M[i][j]·I_j / ρ).
RSCF_CONTACT_MATRIX_2: tuple[tuple[float, float], ...] = (
    (8.55577786, 5.87935504),
    (0.55266746, 4.65762732),
)


def normalized_contact_matrix(matrix: Sequence[Sequence[float]]) -> np.ndarray:
    """Матрица, делённая на спектральный радиус (масштаб задаётся отдельным множителем)."""

    m = np.asarray(matrix, dtype=float)
    radius = float(np.max(np.abs(np.linalg.eigvals(m))))
    if not radius > 0:
        raise ValueError("Спектральный радиус матрицы контактов должен быть положительным.")
    return m / radius


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

    def contact(self, params: ModelParams) -> np.ndarray:
        """Матрица интенсивностей передачи B (groups × groups)."""

        return np.asarray(params.beta, dtype=float).reshape(self.groups, self.groups)

    def reproduction_number(self, params: ModelParams, sim: SimulationResult) -> np.ndarray:
        """Эффективное репродуктивное число R_t по суткам.

        Один заражённый группы i за время болезни создаёт силу инфекции B[j, i]·g_τ/ρ
        для каждого восприимчивого группы j, поэтому матрица следующего поколения
        K_t[j, i] = S^j_t · B[j, i] · Σg / ρ, а R_t — её спектральный радиус
        (для одной группы R_t = β · S_t · Σg / ρ). Насыщение min(·, S) не учитывается.
        """

        b = self.contact(params)
        s = np.asarray(sim.susceptible, dtype=float)
        k = float(self.g.sum()) / float(params.population_size)
        out = np.empty(s.shape[1])
        for day in range(s.shape[1]):
            ngm = s[:, day][:, None] * b * k
            out[day] = float(np.max(np.abs(np.linalg.eigvals(ngm)))) if ngm.shape[0] > 1 else float(ngm[0, 0])
        return out

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

    Если задана внешняя ``contact_matrix`` M (например, ``RSCF_CONTACT_MATRIX_2``), её структура
    фиксируется: B = beta[0] · M / r(M), где r(M) — спектральный радиус, и калибруется один
    масштабный множитель вместо четырёх коэффициентов.

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

    def __init__(
        self,
        infectivity_profile: Sequence[float] = DEFAULT_INFECTIVITY_PROFILE,
        contact_matrix: Sequence[Sequence[float]] | None = None,
    ) -> None:
        super().__init__(infectivity_profile)
        self.contact_structure = None if contact_matrix is None else normalized_contact_matrix(contact_matrix)
        if self.contact_structure is not None:
            if self.contact_structure.shape != (self.groups, self.groups):
                raise ValueError("Матрица контактов должна быть 2×2.")
            self.beta_dim = 1

    def contact(self, params: ModelParams) -> np.ndarray:
        if self.contact_structure is not None:
            return float(params.beta[0]) * self.contact_structure
        return super().contact(params)

    def simulate(self, params: ModelParams, duration_days: int) -> SimulationResult:
        if params.group_population_sizes is None or len(params.group_population_sizes) != self.groups:
            raise ValueError("AgeGroupBRModel требует group_population_sizes для двух групп.")
        rho_g = np.asarray(params.group_population_sizes, dtype=float)
        rho = float(rho_g.sum())
        alpha = np.asarray(params.alpha, dtype=float)
        contact = self.contact(params)
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


# Матрица всех контактов по 5-летним группам из Ajelli & Litvinova (2017), Supplementary File S5
# (опрос в Томске, январь–февраль 2016): элемент [i][j] — среднее число контактов участника
# группы i с людьми группы j в сутки; группы 0–4, 5–9, ..., 50–54, 55+.
AJELLI_LITVINOVA_AGE_BINS: tuple[tuple[int, int | None], ...] = tuple(
    [(5 * k, 5 * k + 4) for k in range(11)] + [(55, None)]
)
AJELLI_LITVINOVA_ALL_CONTACTS: tuple[tuple[float, ...], ...] = (
    (0.23, 0.77, 0.08, 0.23, 0.08, 0.38, 1, 0.85, 0, 0, 0.15, 0.92),
    (0.54, 7.76, 1.06, 0.19, 0.09, 0.24, 0.96, 1.18, 0.77, 0.24, 0.16, 0.65),
    (0.22, 0.66, 8.18, 0.5, 0.17, 0.29, 0.61, 1.1, 0.94, 0.54, 0.28, 0.88),
    (0.14, 0.26, 0.61, 8.29, 0.37, 0.35, 0.42, 0.93, 1.09, 0.65, 0.37, 0.66),
    (0, 0.09, 0.09, 0.21, 3.79, 1.83, 1.23, 1, 0.75, 0.9, 0.73, 0.85),
    (0.64, 0.93, 0.29, 0.11, 0.29, 1.05, 2.13, 1.38, 1.05, 0.58, 1.05, 1.72),
    (0.56, 0.65, 0.51, 0.19, 0.34, 1.07, 1.82, 0.96, 0.78, 0.6, 0.4, 1.17),
    (0.31, 0.55, 1.55, 1.18, 0.47, 0.94, 1.11, 1.52, 1.18, 0.74, 0.62, 1.12),
    (0.08, 0.39, 0.81, 0.69, 0.58, 1.05, 1.12, 1.63, 2.17, 0.92, 0.89, 1.98),
    (0.05, 0.16, 0.74, 0.89, 0.32, 0.26, 0.36, 1.12, 1.55, 0.88, 0.76, 1.28),
    (0, 0.22, 0.22, 0.56, 0.44, 0, 0.58, 0.83, 1.22, 1.59, 1, 2.33),
    (0, 0.27, 0.45, 0.18, 0.36, 0.44, 0.64, 0.47, 0.8, 0.56, 0.71, 2.29),
)


# Матрица 0–6 / 7–14 / 15+, агрегированная авторами (M. Litvinova, письмо, ноябрь 2026) по данным
# того же опроса; группы 0–2 и 3–6 объединены из-за малой выборки дошкольников.
AJELLI_LITVINOVA_3_GROUPS: tuple[tuple[float, float, float], ...] = (
    (0.23, 0.85, 3.62),
    (0.53, 8.66, 4.97),
    (0.36, 1.14, 9.76),
)


class FineAgeContactBRModel(BaroyanRvachevModel):
    """Возрастная модель БР с силой инфекции по однородным возрастным группам.

    Как советуют авторы матрицы (Ajelli, Litvinova): сила инфекции задаётся матрицей с
    одинаковыми по ширине группами (5 лет) и численностью этих групп, а выход модели
    суммируется до наблюдаемых групп (0–14 / 15+). Агрегирование самой матрицы до
    неоднородных групп смещает доминирующее собственное значение к самой большой группе.

    Для внутренней группы k (наблюдаемая группа g(k))::

        rho_k     = kappa_g(k) · N_k,  S^k_0 = floor(alpha_g(k) · rho_k)
        I^k_t     = min( sum_tau j^k_{t-tau} g_tau , rho_k )
        j^k_{t+1} = min( S^k_t · beta · sum_m M[k, m] · I^m_t / rho_m , S^k_t )

    — как в SIR-модели статьи (``beta · M_ij · I_j / N_j``), с учётом недоучёта: в модели
    группа m содержит регистрируемую долю kappa заражённых, истинное число I_m / kappa_m,
    численность N_m, поэтому I_m / rho_m = (I_m / kappa_m) / N_m.
    Калибруются alpha по наблюдаемым группам и один множитель beta.
    """

    alpha_dim = 2
    beta_dim = 1
    name = "age_fine"

    def __init__(
        self,
        internal_rho: Sequence[float],
        group_index: Sequence[int],
        contact_matrix: Sequence[Sequence[float]] = AJELLI_LITVINOVA_ALL_CONTACTS,
        infectivity_profile: Sequence[float] = DEFAULT_INFECTIVITY_PROFILE,
    ) -> None:
        super().__init__(infectivity_profile)
        self.matrix = np.asarray(contact_matrix, dtype=float)
        self.internal_rho = np.asarray(internal_rho, dtype=float)
        self.group_index = np.asarray(group_index, dtype=int)
        n = len(self.internal_rho)
        if self.matrix.shape != (n, n) or len(self.group_index) != n:
            raise ValueError("Размеры матрицы, численностей и отображения групп не совпадают.")
        self.groups = int(self.group_index.max()) + 1
        self.alpha_dim = self.groups
        self._agg = np.zeros((self.groups, n))
        self._agg[self.group_index, np.arange(n)] = 1.0

    def contact(self, params: ModelParams) -> np.ndarray:
        return float(params.beta[0]) * self.matrix

    def simulate(self, params: ModelParams, duration_days: int) -> SimulationResult:
        rho = self.internal_rho
        alpha = np.asarray(params.alpha, dtype=float)[self.group_index]
        b = self.contact(params) / rho[None, :]
        i0_g = np.asarray(params.initial_infectious, dtype=float)
        share = rho / (self._agg @ rho)[self.group_index]
        i0 = i0_g[self.group_index] * share
        n = len(rho)
        newly = np.zeros((n, duration_days))
        susceptible = np.zeros((n, duration_days))
        infected = np.zeros((n, duration_days))
        newly[:, 0] = i0
        susceptible[:, 0] = np.floor(alpha * rho)
        g = self.g
        for day in range(duration_days - 1):
            lo = max(0, day - len(g) + 1)
            window = newly[:, lo : day + 1][:, ::-1]
            infected[:, day] = np.minimum(window @ g[: window.shape[1]], rho)
            force = b @ infected[:, day]
            newly[:, day + 1] = np.minimum(susceptible[:, day] * force, susceptible[:, day])
            susceptible[:, day + 1] = susceptible[:, day] - newly[:, day + 1]
        self._last_internal = SimulationResult(newly, infected, susceptible)
        return SimulationResult(self._agg @ newly, self._agg @ infected, self._agg @ susceptible)

    def reproduction_number(self, params: ModelParams, sim: SimulationResult) -> np.ndarray:
        """R_t по внутренним группам: K_t[k, m] = S^k_t · beta · M[k, m] · Σg / rho_m."""

        self.simulate(params, sim.newly_infected.shape[1])  # внутренние группы
        internal = self._last_internal
        b = self.contact(params) / self.internal_rho[None, :] * float(self.g.sum())
        s = internal.susceptible
        return np.array([float(np.max(np.abs(np.linalg.eigvals(s[:, d][:, None] * b)))) for d in range(s.shape[1])])


def fine_age_populations(
    pop_0_2: float, pop_3_6: float, pop_7_14: float, pop_15_64: float, pop_65_plus: float
) -> np.ndarray:
    """Численности 5-летних групп (0–4, …, 50–54, 55+) по группам выгрузки.

    Внутри групп выгрузки возраст распределён равномерно по годам (0–4 = 0–2 + 2/4 · 3–6,
    5–9 = 2/4 · 3–6 + 3/8 · 7–14, 10–14 = 5/8 · 7–14; 15–64 — 50 одинаковых лет); 55+ =
    10/50 · (15–64) + 65+. Это приближение: возрастные «ямы» взрослых не учитываются.
    """

    adult_year = pop_15_64 / 50.0
    out = [pop_0_2 + 0.5 * pop_3_6, 0.5 * pop_3_6 + 3 / 8 * pop_7_14, 5 / 8 * pop_7_14]
    out += [5 * adult_year] * 8
    out += [10 * adult_year + pop_65_plus]
    return np.asarray(out, dtype=float)


def get_model(kind: str, *, contact_matrix: Sequence[Sequence[float]] | None = None) -> BaroyanRvachevModel:
    """``"total"`` или ``"age"`` -> экземпляр модели (``contact_matrix`` — только для ``"age"``)."""

    match kind:
        case "total":
            return TotalBRModel()
        case "age":
            return AgeGroupBRModel(contact_matrix=contact_matrix)
    raise ValueError(f"Неизвестный тип модели Барояна–Рвачёва: {kind!r} (ожидается 'total' или 'age').")


__all__ = [
    "AJELLI_LITVINOVA_3_GROUPS",
    "AJELLI_LITVINOVA_AGE_BINS",
    "AJELLI_LITVINOVA_ALL_CONTACTS",
    "DEFAULT_INFECTIVITY_PROFILE",
    "RSCF_CONTACT_MATRIX_2",
    "AgeGroupBRModel",
    "BaroyanRvachevModel",
    "FineAgeContactBRModel",
    "ModelParams",
    "SimulationResult",
    "TotalBRModel",
    "fine_age_populations",
    "get_model",
    "normalized_contact_matrix",
]
