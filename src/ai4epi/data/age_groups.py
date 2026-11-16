"""
Недельные ряды расчётной заболеваемости гриппом по возрастным группам.

Для группы g на неделе t::

    flu_cases_g(t) = flu_cases(t) * ARI_g(t) / ARI_total(t)
    inc_per_10k_g(t) = flu_cases_g(t) / N_g(t) * 10_000

где ``flu_cases`` — оценка числа случаев гриппа по всем возрастам (доля
положительных проб по штаммам × обращения с ОРВИ, ``weekly_incidence_from_cases``).
По умолчанию предполагается, что доля гриппа среди ОРВИ одинакова во всех
возрастных группах — то же допущение, что в режиме ``age`` Influenza_bulletin.

Если в таблице есть ПЦР по возрастам (столбцы ``age_pcr_tested_k`` /
``age_pcr_positive_k``, k = 0..3: 0–2, 3–6, 7–14, 15+; см. ``ai4epi.data.age_pcr``),
на неделях с достаточным числом проб общая оценка гриппа распределяется по
группам пропорционально ``ARI_k · p_k``::

    p_k = (positive_k + m · p_week) / (tested_k + m)       # сжатие к общей доле недели
    flu_cases_g(t) = flu_cases(t) · Σ_{k∈g} ARI_k p_k / Σ_{k=0..3} ARI_k p_k

Сумма по группам равна общей оценке (общий ряд и возрастные согласованы), меняется
только распределение между возрастами. Недели без возрастной ПЦР — по допущению
одинаковой доли; признак ``age_pcr_used`` в ряду группы.

Группы по умолчанию — 0–14 и 15+ лет, как в возрастной модели
Барояна–Рвачёва (``AgeGroupBRModel``). Ряд каждой группы имеет тот же
контракт, что и общий недельный ряд (``datetime``, ``iso_year``,
``iso_week``, ``total_population``, ``total_cases_formula``, ``inc_per_10k``),
поэтому к нему применимы и ML-прогноз, и модель Барояна–Рвачёва.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ai4epi.data.influenza import weekly_incidence_from_cases


@dataclass(frozen=True)
class AgeGroupSplit:
    """Возрастная группа как сумма исходных столбцов ``sars_cases_age_group_k``."""

    code: str
    label_ru: str
    source_indices: tuple[int, ...]


DEFAULT_AGE_SPLIT: tuple[AgeGroupSplit, ...] = (
    AgeGroupSplit("age_0_14", "0–14 лет", (0, 1, 2)),
    AgeGroupSplit("age_15_plus", "15 лет и старше", (3,)),
)


class AgeGroupDataError(ValueError):
    """Таблица cases не позволяет построить возрастные ряды."""


AGE_PCR_SOURCE_INDICES: tuple[int, ...] = (0, 1, 2, 3)


def age_positivity_weights(
    df: pd.DataFrame,
    *,
    prior_tests: float = 10.0,
    min_week_tests: float = 20.0,
) -> pd.DataFrame | None:
    """Веса ``ARI_k · p_k`` (k = 0..3) по неделям с возрастной ПЦР; NaN — недели без неё.

    ``prior_tests`` — сила сжатия доли положительных группы к общей доле недели (в «пробах»);
    недели, где по всем группам протестировано меньше ``min_week_tests``, не используются.
    """

    tested_cols = [f"age_pcr_tested_{k}" for k in AGE_PCR_SOURCE_INDICES]
    positive_cols = [f"age_pcr_positive_{k}" for k in AGE_PCR_SOURCE_INDICES]
    if not set(tested_cols + positive_cols) <= set(df.columns):
        return None
    tested = df[tested_cols].to_numpy(float)
    positive = df[positive_cols].to_numpy(float)
    week_tested = tested.sum(axis=1)
    usable = np.isfinite(week_tested) & (week_tested >= min_week_tests)
    with np.errstate(invalid="ignore", divide="ignore"):
        p_week = positive.sum(axis=1) / week_tested
        p = (positive + prior_tests * p_week[:, None]) / (tested + prior_tests)
    ari = df[[f"sars_cases_age_group_{k}" for k in AGE_PCR_SOURCE_INDICES]].to_numpy(float)
    weights = ari * p
    usable &= np.isfinite(weights).all(axis=1) & (weights.sum(axis=1) > 0)
    weights[~usable] = np.nan
    return pd.DataFrame(weights, index=df.index, columns=list(AGE_PCR_SOURCE_INDICES))


def build_age_group_weekly(
    cases: pd.DataFrame,
    *,
    groups: tuple[AgeGroupSplit, ...] = DEFAULT_AGE_SPLIT,
    use_age_pcr: bool = True,
    prior_tests: float = 10.0,
    min_week_tests: float = 20.0,
) -> dict[str, pd.DataFrame]:
    """Построить недельные ряды по возрастным группам из таблицы ``cases``.

    ``use_age_pcr=False`` — всегда допущение одинаковой доли гриппа среди ОРВИ.
    """

    df = cases.copy()
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)
    no_15_plus = "sars_cases_age_group_3" not in df or df["sars_cases_age_group_3"].isna().all()
    if no_15_plus and {"sars_cases_age_group_4", "sars_cases_age_group_5"} <= set(df.columns):
        df["sars_cases_age_group_3"] = df["sars_cases_age_group_4"] + df["sars_cases_age_group_5"]
        df["population_age_group_3"] = df["population_age_group_4"] + df["population_age_group_5"]

    total = weekly_incidence_from_cases(df).set_index("datetime")
    ari_total = df.set_index("datetime")["sars_total_cases"]
    weights = age_positivity_weights(df, prior_tests=prior_tests, min_week_tests=min_week_tests) if use_age_pcr else None
    if weights is not None:
        weights.index = pd.DatetimeIndex(df["datetime"])
        age_used = weights.notna().all(axis=1)
        weight_sum = weights.sum(axis=1, min_count=len(AGE_PCR_SOURCE_INDICES))
    out: dict[str, pd.DataFrame] = {}
    for group in groups:
        case_cols = [f"sars_cases_age_group_{k}" for k in group.source_indices]
        pop_cols = [f"population_age_group_{k}" for k in group.source_indices]
        missing = [c for c in case_cols + pop_cols if c not in df.columns]
        if missing:
            raise AgeGroupDataError(f"Для группы {group.code} отсутствуют столбцы {missing}.")
        indexed = df.set_index("datetime")
        ari_g = indexed[case_cols].sum(axis=1, min_count=len(case_cols))
        pop_g = indexed[pop_cols].sum(axis=1, min_count=len(pop_cols))
        share = (ari_g / ari_total).where(ari_total > 0)
        if weights is not None and set(group.source_indices) <= set(AGE_PCR_SOURCE_INDICES):
            age_share = weights[list(group.source_indices)].sum(axis=1) / weight_sum
            share = age_share.where(age_used, share)
        frame = pd.DataFrame(index=total.index)
        frame["iso_year"] = total["iso_year"]
        frame["iso_week"] = total["iso_week"]
        frame["total_population"] = pop_g.reindex(total.index)
        frame["total_cases_formula"] = (total["total_cases_formula"] * share.reindex(total.index)).round()
        frame["inc_per_10k"] = frame["total_cases_formula"] / frame["total_population"] * 10_000
        # численности исходных групп (0–2, 3–6, 7–14, 15+, 15–64, 65+) — для модели с 5-летними группами
        for k in range(6):
            col = f"population_age_group_{k}"
            if col in indexed.columns:
                frame[f"source_{col}"] = indexed[col].reindex(total.index)
        if weights is not None:
            frame["age_pcr_used"] = age_used.reindex(total.index).fillna(False).astype(bool)
        if "pcr_imputed" in indexed.columns:
            frame["pcr_imputed"] = indexed["pcr_imputed"].reindex(total.index).fillna(False).astype(bool)
        frame = frame.reset_index().dropna(subset=["inc_per_10k", "total_population"]).reset_index(drop=True)
        frame["iso_year"] = frame["iso_year"].astype(int)
        frame["iso_week"] = frame["iso_week"].astype(int)
        out[group.code] = frame
    return out


def age_group_labels(groups: tuple[AgeGroupSplit, ...] = DEFAULT_AGE_SPLIT) -> dict[str, str]:
    return {g.code: g.label_ru for g in groups}


__all__ = [
    "DEFAULT_AGE_SPLIT",
    "AgeGroupDataError",
    "AgeGroupSplit",
    "age_group_labels",
    "age_positivity_weights",
    "build_age_group_weekly",
]
