"""
Недельные ряды расчётной заболеваемости гриппом по возрастным группам.

Для группы g на неделе t::

    flu_cases_g(t) = flu_cases(t) * ARI_g(t) / ARI_total(t)
    inc_per_10k_g(t) = flu_cases_g(t) / N_g(t) * 10_000

где ``flu_cases`` — оценка числа случаев гриппа по всем возрастам (доля
положительных проб по штаммам × обращения с ОРВИ, ``weekly_incidence_from_cases``).
Предполагается, что доля гриппа среди ОРВИ одинакова во всех возрастных
группах (лабораторные данные по возрастам в выгрузках отсутствуют) — то же
допущение, что в режиме ``age`` Influenza_bulletin.

Группы по умолчанию — 0–14 и 15+ лет, как в возрастной модели
Барояна–Рвачёва (``AgeGroupBRModel``). Ряд каждой группы имеет тот же
контракт, что и общий недельный ряд (``datetime``, ``iso_year``,
``iso_week``, ``total_population``, ``total_cases_formula``, ``inc_per_10k``),
поэтому к нему применимы и ML-прогноз, и модель Барояна–Рвачёва.
"""

from __future__ import annotations

from dataclasses import dataclass

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


def build_age_group_weekly(
    cases: pd.DataFrame,
    *,
    groups: tuple[AgeGroupSplit, ...] = DEFAULT_AGE_SPLIT,
) -> dict[str, pd.DataFrame]:
    """Построить недельные ряды по возрастным группам из таблицы ``cases``."""

    df = cases.copy()
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)
    no_15_plus = "sars_cases_age_group_3" not in df or df["sars_cases_age_group_3"].isna().all()
    if no_15_plus and {"sars_cases_age_group_4", "sars_cases_age_group_5"} <= set(df.columns):
        df["sars_cases_age_group_3"] = df["sars_cases_age_group_4"] + df["sars_cases_age_group_5"]
        df["population_age_group_3"] = df["population_age_group_4"] + df["population_age_group_5"]

    total = weekly_incidence_from_cases(df).set_index("datetime")
    ari_total = df.set_index("datetime")["sars_total_cases"]
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
        frame = pd.DataFrame(index=total.index)
        frame["iso_year"] = total["iso_year"]
        frame["iso_week"] = total["iso_week"]
        frame["total_population"] = pop_g.reindex(total.index)
        frame["total_cases_formula"] = (total["total_cases_formula"] * share.reindex(total.index)).round()
        frame["inc_per_10k"] = frame["total_cases_formula"] / frame["total_population"] * 10_000
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
    "build_age_group_weekly",
]
