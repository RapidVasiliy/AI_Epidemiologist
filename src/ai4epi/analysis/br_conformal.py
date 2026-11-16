"""
Конформная полоса прогноза модели Барояна–Рвачёва по ошибкам прошлых прогнозов.

Полоса пучка траекторий модели БР (MCMC с ε = 1) накрывает будущие наблюдения лишь в 5–15 %
случаев при номинальных 90 % (бэктест СПб, ``docs/mechanistic_wave.md``): она отражает только
неопределённость параметров, а не структурную ошибку модели. Здесь полоса строится так же,
как конформные интервалы ML-модели, но по ошибкам центральной траектории БР на бэктесте:

    s = |y − br| / (y_0 + c)                      — ошибка относительно текущего уровня y_0, c = ``offset``
    q_{фаза, h} = квантиль s уровня ⌈(n + 1)·level⌉ / n
    [max(0, br − q·(y_0 + c)),  br + q·(y_0 + c)] — полоса для горизонта h

Масштаб — наблюдаемое значение текущей недели y_0 (на бэктесте СПб и Челябинска при исключении
сезона покрытие по сезонам 0.84–0.94 при номинальных 0.9). Масштаб по самому прогнозу br
неустойчив: когда модель почти обнуляет волну, а она продолжается, ошибка «взрывается»
(покрытие по сезонам от 0.38 до 1.0).

Квантили считаются отдельно для фазы роста (``growth``) и остальных фаз волны (там ошибки
различаются в разы) и для каждого горизонта; для горизонтов дальше таблицы используется
последний. Таблица строится командой ``ai4epi br-conformal-fit`` по ``predictions.csv`` из
``ai4epi backtest-local`` и сохраняется в JSON; для другого города или метода калибровки её
нужно пересчитать по бэктесту на его данных.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

DEFAULT_BR_CONFORMAL_PATH = Path("data") / "calibration" / "br_conformal.json"
PHASE_GROUPS: dict[str, str] = {"growth": "growth", "peak": "other", "decline": "other", "post_wave": "post_wave"}
WAVE_PHASES = ("growth", "peak", "decline")


def phase_group(phase: str | None) -> str:
    return PHASE_GROUPS.get(str(phase), "other")


def conformal_quantile(scores: np.ndarray, level: float) -> float:
    """Квантиль с поправкой на конечную выборку: ⌈(n + 1)·level⌉-я порядковая статистика."""

    s = np.sort(np.asarray(scores, dtype=float))
    s = s[np.isfinite(s)]
    n = len(s)
    if n == 0:
        return float("nan")
    k = math.ceil((n + 1) * level)
    return float(s[min(k, n) - 1])


def fit_br_conformal(
    predictions: pd.DataFrame,
    *,
    level: float = 0.9,
    offset: float = 1.0,
    min_points: int = 15,
    phases: Iterable[str] = WAVE_PHASES,
    sources: list[str] | None = None,
) -> dict[str, Any]:
    """Таблица квантилей по прогнозам бэктеста (столбцы ``horizon``, ``phase``, ``br``, ``y_true``).

    Используются только недели волны (``phases``). Если в группе фаз меньше ``min_points`` точек
    на горизонт, берётся общий квантиль по всем фазам волны для этого горизонта.
    """

    d = predictions.copy()
    d = d[d["br"].notna() & d["y_true"].notna() & d["phase"].isin(list(phases))]
    if d.empty:
        raise ValueError("В прогнозах нет недель волны с прогнозом модели БР и фактическим значением.")
    d["score"] = (d["y_true"] - d["br"]).abs() / (d["y_origin"].clip(lower=0) + offset)
    d["group"] = d["phase"].map(phase_group)
    horizons = sorted(int(h) for h in d["horizon"].unique())
    table: dict[str, dict[str, float]] = {}
    counts: dict[str, dict[str, int]] = {}
    for group in ("growth", "other", "all"):
        table[group], counts[group] = {}, {}
        for h in horizons:
            part = d[d["horizon"] == h] if group == "all" else d[(d["horizon"] == h) & (d["group"] == group)]
            n = len(part)
            counts[group][str(h)] = n
            if group != "all" and n < min_points:
                part = d[d["horizon"] == h]
            table[group][str(h)] = round(conformal_quantile(part["score"].to_numpy(), level), 4)
    return {
        "level": level,
        "offset": offset,
        "phases": list(phases),
        "quantiles": table,
        "counts": counts,
        "n_points": len(d),
        "sources": sources or [],
        "definition": "q — квантиль |y − br| / (y0 + offset), y0 — значение текущей недели; полоса br ± q·(y0 + offset)",
    }


def load_br_conformal(path: str | Path | None = DEFAULT_BR_CONFORMAL_PATH) -> dict[str, Any] | None:
    if path is None:
        return None
    p = Path(path)
    if not p.is_file():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def conformal_band(
    best: np.ndarray,
    start_index: int,
    table: Mapping[str, Any],
    phase: str | None,
    current_value: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Нижняя и верхняя граница для точек ``best[start_index:]`` (горизонты 1, 2, …).

    ``current_value`` — значение текущей недели (масштаб ошибки).
    """

    group = phase_group(phase)
    quantiles = table["quantiles"].get(group) or table["quantiles"]["all"]
    hs = sorted(int(h) for h in quantiles)
    offset = float(table["offset"])
    best = np.asarray(best, dtype=float)
    lo = np.full(len(best), np.nan)
    hi = np.full(len(best), np.nan)
    for j in range(start_index, len(best)):
        h = min(j - start_index + 1, hs[-1])
        q = float(quantiles[str(h)])
        width = q * (max(current_value, 0.0) + offset)
        lo[j] = max(0.0, best[j] - width)
        hi[j] = best[j] + width
    return lo, hi


def evaluate_leave_one_out(
    predictions: pd.DataFrame,
    *,
    group_col: str = "season",
    level: float = 0.9,
    offset: float = 1.0,
    min_points: int = 15,
) -> pd.DataFrame:
    """Покрытие конформной полосы, построенной без сезона (города), на этом сезоне."""

    rows = []
    d = predictions[predictions["br"].notna() & predictions["y_true"].notna() & predictions["phase"].isin(WAVE_PHASES)]
    for key, held in d.groupby(group_col):
        table = fit_br_conformal(d[d[group_col] != key], level=level, offset=offset, min_points=min_points)
        for _, r in held.iterrows():
            group = phase_group(r["phase"])
            quantiles = table["quantiles"][group]
            h = min(int(r["horizon"]), max(int(x) for x in quantiles))
            width = float(quantiles[str(h)]) * (max(float(r["y_origin"]), 0.0) + offset)
            lo, hi = max(0.0, r["br"] - width), r["br"] + width
            rows.append(
                {
                    group_col: key,
                    "phase": r["phase"],
                    "horizon": int(r["horizon"]),
                    "inside": bool(lo <= r["y_true"] <= hi),
                    "width": hi - lo,
                    "inside_model": bool(r["br_lo"] <= r["y_true"] <= r["br_hi"]) if pd.notna(r.get("br_lo")) else np.nan,
                    "width_model": (r["br_hi"] - r["br_lo"]) if pd.notna(r.get("br_lo")) else np.nan,
                }
            )
    return pd.DataFrame(rows)


__all__ = [
    "DEFAULT_BR_CONFORMAL_PATH",
    "conformal_band",
    "conformal_quantile",
    "evaluate_leave_one_out",
    "fit_br_conformal",
    "load_br_conformal",
    "phase_group",
]
