"""Графики к docs/backtest.md по результатам ``ai4epi backtest-local``.

Использование::

    python scripts/plot_backtest_figures.py <каталог с подкаталогами бэктестов> docs/figures

Каждый подкаталог — вывод одной команды ``backtest-local`` (predictions.csv).
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
WAVE = {"growth", "peak", "decline"}
SEASON_LABELS = {
    "spb_2016-05-02": "СПб 2016/17",
    "spb_2018-05-07": "СПб 2018/19",
    "spb_2023-05-01": "СПб 2023/24",
    "chelyabinsk_2018-05-07": "Челябинск 2018/19",
    "chelyabinsk_2023-05-01": "Челябинск 2023/24",
}


def _style(ax) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _bar_labels(ax, bars, fmt="{:.2f}") -> None:
    for b in bars:
        h = b.get_height()
        ax.annotate(fmt.format(h), (b.get_x() + b.get_width() / 2, h), xytext=(0, 3),
                    textcoords="offset points", ha="center", fontsize=8, color=INK)


def load(root: Path) -> pd.DataFrame:
    frames = []
    for path in sorted(root.glob("*/predictions.csv")):
        df = pd.read_csv(path, parse_dates=["origin", "target_date"])
        df["run"] = path.parent.name
        frames.append(df)
    data = pd.concat(frames, ignore_index=True)
    ml = "ml_hgb_fast" if "ml_hgb_fast" in data else "ml_hgb"
    data["ml"] = data["ml_hgb"] if "ml_hgb" in data else data[ml]
    return data[data["ml"].notna() & data["y_true"].notna()]


def fig_timeline(p: pd.DataFrame, out: Path, run: str = "spb_2018-05-07", horizon: int = 2) -> None:
    d = p[(p["run"] == run) & (p["horizon"] == horizon)].sort_values("target_date")
    fig, axes = plt.subplots(1, 2, figsize=(13, 4), sharey=True)
    for ax, (lo, hi, title) in zip(
        axes,
        (("ml_lo", "ml_hi", "Абсолютный конформный интервал (было)"),
         ("ml_lo_level", "ml_hi_level", "Интервал, масштабированный по уровню (стало)")),
        strict=True,
    ):
        inside = (d["y_true"] >= d[lo]) & (d["y_true"] <= d[hi])
        ax.fill_between(d["target_date"], d[lo], d[hi], color=BLUE, alpha=0.18, linewidth=0, label="80 % интервал ML")
        ax.plot(d["target_date"], d["ml"], color=BLUE, linewidth=2, label="прогноз ML")
        ax.plot(d["target_date"], d["y_true"], color=INK, linewidth=1.2, label="факт")
        miss = d[~inside & d[lo].notna()]
        ax.scatter(miss["target_date"], miss["y_true"], s=36, facecolor="white", edgecolor=ORANGE, linewidth=1.8,
                   zorder=5, label="факт вне интервала")
        cov = inside[d[lo].notna()].mean()
        ax.set_title(f"{title}\nпокрытие {cov:.0%} при номинале 80 %", fontsize=10, color=INK, loc="left")
        _style(ax)
    axes[0].set_ylabel("заболеваемость гриппом на 10 тыс.", color=INK2)
    axes[0].legend(fontsize=8, frameon=False, loc="upper left")
    fig.suptitle(f"{SEASON_LABELS.get(run, run)}: прогноз на {horizon} нед. вперёд, еженедельное переобучение", fontsize=11,
                 color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out / "backtest_intervals_timeline.png", dpi=160)
    plt.close(fig)


def fig_coverage(p: pd.DataFrame, out: Path) -> None:
    wave = p["phase"].isin(WAVE)
    rows = []
    for lo, hi in (("ml_lo", "ml_hi"), ("ml_lo_level", "ml_hi_level")):
        ok = p[lo].notna()
        inside = (p["y_true"] >= p[lo]) & (p["y_true"] <= p[hi])
        rows.append([inside[ok & wave].mean(), inside[ok & ~wave].mean()])
    fig, ax = plt.subplots(figsize=(6.5, 4))
    x = np.arange(2)
    w = 0.36
    b1 = ax.bar(x - w / 2 - 0.01, rows[0], w, color=ORANGE, label="абсолютный (было)")
    b2 = ax.bar(x + w / 2 + 0.01, rows[1], w, color=BLUE, label="по уровню (стало)")
    _bar_labels(ax, b1)
    _bar_labels(ax, b2)
    ax.axhline(0.8, color=INK2, linestyle="--", linewidth=1)
    ax.text(-0.5, 0.81, "номинал 0.8", color=INK2, fontsize=8, ha="left", va="bottom")
    ax.set_xticks(x, ["на эпидемической волне", "вне волны"])
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("доля фактов внутри 80 % интервала", color=INK2)
    ax.set_title("Покрытие прогнозных интервалов ML, 5 сезонов", fontsize=11, color=INK, loc="left")
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    _style(ax)
    fig.tight_layout()
    fig.savefig(out / "backtest_interval_coverage.png", dpi=160)
    plt.close(fig)


def fig_agreement(p: pd.DataFrame, out: Path) -> None:
    a = p[p["agreement"].notna()]
    labels = {"agree": "модели согласны", "disagree": "модели расходятся"}
    mae, cov, n = [], [], []
    for code in ("agree", "disagree"):
        part = a[a["agreement"] == code]
        mae.append(float(np.abs(part["y_true"] - part["ml"]).mean()))
        ok = part["ml_lo"].notna()
        cov.append(float(((part["y_true"] >= part["ml_lo"]) & (part["y_true"] <= part["ml_hi"]))[ok].mean()))
        n.append(part.groupby(["run", "origin"]).ngroups)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    names = [f"{labels[c]}\n({k} нед.)" for c, k in zip(("agree", "disagree"), n, strict=True)]
    _bar_labels(axes[0], axes[0].bar(names, mae, color=BLUE, width=0.5))
    axes[0].set_title("Ошибка ML-прогноза (MAE, на 10 тыс.)", fontsize=10, color=INK, loc="left")
    _bar_labels(axes[1], axes[1].bar(names, cov, color=BLUE, width=0.5))
    axes[1].axhline(0.8, color=INK2, linestyle="--", linewidth=1)
    axes[1].set_ylim(0, 1)
    axes[1].set_title("Покрытие 80 % интервала ML (абсолютного)", fontsize=10, color=INK, loc="left")
    for ax in axes:
        _style(ax)
    fig.suptitle("Недели эпидемической волны: согласие ML и модели Барояна–Рвачёва", fontsize=11, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out / "backtest_agreement.png", dpi=160)
    plt.close(fig)


def fig_seasons(p: pd.DataFrame, out: Path) -> None:
    runs = [r for r in SEASON_LABELS if r in set(p["run"])]
    vals = {"ML (HGB)": [], "наивный": [], "климатический": []}
    for r in runs:
        d = p[p["run"] == r]
        vals["ML (HGB)"].append(float(np.abs(d["y_true"] - d["ml"]).mean()))
        vals["наивный"].append(float(np.abs(d["y_true"] - d["persistence"]).mean()))
        m = d["climatology"].notna()
        vals["климатический"].append(float(np.abs(d["y_true"] - d["climatology"])[m].mean()))
    fig, ax = plt.subplots(figsize=(11, 4))
    x = np.arange(len(runs))
    w = 0.26
    for i, (name, color) in enumerate(zip(vals, (BLUE, ORANGE, AQUA), strict=True)):
        bars = ax.bar(x + (i - 1) * (w + 0.015), vals[name], w, color=color, label=name)
        _bar_labels(ax, bars, "{:.1f}")
    ax.set_xticks(x, [SEASON_LABELS[r] for r in runs])
    ax.set_ylabel("MAE, на 10 тыс. (меньше — лучше)", color=INK2)
    ax.set_title("Ошибка прогноза на 1–4 недели: ML против базовых прогнозов", fontsize=11, color=INK, loc="left")
    ax.legend(fontsize=8, frameon=False)
    _style(ax)
    fig.tight_layout()
    fig.savefig(out / "backtest_seasons_mae.png", dpi=160)
    plt.close(fig)


def main() -> None:
    root, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    p = load(root)
    fig_timeline(p, out)
    fig_coverage(p, out)
    fig_agreement(p, out)
    fig_seasons(p, out)
    print("saved to", out)


if __name__ == "__main__":
    main()
