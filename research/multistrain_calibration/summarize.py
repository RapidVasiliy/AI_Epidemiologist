"""Сводная таблица и рисунки по результатам run_experiments.py.

python summarize.py   # читает results/<city>.json, пишет results/summary.md и results/fits_<city>.png
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
CITIES = {"msk": "Москва", "spb": "Санкт-Петербург", "nsk": "Новосибирск", "oms": "Омск", "vor": "Воронеж", "krs": "Краснодар"}
STRAINS = ["H1N1", "H3N2", "B"]
COL = ["#2a78d6", "#eb6834", "#1baf7a"]
INK2 = "#52514e"
# штамм считается «без волны», если его пик < 15 % от пика сезона (R² по шуму не информативен)
NO_WAVE_SHARE = 0.15


def fmt_r2(values, peaks):
    top = max(peaks)
    out = []
    for v, pk in zip(values, peaks, strict=True):
        s = f"{v:.2f}" if v > -10 else f"{v:.0f}"
        out.append(f"({s})" if pk < NO_WAVE_SHARE * top else s)
    return " / ".join(out)


def main():
    lines = ["# Калибровка по городам: взвешенный R² (H1N1 / H3N2 / B)", "",
             "В скобках — штамм без выраженной волны (пик < 15 % от пика сезона): R² по такому ряду не информативен.", "",
             "| Город | Сезон | исходная процедура (3 запуска) | та же модель, глобальный поиск | + μ оценивается | + свой сдвиг штамма | все изменения | μ (все изм.) |",
             "|---|---|---|---|---|---|---|---|"]
    for code, name in CITIES.items():
        path = RES / f"{code}.json"
        if not path.exists():
            continue
        d = json.loads(path.read_text())
        years = sorted({k.split("|")[0] for k in d})
        fig, axes = plt.subplots(len(years), 2, figsize=(13, 4.2 * len(years)), squeeze=False)
        for r, year in enumerate(years):
            peaks = d[f"{year}|статья"]["data_peaks"]
            orig = [d[f"{year}|исходная процедура {s}"]["r2_weighted"] for s in range(3) if f"{year}|исходная процедура {s}" in d]
            orig_txt = "; ".join(fmt_r2(o, peaks) for o in orig)
            row = [fmt_r2(d[f"{year}|{c}"]["r2_weighted"], peaks) for c in ("статья", "+μ оценивается", "+свой сдвиг штамма", "все изменения")]
            mus = [s["mu"] for s in d[f"{year}|все изменения"]["seeds"]]
            lines.append(f"| {name} | {year}/{int(year) + 1 - 2000} | {orig_txt} | " + " | ".join(row) + f" | {min(mus):.3f}–{max(mus):.3f} |")
            for c, cfg in enumerate(("статья", "все изменения")):
                res = d[f"{year}|{cfg}"]
                ax = axes[r, c]
                x = np.arange(len(res["weeks"]))
                for s in range(3):
                    ax.plot(x, np.array(res["data"][s], float), "o", color=COL[s], ms=4)
                    ax.plot(x, np.array(res["curves"][s]), "-", color=COL[s], lw=2, label=STRAINS[s])
                title = "та же модель, глобальный поиск, μ = 0.9" if cfg == "статья" else f"все изменения, μ = {res['mu']:.3f}"
                ax.set_title(f"{name} {year}/{int(year) + 1}: {title}\nR² = {fmt_r2(res['r2_weighted'], peaks)}", fontsize=10, loc="left")
                step = max(1, len(x) // 8)
                ax.set_xticks(x[::step], [res["weeks"][i] for i in x[::step]], fontsize=8)
                ax.spines[["top", "right"]].set_visible(False)
                ax.grid(axis="y", color="#e4e3df")
                ax.tick_params(colors=INK2)
                if c == 0:
                    ax.set_ylabel("случаев в неделю", color=INK2)
                if r == 0 and c == 0:
                    ax.legend(frameon=False, fontsize=9)
        fig.tight_layout()
        fig.savefig(RES / f"fits_{code}.png", dpi=110)
        plt.close(fig)
    (RES / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
