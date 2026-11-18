"""Калибровка многоштаммовой модели Барояна–Рвачёва: исходная постановка и её изменения.

python run_experiments.py msk 2017 2018 --out results/msk.json
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
from msfast import Problem, normalize_exposed
from scipy.optimize import differential_evolution, minimize

HERE = Path(__file__).resolve().parent
BOUNDS = [(0.1, 0.4)] * 3 + [(0.06, 0.12), (0.07, 0.13), (0.06, 0.13)] + [(0.05, 0.5)]  # как в v4b
MU_B = (0.8, 0.995)
CONFIGS = {
    "статья": {},
    "+μ оценивается": {"fit_mu": True},
    "+свой сдвиг штамма": {"align": "per_strain"},
    "все изменения": {"calendar": True, "fit_mu": True, "align": "per_strain", "share_ab": False},
}
CITY_FILES = {c: HERE / f"data/incidence_strains_{c}.csv" for c in ("msk", "spb", "nsk", "oms", "vor", "krs")}


def make(csv, year, cfg):
    fit_mu = cfg.get("fit_mu", False)
    p = Problem(csv, year, calendar=cfg.get("calendar", False), share_ab=cfg.get("share_ab", True),
                align=cfg.get("align", "dominant"))
    bounds = BOUNDS + ([MU_B] if fit_mu else [])

    def unpack(k):
        ex4, kk = normalize_exposed(k[:7])
        return ex4, kk[3:6], kk[6], (k[7] if fit_mu else None)

    def obj(k):
        ex4, lam, a, mu = unpack(k)
        return p.fit_values(ex4, lam, a, mu).sum()

    return p, bounds, unpack, obj


def report(p, unpack, k):
    ex4, lam, a, mu = unpack(k)
    dist, (aligned, deltas, _) = p.fit_values(ex4, lam, a, mu, return_curves=True)
    peaks = np.nanmax(np.where(p.mask, p.x, np.nan), axis=1)
    return {"r2_weighted": p.r2(dist).round(3).tolist(), "r2": p.unweighted_r2(aligned).round(3).tolist(),
                "exposed": np.round(ex4[:3], 3).tolist(), "lam": np.round(lam, 4).tolist(), "a": round(float(a), 3),
                "mu": round(float(mu if mu is not None else p.mu), 4), "deltas": [int(d) for d in deltas],
                "obj": float(dist.sum()), "curves": aligned.tolist(), "data": np.where(p.mask, p.x, np.nan).tolist(),
                "weeks": [f"{int(a) % 100}.{int(b)}" for a, b in zip(p.season["Год"], p.season["Неделя"], strict=False)],
                "data_peaks": peaks.round(0).tolist(), "pop": p.pop}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("city", choices=sorted(CITY_FILES))
    ap.add_argument("years", type=int, nargs="+")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    out = {}
    for year in args.years:
        for name, cfg in CONFIGS.items():
            p, bounds, unpack, obj = make(CITY_FILES[args.city], year, cfg)
            t = time.time()
            runs = [report(p, unpack, differential_evolution(obj, bounds, seed=s, popsize=20, maxiter=300, tol=1e-8).x)
                    for s in range(args.seeds)]
            best = min(runs, key=lambda r: r["obj"])
            best["seeds"] = [{k: r[k] for k in ("r2_weighted", "exposed", "lam", "a", "mu")} for r in runs]
            out[f"{year}|{name}"] = best
            print(args.city, year, f"{name:20s} R²w={best['r2_weighted']} μ={best['mu']} ({time.time() - t:.0f} c)", flush=True)
        p, bounds, unpack, obj = make(CITY_FILES[args.city], year, {})
        for seed in range(args.seeds):  # исходная процедура: 1000 случайных точек + L-BFGS-B
            rng = np.random.default_rng(seed)
            pts = rng.uniform([b[0] for b in bounds], [b[1] for b in bounds], size=(1000, len(bounds)))
            x0 = pts[np.argmin([obj(x) for x in pts])]
            out[f"{year}|исходная процедура {seed}"] = report(p, unpack, minimize(obj, x0, method="L-BFGS-B", bounds=bounds).x)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    main()
