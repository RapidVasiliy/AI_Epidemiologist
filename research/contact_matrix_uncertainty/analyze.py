import json, sys
import numpy as np, pandas as pd
M0 = np.array([[8.55577786, 5.87935504], [0.55266746, 4.65762732]])
rows = [json.loads(l) for l in open(sys.argv[1])]
d = pd.DataFrame(rows)
codes = ["age_0_14", "age_15_plus"]
def mae(r, c): 
    p, o = np.array(r["proj"][c], float), np.array(r["obs"][c], float); return float(np.nanmean(np.abs(p - o)))
d["mae_k"] = [mae(r, codes[0]) for r in rows]; d["mae_a"] = [mae(r, codes[1]) for r in rows]
d["r2k"] = d.r2.str[0]; d["r2a"] = d.r2.str[1]
d["pk"] = pd.to_datetime(d.peaks.str[0]); d["pa"] = pd.to_datetime(d.peaks.str[1])
# форма матрицы: отклонения логарифмов отношений к M00
def shape(m):
    m = np.array(m); return np.log(m[0, 1] / m[0, 0]) - np.log(M0[0, 1] / M0[0, 0]), np.log(m[1, 0] / m[0, 0]) - np.log(M0[1, 0] / M0[0, 0]), np.log(m[1, 1] / m[0, 0]) - np.log(M0[1, 1] / M0[0, 0])
sh = np.array([shape(m) for m in d.m]); d["d01"], d["d10"], d["d11"] = sh.T
q = lambda s, a, b: f"{s.quantile(a):.2f}…{s.quantile(b):.2f}"
for (city, origin), g in d.groupby(["city", "origin"], sort=False):
    base = g[g.name == "base"].iloc[0]; rec = g[g.name == "reciprocal"].iloc[0]
    print(f"\n=== {city} {origin}: base R² {base.r2k:.2f}/{base.r2a:.2f} joint {base.fit_r2_joint:.3f} lead {base.lead} Rt {base.R_t} peaks {base.pk.date()} {base.pa.date()} MAE {base.mae_k:.1f}/{base.mae_a:.2f}")
    print(f"    reciprocal: R² {rec.r2k:.2f}/{rec.r2a:.2f} joint {rec.fit_r2_joint:.3f} lead {rec.lead} Rt {rec.R_t} MAE {rec.mae_k:.1f}/{rec.mae_a:.2f}")
    for s in [0.25, 0.5]:
        x = g[g.sigma == s]
        better = (x.fit_r2_joint > base.fit_r2_joint + 0.01).mean(); worse = (x.fit_r2_joint < base.fit_r2_joint - 0.05).mean()
        # полоса только от матрицы (5–95 % по матрицам) и покрытие наблюдений h1–4
        cov = []
        for c, oc in ((codes[0], "k"), (codes[1], "a")):
            P = np.array([r for r in x.proj.map(lambda p: p[c])], float); O = np.array(base.obs[c], float)
            lo, hi = np.percentile(P, 5, axis=0), np.percentile(P, 95, axis=0)
            cov.append(f"{np.mean((O >= lo) & (O <= hi)):.2f} (ширина h4 {hi[-1]-lo[-1]:.1f}, база h4 {base.proj[c][-1]:.1f})")
        top = x.nlargest(max(1, len(x) // 10), "fit_r2_joint")
        print(f"  σ={s}: R² дети {q(x.r2k,.05,.95)} взр {q(x.r2a,.05,.95)} | joint лучше базы {better:.0%}, хуже на >0.05 {worse:.0%} | "
              f"lead {sorted(x.lead.value_counts().to_dict().items())} | Rt {q(x.R_t,.05,.95)} | пик детей {x.pk.min().date()}…{x.pk.max().date()} взр {x.pa.min().date()}…{x.pa.max().date()}")
        print(f"         MAE h1–4 дети {q(x.mae_k,.05,.95)} взр {q(x.mae_a,.05,.95)} | покрытие полосой матриц: дети {cov[0]}, взр {cov[1]}")
        print(f"         лучшие 10%: сдвиг формы d01 {top.d01.mean():+.2f} d10 {top.d10.mean():+.2f} d11 {top.d11.mean():+.2f}; корр(joint, d01/d10/d11) "
              f"{x.fit_r2_joint.corr(x.d01):+.2f}/{x.fit_r2_joint.corr(x.d10):+.2f}/{x.fit_r2_joint.corr(x.d11):+.2f}")
