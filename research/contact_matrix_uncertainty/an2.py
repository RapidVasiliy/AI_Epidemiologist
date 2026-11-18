import json, sys
import numpy as np, pandas as pd
rows = [json.loads(l) for l in open(sys.argv[1])]
for origin in ["2019-01-21", "2018-02-12"]:
    g = [r for r in rows if r["origin"] == origin]
    base = next(r for r in g if r["name"] == "base"); rec = next(r for r in g if r["name"] == "reciprocal")
    def mae(r, c): return np.nanmean(np.abs(np.array(r["proj"][c]) - np.array(r["obs"][c])))
    print(f"\n== {origin}: base joint {base['joint']:.3f} R² {base['r2']} peaks {base['peaks']} Rt {base['R_t']} h4 {base['proj']['age_0_14'][-1]}/{base['proj']['age_15_plus'][-1]} MAE {mae(base,'age_0_14'):.1f}/{mae(base,'age_15_plus'):.2f}; obs h1-4 kids {base['obs']['age_0_14']}")
    print(f"   reciprocal joint {rec['joint']:.3f} R² {rec['r2']} peaks {rec['peaks']} Rt {rec['R_t']} h4 {rec['proj']['age_0_14'][-1]}/{rec['proj']['age_15_plus'][-1]} MAE {mae(rec,'age_0_14'):.1f}/{mae(rec,'age_15_plus'):.2f}")
    for s in [0.25, 0.5]:
        x = [r for r in g if r["sigma"] == s]
        J = np.array([r["joint"] for r in x]); K = np.array([r["proj"]["age_0_14"] for r in x]); A = np.array([r["proj"]["age_15_plus"] for r in x])
        Ok = np.array(base["obs"]["age_0_14"]); Oa = np.array(base["obs"]["age_15_plus"])
        pk = pd.to_datetime([r["peaks"][0] for r in x]); pa = pd.to_datetime([r["peaks"][1] for r in x]); rt = np.array([r["R_t"] for r in x])
        cov = lambda P, O: np.mean((O >= np.percentile(P, 5, 0)) & (O <= np.percentile(P, 95, 0)))
        print(f"  σ={s}: joint {np.percentile(J,5):.3f}…{np.percentile(J,95):.3f} (лучше базы на >0.01: {(J>base['joint']+0.01).mean():.0%}) | "
              f"h4 дети {np.percentile(K[:,-1],5):.0f}…{np.percentile(K[:,-1],95):.0f}, взр {np.percentile(A[:,-1],5):.1f}…{np.percentile(A[:,-1],95):.1f} | "
              f"пик детей {pk.min().date()}…{pk.max().date()} (мода {pk.to_series().mode()[0].date()}), взр {pa.min().date()}…{pa.max().date()} | Rt {np.percentile(rt,5):.2f}…{np.percentile(rt,95):.2f} | покрытие h1–4 дети {cov(K,Ok):.2f} взр {cov(A,Oa):.2f}")
