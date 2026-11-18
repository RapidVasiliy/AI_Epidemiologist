import json, sys, glob, os
import numpy as np, pandas as pd
from ai4epi.analysis.br_conformal import evaluate_leave_one_out
K = "/tmp/claude-0/-home-user/4f06e2ce-f8b8-5274-beb2-32b879306b77/scratchpad/kern"
BASE = "/tmp/claude-0/-home-user/4f06e2ce-f8b8-5274-beb2-32b879306b77/scratchpad/btc/out"
seasons = [l.split() for l in open(sys.argv[1])]
variants = {"br_mcmc": BASE, **{v: f"{K}/out_{v}" for v in sys.argv[2:]}}
frames = {}
for name, root in variants.items():
    parts = []
    for city, y in seasons:
        f = f"{root}/{city}_{y}/predictions.csv"
        if os.path.exists(f):
            d = pd.read_csv(f, parse_dates=["origin", "target_date"]); d["season"] = f"{city}_{y}"; parts.append(d)
    frames[name] = pd.concat(parts)
key = ["season", "origin", "horizon"]
wave = ["growth", "peak", "decline"]
common = None
for name, d in frames.items():
    k = d[d.br.notna() & d.y_true.notna() & d.phase.isin(wave)][key]
    common = k if common is None else common.merge(k, on=key)
print("common wave points:", len(common), "seasons:", common.season.nunique())
rows = []
for name, d in frames.items():
    c = d.merge(common, on=key)
    o = c.drop_duplicates(["season", "origin"])
    err = (c.br - c.y_true).abs()
    loo = evaluate_leave_one_out(c, group_col="season")
    rows.append({"variant": name, "MAE": err.mean(), **{f"MAE_{p}": err[c.phase == p].mean() for p in wave},
                 "MAE_h4": err[c.horizon == 4].mean(), "fit_R2_med": o.br_fit_r2.median(),
                 "rhat_med": o.br_rhat.median(), "rhat_ok": (o.br_rhat <= 1.05).mean(),
                 "conf_cov": loo.inside.mean(), "conf_width": loo.width.mean()})
print(pd.DataFrame(rows).set_index("variant").round(3).to_string())
per = {name: (d.merge(common, on=key).assign(e=lambda x: (x.br - x.y_true).abs()).groupby("season").e.mean()) for name, d in frames.items()}
print(pd.DataFrame(per).round(2).to_string())
