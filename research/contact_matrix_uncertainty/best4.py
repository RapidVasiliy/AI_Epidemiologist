import json, sys, warnings; warnings.filterwarnings("ignore")
from multiprocessing import Pool
import numpy as np, pandas as pd
sys.path.insert(0, sys.argv[1])
import run as R
R.K = 30

def job(a):
    city, origin, name, sigma, m = a
    from ai4epi.analysis.mechanistic_wave import MechanisticWaveConfig, run_age_group_mechanistic
    from ai4epi.data.age_groups import age_group_labels
    age, start, _ = R.setup(city, origin)
    best = None
    for seed in range(4):
        cfg = MechanisticWaveConfig(method="annealing", random_seed=seed, age_groups=True, age_contact_matrix=tuple(map(tuple, m)))
        b, traj, cal, _ = run_age_group_mechanistic(age, calibration_start=start, origin=pd.Timestamp(origin), labels=age_group_labels(), ml_forecasts={}, config=cfg)
        if best is None or cal.diagnostics["best_r2"] > best[0]:
            best = (cal.diagnostics["best_r2"], b, traj)
    r2, b, traj = best
    out = {"city": city, "origin": origin, "name": name, "sigma": sigma, "joint": r2, "r2": [g["fit_r2_in_sample"] for g in b["groups"]],
           "peaks": [g["peak"]["date"] for g in b["groups"]], "R_t": b["reproduction_number_current"], "lead": b["children_peak_lead_weeks"], "proj": {}, "obs": {}}
    for code, f in age.items():
        p = traj[(traj.age_group_code == code) & traj.is_projection].head(4)
        out["proj"][code] = p["best"].round(2).tolist(); out["obs"][code] = f.set_index("datetime")["inc_per_10k"].reindex(p["datetime"]).round(2).tolist()
    return out

if __name__ == "__main__":
    jobs = []
    for city, origin in [("spb", "2019-01-21"), ("spb", "2018-02-12")]:
        _, _, pops = R.setup(city, origin)
        jobs += [(city, origin, n, s, m) for n, s, m in R.matrices(pops)]
    with Pool(4) as pool, open(sys.argv[2], "w") as fh:
        for r in pool.imap_unordered(job, jobs, chunksize=2):
            fh.write(json.dumps(r) + "\n"); fh.flush()
