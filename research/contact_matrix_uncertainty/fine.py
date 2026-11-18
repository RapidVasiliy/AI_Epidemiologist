import json, sys, time, warnings; warnings.filterwarnings("ignore")
from multiprocessing import Pool
import numpy as np, pandas as pd
sys.path.insert(0, "/tmp/claude-0/-home-user/4f06e2ce-f8b8-5274-beb2-32b879306b77/scratchpad/exp/cm")
import run as R

def job(a):
    city, origin, structure, method = a
    from ai4epi.analysis.mechanistic_wave import MechanisticWaveConfig, run_age_group_mechanistic
    from ai4epi.data.age_groups import age_group_labels
    age, start, _ = R.setup(city, origin)
    seeds = range(4) if method == "annealing" else [42]
    best = None; t = time.time()
    for seed in seeds:
        cfg = MechanisticWaveConfig(method=method, random_seed=seed, age_groups=True, age_contact_structure=structure, cores=1, chains=4)
        b, traj, cal, _ = run_age_group_mechanistic(age, calibration_start=start, origin=pd.Timestamp(origin), labels=age_group_labels(), ml_forecasts={}, config=cfg)
        score = cal.diagnostics.get("best_r2", 0)
        if best is None or score > best[0]:
            best = (score, b, traj, cal)
    _, b, traj, cal = best
    mae = []
    for code, f in age.items():
        p = traj[(traj.age_group_code == code) & traj.is_projection].head(4).set_index("datetime")["best"]
        o = f.set_index("datetime")["inc_per_10k"].reindex(p.index)
        mae.append(round(float((p - o).abs().mean()), 2))
    return dict(city=city, origin=origin, structure=structure, method=method, sec=round(time.time() - t, 1),
                r2=[g["fit_r2_in_sample"] for g in b["groups"]], rhat=b["diagnostics"]["r_hat_max"],
                peaks=[g["peak"]["date"] for g in b["groups"]], lead=b["children_peak_lead_weeks"],
                Rt=b["reproduction_number_current"], alpha=b["alpha"], mae=mae)

if __name__ == "__main__":
    jobs = [(c, o, s, m) for c, o in R.CASES for s in ("authors_3g",) for m in ("annealing", "mcmc")]
    with Pool(4) as pool:
        res = pool.map(job, jobs)
    with open(sys.argv[1], "w") as fh:
        for r in res: fh.write(json.dumps(r) + "\n")
    print(pd.DataFrame(res).to_string())
