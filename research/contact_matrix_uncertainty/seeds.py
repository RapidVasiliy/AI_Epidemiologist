import sys, warnings; warnings.filterwarnings("ignore")
from multiprocessing import Pool
import numpy as np, pandas as pd
sys.path.insert(0, sys.argv[1])
import run as R

def job(a):
    city, origin, seed = a
    from ai4epi.analysis.mechanistic_wave import MechanisticWaveConfig, run_age_group_mechanistic
    from ai4epi.data.age_groups import age_group_labels
    age, start, _ = R.setup(city, origin)
    cfg = MechanisticWaveConfig(method="annealing", random_seed=seed, age_groups=True)
    b, traj, cal, _ = run_age_group_mechanistic(age, calibration_start=start, origin=pd.Timestamp(origin), labels=age_group_labels(), ml_forecasts={}, config=cfg)
    p = traj[(traj.age_group_code == "age_0_14") & traj.is_projection].head(4)["best"].to_numpy()
    return city, origin, seed, round(cal.diagnostics["best_r2"], 4), b["groups"][0]["peak"]["date"], round(p[-1], 1)

if __name__ == "__main__":
    jobs = [(c, o, s) for c, o in [("spb", "2019-01-21"), ("spb", "2018-02-12")] for s in range(10)]
    with Pool(4) as pool:
        res = pool.map(job, jobs)
    df = pd.DataFrame(res, columns=["city", "origin", "seed", "joint_r2", "peak_kids", "kids_h4"])
    print(df.to_string())
    for o, g in df.groupby("origin"):
        print(o, "h4 дети 5–95%:", np.percentile(g.kids_h4, 5).round(1), np.percentile(g.kids_h4, 95).round(1), "пики", sorted(g.peak_kids.unique()))
