"""Неопределённость матрицы контактов: калибровка возрастной модели БР на возмущённых матрицах."""
import json, sys, warnings
from functools import lru_cache
from multiprocessing import Pool
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")

CASES = [("spb", "2019-01-21"), ("spb", "2019-03-11"), ("spb", "2024-01-29"), ("chelyabinsk", "2023-01-16"), ("spb", "2018-02-12")]
M0 = np.array([[8.55577786, 5.87935504], [0.55266746, 4.65762732]])
K = 40
SIGMAS = [0.25, 0.5]


@lru_cache(maxsize=None)
def setup(city, origin):
    from ai4epi.data.local_excel import load_local_influenza
    from ai4epi.data.age_groups import build_age_group_weekly
    from ai4epi.analysis.mechanistic_wave import MechanisticWaveConfig, run_mechanistic_wave_analysis
    b = load_local_influenza(city)
    age = build_age_group_weekly(b.cases)
    r = run_mechanistic_wave_analysis(b.weekly, origin_date=origin, config=MechanisticWaveConfig(method="annealing", random_seed=0))
    start = pd.Timestamp(r.context_block["calibration_start_date"])
    pops = np.array([float(age[c].set_index("datetime")["total_population"].loc[pd.Timestamp(origin)]) for c in age])
    return age, start, pops


def matrices(pops):
    rng = np.random.default_rng(12345)
    out = [("base", 0.0, M0)]
    c = M0 * pops[:, None]
    out.append(("reciprocal", 0.0, ((c + c.T) / 2) / pops[:, None]))
    for s in SIGMAS:
        for k in range(K):
            out.append((f"s{s}_{k}", s, M0 * np.exp(rng.normal(0, s, (2, 2)))))
    return out


def job(args):
    city, origin, name, sigma, m = args
    from ai4epi.analysis.mechanistic_wave import MechanisticWaveConfig, run_age_group_mechanistic
    from ai4epi.data.age_groups import age_group_labels
    age, start, _ = setup(city, origin)
    cfg = MechanisticWaveConfig(method="annealing", random_seed=0, age_groups=True, age_contact_matrix=tuple(map(tuple, m)))
    block, traj, cal, _ = run_age_group_mechanistic(age, calibration_start=start, origin=pd.Timestamp(origin),
                                                   labels=age_group_labels(), ml_forecasts={}, config=cfg)
    out = {"city": city, "origin": origin, "name": name, "sigma": sigma, "m": m.tolist(),
           "fit_r2_joint": cal.diagnostics.get("best_r2"), "r2": [g["fit_r2_in_sample"] for g in block["groups"]],
           "alpha": block["alpha"], "R_t": block["reproduction_number_current"], "lead": block["children_peak_lead_weeks"],
           "peaks": [g["peak"]["date"] for g in block["groups"]], "peak_vals": [g["peak"]["value"] for g in block["groups"]],
           "proj": {}, "obs": {}}
    for code, f in age.items():
        p = traj[(traj.age_group_code == code) & traj.is_projection].head(4)
        o = f.set_index("datetime")["inc_per_10k"].reindex(p["datetime"])
        out["proj"][code] = p["best"].round(3).tolist(); out["obs"][code] = o.round(3).tolist()
    return out


if __name__ == "__main__":
    jobs = []
    for city, origin in CASES:
        _, _, pops = setup(city, origin)
        jobs += [(city, origin, n, s, m) for n, s, m in matrices(pops)]
    with Pool(4) as pool, open(sys.argv[1], "w") as fh:
        for i, r in enumerate(pool.imap_unordered(job, jobs, chunksize=4)):
            fh.write(json.dumps(r) + "\n"); fh.flush()
