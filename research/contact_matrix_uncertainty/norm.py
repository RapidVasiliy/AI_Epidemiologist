import json, sys, warnings; warnings.filterwarnings("ignore")
from multiprocessing import Pool
import numpy as np, pandas as pd
sys.path.insert(0, "/tmp/claude-0/-home-user/4f06e2ce-f8b8-5274-beb2-32b879306b77/scratchpad/exp/cm")
import run as R
RSCF = np.array([[8.55577786, 5.87935504], [0.55266746, 4.65762732]])
S5_2 = np.array([[6.5, 4.467], [0.961, 7.944]])   # прямое агрегирование S5 (возраст участника — равные веса)
KAPPA = np.array([1.0, 0.1])

def variants(pops):
    rho = KAPPA * pops
    out = {}
    for mname, m in (("rscf", RSCF), ("s5", S5_2)):
        out[f"{mname}/kappa"] = m / KAPPA[None, :]            # текущая реализация
        out[f"{mname}/per_capita"] = m * (rho.sum() / rho)[None, :]  # как в SIR статьи: M_ij I_j / N_j (с учётом недоучёта)
    return out

def job(a):
    city, origin, name, m = a
    from ai4epi.analysis.mechanistic_wave import MechanisticWaveConfig, run_age_group_mechanistic
    from ai4epi.data.age_groups import age_group_labels
    age, start, _ = R.setup(city, origin)
    best = None
    for seed in range(4):
        cfg = MechanisticWaveConfig(method="annealing", random_seed=seed, age_groups=True,
                                    age_contact_matrix=tuple(map(tuple, m)), age_contact_kappa_correction=False)
        b, traj, cal, _ = run_age_group_mechanistic(age, calibration_start=start, origin=pd.Timestamp(origin), labels=age_group_labels(), ml_forecasts={}, config=cfg)
        if best is None or cal.diagnostics["best_r2"] > best[0]:
            best = (cal.diagnostics["best_r2"], b, traj)
    r2, b, traj = best
    mae = []
    for code, f in age.items():
        p = traj[(traj.age_group_code == code) & traj.is_projection].head(4).set_index("datetime")["best"]
        o = f.set_index("datetime")["inc_per_10k"].reindex(p.index)
        mae.append(round(float((p - o).abs().mean()), 2))
    return dict(city=city, origin=origin, variant=name, joint=round(r2, 3), r2=[g["fit_r2_in_sample"] for g in b["groups"]],
                peaks=[g["peak"]["date"] for g in b["groups"]], lead=b["children_peak_lead_weeks"], Rt=b["reproduction_number_current"], mae=mae)

if __name__ == "__main__":
    jobs = []
    for city, origin in R.CASES:
        _, _, pops = R.setup(city, origin)
        jobs += [(city, origin, n, m) for n, m in variants(pops).items()]
    with Pool(4) as pool:
        res = pool.map(job, jobs)
    with open(sys.argv[1], "w") as fh:
        for r in res: fh.write(json.dumps(r) + "\n")
    df = pd.DataFrame(res)
    print(df.to_string())
