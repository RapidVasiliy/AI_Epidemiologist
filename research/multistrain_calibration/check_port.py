"""Проверка порта: суточные ряды и R² совпадают с исходным кодом v4b."""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "legacy_v4b"))
import BR_model_strains as BR  # noqa: E402
from msfast import Problem, Q, normalize_exposed, simulate  # noqa: E402

rng = np.random.default_rng(1)
for _ in range(5):
    k = np.r_[rng.uniform(0.1, 0.4, 3), rng.uniform(0.06, 0.13, 3), rng.uniform(0.05, 0.5)]
    ex4, _ = normalize_exposed(k)
    rho = 12_263_854 * 0.1 + 3
    m = BR.BR_model.init(list(Q), rho, np.ones(3), 1800)
    m.initSimulParams(list(ex4), list(k[3:6]), k[6])
    yo = np.array(m.MakeSimulation())
    yf = simulate(ex4, k[3:6], k[6], rho, 1.0, 1800, True)
    print(f"max |разница| суточных рядов = {np.abs(yo - yf).max():.2e} при масштабе {yo.max():.0f}")
for year in (2017, 2018):
    v = np.loadtxt(HERE / f"legacy_v4b/out/K_out_Moscow_{year}.txt")
    ex4, _ = normalize_exposed(np.r_[v[5:8], v[8:11], v[4]])
    p = Problem(HERE / "data/incidence_strains_msk.csv", year)
    print(year, "R² из K_out:", np.round(v[1:4], 3), "порт:", np.round(p.r2(p.fit_values(ex4, v[8:11], v[4])), 3))
