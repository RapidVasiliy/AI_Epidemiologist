"""Проверка: brmodel.simulate_v2 совпадает с исходным BRModel (v2) и эквивалентна модели v4b."""
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
# Исходный BR_model_new несовместим с numpy >= 2: total_pop_size = sum(rho) — массив формы (1,),
# и присваивание x[i, h, t + 1] -= ... падает. Для сверки загружаем исходник с единственной правкой.
_src = (HERE / "reference_v2" / "BR_model_new.py").read_text(encoding="utf-8")
_src = _src.replace("total_pop_size = sum(rho)", "total_pop_size = float(np.sum(rho))")
_ns: dict = {}
exec(compile(_src, "BR_model_new.py", "exec"), _ns)
BRModel = _ns["BRModel"]

from brmodel import lam_v4b_to_v2, simulate_v2  # noqa: E402
from msfast import normalize_exposed, simulate  # noqa: E402

rng = np.random.default_rng(0)
M2 = np.array([[8.55577786, 5.87935504], [0.55266746, 4.65762732]])
cases = {
    "total": {"groups": ["total"], "strains": ["generic"], "H": 2, "M": [[6.528]], "pop": [1_000_000.0], "lam": lambda: [rng.uniform(0.1, 0.4)]},
    "age-group": {"groups": ["0-14", "15+"], "strains": ["generic"], "H": 2, "M": M2, "pop": [540_000.0, 4_040_000.0], "lam": lambda: [rng.uniform(0.2, 0.5)]},
    "strain": {"groups": ["total"], "strains": ["A(H1N1)pdm09", "A(H3N2)", "B"], "H": 4, "M": [[6.528]], "pop": [2_000_000.0], "lam": lambda: list(rng.uniform(0.8, 1.3, 3))},
    "strain_age-group": {"groups": ["0-14", "15+"], "strains": ["A(H1N1)pdm09", "A(H3N2)", "B"], "H": 4, "M": M2, "pop": [540_000.0, 4_040_000.0], "lam": lambda: list(rng.uniform(0.8, 1.3, 3))},
}
for inc, c in cases.items():
    mu = 0.2 if inc in ("total", "age-group") else 0.9
    exposed = rng.dirichlet(np.ones(c["H"]), size=len(c["groups"]))
    lam, a = c["lam"](), float(rng.uniform(0.05, 0.5))
    m = BRModel(c["M"], c["pop"], mu, inc, list(c["groups"]), list(c["strains"]))
    m.N = 400
    m.set_attributes()
    m.init_simul_params([list(e) for e in exposed] if len(c["groups"]) > 1 else list(exposed[0]), lam, a)
    y_ref = m.make_simulation()[0]
    shift = np.array([0, 10, 0]) if len(c["strains"]) == 3 else np.array([0])
    y = simulate_v2(exposed, np.array(lam, float), a, np.array(c["M"], float), np.array(c["pop"]), mu, shift, 400,
                    inc in ("strain", "strain_age-group"))
    print(f"{inc:17s} max |разница| = {np.abs(y_ref - y).max():.2e} при масштабе {y_ref.max():.0f}")

# эквивалентность v4b ↔ v2 в режиме штаммов (без сдвига заражения)
N, mu = 12_263_854.0, 0.93
k = np.r_[0.3, 0.2, 0.25, 0.085, 0.08, 0.09, 0.2]
ex4, _ = normalize_exposed(k)
y4 = simulate(ex4, k[3:6], k[6], N * (1 - mu) + 3, 1.0, 1800, True)
y2 = simulate_v2(ex4.reshape(1, 4), lam_v4b_to_v2(k[3:6], mu, N), k[6], np.array([[6.528]]), np.array([N]), mu,
                 np.zeros(3, np.int64), 1800, True)[0]
print(f"v4b ↔ v2 (λ пересчитаны): max относительная разница пиков = {np.abs(y4.max(1) - y2.max(1)).max() / y4.max():.2e}")
