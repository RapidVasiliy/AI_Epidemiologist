from __future__ import annotations

import numpy as np
import pandas as pd

from ai4epi.analysis.age_group_forecast import add_cross_group_features
from ai4epi.data.age_groups import build_age_group_weekly
from ai4epi.data.local_excel import load_local_influenza


def test_age_group_series_split_total_flu_cases(data_dir):
    bundle = load_local_influenza("spb", data_dir=data_dir, end_date="2019-03-11")
    groups = build_age_group_weekly(bundle.cases)
    assert set(groups) == {"age_0_14", "age_15_plus"}
    total = bundle.weekly.set_index("datetime")["total_cases_formula"]
    summed = sum(g.set_index("datetime")["total_cases_formula"] for g in groups.values())
    # доли групп в ОРВИ 0–14 и 15+ дают 100 %, поэтому сумма = общей оценке (с точностью до округления)
    np.testing.assert_allclose(summed.reindex(total.index).to_numpy(), total.to_numpy(), atol=2.0)
    children = groups["age_0_14"].set_index("datetime")
    assert children["inc_per_10k"].max() > groups["age_15_plus"]["inc_per_10k"].max()
    assert {"datetime", "iso_year", "iso_week", "total_population", "total_cases_formula", "inc_per_10k"} <= set(children.reset_index().columns)


def test_cross_group_features_are_lagged():
    dates = pd.date_range("2020-01-06", periods=5, freq="7D")
    a = pd.DataFrame({"datetime": dates, "inc_per_10k": [1.0, 2, 3, 4, 5]})
    b = pd.DataFrame({"datetime": dates, "inc_per_10k": [10.0, 20, 30, 40, 50]})
    out = add_cross_group_features(a, {"b": b}, lags=(0, 2))
    assert out["xg_b_lag0"].tolist() == [10, 20, 30, 40, 50]
    assert np.isnan(out["xg_b_lag2"].iloc[1]) and out["xg_b_lag2"].iloc[4] == 30
