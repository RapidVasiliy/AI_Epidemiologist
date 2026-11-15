from __future__ import annotations

import numpy as np
import pandas as pd

from ai4epi.data.age_groups import build_age_group_weekly
from ai4epi.data.age_pcr import attach_age_pcr, load_age_pcr, season_positivity
from ai4epi.data.local_excel import load_local_cases


def _cases(n: int = 6) -> pd.DataFrame:
    dates = pd.date_range("2018-01-01", periods=n, freq="7D")
    df = pd.DataFrame({"datetime": dates, "sars_total_cases": 1000.0})
    for k, (ari, pop) in enumerate([(200, 1e4), (200, 1e4), (100, 2e4), (500, 1e5)]):
        df[f"sars_cases_age_group_{k}"] = float(ari)
        df[f"population_age_group_{k}"] = pop
    df["total_population"] = 1.4e5
    for i in range(4):
        df[f"rel_strain_{i}"] = 0.0
        df[f"real_cases_strain_{i}"] = 0.0
    df["real_cases_strain_1"] = 200.0  # оценка гриппа по всем возрастам
    return df


def _age_pcr(dates, positives) -> pd.DataFrame:
    table = pd.DataFrame(index=pd.DatetimeIndex(dates, name="datetime"))
    for g, pos in zip(["0_2", "3_6", "7_14", "15"], positives, strict=True):
        table[f"tested_{g}"] = 100.0
        table[f"positive_{g}"] = float(pos)
    table["tested_total"] = 400.0
    table["positive_total"] = float(sum(positives))
    return table


def test_without_age_pcr_shares_follow_ari():
    groups = build_age_group_weekly(_cases())
    kids = groups["age_0_14"]["total_cases_formula"].to_numpy()
    adults = groups["age_15_plus"]["total_cases_formula"].to_numpy()
    np.testing.assert_allclose(kids, 200 * 500 / 1000)
    np.testing.assert_allclose(adults, 200 * 500 / 1000)
    assert "age_pcr_used" not in groups["age_0_14"]


def test_age_pcr_redistributes_but_keeps_total():
    cases = _cases()
    pcr = _age_pcr(cases["datetime"].iloc[:3], positives=[5, 5, 5, 40])  # у взрослых доля выше
    groups = build_age_group_weekly(attach_age_pcr(cases, pcr))
    kids = groups["age_0_14"].set_index("datetime")
    adults = groups["age_15_plus"].set_index("datetime")
    total = kids["total_cases_formula"] + adults["total_cases_formula"]
    np.testing.assert_allclose(total.to_numpy(), 200.0, atol=1.0)
    assert kids["age_pcr_used"].tolist() == [True] * 3 + [False] * 3
    # недели с возрастной ПЦР: взрослым достаётся больше гриппа, чем по доле ОРВИ
    assert (adults["total_cases_formula"].iloc[:3] > 150).all()
    np.testing.assert_allclose(adults["total_cases_formula"].iloc[3:], 100.0)
    no_pcr = build_age_group_weekly(attach_age_pcr(cases, pcr), use_age_pcr=False)
    np.testing.assert_allclose(no_pcr["age_15_plus"]["total_cases_formula"], 100.0)


def test_small_samples_are_ignored():
    cases = _cases()
    pcr = _age_pcr(cases["datetime"].iloc[:2], positives=[0, 0, 0, 1])
    for g in ["0_2", "3_6", "7_14", "15"]:
        pcr[f"tested_{g}"] = 2.0
    groups = build_age_group_weekly(attach_age_pcr(cases, pcr), min_week_tests=20)
    assert not groups["age_0_14"]["age_pcr_used"].any()


def test_national_export_for_spb(data_dir):
    pcr = load_age_pcr("spb", path=data_dir.parent / "national" / "age_pcr_2017w40_2018w39.csv")
    assert pcr is not None and len(pcr) == 52
    share = season_positivity(pcr)
    assert share["0_2"] < share["15"]
    cases, _ = load_local_cases("spb", data_dir=data_dir, end_date="2018-06-01")
    assert cases["age_pcr_tested_3"].notna().sum() > 30
    assert load_age_pcr("unknown_city") is None
