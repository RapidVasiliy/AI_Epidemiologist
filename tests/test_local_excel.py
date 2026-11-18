from __future__ import annotations

import pandas as pd
import pytest

from ai4epi.data.local_excel import (
    LocalDataError,
    list_local_cities,
    load_local_influenza,
    week_start_from_label,
)


def test_week_start_from_label():
    assert week_start_from_label("2010.1 (04.01.2010 - 10.01.2010)") == pd.Timestamp("2010-01-04")
    assert week_start_from_label(pd.Timestamp("2020-03-02")) == pd.Timestamp("2020-03-02")
    with pytest.raises(LocalDataError):
        week_start_from_label("неделя без даты")


def test_cities_listed(data_dir):
    assert {"spb", "samara", "chelyabinsk"} <= set(list_local_cities(data_dir))


def test_spb_weekly_matches_legacy_formula(data_dir):
    bundle = load_local_influenza("spb", data_dir=data_dir, end_date="2019-01-21")
    weekly = bundle.weekly
    assert weekly["datetime"].is_monotonic_increasing
    assert weekly["datetime"].diff().dropna().dt.days.eq(7).all()
    last = weekly.iloc[-1]
    assert last["datetime"] == pd.Timestamp("2019-01-21")
    # Σ round(ОРВИ · доля положительных по штамму), как в model_complex.EpidData
    assert last["total_cases_formula"] == 20697
    assert last["inc_per_10k"] == pytest.approx(20697 / last["total_population"] * 10_000)


def test_pcr_gaps_interpolated_and_flagged(data_dir):
    interp = load_local_influenza("chelyabinsk", data_dir=data_dir, start_date="2022-11-28", end_date="2023-01-02")
    zero = load_local_influenza(
        "chelyabinsk", data_dir=data_dir, start_date="2022-11-28", end_date="2023-01-02", pcr_gap_policy="zero"
    )
    w_i = interp.weekly.set_index("datetime")
    w_z = zero.weekly.set_index("datetime")
    gap = pd.Timestamp("2022-12-12")
    assert bool(w_i.loc[gap, "pcr_imputed"])
    assert w_z.loc[gap, "inc_per_10k"] == 0.0
    assert w_i.loc[gap, "inc_per_10k"] > 50
    assert not w_i["pcr_imputed"].loc[pd.Timestamp("2022-12-19")]
    assert interp.pcr_imputed_weeks == 2


def test_unknown_city(data_dir):
    with pytest.raises(LocalDataError, match="Доступные города"):
        load_local_influenza("atlantis", data_dir=data_dir)


def test_unsubtyped_a_check():
    import warnings

    from ai4epi.data.influenza import StrainDoubleCountWarning, check_unsubtyped_a

    weeks = 20
    ok = pd.DataFrame({"tested_total": [100] * weeks, "tested_strain_0": [0, 1] * 10,
                       "tested_strain_1": [5] * weeks, "tested_strain_2": [3] * weeks, "tested_strain_3": [2] * weeks})
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        res = check_unsubtyped_a(ok)
    assert res["suspected_double_count"] is False and res["share_a_not_less_than_subtypes"] == 0.0
    # A = все положительные на A (субтипы — подмножество): A ≥ H1 + H3 почти всегда
    total_a = ok.assign(tested_strain_0=ok["tested_strain_1"] + ok["tested_strain_2"] + 1)
    with pytest.warns(StrainDoubleCountWarning, match="двойной счёт"):
        res = check_unsubtyped_a(total_a)
    assert res["suspected_double_count"] is True


def test_real_spb_pcr_has_unsubtyped_a(data_dir):
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")  # на реальных данных проверка не должна срабатывать
        from ai4epi.data.local_excel import load_local_cases

        for city in ("spb", "samara", "chelyabinsk"):
            load_local_cases(city, data_dir=data_dir, age_pcr_path=None)
