from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from ai4epi.data.local_excel import load_local_cases
from ai4epi.data.raw_regions import (
    GROUP_LABELS,
    RAW_CITIES,
    RawFormatError,
    convert_raw_city,
    normalize_group_label,
    read_raw_cases,
)


def _labels(n: int) -> list[str]:
    start = pd.Timestamp("2010-01-04")
    out = []
    for k in range(n):
        d = start + pd.Timedelta(weeks=k)
        iso = d.isocalendar()
        end = d + pd.Timedelta(days=6)
        out.append(f"{iso.year}.{iso.week}\xa0({d:%d.%m.%Y} - {end:%d.%m.%Y})")
    return out


def _raw_cases(n: int) -> pd.DataFrame:
    # заголовки групп так, как их испортил Excel: даты и '3-Jun' / 'Jul-14'
    groups = ["Всего", "0 - 2", datetime(2021, 6, 3), datetime(2014, 7, 1), "15 и ст.", "15 - 64", "65 и ст.", datetime(2021, 4, 3)]
    groups_txt = ["Всего", "0 - 2", "3-Jun", "Jul-14", "15 и ст.", "15 - 64", "65 и ст.", "3 - 4"]
    rows = [["Регион , Город , ЦГиЭ"] + [None] * 25,
            [None, None, "ОРЗ"] + [None] * 7 + ["Грипп"] + [None] * 7 + ["Население"] + [None] * 7,
            [None, None, *groups, *groups_txt, *groups]]
    for k, label in enumerate(_labels(n)):
        ari = [1000 + k, 200, 200, 100, 500, 450, 50, 80]
        pop = [100_000, 3_000, 4_000, 8_000, 85_000, 70_000, 15_000, 2_000]
        rows.append([label, "Н", *ari, *([np.nan] * 8), *pop])
    return pd.DataFrame(rows)


def _raw_pcr(n: int) -> pd.DataFrame:
    header = ["Регион , Город", None, "Число образцов тестированных на грипп", "A (субтип не определен)", "A(H1)pdm09", "A(H3)", "B"]
    rows = [header]
    for k, label in enumerate(_labels(n)):
        rows.append([label, "Н", np.nan if k == 0 else 50, 0, 5, 3, 2])
    return pd.DataFrame(rows)


def test_normalize_group_label():
    assert normalize_group_label(datetime(2021, 6, 3)) == "3 - 6"
    assert normalize_group_label(pd.Timestamp("2014-07-01")) == "7 - 14"
    assert normalize_group_label("Jul-14") == "7 - 14"
    assert normalize_group_label("15\xa0и ст.") == "15 и ст."
    with pytest.raises(RawFormatError):
        normalize_group_label(datetime(2020, 1, 1))


def test_convert_and_load(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    spec = RAW_CITIES["omsk"]
    _raw_cases(12).to_excel(raw / spec.cases_file, header=False, index=False)
    _raw_pcr(12).to_excel(raw / spec.pcr_file, header=False, index=False)

    cases = read_raw_cases(raw / spec.cases_file)
    assert [c for c in cases.columns if c.endswith("_орви")] == [f"{g}_орви" for g in GROUP_LABELS]
    assert "\xa0" not in cases["Дата"].iloc[0]

    report = convert_raw_city("omsk", raw_dir=raw, out_dir=tmp_path / "regions")
    assert report.cases_rows == 12 and report.pcr_missing_weeks == 1
    loaded, dropped = load_local_cases("omsk", data_dir=tmp_path / "regions", age_pcr_path=None)
    assert dropped == 0
    # первая неделя без ПЦР отбрасывается (интерполяция только внутри ряда)
    assert len(loaded) == 11
    row = loaded.iloc[0]
    assert row["sars_cases_age_group_1"] == 200 and row["population_age_group_3"] == 85_000
    assert row["real_cases_strain_1"] == round(5 / 50 * (1000 + 1))


def test_convert_arbitrary_city_writes_meta(tmp_path):
    import json

    _raw_cases(8).to_excel(tmp_path / "Заб Томск.xlsx", header=False, index=False)
    _raw_pcr(8).to_excel(tmp_path / "ПЦР Томск.xlsx", header=False, index=False)
    convert_raw_city("tomsk", out_dir=tmp_path / "regions", cases_file=tmp_path / "Заб Томск.xlsx",
                     pcr_file=tmp_path / "ПЦР Томск.xlsx", name_ru="Томск")
    meta = json.loads((tmp_path / "regions" / "tomsk" / "meta.json").read_text(encoding="utf-8"))
    assert meta == {"name_ru": "Томск", "district": "Томск"}
    cases, _ = load_local_cases("tomsk", data_dir=tmp_path / "regions")
    assert cases["region_name"].iloc[0] == "Томск"
    with pytest.raises(RawFormatError):
        convert_raw_city("tomsk", out_dir=tmp_path / "regions")
