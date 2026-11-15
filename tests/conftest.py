from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ai4epi.mechanistic import ModelParams, get_model, model_observable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "regions"


@pytest.fixture(scope="session")
def data_dir() -> Path:
    if not (DATA_DIR / "spb" / "cases.xlsx").is_file():
        pytest.skip("локальные данные data/regions не найдены")
    return DATA_DIR


def _synthetic_weekly(
    *,
    alpha: float = 0.6,
    beta: float = 0.7,
    population: int = 5_000_000,
    kappa: float = 0.1,
    lead_weeks: int = 6,
    wave_weeks: int = 30,
    start: str = "2018-09-03",
) -> pd.DataFrame:
    """Недельный ряд: 6 недель межсезонья, затем волна БР с известными параметрами."""

    model = get_model("total")
    params = ModelParams([alpha], [beta], int(population * kappa), [100.0])
    wave = model_observable(model, params, wave_weeks)
    cases = np.concatenate([np.zeros(lead_weeks), wave])
    dates = pd.date_range(start, periods=len(cases), freq="7D")
    iso = dates.isocalendar()
    return pd.DataFrame(
        {
            "datetime": dates,
            "iso_year": iso.year.to_numpy(int),
            "iso_week": iso.week.to_numpy(int),
            "total_population": float(population),
            "total_cases_formula": cases,
            "inc_per_10k": cases / population * 10_000,
        }
    )


@pytest.fixture
def synthetic_weekly():
    return _synthetic_weekly
