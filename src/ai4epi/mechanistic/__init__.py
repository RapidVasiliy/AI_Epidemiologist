"""Механистическая модель эпидемической волны (Барояна–Рвачёва).

Перенесено из репозитория Influenza_bulletin (``model_complex``); численный
этап pipeline, использующий модель, — ``ai4epi.analysis.mechanistic_wave``.
"""

from ai4epi.mechanistic.calibration import (
    CALIBRATION_METHODS,
    CalibrationError,
    CalibrationResult,
    calibrate,
    model_observable,
    negative_binomial_noise,
    r2_score_np,
)
from ai4epi.mechanistic.models import (
    AJELLI_LITVINOVA_3_GROUPS,
    AJELLI_LITVINOVA_ALL_CONTACTS,
    DEFAULT_INFECTIVITY_PROFILE,
    RSCF_CONTACT_MATRIX_2,
    AgeGroupBRModel,
    BaroyanRvachevModel,
    FineAgeContactBRModel,
    ModelParams,
    SimulationResult,
    TotalBRModel,
    fine_age_populations,
    get_model,
)

__all__ = [
    "AJELLI_LITVINOVA_3_GROUPS",
    "AJELLI_LITVINOVA_ALL_CONTACTS",
    "CALIBRATION_METHODS",
    "DEFAULT_INFECTIVITY_PROFILE",
    "RSCF_CONTACT_MATRIX_2",
    "AgeGroupBRModel",
    "BaroyanRvachevModel",
    "CalibrationError",
    "CalibrationResult",
    "FineAgeContactBRModel",
    "ModelParams",
    "SimulationResult",
    "TotalBRModel",
    "calibrate",
    "fine_age_populations",
    "get_model",
    "model_observable",
    "negative_binomial_noise",
    "r2_score_np",
]
