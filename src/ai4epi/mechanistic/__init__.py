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
    r2_score_np,
)
from ai4epi.mechanistic.models import (
    DEFAULT_INFECTIVITY_PROFILE,
    AgeGroupBRModel,
    BaroyanRvachevModel,
    ModelParams,
    SimulationResult,
    TotalBRModel,
    get_model,
)

__all__ = [
    "CALIBRATION_METHODS",
    "DEFAULT_INFECTIVITY_PROFILE",
    "AgeGroupBRModel",
    "BaroyanRvachevModel",
    "CalibrationError",
    "CalibrationResult",
    "ModelParams",
    "SimulationResult",
    "TotalBRModel",
    "calibrate",
    "get_model",
    "model_observable",
    "r2_score_np",
]
