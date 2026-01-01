"""
Calibration Module - Bayesian calibration with surrogate models.

Provides uncertainty-quantified parameter estimation for building energy models.

Key components:
- SurrogateTrainer: Train Gaussian Process models on E+ simulation results
- SurrogatePredictor: Fast predictions using trained surrogates
- ABCSMCCalibrator: Bayesian calibration using ABC-SMC algorithm
- BayesianCalibrator: Unified interface for calibration with uncertainty

Usage:
    from src.calibration import BayesianCalibrator

    calibrator = BayesianCalibrator(surrogate_dir=Path("./surrogates"))
    result = calibrator.calibrate(
        archetype_id="mfh_1961_1975",
        measured_kwh_m2=85.0,
    )

    print(f"Infiltration: {result.posterior.means['infiltration_ach']:.3f}")
    print(f"90% CI: {result.posterior.ci_90['infiltration_ach']}")
"""

from .surrogate import (
    SurrogateConfig,
    SurrogateTrainer,
    SurrogatePredictor,
    TrainedSurrogate,
    FixedParamPredictor,
)
from .bayesian import (
    Prior,
    CalibrationPriors,
    PosteriorSample,
    CalibrationPosterior,
    ABCSMCCalibrator,
    UncertaintyPropagator,
    ECMUncertaintyPropagator,
    ECM_PARAMETER_EFFECTS,
    get_ecm_effect,
)
from .calibrator_v2 import (
    CalibrationResultV2,
    BayesianCalibrator,
)
from .metrics import (
    CalibrationMetrics,
    compute_uncertainty_adjusted_metrics,
)
from .pipeline import (
    CalibrationResult,
    BayesianCalibrationPipeline,
)
from .sensitivity import (
    MorrisResults,
    MorrisScreening,
    run_morris_analysis,
    AdaptiveCalibration,
)

__all__ = [
    # Surrogate
    "SurrogateConfig",
    "SurrogateTrainer",
    "SurrogatePredictor",
    "TrainedSurrogate",
    "FixedParamPredictor",
    # Bayesian
    "Prior",
    "CalibrationPriors",
    "PosteriorSample",
    "CalibrationPosterior",
    "ABCSMCCalibrator",
    "UncertaintyPropagator",
    # ECM Uncertainty
    "ECMUncertaintyPropagator",
    "ECM_PARAMETER_EFFECTS",
    "get_ecm_effect",
    # Interface
    "CalibrationResultV2",
    "BayesianCalibrator",
    # Metrics
    "CalibrationMetrics",
    "compute_uncertainty_adjusted_metrics",
    # Pipeline
    "CalibrationResult",
    "BayesianCalibrationPipeline",
    # Sensitivity
    "MorrisResults",
    "MorrisScreening",
    "run_morris_analysis",
    "AdaptiveCalibration",
]
