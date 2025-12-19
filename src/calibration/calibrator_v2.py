"""
Unified Bayesian calibration interface.

High-level API that orchestrates surrogate model loading/training
and Bayesian calibration to deliver uncertainty-quantified results.

Usage:
    from src.calibration import BayesianCalibrator

    calibrator = BayesianCalibrator(surrogate_dir=Path("./surrogates"))
    result = calibrator.calibrate(
        archetype_id="mfh_1961_1975",
        measured_kwh_m2=85.0,
    )

    # Results with uncertainty
    print(f"Best infiltration: {result.best_params['infiltration_ach']:.3f}")
    print(f"90% CI: {result.ci_90['infiltration_ach']}")
    print(f"Predicted heating: {result.predicted_kwh_m2:.1f} ± {result.prediction_std:.1f}")
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import logging

import numpy as np

from .surrogate import (
    SurrogateConfig,
    SurrogateTrainer,
    SurrogatePredictor,
    TrainedSurrogate,
)
from .bayesian import (
    CalibrationPriors,
    CalibrationPosterior,
    ABCSMCCalibrator,
    UncertaintyPropagator,
)

logger = logging.getLogger(__name__)


@dataclass
class CalibrationResultV2:
    """
    Complete calibration result with uncertainty quantification.

    Contains:
    - Best-fit parameters
    - Parameter uncertainty (credible intervals)
    - Prediction with uncertainty
    - Full posterior for advanced analysis
    """

    archetype_id: str
    measured_kwh_m2: float

    # Best-fit parameters
    best_params: Dict[str, float]

    # Uncertainty quantification
    param_stds: Dict[str, float]
    ci_90: Dict[str, Tuple[float, float]]
    ci_95: Dict[str, Tuple[float, float]]

    # Predictions
    predicted_kwh_m2: float
    prediction_std: float
    prediction_ci_90: Tuple[float, float]

    # Calibration quality
    calibration_error_percent: float
    final_tolerance: float
    n_accepted_samples: int

    # Full posterior (for advanced users)
    posterior: CalibrationPosterior

    # Metadata
    surrogate_r2: float
    surrogate_rmse: float

    def to_dict(self) -> Dict[str, Any]:
        """Export result as dictionary (for JSON serialization)."""
        return {
            "archetype_id": self.archetype_id,
            "measured_kwh_m2": self.measured_kwh_m2,
            "best_params": self.best_params,
            "param_stds": self.param_stds,
            "ci_90": {k: list(v) for k, v in self.ci_90.items()},
            "ci_95": {k: list(v) for k, v in self.ci_95.items()},
            "predicted_kwh_m2": self.predicted_kwh_m2,
            "prediction_std": self.prediction_std,
            "prediction_ci_90": list(self.prediction_ci_90),
            "calibration_error_percent": self.calibration_error_percent,
            "final_tolerance": self.final_tolerance,
            "n_accepted_samples": self.n_accepted_samples,
            "surrogate_r2": self.surrogate_r2,
            "surrogate_rmse": self.surrogate_rmse,
        }

    def summary(self) -> str:
        """Human-readable summary of calibration results."""
        lines = [
            f"Calibration Results for {self.archetype_id}",
            "=" * 50,
            f"Measured: {self.measured_kwh_m2:.1f} kWh/m²",
            f"Predicted: {self.predicted_kwh_m2:.1f} ± {self.prediction_std:.1f} kWh/m²",
            f"Error: {self.calibration_error_percent:.1f}%",
            "",
            "Calibrated Parameters (mean ± std):",
        ]

        for param in self.best_params:
            mean = self.best_params[param]
            std = self.param_stds[param]
            ci = self.ci_90[param]
            lines.append(f"  {param}: {mean:.4f} ± {std:.4f} (90% CI: {ci[0]:.4f} - {ci[1]:.4f})")

        lines.extend([
            "",
            f"Surrogate model: R²={self.surrogate_r2:.4f}, RMSE={self.surrogate_rmse:.2f} kWh/m²",
            f"Accepted samples: {self.n_accepted_samples}",
        ])

        return "\n".join(lines)


class BayesianCalibrator:
    """
    High-level interface for Bayesian building calibration.

    Handles:
    - Surrogate model loading/caching
    - Prior selection based on archetype
    - ABC-SMC calibration
    - Uncertainty propagation
    - Result packaging

    Usage:
        calibrator = BayesianCalibrator(surrogate_dir=Path("./surrogates"))

        # Calibrate to measured data
        result = calibrator.calibrate(
            archetype_id="mfh_1961_1975",
            measured_kwh_m2=85.0,
        )

        # Use calibrated parameters
        print(result.best_params)
        print(result.ci_90)
    """

    def __init__(
        self,
        surrogate_dir: Optional[Path] = None,
        n_particles: int = 1000,
        n_generations: int = 8,
        random_state: int = 42,
    ):
        """
        Initialize calibrator.

        Args:
            surrogate_dir: Directory containing pre-trained surrogate models
            n_particles: Number of ABC-SMC particles
            n_generations: Number of SMC generations
            random_state: Random seed for reproducibility
        """
        self.surrogate_dir = Path(surrogate_dir) if surrogate_dir else None
        self.n_particles = n_particles
        self.n_generations = n_generations
        self.random_state = random_state

        # Cache loaded surrogates
        self._surrogate_cache: Dict[str, TrainedSurrogate] = {}

    def calibrate(
        self,
        archetype_id: str,
        measured_kwh_m2: float,
        custom_priors: Optional[CalibrationPriors] = None,
        initial_tolerance_percent: float = 20.0,
    ) -> CalibrationResultV2:
        """
        Calibrate building parameters to measured energy consumption.

        Args:
            archetype_id: Building archetype identifier
            measured_kwh_m2: Measured annual heating energy
            custom_priors: Optional custom prior distributions
            initial_tolerance_percent: Initial ABC acceptance tolerance

        Returns:
            CalibrationResultV2 with uncertainty-quantified results
        """
        logger.info(f"Starting calibration for {archetype_id}, target={measured_kwh_m2} kWh/m²")

        # Load or create surrogate
        surrogate = self._get_surrogate(archetype_id)
        predictor = SurrogatePredictor(surrogate)

        # Get priors (archetype-informed or custom)
        if custom_priors:
            priors = custom_priors
        else:
            priors = CalibrationPriors.from_archetype(archetype_id)

        # Run ABC-SMC calibration
        abc = ABCSMCCalibrator(
            predictor=predictor,
            priors=priors,
            n_particles=self.n_particles,
            n_generations=self.n_generations,
            random_state=self.random_state,
        )

        posterior = abc.calibrate(
            measured_kwh_m2=measured_kwh_m2,
            tolerance_percent=initial_tolerance_percent,
        )

        # Propagate uncertainty to predictions
        propagator = UncertaintyPropagator(predictor, posterior)
        pred_mean, pred_std, pred_ci = propagator.predict_with_uncertainty()

        # Calculate calibration error
        error_pct = abs(pred_mean - measured_kwh_m2) / measured_kwh_m2 * 100

        return CalibrationResultV2(
            archetype_id=archetype_id,
            measured_kwh_m2=measured_kwh_m2,
            best_params=posterior.means,
            param_stds=posterior.stds,
            ci_90=posterior.ci_90,
            ci_95=posterior.ci_95,
            predicted_kwh_m2=pred_mean,
            prediction_std=pred_std,
            prediction_ci_90=pred_ci,
            calibration_error_percent=error_pct,
            final_tolerance=posterior.epsilon_final,
            n_accepted_samples=len(posterior.samples),
            posterior=posterior,
            surrogate_r2=surrogate.training_r2,
            surrogate_rmse=surrogate.training_rmse,
        )

    def _get_surrogate(self, archetype_id: str) -> TrainedSurrogate:
        """Load surrogate from cache or disk."""
        if archetype_id in self._surrogate_cache:
            return self._surrogate_cache[archetype_id]

        if self.surrogate_dir:
            path = self.surrogate_dir / f"surrogate_{archetype_id}.joblib"
            if path.exists():
                surrogate = SurrogateTrainer.load(path)
                self._surrogate_cache[archetype_id] = surrogate
                logger.info(f"Loaded surrogate from {path}")
                return surrogate

        raise FileNotFoundError(
            f"No surrogate model found for '{archetype_id}'. "
            f"Train one first using SurrogateTrainer."
        )

    def train_surrogate(
        self,
        archetype_id: str,
        X: np.ndarray,
        y: np.ndarray,
        save: bool = True,
    ) -> TrainedSurrogate:
        """
        Train a new surrogate model.

        Args:
            archetype_id: Identifier for this archetype
            X: Parameter samples (n_samples, n_params)
            y: Corresponding heating_kwh_m2 results
            save: Whether to save to surrogate_dir

        Returns:
            TrainedSurrogate ready for calibration
        """
        trainer = SurrogateTrainer()
        surrogate = trainer.train(archetype_id, X, y)

        if save and self.surrogate_dir:
            trainer.save(surrogate, self.surrogate_dir)

        self._surrogate_cache[archetype_id] = surrogate
        return surrogate

    def list_available_surrogates(self) -> List[str]:
        """List archetype IDs with available surrogate models."""
        if not self.surrogate_dir or not self.surrogate_dir.exists():
            return []

        surrogates = []
        for path in self.surrogate_dir.glob("surrogate_*.joblib"):
            # Extract archetype_id from filename
            archetype_id = path.stem.replace("surrogate_", "")
            surrogates.append(archetype_id)

        return sorted(surrogates)


def quick_calibrate(
    measured_kwh_m2: float,
    archetype_id: str = "generic",
    surrogate_dir: Optional[Path] = None,
) -> CalibrationResultV2:
    """
    Convenience function for quick calibration.

    Args:
        measured_kwh_m2: Measured annual heating
        archetype_id: Building archetype
        surrogate_dir: Path to surrogate models

    Returns:
        CalibrationResultV2
    """
    calibrator = BayesianCalibrator(surrogate_dir=surrogate_dir)
    return calibrator.calibrate(
        archetype_id=archetype_id,
        measured_kwh_m2=measured_kwh_m2,
    )
