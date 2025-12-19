"""
Surrogate model training for fast Bayesian calibration.

Trains Gaussian Process models on E+ simulation results,
enabling instant predictions for parameter combinations.

Usage:
    trainer = SurrogateTrainer()
    samples = trainer.generate_samples()
    # ... run E+ simulations for samples ...
    surrogate = trainer.train(archetype_id, X, y)
    trainer.save(surrogate, output_dir)

    # Later, for prediction:
    predictor = SurrogatePredictor(surrogate)
    heating, std = predictor.predict(params, return_std=True)
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
from sklearn.preprocessing import StandardScaler
from scipy.stats import qmc
import joblib

logger = logging.getLogger(__name__)


@dataclass
class SurrogateConfig:
    """Configuration for surrogate model training."""

    n_samples: int = 100  # Latin Hypercube samples
    random_state: int = 42

    # Parameter bounds (Swedish buildings)
    param_bounds: Dict[str, Tuple[float, float]] = field(default_factory=dict)

    def __post_init__(self):
        if not self.param_bounds:
            self.param_bounds = {
                'infiltration_ach': (0.02, 0.20),
                'wall_u_value': (0.15, 1.50),
                'roof_u_value': (0.10, 0.60),
                'floor_u_value': (0.15, 0.80),
                'window_u_value': (0.70, 2.50),
                'heat_recovery_eff': (0.0, 0.90),
                'heating_setpoint': (18.0, 23.0),
            }


@dataclass
class TrainedSurrogate:
    """Trained surrogate model with metadata."""

    archetype_id: str
    gp_model: GaussianProcessRegressor
    scaler_X: StandardScaler
    scaler_y: StandardScaler
    param_names: List[str]
    param_bounds: Dict[str, Tuple[float, float]]
    training_r2: float
    training_rmse: float
    n_training_samples: int


class SurrogateTrainer:
    """
    Train Gaussian Process surrogate models for each archetype.

    Workflow:
    1. Generate Latin Hypercube samples of parameter space
    2. Run E+ simulations for all samples
    3. Train GP regressor on results
    4. Save model for fast inference
    """

    def __init__(self, config: Optional[SurrogateConfig] = None):
        self.config = config or SurrogateConfig()
        self.param_names = list(self.config.param_bounds.keys())

    def generate_samples(self) -> np.ndarray:
        """Generate Latin Hypercube samples of parameter space."""
        n_params = len(self.param_names)

        # Latin Hypercube sampling (better coverage than random)
        sampler = qmc.LatinHypercube(d=n_params, seed=self.config.random_state)
        samples_unit = sampler.random(n=self.config.n_samples)

        # Scale to parameter bounds
        lower = np.array([self.config.param_bounds[p][0] for p in self.param_names])
        upper = np.array([self.config.param_bounds[p][1] for p in self.param_names])
        samples = qmc.scale(samples_unit, lower, upper)

        return samples

    def samples_to_dicts(self, samples: np.ndarray) -> List[Dict[str, float]]:
        """Convert sample array to list of parameter dicts."""
        return [
            {name: samples[i, j] for j, name in enumerate(self.param_names)}
            for i in range(samples.shape[0])
        ]

    def train(
        self,
        archetype_id: str,
        X: np.ndarray,  # Parameter samples (n_samples, n_params)
        y: np.ndarray,  # E+ results: heating_kwh_m2 (n_samples,)
    ) -> TrainedSurrogate:
        """
        Train Gaussian Process surrogate on simulation results.

        Args:
            archetype_id: Identifier for this archetype
            X: Parameter samples (n_samples, n_params)
            y: Corresponding heating_kwh_m2 results

        Returns:
            TrainedSurrogate ready for inference
        """
        logger.info(f"Training surrogate for {archetype_id} with {len(X)} samples")

        # Standardize inputs and outputs
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()

        X_scaled = scaler_X.fit_transform(X)
        y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()

        # Define GP kernel (Matern is robust for physical systems)
        kernel = (
            ConstantKernel(1.0, (1e-3, 1e3)) *
            Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e2), nu=2.5) +
            WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-5, 1e1))
        )

        # Train GP
        gp = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=10,
            normalize_y=False,  # Already normalized
            random_state=self.config.random_state,
        )
        gp.fit(X_scaled, y_scaled)

        # Evaluate training performance
        y_pred_scaled = gp.predict(X_scaled)
        y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()

        r2 = 1 - np.sum((y - y_pred)**2) / np.sum((y - y.mean())**2)
        rmse = np.sqrt(np.mean((y - y_pred)**2))

        logger.info(f"Training R²: {r2:.4f}, RMSE: {rmse:.2f} kWh/m²")

        return TrainedSurrogate(
            archetype_id=archetype_id,
            gp_model=gp,
            scaler_X=scaler_X,
            scaler_y=scaler_y,
            param_names=self.param_names,
            param_bounds=self.config.param_bounds,
            training_r2=r2,
            training_rmse=rmse,
            n_training_samples=len(X),
        )

    def save(self, surrogate: TrainedSurrogate, output_dir: Path) -> Path:
        """Save trained surrogate to disk."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / f"surrogate_{surrogate.archetype_id}.joblib"
        joblib.dump(surrogate, path)
        logger.info(f"Saved surrogate to {path}")
        return path

    @staticmethod
    def load(path: Path) -> TrainedSurrogate:
        """Load trained surrogate from disk."""
        return joblib.load(path)


class SurrogatePredictor:
    """
    Fast predictions using trained surrogate.

    Provides both point predictions and uncertainty estimates.
    """

    def __init__(self, surrogate: TrainedSurrogate):
        self.surrogate = surrogate

    def predict(
        self,
        params: Dict[str, float],
        return_std: bool = False,
    ) -> Tuple[float, Optional[float]]:
        """
        Predict heating_kwh_m2 for given parameters.

        Args:
            params: Parameter dict
            return_std: Whether to return uncertainty estimate

        Returns:
            (prediction, std) if return_std else prediction
        """
        # Convert to array in correct order
        X = np.array([[params.get(name, 0) for name in self.surrogate.param_names]])
        X_scaled = self.surrogate.scaler_X.transform(X)

        if return_std:
            y_scaled, std_scaled = self.surrogate.gp_model.predict(X_scaled, return_std=True)
            y = self.surrogate.scaler_y.inverse_transform(y_scaled.reshape(-1, 1)).ravel()[0]
            # Approximate std transformation
            std = std_scaled[0] * self.surrogate.scaler_y.scale_[0]
            return y, std
        else:
            y_scaled = self.surrogate.gp_model.predict(X_scaled)
            y = self.surrogate.scaler_y.inverse_transform(y_scaled.reshape(-1, 1)).ravel()[0]
            return y

    def predict_batch(
        self,
        params_list: List[Dict[str, float]],
    ) -> np.ndarray:
        """Predict for multiple parameter sets (vectorized)."""
        X = np.array([
            [params.get(name, 0) for name in self.surrogate.param_names]
            for params in params_list
        ])
        X_scaled = self.surrogate.scaler_X.transform(X)
        y_scaled = self.surrogate.gp_model.predict(X_scaled)
        y = self.surrogate.scaler_y.inverse_transform(y_scaled.reshape(-1, 1)).ravel()
        return y
