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

    n_samples: int = 150  # Latin Hypercube samples (increased from 100, literature: 10-20 per dim)
    random_state: int = 42

    # Parameter bounds (Swedish buildings)
    param_bounds: Dict[str, Tuple[float, float]] = field(default_factory=dict)

    def __post_init__(self):
        if not self.param_bounds:
            # EXPANDED BOUNDS for simulating poorly-performing buildings
            # Based on Swedish building stock data:
            # - Pre-1975 buildings can have infiltration 0.4-0.6 ACH
            # - Uninsulated concrete/brick walls: U = 1.5-2.5 W/m²K
            # - Old double-pane windows: U = 2.5-3.5 W/m²K
            # - 86% of Sjöstad buildings have non-functional FTX
            self.param_bounds = {
                'infiltration_ach': (0.02, 0.50),  # Expanded from 0.20 for leaky buildings
                'wall_u_value': (0.15, 2.50),      # Expanded from 1.50 for poor insulation
                'roof_u_value': (0.10, 1.50),      # Expanded from 0.60 for flat roofs
                'floor_u_value': (0.15, 1.50),     # Expanded from 0.80 for slab-on-grade
                'window_u_value': (0.70, 4.00),    # Expanded from 2.50 for old windows
                'heat_recovery_eff': (0.0, 0.90),
                'heating_setpoint': (18.0, 24.0),  # Expanded from 23.0 for overheated buildings
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
    # Cross-validation metrics (new for overfitting detection)
    test_r2: float = 0.0
    test_rmse: float = 0.0
    is_overfit: bool = False  # True if train_r2 - test_r2 > 0.10

    @property
    def train_r2(self) -> float:
        """Alias for training_r2 for compatibility."""
        return self.training_r2

    @property
    def model(self):
        """Alias for gp_model for compatibility."""
        return self.gp_model


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
        test_split: float = 0.2,  # Holdout fraction for validation
    ) -> TrainedSurrogate:
        """
        Train Gaussian Process surrogate on simulation results.

        Uses 80/20 train/test split to detect overfitting (literature best practice).
        Reports both train_r2 and test_r2 - flags if difference > 0.10.

        Args:
            archetype_id: Identifier for this archetype
            X: Parameter samples (n_samples, n_params)
            y: Corresponding heating_kwh_m2 results
            test_split: Fraction of data to holdout for testing (default 0.2)

        Returns:
            TrainedSurrogate ready for inference
        """
        logger.info(f"Training surrogate for {archetype_id} with {len(X)} samples")

        # === TRAIN/TEST SPLIT (80/20) ===
        n_samples = len(X)
        n_test = max(int(n_samples * test_split), 5)  # At least 5 test samples
        n_train = n_samples - n_test

        # Shuffle indices
        rng = np.random.default_rng(self.config.random_state)
        indices = rng.permutation(n_samples)
        train_idx = indices[:n_train]
        test_idx = indices[n_train:]

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        logger.info(f"Split: {n_train} train, {n_test} test samples")

        # Standardize using TRAINING data only (prevent data leakage)
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()

        X_train_scaled = scaler_X.fit_transform(X_train)
        y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()

        # Transform test data using training scalers
        X_test_scaled = scaler_X.transform(X_test)

        # Define GP kernel (Matern 5/2 is robust for physical systems)
        kernel = (
            ConstantKernel(1.0, (1e-3, 1e3)) *
            Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e2), nu=2.5) +
            WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-5, 1e1))
        )

        # Train GP on training data only
        gp = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=30,  # Increased from 10 (literature recommendation)
            normalize_y=False,  # Already normalized
            random_state=self.config.random_state,
        )
        gp.fit(X_train_scaled, y_train_scaled)

        # === EVALUATE ON TRAINING DATA ===
        y_train_pred_scaled = gp.predict(X_train_scaled)
        y_train_pred = scaler_y.inverse_transform(y_train_pred_scaled.reshape(-1, 1)).ravel()

        train_r2 = 1 - np.sum((y_train - y_train_pred)**2) / np.sum((y_train - y_train.mean())**2)
        train_rmse = np.sqrt(np.mean((y_train - y_train_pred)**2))

        # === EVALUATE ON TEST DATA (unseen) ===
        y_test_pred_scaled = gp.predict(X_test_scaled)
        y_test_pred = scaler_y.inverse_transform(y_test_pred_scaled.reshape(-1, 1)).ravel()

        # Handle edge case where test set has no variance
        if np.std(y_test) > 0:
            test_r2 = 1 - np.sum((y_test - y_test_pred)**2) / np.sum((y_test - y_test.mean())**2)
        else:
            test_r2 = 0.0  # Can't compute R² with zero variance
        test_rmse = np.sqrt(np.mean((y_test - y_test_pred)**2))

        # === OVERFITTING DETECTION ===
        is_overfit = (train_r2 - test_r2) > 0.10

        logger.info(f"Train R²: {train_r2:.4f}, RMSE: {train_rmse:.2f} kWh/m²")
        logger.info(f"Test  R²: {test_r2:.4f}, RMSE: {test_rmse:.2f} kWh/m²")

        if is_overfit:
            logger.warning(
                f"⚠️ OVERFITTING DETECTED: Train R²={train_r2:.3f} vs Test R²={test_r2:.3f} "
                f"(gap={train_r2 - test_r2:.3f} > 0.10). Consider more samples or regularization."
            )
        else:
            logger.info(f"✓ No overfitting (gap={train_r2 - test_r2:.3f} ≤ 0.10)")

        # Re-fit on ALL data for final model (common practice after validation)
        X_all_scaled = scaler_X.fit_transform(X)
        y_all_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()
        gp.fit(X_all_scaled, y_all_scaled)

        return TrainedSurrogate(
            archetype_id=archetype_id,
            gp_model=gp,
            scaler_X=scaler_X,
            scaler_y=scaler_y,
            param_names=self.param_names,
            param_bounds=self.config.param_bounds,
            training_r2=train_r2,
            training_rmse=train_rmse,
            n_training_samples=len(X),
            test_r2=test_r2,
            test_rmse=test_rmse,
            is_overfit=is_overfit,
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

    def train_mock(self, archetype_id: str) -> TrainedSurrogate:
        """
        Train surrogate using synthetic data based on archetype physics.

        This is useful for testing the pipeline without running actual
        EnergyPlus simulations. The synthetic model uses a simplified
        heat balance equation calibrated to Swedish archetypes.

        Args:
            archetype_id: Archetype identifier

        Returns:
            TrainedSurrogate (less accurate than real E+ training)
        """
        logger.info(f"Training MOCK surrogate for {archetype_id} (no E+ simulations)")

        # Generate LHS samples
        X = self.generate_samples()
        n_samples = len(X)

        # Get archetype defaults for baseline calibration
        try:
            from src.baseline import get_archetype
            archetype = get_archetype(archetype_id)
            if archetype:
                # Handle both SwedishArchetype (envelope) and DetailedArchetype (wall_constructions)
                if hasattr(archetype, 'envelope') and archetype.envelope:
                    base_wall_u = archetype.envelope.wall_u_value
                    base_roof_u = archetype.envelope.roof_u_value or 0.3
                    base_window_u = archetype.envelope.window_u_value or 2.0
                elif hasattr(archetype, 'wall_constructions') and archetype.wall_constructions:
                    # DetailedArchetype structure
                    base_wall_u = archetype.wall_constructions[0].u_value
                    base_roof_u = archetype.roof_construction.u_value if archetype.roof_construction else 0.3
                    if archetype.window_construction:
                        base_window_u = getattr(archetype.window_construction, 'u_value_installed',
                                               getattr(archetype.window_construction, 'u_value_glass', 2.0))
                    else:
                        base_window_u = 2.0
                else:
                    base_wall_u, base_roof_u, base_window_u = 0.5, 0.3, 2.0
            else:
                base_wall_u, base_roof_u, base_window_u = 0.5, 0.3, 2.0
        except Exception:
            base_wall_u, base_roof_u, base_window_u = 0.5, 0.3, 2.0

        # Simplified heat balance model for Swedish climate
        # Q_heat = (UA_envelope + UA_ventilation - Q_internal) * HDD
        # Typical Stockholm: ~4000 HDD (base 17°C)

        HDD = 4000  # Heating degree days (Stockholm)
        INTERNAL_GAINS = 25  # kWh/m² from occupants, appliances, lighting
        WWR = 0.20  # Window-to-wall ratio
        VENT_RATE = 0.35  # L/s/m² (Swedish standard)

        y = np.zeros(n_samples)

        for i in range(n_samples):
            params = {name: X[i, j] for j, name in enumerate(self.param_names)}

            # Envelope losses (simplified)
            wall_u = params.get('wall_u_value', base_wall_u)
            roof_u = params.get('roof_u_value', base_roof_u)
            window_u = params.get('window_u_value', base_window_u)
            floor_u = params.get('floor_u_value', 0.3)

            # Composite U-value (weighted by typical Swedish MFH areas)
            # Wall: 40%, Roof: 25%, Floor: 20%, Windows: 15%
            opaque_u = 0.40 * wall_u * (1 - WWR) + 0.40 * window_u * WWR
            opaque_u += 0.25 * roof_u + 0.20 * floor_u

            # Infiltration losses
            infiltration = params.get('infiltration_ach', 0.06)
            # Q_inf = 0.34 * ACH * V * dT (approx 3 kWh/m² per 0.1 ACH)
            inf_loss = infiltration * 30  # kWh/m² per ACH

            # Ventilation losses with heat recovery
            heat_recovery = params.get('heat_recovery_eff', 0.0)
            vent_loss = VENT_RATE * 3600 * 1.2 * 1005 / 3600000  # W/m²/K
            vent_loss_annual = vent_loss * HDD * 24 / 1000  # kWh/m²
            vent_loss_net = vent_loss_annual * (1 - heat_recovery)

            # Setpoint adjustment
            setpoint = params.get('heating_setpoint', 21.0)
            setpoint_factor = 1.0 + (setpoint - 21.0) * 0.05  # 5% per degree

            # Total heating demand
            envelope_loss = opaque_u * HDD * 24 / 1000  # kWh/m²
            total_loss = envelope_loss + inf_loss + vent_loss_net
            heating = max(0, total_loss - INTERNAL_GAINS) * setpoint_factor

            # Add noise (±5%)
            noise = np.random.normal(0, 0.05 * heating)
            y[i] = max(0, heating + noise)

        logger.info(f"Generated {n_samples} synthetic samples: "
                    f"min={y.min():.1f}, max={y.max():.1f}, mean={y.mean():.1f} kWh/m²")

        # Train on synthetic data
        return self.train(archetype_id, X, y)

    def train_with_simulation(
        self,
        archetype_id: str,
        base_idf_path: Path,
        weather_path: Path,
        output_dir: Path,
        n_workers: int = 4,
    ) -> TrainedSurrogate:
        """
        Train surrogate by running actual EnergyPlus simulations.

        This method:
        1. Generates Latin Hypercube samples of parameter space
        2. For each sample, modifies the IDF and runs E+ simulation
        3. Collects heating_kwh_m2 results
        4. Trains GP surrogate on the results

        Args:
            archetype_id: Archetype identifier
            base_idf_path: Path to base IDF file for this archetype
            weather_path: Path to weather file (.epw)
            output_dir: Directory for simulation outputs
            n_workers: Number of parallel simulation workers

        Returns:
            TrainedSurrogate ready for inference
        """
        from concurrent.futures import ProcessPoolExecutor, as_completed

        logger.info(f"Training surrogate for {archetype_id} with E+ simulations")
        logger.info(f"  Base IDF: {base_idf_path}")
        logger.info(f"  Weather: {weather_path}")
        logger.info(f"  Samples: {self.config.n_samples}")
        logger.info(f"  Workers: {n_workers}")

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Generate LHS samples
        X = self.generate_samples()
        n_samples = len(X)
        param_dicts = self.samples_to_dicts(X)

        # Initialize results
        y = np.zeros(n_samples)
        successful = np.ones(n_samples, dtype=bool)

        # Run simulations in parallel using module-level function
        # (required for ProcessPoolExecutor pickling)
        logger.info(f"Running {n_samples} E+ simulations with {n_workers} workers...")

        # Prepare arguments for module-level function
        sim_args = [
            (i, params, str(base_idf_path), str(weather_path), str(output_dir / f"sim_{i:04d}"))
            for i, params in enumerate(param_dicts)
        ]

        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {
                executor.submit(_run_single_ep_simulation, args): args[0]
                for args in sim_args
            }

            completed = 0
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    result_idx, heating = future.result()
                    if heating is not None:
                        y[result_idx] = heating
                    else:
                        successful[result_idx] = False
                except Exception as e:
                    logger.error(f"Worker exception for simulation {idx}: {e}")
                    successful[idx] = False

                completed += 1
                if completed % 10 == 0:
                    logger.info(f"Progress: {completed}/{n_samples} simulations")

        # Filter to successful simulations
        n_success = successful.sum()
        logger.info(f"Completed: {n_success}/{n_samples} successful simulations")

        if n_success < 20:
            raise RuntimeError(
                f"Too few successful simulations ({n_success}). "
                "Cannot train reliable surrogate."
            )

        X_success = X[successful]
        y_success = y[successful]

        logger.info(f"Training on {len(X_success)} samples: "
                    f"min={y_success.min():.1f}, max={y_success.max():.1f}, "
                    f"mean={y_success.mean():.1f} kWh/m²")

        # Train surrogate
        return self.train(archetype_id, X_success, y_success)

    @property
    def train_r2(self) -> float:
        """Alias for training_r2 for compatibility."""
        return getattr(self, 'training_r2', 0.0)


def _get_floor_area_from_idf(idf) -> float:
    """Extract total floor area from IDF."""
    try:
        zones = idf.idfobjects['ZONE']
        if not zones:
            return 1000.0  # Default fallback

        # Try to get from Zone list (using floor area if available)
        total_area = 0.0
        for zone in zones:
            # Some IDFs have floor area in zone definition
            area = getattr(zone, 'Floor_Area', None)
            if area and area > 0:
                total_area += area

        if total_area > 0:
            return total_area

        # Fallback: estimate from building geometry
        # (would need more complex calculation from surfaces)
        return 1000.0

    except Exception:
        return 1000.0


def _run_single_ep_simulation(args: Tuple) -> Tuple[int, Optional[float]]:
    """
    Run a single E+ simulation for surrogate training.

    Must be at module level for ProcessPoolExecutor pickling.

    Args:
        args: Tuple of (idx, params_dict, base_idf_path, weather_path, sim_dir)

    Returns:
        (idx, heating_kwh_m2) or (idx, None) on failure
    """
    from pathlib import Path
    idx, params, base_idf_path, weather_path, sim_dir = args

    try:
        # Import inside function to avoid pickling issues
        from ..core.idf_parser import IDFParser
        from ..simulation import SimulationRunner, ResultsParser

        sim_dir = Path(sim_dir)
        sim_dir.mkdir(exist_ok=True)

        # Initialize
        idf_parser = IDFParser()
        runner = SimulationRunner()
        results_parser = ResultsParser()

        # Load and modify IDF
        idf = idf_parser.load(Path(base_idf_path))

        # Apply ALL 6 calibration parameters
        if 'infiltration_ach' in params:
            idf_parser.set_infiltration_ach(idf, params['infiltration_ach'])

        if 'window_u_value' in params:
            idf_parser.set_window_u_value(idf, params['window_u_value'])

        if 'heat_recovery_eff' in params:
            idf_parser.set_heat_recovery_effectiveness(idf, params['heat_recovery_eff'])

        if 'wall_u_value' in params:
            idf_parser.set_wall_u_value(idf, params['wall_u_value'])

        if 'roof_u_value' in params:
            idf_parser.set_roof_u_value(idf, params['roof_u_value'])

        if 'heating_setpoint' in params:
            idf_parser.set_heating_setpoint(idf, params['heating_setpoint'])

        # Save modified IDF
        modified_idf_path = sim_dir / "modified.idf"
        idf_parser.save(idf, modified_idf_path)

        # Run simulation
        result = runner.run(modified_idf_path, Path(weather_path), sim_dir)

        if result.success:
            # Parse results
            annual = results_parser.parse(sim_dir)
            if annual and annual.heating_kwh > 0:
                # AnnualResults already has heating_kwh_m2 calculated
                return idx, annual.heating_kwh_m2
            else:
                logger.warning(f"Simulation {idx}: No heating results")
                return idx, None
        else:
            logger.warning(f"Simulation {idx} failed: {result.error}")
            return idx, None

    except Exception as e:
        logger.error(f"Simulation {idx} exception: {e}")
        return idx, None


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


class FixedParamPredictor:
    """
    Wrapper around SurrogatePredictor that injects fixed parameter values.

    Used when Morris screening identifies some parameters as non-influential.
    Those parameters are fixed at archetype defaults, reducing the
    dimensionality of the calibration problem.

    This follows Kennedy & O'Hagan (2001) guidance to only calibrate
    identifiable parameters.
    """

    def __init__(
        self,
        predictor: SurrogatePredictor,
        fixed_params: Dict[str, float],
    ):
        """
        Args:
            predictor: Base surrogate predictor
            fixed_params: Parameters to fix at constant values
        """
        self.predictor = predictor
        self.fixed_params = fixed_params
        self.surrogate = predictor.surrogate
        logger.info(
            f"FixedParamPredictor: {len(fixed_params)} params fixed at defaults: "
            f"{', '.join(f'{k}={v:.3f}' for k, v in fixed_params.items())}"
        )

    def predict(
        self,
        params: Dict[str, float],
        return_std: bool = False,
    ) -> Tuple[float, Optional[float]]:
        """Predict with fixed params injected."""
        full_params = {**self.fixed_params, **params}
        return self.predictor.predict(full_params, return_std=return_std)

    def predict_batch(
        self,
        params_list: List[Dict[str, float]],
    ) -> np.ndarray:
        """Batch predict with fixed params injected."""
        full_params_list = [
            {**self.fixed_params, **p} for p in params_list
        ]
        return self.predictor.predict_batch(full_params_list)
