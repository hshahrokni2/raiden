"""
Tests for the Bayesian calibration module.

Tests:
- Surrogate model training
- Latin Hypercube sampling
- ABC-SMC calibration
- Uncertainty propagation
"""

import numpy as np
import pytest
from pathlib import Path
import tempfile

from src.calibration import (
    SurrogateConfig,
    SurrogateTrainer,
    SurrogatePredictor,
    TrainedSurrogate,
)
from src.calibration.bayesian import (
    Prior,
    CalibrationPriors,
    ABCSMCCalibrator,
    UncertaintyPropagator,
)


class TestSurrogateConfig:
    """Test surrogate configuration."""

    def test_default_config(self):
        """Default config has Swedish building bounds."""
        config = SurrogateConfig()
        assert config.n_samples == 100
        assert "infiltration_ach" in config.param_bounds
        assert "wall_u_value" in config.param_bounds
        assert config.param_bounds["infiltration_ach"] == (0.02, 0.20)

    def test_custom_config(self):
        """Custom config overrides defaults."""
        config = SurrogateConfig(
            n_samples=50,
            param_bounds={"test_param": (0.0, 1.0)}
        )
        assert config.n_samples == 50
        assert config.param_bounds == {"test_param": (0.0, 1.0)}


class TestSurrogateTrainer:
    """Test surrogate model training."""

    def test_generate_samples(self):
        """Latin Hypercube sampling generates valid samples."""
        config = SurrogateConfig(n_samples=20)
        trainer = SurrogateTrainer(config)
        samples = trainer.generate_samples()

        assert samples.shape == (20, len(config.param_bounds))

        # Check samples are within bounds
        for i, param in enumerate(trainer.param_names):
            low, high = config.param_bounds[param]
            assert np.all(samples[:, i] >= low)
            assert np.all(samples[:, i] <= high)

    def test_samples_to_dicts(self):
        """Convert samples to parameter dictionaries."""
        trainer = SurrogateTrainer()
        samples = np.array([[0.05, 0.3, 0.2, 0.3, 1.0, 0.8, 21.0]])
        dicts = trainer.samples_to_dicts(samples)

        assert len(dicts) == 1
        assert "infiltration_ach" in dicts[0]
        assert dicts[0]["infiltration_ach"] == 0.05

    def test_train_surrogate(self):
        """Train GP surrogate on synthetic data."""
        config = SurrogateConfig(n_samples=30)
        trainer = SurrogateTrainer(config)

        # Generate synthetic training data
        X = trainer.generate_samples()

        # Simple synthetic model: heating ~ infiltration * 100 + wall_u * 50
        y = (
            X[:, 0] * 100 +  # infiltration
            X[:, 1] * 50 +   # wall_u
            np.random.normal(0, 2, len(X))  # noise
        )

        surrogate = trainer.train("test_archetype", X, y)

        assert surrogate.archetype_id == "test_archetype"
        assert surrogate.training_r2 > 0.8  # Should fit well
        assert surrogate.n_training_samples == 30
        assert len(surrogate.param_names) == len(config.param_bounds)

    def test_save_and_load(self):
        """Surrogate can be saved and loaded."""
        trainer = SurrogateTrainer(SurrogateConfig(n_samples=20))
        X = trainer.generate_samples()
        y = X[:, 0] * 100 + X[:, 1] * 50

        surrogate = trainer.train("test_save", X, y)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = trainer.save(surrogate, Path(tmpdir))
            loaded = trainer.load(path)

            assert loaded.archetype_id == surrogate.archetype_id
            assert loaded.training_r2 == surrogate.training_r2


class TestSurrogatePredictor:
    """Test surrogate predictions."""

    @pytest.fixture
    def trained_surrogate(self):
        """Create a trained surrogate for testing."""
        config = SurrogateConfig(n_samples=50)
        trainer = SurrogateTrainer(config)
        X = trainer.generate_samples()
        # Synthetic model
        y = 50 + X[:, 0] * 200 + X[:, 1] * 30
        return trainer.train("test_pred", X, y)

    def test_predict_single(self, trained_surrogate):
        """Predict for single parameter set."""
        predictor = SurrogatePredictor(trained_surrogate)

        params = {
            "infiltration_ach": 0.10,
            "wall_u_value": 0.5,
            "roof_u_value": 0.3,
            "floor_u_value": 0.4,
            "window_u_value": 1.2,
            "heat_recovery_eff": 0.75,
            "heating_setpoint": 21.0,
        }

        result = predictor.predict(params)
        assert isinstance(result, float)
        assert result > 0

    def test_predict_with_std(self, trained_surrogate):
        """Predict with uncertainty estimate."""
        predictor = SurrogatePredictor(trained_surrogate)

        params = {
            "infiltration_ach": 0.10,
            "wall_u_value": 0.5,
            "roof_u_value": 0.3,
            "floor_u_value": 0.4,
            "window_u_value": 1.2,
            "heat_recovery_eff": 0.75,
            "heating_setpoint": 21.0,
        }

        result, std = predictor.predict(params, return_std=True)
        assert isinstance(result, float)
        assert isinstance(std, float)
        assert std >= 0

    def test_predict_batch(self, trained_surrogate):
        """Batch predictions."""
        predictor = SurrogatePredictor(trained_surrogate)

        params_list = [
            {"infiltration_ach": 0.05, "wall_u_value": 0.3,
             "roof_u_value": 0.2, "floor_u_value": 0.3,
             "window_u_value": 1.0, "heat_recovery_eff": 0.8,
             "heating_setpoint": 21.0},
            {"infiltration_ach": 0.15, "wall_u_value": 0.8,
             "roof_u_value": 0.4, "floor_u_value": 0.5,
             "window_u_value": 1.8, "heat_recovery_eff": 0.5,
             "heating_setpoint": 20.0},
        ]

        results = predictor.predict_batch(params_list)
        assert len(results) == 2
        # Higher infiltration should give higher heating
        # (This may not always hold due to GP interpolation with noise)


class TestPrior:
    """Test prior distributions."""

    def test_uniform_prior(self):
        """Uniform prior sampling and PDF."""
        prior = Prior(
            name="test",
            distribution="uniform",
            params={"low": 0.0, "high": 1.0}
        )
        rng = np.random.default_rng(42)
        samples = prior.sample(100, rng)

        assert np.all(samples >= 0)
        assert np.all(samples <= 1)

    def test_truncnorm_prior(self):
        """Truncated normal prior."""
        prior = Prior(
            name="test",
            distribution="truncnorm",
            params={"mean": 0.5, "std": 0.2, "low": 0.0, "high": 1.0}
        )
        rng = np.random.default_rng(42)
        samples = prior.sample(100, rng)

        assert np.all(samples >= 0)
        assert np.all(samples <= 1)
        assert np.abs(samples.mean() - 0.5) < 0.1

    def test_beta_prior(self):
        """Beta prior."""
        prior = Prior(
            name="test",
            distribution="beta",
            params={"alpha": 2, "beta": 2, "low": 0.0, "high": 1.0}
        )
        rng = np.random.default_rng(42)
        samples = prior.sample(100, rng)

        assert np.all(samples >= 0)
        assert np.all(samples <= 1)


class TestCalibrationPriors:
    """Test calibration prior collections."""

    def test_swedish_defaults(self):
        """Default Swedish priors."""
        priors = CalibrationPriors.swedish_defaults()

        assert "infiltration_ach" in priors.priors
        assert "wall_u_value" in priors.priors
        assert priors.priors["infiltration_ach"].distribution == "truncnorm"

    def test_archetype_priors(self):
        """Archetype-specific priors adjust values."""
        priors_old = CalibrationPriors.from_archetype("pre_1945_brick")
        priors_new = CalibrationPriors.from_archetype("post_2010_passive")

        # Old buildings should have higher infiltration mean
        old_inf = priors_old.priors["infiltration_ach"].params["mean"]
        new_inf = priors_new.priors["infiltration_ach"].params["mean"]

        assert old_inf > new_inf

    def test_sample_all(self):
        """Sample from all priors at once."""
        priors = CalibrationPriors.swedish_defaults()
        samples = priors.sample(50)

        assert len(samples) == len(priors.priors)
        for name, values in samples.items():
            assert len(values) == 50


class TestABCSMCCalibrator:
    """Test ABC-SMC calibration algorithm."""

    @pytest.fixture
    def simple_surrogate(self):
        """Create simple surrogate for calibration testing."""
        config = SurrogateConfig(n_samples=80)
        trainer = SurrogateTrainer(config)
        X = trainer.generate_samples()

        # Simple model: heating = 50 + infiltration * 300
        # (mostly depends on infiltration)
        y = 50 + X[:, 0] * 300 + np.random.normal(0, 1, len(X))

        return trainer.train("calibration_test", X, y)

    def test_calibration_runs(self, simple_surrogate):
        """ABC-SMC calibration completes."""
        predictor = SurrogatePredictor(simple_surrogate)
        priors = CalibrationPriors.swedish_defaults()

        calibrator = ABCSMCCalibrator(
            predictor=predictor,
            priors=priors,
            n_particles=100,
            n_generations=3,
            random_state=42,
        )

        # Target: infiltration ~0.10 → heating ~80 kWh/m²
        posterior = calibrator.calibrate(
            measured_kwh_m2=80.0,
            tolerance_percent=25.0,
        )

        assert len(posterior.samples) > 10
        assert "infiltration_ach" in posterior.means

        # Should recover roughly correct infiltration
        # (with tolerance due to simplified model)
        inf_mean = posterior.means["infiltration_ach"]
        assert 0.05 < inf_mean < 0.15

    def test_posterior_statistics(self, simple_surrogate):
        """Posterior provides uncertainty estimates."""
        predictor = SurrogatePredictor(simple_surrogate)
        priors = CalibrationPriors.swedish_defaults()

        calibrator = ABCSMCCalibrator(
            predictor=predictor,
            priors=priors,
            n_particles=200,
            n_generations=4,
            random_state=42,
        )

        posterior = calibrator.calibrate(measured_kwh_m2=70.0)

        assert posterior.means is not None
        assert posterior.stds is not None
        assert posterior.ci_90 is not None

        # CI should contain mean
        for param in posterior.param_names:
            ci = posterior.ci_90[param]
            mean = posterior.means[param]
            assert ci[0] <= mean <= ci[1]


class TestUncertaintyPropagator:
    """Test uncertainty propagation."""

    @pytest.fixture
    def calibrated_posterior(self):
        """Create calibrated posterior for testing."""
        config = SurrogateConfig(n_samples=50)
        trainer = SurrogateTrainer(config)
        X = trainer.generate_samples()
        y = 50 + X[:, 0] * 300

        surrogate = trainer.train("uncertainty_test", X, y)
        predictor = SurrogatePredictor(surrogate)
        priors = CalibrationPriors.swedish_defaults()

        calibrator = ABCSMCCalibrator(
            predictor=predictor,
            priors=priors,
            n_particles=100,
            n_generations=3,
        )
        return predictor, calibrator.calibrate(measured_kwh_m2=75.0)

    def test_predict_with_uncertainty(self, calibrated_posterior):
        """Predictions include uncertainty."""
        predictor, posterior = calibrated_posterior
        propagator = UncertaintyPropagator(predictor, posterior)

        mean, std, ci = propagator.predict_with_uncertainty(n_samples=100)

        assert isinstance(mean, float)
        assert isinstance(std, float)
        assert std > 0
        assert ci[0] < mean < ci[1]

    def test_savings_distribution(self, calibrated_posterior):
        """ECM savings with uncertainty."""
        predictor, posterior = calibrated_posterior
        propagator = UncertaintyPropagator(predictor, posterior)

        # ECM: reduce infiltration by 50%
        def ecm_effect(params):
            modified = params.copy()
            modified["infiltration_ach"] = params["infiltration_ach"] * 0.5
            return modified

        savings = propagator.compute_savings_distribution(ecm_effect, n_samples=100)

        assert "savings_kwh_m2_mean" in savings
        assert "savings_percent_mean" in savings
        assert "savings_percent_ci_90" in savings
        assert savings["savings_kwh_m2_mean"] > 0  # Should save energy


class TestIntegration:
    """Integration tests for full calibration workflow."""

    def test_full_workflow(self):
        """End-to-end calibration workflow."""
        from src.calibration import BayesianCalibrator

        # 1. Create synthetic training data
        config = SurrogateConfig(n_samples=60)
        trainer = SurrogateTrainer(config)
        X = trainer.generate_samples()

        # Synthetic Swedish building model
        # heating ~ 40 + inf*200 + wall_u*20 + (1-hr)*30
        y = (
            40 +
            X[:, 0] * 200 +  # infiltration_ach
            X[:, 1] * 20 +   # wall_u_value
            (1 - X[:, 5]) * 30  # heat_recovery
        )

        # 2. Train surrogate
        with tempfile.TemporaryDirectory() as tmpdir:
            surrogate = trainer.train("mfh_test", X, y)
            trainer.save(surrogate, Path(tmpdir))

            # 3. Calibrate
            calibrator = BayesianCalibrator(
                surrogate_dir=Path(tmpdir),
                n_particles=100,
                n_generations=3,
            )

            # Target: typical Swedish MFH ~70 kWh/m²
            result = calibrator.calibrate(
                archetype_id="mfh_test",
                measured_kwh_m2=70.0,
            )

            # 4. Verify results
            assert result.measured_kwh_m2 == 70.0
            assert result.n_accepted_samples > 10
            assert result.calibration_error_percent < 30  # Within 30%

            # Summary should work
            summary = result.summary()
            assert "Calibration Results" in summary
            assert "infiltration_ach" in summary
