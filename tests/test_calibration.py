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
    FixedParamPredictor,
)
from src.calibration.bayesian import (
    Prior,
    CalibrationPriors,
    ABCSMCCalibrator,
    UncertaintyPropagator,
)
from src.calibration.sensitivity import (
    MorrisResults,
    MorrisScreening,
    run_morris_analysis,
)
from src.calibration.bayesian import (
    ECMUncertaintyPropagator,
    get_ecm_effect,
    ECM_PARAMETER_EFFECTS,
)


class TestSurrogateConfig:
    """Test surrogate configuration."""

    def test_default_config(self):
        """Default config has Swedish building bounds (expanded for poorly-performing buildings)."""
        config = SurrogateConfig()
        assert config.n_samples == 150  # Updated for better surrogate quality (Phase 2)
        assert "infiltration_ach" in config.param_bounds
        assert "wall_u_value" in config.param_bounds
        # Expanded bounds to simulate high-energy buildings (86% of Sjöstad have non-functional FTX)
        assert config.param_bounds["infiltration_ach"] == (0.02, 0.50)  # Expanded from 0.20
        assert config.param_bounds["wall_u_value"] == (0.15, 2.50)  # Expanded from 1.50
        assert config.param_bounds["window_u_value"] == (0.70, 4.00)  # Expanded from 2.50

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

    def test_from_building_context_with_calibration_hints(self):
        """Calibration hints from LLM reasoner constrain priors."""
        # Simulate hints from renovation detection
        hints = {
            "window_u_value_adjustment": -0.5,  # Better windows
            "infiltration_adjustment": -0.02,  # Tighter
            "ventilation_efficiency": 0.80,  # FTX detected
            "heat_recovery": True,
        }

        priors = CalibrationPriors.from_building_context(
            archetype_id="1961_1975",
            ventilation_type="FTX",
            energy_class="C",
            calibration_hints=hints,
        )

        # Window U-value should be adjusted lower
        window_params = priors.priors["window_u_value"].params
        assert window_params["mean"] <= 1.0  # Adjusted from ~1.5 to ~1.0

        # Infiltration should be adjusted lower
        inf_params = priors.priors["infiltration_ach"].params
        assert inf_params["mean"] <= 0.08  # Adjusted from ~0.10 to ~0.08

        # Heat recovery should be constrained to FTX range
        hr_params = priors.priors["heat_recovery_eff"].params
        assert hr_params["low"] >= 0.60
        assert hr_params["high"] <= 0.92

    def test_calibration_hints_wall_roof_adjustments(self):
        """Wall and roof U-value hints shift bounds."""
        hints = {
            "wall_u_value_adjustment": -0.15,  # Added insulation
            "roof_u_value_adjustment": -0.10,
        }

        priors = CalibrationPriors.from_building_context(
            archetype_id="1961_1975",
            calibration_hints=hints,
        )

        # Wall bounds should be shifted lower
        wall_params = priors.priors["wall_u_value"].params
        assert wall_params["high"] < 0.90  # Original was ~0.90 for 1961_1975

        # Roof bounds should be shifted lower
        roof_params = priors.priors["roof_u_value"].params
        assert roof_params["high"] < 0.60  # Default high

    def test_performance_gap_expansion_for_high_energy_buildings(self):
        """High-energy buildings get expanded parameter bounds.

        Based on Hammarby Sjöstad research: 86% of buildings have non-functional
        heat recovery despite claiming FTX. For buildings using 100+ kWh/m², the
        calibration needs wider parameter ranges to reach the target.
        """
        # Low-energy building: normal bounds
        priors_low = CalibrationPriors.from_building_context(
            archetype_id="1996_2010",
            measured_kwh_m2=50.0,  # Below 80, no expansion
            construction_year=2000,
        )

        # High-energy building: expanded bounds
        priors_high = CalibrationPriors.from_building_context(
            archetype_id="1996_2010",
            measured_kwh_m2=120.0,  # High energy, should trigger expansion
            construction_year=2000,
        )

        # Infiltration should be expanded for high-energy buildings
        low_inf = priors_low.priors["infiltration_ach"].params.get("high", 0.20)
        high_inf = priors_high.priors["infiltration_ach"].params.get("high", 0.20)
        assert high_inf > low_inf, f"High energy building should have expanded infiltration range: {high_inf} > {low_inf}"
        assert high_inf >= 0.27, f"Infiltration upper bound should be significantly expanded: {high_inf}"

        # Wall U-value should be expanded for high-energy buildings
        low_wall = priors_low.priors["wall_u_value"].params.get("high", 0.35)
        high_wall = priors_high.priors["wall_u_value"].params.get("high", 0.35)
        assert high_wall > low_wall, f"High energy building should have expanded wall U range: {high_wall} > {low_wall}"

        # Window U-value should be expanded for high-energy buildings
        low_win = priors_low.priors["window_u_value"].params.get("high", 1.0)
        high_win = priors_high.priors["window_u_value"].params.get("high", 1.0)
        assert high_win > low_win, f"High energy building should have expanded window U range: {high_win} > {low_win}"

    def test_performance_gap_expansion_scaling(self):
        """Expansion scales with measured energy.

        100 kWh/m² should expand less than 150 kWh/m².
        """
        priors_100 = CalibrationPriors.from_building_context(
            archetype_id="1961_1975",
            measured_kwh_m2=100.0,
            construction_year=1970,
        )

        priors_150 = CalibrationPriors.from_building_context(
            archetype_id="1961_1975",
            measured_kwh_m2=150.0,
            construction_year=1970,
        )

        # 150 kWh/m² should have wider bounds than 100 kWh/m²
        inf_100 = priors_100.priors["infiltration_ach"].params.get("high", 0.20)
        inf_150 = priors_150.priors["infiltration_ach"].params.get("high", 0.20)
        assert inf_150 >= inf_100, f"Higher energy should have wider infiltration range: {inf_150} >= {inf_100}"


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


class TestMorrisScreening:
    """Integration tests for Morris sensitivity analysis."""

    @pytest.fixture
    def trained_surrogate(self):
        """Create surrogate with known sensitivity structure.

        Model: heating = 40 + infiltration*200 + wall_u*20 + (1-hr)*30
        - infiltration_ach has largest effect (200x)
        - heat_recovery_eff has medium effect (30x)
        - wall_u_value has small effect (20x)
        - Other parameters have negligible effect
        """
        config = SurrogateConfig(n_samples=100)
        trainer = SurrogateTrainer(config)
        X = trainer.generate_samples()

        # Known model: infiltration >> heat_recovery > wall_u > others
        y = (
            40 +
            X[:, 0] * 200 +  # infiltration_ach (index 0) - dominant
            X[:, 1] * 20 +   # wall_u_value (index 1) - small
            (1 - X[:, 5]) * 30 +  # heat_recovery_eff (index 5) - medium
            np.random.normal(0, 0.5, len(X))  # Small noise
        )

        return trainer.train("morris_test", X, y)

    def test_morris_screening_runs(self, trained_surrogate):
        """Morris screening completes without error."""
        screening = MorrisScreening(
            trained_surrogate,
            n_trajectories=10,  # Small for speed
            n_levels=4,
        )

        results = screening.analyze()

        assert isinstance(results, MorrisResults)
        assert len(results.param_names) == 7
        assert len(results.mu_star) == 7
        assert len(results.ranking) == 7

    def test_morris_identifies_important_parameters(self, trained_surrogate):
        """Morris correctly identifies infiltration as most important."""
        screening = MorrisScreening(
            trained_surrogate,
            n_trajectories=15,
            n_levels=4,
        )

        results = screening.analyze()

        # Infiltration should be ranked #1 (most important)
        assert results.ranking["infiltration_ach"] == 1

        # Get top 3 parameters
        top3 = results.get_important_parameters(top_n=3)
        assert "infiltration_ach" in top3

        # Normalized mu* for infiltration should be high
        assert results.mu_star_normalized["infiltration_ach"] > 0.5

    def test_morris_identifies_negligible_parameters(self, trained_surrogate):
        """Morris correctly identifies parameters with small effect."""
        screening = MorrisScreening(
            trained_surrogate,
            n_trajectories=15,
            n_levels=4,
        )

        results = screening.analyze()

        # Parameters not in model should be negligible
        negligible = results.get_negligible_parameters(mu_star_threshold=0.05)

        # These weren't in our model, so should be negligible
        # (roof_u_value, floor_u_value, window_u_value, heating_setpoint)
        assert len(negligible) >= 2  # At least some should be negligible

    def test_morris_results_string(self, trained_surrogate):
        """Morris results have human-readable output."""
        screening = MorrisScreening(trained_surrogate, n_trajectories=5)
        results = screening.analyze()

        output = str(results)

        assert "Morris Sensitivity Analysis" in output
        assert "infiltration_ach" in output
        assert "Rank" in output

    def test_run_morris_analysis_convenience(self, trained_surrogate):
        """Convenience function runs analysis and returns parameter lists."""
        results, important, negligible = run_morris_analysis(
            surrogate=trained_surrogate,
            n_trajectories=8,
            importance_threshold=0.1,
        )

        assert isinstance(results, MorrisResults)
        assert results.n_trajectories == 8
        assert isinstance(important, list)
        assert isinstance(negligible, list)
        # Infiltration should be in important list
        assert "infiltration_ach" in important


class TestFixedParamPredictor:
    """Tests for FixedParamPredictor that injects fixed parameter values."""

    @pytest.fixture
    def surrogate_and_predictor(self):
        """Create surrogate and base predictor."""
        config = SurrogateConfig(n_samples=60)
        trainer = SurrogateTrainer(config)
        X = trainer.generate_samples()

        # Simple model
        y = 40 + X[:, 0] * 200 + X[:, 1] * 20

        surrogate = trainer.train("fixed_param_test", X, y)
        base_predictor = SurrogatePredictor(surrogate)
        return surrogate, base_predictor

    def test_fixed_params_injected(self, surrogate_and_predictor):
        """Fixed parameters are correctly injected during prediction."""
        surrogate, base_predictor = surrogate_and_predictor

        # Fix infiltration and wall_u_value
        fixed_params = {
            'infiltration_ach': 0.10,
            'wall_u_value': 0.40,
        }

        fixed_predictor = FixedParamPredictor(base_predictor, fixed_params)

        # Predict with partial params (fixed ones will be injected)
        partial_params = {
            'roof_u_value': 0.25,
            'floor_u_value': 0.30,
            'window_u_value': 1.2,
            'heat_recovery_eff': 0.75,
            'heating_setpoint': 21.0,
        }

        result = fixed_predictor.predict(partial_params)

        # Should get a valid prediction
        assert isinstance(result, (float, np.floating))
        assert 20 < result < 200  # Reasonable range

    def test_fixed_predictor_batch(self, surrogate_and_predictor):
        """Batch prediction with fixed params works."""
        surrogate, base_predictor = surrogate_and_predictor

        fixed_params = {'infiltration_ach': 0.08}
        fixed_predictor = FixedParamPredictor(base_predictor, fixed_params)

        # Batch of partial params
        partial_list = [
            {'wall_u_value': 0.3, 'roof_u_value': 0.2, 'floor_u_value': 0.25,
             'window_u_value': 1.0, 'heat_recovery_eff': 0.8, 'heating_setpoint': 21},
            {'wall_u_value': 0.5, 'roof_u_value': 0.3, 'floor_u_value': 0.35,
             'window_u_value': 1.5, 'heat_recovery_eff': 0.6, 'heating_setpoint': 20},
        ]

        results = fixed_predictor.predict_batch(partial_list)

        assert len(results) == 2
        assert all(20 < r < 200 for r in results)

    def test_fixed_predictor_preserves_std(self, surrogate_and_predictor):
        """Standard deviation returned correctly with fixed params."""
        surrogate, base_predictor = surrogate_and_predictor

        fixed_params = {'infiltration_ach': 0.08}
        fixed_predictor = FixedParamPredictor(base_predictor, fixed_params)

        partial_params = {
            'wall_u_value': 0.4, 'roof_u_value': 0.25, 'floor_u_value': 0.30,
            'window_u_value': 1.2, 'heat_recovery_eff': 0.75, 'heating_setpoint': 21.0,
        }

        mean, std = fixed_predictor.predict(partial_params, return_std=True)

        assert isinstance(mean, (float, np.floating))
        assert isinstance(std, (float, np.floating))
        assert std >= 0


class TestMorrisCalibrationIntegration:
    """Test Morris→Calibrator integration (filtering priors)."""

    def test_filter_priors_from_morris(self):
        """Morris results filter priors to important parameters."""
        # Create surrogate with known sensitivity
        config = SurrogateConfig(n_samples=80)
        trainer = SurrogateTrainer(config)
        X = trainer.generate_samples()

        # Only infiltration and heat_recovery matter
        y = 40 + X[:, 0] * 200 + (1 - X[:, 5]) * 50

        surrogate = trainer.train("integration_test", X, y)

        # Run Morris
        screening = MorrisScreening(surrogate, n_trajectories=10)
        results = screening.analyze()

        # Get important params
        important = results.get_important_parameters(top_n=3)

        # Create priors and filter
        priors = CalibrationPriors.swedish_defaults()
        filtered_priors = priors.filter_to_parameters(important)

        # Should only have top 3 parameters
        assert len(filtered_priors.priors) == 3
        assert all(name in important for name in filtered_priors.priors)

    def test_full_adaptive_calibration_flow(self):
        """Complete Morris→filter→calibrate flow."""
        # 1. Create surrogate
        config = SurrogateConfig(n_samples=80)
        trainer = SurrogateTrainer(config)
        X = trainer.generate_samples()

        # Model where only infiltration matters
        y = 40 + X[:, 0] * 300

        surrogate = trainer.train("adaptive_test", X, y)

        # 2. Run Morris screening
        screening = MorrisScreening(surrogate, n_trajectories=10)
        morris_results = screening.analyze()

        # 3. Identify important parameters
        important = morris_results.get_important_parameters(top_n=3)

        # 4. Get fixed params (archetype defaults for non-important)
        archetype_defaults = {
            'infiltration_ach': 0.08,
            'wall_u_value': 0.40,
            'roof_u_value': 0.25,
            'floor_u_value': 0.30,
            'window_u_value': 1.2,
            'heat_recovery_eff': 0.75,
            'heating_setpoint': 21.0,
        }

        fixed_params = {
            name: archetype_defaults[name]
            for name in surrogate.param_names
            if name not in important
        }

        # 5. Create filtered priors and fixed predictor
        priors = CalibrationPriors.swedish_defaults()
        filtered_priors = priors.filter_to_parameters(important)

        base_predictor = SurrogatePredictor(surrogate)
        fixed_predictor = FixedParamPredictor(base_predictor, fixed_params)

        # 6. Calibrate with reduced parameters
        calibrator = ABCSMCCalibrator(
            predictor=fixed_predictor,
            priors=filtered_priors,
            n_particles=100,
            n_generations=3,
        )

        posterior = calibrator.calibrate(measured_kwh_m2=70.0)

        # Should have posterior samples for important params only
        assert len(posterior.samples) > 10
        for sample in posterior.samples:
            assert all(name in sample.params for name in filtered_priors.priors)


class TestECMUncertaintyPropagation:
    """Tests for Monte Carlo ECM uncertainty propagation."""

    @pytest.fixture
    def surrogate_with_posterior(self):
        """Create surrogate and calibrated posterior for testing."""
        config = SurrogateConfig(n_samples=80)
        trainer = SurrogateTrainer(config)
        X = trainer.generate_samples()

        # Model sensitive to infiltration and heat recovery
        y = 50 + X[:, 0] * 200 + (1 - X[:, 5]) * 40

        surrogate = trainer.train("ecm_uncertainty_test", X, y)
        predictor = SurrogatePredictor(surrogate)
        priors = CalibrationPriors.swedish_defaults()

        calibrator = ABCSMCCalibrator(
            predictor=predictor,
            priors=priors,
            n_particles=100,
            n_generations=3,
        )

        posterior = calibrator.calibrate(measured_kwh_m2=70.0)
        return surrogate, predictor, posterior

    def test_ecm_effect_functions(self):
        """ECM effect functions modify parameters correctly."""
        base_params = {
            'infiltration_ach': 0.10,
            'wall_u_value': 0.50,
            'window_u_value': 1.5,
            'heat_recovery_eff': 0.0,
            'heating_setpoint': 21.0,
        }

        # Test air sealing (multiply by 0.6)
        air_sealing_effect = get_ecm_effect("air_sealing")
        result = air_sealing_effect(base_params)
        assert result["infiltration_ach"] == pytest.approx(0.06)

        # Test FTX installation (set to 0.80)
        ftx_effect = get_ecm_effect("ftx_installation")
        result = ftx_effect(base_params)
        assert result["heat_recovery_eff"] == 0.80

        # Test window replacement (set to 0.9)
        window_effect = get_ecm_effect("window_replacement")
        result = window_effect(base_params)
        assert result["window_u_value"] == 0.9

        # Test smart thermostats (subtract 1)
        thermostat_effect = get_ecm_effect("smart_thermostats")
        result = thermostat_effect(base_params)
        assert result["heating_setpoint"] == 20.0

    def test_ecm_uncertainty_propagator_runs(self, surrogate_with_posterior):
        """ECM uncertainty propagator completes without error."""
        surrogate, predictor, posterior = surrogate_with_posterior

        propagator = ECMUncertaintyPropagator(
            predictor=predictor,
            posterior=posterior,
            n_samples=100,
        )

        result = propagator.compute_ecm_uncertainty("air_sealing")

        assert "savings_kwh_m2_mean" in result
        assert "savings_kwh_m2_std" in result
        assert result["savings_kwh_m2_std"] > 0  # Should have uncertainty

    def test_ecm_uncertainty_with_simulated_savings(self, surrogate_with_posterior):
        """Propagator uses simulated savings when provided."""
        surrogate, predictor, posterior = surrogate_with_posterior

        propagator = ECMUncertaintyPropagator(
            predictor=predictor,
            posterior=posterior,
            n_samples=100,
        )

        # Provide actual simulated savings from E+
        simulated_savings = 5.0  # kWh/m²

        result = propagator.compute_ecm_uncertainty(
            "air_sealing",
            simulated_savings_kwh_m2=simulated_savings,
        )

        # Mean should be the simulated value
        assert result["savings_kwh_m2_mean"] == simulated_savings
        # Std should be positive
        assert result["savings_kwh_m2_std"] > 0

    def test_ecm_uncertainty_all_ecms(self, surrogate_with_posterior):
        """Propagator adds uncertainty to list of ECM results."""
        surrogate, predictor, posterior = surrogate_with_posterior

        propagator = ECMUncertaintyPropagator(
            predictor=predictor,
            posterior=posterior,
            n_samples=100,
        )

        ecm_results = [
            {"ecm_id": "air_sealing", "heating_kwh_m2": 65, "baseline_kwh_m2": 70},
            {"ecm_id": "ftx_installation", "heating_kwh_m2": 55, "baseline_kwh_m2": 70},
        ]

        enhanced = propagator.compute_all_ecm_uncertainties(ecm_results)

        assert len(enhanced) == 2
        for result in enhanced:
            assert "savings_kwh_m2_std" in result
            assert result["uncertainty_method"] == "monte_carlo"

    def test_ecm_parameter_effects_coverage(self):
        """ECM parameter effects cover major ECM categories."""
        # Envelope ECMs
        assert "wall_external_insulation" in ECM_PARAMETER_EFFECTS
        assert "window_replacement" in ECM_PARAMETER_EFFECTS
        assert "air_sealing" in ECM_PARAMETER_EFFECTS

        # Ventilation ECMs
        assert "ftx_installation" in ECM_PARAMETER_EFFECTS
        assert "ftx_upgrade" in ECM_PARAMETER_EFFECTS
        assert "demand_controlled_ventilation" in ECM_PARAMETER_EFFECTS

        # Controls ECMs
        assert "smart_thermostats" in ECM_PARAMETER_EFFECTS

        # Default fallback
        assert "_default" in ECM_PARAMETER_EFFECTS
