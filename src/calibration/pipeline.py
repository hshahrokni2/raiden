"""
Bayesian Calibration Pipeline - High-level interface for model calibration.

This wraps the surrogate training and ABC-SMC calibration into a simple
pipeline that can be used by FullPipelineAnalyzer.

Workflow:
1. Check for cached surrogate (per archetype)
2. If not cached: train surrogate with Latin Hypercube sampling
3. Run ABC-SMC calibration to get posterior
4. Return calibrated parameters with uncertainty

Usage:
    pipeline = BayesianCalibrationPipeline(
        runner=EnergyPlusRunner(),
        weather_path=Path('./weather.epw'),
        cache_dir=Path('./cache'),
    )

    result = pipeline.calibrate(
        baseline_idf=Path('./baseline.idf'),
        archetype_id='1996_2010',
        measured_kwh_m2=53.0,
        atemp_m2=15350,
    )

    print(f"Infiltration: {result.means['infiltration_ach']:.3f} ± {result.stds['infiltration_ach']:.3f}")
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import logging
import json
import shutil
import tempfile

import numpy as np

from .surrogate import SurrogateTrainer, SurrogatePredictor, TrainedSurrogate, SurrogateConfig, FixedParamPredictor
from .bayesian import ABCSMCCalibrator, CalibrationPriors, CalibrationPosterior, UncertaintyPropagator
from .metrics import CalibrationMetrics, compute_uncertainty_adjusted_metrics
from .sensitivity import MorrisScreening, MorrisResults, run_morris_analysis

logger = logging.getLogger(__name__)


@dataclass
class CalibrationResult:
    """Result of Bayesian calibration with uncertainty."""

    # Point estimates
    calibrated_kwh_m2: float
    calibrated_params: Dict[str, float]

    # Uncertainty
    kwh_m2_std: float
    kwh_m2_ci_90: Tuple[float, float]
    param_stds: Dict[str, float]
    param_ci_90: Dict[str, Tuple[float, float]]

    # Posterior for downstream use
    posterior: Optional[CalibrationPosterior] = None

    # Metadata
    archetype_id: str = ""
    measured_kwh_m2: float = 0.0
    calibration_error: float = 0.0  # |measured - calibrated|
    n_posterior_samples: int = 0
    surrogate_r2: float = 0.0  # Training R² (may overfit)
    surrogate_test_r2: float = 0.0  # Test R² (true generalization)
    surrogate_is_overfit: bool = False  # Warning flag

    # ASHRAE Guideline 14 metrics
    ashrae_nmbe: float = 0.0  # Normalized Mean Bias Error (%)
    ashrae_cvrmse: float = 0.0  # CV of RMSE (%)
    ashrae_passes: bool = False  # Whether model passes ASHRAE criteria
    ashrae_pass_probability: float = 0.0  # Probability of passing given uncertainty

    # Morris sensitivity analysis
    morris_results: Optional[MorrisResults] = None
    calibrated_param_list: Optional[List[str]] = None  # Parameters that were calibrated
    fixed_param_values: Optional[Dict[str, float]] = None  # Parameters fixed at defaults

    # E+ Verification (ground truth check)
    eplus_verified: bool = False  # Whether E+ verification was run
    eplus_verified_kwh_m2: float = 0.0  # Actual E+ result with calibrated params
    surrogate_eplus_discrepancy: float = 0.0  # |surrogate - eplus| / eplus

    def to_dict(self) -> Dict[str, Any]:
        """Export to dictionary for JSON serialization."""
        return {
            "calibrated_kwh_m2": self.calibrated_kwh_m2,
            "calibrated_kwh_m2_std": self.kwh_m2_std,
            "calibrated_kwh_m2_ci_90": list(self.kwh_m2_ci_90),
            "calibrated_params": self.calibrated_params,
            "param_uncertainties": {
                name: {
                    "mean": self.calibrated_params[name],
                    "std": self.param_stds.get(name, 0),
                    "ci_90": list(self.param_ci_90.get(name, (0, 0))),
                }
                for name in self.calibrated_params
            },
            "measured_kwh_m2": self.measured_kwh_m2,
            "calibration_error_percent": 100 * self.calibration_error / self.measured_kwh_m2 if self.measured_kwh_m2 > 0 else 0,
            "archetype_id": self.archetype_id,
            "n_posterior_samples": self.n_posterior_samples,
            "surrogate_r2": self.surrogate_r2,
            "surrogate_test_r2": self.surrogate_test_r2,
            "surrogate_is_overfit": self.surrogate_is_overfit,
            # ASHRAE Guideline 14 metrics
            "ashrae_nmbe": self.ashrae_nmbe,
            "ashrae_cvrmse": self.ashrae_cvrmse,
            "ashrae_passes": self.ashrae_passes,
            "ashrae_pass_probability": self.ashrae_pass_probability,
            # Morris sensitivity
            "morris_ranking": self.morris_results.ranking if self.morris_results else None,
            "calibrated_param_list": self.calibrated_param_list,
            "fixed_param_values": self.fixed_param_values,
            # E+ verification
            "eplus_verified": self.eplus_verified,
            "eplus_verified_kwh_m2": self.eplus_verified_kwh_m2,
            "surrogate_eplus_discrepancy": self.surrogate_eplus_discrepancy,
        }


class BayesianCalibrationPipeline:
    """
    End-to-end Bayesian calibration pipeline.

    Manages surrogate training, caching, and ABC-SMC inference.
    """

    def __init__(
        self,
        runner,  # EnergyPlusRunner or SimulationRunner
        weather_path: Path,
        cache_dir: Path = None,
        n_surrogate_samples: int = 200,  # Increased from 100 (literature: 10-20 per parameter)
        n_abc_particles: int = 500,
        n_abc_generations: int = 8,
        parallel_sims: int = 4,
        use_adaptive_calibration: bool = True,  # Use Morris screening
        min_calibration_params: int = 3,
        max_calibration_params: int = 5,
        verify_with_eplus: bool = False,  # Run final E+ verification after calibration
    ):
        """
        Initialize calibration pipeline.

        Args:
            runner: Simulation runner for E+ execution
            weather_path: Path to EPW weather file
            cache_dir: Directory for caching surrogates (None = no caching)
            n_surrogate_samples: LHS samples for surrogate training
            n_abc_particles: Particles for ABC-SMC
            n_abc_generations: SMC generations
            parallel_sims: Parallel E+ simulations
            use_adaptive_calibration: Use Morris screening to focus on important params
            min_calibration_params: Minimum parameters to calibrate (adaptive mode)
            max_calibration_params: Maximum parameters to calibrate (adaptive mode)
            verify_with_eplus: Run final E+ simulation with calibrated params to verify surrogate accuracy
        """
        self.runner = runner
        self.weather_path = Path(weather_path)
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.n_surrogate_samples = n_surrogate_samples
        self.n_abc_particles = n_abc_particles
        self.n_abc_generations = n_abc_generations
        self.parallel_sims = parallel_sims
        self.use_adaptive_calibration = use_adaptive_calibration
        self.min_calibration_params = min_calibration_params
        self.max_calibration_params = max_calibration_params
        self.verify_with_eplus = verify_with_eplus

        # Cached surrogates and Morris results
        self._surrogates: Dict[str, TrainedSurrogate] = {}
        self._morris_results: Dict[str, MorrisResults] = {}

        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def calibrate(
        self,
        baseline_idf: Path,
        archetype_id: str,
        measured_kwh_m2: float,
        atemp_m2: float,
        output_dir: Path = None,
        force_retrain: bool = False,
        # Context-aware prior constraints
        existing_measures: set = None,
        ventilation_type: str = None,
        heating_system: str = None,
        energy_class: str = None,
        calibration_hints: dict = None,
        construction_year: int = None,
        # Mixed-use building adjustments
        restaurant_pct: float = 0.0,
        commercial_pct: float = 0.0,
    ) -> CalibrationResult:
        """
        Run Bayesian calibration.

        Args:
            baseline_idf: Path to baseline IDF file
            archetype_id: Building archetype identifier
            measured_kwh_m2: Measured annual heating (target)
            atemp_m2: Heated floor area
            output_dir: Directory for outputs
            force_retrain: Force surrogate retraining
            existing_measures: Set of ExistingMeasure enums from building context
            ventilation_type: Detected ventilation type (F, FT, FTX, S)
            heating_system: Detected heating system
            energy_class: Energy declaration class (A-G)
            calibration_hints: Dict from LLM archetype reasoner (renovation hints)

        Returns:
            CalibrationResult with parameters and uncertainty
        """
        logger.info(f"Starting Bayesian calibration for {archetype_id}")
        logger.info(f"Target: {measured_kwh_m2} kWh/m², Area: {atemp_m2} m²")

        # Get or train surrogate
        surrogate = self._get_surrogate(
            baseline_idf=baseline_idf,
            archetype_id=archetype_id,
            atemp_m2=atemp_m2,
            output_dir=output_dir,
            force_retrain=force_retrain,
        )

        # Run Morris screening if enabled
        morris_results = None
        calibrated_params = None
        fixed_params = None

        if self.use_adaptive_calibration:
            logger.info("Running Morris sensitivity screening...")
            morris_results = self._get_morris_results(surrogate, archetype_id)

            # Determine which parameters to calibrate
            important = morris_results.get_important_parameters(
                mu_star_threshold=0.1,
                top_n=self.max_calibration_params,
            )

            # Ensure minimum number of parameters
            if len(important) < self.min_calibration_params:
                sorted_params = sorted(morris_results.ranking.items(), key=lambda x: x[1])
                important = [p for p, _ in sorted_params[:self.min_calibration_params]]

            calibrated_params = important
            fixed_params = {
                name: self._get_archetype_default(archetype_id, name)
                for name in surrogate.param_names
                if name not in calibrated_params
            }

            logger.info(f"Calibrating {len(calibrated_params)} parameters: {calibrated_params}")
            logger.info(f"Fixed {len(fixed_params)} parameters: {list(fixed_params.keys())}")

        # Create context-aware priors (Kennedy & O'Hagan best practice)
        # Now includes REALITY CHECK using actual measured energy to infer
        # real system performance (e.g., non-functional FTX)
        if any([existing_measures, ventilation_type, heating_system, energy_class, calibration_hints, measured_kwh_m2]):
            logger.info("Using context-aware priors (building data constraints + reality check)")
            priors = CalibrationPriors.from_building_context(
                archetype_id=archetype_id,
                existing_measures=existing_measures,
                ventilation_type=ventilation_type,
                heating_system=heating_system,
                energy_class=energy_class,
                calibration_hints=calibration_hints,
                measured_kwh_m2=measured_kwh_m2,  # For reality check
                construction_year=construction_year,  # For expected energy calculation
                restaurant_pct=restaurant_pct,  # For mixed-use adjustment
                commercial_pct=commercial_pct,  # For mixed-use adjustment
            )
        else:
            # Fall back to archetype-only priors
            priors = CalibrationPriors.from_archetype(archetype_id)

        # Create predictor and calibrator
        base_predictor = SurrogatePredictor(surrogate)

        # Apply Morris screening results: filter priors and wrap predictor
        if self.use_adaptive_calibration and calibrated_params:
            # Filter priors to only important parameters (Kennedy & O'Hagan guidance)
            priors = priors.filter_to_parameters(calibrated_params)

            # Wrap predictor to inject fixed parameter values
            if fixed_params:
                predictor = FixedParamPredictor(base_predictor, fixed_params)
            else:
                predictor = base_predictor
        else:
            predictor = base_predictor
            calibrated_params = None
            fixed_params = None

        calibrator = ABCSMCCalibrator(
            predictor=predictor,
            priors=priors,
            n_particles=self.n_abc_particles,
            n_generations=self.n_abc_generations,
        )

        # Run ABC-SMC
        logger.info("Running ABC-SMC calibration...")
        posterior = calibrator.calibrate(
            measured_kwh_m2=measured_kwh_m2,
            tolerance_percent=20.0,
        )

        logger.info(f"Calibration complete. {len(posterior.samples)} posterior samples")

        # Compute calibrated prediction with uncertainty
        propagator = UncertaintyPropagator(predictor, posterior)
        pred_mean, pred_std, pred_ci = propagator.predict_with_uncertainty(n_samples=500)

        # Compute ASHRAE Guideline 14 metrics with uncertainty
        ashrae_metrics, ashrae_prob = compute_uncertainty_adjusted_metrics(
            measured_kwh_m2=measured_kwh_m2,
            simulated_kwh_m2=pred_mean,
            simulated_std=pred_std,
        )

        # Optional: Run final E+ verification with calibrated parameters
        eplus_verified = False
        eplus_verified_kwh_m2 = 0.0
        surrogate_eplus_discrepancy = 0.0

        if self.verify_with_eplus and baseline_idf and output_dir:
            logger.info("Running E+ verification with calibrated parameters...")
            try:
                eplus_result = self._run_eplus_verification(
                    baseline_idf=baseline_idf,
                    calibrated_params=posterior.means,
                    output_dir=output_dir,
                    atemp_m2=atemp_m2,
                )
                if eplus_result is not None:
                    eplus_verified = True
                    eplus_verified_kwh_m2 = eplus_result
                    surrogate_eplus_discrepancy = abs(pred_mean - eplus_result) / eplus_result if eplus_result > 0 else 0

                    logger.info(f"E+ Verification: {eplus_result:.1f} kWh/m² (surrogate: {pred_mean:.1f})")
                    if surrogate_eplus_discrepancy > 0.10:
                        logger.warning(
                            f"⚠️ Surrogate-E+ discrepancy: {surrogate_eplus_discrepancy:.1%} "
                            f"(exceeds 10% threshold). Consider retraining surrogate."
                        )
                    else:
                        logger.info(f"✓ Surrogate-E+ discrepancy: {surrogate_eplus_discrepancy:.1%} (OK)")
            except Exception as e:
                logger.warning(f"E+ verification failed: {e}")

        # Build result
        result = CalibrationResult(
            calibrated_kwh_m2=pred_mean,
            calibrated_params=posterior.means,
            kwh_m2_std=pred_std,
            kwh_m2_ci_90=pred_ci,
            param_stds=posterior.stds,
            param_ci_90=posterior.ci_90,
            posterior=posterior,
            archetype_id=archetype_id,
            measured_kwh_m2=measured_kwh_m2,
            calibration_error=abs(pred_mean - measured_kwh_m2),
            n_posterior_samples=len(posterior.samples),
            surrogate_r2=surrogate.training_r2,
            surrogate_test_r2=surrogate.test_r2,
            surrogate_is_overfit=surrogate.is_overfit,
            # ASHRAE metrics
            ashrae_nmbe=ashrae_metrics.nmbe,
            ashrae_cvrmse=ashrae_metrics.cvrmse,
            ashrae_passes=ashrae_metrics.passes_ashrae,
            ashrae_pass_probability=ashrae_prob,
            # Morris sensitivity analysis
            morris_results=morris_results,
            calibrated_param_list=calibrated_params,
            fixed_param_values=fixed_params,
            # E+ verification
            eplus_verified=eplus_verified,
            eplus_verified_kwh_m2=eplus_verified_kwh_m2,
            surrogate_eplus_discrepancy=surrogate_eplus_discrepancy,
        )

        # Log ASHRAE compliance
        if ashrae_metrics.passes_ashrae:
            logger.info(f"✓ ASHRAE Guideline 14: PASSES (NMBE={ashrae_metrics.nmbe:+.1f}%, CVRMSE={ashrae_metrics.cvrmse:.1f}%)")
        else:
            logger.warning(f"⚠️ ASHRAE Guideline 14: FAILS (NMBE={ashrae_metrics.nmbe:+.1f}%, CVRMSE={ashrae_metrics.cvrmse:.1f}%)")
        logger.info(f"  Probability of passing (with uncertainty): {ashrae_prob:.1%}")

        # Log overfitting warning if detected
        if surrogate.is_overfit:
            logger.warning(
                f"⚠️ Surrogate overfitting detected: Train R²={surrogate.training_r2:.3f} vs "
                f"Test R²={surrogate.test_r2:.3f}. Consider more samples."
            )

        # Log results
        logger.info(f"Calibrated: {pred_mean:.1f} ± {pred_std:.1f} kWh/m²")
        logger.info(f"90% CI: [{pred_ci[0]:.1f}, {pred_ci[1]:.1f}] kWh/m²")
        for name, val in posterior.means.items():
            std = posterior.stds[name]
            logger.info(f"  {name}: {val:.4f} ± {std:.4f}")

        return result

    def _get_surrogate(
        self,
        baseline_idf: Path,
        archetype_id: str,
        atemp_m2: float,
        output_dir: Path = None,
        force_retrain: bool = False,
    ) -> TrainedSurrogate:
        """Get surrogate from cache or train new one."""

        # Check memory cache
        if archetype_id in self._surrogates and not force_retrain:
            logger.info(f"Using cached surrogate for {archetype_id}")
            return self._surrogates[archetype_id]

        # Check disk cache
        if self.cache_dir and not force_retrain:
            cache_path = self.cache_dir / f"surrogate_{archetype_id}.joblib"
            if cache_path.exists():
                logger.info(f"Loading surrogate from {cache_path}")
                surrogate = SurrogateTrainer.load(cache_path)
                self._surrogates[archetype_id] = surrogate
                return surrogate

        # Train new surrogate
        logger.info(f"Training new surrogate for {archetype_id} ({self.n_surrogate_samples} samples)")
        surrogate = self._train_surrogate(
            baseline_idf=baseline_idf,
            archetype_id=archetype_id,
            atemp_m2=atemp_m2,
            output_dir=output_dir,
        )

        # Cache it
        self._surrogates[archetype_id] = surrogate
        if self.cache_dir:
            trainer = SurrogateTrainer()
            trainer.save(surrogate, self.cache_dir)

        return surrogate

    def _train_surrogate(
        self,
        baseline_idf: Path,
        archetype_id: str,
        atemp_m2: float,
        output_dir: Path = None,
    ) -> TrainedSurrogate:
        """Train surrogate model with Latin Hypercube sampling."""
        from concurrent.futures import ThreadPoolExecutor, as_completed
        from ..ecm.idf_modifier import IDFModifier
        from ..simulation.results import ResultsParser

        # Create trainer with archetype-appropriate bounds
        config = SurrogateConfig(n_samples=self.n_surrogate_samples)

        # Adjust bounds based on archetype
        if "post_2010" in archetype_id.lower() or "2011" in archetype_id:
            config.param_bounds['infiltration_ach'] = (0.02, 0.08)
            config.param_bounds['wall_u_value'] = (0.10, 0.30)
            config.param_bounds['window_u_value'] = (0.70, 1.30)
        elif "1996" in archetype_id.lower() or "modern" in archetype_id.lower():
            config.param_bounds['infiltration_ach'] = (0.03, 0.10)
            config.param_bounds['wall_u_value'] = (0.15, 0.40)
            config.param_bounds['window_u_value'] = (0.80, 1.50)

        trainer = SurrogateTrainer(config)

        # Generate LHS samples
        samples = trainer.generate_samples()
        params_list = trainer.samples_to_dicts(samples)

        logger.info(f"Running {len(params_list)} E+ simulations for surrogate training...")

        # Run E+ simulations
        results = []
        work_dir = Path(output_dir or tempfile.mkdtemp()) / "surrogate_training"
        work_dir.mkdir(parents=True, exist_ok=True)

        modifier = IDFModifier()
        results_parser = ResultsParser()

        def run_single(idx: int, params: Dict) -> Tuple[int, float]:
            """Run single E+ simulation with given parameters."""
            try:
                # Copy baseline and modify
                sim_dir = work_dir / f"sample_{idx:03d}"
                sim_dir.mkdir(exist_ok=True)

                modified_idf = sim_dir / "model.idf"
                shutil.copy(baseline_idf, modified_idf)

                # Apply parameter modifications
                self._apply_params_to_idf(modified_idf, params)

                # Run simulation
                sim_result = self.runner.run(
                    idf_path=modified_idf,
                    weather_path=self.weather_path,
                    output_dir=sim_dir,
                )

                # Parse results using ResultsParser
                if sim_result and sim_result.success:
                    parsed = results_parser.parse(sim_dir)
                    if parsed and parsed.heating_kwh_m2 > 0:
                        return idx, parsed.heating_kwh_m2
                    else:
                        return idx, np.nan
                else:
                    return idx, np.nan

            except Exception as e:
                logger.warning(f"Sample {idx} failed: {e}")
                return idx, np.nan

        # Run simulations (parallel or sequential)
        y = np.zeros(len(params_list))

        with ThreadPoolExecutor(max_workers=self.parallel_sims) as executor:
            futures = {
                executor.submit(run_single, i, p): i
                for i, p in enumerate(params_list)
            }

            completed = 0
            for future in as_completed(futures):
                idx, heating = future.result()
                y[idx] = heating
                completed += 1
                if completed % 10 == 0:
                    logger.info(f"  Completed {completed}/{len(params_list)} simulations")

        # Filter out failed simulations
        valid_mask = ~np.isnan(y)
        X_valid = samples[valid_mask]
        y_valid = y[valid_mask]

        logger.info(f"Training surrogate on {len(y_valid)}/{len(y)} valid samples")

        # Minimum samples for GP training (lower = faster but less accurate)
        min_samples = max(10, self.n_surrogate_samples // 2)
        if len(y_valid) < min_samples:
            raise ValueError(f"Too few valid samples ({len(y_valid)}/{min_samples}) for surrogate training")

        # Train GP surrogate
        surrogate = trainer.train(archetype_id, X_valid, y_valid)

        return surrogate

    def _apply_params_to_idf(self, idf_path: Path, params: Dict[str, float]):
        """Apply calibration parameters to IDF file."""
        from eppy.modeleditor import IDF as EppyIDF

        # Set IDD if needed - check multiple locations
        idd_paths = [
            '/Applications/EnergyPlus-25-1-0/Energy+.idd',
            '/usr/local/EnergyPlus-25-1-0/Energy+.idd',
            '/opt/EnergyPlus-25-1-0/Energy+.idd',
        ]
        try:
            for idd_path in idd_paths:
                if Path(idd_path).exists():
                    EppyIDF.setiddname(idd_path)
                    break
        except:
            pass

        idf = EppyIDF(str(idf_path))

        # Apply infiltration
        if 'infiltration_ach' in params:
            for inf in idf.idfobjects['ZONEINFILTRATION:DESIGNFLOWRATE']:
                inf.Air_Changes_per_Hour = params['infiltration_ach']

        # Apply heat recovery
        if 'heat_recovery_eff' in params:
            for hr in idf.idfobjects.get('HEATEXCHANGER:AIRTOAIR:SENSIBLEANDLATENT', []):
                hr.Sensible_Effectiveness_at_100_Heating_Air_Flow = params['heat_recovery_eff']
                hr.Sensible_Effectiveness_at_75_Heating_Air_Flow = params['heat_recovery_eff']

        # Apply window U-value
        if 'window_u_value' in params:
            for win in idf.idfobjects.get('WINDOWMATERIAL:SIMPLEGLAZINGSYSTEM', []):
                win.UFactor = params['window_u_value']

        # Apply heating setpoint
        if 'heating_setpoint' in params:
            for sched in idf.idfobjects.get('SCHEDULETYPELIMITS', []):
                pass  # Would need to find and modify thermostat schedules

        idf.save()

    def _get_morris_results(
        self,
        surrogate: TrainedSurrogate,
        archetype_id: str,
    ) -> MorrisResults:
        """Get or compute Morris sensitivity results."""
        # Check cache
        if archetype_id in self._morris_results:
            return self._morris_results[archetype_id]

        # Run Morris screening
        screening = MorrisScreening(surrogate, n_trajectories=20)
        results = screening.analyze()

        # Log results
        logger.info(f"\nMorris Sensitivity Ranking:")
        for name, rank in sorted(results.ranking.items(), key=lambda x: x[1]):
            mu_star = results.mu_star_normalized[name]
            logger.info(f"  {rank}. {name}: μ*={mu_star:.3f}")

        # Cache
        self._morris_results[archetype_id] = results
        return results

    def _get_archetype_default(self, archetype_id: str, param_name: str) -> float:
        """Get default parameter value from archetype."""
        # Default values based on archetype (simplified)
        defaults = {
            'infiltration_ach': 0.08,
            'wall_u_value': 0.30,
            'roof_u_value': 0.25,
            'floor_u_value': 0.35,
            'window_u_value': 1.20,
            'heat_recovery_eff': 0.70,
            'heating_setpoint': 21.0,
        }

        # Adjust for archetype era
        if '2011' in archetype_id or 'low_energy' in archetype_id.lower():
            defaults.update({
                'infiltration_ach': 0.04,
                'wall_u_value': 0.15,
                'window_u_value': 0.90,
                'heat_recovery_eff': 0.82,
            })
        elif '1996' in archetype_id or 'modern' in archetype_id.lower():
            defaults.update({
                'infiltration_ach': 0.06,
                'wall_u_value': 0.20,
                'window_u_value': 1.10,
            })
        elif '1976' in archetype_id or '1985' in archetype_id:
            defaults.update({
                'infiltration_ach': 0.10,
                'wall_u_value': 0.30,
            })
        elif '1961' in archetype_id or '1975' in archetype_id or 'miljon' in archetype_id.lower():
            defaults.update({
                'infiltration_ach': 0.15,
                'wall_u_value': 0.60,
                'window_u_value': 2.00,
                'heat_recovery_eff': 0.0,
            })
        elif 'pre_1945' in archetype_id or 'brick' in archetype_id.lower():
            defaults.update({
                'infiltration_ach': 0.20,
                'wall_u_value': 1.20,
                'window_u_value': 2.50,
                'heat_recovery_eff': 0.0,
            })

        return defaults.get(param_name, 0.5)

    def _run_eplus_verification(
        self,
        baseline_idf: Path,
        calibrated_params: Dict[str, float],
        output_dir: Path,
        atemp_m2: float,
    ) -> Optional[float]:
        """
        Run E+ simulation with calibrated parameters to verify surrogate accuracy.

        This is the ground-truth check recommended by Chong & Menberg (2018).
        After surrogate-based calibration, running actual E+ confirms the
        surrogate was accurate in the region of interest.

        Args:
            baseline_idf: Path to baseline IDF
            calibrated_params: MAP parameters from calibration
            output_dir: Directory for verification run
            atemp_m2: Heated floor area for normalization

        Returns:
            Verified heating_kwh_m2 from actual E+ run, or None on failure
        """
        import shutil

        try:
            # Create verification directory
            verify_dir = Path(output_dir) / "eplus_verification"
            verify_dir.mkdir(parents=True, exist_ok=True)

            # Copy and modify IDF
            modified_idf = verify_dir / "verified_model.idf"
            shutil.copy(baseline_idf, modified_idf)

            # Apply calibrated parameters
            self._apply_params_to_idf(modified_idf, calibrated_params)

            # Run E+ simulation
            sim_result = self.runner.run(
                idf_path=modified_idf,
                weather_path=self.weather_path,
                output_dir=verify_dir,
            )

            if sim_result and sim_result.success:
                # Parse results
                from ..simulation.results import ResultsParser
                parser = ResultsParser()
                parsed = parser.parse(verify_dir)

                if parsed and parsed.heating_kwh_m2 > 0:
                    return parsed.heating_kwh_m2
                elif parsed and parsed.heating_kwh > 0 and atemp_m2 > 0:
                    return parsed.heating_kwh / atemp_m2

            logger.warning("E+ verification simulation did not produce valid results")
            return None

        except Exception as e:
            logger.error(f"E+ verification failed: {e}")
            return None

    def compute_ecm_savings_with_uncertainty(
        self,
        ecm_params_effect: Dict[str, float],
        result: CalibrationResult,
        n_samples: int = 500,
    ) -> Dict[str, Any]:
        """
        Compute ECM savings with uncertainty from calibration posterior.

        Args:
            ecm_params_effect: Parameter modifications from ECM
                              e.g., {'heat_recovery_eff': 0.85} for FTX upgrade
            result: CalibrationResult from calibrate()
            n_samples: Monte Carlo samples

        Returns:
            Dict with savings mean, std, and confidence intervals
        """
        if result.posterior is None:
            raise ValueError("CalibrationResult has no posterior - was Bayesian calibration used?")

        # Get surrogate for this archetype
        surrogate = self._surrogates.get(result.archetype_id)
        if surrogate is None:
            raise ValueError(f"No surrogate found for {result.archetype_id}")

        predictor = SurrogatePredictor(surrogate)
        propagator = UncertaintyPropagator(predictor, result.posterior)

        # Define ECM effect function
        def ecm_effect(params: Dict[str, float]) -> Dict[str, float]:
            modified = params.copy()
            modified.update(ecm_params_effect)
            return modified

        # Compute savings distribution
        savings = propagator.compute_savings_distribution(ecm_effect, n_samples)

        return savings
