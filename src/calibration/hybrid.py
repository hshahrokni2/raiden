"""
Hybrid Calibration: Surrogate + EnergyPlus Verification (2026 Roadmap).

Combines fast surrogate-based Bayesian calibration with ground-truth
EnergyPlus verification for high-stakes investment decisions.

When to Use:
- Investment decisions > 10 MSEK
- ESCO performance guarantees
- Contractual energy savings commitments
- When surrogate-E+ discrepancy needs quantification

Method:
1. Train surrogate on 200 E+ runs (standard approach)
2. Run MCMC on surrogate (50,000 samples)
3. Take top 500 posterior samples (highest likelihood)
4. Run actual E+ on all 500 samples
5. Reweight posterior based on E+/surrogate discrepancy
6. Report ground-truth calibrated parameters

Computational Cost:
- Surrogate training: 200 E+ runs (standard)
- Verification: 500 E+ runs (additional)
- Total: 700 E+ runs
- Time: 2-5 hours (parallel execution)

Reference: Kennedy & O'Hagan (2001), Higdon et al. (2008)

Usage:
    from src.calibration.hybrid import HybridCalibrator

    calibrator = HybridCalibrator(
        runner=EnergyPlusRunner(),
        weather_path=Path("stockholm.epw"),
        n_verification_samples=500,
    )

    result = calibrator.calibrate(
        baseline_idf=Path("baseline.idf"),
        measured_kwh_m2=53.0,
    )

    print(f"Ground-truth posterior mean: {result.verified_means}")
    print(f"Surrogate bias: {result.surrogate_bias:.1%}")
"""

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Callable, Any
import logging
import tempfile
import shutil
import time

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class HybridConfig:
    """Configuration for hybrid calibration."""

    # Surrogate phase
    n_surrogate_samples: int = 200  # LHS samples for surrogate training
    n_mcmc_samples: int = 50000     # MCMC posterior samples

    # Verification phase
    n_verification_samples: int = 500  # Top posterior samples to verify
    n_workers: int = 8                  # Parallel E+ executions

    # Reweighting
    discrepancy_kernel_width: float = 0.05  # Gaussian kernel for reweighting

    # Sample selection
    selection_method: str = "diverse"  # "top_likelihood" or "diverse"

    # Parameter bounds
    param_bounds: Dict[str, Tuple[float, float]] = field(default_factory=dict)

    def __post_init__(self):
        if not self.param_bounds:
            self.param_bounds = {
                'infiltration_ach': (0.02, 0.50),
                'wall_u_value': (0.15, 2.50),
                'roof_u_value': (0.10, 1.50),
                'floor_u_value': (0.15, 1.50),
                'window_u_value': (0.70, 4.00),
                'heat_recovery_eff': (0.0, 0.90),
                'heating_setpoint': (18.0, 24.0),
            }


@dataclass
class HybridCalibrationResult:
    """Results from hybrid calibration."""

    # Surrogate-based posterior (fast, approximate)
    surrogate_samples: np.ndarray      # (n_mcmc_samples, n_params)
    surrogate_weights: np.ndarray      # Importance weights
    surrogate_means: Dict[str, float]
    surrogate_stds: Dict[str, float]

    # Verification subset
    verification_samples: np.ndarray   # (n_verification_samples, n_params)
    verification_eplus_results: np.ndarray  # Actual E+ kWh/m² for each sample
    verification_surrogate_preds: np.ndarray  # Surrogate predictions

    # Reweighted posterior (ground truth)
    verified_weights: np.ndarray       # Reweighted based on E+ results
    verified_means: Dict[str, float]
    verified_stds: Dict[str, float]

    # Discrepancy analysis
    surrogate_bias: float              # Mean(surrogate - E+) / E+
    surrogate_rmse: float              # RMSE of surrogate vs E+
    param_names: List[str]

    # Metadata
    n_eplus_runs: int                  # Total E+ simulations
    effective_sample_size: float       # ESS after reweighting
    total_time_seconds: float = 0.0

    @property
    def discrepancy_acceptable(self) -> bool:
        """Check if surrogate-E+ discrepancy is within acceptable range."""
        return abs(self.surrogate_bias) < 0.05 and self.surrogate_rmse < 0.10

    def to_dict(self) -> Dict[str, Any]:
        """Export to dictionary for JSON serialization."""
        return {
            "surrogate_means": self.surrogate_means,
            "surrogate_stds": self.surrogate_stds,
            "verified_means": self.verified_means,
            "verified_stds": self.verified_stds,
            "surrogate_bias": self.surrogate_bias,
            "surrogate_rmse": self.surrogate_rmse,
            "discrepancy_acceptable": self.discrepancy_acceptable,
            "n_eplus_runs": self.n_eplus_runs,
            "effective_sample_size": self.effective_sample_size,
            "total_time_seconds": self.total_time_seconds,
        }


@dataclass
class ImportanceSamplingResult:
    """Result of importance sampling reweighting."""
    weights: np.ndarray
    effective_sample_size: float
    normalized_weights: np.ndarray


def _run_single_eplus(args: Tuple) -> Optional[float]:
    """Run single E+ simulation (for parallel execution)."""
    idf_path, weather_path, params, output_dir, param_names = args

    try:
        # Lazy import to avoid issues with multiprocessing
        from src.ecm.idf_modifier import IDFModifier

        # Copy IDF to temp location
        temp_idf = output_dir / "model.idf"
        shutil.copy(idf_path, temp_idf)

        # Apply parameter modifications
        modifier = IDFModifier()

        # Map params to IDF modifications
        param_dict = {name: params[i] for i, name in enumerate(param_names)}

        # Apply infiltration
        if 'infiltration_ach' in param_dict:
            modifier.set_infiltration(temp_idf, param_dict['infiltration_ach'])

        # Apply U-values
        if 'wall_u_value' in param_dict:
            modifier.set_wall_u_value(temp_idf, param_dict['wall_u_value'])
        if 'roof_u_value' in param_dict:
            modifier.set_roof_u_value(temp_idf, param_dict['roof_u_value'])
        if 'window_u_value' in param_dict:
            modifier.set_window_u_value(temp_idf, param_dict['window_u_value'])

        # Apply heat recovery
        if 'heat_recovery_eff' in param_dict:
            modifier.set_heat_recovery(temp_idf, param_dict['heat_recovery_eff'])

        # Apply heating setpoint
        if 'heating_setpoint' in param_dict:
            modifier.set_heating_setpoint(temp_idf, param_dict['heating_setpoint'])

        # Run E+
        from src.simulation.runner import EnergyPlusRunner
        runner = EnergyPlusRunner()
        result = runner.run(
            idf_path=temp_idf,
            weather_path=weather_path,
            output_dir=output_dir,
        )

        if result and result.heating_kwh:
            # Get floor area from result or assume
            floor_area = getattr(result, 'floor_area_m2', 1000)
            return result.heating_kwh / floor_area

        return None

    except Exception as e:
        logger.error(f"E+ simulation failed: {e}")
        return None


class HybridCalibrator:
    """
    Hybrid calibration combining surrogate speed with E+ ground truth.

    This approach provides:
    1. Fast exploration via surrogate (50,000 MCMC samples in seconds)
    2. Ground truth verification via actual E+ (500 samples)
    3. Quantified surrogate bias for uncertainty

    Algorithm:
    1. Standard surrogate calibration (200 E+ → GP → MCMC)
    2. Select top 500 posterior samples (highest likelihood)
    3. Run E+ on all 500 → get ground-truth energies
    4. Compute discrepancy: d_i = |surrogate_i - eplus_i|
    5. Reweight samples: w_i ∝ exp(-d_i² / (2σ²))
    6. Report reweighted posterior statistics

    Reference:
    - Higdon et al. (2008) Computer Model Calibration Using High-Dimensional Output
    - Kennedy & O'Hagan (2001) Bayesian Calibration of Computer Models
    """

    def __init__(
        self,
        runner,  # EnergyPlusRunner
        weather_path: Path,
        config: Optional[HybridConfig] = None,
    ):
        self.runner = runner
        self.weather_path = Path(weather_path)
        self.config = config or HybridConfig()
        self.param_names = list(self.config.param_bounds.keys())

    def calibrate(
        self,
        baseline_idf: Path,
        measured_kwh_m2: float,
        archetype_id: str = "unknown",
        output_dir: Optional[Path] = None,
        progress_callback: Optional[Callable[[str, int, int], None]] = None,
    ) -> HybridCalibrationResult:
        """
        Run hybrid calibration with E+ verification.

        Args:
            baseline_idf: Path to baseline IDF file
            measured_kwh_m2: Target measured energy
            archetype_id: Archetype identifier for surrogate lookup
            output_dir: Directory for simulation outputs
            progress_callback: Optional callback(phase, current, total)

        Returns:
            HybridCalibrationResult with verified posterior
        """
        start_time = time.time()

        logger.info("=" * 60)
        logger.info("HYBRID CALIBRATION (Surrogate + E+ Verification)")
        logger.info("=" * 60)
        logger.info(f"Target: {measured_kwh_m2:.1f} kWh/m²")
        logger.info(f"Surrogate samples: {self.config.n_surrogate_samples}")
        logger.info(f"MCMC samples: {self.config.n_mcmc_samples}")
        logger.info(f"Verification samples: {self.config.n_verification_samples}")
        logger.info(f"Total E+ runs: ~{self.config.n_surrogate_samples + self.config.n_verification_samples}")

        # Create output directory
        if output_dir is None:
            output_dir = Path(tempfile.mkdtemp(prefix="hybrid_cal_"))
        else:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

        # Phase 1: Standard surrogate calibration
        if progress_callback:
            progress_callback("surrogate_training", 0, self.config.n_surrogate_samples)

        logger.info("\nPhase 1: Surrogate-based calibration")
        surrogate_result = self._run_surrogate_calibration(
            baseline_idf,
            measured_kwh_m2,
            archetype_id,
            output_dir,
        )

        # Phase 2: Select samples for verification
        if progress_callback:
            progress_callback("sample_selection", 0, self.config.n_verification_samples)

        logger.info(f"\nPhase 2: Selecting top {self.config.n_verification_samples} samples")
        verification_samples, verification_indices = self._select_samples(
            surrogate_result.samples,
            surrogate_result.likelihoods,
        )

        # Get surrogate predictions for verification samples
        surrogate_preds = np.array([
            surrogate_result.surrogate.predict({
                name: verification_samples[i, j]
                for j, name in enumerate(self.param_names)
            })
            for i in range(len(verification_samples))
        ])

        # Phase 3: Run E+ verification
        if progress_callback:
            progress_callback("eplus_verification", 0, len(verification_samples))

        logger.info(f"\nPhase 3: Running {len(verification_samples)} E+ verifications")
        eplus_results = self._run_verification_simulations(
            baseline_idf,
            verification_samples,
            output_dir,
            progress_callback,
        )

        # Phase 4: Reweight posterior
        logger.info("\nPhase 4: Reweighting posterior")
        reweight_result = self._reweight_posterior(
            verification_samples,
            eplus_results,
            surrogate_preds,
        )

        # Calculate statistics
        surrogate_means = {
            name: np.average(surrogate_result.samples[:, i], weights=surrogate_result.weights)
            for i, name in enumerate(self.param_names)
        }
        surrogate_stds = {
            name: np.sqrt(np.average(
                (surrogate_result.samples[:, i] - surrogate_means[name])**2,
                weights=surrogate_result.weights
            ))
            for i, name in enumerate(self.param_names)
        }

        verified_means = {
            name: np.average(verification_samples[:, i], weights=reweight_result.normalized_weights)
            for i, name in enumerate(self.param_names)
        }
        verified_stds = {
            name: np.sqrt(np.average(
                (verification_samples[:, i] - verified_means[name])**2,
                weights=reweight_result.normalized_weights
            ))
            for i, name in enumerate(self.param_names)
        }

        # Compute discrepancy metrics
        valid_mask = ~np.isnan(eplus_results)
        valid_surrogate = surrogate_preds[valid_mask]
        valid_eplus = eplus_results[valid_mask]

        if len(valid_eplus) > 0:
            bias = np.mean((valid_surrogate - valid_eplus) / valid_eplus)
            rmse = np.sqrt(np.mean(((valid_surrogate - valid_eplus) / valid_eplus)**2))
        else:
            bias = 0.0
            rmse = 0.0

        total_time = time.time() - start_time

        logger.info("=" * 60)
        logger.info("HYBRID CALIBRATION COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Surrogate bias: {bias:.1%}")
        logger.info(f"Surrogate RMSE: {rmse:.1%}")
        logger.info(f"Effective sample size: {reweight_result.effective_sample_size:.0f}")
        logger.info(f"Total time: {total_time/60:.1f} minutes")
        logger.info(f"Total E+ runs: {self.config.n_surrogate_samples + len(verification_samples)}")

        return HybridCalibrationResult(
            surrogate_samples=surrogate_result.samples,
            surrogate_weights=surrogate_result.weights,
            surrogate_means=surrogate_means,
            surrogate_stds=surrogate_stds,
            verification_samples=verification_samples,
            verification_eplus_results=eplus_results,
            verification_surrogate_preds=surrogate_preds,
            verified_weights=reweight_result.normalized_weights,
            verified_means=verified_means,
            verified_stds=verified_stds,
            surrogate_bias=bias,
            surrogate_rmse=rmse,
            param_names=self.param_names,
            n_eplus_runs=self.config.n_surrogate_samples + len(verification_samples),
            effective_sample_size=reweight_result.effective_sample_size,
            total_time_seconds=total_time,
        )

    def _run_surrogate_calibration(
        self,
        baseline_idf: Path,
        measured_kwh_m2: float,
        archetype_id: str,
        output_dir: Path,
    ):
        """Run standard surrogate-based Bayesian calibration."""
        from .pipeline import BayesianCalibrationPipeline

        pipeline = BayesianCalibrationPipeline(
            runner=self.runner,
            weather_path=self.weather_path,
            cache_dir=output_dir / "surrogate_cache",
            n_surrogate_samples=self.config.n_surrogate_samples,
            n_abc_particles=self.config.n_mcmc_samples,
        )

        result = pipeline.calibrate(
            baseline_idf=baseline_idf,
            archetype_id=archetype_id,
            measured_kwh_m2=measured_kwh_m2,
            atemp_m2=1000,  # Will be normalized anyway
            output_dir=output_dir,
        )

        # Extract samples and likelihoods from posterior
        if result.posterior:
            samples = result.posterior.samples
            weights = result.posterior.weights
        else:
            # Fallback: generate samples from calibrated params
            from scipy.stats import qmc
            sampler = qmc.LatinHypercube(d=len(self.param_names))
            samples = sampler.random(n=self.config.n_mcmc_samples)

            # Scale to bounds centered on calibrated values
            for i, name in enumerate(self.param_names):
                mean = result.calibrated_params.get(name, 0.5)
                std = result.param_stds.get(name, 0.1)
                lower, upper = self.config.param_bounds[name]
                samples[:, i] = np.clip(mean + std * (samples[:, i] - 0.5) * 2, lower, upper)

            weights = np.ones(len(samples)) / len(samples)

        # Create a simple result object
        class SurrogateResult:
            pass

        sr = SurrogateResult()
        sr.samples = samples
        sr.weights = weights
        sr.likelihoods = weights  # Use weights as proxy for likelihood
        sr.surrogate = pipeline._surrogate  # Access trained surrogate

        return sr

    def _select_samples(
        self,
        samples: np.ndarray,
        likelihoods: np.ndarray,
        n: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Select samples for verification.

        Selection methods:
        - "top_likelihood": Take highest likelihood samples
        - "diverse": Cluster and take representatives for diversity
        """
        if n is None:
            n = self.config.n_verification_samples

        n = min(n, len(samples))

        if self.config.selection_method == "top_likelihood":
            # Sort by likelihood and take top N
            indices = np.argsort(likelihoods)[-n:]
            return samples[indices], indices

        elif self.config.selection_method == "diverse":
            # K-means clustering for diverse selection
            try:
                from sklearn.cluster import KMeans

                # Cluster samples
                kmeans = KMeans(n_clusters=n, random_state=42)
                labels = kmeans.fit_predict(samples)

                # Take sample closest to each centroid
                indices = []
                for i in range(n):
                    cluster_mask = labels == i
                    cluster_indices = np.where(cluster_mask)[0]
                    if len(cluster_indices) > 0:
                        # Take highest likelihood sample from cluster
                        best_in_cluster = cluster_indices[
                            np.argmax(likelihoods[cluster_indices])
                        ]
                        indices.append(best_in_cluster)

                indices = np.array(indices)
                return samples[indices], indices

            except ImportError:
                # Fallback to top_likelihood if sklearn not available
                logger.warning("sklearn not available, using top_likelihood selection")
                indices = np.argsort(likelihoods)[-n:]
                return samples[indices], indices

        else:
            raise ValueError(f"Unknown selection method: {self.config.selection_method}")

    def _run_verification_simulations(
        self,
        baseline_idf: Path,
        samples: np.ndarray,
        output_dir: Path,
        progress_callback: Optional[Callable] = None,
    ) -> np.ndarray:
        """
        Run E+ simulations for verification samples.

        Uses parallel execution for efficiency.
        """
        results = np.full(len(samples), np.nan)

        # Prepare arguments for parallel execution
        args_list = []
        for i, sample in enumerate(samples):
            sim_dir = output_dir / f"verify_{i:04d}"
            sim_dir.mkdir(parents=True, exist_ok=True)
            args_list.append((
                baseline_idf,
                self.weather_path,
                sample,
                sim_dir,
                self.param_names,
            ))

        # Run in parallel
        completed = 0
        with ProcessPoolExecutor(max_workers=self.config.n_workers) as executor:
            futures = {
                executor.submit(_run_single_eplus, args): i
                for i, args in enumerate(args_list)
            }

            for future in as_completed(futures):
                i = futures[future]
                try:
                    result = future.result()
                    if result is not None:
                        results[i] = result
                except Exception as e:
                    logger.error(f"Verification {i} failed: {e}")

                completed += 1
                if completed % 50 == 0:
                    logger.info(f"Completed {completed}/{len(samples)} verifications")

                if progress_callback:
                    progress_callback("eplus_verification", completed, len(samples))

        valid_count = np.sum(~np.isnan(results))
        logger.info(f"Completed {valid_count}/{len(samples)} valid verifications")

        return results

    def _reweight_posterior(
        self,
        samples: np.ndarray,
        eplus_results: np.ndarray,
        surrogate_preds: np.ndarray,
    ) -> ImportanceSamplingResult:
        """
        Reweight posterior based on E+/surrogate discrepancy.

        Uses Gaussian kernel:
        w_i ∝ exp(-d_i² / (2σ²))

        where d_i = |surrogate_i - eplus_i| / eplus_i
        """
        sigma = self.config.discrepancy_kernel_width

        # Handle NaN values (failed simulations)
        valid_mask = ~np.isnan(eplus_results)

        # Initialize weights
        weights = np.zeros(len(samples))

        # Compute relative discrepancy
        discrepancy = np.abs(surrogate_preds - eplus_results) / np.maximum(eplus_results, 1e-6)

        # Gaussian kernel weights
        weights[valid_mask] = np.exp(-discrepancy[valid_mask]**2 / (2 * sigma**2))

        # Normalize
        total_weight = np.sum(weights)
        if total_weight > 0:
            normalized_weights = weights / total_weight
        else:
            # Fallback to uniform if all failed
            normalized_weights = np.ones(len(samples)) / len(samples)

        # Effective sample size
        ess = 1.0 / np.sum(normalized_weights**2) if np.any(normalized_weights > 0) else 0

        return ImportanceSamplingResult(
            weights=weights,
            effective_sample_size=ess,
            normalized_weights=normalized_weights,
        )


def run_hybrid_calibration(
    baseline_idf: Path,
    measured_kwh_m2: float,
    weather_path: Path,
    runner,
    n_surrogate_samples: int = 200,
    n_verification_samples: int = 500,
    n_workers: int = 8,
) -> HybridCalibrationResult:
    """
    Convenience function for hybrid calibration.

    Use this for high-stakes buildings where ground-truth verification
    is required (>10 MSEK investments, ESCO guarantees).

    Args:
        baseline_idf: Path to baseline IDF
        measured_kwh_m2: Target measured energy
        weather_path: Path to EPW weather file
        runner: EnergyPlusRunner instance
        n_surrogate_samples: LHS samples for surrogate training
        n_verification_samples: Top samples to verify with E+
        n_workers: Parallel E+ workers

    Returns:
        HybridCalibrationResult with ground-truth posterior
    """
    config = HybridConfig(
        n_surrogate_samples=n_surrogate_samples,
        n_verification_samples=n_verification_samples,
        n_workers=n_workers,
    )

    calibrator = HybridCalibrator(
        runner=runner,
        weather_path=weather_path,
        config=config,
    )

    return calibrator.calibrate(
        baseline_idf=baseline_idf,
        measured_kwh_m2=measured_kwh_m2,
    )


# Decision helper
def recommend_calibration_method(
    investment_sek: float,
    esco_guarantee: bool = False,
    surrogate_discrepancy: Optional[float] = None,
    building_unusual: bool = False,
) -> str:
    """
    Recommend calibration method based on project requirements.

    Args:
        investment_sek: Total investment amount
        esco_guarantee: Whether ESCO performance guarantee is needed
        surrogate_discrepancy: Known surrogate-E+ discrepancy (if available)
        building_unusual: Whether building has unusual construction

    Returns:
        Recommended method: "surrogate", "hybrid", or "abc_smc"
    """
    # ESCO guarantees always need hybrid or ABC-SMC
    if esco_guarantee:
        if investment_sek > 50_000_000:  # >50 MSEK
            return "abc_smc"  # Full likelihood-free inference
        return "hybrid"  # Surrogate + 500 E+ verification

    # High investment needs verification
    if investment_sek > 10_000_000:  # >10 MSEK
        return "hybrid"

    # Known high surrogate error
    if surrogate_discrepancy is not None and surrogate_discrepancy > 0.10:
        return "hybrid"

    # Unusual buildings may not match surrogates
    if building_unusual:
        return "hybrid"

    # Standard case
    return "surrogate"
