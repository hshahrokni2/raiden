"""
Approximate Bayesian Computation with Sequential Monte Carlo (ABC-SMC).

For high-stakes buildings where surrogate-based calibration may not be trusted,
ABC-SMC runs actual EnergyPlus simulations to calibrate parameters.

Reference: https://www.sciencedirect.com/science/article/abs/pii/S0360544225014653

When to Use:
- Investment decisions > 10 MSEK
- ESCO performance guarantees
- Buildings where surrogate may not generalize (unusual construction)
- When surrogate-E+ discrepancy > 10%

Computational Cost:
- 2,000-5,000 actual E+ simulations
- 10-50 hours on a workstation
- $50-200 on cloud (parallel execution)

Usage:
    from src.calibration.abc_smc import ABCSMCDirectCalibrator

    calibrator = ABCSMCDirectCalibrator(
        runner=EnergyPlusRunner(),
        weather_path=Path("stockholm.epw"),
        n_particles=1000,
        n_generations=5,
    )

    posterior = calibrator.calibrate(
        baseline_idf=Path("baseline.idf"),
        measured_kwh_m2=53.0,
        initial_tolerance=20.0,  # Accept within ±20%
    )
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Callable
import logging
import tempfile
import shutil

import numpy as np
from scipy.stats import qmc, truncnorm

logger = logging.getLogger(__name__)


@dataclass
class ABCSMCConfig:
    """Configuration for ABC-SMC calibration."""

    n_particles: int = 1000  # Number of samples to maintain
    n_generations: int = 5   # SMC iterations
    alpha: float = 0.75      # Quantile for tolerance adaptation

    # Parallel execution
    n_workers: int = 8       # Parallel E+ simulations

    # Parameter bounds (same as SurrogateConfig)
    param_bounds: Dict[str, Tuple[float, float]] = field(default_factory=dict)

    # Stopping criteria
    min_acceptance_rate: float = 0.01  # Stop if acceptance < 1%
    max_simulations: int = 10000       # Hard limit on E+ runs

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
class ABCSMCPosterior:
    """Posterior samples from ABC-SMC calibration."""

    samples: np.ndarray            # (n_particles, n_params)
    weights: np.ndarray            # Importance weights
    param_names: List[str]
    n_simulations: int             # Total E+ simulations run
    final_tolerance: float         # Final epsilon
    effective_sample_size: float   # ESS

    @property
    def means(self) -> Dict[str, float]:
        """Weighted posterior means."""
        return {
            name: np.average(self.samples[:, i], weights=self.weights)
            for i, name in enumerate(self.param_names)
        }

    @property
    def stds(self) -> Dict[str, float]:
        """Weighted posterior standard deviations."""
        means = self.means
        return {
            name: np.sqrt(np.average((self.samples[:, i] - means[name])**2, weights=self.weights))
            for i, name in enumerate(self.param_names)
        }


class ABCSMCDirectCalibrator:
    """
    ABC-SMC calibration using actual EnergyPlus simulations.

    Unlike surrogate-based calibration, this runs real E+ simulations
    for each sample evaluation. More accurate but much more expensive.

    Algorithm:
    1. Draw initial particles from prior
    2. For each generation:
       a. Reduce tolerance (epsilon)
       b. Resample particles based on weights
       c. Perturb particles with MCMC kernel
       d. Run E+ for each particle
       e. Accept particles where |simulated - measured| < epsilon
       f. Update weights based on acceptance

    Reference:
    - Sisson et al. (2007) Sequential Monte Carlo without likelihoods
    - Chong & Menberg (2018) Bayesian calibration for building energy
    """

    def __init__(
        self,
        runner,  # EnergyPlusRunner
        weather_path: Path,
        config: Optional[ABCSMCConfig] = None,
    ):
        self.runner = runner
        self.weather_path = Path(weather_path)
        self.config = config or ABCSMCConfig()
        self.param_names = list(self.config.param_bounds.keys())

    def calibrate(
        self,
        baseline_idf: Path,
        measured_kwh_m2: float,
        output_dir: Optional[Path] = None,
        priors: Optional[Dict[str, Tuple[float, float, float, float]]] = None,
    ) -> ABCSMCPosterior:
        """
        Run ABC-SMC calibration with actual E+ simulations.

        Args:
            baseline_idf: Path to baseline IDF file
            measured_kwh_m2: Target measured energy
            output_dir: Directory for simulation outputs
            priors: Optional prior distributions {param: (mean, std, min, max)}

        Returns:
            ABCSMCPosterior with calibrated parameter samples
        """
        logger.info("=" * 60)
        logger.info("ABC-SMC CALIBRATION (Actual E+ Simulations)")
        logger.info("=" * 60)
        logger.info(f"Target: {measured_kwh_m2:.1f} kWh/m²")
        logger.info(f"Particles: {self.config.n_particles}")
        logger.info(f"Generations: {self.config.n_generations}")
        logger.info(f"Max simulations: {self.config.max_simulations}")
        logger.warning("⚠️ This will run thousands of E+ simulations!")

        # TODO: Implement ABC-SMC algorithm
        # This is a stub - full implementation requires:
        # 1. Prior sampling
        # 2. E+ simulation execution (parallel)
        # 3. Distance calculation
        # 4. Tolerance adaptation
        # 5. MCMC perturbation kernel
        # 6. Weight update

        raise NotImplementedError(
            "ABC-SMC direct calibration not yet implemented. "
            "Use surrogate-based calibration with verify_with_eplus=True "
            "for a hybrid approach that validates with actual E+ runs."
        )

    def _run_simulation(
        self,
        baseline_idf: Path,
        params: Dict[str, float],
        sim_dir: Path,
    ) -> Optional[float]:
        """Run single E+ simulation and return heating_kwh_m2."""
        # TODO: Copy IDF, apply params, run simulation, parse results
        raise NotImplementedError()

    def _sample_prior(self, n: int) -> np.ndarray:
        """Sample n particles from prior distribution."""
        # Latin Hypercube for better coverage
        sampler = qmc.LatinHypercube(d=len(self.param_names))
        samples_unit = sampler.random(n=n)

        # Scale to bounds
        lower = np.array([self.config.param_bounds[p][0] for p in self.param_names])
        upper = np.array([self.config.param_bounds[p][1] for p in self.param_names])

        return qmc.scale(samples_unit, lower, upper)

    def _compute_distance(
        self,
        simulated: float,
        measured: float,
    ) -> float:
        """Compute distance between simulated and measured."""
        return abs(simulated - measured) / measured  # Relative error


# Convenience function
def run_abc_smc_calibration(
    baseline_idf: Path,
    measured_kwh_m2: float,
    weather_path: Path,
    runner,
    n_particles: int = 1000,
    n_generations: int = 5,
) -> ABCSMCPosterior:
    """
    Convenience function for ABC-SMC calibration.

    WARNING: This runs thousands of E+ simulations!
    """
    config = ABCSMCConfig(
        n_particles=n_particles,
        n_generations=n_generations,
    )

    calibrator = ABCSMCDirectCalibrator(
        runner=runner,
        weather_path=weather_path,
        config=config,
    )

    return calibrator.calibrate(
        baseline_idf=baseline_idf,
        measured_kwh_m2=measured_kwh_m2,
    )
