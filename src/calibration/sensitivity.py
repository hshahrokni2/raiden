"""
Morris Method Sensitivity Analysis for Building Calibration.

Implements Morris screening to identify which parameters have the largest
influence on heating energy, enabling focused calibration on identifiable parameters.

Reference: Morris, M. D. (1991). Factorial Sampling Plans for Preliminary
           Computational Experiments. Technometrics, 33(2), 161-174.

Usage:
    from src.calibration.sensitivity import MorrisScreening, run_morris_analysis

    # Using trained surrogate
    screening = MorrisScreening(surrogate)
    results = screening.analyze()

    # Get important parameters
    important = results.get_important_parameters(mu_star_threshold=0.1)
    print(f"Important parameters: {important}")
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Callable
import logging
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class MorrisResults:
    """
    Results of Morris sensitivity analysis.

    Attributes:
        param_names: List of parameter names
        mu: Mean of elementary effects (can cancel out)
        mu_star: Mean of absolute elementary effects (magnitude of influence)
        sigma: Standard deviation of elementary effects (nonlinearity/interactions)
        mu_star_normalized: Normalized μ* (0-1 scale for comparison)
        ranking: Parameters ranked by importance (1 = most important)
        n_trajectories: Number of Morris trajectories used
    """

    param_names: List[str]
    mu: Dict[str, float]
    mu_star: Dict[str, float]
    sigma: Dict[str, float]
    mu_star_normalized: Dict[str, float]
    ranking: Dict[str, int]
    n_trajectories: int = 0

    def get_important_parameters(
        self,
        mu_star_threshold: float = 0.1,
        top_n: Optional[int] = None,
    ) -> List[str]:
        """
        Get parameters that have significant influence.

        Args:
            mu_star_threshold: Normalized μ* threshold (0-1)
            top_n: Return top N parameters (overrides threshold)

        Returns:
            List of important parameter names
        """
        if top_n is not None:
            # Return top N by ranking
            sorted_params = sorted(self.ranking.items(), key=lambda x: x[1])
            return [p for p, _ in sorted_params[:top_n]]

        # Filter by threshold
        return [
            name for name, value in self.mu_star_normalized.items()
            if value >= mu_star_threshold
        ]

    def get_negligible_parameters(
        self,
        mu_star_threshold: float = 0.05,
    ) -> List[str]:
        """
        Get parameters with negligible influence (can be fixed).

        Args:
            mu_star_threshold: Parameters below this are negligible

        Returns:
            List of negligible parameter names
        """
        return [
            name for name, value in self.mu_star_normalized.items()
            if value < mu_star_threshold
        ]

    def get_nonlinear_parameters(
        self,
        sigma_ratio_threshold: float = 0.5,
    ) -> List[str]:
        """
        Get parameters with high nonlinearity/interactions.

        High σ/μ* ratio indicates nonlinear effects or interactions.

        Args:
            sigma_ratio_threshold: σ/μ* threshold for nonlinearity

        Returns:
            List of nonlinear parameter names
        """
        nonlinear = []
        for name in self.param_names:
            if self.mu_star[name] > 0:
                ratio = self.sigma[name] / self.mu_star[name]
                if ratio > sigma_ratio_threshold:
                    nonlinear.append(name)
        return nonlinear

    def to_dict(self) -> Dict[str, Any]:
        """Export to dictionary."""
        return {
            "param_names": self.param_names,
            "mu": self.mu,
            "mu_star": self.mu_star,
            "sigma": self.sigma,
            "mu_star_normalized": self.mu_star_normalized,
            "ranking": self.ranking,
            "n_trajectories": self.n_trajectories,
        }

    def __str__(self) -> str:
        """Human-readable summary."""
        lines = [
            f"Morris Sensitivity Analysis (n_trajectories={self.n_trajectories})",
            "",
            "Rank  Parameter                μ*norm   μ*        σ        Type",
            "─" * 70,
        ]

        sorted_params = sorted(self.ranking.items(), key=lambda x: x[1])
        for name, rank in sorted_params:
            mu_star_n = self.mu_star_normalized[name]
            mu_star = self.mu_star[name]
            sigma = self.sigma[name]

            # Classify parameter
            if mu_star_n < 0.05:
                ptype = "negligible"
            elif sigma / max(mu_star, 1e-6) > 0.5:
                ptype = "nonlinear"
            else:
                ptype = "linear"

            lines.append(
                f"{rank:4d}  {name:24s} {mu_star_n:6.3f}   {mu_star:8.2f}  {sigma:8.2f}  {ptype}"
            )

        return "\n".join(lines)


class MorrisScreening:
    """
    Morris method sensitivity screening using surrogate model.

    Uses the trained GP surrogate for fast evaluation, avoiding expensive E+ runs.
    """

    def __init__(
        self,
        surrogate,  # TrainedSurrogate
        n_trajectories: int = 20,  # Number of Morris trajectories (10-50 typical)
        n_levels: int = 4,  # Number of levels in parameter grid
        seed: int = 42,
    ):
        """
        Initialize Morris screening.

        Args:
            surrogate: Trained surrogate model
            n_trajectories: Number of Morris trajectories (more = more accurate)
            n_levels: Grid resolution (4-10 typical)
            seed: Random seed for reproducibility
        """
        self.surrogate = surrogate
        self.n_trajectories = n_trajectories
        self.n_levels = n_levels
        self.seed = seed

        self.param_names = surrogate.param_names
        self.param_bounds = surrogate.param_bounds

    def analyze(self) -> MorrisResults:
        """
        Run Morris sensitivity analysis.

        Returns:
            MorrisResults with importance metrics
        """
        try:
            from SALib.sample import morris as morris_sample
            from SALib.analyze import morris as morris_analyze
        except ImportError:
            logger.warning("SALib not installed, using fallback implementation")
            return self._analyze_fallback()

        # Define problem for SALib
        problem = {
            'num_vars': len(self.param_names),
            'names': self.param_names,
            'bounds': [
                list(self.param_bounds[name])
                for name in self.param_names
            ],
        }

        # Generate Morris samples
        param_values = morris_sample.sample(
            problem,
            N=self.n_trajectories,
            num_levels=self.n_levels,
            seed=self.seed,
        )

        logger.info(f"Running Morris analysis with {len(param_values)} samples")

        # Evaluate surrogate at all sample points
        from .surrogate import SurrogatePredictor
        predictor = SurrogatePredictor(self.surrogate)

        # Convert to parameter dicts
        params_list = [
            {name: param_values[i, j] for j, name in enumerate(self.param_names)}
            for i in range(len(param_values))
        ]

        # Batch predict
        Y = predictor.predict_batch(params_list)

        # Morris analysis
        Si = morris_analyze.analyze(
            problem,
            param_values,
            Y,
            conf_level=0.95,
            print_to_console=False,
            seed=self.seed,
        )

        # Extract results
        mu = dict(zip(self.param_names, Si['mu']))
        mu_star = dict(zip(self.param_names, Si['mu_star']))
        sigma = dict(zip(self.param_names, Si['sigma']))

        # Normalize μ*
        max_mu_star = max(mu_star.values()) if mu_star.values() else 1.0
        mu_star_normalized = {
            name: value / max_mu_star if max_mu_star > 0 else 0
            for name, value in mu_star.items()
        }

        # Create ranking
        sorted_params = sorted(mu_star.items(), key=lambda x: -x[1])
        ranking = {name: rank + 1 for rank, (name, _) in enumerate(sorted_params)}

        results = MorrisResults(
            param_names=self.param_names,
            mu=mu,
            mu_star=mu_star,
            sigma=sigma,
            mu_star_normalized=mu_star_normalized,
            ranking=ranking,
            n_trajectories=self.n_trajectories,
        )

        logger.info(f"Morris analysis complete. Top parameter: {sorted_params[0][0]}")
        return results

    def _analyze_fallback(self) -> MorrisResults:
        """
        Fallback Morris implementation without SALib.

        Uses simplified one-at-a-time approach.
        """
        from .surrogate import SurrogatePredictor
        predictor = SurrogatePredictor(self.surrogate)

        n_params = len(self.param_names)
        elementary_effects = {name: [] for name in self.param_names}

        rng = np.random.default_rng(self.seed)

        for _ in range(self.n_trajectories):
            # Random base point
            base = {
                name: rng.uniform(*self.param_bounds[name])
                for name in self.param_names
            }

            y_base = predictor.predict(base)

            # Compute elementary effect for each parameter
            for name in self.param_names:
                bounds = self.param_bounds[name]
                delta = (bounds[1] - bounds[0]) / (self.n_levels - 1)

                # Perturb parameter
                perturbed = base.copy()
                direction = rng.choice([-1, 1])

                new_val = base[name] + direction * delta
                new_val = max(bounds[0], min(bounds[1], new_val))
                perturbed[name] = new_val

                y_perturbed = predictor.predict(perturbed)

                # Elementary effect
                if abs(new_val - base[name]) > 1e-10:
                    ee = (y_perturbed - y_base) / (new_val - base[name])
                    elementary_effects[name].append(ee)

        # Compute statistics
        mu = {}
        mu_star = {}
        sigma = {}

        for name in self.param_names:
            effects = np.array(elementary_effects[name])
            if len(effects) > 0:
                mu[name] = np.mean(effects)
                mu_star[name] = np.mean(np.abs(effects))
                sigma[name] = np.std(effects)
            else:
                mu[name] = 0.0
                mu_star[name] = 0.0
                sigma[name] = 0.0

        # Normalize
        max_mu_star = max(mu_star.values()) if mu_star.values() else 1.0
        mu_star_normalized = {
            name: value / max_mu_star if max_mu_star > 0 else 0
            for name, value in mu_star.items()
        }

        # Ranking
        sorted_params = sorted(mu_star.items(), key=lambda x: -x[1])
        ranking = {name: rank + 1 for rank, (name, _) in enumerate(sorted_params)}

        return MorrisResults(
            param_names=self.param_names,
            mu=mu,
            mu_star=mu_star,
            sigma=sigma,
            mu_star_normalized=mu_star_normalized,
            ranking=ranking,
            n_trajectories=self.n_trajectories,
        )


def run_morris_analysis(
    surrogate,
    n_trajectories: int = 20,
    importance_threshold: float = 0.1,
) -> Tuple[MorrisResults, List[str], List[str]]:
    """
    Convenience function to run Morris analysis and get parameter lists.

    Args:
        surrogate: Trained surrogate model
        n_trajectories: Number of Morris trajectories
        importance_threshold: Threshold for important parameters

    Returns:
        Tuple of (results, important_params, negligible_params)
    """
    screening = MorrisScreening(surrogate, n_trajectories=n_trajectories)
    results = screening.analyze()

    important = results.get_important_parameters(mu_star_threshold=importance_threshold)
    negligible = results.get_negligible_parameters(mu_star_threshold=0.05)

    logger.info(f"Important parameters ({len(important)}): {important}")
    logger.info(f"Negligible parameters ({len(negligible)}): {negligible}")

    return results, important, negligible


class AdaptiveCalibration:
    """
    Adaptive calibration that focuses on identifiable parameters.

    Uses Morris screening to determine which parameters to calibrate
    and which to fix at archetype defaults.
    """

    def __init__(
        self,
        surrogate,
        archetype_defaults: Dict[str, float],
        min_important_params: int = 3,
        max_important_params: int = 5,
    ):
        """
        Initialize adaptive calibration.

        Args:
            surrogate: Trained surrogate
            archetype_defaults: Default parameter values from archetype
            min_important_params: Minimum parameters to calibrate
            max_important_params: Maximum parameters to calibrate
        """
        self.surrogate = surrogate
        self.archetype_defaults = archetype_defaults
        self.min_params = min_important_params
        self.max_params = max_important_params

        self._morris_results: Optional[MorrisResults] = None
        self._calibration_params: Optional[List[str]] = None
        self._fixed_params: Optional[Dict[str, float]] = None

    def screen_parameters(self) -> Tuple[List[str], Dict[str, float]]:
        """
        Screen parameters and determine which to calibrate.

        Returns:
            Tuple of (params_to_calibrate, fixed_params)
        """
        screening = MorrisScreening(self.surrogate, n_trajectories=20)
        self._morris_results = screening.analyze()

        # Get parameters sorted by importance
        sorted_by_importance = sorted(
            self._morris_results.ranking.items(),
            key=lambda x: x[1]
        )

        # Select top N parameters for calibration
        n_calibrate = min(
            self.max_params,
            max(self.min_params, len([
                p for p, v in self._morris_results.mu_star_normalized.items()
                if v >= 0.1
            ]))
        )

        calibration_params = [p for p, _ in sorted_by_importance[:n_calibrate]]
        fixed_params = {
            p: self.archetype_defaults.get(p, 0)
            for p in self.surrogate.param_names
            if p not in calibration_params
        }

        self._calibration_params = calibration_params
        self._fixed_params = fixed_params

        logger.info(f"Calibrating {len(calibration_params)} parameters: {calibration_params}")
        logger.info(f"Fixed {len(fixed_params)} parameters at archetype defaults")

        return calibration_params, fixed_params

    @property
    def morris_results(self) -> Optional[MorrisResults]:
        """Get Morris analysis results."""
        return self._morris_results

    @property
    def calibration_params(self) -> Optional[List[str]]:
        """Get list of parameters to calibrate."""
        return self._calibration_params

    @property
    def fixed_params(self) -> Optional[Dict[str, float]]:
        """Get fixed parameter values."""
        return self._fixed_params
