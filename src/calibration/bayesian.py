"""
Bayesian calibration using ABC-SMC (Approximate Bayesian Computation).

Uses surrogate models for fast likelihood-free inference to estimate
building parameters from measured energy consumption.

Key classes:
    - Prior: Parameter prior distributions (uniform, normal, etc.)
    - CalibrationPriors: Collection of priors for all parameters
    - ABCSMCCalibrator: The main calibration engine
    - UncertaintyPropagator: Propagate parameter uncertainty to predictions

Usage:
    priors = CalibrationPriors.swedish_defaults()
    calibrator = ABCSMCCalibrator(surrogate, priors)
    posterior = calibrator.calibrate(measured_kwh_m2=85.0)

    print(f"Infiltration: {posterior.means['infiltration_ach']:.3f}")
    print(f"90% CI: {posterior.ci_90['infiltration_ach']}")
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Callable, Literal
import logging

import numpy as np
from scipy import stats

from .surrogate import SurrogatePredictor, TrainedSurrogate

logger = logging.getLogger(__name__)


@dataclass
class Prior:
    """Prior distribution for a single parameter."""

    name: str
    distribution: Literal["uniform", "normal", "truncnorm", "beta"]
    params: Dict[str, float]  # Distribution-specific parameters

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """Sample n values from this prior."""
        if self.distribution == "uniform":
            return rng.uniform(
                self.params["low"],
                self.params["high"],
                size=n
            )
        elif self.distribution == "normal":
            return rng.normal(
                self.params["mean"],
                self.params["std"],
                size=n
            )
        elif self.distribution == "truncnorm":
            # Truncated normal (bounded)
            a = (self.params["low"] - self.params["mean"]) / self.params["std"]
            b = (self.params["high"] - self.params["mean"]) / self.params["std"]
            return stats.truncnorm.rvs(
                a, b,
                loc=self.params["mean"],
                scale=self.params["std"],
                size=n,
                random_state=rng
            )
        elif self.distribution == "beta":
            # Beta distribution scaled to [low, high]
            samples = rng.beta(
                self.params["alpha"],
                self.params["beta"],
                size=n
            )
            return self.params["low"] + samples * (self.params["high"] - self.params["low"])
        else:
            raise ValueError(f"Unknown distribution: {self.distribution}")

    def pdf(self, x: np.ndarray) -> np.ndarray:
        """Evaluate probability density at x."""
        if self.distribution == "uniform":
            return stats.uniform.pdf(
                x,
                loc=self.params["low"],
                scale=self.params["high"] - self.params["low"]
            )
        elif self.distribution == "normal":
            return stats.norm.pdf(x, self.params["mean"], self.params["std"])
        elif self.distribution == "truncnorm":
            a = (self.params["low"] - self.params["mean"]) / self.params["std"]
            b = (self.params["high"] - self.params["mean"]) / self.params["std"]
            return stats.truncnorm.pdf(
                x, a, b,
                loc=self.params["mean"],
                scale=self.params["std"]
            )
        elif self.distribution == "beta":
            # Transform x to [0, 1] for beta PDF
            x_scaled = (x - self.params["low"]) / (self.params["high"] - self.params["low"])
            return stats.beta.pdf(x_scaled, self.params["alpha"], self.params["beta"])
        else:
            raise ValueError(f"Unknown distribution: {self.distribution}")


@dataclass
class CalibrationPriors:
    """Collection of prior distributions for calibration parameters."""

    priors: Dict[str, Prior] = field(default_factory=dict)

    @classmethod
    def swedish_defaults(cls) -> "CalibrationPriors":
        """Default priors for Swedish multi-family buildings."""
        return cls(priors={
            "infiltration_ach": Prior(
                name="infiltration_ach",
                distribution="truncnorm",
                params={"mean": 0.08, "std": 0.04, "low": 0.02, "high": 0.20}
            ),
            "wall_u_value": Prior(
                name="wall_u_value",
                distribution="uniform",
                params={"low": 0.15, "high": 1.50}
            ),
            "roof_u_value": Prior(
                name="roof_u_value",
                distribution="uniform",
                params={"low": 0.10, "high": 0.60}
            ),
            "floor_u_value": Prior(
                name="floor_u_value",
                distribution="uniform",
                params={"low": 0.15, "high": 0.80}
            ),
            "window_u_value": Prior(
                name="window_u_value",
                distribution="truncnorm",
                params={"mean": 1.2, "std": 0.5, "low": 0.70, "high": 2.50}
            ),
            "heat_recovery_eff": Prior(
                name="heat_recovery_eff",
                distribution="beta",
                params={"alpha": 2, "beta": 2, "low": 0.0, "high": 0.90}
            ),
            "heating_setpoint": Prior(
                name="heating_setpoint",
                distribution="truncnorm",
                params={"mean": 21.0, "std": 1.0, "low": 18.0, "high": 23.0}
            ),
        })

    @classmethod
    def from_archetype(cls, archetype_id: str) -> "CalibrationPriors":
        """
        Create priors informed by archetype typical values.

        Tighter priors around expected values for the building era.
        """
        # Archetype-specific prior adjustments
        archetype_priors = {
            "pre_1945": {
                "infiltration_ach": {"mean": 0.15, "std": 0.05},
                "wall_u_value": {"low": 0.80, "high": 1.50},
                "window_u_value": {"mean": 2.0, "std": 0.4},
            },
            "1945_1960": {
                "infiltration_ach": {"mean": 0.12, "std": 0.04},
                "wall_u_value": {"low": 0.60, "high": 1.20},
                "window_u_value": {"mean": 1.8, "std": 0.4},
            },
            "1961_1975": {
                "infiltration_ach": {"mean": 0.10, "std": 0.04},
                "wall_u_value": {"low": 0.40, "high": 0.90},
                "window_u_value": {"mean": 1.5, "std": 0.4},
            },
            "1976_1985": {
                "infiltration_ach": {"mean": 0.08, "std": 0.03},
                "wall_u_value": {"low": 0.25, "high": 0.60},
                "window_u_value": {"mean": 1.3, "std": 0.3},
            },
            "1986_1995": {
                "infiltration_ach": {"mean": 0.06, "std": 0.02},
                "wall_u_value": {"low": 0.20, "high": 0.45},
                "window_u_value": {"mean": 1.1, "std": 0.2},
            },
            "1996_2010": {
                "infiltration_ach": {"mean": 0.05, "std": 0.02},
                "wall_u_value": {"low": 0.15, "high": 0.35},
                "window_u_value": {"mean": 1.0, "std": 0.2},
            },
            "post_2010": {
                "infiltration_ach": {"mean": 0.04, "std": 0.01},
                "wall_u_value": {"low": 0.10, "high": 0.25},
                "window_u_value": {"mean": 0.9, "std": 0.15},
            },
        }

        # Start with defaults
        priors = cls.swedish_defaults()

        # Match archetype and adjust
        for key, adjustments in archetype_priors.items():
            if key in archetype_id.lower():
                for param, adj in adjustments.items():
                    if param in priors.priors:
                        priors.priors[param].params.update(adj)
                break

        return priors

    def sample(self, n: int, rng: Optional[np.random.Generator] = None) -> Dict[str, np.ndarray]:
        """Sample n values from all priors."""
        if rng is None:
            rng = np.random.default_rng()
        return {name: prior.sample(n, rng) for name, prior in self.priors.items()}


@dataclass
class PosteriorSample:
    """Single sample from the posterior distribution."""

    params: Dict[str, float]
    weight: float
    distance: float  # Distance from observed data


@dataclass
class CalibrationPosterior:
    """Posterior distribution from ABC-SMC calibration."""

    samples: List[PosteriorSample]
    param_names: List[str]
    measured_value: float
    epsilon_final: float  # Final acceptance threshold

    @property
    def weights(self) -> np.ndarray:
        """Normalized importance weights."""
        w = np.array([s.weight for s in self.samples])
        return w / w.sum()

    @property
    def means(self) -> Dict[str, float]:
        """Weighted posterior means for each parameter."""
        weights = self.weights
        return {
            name: np.average(
                [s.params[name] for s in self.samples],
                weights=weights
            )
            for name in self.param_names
        }

    @property
    def stds(self) -> Dict[str, float]:
        """Weighted posterior standard deviations."""
        weights = self.weights
        means = self.means
        return {
            name: np.sqrt(np.average(
                [(s.params[name] - means[name])**2 for s in self.samples],
                weights=weights
            ))
            for name in self.param_names
        }

    @property
    def ci_90(self) -> Dict[str, Tuple[float, float]]:
        """90% credible intervals for each parameter."""
        return self._credible_interval(0.90)

    @property
    def ci_95(self) -> Dict[str, Tuple[float, float]]:
        """95% credible intervals for each parameter."""
        return self._credible_interval(0.95)

    def _credible_interval(self, level: float) -> Dict[str, Tuple[float, float]]:
        """Compute credible intervals at given level."""
        alpha = (1 - level) / 2
        result = {}

        for name in self.param_names:
            values = np.array([s.params[name] for s in self.samples])
            weights = self.weights

            # Sort and compute cumulative weights
            sorted_idx = np.argsort(values)
            sorted_values = values[sorted_idx]
            sorted_weights = weights[sorted_idx]
            cumsum = np.cumsum(sorted_weights)

            # Find quantiles
            low_idx = np.searchsorted(cumsum, alpha)
            high_idx = np.searchsorted(cumsum, 1 - alpha)

            result[name] = (
                sorted_values[max(0, low_idx)],
                sorted_values[min(len(sorted_values) - 1, high_idx)]
            )

        return result

    def to_dict(self) -> Dict:
        """Export posterior summary as dictionary."""
        return {
            "measured_value": self.measured_value,
            "epsilon_final": self.epsilon_final,
            "n_samples": len(self.samples),
            "parameters": {
                name: {
                    "mean": self.means[name],
                    "std": self.stds[name],
                    "ci_90": self.ci_90[name],
                    "ci_95": self.ci_95[name],
                }
                for name in self.param_names
            }
        }


class ABCSMCCalibrator:
    """
    ABC-SMC (Approximate Bayesian Computation - Sequential Monte Carlo) calibration.

    Uses surrogate model for fast forward simulation, then applies ABC
    to estimate posterior distribution of building parameters given
    measured energy consumption.

    Algorithm:
    1. Sample from prior
    2. Simulate with surrogate model
    3. Accept samples within epsilon of measured value
    4. Resample and perturb (importance sampling)
    5. Reduce epsilon and repeat

    Reference: Beaumont et al. (2009) "Adaptive ABC"
    """

    def __init__(
        self,
        predictor: SurrogatePredictor,
        priors: CalibrationPriors,
        n_particles: int = 1000,
        n_generations: int = 8,
        alpha: float = 0.5,  # Quantile for epsilon schedule
        random_state: int = 42,
    ):
        self.predictor = predictor
        self.priors = priors
        self.n_particles = n_particles
        self.n_generations = n_generations
        self.alpha = alpha
        self.rng = np.random.default_rng(random_state)

        # Get parameter names that overlap between priors and surrogate
        surrogate_params = set(predictor.surrogate.param_names)
        prior_params = set(priors.priors.keys())
        self.param_names = list(surrogate_params & prior_params)

        if not self.param_names:
            raise ValueError("No overlapping parameters between priors and surrogate")

        logger.info(f"Calibrating {len(self.param_names)} parameters: {self.param_names}")

    def calibrate(
        self,
        measured_kwh_m2: float,
        tolerance_percent: float = 20.0,
    ) -> CalibrationPosterior:
        """
        Run ABC-SMC calibration to estimate parameters.

        Args:
            measured_kwh_m2: Measured annual heating energy
            tolerance_percent: Initial acceptance tolerance (%)

        Returns:
            CalibrationPosterior with weighted samples
        """
        logger.info(f"Starting ABC-SMC calibration (target: {measured_kwh_m2} kWh/m²)")

        # Initial epsilon from tolerance
        epsilon = measured_kwh_m2 * tolerance_percent / 100

        # Generation 0: Sample from prior
        particles = self._sample_from_prior(self.n_particles)
        distances = self._compute_distances(particles, measured_kwh_m2)
        weights = np.ones(self.n_particles) / self.n_particles

        # Accept particles within epsilon
        accepted_idx = distances <= epsilon
        n_accepted = accepted_idx.sum()
        logger.info(f"Gen 0: epsilon={epsilon:.2f}, accepted={n_accepted}/{self.n_particles}")

        if n_accepted < 10:
            logger.warning("Very few particles accepted - increasing epsilon")
            epsilon = np.percentile(distances, 50)
            accepted_idx = distances <= epsilon

        # SMC iterations
        for gen in range(1, self.n_generations):
            # Adaptive epsilon: alpha quantile of current distances
            epsilon = np.percentile(distances[accepted_idx], self.alpha * 100)
            epsilon = max(epsilon, 1.0)  # Minimum 1 kWh/m² tolerance

            # Resample and perturb
            particles, weights = self._resample_and_perturb(
                particles, weights, accepted_idx, gen
            )

            # Evaluate new particles
            distances = self._compute_distances(particles, measured_kwh_m2)
            accepted_idx = distances <= epsilon
            n_accepted = accepted_idx.sum()

            logger.info(f"Gen {gen}: epsilon={epsilon:.2f}, accepted={n_accepted}/{self.n_particles}")

            if epsilon < 2.0:  # Good enough
                break

        # Build posterior from final particles
        samples = []
        for i in range(self.n_particles):
            if accepted_idx[i]:
                samples.append(PosteriorSample(
                    params={name: particles[name][i] for name in self.param_names},
                    weight=weights[i],
                    distance=distances[i],
                ))

        return CalibrationPosterior(
            samples=samples,
            param_names=self.param_names,
            measured_value=measured_kwh_m2,
            epsilon_final=epsilon,
        )

    def _sample_from_prior(self, n: int) -> Dict[str, np.ndarray]:
        """Sample n particles from prior distributions."""
        particles = {}
        for name in self.param_names:
            if name in self.priors.priors:
                particles[name] = self.priors.priors[name].sample(n, self.rng)
            else:
                # Use surrogate bounds as uniform prior
                bounds = self.predictor.surrogate.param_bounds[name]
                particles[name] = self.rng.uniform(bounds[0], bounds[1], size=n)
        return particles

    def _compute_distances(
        self,
        particles: Dict[str, np.ndarray],
        measured: float
    ) -> np.ndarray:
        """Compute distance from measured value for all particles."""
        n = len(list(particles.values())[0])
        predictions = np.zeros(n)

        # Batch predict using surrogate
        params_list = [
            {name: particles[name][i] for name in self.param_names}
            for i in range(n)
        ]
        predictions = self.predictor.predict_batch(params_list)

        return np.abs(predictions - measured)

    def _resample_and_perturb(
        self,
        particles: Dict[str, np.ndarray],
        weights: np.ndarray,
        accepted_idx: np.ndarray,
        generation: int,
    ) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
        """Resample accepted particles and perturb with kernel."""
        # Normalize weights for accepted particles
        accepted_weights = weights.copy()
        accepted_weights[~accepted_idx] = 0
        accepted_weights /= accepted_weights.sum()

        # Resample indices
        indices = self.rng.choice(
            self.n_particles,
            size=self.n_particles,
            p=accepted_weights
        )

        # Compute perturbation kernel (adaptive bandwidth)
        bandwidths = {}
        for name in self.param_names:
            accepted_values = particles[name][accepted_idx]
            bandwidths[name] = 2 * np.std(accepted_values) / (generation + 1)

        # Perturb resampled particles
        new_particles = {}
        for name in self.param_names:
            values = particles[name][indices]
            perturbation = self.rng.normal(0, bandwidths[name], size=self.n_particles)
            new_values = values + perturbation

            # Clip to bounds
            if name in self.priors.priors:
                prior = self.priors.priors[name]
                if "low" in prior.params:
                    new_values = np.clip(new_values, prior.params["low"], prior.params.get("high", np.inf))
            else:
                bounds = self.predictor.surrogate.param_bounds[name]
                new_values = np.clip(new_values, bounds[0], bounds[1])

            new_particles[name] = new_values

        # Compute new weights (importance weights)
        new_weights = np.ones(self.n_particles)

        return new_particles, new_weights


class UncertaintyPropagator:
    """
    Propagate parameter uncertainty to energy predictions.

    Given a calibrated posterior, compute prediction intervals
    for baseline and ECM scenarios.
    """

    def __init__(
        self,
        predictor: SurrogatePredictor,
        posterior: CalibrationPosterior,
    ):
        self.predictor = predictor
        self.posterior = posterior

    def predict_with_uncertainty(
        self,
        n_samples: int = 500,
    ) -> Tuple[float, float, Tuple[float, float]]:
        """
        Predict heating with uncertainty from posterior.

        Returns:
            (mean, std, (ci_low, ci_high))
        """
        # Sample from posterior
        weights = self.posterior.weights
        indices = np.random.choice(
            len(self.posterior.samples),
            size=n_samples,
            p=weights
        )

        predictions = []
        for idx in indices:
            sample = self.posterior.samples[idx]
            pred = self.predictor.predict(sample.params)
            predictions.append(pred)

        predictions = np.array(predictions)

        return (
            np.mean(predictions),
            np.std(predictions),
            (np.percentile(predictions, 5), np.percentile(predictions, 95))
        )

    def compute_savings_distribution(
        self,
        ecm_effect: Callable[[Dict[str, float]], Dict[str, float]],
        n_samples: int = 500,
    ) -> Dict[str, float]:
        """
        Compute savings distribution accounting for uncertainty.

        Args:
            ecm_effect: Function that modifies parameters for ECM scenario
            n_samples: Number of Monte Carlo samples

        Returns:
            Dict with mean, std, ci_90 for savings
        """
        weights = self.posterior.weights
        indices = np.random.choice(
            len(self.posterior.samples),
            size=n_samples,
            p=weights
        )

        baseline_preds = []
        ecm_preds = []

        for idx in indices:
            sample = self.posterior.samples[idx]

            # Baseline prediction
            baseline = self.predictor.predict(sample.params)
            baseline_preds.append(baseline)

            # ECM scenario
            ecm_params = ecm_effect(sample.params)
            ecm = self.predictor.predict(ecm_params)
            ecm_preds.append(ecm)

        baseline_preds = np.array(baseline_preds)
        ecm_preds = np.array(ecm_preds)
        savings = baseline_preds - ecm_preds
        savings_pct = 100 * savings / baseline_preds

        return {
            "savings_kwh_m2_mean": np.mean(savings),
            "savings_kwh_m2_std": np.std(savings),
            "savings_percent_mean": np.mean(savings_pct),
            "savings_percent_std": np.std(savings_pct),
            "savings_percent_ci_90": (
                np.percentile(savings_pct, 5),
                np.percentile(savings_pct, 95)
            ),
        }
