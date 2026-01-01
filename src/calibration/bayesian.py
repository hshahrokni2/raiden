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
from typing import Any, Dict, List, Tuple, Optional, Callable, Literal
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

    @classmethod
    def from_building_context(
        cls,
        archetype_id: str,
        existing_measures: Optional[set] = None,
        ventilation_type: Optional[str] = None,
        heating_system: Optional[str] = None,
        energy_class: Optional[str] = None,
        calibration_hints: Optional[Dict[str, Any]] = None,
        measured_kwh_m2: Optional[float] = None,
        construction_year: Optional[int] = None,
        restaurant_pct: float = 0.0,
        commercial_pct: float = 0.0,
    ) -> "CalibrationPriors":
        """
        Create priors constrained by building context (detected existing measures).

        Based on Kennedy & O'Hagan guidelines: Use informative priors when evidence
        exists, but don't make them too strong to dominate data.

        IMPORTANT: This method includes a "reality check" that uses actual measured
        energy to infer real system performance. For example, if a building claims
        FTX but uses 100+ kWh/m² (vs expected ~50), the FTX is clearly not working.
        This is based on real-world data from Hammarby Sjöstad where 86% of buildings
        have non-functional heat recovery despite claiming FTX.

        Mixed-use adjustment: Restaurants typically use F-only ventilation (no heat
        recovery) with very high airflow for kitchen exhaust. A 6% restaurant share
        can add 5-10 kWh/m² to building average due to ventilation losses.

        Args:
            archetype_id: Building era identifier (e.g., "1996_2010")
            existing_measures: Set of ExistingMeasure enums from building context
            ventilation_type: Detected ventilation (F, FT, FTX, S)
            heating_system: Detected heating (district_heating, heat_pump, etc.)
            energy_class: Energy declaration class (A-G)
            calibration_hints: Hints from LLM archetype reasoner (renovation detection)
                Keys may include:
                - window_u_value_adjustment: float (e.g., -0.5 for better windows)
                - infiltration_adjustment: float (e.g., -0.02 for tighter)
                - wall_u_value_adjustment: float (e.g., -0.15 for added insulation)
                - roof_u_value_adjustment: float (e.g., -0.10 for added insulation)
                - ventilation_efficiency: float (e.g., 0.80 for FTX)
                - heat_recovery: bool (True if FTX detected)
            measured_kwh_m2: Actual measured space heating energy (for reality check)
            construction_year: Building construction year (for expected energy calc)
            restaurant_pct: Percentage of Atemp that is restaurant (0-100)
            commercial_pct: Total commercial percentage (retail + office + restaurant)

        Returns:
            CalibrationPriors with context-constrained distributions
        """
        # Start with archetype-based priors
        priors = cls.from_archetype(archetype_id)

        # Import here to avoid circular imports
        try:
            from ..core.building_context import ExistingMeasure
        except ImportError:
            ExistingMeasure = None

        if existing_measures is None:
            existing_measures = set()

        # ═══════════════════════════════════════════════════════════════════════
        # REALITY CHECK: Use actual energy data to infer real system performance
        # ═══════════════════════════════════════════════════════════════════════
        # Based on Hammarby Sjöstad research: 86% of buildings have non-functional
        # heat recovery despite claiming FTX. Use measured energy to reality-check.

        ftx_is_functional = True  # Default assumption
        inferred_heat_recovery = None

        if measured_kwh_m2 is not None and construction_year is not None:
            # Calculate expected energy for era WITH working FTX
            # Swedish buildings with functional FTX typically achieve:
            expected_with_ftx = {
                2010: 45,  # Post-2010: ~45 kWh/m²
                2005: 50,  # 2005-2010: ~50 kWh/m²
                2000: 55,  # 2000-2005: ~55 kWh/m²
                1995: 60,  # 1995-2000: ~60 kWh/m²
                1990: 70,  # 1990-1995: ~70 kWh/m²
                1985: 80,  # 1985-1990: ~80 kWh/m²
                1980: 90,  # 1980-1985: ~90 kWh/m²
                1975: 100, # 1975-1980: ~100 kWh/m²
                1960: 120, # Miljonprogrammet: ~120 kWh/m²
                1945: 140, # Post-war: ~140 kWh/m²
                0: 160,    # Pre-war: ~160 kWh/m²
            }

            # Find expected for this era (residential with working FTX)
            expected_residential = 55  # Default
            for year, kwh in sorted(expected_with_ftx.items(), reverse=True):
                if construction_year >= year:
                    expected_residential = kwh
                    break

            # ═══════════════════════════════════════════════════════════════════
            # MIXED-USE ADJUSTMENT: Commercial/restaurant uses F-only ventilation
            # ═══════════════════════════════════════════════════════════════════
            # Restaurants: ~180 kWh/m² (kitchen exhaust, no heat recovery possible)
            # Retail: ~120 kWh/m² (F-only, higher occupancy)
            # These areas have NO heat recovery regardless of residential FTX
            restaurant_energy = 180  # kWh/m² for restaurant areas
            retail_energy = 120  # kWh/m² for retail/office areas

            residential_pct_frac = max(0.0, 1.0 - (restaurant_pct + commercial_pct) / 100)
            restaurant_pct_frac = restaurant_pct / 100
            other_commercial_pct_frac = max(0.0, commercial_pct / 100 - restaurant_pct_frac)

            # Weighted expected energy accounting for mixed-use
            expected_kwh = (
                residential_pct_frac * expected_residential +
                restaurant_pct_frac * restaurant_energy +
                other_commercial_pct_frac * retail_energy
            )

            if restaurant_pct > 0 or commercial_pct > 0:
                logger.info(
                    f"MIXED-USE ADJUSTMENT: Residential {residential_pct_frac:.0%} @ {expected_residential} + "
                    f"Restaurant {restaurant_pct_frac:.0%} @ {restaurant_energy} + "
                    f"Other commercial {other_commercial_pct_frac:.0%} @ {retail_energy} = "
                    f"Expected {expected_kwh:.1f} kWh/m²"
                )

            # Calculate performance gap
            performance_ratio = measured_kwh_m2 / expected_kwh if expected_kwh > 0 else 1.0

            if performance_ratio > 1.5:
                # Building uses 50%+ more energy than expected
                # FTX is likely non-functional or degraded
                ftx_is_functional = False

                # Infer actual heat recovery from energy gap
                # If using 2x expected, heat recovery is ~0%
                # If using 1.5x expected, heat recovery is ~25%
                # Linear interpolation between 0% at 2x and 75% at 1x
                inferred_heat_recovery = max(0.0, min(0.75, 1.5 - performance_ratio))

                logger.warning(
                    f"REALITY CHECK: Measured {measured_kwh_m2:.0f} kWh/m² vs expected "
                    f"{expected_kwh:.0f} kWh/m² (ratio={performance_ratio:.1f}x). "
                    f"FTX likely non-functional. Inferred heat recovery: {inferred_heat_recovery:.0%}"
                )

            elif performance_ratio > 1.2:
                # Moderately higher than expected - FTX may be degraded
                inferred_heat_recovery = max(0.30, 0.85 - (performance_ratio - 1.0) * 0.5)
                logger.info(
                    f"REALITY CHECK: Measured {measured_kwh_m2:.0f} kWh/m² vs expected "
                    f"{expected_kwh:.0f} kWh/m² (ratio={performance_ratio:.1f}x). "
                    f"FTX may be degraded. Inferred heat recovery: {inferred_heat_recovery:.0%}"
                )

        # ═══════════════════════════════════════════════════════════════════════
        # PERFORMANCE GAP EXPANSION: When building uses much more energy than
        # expected, expand parameter bounds to allow the model to reach high
        # energy values. This is critical for simulating poorly-performing
        # buildings like many in Hammarby Sjöstad (86% have non-functional FTX).
        # ═══════════════════════════════════════════════════════════════════════
        if measured_kwh_m2 is not None and measured_kwh_m2 > 80:
            # High-energy building: expand bounds to allow model to reach target

            # Scale expansion based on how much energy the building uses
            # 80 kWh/m² → mild expansion, 150 kWh/m² → strong expansion
            expansion_factor = min(2.0, max(1.0, (measured_kwh_m2 - 50) / 50))

            logger.info(
                f"PERFORMANCE GAP EXPANSION: Building uses {measured_kwh_m2:.0f} kWh/m². "
                f"Expansion factor: {expansion_factor:.1f}x"
            )

            # INFILTRATION: Old/leaky buildings can have 0.3-0.5+ ACH
            # Swedish pre-1975 buildings often have 0.4-0.6 ACH
            current_inf = priors.priors.get("infiltration_ach")
            if current_inf:
                old_high = current_inf.params.get("high", 0.20)
                # Expand to allow up to 0.5 ACH for high-energy buildings
                new_high = min(0.50, old_high * expansion_factor)
                priors.priors["infiltration_ach"] = Prior(
                    name="infiltration_ach",
                    distribution="uniform",
                    params={"low": 0.02, "high": new_high}
                )
                logger.info(f"  infiltration_ach expanded: [0.02, {new_high:.2f}]")

            # WALL U-VALUE: Poor insulation can have U > 1.5 W/m²K
            # Uninsulated concrete panel: ~1.5-2.0, brick: ~1.2-1.8
            # For high-energy buildings, scale toward full range [0.15, 2.5]
            current_wall = priors.priors.get("wall_u_value")
            if current_wall:
                old_high = current_wall.params.get("high", 1.5)
                max_possible = 2.5  # Maximum for Swedish building stock
                # Interpolate between old_high and max_possible based on expansion
                new_high = min(max_possible, old_high + (max_possible - old_high) * (expansion_factor - 1.0))
                new_low = min(0.3, current_wall.params.get("low", 0.15))
                priors.priors["wall_u_value"] = Prior(
                    name="wall_u_value",
                    distribution="uniform",
                    params={"low": new_low, "high": new_high}
                )
                logger.info(f"  wall_u_value expanded: [{new_low:.2f}, {new_high:.2f}]")

            # ROOF U-VALUE: Uninsulated flat roofs can be 0.8-1.5 W/m²K
            current_roof = priors.priors.get("roof_u_value")
            if current_roof:
                old_high = current_roof.params.get("high", 0.6)
                max_possible = 1.5
                new_high = min(max_possible, old_high + (max_possible - old_high) * (expansion_factor - 1.0))
                priors.priors["roof_u_value"] = Prior(
                    name="roof_u_value",
                    distribution="uniform",
                    params={"low": 0.10, "high": new_high}
                )
                logger.info(f"  roof_u_value expanded: [0.10, {new_high:.2f}]")

            # FLOOR U-VALUE: Uninsulated slab-on-grade can be 0.8-1.5 W/m²K
            current_floor = priors.priors.get("floor_u_value")
            if current_floor:
                old_high = current_floor.params.get("high", 0.8)
                max_possible = 1.5
                new_high = min(max_possible, old_high + (max_possible - old_high) * (expansion_factor - 1.0))
                priors.priors["floor_u_value"] = Prior(
                    name="floor_u_value",
                    distribution="uniform",
                    params={"low": 0.15, "high": new_high}
                )
                logger.info(f"  floor_u_value expanded: [0.15, {new_high:.2f}]")

            # WINDOW U-VALUE: Old single-pane windows: 4.5-5.5 W/m²K
            # Double-pane without low-e: 2.5-3.0, old double-pane: 2.8-3.5
            current_win = priors.priors.get("window_u_value")
            if current_win:
                old_high = current_win.params.get("high", 2.5)
                max_possible = 4.0  # Old double-pane windows
                new_high = min(max_possible, old_high + (max_possible - old_high) * (expansion_factor - 1.0))
                priors.priors["window_u_value"] = Prior(
                    name="window_u_value",
                    distribution="uniform",
                    params={"low": 0.7, "high": new_high}
                )
                logger.info(f"  window_u_value expanded: [0.7, {new_high:.2f}]")

            # HEATING SETPOINT: Some buildings run at 22-24°C
            current_setpoint = priors.priors.get("heating_setpoint")
            if current_setpoint and measured_kwh_m2 > 100:
                priors.priors["heating_setpoint"] = Prior(
                    name="heating_setpoint",
                    distribution="uniform",
                    params={"low": 19.0, "high": 24.0}
                )
                logger.info(f"  heating_setpoint expanded: [19.0, 24.0]")

        # === VENTILATION SYSTEM CONSTRAINTS ===
        # FTX detected → narrow heat recovery to realistic range
        has_ftx = (
            ventilation_type in ("FTX", "ftx") or
            (ExistingMeasure and "FTX_SYSTEM" in str(existing_measures))
        )

        if has_ftx and ftx_is_functional:
            # FTX systems typically have 70-85% efficiency
            # Use Beta distribution peaked around 0.78
            priors.priors["heat_recovery_eff"] = Prior(
                name="heat_recovery_eff",
                distribution="beta",
                params={"alpha": 8, "beta": 2.5, "low": 0.65, "high": 0.90}
            )
            logger.info("Context: FTX detected → heat_recovery_eff constrained to [0.65, 0.90]")

        elif has_ftx and not ftx_is_functional:
            # FTX claimed but REALITY CHECK says it's not working
            # Use inferred heat recovery from actual energy data
            if inferred_heat_recovery is not None:
                # Center prior around inferred value
                low = max(0.0, inferred_heat_recovery - 0.15)
                high = min(0.60, inferred_heat_recovery + 0.15)
                priors.priors["heat_recovery_eff"] = Prior(
                    name="heat_recovery_eff",
                    distribution="uniform",
                    params={"low": low, "high": high}
                )
                logger.warning(
                    f"Context: FTX claimed but non-functional → heat_recovery_eff "
                    f"constrained to [{low:.0%}, {high:.0%}] based on actual energy"
                )

        elif ventilation_type in ("FT", "ft"):
            # FT (exhaust + supply, no recovery) → minimal heat recovery
            priors.priors["heat_recovery_eff"] = Prior(
                name="heat_recovery_eff",
                distribution="beta",
                params={"alpha": 2, "beta": 8, "low": 0.0, "high": 0.30}
            )
            logger.info("Context: FT detected → heat_recovery_eff constrained to [0.0, 0.30]")

        elif ventilation_type in ("F", "f", "S", "s"):
            # F (exhaust only) or S (natural) → no heat recovery
            priors.priors["heat_recovery_eff"] = Prior(
                name="heat_recovery_eff",
                distribution="beta",
                params={"alpha": 1, "beta": 10, "low": 0.0, "high": 0.10}
            )
            logger.info(f"Context: {ventilation_type} detected → heat_recovery_eff ≈ 0")

        # === AIR SEALING CONSTRAINTS ===
        has_air_sealing = ExistingMeasure and any(
            "AIR_SEALING" in str(m) for m in existing_measures
        )

        if has_air_sealing:
            # Well-sealed building
            priors.priors["infiltration_ach"] = Prior(
                name="infiltration_ach",
                distribution="truncnorm",
                params={"mean": 0.04, "std": 0.01, "low": 0.02, "high": 0.08}
            )
            logger.info("Context: Air sealing detected → infiltration_ach constrained to [0.02, 0.08]")

        # === WINDOW CONSTRAINTS ===
        has_new_windows = ExistingMeasure and any(
            "WINDOW" in str(m) for m in existing_measures
        )

        if has_new_windows:
            # Modern windows (triple-glazed)
            priors.priors["window_u_value"] = Prior(
                name="window_u_value",
                distribution="truncnorm",
                params={"mean": 0.9, "std": 0.15, "low": 0.7, "high": 1.3}
            )
            logger.info("Context: New windows detected → window_u_value constrained to [0.7, 1.3]")

        # === ENERGY CLASS CONSTRAINTS ===
        # Good energy class suggests well-performing envelope
        if energy_class in ("A", "B"):
            # Very efficient building → tighten all envelope U-values
            for param in ["wall_u_value", "roof_u_value", "floor_u_value"]:
                if param in priors.priors:
                    current = priors.priors[param].params
                    # Reduce upper bound significantly
                    new_high = min(current.get("high", 0.5), 0.35)
                    priors.priors[param] = Prior(
                        name=param,
                        distribution="uniform",
                        params={"low": current.get("low", 0.10), "high": new_high}
                    )
            logger.info(f"Context: Energy class {energy_class} → envelope U-values constrained")

        elif energy_class in ("F", "G"):
            # Poor energy class → likely older, less insulated
            for param in ["wall_u_value", "roof_u_value"]:
                if param in priors.priors:
                    current = priors.priors[param].params
                    # Increase lower bound
                    new_low = max(current.get("low", 0.3), 0.40)
                    priors.priors[param] = Prior(
                        name=param,
                        distribution="uniform",
                        params={"low": new_low, "high": current.get("high", 1.5)}
                    )
            logger.info(f"Context: Energy class {energy_class} → envelope U-values relaxed upward")

        # === HEATING SETPOINT CONSTRAINTS ===
        # Modern buildings often have lower setpoints
        if energy_class in ("A", "B", "C"):
            priors.priors["heating_setpoint"] = Prior(
                name="heating_setpoint",
                distribution="truncnorm",
                params={"mean": 20.5, "std": 0.8, "low": 19.0, "high": 22.0}
            )

        # === CALIBRATION HINTS FROM LLM REASONER ===
        # Apply hints from renovation detection / anomaly analysis
        if calibration_hints:
            logger.info(f"Applying calibration hints from LLM: {list(calibration_hints.keys())}")

            # Window U-value adjustment (negative = better windows)
            if "window_u_value_adjustment" in calibration_hints:
                adj = calibration_hints["window_u_value_adjustment"]
                if "window_u_value" in priors.priors:
                    current = priors.priors["window_u_value"].params
                    new_mean = current.get("mean", 1.2) + adj
                    new_mean = max(0.7, min(2.5, new_mean))  # Clamp to valid range
                    priors.priors["window_u_value"] = Prior(
                        name="window_u_value",
                        distribution="truncnorm",
                        params={
                            "mean": new_mean,
                            "std": current.get("std", 0.3) * 0.7,  # Tighter std with hint
                            "low": max(0.7, new_mean - 0.4),
                            "high": min(2.5, new_mean + 0.4),
                        }
                    )
                    logger.info(f"Hint: window_u_value adjusted by {adj:.2f} → mean={new_mean:.2f}")

            # Infiltration adjustment (negative = tighter building)
            if "infiltration_adjustment" in calibration_hints:
                adj = calibration_hints["infiltration_adjustment"]
                if "infiltration_ach" in priors.priors:
                    current = priors.priors["infiltration_ach"].params
                    new_mean = current.get("mean", 0.08) + adj
                    new_mean = max(0.02, min(0.20, new_mean))
                    priors.priors["infiltration_ach"] = Prior(
                        name="infiltration_ach",
                        distribution="truncnorm",
                        params={
                            "mean": new_mean,
                            "std": current.get("std", 0.02) * 0.7,
                            "low": max(0.02, new_mean - 0.03),
                            "high": min(0.20, new_mean + 0.03),
                        }
                    )
                    logger.info(f"Hint: infiltration_ach adjusted by {adj:.3f} → mean={new_mean:.3f}")

            # Wall U-value adjustment (negative = added insulation)
            if "wall_u_value_adjustment" in calibration_hints:
                adj = calibration_hints["wall_u_value_adjustment"]
                if "wall_u_value" in priors.priors:
                    current = priors.priors["wall_u_value"].params
                    # For uniform, shift both bounds
                    new_low = max(0.15, current.get("low", 0.3) + adj)
                    new_high = max(new_low + 0.2, current.get("high", 1.0) + adj)
                    new_high = min(1.5, new_high)
                    priors.priors["wall_u_value"] = Prior(
                        name="wall_u_value",
                        distribution="uniform",
                        params={"low": new_low, "high": new_high}
                    )
                    logger.info(f"Hint: wall_u_value bounds shifted by {adj:.2f} → [{new_low:.2f}, {new_high:.2f}]")

            # Roof U-value adjustment
            if "roof_u_value_adjustment" in calibration_hints:
                adj = calibration_hints["roof_u_value_adjustment"]
                if "roof_u_value" in priors.priors:
                    current = priors.priors["roof_u_value"].params
                    new_low = max(0.10, current.get("low", 0.15) + adj)
                    new_high = max(new_low + 0.1, current.get("high", 0.5) + adj)
                    new_high = min(0.60, new_high)
                    priors.priors["roof_u_value"] = Prior(
                        name="roof_u_value",
                        distribution="uniform",
                        params={"low": new_low, "high": new_high}
                    )
                    logger.info(f"Hint: roof_u_value bounds shifted by {adj:.2f} → [{new_low:.2f}, {new_high:.2f}]")

            # Ventilation efficiency hint (from detected FTX)
            if "ventilation_efficiency" in calibration_hints:
                eff = calibration_hints["ventilation_efficiency"]
                if "heat_recovery_eff" in priors.priors:
                    # Tight prior around the detected efficiency
                    priors.priors["heat_recovery_eff"] = Prior(
                        name="heat_recovery_eff",
                        distribution="beta",
                        params={
                            "alpha": 10,  # Strong prior
                            "beta": 2.5,
                            "low": max(0.60, eff - 0.10),
                            "high": min(0.92, eff + 0.10),
                        }
                    )
                    logger.info(f"Hint: heat_recovery_eff set to ~{eff:.0%} (detected FTX)")

            # Heat recovery flag
            if calibration_hints.get("heat_recovery") is True:
                if "heat_recovery_eff" in priors.priors:
                    # Ensure we're modeling heat recovery
                    current = priors.priors["heat_recovery_eff"].params
                    if current.get("high", 0) < 0.5:
                        # Override to FTX range
                        priors.priors["heat_recovery_eff"] = Prior(
                            name="heat_recovery_eff",
                            distribution="beta",
                            params={"alpha": 6, "beta": 2, "low": 0.60, "high": 0.88}
                        )
                        logger.info("Hint: heat_recovery enabled → efficiency [0.60, 0.88]")

            # Log renovation note if present
            if "renovation_note" in calibration_hints:
                logger.info(f"LLM Renovation Analysis: {calibration_hints['renovation_note']}")

        return priors

    def sample(self, n: int, rng: Optional[np.random.Generator] = None) -> Dict[str, np.ndarray]:
        """Sample n values from all priors."""
        if rng is None:
            rng = np.random.default_rng()
        return {name: prior.sample(n, rng) for name, prior in self.priors.items()}

    def filter_to_parameters(self, param_names: List[str]) -> "CalibrationPriors":
        """
        Create new CalibrationPriors with only specified parameters.

        Used by Morris screening to focus calibration on identifiable params.
        Non-important parameters should be fixed at archetype defaults.

        Args:
            param_names: Parameters to keep (e.g., from Morris ranking)

        Returns:
            New CalibrationPriors with filtered parameters
        """
        filtered = {
            name: self.priors[name]
            for name in param_names
            if name in self.priors
        }
        logger.info(
            f"Filtered priors: {len(self.priors)} → {len(filtered)} parameters "
            f"({', '.join(filtered.keys())})"
        )
        return CalibrationPriors(priors=filtered)


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


# ============================================================================
# ECM UNCERTAINTY PROPAGATION
# ============================================================================

# ECM effects on surrogate model parameters
# Maps ECM IDs to parameter modifications
ECM_PARAMETER_EFFECTS: Dict[str, Dict[str, Any]] = {
    # Envelope ECMs
    "wall_external_insulation": {
        "wall_u_value": {"operation": "subtract", "value": 0.15},  # -0.15 W/m²K
    },
    "wall_internal_insulation": {
        "wall_u_value": {"operation": "subtract", "value": 0.10},
    },
    "roof_insulation": {
        "roof_u_value": {"operation": "subtract", "value": 0.10},
    },
    "floor_insulation": {
        "floor_u_value": {"operation": "subtract", "value": 0.10},
    },
    "window_replacement": {
        "window_u_value": {"operation": "set", "value": 0.9},  # Triple-glazed
    },
    "air_sealing": {
        "infiltration_ach": {"operation": "multiply", "value": 0.6},  # -40%
    },

    # Ventilation ECMs
    "ftx_installation": {
        "heat_recovery_eff": {"operation": "set", "value": 0.80},
    },
    "ftx_upgrade": {
        "heat_recovery_eff": {"operation": "set", "value": 0.85},
    },
    "ftx_overhaul": {
        # Restore malfunctioning FTX to working state
        "heat_recovery_eff": {"operation": "set", "value": 0.75},
        "infiltration_ach": {"operation": "multiply", "value": 0.85},  # Fixed seals/dampers
    },
    "demand_controlled_ventilation": {
        # DCV reduces average ventilation → affects heating indirectly
        # Approximate as infiltration reduction for surrogate
        "infiltration_ach": {"operation": "multiply", "value": 0.85},
    },

    # Controls ECMs
    "smart_thermostats": {
        "heating_setpoint": {"operation": "subtract", "value": 1.0},  # -1°C
    },

    # Additional Envelope ECMs
    "basement_insulation": {
        "floor_u_value": {"operation": "subtract", "value": 0.20},  # Significant improvement
    },
    "thermal_bridge_remediation": {
        "infiltration_ach": {"operation": "multiply", "value": 0.90},  # 10% reduction
    },
    "facade_renovation": {
        # Comprehensive: wall + windows + air sealing
        "wall_u_value": {"operation": "subtract", "value": 0.15},
        "window_u_value": {"operation": "set", "value": 1.0},
        "infiltration_ach": {"operation": "multiply", "value": 0.6},
    },
    "entrance_door_replacement": {
        "infiltration_ach": {"operation": "multiply", "value": 0.95},  # Small improvement
    },
    "pipe_insulation": {
        # Distribution losses, approximate as 2% overall reduction
        "infiltration_ach": {"operation": "multiply", "value": 0.98},
    },

    # HVAC & Controls ECMs
    "radiator_fans": {
        "heating_setpoint": {"operation": "subtract", "value": 1.5},  # Enable lower setpoint
    },
    "heat_recovery_dhw": {
        # DHW not in surrogate, approximate as small heating reduction
        "heating_setpoint": {"operation": "subtract", "value": 0.3},
    },
    "vrf_system": {
        # High efficiency HP
        "heat_recovery_eff": {"operation": "set", "value": 0.85},
    },
    "occupancy_sensors": {
        # Reduces lighting and indirectly heating (less internal gains)
        "heating_setpoint": {"operation": "subtract", "value": 0.2},
    },
    "daylight_sensors": {
        "heating_setpoint": {"operation": "subtract", "value": 0.2},
    },
    "predictive_control": {
        "heating_setpoint": {"operation": "subtract", "value": 0.5},
    },
    "fault_detection": {
        "infiltration_ach": {"operation": "multiply", "value": 0.95},
    },
    "building_automation_system": {
        "heating_setpoint": {"operation": "subtract", "value": 0.5},
    },

    # Metering & Monitoring
    "individual_metering": {
        # Behavioral savings
        "heating_setpoint": {"operation": "subtract", "value": 0.5},
    },
    "energy_monitoring": {
        "infiltration_ach": {"operation": "multiply", "value": 0.97},
    },
    "recommissioning": {
        "heating_setpoint": {"operation": "subtract", "value": 0.3},
    },

    # Lighting (affects internal gains → heating)
    "led_lighting": {
        # Less internal gains → more heating needed (negative effect in heating season)
        "heating_setpoint": {"operation": "add", "value": 0.3},
    },
    "led_common_areas": {
        "heating_setpoint": {"operation": "add", "value": 0.1},
    },
    "led_outdoor": {
        # No heating impact
        "infiltration_ach": {"operation": "multiply", "value": 1.0},
    },

    # DHW ECMs
    "dhw_circulation_optimization": {
        "infiltration_ach": {"operation": "multiply", "value": 0.99},
    },
    "dhw_tank_insulation": {
        "infiltration_ach": {"operation": "multiply", "value": 0.99},
    },
    "low_flow_fixtures": {
        "infiltration_ach": {"operation": "multiply", "value": 0.99},
    },

    # Renewables & Storage
    "solar_pv": {
        # No direct heating impact in E+ model
        "infiltration_ach": {"operation": "multiply", "value": 1.0},
    },
    "solar_thermal": {
        # Reduces heating demand for DHW
        "heating_setpoint": {"operation": "subtract", "value": 0.2},
    },
    "battery_storage": {
        # No direct heating impact
        "infiltration_ach": {"operation": "multiply", "value": 1.0},
    },

    # Heat pump types
    "air_source_heat_pump": {
        # Similar to ground source but slightly less efficient tightening
        "infiltration_ach": {"operation": "multiply", "value": 0.88},
    },
    "heat_pump_water_heater": {
        # No direct space heating impact (DHW only)
        "infiltration_ach": {"operation": "multiply", "value": 1.0},
    },

    # Default for unknown ECMs
    "_default": {
        # Conservative 5% reduction in heating need
        "infiltration_ach": {"operation": "multiply", "value": 0.95},
    },
}


def get_ecm_effect(ecm_id: str) -> Callable[[Dict[str, float]], Dict[str, float]]:
    """
    Get parameter modification function for an ECM.

    Args:
        ecm_id: ECM identifier

    Returns:
        Function that takes baseline params and returns ECM params
    """
    effects = ECM_PARAMETER_EFFECTS.get(ecm_id, ECM_PARAMETER_EFFECTS["_default"])

    def apply_effect(params: Dict[str, float]) -> Dict[str, float]:
        modified = params.copy()
        for param_name, effect in effects.items():
            if param_name in modified:
                op = effect["operation"]
                val = effect["value"]
                if op == "set":
                    modified[param_name] = val
                elif op == "add":
                    modified[param_name] = modified[param_name] + val
                elif op == "subtract":
                    modified[param_name] = max(0.01, modified[param_name] - val)
                elif op == "multiply":
                    modified[param_name] = modified[param_name] * val
        return modified

    return apply_effect


class ECMUncertaintyPropagator:
    """
    Monte Carlo uncertainty propagation for ECM savings.

    Uses posterior samples to properly propagate parameter uncertainty
    through ECM effects, avoiding the sqrt(2) approximation.
    """

    def __init__(
        self,
        predictor: SurrogatePredictor,
        posterior: CalibrationPosterior,
        n_samples: int = 500,
    ):
        """
        Initialize the propagator.

        Args:
            predictor: Trained surrogate predictor
            posterior: Calibration posterior from ABC-SMC
            n_samples: Number of MC samples (default 500)
        """
        self.predictor = predictor
        self.posterior = posterior
        self.n_samples = n_samples
        self._base_propagator = UncertaintyPropagator(predictor, posterior)

    def compute_ecm_uncertainty(
        self,
        ecm_id: str,
        simulated_savings_kwh_m2: Optional[float] = None,
    ) -> Dict[str, float]:
        """
        Compute MC uncertainty for a specific ECM.

        If simulated_savings_kwh_m2 is provided (from E+ simulation),
        we use that as the mean and only compute the uncertainty.

        Args:
            ecm_id: ECM identifier
            simulated_savings_kwh_m2: Optional savings from actual simulation

        Returns:
            Dict with mean, std, ci_90 for savings
        """
        ecm_effect = get_ecm_effect(ecm_id)

        # Use the base propagator for MC
        result = self._base_propagator.compute_savings_distribution(
            ecm_effect=ecm_effect,
            n_samples=self.n_samples,
        )

        # If we have actual simulation results, use that as mean
        # but keep the relative uncertainty from MC
        if simulated_savings_kwh_m2 is not None:
            mc_mean = result["savings_kwh_m2_mean"]
            mc_std = result["savings_kwh_m2_std"]

            if abs(mc_mean) > 0.1:
                # Scale std by ratio of simulated to MC mean
                cv = mc_std / abs(mc_mean)  # Coefficient of variation
                adjusted_std = simulated_savings_kwh_m2 * cv
            else:
                adjusted_std = mc_std

            result["savings_kwh_m2_mean"] = simulated_savings_kwh_m2
            result["savings_kwh_m2_std"] = adjusted_std

            # Recompute CI based on adjusted values
            result["savings_percent_ci_90"] = (
                max(0, simulated_savings_kwh_m2 - 1.645 * adjusted_std),
                simulated_savings_kwh_m2 + 1.645 * adjusted_std,
            )

        return result

    def compute_all_ecm_uncertainties(
        self,
        ecm_results: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Add uncertainty estimates to a list of ECM results.

        Args:
            ecm_results: List of ECM result dicts with 'ecm_id' and 'savings_percent'

        Returns:
            Same list with added uncertainty fields
        """
        enhanced = []

        for result in ecm_results:
            ecm_id = result.get("ecm_id")
            if not ecm_id:
                enhanced.append(result)
                continue

            # Get simulated savings if available
            simulated_savings = result.get("savings_kwh_m2")
            if simulated_savings is None and "heating_kwh_m2" in result:
                # Compute from baseline and ECM heating
                baseline = result.get("baseline_kwh_m2", 0)
                if baseline > 0:
                    simulated_savings = baseline - result["heating_kwh_m2"]

            # Compute MC uncertainty
            try:
                uncertainty = self.compute_ecm_uncertainty(
                    ecm_id=ecm_id,
                    simulated_savings_kwh_m2=simulated_savings,
                )

                # Add uncertainty fields to result
                result["savings_kwh_m2_std"] = uncertainty["savings_kwh_m2_std"]
                result["savings_kwh_m2_ci_90"] = uncertainty["savings_percent_ci_90"]
                result["uncertainty_method"] = "monte_carlo"

            except Exception as e:
                logger.warning(f"MC uncertainty failed for {ecm_id}: {e}")
                # Fall back to sqrt(2) approximation
                if "kwh_m2_std" in result:
                    import math
                    result["savings_kwh_m2_std"] = math.sqrt(2) * result["kwh_m2_std"]
                    result["uncertainty_method"] = "sqrt2_approx"

            enhanced.append(result)

        return enhanced
