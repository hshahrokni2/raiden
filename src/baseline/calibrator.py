"""
Baseline Calibrator - Calibrate model to energy declaration.

Takes:
- Generated baseline model
- Actual energy consumption (from energy declaration)

Adjusts within plausible ranges:
- Infiltration rate
- Heat recovery efficiency
- Window U-value

Until simulated ≈ measured (within ±10%)
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
from pathlib import Path
import shutil
import logging

from ..simulation import SimulationRunner, ResultsParser
from ..core.idf_parser import IDFParser

logger = logging.getLogger(__name__)


@dataclass
class CalibrationParameter:
    """A parameter that can be adjusted during calibration."""
    name: str
    current_value: float
    min_value: float
    max_value: float
    sensitivity: float  # kWh/m² change per unit change (negative = reduces heating)


@dataclass
class CalibrationResult:
    """Result of calibration process."""
    success: bool
    iterations: int
    final_error_percent: float

    # Adjusted parameters
    adjusted_infiltration_ach: float
    adjusted_heat_recovery: float
    adjusted_window_u: float

    # Energy values
    measured_kwh_m2: float
    initial_kwh_m2: float
    calibrated_kwh_m2: float

    # Calibrated IDF path
    calibrated_idf_path: Optional[Path] = None


class BaselineCalibrator:
    """
    Calibrate baseline model to actual energy consumption.

    Uses iterative adjustment within physically plausible bounds:
    - Won't make building impossibly airtight
    - Won't exceed manufacturer HR specs
    - Won't assume unrealistic window performance

    Usage:
        calibrator = BaselineCalibrator()
        result = calibrator.calibrate(
            idf_path=Path('./baseline.idf'),
            weather_path=Path('./weather.epw'),
            measured_heating_kwh_m2=33.0,
            output_dir=Path('./calibrated')
        )
    """

    # Calibration bounds (physically plausible for Swedish buildings)
    PARAM_BOUNDS = {
        'infiltration': (0.02, 0.15),  # ACH
        'heat_recovery': (0.60, 0.90),  # Effectiveness
        'window_u': (0.7, 1.5),  # W/m²K
    }

    # Approximate sensitivities (kWh/m² per unit change)
    # Negative means reducing heating demand
    PARAM_SENSITIVITIES = {
        'infiltration': 80.0,  # ~80 kWh/m² per ACH change (higher infiltration = more heating)
        'heat_recovery': -50.0,  # ~50 kWh/m² per 0.1 HR change (higher HR = less heating)
        'window_u': 8.0,  # ~8 kWh/m² per W/m²K change (higher U = more heating)
    }

    # Convergence criteria
    MAX_ITERATIONS = 10
    CONVERGENCE_THRESHOLD = 0.10  # 10% error acceptable

    def __init__(self, energyplus_path: Optional[str] = None):
        """
        Initialize calibrator.

        Args:
            energyplus_path: Path to EnergyPlus executable (auto-detect if None)
        """
        self.runner = SimulationRunner(energyplus_path)
        self.results_parser = ResultsParser()
        self.idf_parser = IDFParser()

    def calibrate(
        self,
        idf_path: Path,
        weather_path: Path,
        measured_heating_kwh_m2: float,
        output_dir: Path,
        initial_params: Optional[Dict[str, float]] = None
    ) -> CalibrationResult:
        """
        Calibrate model to measured energy consumption.

        Args:
            idf_path: Path to baseline IDF
            weather_path: Path to weather file
            measured_heating_kwh_m2: Actual energy use from declaration
            output_dir: Directory for calibrated IDF and simulation output
            initial_params: Optional initial parameter values

        Returns:
            CalibrationResult with adjusted parameters
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Read initial IDF
        with open(idf_path) as f:
            idf_content = f.read()

        # Extract current parameter values from IDF
        current_params = self._extract_parameters(idf_content)
        if initial_params:
            current_params.update(initial_params)

        logger.info("Initial parameters:")
        logger.info(f"  Infiltration: {current_params['infiltration']:.3f} ACH")
        logger.info(f"  Heat recovery: {current_params['heat_recovery']:.2f}")
        logger.info(f"  Window U-value: {current_params['window_u']:.2f} W/m²K")

        # Run initial simulation
        logger.info("Running initial simulation...")
        initial_kwh_m2 = self._run_simulation(idf_content, weather_path, output_dir / "iter_0")

        if initial_kwh_m2 is None:
            logger.error("Initial simulation failed!")
            return CalibrationResult(
                success=False,
                iterations=0,
                final_error_percent=float('inf'),
                adjusted_infiltration_ach=current_params['infiltration'],
                adjusted_heat_recovery=current_params['heat_recovery'],
                adjusted_window_u=current_params['window_u'],
                measured_kwh_m2=measured_heating_kwh_m2,
                initial_kwh_m2=0.0,
                calibrated_kwh_m2=0.0,
            )

        logger.info(f"Initial heating: {initial_kwh_m2:.1f} kWh/m²")
        logger.info(f"Target: {measured_heating_kwh_m2:.1f} kWh/m²")

        # Handle heat pump buildings where space heating target is 0 or very small
        if measured_heating_kwh_m2 <= 1.0:
            logger.warning(
                f"Space heating target is {measured_heating_kwh_m2:.1f} kWh/m² - "
                "building likely has heat pump providing all heating. "
                "Skipping iterative calibration."
            )
            calibrated_idf = output_dir / f"{idf_path.stem}_calibrated.idf"
            shutil.copy(idf_path, calibrated_idf)
            return CalibrationResult(
                success=True,
                iterations=0,
                final_error_percent=0.0,  # Can't calculate error when target is 0
                adjusted_infiltration_ach=current_params['infiltration'],
                adjusted_heat_recovery=current_params['heat_recovery'],
                adjusted_window_u=current_params['window_u'],
                measured_kwh_m2=measured_heating_kwh_m2,
                initial_kwh_m2=initial_kwh_m2,
                calibrated_kwh_m2=initial_kwh_m2,  # Use initial (archetype defaults)
                calibrated_idf_path=calibrated_idf,
            )

        error_pct = (initial_kwh_m2 - measured_heating_kwh_m2) / measured_heating_kwh_m2
        logger.info(f"Initial error: {error_pct:+.1%}")

        if abs(error_pct) <= self.CONVERGENCE_THRESHOLD:
            logger.info("Already within tolerance!")
            calibrated_idf = output_dir / f"{idf_path.stem}_calibrated.idf"
            shutil.copy(idf_path, calibrated_idf)
            return CalibrationResult(
                success=True,
                iterations=0,
                final_error_percent=error_pct * 100,
                adjusted_infiltration_ach=current_params['infiltration'],
                adjusted_heat_recovery=current_params['heat_recovery'],
                adjusted_window_u=current_params['window_u'],
                measured_kwh_m2=measured_heating_kwh_m2,
                initial_kwh_m2=initial_kwh_m2,
                calibrated_kwh_m2=initial_kwh_m2,
                calibrated_idf_path=calibrated_idf,
            )

        # Iterative calibration
        best_params = current_params.copy()
        best_error = abs(error_pct)
        best_kwh_m2 = initial_kwh_m2
        best_idf_content = idf_content

        for iteration in range(1, self.MAX_ITERATIONS + 1):
            logger.info(f"--- Iteration {iteration} ---")

            # Calculate required delta
            delta_kwh_m2 = measured_heating_kwh_m2 - best_kwh_m2

            # Adjust parameters proportionally
            new_params = self._calculate_adjustments(
                current_params=best_params,
                delta_kwh_m2=delta_kwh_m2,
                damping=0.7  # Don't overshoot
            )

            logger.info("Adjusted parameters:")
            logger.info(f"  Infiltration: {new_params['infiltration']:.3f} ACH")
            logger.info(f"  Heat recovery: {new_params['heat_recovery']:.2f}")
            logger.info(f"  Window U-value: {new_params['window_u']:.2f} W/m²K")

            # Modify IDF
            modified_idf = self._modify_idf(idf_content, new_params)

            # Run simulation
            iter_output = output_dir / f"iter_{iteration}"
            new_kwh_m2 = self._run_simulation(modified_idf, weather_path, iter_output)

            if new_kwh_m2 is None:
                logger.warning(f"Simulation failed in iteration {iteration}")
                continue

            new_error = abs(new_kwh_m2 - measured_heating_kwh_m2) / measured_heating_kwh_m2
            logger.info(f"New heating: {new_kwh_m2:.1f} kWh/m² (error: {new_error:.1%})")

            # Check if better
            if new_error < best_error:
                best_params = new_params.copy()
                best_error = new_error
                best_kwh_m2 = new_kwh_m2
                best_idf_content = modified_idf
                logger.info("New best solution found!")

            # Check convergence
            if new_error <= self.CONVERGENCE_THRESHOLD:
                logger.info(f"Converged after {iteration} iterations!")
                break

        # Save calibrated IDF
        calibrated_idf = output_dir / f"{idf_path.stem}_calibrated.idf"
        with open(calibrated_idf, 'w') as f:
            f.write(best_idf_content)

        final_error = (best_kwh_m2 - measured_heating_kwh_m2) / measured_heating_kwh_m2 * 100

        logger.info("=== Calibration Complete ===")
        logger.info(f"Final heating: {best_kwh_m2:.1f} kWh/m² (target: {measured_heating_kwh_m2:.1f})")
        logger.info(f"Final error: {final_error:+.1f}%")
        logger.info(f"Calibrated IDF: {calibrated_idf}")

        return CalibrationResult(
            success=abs(final_error) <= self.CONVERGENCE_THRESHOLD * 100,
            iterations=iteration,
            final_error_percent=final_error,
            adjusted_infiltration_ach=best_params['infiltration'],
            adjusted_heat_recovery=best_params['heat_recovery'],
            adjusted_window_u=best_params['window_u'],
            measured_kwh_m2=measured_heating_kwh_m2,
            initial_kwh_m2=initial_kwh_m2,
            calibrated_kwh_m2=best_kwh_m2,
            calibrated_idf_path=calibrated_idf,
        )

    def _extract_parameters(self, idf_content: str) -> Dict[str, float]:
        """Extract current parameter values from IDF using structured parser."""
        # Default values if extraction fails
        defaults = {
            'infiltration': 0.06,  # Default ACH
            'heat_recovery': 0.75,  # Default effectiveness
            'window_u': 1.0,  # Default U-value
        }

        try:
            idf = self.idf_parser.load_string(idf_content)
            params = self.idf_parser.extract_calibration_parameters(idf)

            # Use defaults for any None values
            for key, default in defaults.items():
                if params.get(key) is None:
                    params[key] = default

            return params
        except Exception as e:
            logger.warning(f"Structured parsing failed, using defaults: {e}")
            return defaults

    def _calculate_adjustments(
        self,
        current_params: Dict[str, float],
        delta_kwh_m2: float,
        damping: float = 0.7
    ) -> Dict[str, float]:
        """
        Calculate parameter adjustments to achieve target delta.

        Uses sensitivities to distribute the adjustment across parameters.
        """
        new_params = current_params.copy()

        # Calculate adjustment for each parameter
        # Priority: heat_recovery > infiltration > window_u
        # (based on what's most likely to need adjustment in Swedish buildings)

        # Distribute adjustment: 50% HR, 30% infiltration, 20% windows
        weights = {'heat_recovery': 0.5, 'infiltration': 0.3, 'window_u': 0.2}

        for param, weight in weights.items():
            sensitivity = self.PARAM_SENSITIVITIES[param]
            bounds = self.PARAM_BOUNDS[param]

            # Calculate required change in this parameter
            param_delta = (delta_kwh_m2 * weight) / sensitivity

            # Apply damping
            param_delta *= damping

            # Apply change
            new_value = current_params[param] + param_delta

            # Clamp to bounds
            new_value = max(bounds[0], min(bounds[1], new_value))

            new_params[param] = new_value

        return new_params

    def _modify_idf(self, idf_content: str, params: Dict[str, float]) -> str:
        """Modify IDF with new parameter values using structured parser."""
        import re

        try:
            if hasattr(self, 'idf_parser') and self.idf_parser:
                idf = self.idf_parser.load_string(idf_content)
                self.idf_parser.apply_calibration_parameters(idf, params)
                return self.idf_parser.to_string(idf)
        except Exception as e:
            logger.warning(f"Structured modification failed: {e}")

        # Regex fallback for compatibility and testing
        modified = idf_content

        # Modify infiltration
        modified = re.sub(
            r'(AirChanges/Hour,)([\s,]*)([\d.]+)(;)',
            lambda m: f"{m.group(1)}{m.group(2)}{params['infiltration']:.4f}{m.group(4)}",
            modified,
            flags=re.IGNORECASE
        )

        # Modify heat recovery effectiveness
        def replace_hr(match):
            prefix = match.group(1)
            return f"{prefix}{params['heat_recovery']:.2f}"

        modified = re.sub(
            r'(Sensible Heat Recovery Effectiveness\s*\n\s*)([\d.]+)',
            replace_hr,
            modified,
            flags=re.IGNORECASE
        )

        # Modify window U-value (format with comment)
        def replace_window_u(match):
            return f"{params['window_u']:.2f},                     !- U-Factor"

        modified = re.sub(
            r'[\d.]+,\s*!-\s*U-Factor[^\n]*',
            replace_window_u,
            modified,
            flags=re.IGNORECASE
        )

        # Modify window U-value (SimpleGlazingSystem format)
        def replace_simple_glazing(match):
            prefix = match.group(1)
            shgc = match.group(2)
            return f"{prefix}{params['window_u']:.2f},{shgc}"

        modified = re.sub(
            r'(WindowMaterial:SimpleGlazingSystem,\s*\w+,\s*)([\d.]+)(,[\d.]+)',
            replace_simple_glazing,
            modified,
            flags=re.IGNORECASE
        )

        return modified

    def _run_simulation(
        self,
        idf_content: str,
        weather_path: Path,
        output_dir: Path
    ) -> Optional[float]:
        """Run simulation and return heating energy (kWh/m²)."""
        output_dir.mkdir(parents=True, exist_ok=True)

        # Write IDF to temp file
        idf_path = output_dir / "model.idf"
        with open(idf_path, 'w') as f:
            f.write(idf_content)

        # Run simulation
        result = self.runner.run(idf_path, weather_path, output_dir)

        if not result.success:
            logger.error(f"Simulation failed: {result.error_message}")
            return None

        # Parse results
        parsed = self.results_parser.parse(output_dir)
        if parsed is None:
            logger.error("Failed to parse results")
            return None

        return parsed.heating_kwh_m2


def calibrate_baseline(
    idf_path: Path,
    weather_path: Path,
    measured_heating_kwh_m2: float,
    output_dir: Path
) -> CalibrationResult:
    """
    Convenience function to calibrate a baseline model.

    Args:
        idf_path: Path to baseline IDF
        weather_path: Path to weather file
        measured_heating_kwh_m2: Target heating intensity
        output_dir: Directory for calibrated output

    Returns:
        CalibrationResult
    """
    calibrator = BaselineCalibrator()
    return calibrator.calibrate(
        idf_path=idf_path,
        weather_path=weather_path,
        measured_heating_kwh_m2=measured_heating_kwh_m2,
        output_dir=output_dir,
    )
