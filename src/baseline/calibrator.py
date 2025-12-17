"""
Baseline Calibrator - Calibrate model to energy declaration.

Takes:
- Generated baseline model
- Actual energy consumption (from energy declaration)

Adjusts within plausible ranges:
- Infiltration rate
- Heat recovery efficiency
- Internal gains

Until simulated ≈ measured (within ±10%)
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional
from pathlib import Path


@dataclass
class CalibrationParameter:
    """A parameter that can be adjusted during calibration."""
    name: str
    current_value: float
    min_value: float
    max_value: float
    sensitivity: float  # kWh/m² change per unit change


@dataclass
class CalibrationResult:
    """Result of calibration process."""
    success: bool
    iterations: int
    final_error_percent: float

    # Adjusted parameters
    adjusted_infiltration_ach: float
    adjusted_heat_recovery: float
    adjusted_internal_gains_factor: float

    # Energy values
    measured_kwh_m2: float
    simulated_kwh_m2: float
    calibrated_kwh_m2: float


class BaselineCalibrator:
    """
    Calibrate baseline model to actual energy consumption.

    Uses iterative adjustment within physically plausible bounds:
    - Won't make building impossibly airtight
    - Won't exceed manufacturer HR specs
    - Won't assume unrealistic occupancy

    Usage:
        calibrator = BaselineCalibrator()
        result = calibrator.calibrate(
            idf_path=Path('./baseline.idf'),
            measured_heating_kwh_m2=95.0,
            floor_area_m2=2240
        )
    """

    # Calibration bounds (what's physically plausible)
    INFILTRATION_BOUNDS = (0.02, 0.40)  # ACH
    HEAT_RECOVERY_BOUNDS = (0.60, 0.90)  # Effectiveness
    INTERNAL_GAINS_FACTOR_BOUNDS = (0.8, 1.3)  # Multiplier on Sveby defaults

    # Convergence criteria
    MAX_ITERATIONS = 20
    CONVERGENCE_THRESHOLD = 0.10  # 10% error acceptable

    def __init__(self, energyplus_path: Optional[str] = None):
        """
        Initialize calibrator.

        Args:
            energyplus_path: Path to EnergyPlus executable (auto-detect if None)
        """
        self.energyplus_path = energyplus_path or self._find_energyplus()

    def calibrate(
        self,
        idf_path: Path,
        weather_path: Path,
        measured_heating_kwh_m2: float,
        floor_area_m2: float
    ) -> CalibrationResult:
        """
        Calibrate model to measured energy consumption.

        Args:
            idf_path: Path to baseline IDF
            weather_path: Path to weather file
            measured_heating_kwh_m2: Actual energy use from declaration
            floor_area_m2: Building floor area (Atemp)

        Returns:
            CalibrationResult with adjusted parameters
        """
        # TODO: Implement
        # 1. Run initial simulation
        # 2. Compare to measured
        # 3. Estimate sensitivities (partial derivatives)
        # 4. Adjust parameters using gradient descent
        # 5. Re-run and iterate until convergence
        # 6. Return calibrated model
        raise NotImplementedError("Implement calibration")

    def _run_simulation(self, idf_path: Path, weather_path: Path) -> float:
        """Run EnergyPlus and return heating energy (kWh)."""
        raise NotImplementedError()

    def _modify_idf_parameter(
        self,
        idf_path: Path,
        parameter: str,
        value: float
    ) -> Path:
        """Modify a parameter in the IDF and return new path."""
        raise NotImplementedError()

    def _estimate_sensitivities(
        self,
        idf_path: Path,
        weather_path: Path,
        parameters: List[CalibrationParameter]
    ) -> List[float]:
        """
        Estimate sensitivity of heating energy to each parameter.

        Uses finite differences: dE/dp ≈ (E(p+Δp) - E(p)) / Δp
        """
        raise NotImplementedError()

    def _find_energyplus(self) -> str:
        """Auto-detect EnergyPlus installation."""
        import shutil
        # Common locations
        candidates = [
            '/usr/local/EnergyPlus-25-1-0/energyplus',
            '/Applications/EnergyPlus-25-1-0/energyplus',
            'energyplus',  # In PATH
        ]
        for path in candidates:
            if shutil.which(path):
                return path
        raise RuntimeError("EnergyPlus not found. Install or specify path.")
