"""
PV Potential Calculator

Calculates solar PV potential from roof geometry and location:
- Available roof area after setbacks and obstructions
- Optimal panel orientation and tilt
- Annual generation estimate (kWh/kWp)
- Shading losses from neighbors and trees

Uses Swedish solar irradiance data and typical system parameters.
"""

from dataclasses import dataclass
from typing import Optional, List, Tuple


@dataclass
class PVSystemSpec:
    """PV system specification."""
    capacity_kwp: float
    panel_area_m2: float
    panel_efficiency: float  # 0.18-0.22 typical
    tilt_deg: float
    azimuth_deg: float  # 180 = south
    annual_yield_kwh_per_kwp: float  # Location-specific


@dataclass
class PVPotential:
    """PV potential assessment results."""
    available_roof_area_m2: float
    max_capacity_kwp: float
    optimal_tilt_deg: float
    optimal_azimuth_deg: float
    annual_yield_kwh_per_kwp: float  # Before shading
    shading_loss_factor: float  # 0-1, multiply yield by (1 - this)
    effective_annual_yield_kwh: float  # After shading

    # Breakdown
    roof_utilization_factor: float  # % of roof usable
    inverter_losses: float  # Typical 3-5%
    soiling_losses: float  # Typical 2-3%


class PVPotentialCalculator:
    """
    Calculate PV potential for a building.

    Usage:
        calculator = PVPotentialCalculator(latitude=59.3)
        potential = calculator.calculate(
            roof_area_m2=320,
            roof_type='flat',
            roof_slope_deg=0,
            roof_azimuth_deg=180,
            shading_objects=[...]
        )
    """

    # Swedish solar irradiance by latitude (kWh/m²/year on horizontal)
    SWEDISH_IRRADIANCE = {
        55.0: 1050,  # Malmö
        57.7: 1000,  # Gothenburg
        59.3: 950,   # Stockholm
        63.8: 900,   # Sundsvall
        67.8: 850,   # Kiruna
    }

    # Typical system parameters
    DEFAULT_PANEL_EFFICIENCY = 0.20
    DEFAULT_SYSTEM_LOSSES = 0.14  # Inverter, wiring, soiling
    ROOF_SETBACK_M = 1.0  # Required setback from edges
    FLAT_ROOF_UTILIZATION = 0.70  # Account for spacing, access
    PITCHED_ROOF_UTILIZATION = 0.85

    def __init__(self, latitude: float = 59.3):
        """
        Initialize calculator for a specific latitude.

        Args:
            latitude: Site latitude (default Stockholm)
        """
        self.latitude = latitude
        self.base_irradiance = self._interpolate_irradiance(latitude)

    def calculate(
        self,
        roof_area_m2: float,
        roof_type: str = 'flat',
        roof_slope_deg: float = 0.0,
        roof_azimuth_deg: float = 180.0,
        shading_objects: Optional[List[dict]] = None
    ) -> PVPotential:
        """
        Calculate PV potential for roof.

        Args:
            roof_area_m2: Total roof area
            roof_type: 'flat' or 'pitched'
            roof_slope_deg: Roof slope (0 for flat)
            roof_azimuth_deg: Roof orientation (180 = south)
            shading_objects: List of shading obstructions

        Returns:
            PVPotential with capacity and yield estimates
        """
        # TODO: Implement
        # 1. Calculate usable area after setbacks
        # 2. Determine optimal tilt (latitude - 10-15° typically)
        # 3. Calculate yield based on orientation
        # 4. Apply shading losses
        # 5. Apply system losses
        raise NotImplementedError("Implement PV calculation")

    def _interpolate_irradiance(self, latitude: float) -> float:
        """Interpolate solar irradiance for latitude."""
        raise NotImplementedError()

    def _orientation_factor(self, tilt: float, azimuth: float) -> float:
        """
        Calculate yield factor for non-optimal orientation.

        Optimal in Sweden: ~40° tilt, 180° azimuth (south)
        """
        raise NotImplementedError()

    def _calculate_shading_loss(self, shading_objects: List[dict]) -> float:
        """Calculate shading loss factor from obstructions."""
        raise NotImplementedError()
