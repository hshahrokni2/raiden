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
        # 1. Calculate usable area after setbacks
        if roof_type == 'flat':
            utilization = self.FLAT_ROOF_UTILIZATION
        else:
            utilization = self.PITCHED_ROOF_UTILIZATION

        available_area = roof_area_m2 * utilization

        # 2. Determine optimal tilt for this latitude
        # In Sweden, optimal is approximately latitude - 10° to 15°
        optimal_tilt = max(10.0, self.latitude - 12.0)
        optimal_azimuth = 180.0  # South-facing

        # 3. Calculate panel configuration
        # For flat roofs with tilted panels, panels need spacing to avoid shading
        if roof_type == 'flat' and roof_slope_deg < 5:
            # Panels will be tilted to optimal angle
            actual_tilt = optimal_tilt
            actual_azimuth = optimal_azimuth
        else:
            # Use roof's existing slope and orientation
            actual_tilt = roof_slope_deg
            actual_azimuth = roof_azimuth_deg

        # 4. Calculate capacity
        # Standard panel: ~400W, ~1.7m², ~0.20 efficiency
        panel_area_per_kwp = 1.0 / self.DEFAULT_PANEL_EFFICIENCY  # m² per kWp
        max_capacity = available_area / panel_area_per_kwp

        # 5. Calculate yield based on orientation
        orientation_factor = self._orientation_factor(actual_tilt, actual_azimuth)

        # Base yield from irradiance (kWh/m²/year → kWh/kWp/year)
        # At optimal tilt, yield improves by ~10-15% over horizontal
        tilt_gain = 1.12  # 12% gain at optimal tilt vs horizontal
        base_yield_kwh_per_kwp = self.base_irradiance * tilt_gain

        # Apply orientation factor
        yield_before_shading = base_yield_kwh_per_kwp * orientation_factor

        # 6. Calculate shading losses
        shading_loss = self._calculate_shading_loss(shading_objects or [])

        # 7. Apply system losses
        inverter_loss = 0.04  # 4% inverter losses
        soiling_loss = 0.02  # 2% soiling/dust losses
        other_losses = 0.03  # Wiring, mismatch, degradation

        total_system_loss = 1 - (1 - inverter_loss) * (1 - soiling_loss) * (1 - other_losses)

        # Final effective yield
        effective_yield = yield_before_shading * (1 - shading_loss) * (1 - total_system_loss)
        total_annual_yield = effective_yield * max_capacity

        return PVPotential(
            available_roof_area_m2=available_area,
            max_capacity_kwp=max_capacity,
            optimal_tilt_deg=optimal_tilt,
            optimal_azimuth_deg=optimal_azimuth,
            annual_yield_kwh_per_kwp=yield_before_shading,
            shading_loss_factor=shading_loss,
            effective_annual_yield_kwh=total_annual_yield,
            roof_utilization_factor=utilization,
            inverter_losses=inverter_loss,
            soiling_losses=soiling_loss,
        )

    def _interpolate_irradiance(self, latitude: float) -> float:
        """Interpolate solar irradiance for latitude."""
        # Sort latitudes for interpolation
        lats = sorted(self.SWEDISH_IRRADIANCE.keys())

        # Handle edge cases
        if latitude <= lats[0]:
            return self.SWEDISH_IRRADIANCE[lats[0]]
        if latitude >= lats[-1]:
            return self.SWEDISH_IRRADIANCE[lats[-1]]

        # Find bracketing latitudes
        for i in range(len(lats) - 1):
            if lats[i] <= latitude <= lats[i + 1]:
                lat_low, lat_high = lats[i], lats[i + 1]
                irr_low = self.SWEDISH_IRRADIANCE[lat_low]
                irr_high = self.SWEDISH_IRRADIANCE[lat_high]

                # Linear interpolation
                fraction = (latitude - lat_low) / (lat_high - lat_low)
                return irr_low + fraction * (irr_high - irr_low)

        # Fallback to Stockholm
        return self.SWEDISH_IRRADIANCE[59.3]

    def _orientation_factor(self, tilt: float, azimuth: float) -> float:
        """
        Calculate yield factor for non-optimal orientation.

        Optimal in Sweden: ~40° tilt, 180° azimuth (south)

        Returns factor 0-1 where 1 is optimal orientation.
        """
        import math

        # Optimal values for Swedish latitudes
        optimal_tilt = self.latitude - 12.0  # Approximately 47° for Stockholm
        optimal_azimuth = 180.0  # Due south

        # Tilt factor - yield drops as tilt deviates from optimal
        # At 0° (horizontal): ~12% less than optimal
        # At 90° (vertical): ~60% less than optimal
        tilt_diff = abs(tilt - optimal_tilt)
        # Use cosine-based reduction (simplified model)
        tilt_factor = 1.0 - 0.005 * tilt_diff - 0.0002 * tilt_diff ** 2
        tilt_factor = max(0.4, min(1.0, tilt_factor))

        # Azimuth factor - yield drops as orientation deviates from south
        # SE/SW (135°/225°): ~5% less
        # E/W (90°/270°): ~20% less
        # NE/NW (45°/315°): ~40% less
        # N (0°/360°): ~60% less
        azimuth_diff = abs(azimuth - optimal_azimuth)
        if azimuth_diff > 180:
            azimuth_diff = 360 - azimuth_diff

        # Cosine-based model for azimuth
        azimuth_factor = math.cos(math.radians(azimuth_diff * 0.8))
        azimuth_factor = max(0.4, min(1.0, azimuth_factor))

        # Combined factor (multiplicative)
        return tilt_factor * azimuth_factor

    def _calculate_shading_loss(self, shading_objects: List[dict]) -> float:
        """
        Calculate shading loss factor from obstructions.

        Args:
            shading_objects: List of dicts with shading info:
                - type: 'building', 'tree', 'chimney', etc.
                - height_m: Height of obstruction above roof
                - distance_m: Distance from roof edge
                - width_m: Width of obstruction
                - azimuth_deg: Direction of obstruction from roof center

        Returns:
            Shading loss factor (0-1), where 0 = no shading, 1 = full shade
        """
        if not shading_objects:
            return 0.0

        import math

        total_shading = 0.0

        for obj in shading_objects:
            obj_type = obj.get('type', 'building')
            height = obj.get('height_m', 0)
            distance = obj.get('distance_m', 10)
            width = obj.get('width_m', 10)
            azimuth = obj.get('azimuth_deg', 180)

            if distance <= 0 or height <= 0:
                continue

            # Calculate shade angle
            shade_angle = math.degrees(math.atan(height / distance))

            # Objects to the south (150-210°) have most impact
            azimuth_diff = abs(azimuth - 180)
            if azimuth_diff > 180:
                azimuth_diff = 360 - azimuth_diff

            # Impact factor based on azimuth (south-facing objects block more sun)
            azimuth_impact = math.cos(math.radians(azimuth_diff))
            azimuth_impact = max(0, azimuth_impact)

            # Shade impact based on angle and width
            # High objects close by = more shading
            # Wider objects = more shading
            angle_factor = min(1.0, shade_angle / 45.0)  # 45° angle = significant shading
            width_factor = min(1.0, width / 20.0)  # 20m wide = full width impact

            obj_shading = angle_factor * width_factor * azimuth_impact

            # Tree-specific: seasonal variation (less leaves in winter when sun is lower)
            if obj_type == 'tree':
                obj_shading *= 0.7  # Trees ~30% less impactful on average

            total_shading += obj_shading

        # Cap total shading at 0.5 (50% max realistic shading loss)
        return min(0.5, total_shading)


def calculate_pv_potential(
    roof_area_m2: float,
    latitude: float = 59.3,
    roof_type: str = 'flat',
    roof_slope_deg: float = 0.0,
    roof_azimuth_deg: float = 180.0,
    shading_objects: Optional[List[dict]] = None
) -> PVPotential:
    """
    Convenience function to calculate PV potential.

    Args:
        roof_area_m2: Total roof area
        latitude: Site latitude (default Stockholm)
        roof_type: 'flat' or 'pitched'
        roof_slope_deg: Roof slope (0 for flat)
        roof_azimuth_deg: Roof orientation (180 = south)
        shading_objects: List of shading obstructions

    Returns:
        PVPotential assessment

    Example:
        potential = calculate_pv_potential(
            roof_area_m2=320,
            latitude=59.3,
            roof_type='flat'
        )
        print(f"Capacity: {potential.max_capacity_kwp:.1f} kWp")
        print(f"Annual yield: {potential.effective_annual_yield_kwh:.0f} kWh")
    """
    calculator = PVPotentialCalculator(latitude=latitude)
    return calculator.calculate(
        roof_area_m2=roof_area_m2,
        roof_type=roof_type,
        roof_slope_deg=roof_slope_deg,
        roof_azimuth_deg=roof_azimuth_deg,
        shading_objects=shading_objects,
    )
