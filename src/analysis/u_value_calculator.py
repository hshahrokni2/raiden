"""
U-value back-calculation from energy consumption.

Uses simplified steady-state heat loss calculations to estimate
building envelope U-values from actual energy consumption data.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

# Swedish climate data (Stockholm region)
HEATING_DEGREE_DAYS = 3900  # °C-days/year for Stockholm
HEATING_HOURS = 5500  # Hours per heating season
AVG_DELTA_T = 17  # Average temperature difference during heating (°C)
SEASONAL_EFFICIENCY = 0.9  # Heat pump/district heating efficiency

# Default assumptions
DEFAULT_WWR = 0.20  # 20% windows
DEFAULT_INFILTRATION_W_PER_SQM_K = 0.15  # W/m²K equivalent
HOT_WATER_KWH_PER_SQM = 25  # kWh/m²/year for hot water
PROPERTY_ELECTRICITY_KWH_PER_SQM = 10  # kWh/m²/year for common areas


@dataclass
class UValueEstimate:
    """Estimated U-values from back-calculation."""

    walls: float  # W/m²K
    roof: float
    floor: float
    windows: float

    # Calculation metadata
    calculation_method: str = "back_calculation"
    confidence: float = 0.0  # 0-1
    notes: list[str] | None = None


@dataclass
class BuildingEnvelope:
    """Building envelope areas for heat loss calculation."""

    heated_area_sqm: float  # Total heated floor area (Atemp)
    wall_area_sqm: float
    roof_area_sqm: float
    floor_area_sqm: float  # Ground floor area
    window_area_sqm: float

    # Optional refinements
    wall_area_by_orientation: dict[str, float] | None = None  # N, S, E, W
    perimeter_m: Optional[float] = None


def calculate_envelope_areas(
    heated_area_sqm: float,
    num_floors: int,
    height_m: float,
    wwr: float = DEFAULT_WWR,
    footprint_perimeter_m: Optional[float] = None,
) -> BuildingEnvelope:
    """
    Calculate building envelope areas from basic dimensions.

    Args:
        heated_area_sqm: Total heated floor area (Atemp)
        num_floors: Number of floors
        height_m: Total building height
        wwr: Window-to-wall ratio (0-1)
        footprint_perimeter_m: Building perimeter (estimated if not provided)

    Returns:
        BuildingEnvelope with calculated areas
    """
    # Estimate footprint from total area and floors
    floor_area_sqm = heated_area_sqm / num_floors

    # Estimate perimeter (assume roughly square footprint)
    if footprint_perimeter_m is None:
        import math
        side = math.sqrt(floor_area_sqm)
        footprint_perimeter_m = 4 * side

    # Gross wall area = perimeter × height
    gross_wall_area = footprint_perimeter_m * height_m
    window_area = gross_wall_area * wwr
    net_wall_area = gross_wall_area * (1 - wwr)

    return BuildingEnvelope(
        heated_area_sqm=heated_area_sqm,
        wall_area_sqm=net_wall_area,
        roof_area_sqm=floor_area_sqm,  # Top floor footprint
        floor_area_sqm=floor_area_sqm,  # Ground floor footprint
        window_area_sqm=window_area,
        perimeter_m=footprint_perimeter_m,
    )


def calculate_heat_loss(
    envelope: BuildingEnvelope,
    u_walls: float,
    u_roof: float,
    u_floor: float,
    u_windows: float,
    infiltration_w_sqm_k: float = DEFAULT_INFILTRATION_W_PER_SQM_K,
) -> float:
    """
    Calculate annual heat loss through building envelope.

    Uses simplified steady-state calculation:
    Q = ΣU×A × ΔT × hours / 1000 (kWh)

    Args:
        envelope: Building envelope areas
        u_*: U-values for each component (W/m²K)
        infiltration_w_sqm_k: Equivalent infiltration heat loss

    Returns:
        Annual heat loss in kWh
    """
    # Heat loss coefficients (W/K)
    h_walls = envelope.wall_area_sqm * u_walls
    h_roof = envelope.roof_area_sqm * u_roof
    h_floor = envelope.floor_area_sqm * u_floor * 0.7  # Ground coupled = 70%
    h_windows = envelope.window_area_sqm * u_windows
    h_infiltration = envelope.heated_area_sqm * infiltration_w_sqm_k

    total_h = h_walls + h_roof + h_floor + h_windows + h_infiltration

    # Annual heat loss (kWh)
    # Using degree-days: Q = H × DD × 24 / 1000
    annual_loss_kwh = total_h * HEATING_DEGREE_DAYS * 24 / 1000

    return annual_loss_kwh


def back_calculate_u_values(
    envelope: BuildingEnvelope,
    actual_heating_kwh: float,
    era_estimates: UValueEstimate,
    solar_gain_kwh: float = 0,
    internal_gain_kwh: float = 0,
) -> UValueEstimate:
    """
    Back-calculate U-values from actual energy consumption.

    Adjusts era-based estimates to match actual consumption.

    Args:
        envelope: Building envelope areas
        actual_heating_kwh: Actual annual heating energy consumption
        era_estimates: Initial U-value estimates from construction era
        solar_gain_kwh: Estimated solar heat gains (reduces heating need)
        internal_gain_kwh: Estimated internal gains (reduces heating need)

    Returns:
        UValueEstimate with back-calculated values
    """
    # Calculate theoretical loss with era estimates
    theoretical_loss = calculate_heat_loss(
        envelope,
        era_estimates.walls,
        era_estimates.roof,
        era_estimates.floor,
        era_estimates.windows,
    )

    # Net heating demand (accounting for gains and system efficiency)
    net_heating_demand = actual_heating_kwh * SEASONAL_EFFICIENCY - solar_gain_kwh - internal_gain_kwh

    # If theoretical matches actual within 20%, keep estimates
    ratio = net_heating_demand / theoretical_loss if theoretical_loss > 0 else 1

    notes = []
    if 0.8 <= ratio <= 1.2:
        notes.append("Era estimates match actual consumption well")
        confidence = 0.8
    elif ratio < 0.8:
        notes.append(f"Building performs {(1-ratio)*100:.0f}% better than era typical")
        confidence = 0.6
    else:
        notes.append(f"Building performs {(ratio-1)*100:.0f}% worse than era typical")
        confidence = 0.5

    # Scale U-values proportionally
    # This is simplified - in reality different components have different impacts
    scale = ratio ** 0.5  # Dampen the adjustment

    # Clamp to reasonable ranges
    def clamp_u(value: float, min_val: float, max_val: float) -> float:
        return max(min_val, min(max_val, value))

    return UValueEstimate(
        walls=clamp_u(era_estimates.walls * scale, 0.10, 1.0),
        roof=clamp_u(era_estimates.roof * scale, 0.08, 0.5),
        floor=clamp_u(era_estimates.floor * scale, 0.10, 0.5),
        windows=clamp_u(era_estimates.windows * scale, 0.8, 3.0),
        calculation_method="back_calculation_from_consumption",
        confidence=confidence,
        notes=notes,
    )


def estimate_from_specific_energy(
    specific_energy_kwh_sqm: float,
    heated_area_sqm: float,
    num_floors: int,
    height_m: float,
    construction_year: int,
    renovation_year: int | None = None,
    wwr: float = DEFAULT_WWR,
    perimeter_m: float | None = None,
) -> UValueEstimate:
    """
    Main entry point: estimate U-values from specific energy use.

    Args:
        specific_energy_kwh_sqm: Actual energy use per m² per year
        heated_area_sqm: Total heated area (Atemp)
        num_floors: Number of stories
        height_m: Building height
        construction_year: Year built
        renovation_year: Year of energy renovation (if any)
        wwr: Window-to-wall ratio (default 0.20)
        perimeter_m: Building perimeter (estimated if not provided)

    Returns:
        UValueEstimate with calculated values and confidence
    """
    # Build envelope model
    envelope = calculate_envelope_areas(
        heated_area_sqm=heated_area_sqm,
        num_floors=num_floors,
        height_m=height_m,
        wwr=wwr,
        footprint_perimeter_m=perimeter_m,
    )

    # Get era-based initial estimates
    era = get_era_estimates(construction_year, renovation_year)

    # Estimate heating consumption (subtract hot water and electricity)
    total_kwh = specific_energy_kwh_sqm * heated_area_sqm
    heating_kwh = total_kwh - (HOT_WATER_KWH_PER_SQM * heated_area_sqm)

    # Estimate internal gains (roughly 5 kWh/m²/year from occupants + equipment)
    internal_gains = 5 * heated_area_sqm

    # Back-calculate
    result = back_calculate_u_values(
        envelope=envelope,
        actual_heating_kwh=max(0, heating_kwh),
        era_estimates=era,
        internal_gain_kwh=internal_gains,
    )

    result.notes = result.notes or []
    result.notes.append(f"Based on {specific_energy_kwh_sqm:.0f} kWh/m²/year specific energy")
    result.notes.append(f"Reference year: {renovation_year or construction_year}")

    return result


def get_era_estimates(construction_year: int, renovation_year: int | None = None) -> UValueEstimate:
    """Get U-value estimates based on Swedish BBR for construction era."""
    ref_year = renovation_year if renovation_year else construction_year

    if ref_year >= 2020:
        return UValueEstimate(walls=0.18, roof=0.13, floor=0.15, windows=1.2, confidence=0.7)
    elif ref_year >= 2010:
        return UValueEstimate(walls=0.20, roof=0.13, floor=0.15, windows=1.3, confidence=0.7)
    elif ref_year >= 2000:
        return UValueEstimate(walls=0.25, roof=0.15, floor=0.20, windows=1.6, confidence=0.6)
    elif ref_year >= 1990:
        return UValueEstimate(walls=0.30, roof=0.20, floor=0.25, windows=2.0, confidence=0.5)
    elif ref_year >= 1975:
        return UValueEstimate(walls=0.40, roof=0.25, floor=0.30, windows=2.5, confidence=0.5)
    elif ref_year >= 1960:
        return UValueEstimate(walls=0.50, roof=0.35, floor=0.40, windows=2.8, confidence=0.4)
    else:
        return UValueEstimate(walls=0.80, roof=0.50, floor=0.50, windows=3.0, confidence=0.3)
