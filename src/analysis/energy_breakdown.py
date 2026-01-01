"""
Energy Breakdown by End-Use.

Swedish energy declarations (energideklarationer) include ALL energy:
- Uppvärmning (space heating)
- Komfortkyla (cooling)
- Tappvarmvatten (domestic hot water)
- Fastighetsel (property electricity: lighting, fans, pumps)

This module tracks energy by end-use so ECM savings are properly attributed.
For example:
- LED lighting saves property electricity, not heating
- DHW temperature reduction saves DHW energy, not heating
- Wall insulation saves heating energy

Typical Swedish multi-family breakdown (kWh/m²/year):
- Heating: 40-80 (varies by age/envelope)
- DHW: 20-25 (relatively constant)
- Property electricity: 12-20 (depends on ventilation type)
- Cooling: 0-5 (often 0 in Sweden)
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, List, Tuple
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class EndUse(Enum):
    """Energy end-use categories (Swedish energideklaration)."""
    HEATING = "heating"           # Uppvärmning
    COOLING = "cooling"           # Komfortkyla
    DHW = "dhw"                   # Tappvarmvatten (domestic hot water)
    PROPERTY_EL = "property_el"   # Fastighetsel (lighting, fans, pumps, elevators)


@dataclass
class EnergyBreakdown:
    """
    Energy breakdown by end-use.

    All values in kWh/m²/year (specific energy).
    """
    heating_kwh_m2: float = 0.0       # Space heating (uppvärmning)
    cooling_kwh_m2: float = 0.0       # Space cooling (komfortkyla)
    dhw_kwh_m2: float = 0.0           # Domestic hot water (tappvarmvatten)
    property_el_kwh_m2: float = 0.0   # Property electricity (fastighetsel)

    # Sub-categories of property electricity (for detailed tracking)
    lighting_kwh_m2: float = 0.0      # Belysning
    ventilation_kwh_m2: float = 0.0   # Fläktar
    pumps_kwh_m2: float = 0.0         # Pumpar
    elevators_kwh_m2: float = 0.0     # Hissar
    other_el_kwh_m2: float = 0.0      # Övrigt

    @property
    def total_kwh_m2(self) -> float:
        """Total energy use (kWh/m²/year)."""
        return (
            self.heating_kwh_m2 +
            self.cooling_kwh_m2 +
            self.dhw_kwh_m2 +
            self.property_el_kwh_m2
        )

    def copy(self) -> "EnergyBreakdown":
        """Create a deep copy."""
        return EnergyBreakdown(
            heating_kwh_m2=self.heating_kwh_m2,
            cooling_kwh_m2=self.cooling_kwh_m2,
            dhw_kwh_m2=self.dhw_kwh_m2,
            property_el_kwh_m2=self.property_el_kwh_m2,
            lighting_kwh_m2=self.lighting_kwh_m2,
            ventilation_kwh_m2=self.ventilation_kwh_m2,
            pumps_kwh_m2=self.pumps_kwh_m2,
            elevators_kwh_m2=self.elevators_kwh_m2,
            other_el_kwh_m2=self.other_el_kwh_m2,
        )

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            "heating_kwh_m2": self.heating_kwh_m2,
            "cooling_kwh_m2": self.cooling_kwh_m2,
            "dhw_kwh_m2": self.dhw_kwh_m2,
            "property_el_kwh_m2": self.property_el_kwh_m2,
            "total_kwh_m2": self.total_kwh_m2,
        }


# =============================================================================
# ECM End-Use Mapping
# =============================================================================

# Which end-use(s) each ECM affects and typical savings percentage
# Format: {ecm_id: {end_use: savings_pct}}
ECM_END_USE_EFFECTS: Dict[str, Dict[str, float]] = {
    # =========================================================================
    # ENVELOPE - Affects heating (and cooling if present)
    # =========================================================================
    "wall_external_insulation": {"heating": 0.15, "cooling": 0.10},
    "wall_internal_insulation": {"heating": 0.10, "cooling": 0.05},
    "roof_insulation": {"heating": 0.08, "cooling": 0.05},
    "basement_insulation": {"heating": 0.05},
    "window_replacement": {"heating": 0.10, "cooling": 0.05},
    "air_sealing": {"heating": 0.12},
    "thermal_bridge_remediation": {"heating": 0.05},
    "facade_renovation": {"heating": 0.20, "cooling": 0.10},  # Comprehensive
    "entrance_door_replacement": {"heating": 0.02},

    # =========================================================================
    # HVAC - Affects heating, sometimes DHW and property_el
    # =========================================================================
    "ftx_upgrade": {"heating": 0.15},
    "ftx_installation": {"heating": 0.30},  # Major impact if no FTX before
    "ftx_overhaul": {"heating": 0.10},
    "demand_controlled_ventilation": {"heating": 0.20, "property_el": 0.15},  # Less fan energy too
    "vrf_system": {"heating": 0.25, "cooling": 0.30},
    "radiator_fans": {"heating": 0.03},
    "heat_pump_integration": {"heating": 0.15},  # Improves heating system
    "exhaust_air_heat_pump": {"heating": 0.25, "dhw": 0.20},  # Often heats DHW too
    "ground_source_heat_pump": {"heating": 0.40, "dhw": 0.30, "cooling": 0.50},
    "air_source_heat_pump": {"heating": 0.30, "dhw": 0.20},
    "heat_pump_water_heater": {"dhw": 0.50},  # Only DHW

    # =========================================================================
    # CONTROLS - Mostly heating, some property_el
    # =========================================================================
    "smart_thermostats": {"heating": 0.05},
    "heating_curve_adjustment": {"heating": 0.05},
    "radiator_balancing": {"heating": 0.05},
    "night_setback": {"heating": 0.03},
    "summer_bypass": {"heating": 0.02},  # Reduces overheating
    "predictive_control": {"heating": 0.08, "cooling": 0.10},
    "fault_detection": {"heating": 0.03, "property_el": 0.02},  # Identifies waste
    "building_automation_system": {"heating": 0.10, "cooling": 0.10, "property_el": 0.08},
    "bms_optimization": {"heating": 0.05, "property_el": 0.03},
    "effektvakt_optimization": {},  # No kWh savings, only SEK (peak demand)

    # =========================================================================
    # LIGHTING - Only property_el
    # =========================================================================
    "led_lighting": {"property_el": 0.08},  # ~50% of lighting load
    "led_common_areas": {"property_el": 0.05},
    "led_outdoor": {"property_el": 0.02},
    "occupancy_sensors": {"property_el": 0.04},
    "daylight_sensors": {"property_el": 0.03},

    # =========================================================================
    # DHW - Only dhw end-use
    # =========================================================================
    "hot_water_temperature": {"dhw": 0.05},  # Lower setpoint
    "dhw_circulation_optimization": {"dhw": 0.08},  # Reduce circulation losses
    "dhw_tank_insulation": {"dhw": 0.03},
    "low_flow_fixtures": {"dhw": 0.10},  # Reduce hot water consumption
    "pipe_insulation": {"dhw": 0.05, "heating": 0.02},  # Reduces distribution losses
    "heat_recovery_dhw": {"dhw": 0.20},  # Drain water heat recovery
    "solar_thermal": {"dhw": 0.40},  # Solar DHW in Sweden ~40%

    # =========================================================================
    # OTHER
    # =========================================================================
    "pump_optimization": {"property_el": 0.02},  # VFD on pumps
    "duc_calibration": {"heating": 0.02},  # District heating substation
    "district_heating_optimization": {"heating": 0.05},
    "recommissioning": {"heating": 0.05, "property_el": 0.03},
    "individual_metering": {"heating": 0.05, "dhw": 0.10},  # Behavioral
    "energy_monitoring": {"heating": 0.02, "property_el": 0.02},  # Awareness

    # =========================================================================
    # RENEWABLES
    # =========================================================================
    "solar_pv": {"property_el": 0.30},  # Offset, not reduction
    "battery_storage": {},  # Peak shifting, no direct savings
}


# =============================================================================
# Sveby/Boverket Default Values
# =============================================================================

# Default DHW consumption (kWh/m²/year) by building type
DHW_DEFAULTS = {
    "multi_family": 22.0,    # Flerbostadshus
    "single_family": 20.0,   # Småhus
    "office": 5.0,           # Kontor
    "school": 8.0,           # Skola
    "retail": 3.0,           # Butik
    "hotel": 40.0,           # Hotell
}

# Default property electricity (kWh/m²/year) by ventilation type
PROPERTY_EL_DEFAULTS = {
    # Multi-family buildings
    "multi_family": {
        "natural": 10.0,      # Självdrag
        "F": 12.0,            # Frånluft only
        "FT": 14.0,           # Från- och tilluft
        "FTX": 18.0,          # FTX (fans use more, but heat recovery saves heating)
    },
    # Commercial buildings have higher property_el
    "office": {
        "natural": 40.0,
        "FTX": 55.0,
    },
}

# Property electricity sub-category breakdown (typical %)
PROPERTY_EL_BREAKDOWN = {
    "multi_family": {
        "ventilation": 0.40,   # Fläktar
        "lighting": 0.25,      # Belysning (common areas)
        "pumps": 0.15,         # Pumpar
        "elevators": 0.10,     # Hissar
        "other": 0.10,         # Övrigt
    },
}


def estimate_baseline_breakdown(
    total_declared_kwh_m2: float,
    simulated_heating_kwh_m2: float,
    building_type: str = "multi_family",
    ventilation_type: str = "FTX",
    has_cooling: bool = False,
    construction_year: int = 2000,
    return_scaling_factor: bool = False,
) -> EnergyBreakdown:
    """
    Estimate energy breakdown when we only have total declared energy.

    Strategy:
    1. Use simulated heating from EnergyPlus
    2. Use default DHW based on building type
    3. Use default property_el based on ventilation
    4. Allocate remaining to cooling or adjust

    Args:
        total_declared_kwh_m2: Total from energy declaration
        simulated_heating_kwh_m2: Heating from EnergyPlus simulation
        building_type: Building category
        ventilation_type: F, FT, FTX, or natural
        has_cooling: Whether building has cooling system
        construction_year: Year built (affects defaults)
        return_scaling_factor: If True, return tuple of (breakdown, scaling_factor)

    Returns:
        EnergyBreakdown with estimated values, or tuple of (EnergyBreakdown, scaling_factor)
        The scaling_factor should be applied to ECM heating results for consistency.
    """
    # Get defaults
    dhw = DHW_DEFAULTS.get(building_type, 22.0)

    property_el_by_vent = PROPERTY_EL_DEFAULTS.get(building_type, PROPERTY_EL_DEFAULTS["multi_family"])
    property_el = property_el_by_vent.get(ventilation_type, 15.0)

    # Adjust for building age (older = less efficient systems)
    if construction_year < 1980:
        property_el *= 1.2  # Older pumps, motors
    elif construction_year > 2010:
        property_el *= 0.85  # Modern efficient equipment

    # Calculate cooling (if any)
    cooling = 0.0
    if has_cooling:
        cooling = 5.0  # Typical Swedish office/retail

    # Use simulated heating
    heating = simulated_heating_kwh_m2

    # Check if our estimates match the declaration
    estimated_total = heating + dhw + property_el + cooling
    gap = total_declared_kwh_m2 - estimated_total

    # Track scaling factor for ECM consistency
    heating_scaling_factor = 1.0

    if abs(gap) > 10:
        # Significant gap - declaration might include things we're missing
        # Or our heating simulation is off
        logger.warning(
            f"Energy breakdown gap: declared={total_declared_kwh_m2:.1f}, "
            f"estimated={estimated_total:.1f}, gap={gap:.1f} kWh/m²"
        )

        if gap > 0:
            # Declaration higher than estimate - allocate gap to most uncertain
            # Typically DHW or property_el varies most
            dhw += gap * 0.4
            property_el += gap * 0.4
            heating += gap * 0.2  # Small adjustment
            # Scaling factor for ECM: heating was increased slightly
            if simulated_heating_kwh_m2 > 0:
                heating_scaling_factor = heating / simulated_heating_kwh_m2
        else:
            # Our estimate is higher - simulation might be off
            # Don't go negative, reduce proportionally
            reduction_factor = total_declared_kwh_m2 / estimated_total
            heating *= reduction_factor
            dhw *= reduction_factor
            property_el *= reduction_factor
            # Scaling factor for ECM: heating was reduced
            heating_scaling_factor = reduction_factor
            logger.info(
                f"Applied heating scaling factor: {heating_scaling_factor:.3f} "
                f"(E+ simulated {simulated_heating_kwh_m2:.1f} → adjusted {heating:.1f} kWh/m²)"
            )

    # Calculate sub-categories of property_el
    breakdown_pcts = PROPERTY_EL_BREAKDOWN.get(building_type, PROPERTY_EL_BREAKDOWN["multi_family"])

    breakdown = EnergyBreakdown(
        heating_kwh_m2=heating,
        cooling_kwh_m2=cooling,
        dhw_kwh_m2=dhw,
        property_el_kwh_m2=property_el,
        lighting_kwh_m2=property_el * breakdown_pcts.get("lighting", 0.25),
        ventilation_kwh_m2=property_el * breakdown_pcts.get("ventilation", 0.40),
        pumps_kwh_m2=property_el * breakdown_pcts.get("pumps", 0.15),
        elevators_kwh_m2=property_el * breakdown_pcts.get("elevators", 0.10),
        other_el_kwh_m2=property_el * breakdown_pcts.get("other", 0.10),
    )

    if return_scaling_factor:
        return breakdown, heating_scaling_factor
    return breakdown


def calculate_ecm_savings(
    ecm_id: str,
    baseline: EnergyBreakdown,
    simulated_heating_result: Optional[float] = None,
) -> Tuple[EnergyBreakdown, Dict[str, float]]:
    """
    Calculate energy savings from an ECM across all end-uses.

    Args:
        ecm_id: ECM identifier
        baseline: Baseline energy breakdown
        simulated_heating_result: If we have E+ simulation result, use it for heating

    Returns:
        Tuple of (result_breakdown, savings_by_end_use)
    """
    result = baseline.copy()
    savings = {}

    # Get ECM effects
    effects = ECM_END_USE_EFFECTS.get(ecm_id, {})

    if not effects:
        logger.debug(f"ECM {ecm_id} has no defined end-use effects")
        return result, savings

    # If we have simulated heating result, use it directly
    if simulated_heating_result is not None and "heating" in effects:
        heating_savings = baseline.heating_kwh_m2 - simulated_heating_result
        savings["heating"] = max(0, heating_savings)
        result.heating_kwh_m2 = simulated_heating_result

        # Apply other effects (cooling, dhw, property_el) from percentages
        for end_use, pct in effects.items():
            if end_use == "heating":
                continue  # Already handled
            if end_use == "cooling":
                saved = baseline.cooling_kwh_m2 * pct
                result.cooling_kwh_m2 -= saved
                savings["cooling"] = saved
            elif end_use == "dhw":
                saved = baseline.dhw_kwh_m2 * pct
                result.dhw_kwh_m2 -= saved
                savings["dhw"] = saved
            elif end_use == "property_el":
                saved = baseline.property_el_kwh_m2 * pct
                result.property_el_kwh_m2 -= saved
                savings["property_el"] = saved
    else:
        # No simulation result - use percentage savings for all
        for end_use, pct in effects.items():
            if end_use == "heating":
                saved = baseline.heating_kwh_m2 * pct
                result.heating_kwh_m2 -= saved
                savings["heating"] = saved
            elif end_use == "cooling":
                saved = baseline.cooling_kwh_m2 * pct
                result.cooling_kwh_m2 -= saved
                savings["cooling"] = saved
            elif end_use == "dhw":
                saved = baseline.dhw_kwh_m2 * pct
                result.dhw_kwh_m2 -= saved
                savings["dhw"] = saved
            elif end_use == "property_el":
                saved = baseline.property_el_kwh_m2 * pct
                result.property_el_kwh_m2 -= saved
                savings["property_el"] = saved

    return result, savings


def format_breakdown_for_report(breakdown: EnergyBreakdown) -> Dict[str, str]:
    """Format breakdown for HTML report display."""
    return {
        "heating": f"{breakdown.heating_kwh_m2:.1f} kWh/m²",
        "cooling": f"{breakdown.cooling_kwh_m2:.1f} kWh/m²",
        "dhw": f"{breakdown.dhw_kwh_m2:.1f} kWh/m²",
        "property_el": f"{breakdown.property_el_kwh_m2:.1f} kWh/m²",
        "total": f"{breakdown.total_kwh_m2:.1f} kWh/m²",
    }
