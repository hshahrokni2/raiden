"""
Automatic HVAC System Selection from Building Data.

Uses real data from:
1. Sweden Buildings GeoJSON (37,489 Stockholm buildings with heating data)
2. Gripen energy declarations (830,610 buildings nationwide)
3. Archetype defaults (fallback)

Priority: Real data > Inferred from era/type
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any
import logging

from .swedish_systems import (
    SwedishHVACSystem,
    VentilationSystem,
    HVACSelection,
    get_hvac_defaults_for_era,
)

logger = logging.getLogger(__name__)

# Import building data types (optional, may not be available)
try:
    from ..ingest.sweden_buildings import SwedishBuilding
except ImportError:
    SwedishBuilding = None

try:
    from ..ingest.gripen_loader import GripenBuilding
except ImportError:
    GripenBuilding = None

try:
    from ..baseline.archetypes import SwedishArchetype, HeatingSystem, VentilationType
except ImportError:
    SwedishArchetype = None
    HeatingSystem = None
    VentilationType = None


def select_hvac_system(
    building: Optional[Any] = None,  # SwedishBuilding from GeoJSON
    gripen: Optional[Any] = None,    # GripenBuilding from energy declaration
    archetype: Optional[Any] = None,  # SwedishArchetype
    construction_year: Optional[int] = None,
    atemp_m2: Optional[float] = None,
) -> HVACSelection:
    """
    Select appropriate HVAC system from available data.

    Priority:
    1. Sweden Buildings GeoJSON (actual heating data - highest confidence)
    2. Gripen energy declaration (actual heating data - high confidence)
    3. Archetype defaults (inferred from era - lower confidence)

    Args:
        building: SwedishBuilding from Stockholm GeoJSON
        gripen: GripenBuilding from Gripen database
        archetype: Matched SwedishArchetype
        construction_year: Building construction year (fallback)
        atemp_m2: Heated floor area (for sizing hints)

    Returns:
        HVACSelection with heating system, ventilation, and parameters
    """
    # 1. Try Sweden Buildings GeoJSON first (best source for Stockholm)
    if building is not None:
        selection = detect_heating_from_sweden_building(building)
        if selection is not None:
            logger.info(
                f"HVAC detected from Sweden GeoJSON: {selection.primary_heating.value}, "
                f"confidence: {selection.confidence:.0%}"
            )
            return selection

    # 2. Try Gripen energy declaration (nationwide coverage)
    if gripen is not None:
        selection = detect_heating_from_gripen(gripen)
        if selection is not None:
            logger.info(
                f"HVAC detected from Gripen: {selection.primary_heating.value}, "
                f"confidence: {selection.confidence:.0%}"
            )
            return selection

    # 3. Use archetype defaults
    if archetype is not None:
        selection = hvac_from_archetype(archetype)
        logger.info(
            f"HVAC from archetype: {selection.primary_heating.value}, "
            f"confidence: {selection.confidence:.0%}"
        )
        return selection

    # 4. Last resort: infer from construction year
    if construction_year is not None:
        selection = hvac_from_era(construction_year)
        logger.info(
            f"HVAC inferred from era ({construction_year}): {selection.primary_heating.value}, "
            f"confidence: {selection.confidence:.0%}"
        )
        return selection

    # Default: District heating (most common in Swedish MFH)
    logger.warning("No building data available, defaulting to district heating")
    return HVACSelection(
        primary_heating=SwedishHVACSystem.DISTRICT_HEATING,
        ventilation=VentilationSystem.F_SYSTEM,
        heat_recovery_eff=0.0,
        detected_from="default",
        confidence=0.5,
    )


def detect_heating_from_sweden_building(building) -> Optional[HVACSelection]:
    """
    Detect heating system from Sweden Buildings GeoJSON.

    The GeoJSON contains actual energy consumption by source:
    - district_heating_kwh
    - ground_source_hp_kwh
    - exhaust_air_hp_kwh
    - electric_heating_kwh
    - oil_kwh, gas_kwh, pellets_kwh

    Returns the dominant heating source.
    """
    if building is None:
        return None

    # Get energy consumption by source (all in kWh/year)
    dh = getattr(building, 'district_heating_kwh', 0) or 0
    gshp = getattr(building, 'ground_source_hp_kwh', 0) or 0
    exhaust_hp = getattr(building, 'exhaust_air_hp_kwh', 0) or 0
    electric = getattr(building, 'electric_heating_kwh', 0) or 0
    oil = getattr(building, 'oil_kwh', 0) or 0
    gas = getattr(building, 'gas_kwh', 0) or 0
    pellets = getattr(building, 'pellets_kwh', 0) or 0

    total = dh + gshp + exhaust_hp + electric + oil + gas + pellets

    if total == 0:
        # No heating data, check ventilation type
        vent_type = getattr(building, 'ventilation_type', None)
        if vent_type:
            return HVACSelection(
                primary_heating=SwedishHVACSystem.DISTRICT_HEATING,  # Assume DH
                ventilation=_map_ventilation_type(vent_type),
                heat_recovery_eff=_get_hr_eff(vent_type),
                detected_from="sweden_geojson_partial",
                confidence=0.70,
            )
        return None

    # Determine dominant heating source
    sources = {
        SwedishHVACSystem.DISTRICT_HEATING: dh,
        SwedishHVACSystem.GROUND_SOURCE_HP: gshp,
        SwedishHVACSystem.EXHAUST_AIR_HP: exhaust_hp,
        SwedishHVACSystem.DIRECT_ELECTRIC: electric,
        SwedishHVACSystem.OIL_BOILER: oil,
        SwedishHVACSystem.PELLET_BOILER: pellets,
    }

    # Find dominant source (>50% of total)
    for system, kwh in sorted(sources.items(), key=lambda x: x[1], reverse=True):
        if kwh / total > 0.5:
            # Get ventilation type
            vent_type = getattr(building, 'ventilation_type', None)
            ventilation = _map_ventilation_type(vent_type) if vent_type else VentilationSystem.F_SYSTEM
            hr_eff = _get_hr_eff(vent_type) if vent_type else 0.0

            # Calculate COP for heat pumps
            cop = _estimate_cop(system, building)

            return HVACSelection(
                primary_heating=system,
                ventilation=ventilation,
                heat_recovery_eff=hr_eff,
                hp_cop=cop,
                detected_from="sweden_geojson",
                confidence=0.90,
            )

    # Mixed system - take largest contributor
    dominant_system = max(sources.items(), key=lambda x: x[1])[0]
    vent_type = getattr(building, 'ventilation_type', None)

    return HVACSelection(
        primary_heating=dominant_system,
        ventilation=_map_ventilation_type(vent_type) if vent_type else VentilationSystem.F_SYSTEM,
        heat_recovery_eff=_get_hr_eff(vent_type) if vent_type else 0.0,
        detected_from="sweden_geojson_mixed",
        confidence=0.80,
    )


def detect_heating_from_gripen(gripen) -> Optional[HVACSelection]:
    """
    Detect heating system from Gripen energy declaration.

    Gripen contains:
    - Primary heating type indicator
    - Ventilation system type
    - Specific energy consumption by source
    """
    if gripen is None:
        return None

    # Get primary heating using the helper method
    primary_heating_str = None
    if hasattr(gripen, 'get_primary_heating'):
        primary_heating_str = gripen.get_primary_heating()

    # Map string to enum
    heating_map = {
        'district_heating': SwedishHVACSystem.DISTRICT_HEATING,
        'fjärrvärme': SwedishHVACSystem.DISTRICT_HEATING,
        'ground_source_hp': SwedishHVACSystem.GROUND_SOURCE_HP,
        'bergvärme': SwedishHVACSystem.GROUND_SOURCE_HP,
        'exhaust_air_hp': SwedishHVACSystem.EXHAUST_AIR_HP,
        'frånluftsvärmepump': SwedishHVACSystem.EXHAUST_AIR_HP,
        'electric': SwedishHVACSystem.DIRECT_ELECTRIC,
        'direktel': SwedishHVACSystem.DIRECT_ELECTRIC,
        'el': SwedishHVACSystem.DIRECT_ELECTRIC,
        'oil': SwedishHVACSystem.OIL_BOILER,
        'olja': SwedishHVACSystem.OIL_BOILER,
        'gas': SwedishHVACSystem.DISTRICT_HEATING,  # Natural gas uncommon in Sweden
        'pellet': SwedishHVACSystem.PELLET_BOILER,
        'pellets': SwedishHVACSystem.PELLET_BOILER,
    }

    primary_heating = heating_map.get(
        primary_heating_str.lower() if primary_heating_str else '',
        SwedishHVACSystem.DISTRICT_HEATING
    )

    # Detect ventilation
    has_ftx = getattr(gripen, 'has_ftx', False) or False
    has_f_only = getattr(gripen, 'has_f_only', False) or False
    has_natural = getattr(gripen, 'has_natural_draft', False) or False

    if has_ftx:
        ventilation = VentilationSystem.FTX
        hr_eff = 0.80
    elif has_f_only:
        ventilation = VentilationSystem.F_SYSTEM
        hr_eff = 0.0
    elif has_natural:
        ventilation = VentilationSystem.NATURAL
        hr_eff = 0.0
    else:
        ventilation = VentilationSystem.F_SYSTEM
        hr_eff = 0.0

    # Special case: FTX-VP (exhaust air HP with FTX)
    if primary_heating == SwedishHVACSystem.EXHAUST_AIR_HP:
        ventilation = VentilationSystem.FTX_VP
        hr_eff = 0.0  # Heat recovery is via HP, not plate exchanger

    return HVACSelection(
        primary_heating=primary_heating,
        ventilation=ventilation,
        heat_recovery_eff=hr_eff,
        hp_cop=_estimate_cop(primary_heating, None),
        detected_from="gripen",
        confidence=0.85,
    )


def hvac_from_archetype(archetype) -> HVACSelection:
    """
    Get HVAC configuration from archetype defaults.

    Uses archetype.hvac properties if available.
    """
    if archetype is None:
        return _default_hvac_selection()

    # Get from archetype HVAC properties
    hvac_props = getattr(archetype, 'hvac', None)
    if hvac_props is None:
        return _default_hvac_selection()

    # Map archetype heating system to our enum
    heating_system = getattr(hvac_props, 'heating_system', None)
    heating_map = {
        'DISTRICT': SwedishHVACSystem.DISTRICT_HEATING,
        'ELECTRIC': SwedishHVACSystem.DIRECT_ELECTRIC,
        'HEAT_PUMP_GROUND': SwedishHVACSystem.GROUND_SOURCE_HP,
        'HEAT_PUMP_AIR': SwedishHVACSystem.AIR_SOURCE_HP,
        'OIL': SwedishHVACSystem.OIL_BOILER,
        'GAS': SwedishHVACSystem.DISTRICT_HEATING,
        'PELLET': SwedishHVACSystem.PELLET_BOILER,
    }

    if heating_system and hasattr(heating_system, 'name'):
        primary = heating_map.get(heating_system.name, SwedishHVACSystem.DISTRICT_HEATING)
    else:
        primary = SwedishHVACSystem.DISTRICT_HEATING

    # Map ventilation type
    vent_type = getattr(hvac_props, 'ventilation_type', None)
    vent_map = {
        'BALANCED': VentilationSystem.FTX,
        'EXHAUST': VentilationSystem.F_SYSTEM,
        'NATURAL': VentilationSystem.NATURAL,
        'BALANCED_NO_HR': VentilationSystem.FT_SYSTEM,
    }

    if vent_type and hasattr(vent_type, 'name'):
        ventilation = vent_map.get(vent_type.name, VentilationSystem.F_SYSTEM)
    else:
        ventilation = VentilationSystem.F_SYSTEM

    # Get heat recovery efficiency
    hr_eff = getattr(hvac_props, 'heat_recovery_efficiency', 0.0) or 0.0

    return HVACSelection(
        primary_heating=primary,
        ventilation=ventilation,
        heat_recovery_eff=hr_eff,
        hp_cop=3.5,
        detected_from="archetype",
        confidence=0.70,
    )


def hvac_from_era(construction_year: int) -> HVACSelection:
    """
    Infer HVAC from construction era (lowest confidence).

    Uses era-based defaults from swedish_systems.py.
    """
    defaults = get_hvac_defaults_for_era(construction_year)

    return HVACSelection(
        primary_heating=defaults.get('heating', SwedishHVACSystem.DISTRICT_HEATING),
        ventilation=defaults.get('ventilation', VentilationSystem.F_SYSTEM),
        heat_recovery_eff=defaults.get('heat_recovery_eff', 0.0),
        hp_cop=defaults.get('hp_cop', 3.5),
        supply_temp_c=defaults.get('supply_temp_c', 55.0),
        return_temp_c=defaults.get('return_temp_c', 45.0),
        detected_from="era_inference",
        confidence=0.60,
    )


def _default_hvac_selection() -> HVACSelection:
    """Return default HVAC selection (district heating, F-system)."""
    return HVACSelection(
        primary_heating=SwedishHVACSystem.DISTRICT_HEATING,
        ventilation=VentilationSystem.F_SYSTEM,
        heat_recovery_eff=0.0,
        detected_from="default",
        confidence=0.50,
    )


def _map_ventilation_type(vent_str: Optional[str]) -> VentilationSystem:
    """Map ventilation type string to enum."""
    if vent_str is None:
        return VentilationSystem.F_SYSTEM

    vent_str = vent_str.upper()

    if 'FTX' in vent_str:
        return VentilationSystem.FTX
    elif vent_str == 'F' or 'FRÅNLUFT' in vent_str:
        return VentilationSystem.F_SYSTEM
    elif vent_str == 'FT':
        return VentilationSystem.FT_SYSTEM
    elif vent_str == 'S' or 'SJÄLV' in vent_str or 'NATURAL' in vent_str:
        return VentilationSystem.NATURAL
    else:
        return VentilationSystem.F_SYSTEM


def _get_hr_eff(vent_str: Optional[str]) -> float:
    """Get heat recovery efficiency from ventilation type."""
    if vent_str is None:
        return 0.0

    vent_str = vent_str.upper()

    if 'FTX' in vent_str:
        return 0.80  # Typical Swedish FTX
    else:
        return 0.0  # No heat recovery


def _estimate_cop(system: SwedishHVACSystem, building) -> float:
    """Estimate heat pump COP based on system type and building data."""
    cop_defaults = {
        SwedishHVACSystem.GROUND_SOURCE_HP: 4.2,    # High COP, stable source
        SwedishHVACSystem.EXHAUST_AIR_HP: 3.5,      # Moderate COP, 20°C source
        SwedishHVACSystem.AIR_SOURCE_HP: 3.0,       # Lower COP, variable source
        SwedishHVACSystem.DISTRICT_HEATING: 1.0,    # Not a HP, but for consistency
        SwedishHVACSystem.DIRECT_ELECTRIC: 1.0,
        SwedishHVACSystem.OIL_BOILER: 0.9,          # Efficiency, not COP
        SwedishHVACSystem.PELLET_BOILER: 0.85,
    }

    return cop_defaults.get(system, 3.5)
