"""
Swedish HVAC System Modeling for EnergyPlus.

This module provides realistic HVAC equipment modeling for Swedish buildings,
replacing the simplified IdealLoadsAirSystem with actual equipment:

- District Heating (fjärrvärme) - 70% of Swedish MFH
- Ground Source Heat Pump (bergvärme) - COP 4.0-5.0
- Exhaust Air Heat Pump (FTX-VP) - COP 3.0-4.0
- Air Source Heat Pump (luft-vatten) - COP 2.5-3.5
- Direct Electric (direktel)

Usage:
    from src.hvac import (
        SwedishHVACSystem,
        HVACTemplate,
        select_hvac_system,
        generate_hvac_idf,
    )

    # Auto-select from building data
    selection = select_hvac_system(
        building=swedish_building,  # From GeoJSON
        gripen=gripen_building,     # From Gripen
        archetype=archetype,        # Fallback
    )

    # Generate EnergyPlus IDF objects
    idf_snippet = generate_hvac_idf(
        system_type=selection.primary_heating,
        zone_names=["Floor1", "Floor2"],
        design_heating_load_w=50000,
    )
"""

from .swedish_systems import (
    SwedishHVACSystem,
    VentilationSystem,
    HVACTemplate,
    HVACSelection,
    generate_hvac_idf,
    DISTRICT_HEATING_TEMPLATE,
    EXHAUST_AIR_HP_TEMPLATE,
    GSHP_TEMPLATE,
    RADIATOR_TEMPLATE,
)

from .hvac_selector import (
    select_hvac_system,
    detect_heating_from_sweden_building,
    detect_heating_from_gripen,
    hvac_from_archetype,
)

__all__ = [
    # Enums
    "SwedishHVACSystem",
    "VentilationSystem",
    # Data classes
    "HVACTemplate",
    "HVACSelection",
    # Functions
    "select_hvac_system",
    "generate_hvac_idf",
    "detect_heating_from_sweden_building",
    "detect_heating_from_gripen",
    "hvac_from_archetype",
    # Templates
    "DISTRICT_HEATING_TEMPLATE",
    "EXHAUST_AIR_HP_TEMPLATE",
    "GSHP_TEMPLATE",
    "RADIATOR_TEMPLATE",
]
