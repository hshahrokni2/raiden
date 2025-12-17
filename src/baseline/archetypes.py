"""
Swedish Building Archetypes

Based on TABULA/EPISCOPE Swedish building stock study and
Boverket building regulations (BBR) by era.

Each archetype defines:
- Envelope U-values (walls, roof, floor, windows)
- Infiltration rate
- HVAC system type
- Internal loads (Sveby defaults)
- Typical WWR

Usage:
    matcher = ArchetypeMatcher()
    archetype = matcher.match(
        construction_year=1968,
        building_type='multi_family',
        facade_material='concrete'
    )
"""

from dataclasses import dataclass
from typing import Dict, Optional, List
from enum import Enum


class BuildingType(Enum):
    """Swedish building type classification."""
    MULTI_FAMILY = "flerbostadshus"
    SINGLE_FAMILY = "småhus"
    ROW_HOUSE = "radhus"
    COMMERCIAL = "lokal"
    MIXED_USE = "blandad"


class HeatingSystem(Enum):
    """Primary heating system type."""
    DISTRICT = "fjärrvärme"
    ELECTRIC = "direktel"
    HEAT_PUMP_AIR = "luftvärmepump"
    HEAT_PUMP_GROUND = "bergvärme"
    OIL = "olja"
    GAS = "gas"
    PELLET = "pellets"


class VentilationType(Enum):
    """Ventilation system type."""
    NATURAL = "självdrag"  # Pre-1970s
    EXHAUST = "frånluft"  # F-system, 1970s+
    BALANCED = "ftx"  # FTX with heat recovery
    BALANCED_NO_HR = "ft"  # Balanced without heat recovery


@dataclass
class EnvelopeProperties:
    """Thermal properties of building envelope."""
    wall_u_value: float  # W/m²K
    roof_u_value: float
    floor_u_value: float
    window_u_value: float
    window_shgc: float  # Solar heat gain coefficient
    infiltration_ach: float  # Air changes per hour at 50Pa / 20


@dataclass
class HVACProperties:
    """HVAC system properties."""
    heating_system: HeatingSystem
    ventilation_type: VentilationType
    heat_recovery_efficiency: float  # 0 for no HR, 0.7-0.9 for FTX
    ventilation_rate_l_s_m2: float  # BBR requirement 0.35
    sfp_kw_per_m3s: float  # Specific fan power


@dataclass
class InternalLoads:
    """Internal heat gains (Sveby-based)."""
    occupancy_m2_per_person: float
    occupancy_heat_w_per_person: float
    lighting_w_m2: float
    equipment_w_m2: float
    dhw_kwh_m2_year: float  # Hot water (not in E+ model)


@dataclass
class SwedishArchetype:
    """Complete Swedish building archetype."""
    name: str
    era_start: int
    era_end: int
    building_types: List[BuildingType]
    facade_materials: List[str]  # ['brick', 'concrete', 'render', 'wood']

    envelope: EnvelopeProperties
    hvac: HVACProperties
    loads: InternalLoads

    # Typical geometry
    typical_wwr: float
    typical_floor_height_m: float

    # Notes
    description: str
    common_issues: List[str]
    typical_ecms: List[str]


# =============================================================================
# SWEDISH ARCHETYPE DATABASE
# =============================================================================

SWEDISH_ARCHETYPES: Dict[str, SwedishArchetype] = {

    "pre_1945_brick": SwedishArchetype(
        name="Pre-1945 Brick (Funkis/Older)",
        era_start=1900,
        era_end=1945,
        building_types=[BuildingType.MULTI_FAMILY],
        facade_materials=['brick'],
        envelope=EnvelopeProperties(
            wall_u_value=1.0,  # Solid brick, no cavity insulation
            roof_u_value=0.30,
            floor_u_value=0.40,
            window_u_value=2.5,  # Original double glazing
            window_shgc=0.70,
            infiltration_ach=0.30,  # Leaky
        ),
        hvac=HVACProperties(
            heating_system=HeatingSystem.DISTRICT,
            ventilation_type=VentilationType.NATURAL,
            heat_recovery_efficiency=0.0,
            ventilation_rate_l_s_m2=0.35,
            sfp_kw_per_m3s=0.0,  # Natural ventilation
        ),
        loads=InternalLoads(
            occupancy_m2_per_person=30,  # Larger apartments
            occupancy_heat_w_per_person=80,
            lighting_w_m2=8,
            equipment_w_m2=8,
            dhw_kwh_m2_year=25,
        ),
        typical_wwr=0.20,
        typical_floor_height_m=3.0,
        description="Pre-war brick buildings, often with ornate facades",
        common_issues=["High infiltration", "No wall insulation", "Cold floors"],
        typical_ecms=["Window replacement", "Attic insulation", "Air sealing"],
    ),

    "1945_1960_brick": SwedishArchetype(
        name="1945-1960 Brick (Folkhemmet)",
        era_start=1945,
        era_end=1960,
        building_types=[BuildingType.MULTI_FAMILY],
        facade_materials=['brick', 'render'],
        envelope=EnvelopeProperties(
            wall_u_value=0.80,  # Some cavity insulation
            roof_u_value=0.25,
            floor_u_value=0.35,
            window_u_value=2.5,
            window_shgc=0.70,
            infiltration_ach=0.20,
        ),
        hvac=HVACProperties(
            heating_system=HeatingSystem.DISTRICT,
            ventilation_type=VentilationType.NATURAL,
            heat_recovery_efficiency=0.0,
            ventilation_rate_l_s_m2=0.35,
            sfp_kw_per_m3s=0.0,
        ),
        loads=InternalLoads(
            occupancy_m2_per_person=28,
            occupancy_heat_w_per_person=80,
            lighting_w_m2=8,
            equipment_w_m2=8,
            dhw_kwh_m2_year=25,
        ),
        typical_wwr=0.18,
        typical_floor_height_m=2.7,
        description="Post-war 'Folkhemmet' era, standardized construction",
        common_issues=["Limited insulation", "Thermal bridges at balconies"],
        typical_ecms=["Window replacement", "Roof insulation", "FTX installation"],
    ),

    "1961_1975_concrete": SwedishArchetype(
        name="1961-1975 Concrete Panel (Miljonprogrammet)",
        era_start=1961,
        era_end=1975,
        building_types=[BuildingType.MULTI_FAMILY],
        facade_materials=['concrete'],
        envelope=EnvelopeProperties(
            wall_u_value=0.50,  # Sandwich panels with insulation
            roof_u_value=0.20,
            floor_u_value=0.30,
            window_u_value=2.0,
            window_shgc=0.65,
            infiltration_ach=0.15,
        ),
        hvac=HVACProperties(
            heating_system=HeatingSystem.DISTRICT,
            ventilation_type=VentilationType.EXHAUST,  # F-system introduced
            heat_recovery_efficiency=0.0,
            ventilation_rate_l_s_m2=0.35,
            sfp_kw_per_m3s=1.0,
        ),
        loads=InternalLoads(
            occupancy_m2_per_person=25,
            occupancy_heat_w_per_person=80,
            lighting_w_m2=10,
            equipment_w_m2=10,
            dhw_kwh_m2_year=25,
        ),
        typical_wwr=0.22,
        typical_floor_height_m=2.6,
        description="Million Programme concrete panel buildings",
        common_issues=["Thermal bridges", "Facade degradation", "F-ventilation losses"],
        typical_ecms=["External wall insulation", "FTX conversion", "Window replacement"],
    ),

    "1976_1985_insulated": SwedishArchetype(
        name="1976-1985 Insulated (Post Oil Crisis)",
        era_start=1976,
        era_end=1985,
        building_types=[BuildingType.MULTI_FAMILY, BuildingType.ROW_HOUSE],
        facade_materials=['brick', 'render', 'wood'],
        envelope=EnvelopeProperties(
            wall_u_value=0.30,  # Better insulation after oil crisis
            roof_u_value=0.15,
            floor_u_value=0.25,
            window_u_value=1.8,  # Triple glazing introduced
            window_shgc=0.60,
            infiltration_ach=0.10,
        ),
        hvac=HVACProperties(
            heating_system=HeatingSystem.DISTRICT,
            ventilation_type=VentilationType.EXHAUST,
            heat_recovery_efficiency=0.0,
            ventilation_rate_l_s_m2=0.35,
            sfp_kw_per_m3s=1.2,
        ),
        loads=InternalLoads(
            occupancy_m2_per_person=25,
            occupancy_heat_w_per_person=80,
            lighting_w_m2=10,
            equipment_w_m2=10,
            dhw_kwh_m2_year=25,
        ),
        typical_wwr=0.20,
        typical_floor_height_m=2.5,
        description="Post oil crisis, improved insulation standards",
        common_issues=["Still F-ventilation", "Some thermal bridges"],
        typical_ecms=["FTX conversion", "Air sealing", "Window upgrade to U=1.0"],
    ),

    "1986_1995_well_insulated": SwedishArchetype(
        name="1986-1995 Well Insulated",
        era_start=1986,
        era_end=1995,
        building_types=[BuildingType.MULTI_FAMILY, BuildingType.ROW_HOUSE],
        facade_materials=['brick', 'render', 'wood'],
        envelope=EnvelopeProperties(
            wall_u_value=0.22,
            roof_u_value=0.12,
            floor_u_value=0.20,
            window_u_value=1.5,
            window_shgc=0.55,
            infiltration_ach=0.08,
        ),
        hvac=HVACProperties(
            heating_system=HeatingSystem.DISTRICT,
            ventilation_type=VentilationType.BALANCED,  # FTX becoming common
            heat_recovery_efficiency=0.70,
            ventilation_rate_l_s_m2=0.35,
            sfp_kw_per_m3s=1.5,
        ),
        loads=InternalLoads(
            occupancy_m2_per_person=25,
            occupancy_heat_w_per_person=80,
            lighting_w_m2=8,
            equipment_w_m2=10,
            dhw_kwh_m2_year=25,
        ),
        typical_wwr=0.18,
        typical_floor_height_m=2.5,
        description="FTX becoming standard, good insulation",
        common_issues=["FTX units aging", "Original windows still adequate"],
        typical_ecms=["FTX upgrade to 85%", "LED lighting", "Solar PV"],
    ),

    "1996_2010_modern": SwedishArchetype(
        name="1996-2010 Modern",
        era_start=1996,
        era_end=2010,
        building_types=[BuildingType.MULTI_FAMILY, BuildingType.ROW_HOUSE],
        facade_materials=['brick', 'render', 'glass'],
        envelope=EnvelopeProperties(
            wall_u_value=0.18,
            roof_u_value=0.10,
            floor_u_value=0.15,
            window_u_value=1.2,
            window_shgc=0.50,
            infiltration_ach=0.06,
        ),
        hvac=HVACProperties(
            heating_system=HeatingSystem.DISTRICT,
            ventilation_type=VentilationType.BALANCED,
            heat_recovery_efficiency=0.75,
            ventilation_rate_l_s_m2=0.35,
            sfp_kw_per_m3s=1.5,
        ),
        loads=InternalLoads(
            occupancy_m2_per_person=25,
            occupancy_heat_w_per_person=80,
            lighting_w_m2=8,
            equipment_w_m2=12,
            dhw_kwh_m2_year=25,
        ),
        typical_wwr=0.20,
        typical_floor_height_m=2.7,
        description="Modern BBR standards, efficient systems",
        common_issues=["Good baseline, limited ECM potential"],
        typical_ecms=["Solar PV", "DCV", "LED lighting"],
    ),

    "2011_plus_low_energy": SwedishArchetype(
        name="2011+ Low Energy",
        era_start=2011,
        era_end=2030,
        building_types=[BuildingType.MULTI_FAMILY, BuildingType.ROW_HOUSE],
        facade_materials=['render', 'wood', 'glass'],
        envelope=EnvelopeProperties(
            wall_u_value=0.12,
            roof_u_value=0.08,
            floor_u_value=0.12,
            window_u_value=0.9,
            window_shgc=0.45,
            infiltration_ach=0.04,
        ),
        hvac=HVACProperties(
            heating_system=HeatingSystem.DISTRICT,
            ventilation_type=VentilationType.BALANCED,
            heat_recovery_efficiency=0.85,
            ventilation_rate_l_s_m2=0.35,
            sfp_kw_per_m3s=1.5,
        ),
        loads=InternalLoads(
            occupancy_m2_per_person=25,
            occupancy_heat_w_per_person=80,
            lighting_w_m2=6,  # LED standard
            equipment_w_m2=12,
            dhw_kwh_m2_year=20,  # Heat pump DHW
        ),
        typical_wwr=0.22,
        typical_floor_height_m=2.7,
        description="BBR 2011+ requirements, approaching passive house",
        common_issues=["Very efficient, minimal ECM potential"],
        typical_ecms=["Solar PV", "Battery storage"],
    ),
}


class ArchetypeMatcher:
    """
    Match building to most appropriate archetype.

    Usage:
        matcher = ArchetypeMatcher()
        archetype = matcher.match(
            construction_year=1968,
            building_type=BuildingType.MULTI_FAMILY,
            facade_material='concrete'
        )
    """

    def __init__(self, archetypes: Dict[str, SwedishArchetype] = None):
        self.archetypes = archetypes or SWEDISH_ARCHETYPES

    def match(
        self,
        construction_year: int,
        building_type: BuildingType = BuildingType.MULTI_FAMILY,
        facade_material: Optional[str] = None
    ) -> SwedishArchetype:
        """
        Find best matching archetype.

        Args:
            construction_year: Year building was constructed
            building_type: Type of building
            facade_material: Detected facade material (optional)

        Returns:
            Best matching SwedishArchetype
        """
        candidates = []

        for key, archetype in self.archetypes.items():
            # Check year range
            if archetype.era_start <= construction_year <= archetype.era_end:
                score = 100  # Base score for year match

                # Boost for building type match
                if building_type in archetype.building_types:
                    score += 20

                # Boost for material match
                if facade_material and facade_material.lower() in [m.lower() for m in archetype.facade_materials]:
                    score += 30

                candidates.append((score, archetype))

        if not candidates:
            # Fallback to closest year
            closest = min(
                self.archetypes.values(),
                key=lambda a: min(abs(construction_year - a.era_start), abs(construction_year - a.era_end))
            )
            return closest

        # Return highest scoring match
        candidates.sort(key=lambda x: x[0], reverse=True)
        return candidates[0][1]

    def list_archetypes(self) -> List[str]:
        """List all available archetype names."""
        return list(self.archetypes.keys())
