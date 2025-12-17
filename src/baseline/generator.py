"""
Baseline Generator - Auto-generate EnergyPlus IDF from building data.

Takes:
- Building geometry (from OSM/Overture)
- Archetype properties (from era/material matching)
- Location (weather file selection)

Outputs:
- Complete EnergyPlus IDF ready for simulation
- Uses IdealLoadsAirSystem for fast, stable simulation

CRITICAL: Uses the E+ 25.1.0 bug workaround:
- ConstantSupplyHumidityRatio for humidity controls (NOT 'None')
- Blank Cooling Sensible Heat Ratio
"""

from dataclasses import dataclass
from typing import Optional
from pathlib import Path

from ..geometry.building_geometry import BuildingGeometry
from .archetypes import SwedishArchetype


@dataclass
class BaselineModel:
    """Generated baseline model."""
    idf_path: Path
    weather_file: str
    archetype_used: str
    floor_area_m2: float
    predicted_heating_kwh_m2: float  # Before calibration


class BaselineGenerator:
    """
    Generate EnergyPlus baseline model.

    Usage:
        generator = BaselineGenerator()
        model = generator.generate(
            geometry=building_geometry,
            archetype=matched_archetype,
            output_dir=Path('./output')
        )
    """

    # EnergyPlus 25.1.0 IdealLoadsAirSystem template
    # CRITICAL: Uses workaround for segfault bug
    IDEAL_LOADS_TEMPLATE = '''
ZoneHVAC:IdealLoadsAirSystem,
    {zone_name}_IdealLoads,      !- Name
    ,                            !- Availability Schedule Name
    {zone_name}_Supply,          !- Zone Supply Air Node Name
    {zone_name}_Exhaust,         !- Zone Exhaust Air Node Name
    ,                            !- System Inlet Air Node Name
    50,                          !- Maximum Heating Supply Air Temperature {{C}}
    13,                          !- Minimum Cooling Supply Air Temperature {{C}}
    0.015,                       !- Maximum Heating Supply Air Humidity Ratio
    0.009,                       !- Minimum Cooling Supply Air Humidity Ratio
    NoLimit,                     !- Heating Limit
    autosize,                    !- Maximum Heating Air Flow Rate
    ,                            !- Maximum Sensible Heating Capacity
    NoLimit,                     !- Cooling Limit
    autosize,                    !- Maximum Cooling Air Flow Rate
    ,                            !- Maximum Total Cooling Capacity
    ,                            !- Heating Availability Schedule Name
    ,                            !- Cooling Availability Schedule Name
    ConstantSupplyHumidityRatio, !- Dehumidification Control Type (NOT None!)
    ,                            !- Cooling Sensible Heat Ratio (BLANK!)
    ConstantSupplyHumidityRatio, !- Humidification Control Type (NOT None!)
    {oa_spec},                   !- Design Specification Outdoor Air Object Name
    {zone_name}_OA,              !- Outdoor Air Inlet Node Name
    None,                        !- Demand Controlled Ventilation Type
    NoEconomizer,                !- Outdoor Air Economizer Type
    {hr_type},                   !- Heat Recovery Type
    {hr_effectiveness},          !- Sensible Heat Recovery Effectiveness
    0.0;                         !- Latent Heat Recovery Effectiveness
'''

    def __init__(self):
        pass

    def generate(
        self,
        geometry: BuildingGeometry,
        archetype: SwedishArchetype,
        output_dir: Path,
        model_name: str = "baseline"
    ) -> BaselineModel:
        """
        Generate EnergyPlus IDF model.

        Args:
            geometry: Calculated building geometry
            archetype: Matched Swedish archetype
            output_dir: Directory for output files
            model_name: Base name for output files

        Returns:
            BaselineModel with path to generated IDF
        """
        # TODO: Implement
        # 1. Generate header (Version, SimulationControl, etc.)
        # 2. Generate Building object
        # 3. Generate Site:Location (use Stockholm default)
        # 4. Generate materials and constructions from archetype
        # 5. Generate zones (one per floor)
        # 6. Generate surfaces (walls, roof, floor) from geometry
        # 7. Generate windows using WWR from geometry
        # 8. Generate IdealLoadsAirSystem for each zone
        # 9. Generate schedules (Sveby-based)
        # 10. Generate internal loads (people, lights, equipment)
        # 11. Generate output variables
        # 12. Write IDF file
        raise NotImplementedError("Implement IDF generation")

    def _generate_materials(self, archetype: SwedishArchetype) -> str:
        """Generate Material and Construction objects."""
        raise NotImplementedError()

    def _generate_zone(self, floor: int, geometry: BuildingGeometry) -> str:
        """Generate Zone object for a floor."""
        raise NotImplementedError()

    def _generate_surfaces(self, floor: int, geometry: BuildingGeometry) -> str:
        """Generate BuildingSurface:Detailed objects."""
        raise NotImplementedError()

    def _generate_windows(self, floor: int, geometry: BuildingGeometry) -> str:
        """Generate FenestrationSurface:Detailed objects."""
        raise NotImplementedError()

    def _generate_ideal_loads(
        self,
        zone_name: str,
        archetype: SwedishArchetype
    ) -> str:
        """Generate IdealLoadsAirSystem for zone."""
        hr_type = "Sensible" if archetype.hvac.heat_recovery_efficiency > 0 else "None"
        hr_eff = archetype.hvac.heat_recovery_efficiency

        return self.IDEAL_LOADS_TEMPLATE.format(
            zone_name=zone_name,
            oa_spec="OA_Spec",
            hr_type=hr_type,
            hr_effectiveness=hr_eff
        )

    def _generate_schedules(self) -> str:
        """Generate Sveby-based schedules."""
        raise NotImplementedError()

    def _generate_internal_loads(
        self,
        zone_name: str,
        floor_area: float,
        archetype: SwedishArchetype
    ) -> str:
        """Generate People, Lights, ElectricEquipment."""
        raise NotImplementedError()

    def _select_weather_file(self, latitude: float = 59.3) -> str:
        """Select appropriate Swedish weather file."""
        # Default to Stockholm
        return "SWE_Stockholm.Arlanda.024600_IWEC.epw"
