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
from typing import Optional, List, Tuple
from pathlib import Path
import math
import logging

from ..geometry.building_geometry import BuildingGeometry
from .archetypes import SwedishArchetype

logger = logging.getLogger(__name__)


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
    Generate EnergyPlus baseline model from BuildingGeometry and SwedishArchetype.

    Creates a simplified multi-zone model with:
    - One thermal zone per floor
    - Rectangular approximation of footprint
    - IdealLoadsAirSystem with FTX heat recovery
    - Sveby-based schedules for Swedish residential

    Usage:
        generator = BaselineGenerator()
        model = generator.generate(
            geometry=building_geometry,
            archetype=matched_archetype,
            output_dir=Path('./output'),
            model_name="my_building"
        )
    """

    def __init__(self):
        pass

    def _validate_inputs(
        self,
        geometry: BuildingGeometry,
        archetype: SwedishArchetype,
        latitude: float,
        longitude: float
    ) -> None:
        """Validate inputs before generation."""
        errors = []

        # Validate latitude/longitude
        if not -90 <= latitude <= 90:
            errors.append(f"Latitude {latitude} out of range [-90, 90]")
        if not -180 <= longitude <= 180:
            errors.append(f"Longitude {longitude} out of range [-180, 180]")

        # Validate geometry
        if geometry.footprint_area_m2 <= 0:
            errors.append(f"Footprint area must be positive, got {geometry.footprint_area_m2}")
        if geometry.gross_floor_area_m2 <= 0:
            errors.append(f"Gross floor area must be positive, got {geometry.gross_floor_area_m2}")
        if geometry.floors <= 0:
            errors.append(f"Number of floors must be positive, got {geometry.floors}")
        if geometry.height_m <= 0:
            errors.append(f"Building height must be positive, got {geometry.height_m}")
        if geometry.perimeter_m <= 0:
            errors.append(f"Perimeter must be positive, got {geometry.perimeter_m}")

        # Validate archetype
        if archetype.envelope is None:
            errors.append("Archetype missing envelope properties")
        if archetype.hvac is None:
            errors.append("Archetype missing HVAC properties")

        # Validate envelope U-values
        if archetype.envelope:
            if archetype.envelope.wall_u_value <= 0:
                errors.append(f"Wall U-value must be positive, got {archetype.envelope.wall_u_value}")
            if archetype.envelope.roof_u_value <= 0:
                errors.append(f"Roof U-value must be positive, got {archetype.envelope.roof_u_value}")
            if archetype.envelope.window_u_value <= 0:
                errors.append(f"Window U-value must be positive, got {archetype.envelope.window_u_value}")

        if errors:
            error_msg = "Invalid inputs:\n  - " + "\n  - ".join(errors)
            logger.error(error_msg)
            raise ValueError(error_msg)

    def generate(
        self,
        geometry: BuildingGeometry,
        archetype: SwedishArchetype,
        output_dir: Path,
        model_name: str = "baseline",
        latitude: float = 59.35,
        longitude: float = 17.95,
    ) -> BaselineModel:
        """
        Generate EnergyPlus IDF model.

        Args:
            geometry: Calculated building geometry
            archetype: Matched Swedish archetype
            output_dir: Directory for output files
            model_name: Base name for output files
            latitude: Site latitude (default Stockholm)
            longitude: Site longitude (default Stockholm)

        Returns:
            BaselineModel with path to generated IDF

        Raises:
            ValueError: If inputs are invalid
        """
        # Input validation
        self._validate_inputs(geometry, archetype, latitude, longitude)

        logger.info(f"Generating baseline model: {model_name}")
        logger.info(f"  Floor area: {geometry.gross_floor_area_m2:.0f} m²")
        logger.info(f"  Floors: {geometry.floors}")
        logger.info(f"  Archetype: {archetype.name}")

        output_dir.mkdir(parents=True, exist_ok=True)
        idf_path = output_dir / f"{model_name}.idf"

        # Calculate simplified rectangular dimensions from footprint
        # Use equivalent rectangle preserving area and approximate aspect ratio
        footprint_area = geometry.footprint_area_m2
        perimeter = geometry.perimeter_m

        # Solve for width and length: A = w*l, P = 2*(w+l)
        # l + w = P/2, l*w = A
        # Quadratic: x^2 - (P/2)*x + A = 0
        half_perim = perimeter / 2
        discriminant = half_perim**2 - 4 * footprint_area
        if discriminant < 0:
            # Fallback to square
            width = math.sqrt(footprint_area)
            length = width
        else:
            length = (half_perim + math.sqrt(discriminant)) / 2
            width = footprint_area / length

        # Ensure width <= length
        if width > length:
            width, length = length, width

        floors = geometry.floors
        floor_height = geometry.floor_height_m
        total_height = geometry.height_m

        # Build IDF content
        idf_parts = []

        # 1. Header
        idf_parts.append(self._generate_header(model_name, latitude, longitude))

        # 2. Schedules
        idf_parts.append(self._generate_schedules())

        # 3. Materials and Constructions
        idf_parts.append(self._generate_materials(archetype))

        # 4. Outdoor air nodes for all floors
        idf_parts.append(self._generate_oa_nodes(floors))

        # 5. Zones and surfaces for each floor
        for floor in range(1, floors + 1):
            floor_z = (floor - 1) * floor_height
            is_ground_floor = (floor == 1)
            is_top_floor = (floor == floors)

            idf_parts.append(self._generate_zone(
                floor=floor,
                width=width,
                length=length,
                floor_height=floor_height,
                floor_z=floor_z,
                floor_area=footprint_area,
            ))

            idf_parts.append(self._generate_surfaces(
                floor=floor,
                width=width,
                length=length,
                floor_height=floor_height,
                floor_z=floor_z,
                is_ground_floor=is_ground_floor,
                is_top_floor=is_top_floor,
                floors_total=floors,
            ))

            idf_parts.append(self._generate_windows(
                floor=floor,
                width=width,
                length=length,
                floor_height=floor_height,
                floor_z=floor_z,
                geometry=geometry,
            ))

            idf_parts.append(self._generate_internal_loads(
                zone_name=f"Floor{floor}",
                floor_area=footprint_area,
                archetype=archetype,
            ))

            idf_parts.append(self._generate_hvac(
                zone_name=f"Floor{floor}",
                archetype=archetype,
            ))

        # 6. Output variables
        idf_parts.append(self._generate_outputs())

        # Write IDF file
        idf_content = "\n".join(idf_parts)
        with open(idf_path, "w") as f:
            f.write(idf_content)

        # Estimate heating demand (rough)
        # Based on Swedish typical: ~40-60 kWh/m² for 2000s buildings
        estimated_heating = self._estimate_heating(geometry, archetype)

        return BaselineModel(
            idf_path=idf_path,
            weather_file=self._select_weather_file(latitude),
            archetype_used=archetype.name,
            floor_area_m2=geometry.gross_floor_area_m2,
            predicted_heating_kwh_m2=estimated_heating,
        )

    def _generate_header(
        self,
        model_name: str,
        latitude: float,
        longitude: float,
    ) -> str:
        """Generate IDF header with simulation settings."""
        return f'''!- ============================================================
!- {model_name} - Auto-generated Swedish Building Model
!- ============================================================
!- Generated by Raiden BaselineGenerator
!- Full 8760-hour annual simulation
!-
!- SOURCES:
!- - TABULA/EPISCOPE Swedish Building Typology
!- - Sveby Brukarindata Bostader v2.0
!- - BBR 6:251 Ventilation Requirements
!- ============================================================

Version,25.1;

Building,
    {model_name},
    0,                       !- North Axis (deg)
    City,                    !- Terrain
    0.04,                    !- Loads Convergence Tolerance
    0.4,                     !- Temperature Convergence Tolerance
    FullExterior,            !- Solar Distribution
    25,                      !- Maximum Number of Warmup Days
    6;                       !- Minimum Number of Warmup Days

Timestep,4;

HeatBalanceAlgorithm,ConductionTransferFunction;

SurfaceConvectionAlgorithm:Inside,TARP;

SurfaceConvectionAlgorithm:Outside,DOE-2;

SimulationControl,
    No,                      !- Do Zone Sizing Calculation
    No,                      !- Do System Sizing Calculation
    No,                      !- Do Plant Sizing Calculation
    No,                      !- Run Simulation for Sizing Periods
    Yes;                     !- Run Simulation for Weather File Run Periods

RunPeriod,
    Annual,
    1,                       !- Begin Month
    1,                       !- Begin Day of Month
    ,                        !- Begin Year
    12,                      !- End Month
    31,                      !- End Day of Month
    ,                        !- End Year
    Sunday,                  !- Day of Week for Start Day
    No,                      !- Use Weather File Holidays
    No,                      !- Use Weather File Daylight Saving Period
    No,                      !- Apply Weekend Holiday Rule
    Yes,                     !- Use Weather File Rain Indicators
    Yes;                     !- Use Weather File Snow Indicators

Site:Location,
    Stockholm,
    {latitude},              !- Latitude
    {longitude},             !- Longitude
    1.0,                     !- Time Zone
    44;                      !- Elevation (m)

GlobalGeometryRules,
    UpperLeftCorner,
    Counterclockwise,
    Relative;

Site:GroundTemperature:BuildingSurface,
    5, 5, 6, 8, 11, 14, 16, 16, 14, 11, 8, 5;
'''

    def _generate_schedules(self) -> str:
        """Generate Sveby-based schedules for Swedish residential."""
        return '''
!- ============================================================
!- SCHEDULES - Sveby Brukarindata Bostader
!- ============================================================

ScheduleTypeLimits,
    Fraction,
    0,
    1,
    Continuous;

ScheduleTypeLimits,
    Temperature,
    -50,
    50,
    Continuous;

ScheduleTypeLimits,
    Any Number,
    ,
    ,
    Continuous;

Schedule:Constant,AlwaysOn,Fraction,1;
Schedule:Constant,HeatSet,Temperature,21;
Schedule:Constant,CoolSet,Temperature,50;
Schedule:Constant,ThermType,Any Number,4;
Schedule:Constant,ActivityLevel,Any Number,120;
Schedule:Constant,FTX_FanSchedule,Fraction,1;

!- Occupancy: High morning/evening, low daytime (people at work)
Schedule:Compact,
    OccupancySchedule,
    Fraction,
    Through: 12/31,
    For: AllDays,
    Until: 07:00, 0.9,
    Until: 09:00, 0.5,
    Until: 17:00, 0.2,
    Until: 22:00, 0.9,
    Until: 24:00, 0.9;

!- Lighting: Low during day, higher evening
Schedule:Compact,
    LightingSchedule,
    Fraction,
    Through: 12/31,
    For: AllDays,
    Until: 06:00, 0.05,
    Until: 08:00, 0.5,
    Until: 17:00, 0.1,
    Until: 22:00, 0.7,
    Until: 24:00, 0.3;

!- Equipment: Background load + activity peaks
Schedule:Compact,
    EquipmentSchedule,
    Fraction,
    Through: 12/31,
    For: AllDays,
    Until: 07:00, 0.3,
    Until: 09:00, 0.6,
    Until: 17:00, 0.3,
    Until: 22:00, 0.7,
    Until: 24:00, 0.4;

!- Outdoor air specification: BBR 6:251 = 0.35 l/s/m²
DesignSpecification:OutdoorAir,
    OA_Spec,
    Flow/Area,
    ,
    0.00035;
'''

    def _generate_oa_nodes(self, floors: int) -> str:
        """Generate outdoor air node list for all floors."""
        lines = ["\n!- Outdoor Air Nodes"]
        for floor in range(1, floors + 1):
            lines.append(f"OutdoorAir:NodeList,Floor{floor}_OA;")
        return "\n".join(lines)

    def _generate_materials(self, archetype: SwedishArchetype) -> str:
        """Generate Material and Construction objects from archetype U-values."""
        # Calculate insulation thicknesses from U-values
        # U = 1 / (Rsi + R_layers + Rse)
        # For simplicity: R_insulation = 1/U - 0.17 (internal) - 0.04 (external) - R_concrete
        # R_concrete_200mm = 0.2/1.0 = 0.2 m²K/W

        wall_u = archetype.envelope.wall_u_value
        roof_u = archetype.envelope.roof_u_value
        floor_u = archetype.envelope.floor_u_value
        window_u = archetype.envelope.window_u_value
        window_shgc = archetype.envelope.window_shgc

        # Calculate required R-values
        r_surface = 0.17 + 0.04  # Internal + external surface resistance
        r_concrete = 0.2  # 200mm concrete

        # Insulation thickness = (1/U - r_surface - r_concrete) * k_insulation
        k_insulation = 0.035  # W/mK for mineral wool

        def calc_insulation_thickness(u_value: float) -> float:
            r_required = 1.0 / u_value
            r_insulation = max(0.05, r_required - r_surface - r_concrete)
            return r_insulation * k_insulation

        wall_ins_thickness = calc_insulation_thickness(wall_u)
        roof_ins_thickness = calc_insulation_thickness(roof_u)
        floor_ins_thickness = calc_insulation_thickness(floor_u)

        return f'''
!- ============================================================
!- MATERIALS AND CONSTRUCTIONS
!- Archetype: {archetype.name}
!- Wall U={wall_u:.2f}, Roof U={roof_u:.2f}, Floor U={floor_u:.2f}, Window U={window_u:.1f}
!- ============================================================

Material,
    Concrete200,
    MediumRough,
    0.2,                     !- Thickness (m)
    1.0,                     !- Conductivity (W/m-K)
    2300,                    !- Density (kg/m3)
    880;                     !- Specific Heat (J/kg-K)

Material,
    WallInsulation,
    MediumRough,
    {wall_ins_thickness:.3f},  !- Thickness (m)
    0.035,                   !- Conductivity (W/m-K)
    30,                      !- Density (kg/m3)
    840;                     !- Specific Heat (J/kg-K)

Material,
    RoofInsulation,
    MediumRough,
    {roof_ins_thickness:.3f},  !- Thickness (m)
    0.035,                   !- Conductivity (W/m-K)
    30,                      !- Density (kg/m3)
    840;                     !- Specific Heat (J/kg-K)

Material,
    FloorInsulation,
    MediumRough,
    {floor_ins_thickness:.3f}, !- Thickness (m)
    0.035,                   !- Conductivity (W/m-K)
    30,                      !- Density (kg/m3)
    840;                     !- Specific Heat (J/kg-K)

Construction,
    ExteriorWall,
    Concrete200,
    WallInsulation;

Construction,
    Roof,
    Concrete200,
    RoofInsulation;

Construction,
    GroundFloor,
    Concrete200,
    FloorInsulation;

Construction,
    InteriorFloor,
    Concrete200;

WindowMaterial:SimpleGlazingSystem,
    GlazingSystem,
    {window_u:.1f},          !- U-Factor (W/m2-K)
    {window_shgc:.2f};       !- Solar Heat Gain Coefficient

Construction,
    Window,
    GlazingSystem;
'''

    def _generate_zone(
        self,
        floor: int,
        width: float,
        length: float,
        floor_height: float,
        floor_z: float,
        floor_area: float,
    ) -> str:
        """Generate Zone object for a floor."""
        volume = floor_area * floor_height
        return f'''
!- ============================================================
!- FLOOR {floor}
!- ============================================================

Zone,
    Floor{floor},
    0,                       !- Direction of Relative North
    0, 0, {floor_z:.1f},     !- Origin X,Y,Z
    1,                       !- Type
    1,                       !- Multiplier
    {floor_height:.1f},      !- Ceiling Height
    {volume:.0f};            !- Volume
'''

    def _generate_surfaces(
        self,
        floor: int,
        width: float,
        length: float,
        floor_height: float,
        floor_z: float,
        is_ground_floor: bool,
        is_top_floor: bool,
        floors_total: int,
    ) -> str:
        """Generate BuildingSurface:Detailed objects for walls, floor, ceiling."""
        zone_name = f"Floor{floor}"
        z_bottom = floor_z
        z_top = floor_z + floor_height

        surfaces = []

        # Floor surface
        if is_ground_floor:
            floor_construction = "GroundFloor"
            floor_boundary = "Ground"
            floor_boundary_obj = ""
        else:
            floor_construction = "InteriorFloor"
            floor_boundary = "Surface"
            floor_boundary_obj = f"Floor{floor-1}_Ceiling"

        surfaces.append(f'''
BuildingSurface:Detailed,
    {zone_name}_Floor,
    Floor,
    {floor_construction},
    {zone_name},
    ,
    {floor_boundary},
    {floor_boundary_obj},
    NoSun,
    NoWind,
    ,
    4,
    0, {length:.1f}, {z_bottom:.1f},
    0, 0, {z_bottom:.1f},
    {width:.1f}, 0, {z_bottom:.1f},
    {width:.1f}, {length:.1f}, {z_bottom:.1f};''')

        # Ceiling surface
        if is_top_floor:
            ceiling_construction = "Roof"
            ceiling_boundary = "Outdoors"
            ceiling_boundary_obj = ""
            sun_wind = "SunExposed,\n    WindExposed"
        else:
            ceiling_construction = "InteriorFloor"
            ceiling_boundary = "Surface"
            ceiling_boundary_obj = f"Floor{floor+1}_Floor"
            sun_wind = "NoSun,\n    NoWind"

        surfaces.append(f'''
BuildingSurface:Detailed,
    {zone_name}_Ceiling,
    {"Roof" if is_top_floor else "Ceiling"},
    {ceiling_construction},
    {zone_name},
    ,
    {ceiling_boundary},
    {ceiling_boundary_obj},
    {sun_wind},
    ,
    4,
    0, 0, {z_top:.1f},
    0, {length:.1f}, {z_top:.1f},
    {width:.1f}, {length:.1f}, {z_top:.1f},
    {width:.1f}, 0, {z_top:.1f};''')

        # South wall (Y=0)
        surfaces.append(f'''
BuildingSurface:Detailed,
    {zone_name}_Wall_S,
    Wall,
    ExteriorWall,
    {zone_name},
    ,
    Outdoors,
    ,
    SunExposed,
    WindExposed,
    ,
    4,
    {width:.1f}, 0, {z_top:.1f},
    {width:.1f}, 0, {z_bottom:.1f},
    0, 0, {z_bottom:.1f},
    0, 0, {z_top:.1f};''')

        # North wall (Y=length)
        surfaces.append(f'''
BuildingSurface:Detailed,
    {zone_name}_Wall_N,
    Wall,
    ExteriorWall,
    {zone_name},
    ,
    Outdoors,
    ,
    SunExposed,
    WindExposed,
    ,
    4,
    0, {length:.1f}, {z_top:.1f},
    0, {length:.1f}, {z_bottom:.1f},
    {width:.1f}, {length:.1f}, {z_bottom:.1f},
    {width:.1f}, {length:.1f}, {z_top:.1f};''')

        # East wall (X=width)
        surfaces.append(f'''
BuildingSurface:Detailed,
    {zone_name}_Wall_E,
    Wall,
    ExteriorWall,
    {zone_name},
    ,
    Outdoors,
    ,
    SunExposed,
    WindExposed,
    ,
    4,
    {width:.1f}, {length:.1f}, {z_top:.1f},
    {width:.1f}, {length:.1f}, {z_bottom:.1f},
    {width:.1f}, 0, {z_bottom:.1f},
    {width:.1f}, 0, {z_top:.1f};''')

        # West wall (X=0)
        surfaces.append(f'''
BuildingSurface:Detailed,
    {zone_name}_Wall_W,
    Wall,
    ExteriorWall,
    {zone_name},
    ,
    Outdoors,
    ,
    SunExposed,
    WindExposed,
    ,
    4,
    0, 0, {z_top:.1f},
    0, 0, {z_bottom:.1f},
    0, {length:.1f}, {z_bottom:.1f},
    0, {length:.1f}, {z_top:.1f};''')

        return "\n".join(surfaces)

    def _generate_windows(
        self,
        floor: int,
        width: float,
        length: float,
        floor_height: float,
        floor_z: float,
        geometry: BuildingGeometry,
    ) -> str:
        """Generate FenestrationSurface:Detailed objects using WWR."""
        zone_name = f"Floor{floor}"
        z_bottom = floor_z
        z_top = floor_z + floor_height

        # Window dimensions: centered on wall, standard height
        window_height = min(1.4, floor_height - 1.0)  # 1.4m or available space
        window_sill = 0.9  # 0.9m sill height
        window_top = window_sill + window_height

        windows = []

        # Get WWR for each orientation from geometry
        wwr = {
            'S': geometry.facades['S'].wwr,
            'N': geometry.facades['N'].wwr,
            'E': geometry.facades['E'].wwr,
            'W': geometry.facades['W'].wwr,
        }

        # South window (on wall at Y=0, wall length = width)
        wall_area_s = width * floor_height
        window_area_s = wall_area_s * wwr['S']
        if window_area_s > 0.5:  # Min window size
            win_width_s = min(window_area_s / window_height, width * 0.8)
            win_offset_s = (width - win_width_s) / 2
            windows.append(f'''
FenestrationSurface:Detailed,
    {zone_name}_Win_S,
    Window,
    Window,
    {zone_name}_Wall_S,
    ,
    ,
    ,
    ,
    4,
    {width - win_offset_s:.1f}, 0, {z_bottom + window_top:.1f},
    {width - win_offset_s:.1f}, 0, {z_bottom + window_sill:.1f},
    {win_offset_s:.1f}, 0, {z_bottom + window_sill:.1f},
    {win_offset_s:.1f}, 0, {z_bottom + window_top:.1f};''')

        # North window (on wall at Y=length, wall length = width)
        wall_area_n = width * floor_height
        window_area_n = wall_area_n * wwr['N']
        if window_area_n > 0.5:
            win_width_n = min(window_area_n / window_height, width * 0.8)
            win_offset_n = (width - win_width_n) / 2
            windows.append(f'''
FenestrationSurface:Detailed,
    {zone_name}_Win_N,
    Window,
    Window,
    {zone_name}_Wall_N,
    ,
    ,
    ,
    ,
    4,
    {win_offset_n:.1f}, {length:.1f}, {z_bottom + window_top:.1f},
    {win_offset_n:.1f}, {length:.1f}, {z_bottom + window_sill:.1f},
    {width - win_offset_n:.1f}, {length:.1f}, {z_bottom + window_sill:.1f},
    {width - win_offset_n:.1f}, {length:.1f}, {z_bottom + window_top:.1f};''')

        # East window (on wall at X=width, wall length = length)
        wall_area_e = length * floor_height
        window_area_e = wall_area_e * wwr['E']
        if window_area_e > 0.5:
            win_width_e = min(window_area_e / window_height, length * 0.8)
            win_offset_e = (length - win_width_e) / 2
            windows.append(f'''
FenestrationSurface:Detailed,
    {zone_name}_Win_E,
    Window,
    Window,
    {zone_name}_Wall_E,
    ,
    ,
    ,
    ,
    4,
    {width:.1f}, {length - win_offset_e:.1f}, {z_bottom + window_top:.1f},
    {width:.1f}, {length - win_offset_e:.1f}, {z_bottom + window_sill:.1f},
    {width:.1f}, {win_offset_e:.1f}, {z_bottom + window_sill:.1f},
    {width:.1f}, {win_offset_e:.1f}, {z_bottom + window_top:.1f};''')

        # West window (on wall at X=0, wall length = length)
        wall_area_w = length * floor_height
        window_area_w = wall_area_w * wwr['W']
        if window_area_w > 0.5:
            win_width_w = min(window_area_w / window_height, length * 0.8)
            win_offset_w = (length - win_width_w) / 2
            windows.append(f'''
FenestrationSurface:Detailed,
    {zone_name}_Win_W,
    Window,
    Window,
    {zone_name}_Wall_W,
    ,
    ,
    ,
    ,
    4,
    0, {win_offset_w:.1f}, {z_bottom + window_top:.1f},
    0, {win_offset_w:.1f}, {z_bottom + window_sill:.1f},
    0, {length - win_offset_w:.1f}, {z_bottom + window_sill:.1f},
    0, {length - win_offset_w:.1f}, {z_bottom + window_top:.1f};''')

        return "\n".join(windows)

    def _generate_internal_loads(
        self,
        zone_name: str,
        floor_area: float,
        archetype: SwedishArchetype,
    ) -> str:
        """Generate People, Lights, ElectricEquipment, Infiltration."""
        # Sveby defaults for Swedish residential
        # Occupancy: ~25 m²/person
        num_people = max(1, floor_area / 25)

        # Lighting: ~8 W/m² (Sveby)
        lighting_w_m2 = 8

        # Equipment: ~6 W/m² (Sveby - excludes cooking)
        equipment_w_m2 = 6

        # Infiltration from archetype
        infiltration_ach = archetype.envelope.infiltration_ach

        # FTX fan power: SFP 1.5 kW/(m³/s), 0.35 l/s/m² = 0.00035 m³/s/m²
        # Fan power = floor_area * 0.00035 * 1500 = floor_area * 0.525 W/m²
        # Per zone: floor_area * 0.525 W
        sfp = archetype.hvac.sfp_kw_per_m3s
        ventilation_rate = 0.00035  # m³/s/m²
        fan_power = floor_area * ventilation_rate * sfp * 1000  # W

        return f'''
People,
    {zone_name}_People,
    {zone_name},
    OccupancySchedule,
    People,
    {num_people:.0f},
    ,
    ,
    0.3,                     !- Fraction Radiant
    ,
    ActivityLevel;

Lights,
    {zone_name}_Lights,
    {zone_name},
    LightingSchedule,
    Watts/Area,
    ,
    {lighting_w_m2},
    ,
    0.2,                     !- Return Air Fraction
    0.6,                     !- Fraction Radiant
    0.2;                     !- Fraction Visible

ElectricEquipment,
    {zone_name}_Equipment,
    {zone_name},
    EquipmentSchedule,
    Watts/Area,
    ,
    {equipment_w_m2},
    ,
    0,                       !- Fraction Latent
    0.3,                     !- Fraction Radiant
    0;                       !- Fraction Lost

ZoneInfiltration:DesignFlowRate,
    {zone_name}_Infiltration,
    {zone_name},
    AlwaysOn,
    AirChanges/Hour,
    ,
    ,
    ,
    {infiltration_ach};

ElectricEquipment,
    {zone_name}_FTX_Fan,
    {zone_name},
    FTX_FanSchedule,
    EquipmentLevel,
    {fan_power:.0f},
    ,
    ,
    0,
    0.5,                     !- Fraction Radiant
    0,
    FTX_Fans;
'''

    def _generate_hvac(
        self,
        zone_name: str,
        archetype: SwedishArchetype,
    ) -> str:
        """Generate IdealLoadsAirSystem and connections for zone."""
        # Heat recovery settings
        hr_eff = archetype.hvac.heat_recovery_efficiency
        hr_type = "Sensible" if hr_eff > 0 else "None"

        return f'''
ZoneHVAC:IdealLoadsAirSystem,
    {zone_name}_IdealLoads,
    ,                        !- Availability Schedule Name
    {zone_name}_Supply,      !- Zone Supply Air Node Name
    {zone_name}_Exhaust,     !- Zone Exhaust Air Node Name
    ,                        !- System Inlet Air Node Name
    50,                      !- Maximum Heating Supply Air Temperature
    13,                      !- Minimum Cooling Supply Air Temperature
    0.015,                   !- Maximum Heating Supply Air Humidity Ratio
    0.009,                   !- Minimum Cooling Supply Air Humidity Ratio
    NoLimit,                 !- Heating Limit
    autosize,                !- Maximum Heating Air Flow Rate
    ,                        !- Maximum Sensible Heating Capacity
    NoLimit,                 !- Cooling Limit
    autosize,                !- Maximum Cooling Air Flow Rate
    ,                        !- Maximum Total Cooling Capacity
    ,                        !- Heating Availability Schedule Name
    ,                        !- Cooling Availability Schedule Name
    ConstantSupplyHumidityRatio, !- Dehumidification Control Type (NOT None!)
    ,                        !- Cooling Sensible Heat Ratio (BLANK!)
    ConstantSupplyHumidityRatio, !- Humidification Control Type (NOT None!)
    OA_Spec,                 !- Design Specification Outdoor Air Object Name
    {zone_name}_OA,          !- Outdoor Air Inlet Node Name
    None,                    !- Demand Controlled Ventilation Type
    NoEconomizer,            !- Outdoor Air Economizer Type
    {hr_type},               !- Heat Recovery Type
    {hr_eff},                !- Sensible Heat Recovery Effectiveness
    0.0;                     !- Latent Heat Recovery Effectiveness

ZoneHVAC:EquipmentList,
    {zone_name}_EquipList,
    SequentialLoad,
    ZoneHVAC:IdealLoadsAirSystem,
    {zone_name}_IdealLoads,
    1,
    1;

ZoneHVAC:EquipmentConnections,
    {zone_name},
    {zone_name}_EquipList,
    {zone_name}_Supply,
    {zone_name}_Exhaust,
    {zone_name}_AirNode,
    {zone_name}_Return;

ZoneControl:Thermostat,
    {zone_name}_Thermostat,
    {zone_name},
    ThermType,
    ThermostatSetpoint:DualSetpoint,
    {zone_name}_DualSetpoint;

ThermostatSetpoint:DualSetpoint,
    {zone_name}_DualSetpoint,
    HeatSet,
    CoolSet;
'''

    def _generate_outputs(self) -> str:
        """Generate Output:Variable and reporting objects."""
        return '''
!- ============================================================
!- OUTPUT VARIABLES
!- ============================================================

Output:Variable,*,Zone Ideal Loads Zone Total Heating Energy,Hourly;
Output:Variable,*,Zone Ideal Loads Zone Total Cooling Energy,Hourly;
Output:Variable,*,Zone Mean Air Temperature,Hourly;
Output:Meter,Heating:EnergyTransfer,Hourly;
Output:Meter,InteriorEquipment:Electricity,Hourly;
Output:Meter,InteriorLights:Electricity,Hourly;
OutputControl:Table:Style,Comma,JtoKWH;
Output:Table:SummaryReports,AllSummary;
'''

    def _estimate_heating(
        self,
        geometry: BuildingGeometry,
        archetype: SwedishArchetype,
    ) -> float:
        """Rough estimate of annual heating demand in kWh/m²."""
        # Simple degree-day based estimation
        # Stockholm: ~3500 heating degree days (base 17°C)
        hdd = 3500

        # Calculate UA value
        wall_ua = geometry.total_wall_area_m2 * archetype.envelope.wall_u_value
        window_ua = geometry.total_window_area_m2 * archetype.envelope.window_u_value
        roof_ua = geometry.roof.total_area_m2 * archetype.envelope.roof_u_value
        floor_ua = geometry.footprint_area_m2 * archetype.envelope.floor_u_value

        # Infiltration UA equivalent
        volume = geometry.volume_m3
        inf_ach = archetype.envelope.infiltration_ach
        inf_ua = volume * inf_ach * 0.34  # 0.34 Wh/m³K

        # Ventilation losses (reduced by heat recovery)
        vent_rate = geometry.gross_floor_area_m2 * 0.35 / 1000  # m³/s
        hr_eff = archetype.hvac.heat_recovery_efficiency
        vent_ua = vent_rate * 1200 * (1 - hr_eff)  # W/K

        total_ua = wall_ua + window_ua + roof_ua + floor_ua + inf_ua + vent_ua

        # Annual heating = UA * HDD * 24 / 1000 (kWh)
        heating_kwh = total_ua * hdd * 24 / 1000

        # Internal gains offset (roughly 15 kWh/m²)
        internal_gains = geometry.gross_floor_area_m2 * 15

        net_heating = max(0, heating_kwh - internal_gains)

        return net_heating / geometry.gross_floor_area_m2

    def _select_weather_file(self, latitude: float = 59.3) -> str:
        """Select appropriate Swedish weather file based on latitude."""
        if latitude >= 66:
            return "SWE_Kiruna.020440_IWEC.epw"
        elif latitude >= 63:
            return "SWE_Ostersund.022220_IWEC.epw"
        elif latitude >= 59:
            return "SWE_Stockholm.Arlanda.024600_IWEC.epw"
        elif latitude >= 57:
            return "SWE_Goteborg.Landvetter.025260_IWEC.epw"
        else:
            return "SWE_Malmo.Sturup.026400_IWEC.epw"


def generate_baseline(
    geometry: BuildingGeometry,
    archetype: SwedishArchetype,
    output_dir: Path,
    model_name: str = "baseline",
    **kwargs,
) -> BaselineModel:
    """
    Convenience function to generate baseline IDF.

    Args:
        geometry: Building geometry from BuildingGeometryCalculator
        archetype: Matched Swedish archetype
        output_dir: Output directory for IDF
        model_name: Model name
        **kwargs: Additional args for BaselineGenerator.generate()

    Returns:
        BaselineModel with IDF path
    """
    generator = BaselineGenerator()
    return generator.generate(geometry, archetype, output_dir, model_name, **kwargs)
