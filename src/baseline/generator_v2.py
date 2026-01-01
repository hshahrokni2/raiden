"""
GeomEppy-based IDF Generator - Accurate polygon geometry.

Unlike the original generator that creates rectangular approximations,
this version uses GeomEppy to build IDF models from actual polygon
footprints (from OSM/Overture).

Benefits:
- Preserves actual building shape
- Accurate wall orientations for solar calculations
- Better window-to-wall ratio distribution
- Supports L-shaped, U-shaped, and complex footprints

Usage:
    from src.baseline.generator_v2 import GeomEppyGenerator

    generator = GeomEppyGenerator()
    model = generator.generate(
        footprint_coords=[(0,0), (10,0), (10,20), (0,20)],
        floors=4,
        archetype=swedish_archetype,
        output_dir=Path('./output'),
    )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any, TYPE_CHECKING
from pathlib import Path
import logging
import math
import tempfile
import shutil

import numpy as np
from shapely.geometry import Polygon

from .archetypes import SwedishArchetype
from .generator import BaselineModel

logger = logging.getLogger(__name__)

# Check for GeomEppy availability
GEOMEPPY_AVAILABLE = False
IDF = None

try:
    # Apply Python 3.11+ compatibility patches BEFORE importing geomeppy
    from ..compat import patch_all
    patch_all()

    from geomeppy import IDF
    GEOMEPPY_AVAILABLE = True
    logger.info("GeomEppy loaded successfully")
except ImportError as e:
    logger.warning(f"GeomEppy not available: {e}. Install with: pip install geomeppy")

if TYPE_CHECKING:
    from geomeppy import IDF


@dataclass
class WallSegment:
    """A wall segment with orientation and area."""
    start: Tuple[float, float]
    end: Tuple[float, float]
    length: float
    azimuth: float  # 0=N, 90=E, 180=S, 270=W
    cardinal: str  # N, NE, E, SE, S, SW, W, NW


@dataclass
class FloorPlan:
    """Analyzed floor plan with wall segments."""
    polygon: Polygon
    area: float
    perimeter: float
    walls: List[WallSegment]
    centroid: Tuple[float, float]


def _azimuth_to_cardinal(azimuth: float) -> str:
    """Convert azimuth (0=N, clockwise) to cardinal direction."""
    # Normalize to 0-360
    azimuth = azimuth % 360

    if azimuth < 22.5 or azimuth >= 337.5:
        return "N"
    elif azimuth < 67.5:
        return "NE"
    elif azimuth < 112.5:
        return "E"
    elif azimuth < 157.5:
        return "SE"
    elif azimuth < 202.5:
        return "S"
    elif azimuth < 247.5:
        return "SW"
    elif azimuth < 292.5:
        return "W"
    else:
        return "NW"


def analyze_footprint(coords: List[Tuple[float, float]]) -> FloorPlan:
    """
    Analyze a footprint polygon to extract wall segments with orientations.

    Args:
        coords: List of (x, y) coordinates in meters, counterclockwise

    Returns:
        FloorPlan with wall segment analysis
    """
    polygon = Polygon(coords)

    # Ensure counterclockwise
    if not polygon.exterior.is_ccw:
        coords = list(reversed(coords))
        polygon = Polygon(coords)

    walls = []
    n = len(coords)

    for i in range(n):
        start = coords[i]
        end = coords[(i + 1) % n]

        # Calculate wall properties
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        length = math.sqrt(dx**2 + dy**2)

        if length < 0.1:  # Skip tiny segments
            continue

        # Wall normal faces outward (90° clockwise from wall direction for CCW polygon)
        # Wall direction: (dx, dy)
        # Outward normal: (dy, -dx) normalized
        # Azimuth: angle from Y-axis (north) clockwise
        wall_azimuth = math.degrees(math.atan2(dy, -dx)) % 360

        walls.append(WallSegment(
            start=start,
            end=end,
            length=length,
            azimuth=wall_azimuth,
            cardinal=_azimuth_to_cardinal(wall_azimuth),
        ))

    return FloorPlan(
        polygon=polygon,
        area=polygon.area,
        perimeter=polygon.length,
        walls=walls,
        centroid=(polygon.centroid.x, polygon.centroid.y),
    )


class GeomEppyGenerator:
    """
    Generate EnergyPlus IDF using GeomEppy for accurate geometry.

    Workflow:
    1. Load minimal IDF template with schedules/materials
    2. Use GeomEppy to create zones from polygon footprint
    3. Add windows based on WWR per orientation
    4. Configure HVAC using archetype properties
    5. Save complete IDF
    """

    def __init__(self, energyplus_dir: Optional[Path] = None):
        """
        Initialize generator.

        Args:
            energyplus_dir: Path to EnergyPlus installation (for IDD file)
        """
        if not GEOMEPPY_AVAILABLE:
            raise ImportError(
                "GeomEppy not available. Install with: pip install geomeppy"
            )

        # Find EnergyPlus IDD
        self.idd_path = self._find_idd(energyplus_dir)
        if self.idd_path:
            IDF.setiddname(str(self.idd_path))

    def _find_idd(self, energyplus_dir: Optional[Path] = None) -> Optional[Path]:
        """Find EnergyPlus IDD file."""
        search_paths = []

        if energyplus_dir:
            search_paths.append(energyplus_dir / "Energy+.idd")

        # Common installation paths
        search_paths.extend([
            Path("/usr/local/EnergyPlus-25-1-0/Energy+.idd"),
            Path("/Applications/EnergyPlus-25-1-0/Energy+.idd"),
            Path("C:/EnergyPlusV25-1-0/Energy+.idd"),
        ])

        for path in search_paths:
            if path.exists():
                return path

        logger.warning("Could not find EnergyPlus IDD file")
        return None

    def generate(
        self,
        footprint_coords: List[Tuple[float, float]],
        floors: int,
        archetype: SwedishArchetype,
        output_dir: Path,
        model_name: str = "baseline_v2",
        floor_height: float = 2.8,
        wwr_per_orientation: Optional[Dict[str, float]] = None,
        latitude: float = 59.35,
        longitude: float = 17.95,
    ) -> BaselineModel:
        """
        Generate IDF from polygon footprint using GeomEppy.

        Args:
            footprint_coords: List of (x, y) coordinates in meters
            floors: Number of floors
            archetype: Swedish building archetype
            output_dir: Output directory
            model_name: Model name
            floor_height: Height per floor (m)
            wwr_per_orientation: WWR dict like {'N': 0.15, 'S': 0.25, 'E': 0.20, 'W': 0.20}
            latitude: Site latitude
            longitude: Site longitude

        Returns:
            BaselineModel with generated IDF path
        """
        logger.info(f"Generating GeomEppy model: {model_name}")

        # Analyze footprint
        floor_plan = analyze_footprint(footprint_coords)
        logger.info(f"  Footprint area: {floor_plan.area:.1f} m²")
        logger.info(f"  Perimeter: {floor_plan.perimeter:.1f} m")
        logger.info(f"  Wall segments: {len(floor_plan.walls)}")

        # Default WWR if not provided
        if wwr_per_orientation is None:
            wwr_per_orientation = {'N': 0.15, 'S': 0.25, 'E': 0.20, 'W': 0.20}

        # Create IDF
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        idf_path = output_dir / f"{model_name}.idf"

        # Start with minimal IDF
        idf = self._create_minimal_idf()

        # Set site location
        self._set_location(idf, latitude, longitude)

        # Add materials and constructions from archetype
        self._add_materials(idf, archetype)

        # Add schedules
        self._add_schedules(idf)

        # Build geometry using GeomEppy
        self._build_geometry(
            idf=idf,
            floor_plan=floor_plan,
            floors=floors,
            floor_height=floor_height,
            archetype=archetype,
        )

        # Add windows
        self._add_windows(
            idf=idf,
            floor_plan=floor_plan,
            floors=floors,
            floor_height=floor_height,
            wwr_per_orientation=wwr_per_orientation,
        )

        # Add internal loads and HVAC
        for floor in range(1, floors + 1):
            zone_name = f"Floor{floor}"
            self._add_internal_loads(idf, zone_name, floor_plan.area, archetype)
            self._add_hvac(idf, zone_name, archetype)

        # Add outputs
        self._add_outputs(idf)

        # Save
        idf.saveas(str(idf_path))
        logger.info(f"Saved IDF to {idf_path}")

        # Estimate heating
        gross_area = floor_plan.area * floors
        estimated_heating = self._estimate_heating(floor_plan, floors, archetype)

        return BaselineModel(
            idf_path=idf_path,
            weather_file=self._select_weather_file(latitude),
            archetype_used=archetype.name,
            floor_area_m2=gross_area,
            predicted_heating_kwh_m2=estimated_heating,
        )

    def _newidfobject(self, idf: IDF, objtype: str, **kwargs):
        """Helper to create IDF objects with uppercase type (GeomEppy requirement)."""
        return idf.newidfobject(objtype.upper(), **kwargs)

    def _create_minimal_idf(self) -> IDF:
        """Create minimal IDF with version and simulation control."""
        # Create empty IDF
        idf = IDF()
        idf.initnew(fname="minimal.idf")

        # Version - Note: GeomEppy requires ALL_CAPS for object types
        self._newidfobject(idf, "VERSION", Version_Identifier="25.1")

        # Building
        self._newidfobject(
            idf, "BUILDING",
            Name="GeomEppyBuilding",
            North_Axis=0,
            Terrain="City",
            Loads_Convergence_Tolerance_Value=0.04,
            Temperature_Convergence_Tolerance_Value=0.4,
            Solar_Distribution="FullExterior",
        )

        # Timestep
        self._newidfobject(idf, "TIMESTEP", Number_of_Timesteps_per_Hour=4)

        # Simulation control
        self._newidfobject(
            idf, "SIMULATIONCONTROL",
            Do_Zone_Sizing_Calculation="No",
            Do_System_Sizing_Calculation="No",
            Do_Plant_Sizing_Calculation="No",
            Run_Simulation_for_Sizing_Periods="No",
            Run_Simulation_for_Weather_File_Run_Periods="Yes",
        )

        # Run period
        self._newidfobject(
            idf, "RUNPERIOD",
            Name="Annual",
            Begin_Month=1,
            Begin_Day_of_Month=1,
            End_Month=12,
            End_Day_of_Month=31,
            Day_of_Week_for_Start_Day="Sunday",
        )

        # Ground temperature
        self._newidfobject(idf, 
            "Site:GroundTemperature:BuildingSurface",
            January_Ground_Temperature=5,
            February_Ground_Temperature=5,
            March_Ground_Temperature=6,
            April_Ground_Temperature=8,
            May_Ground_Temperature=11,
            June_Ground_Temperature=14,
            July_Ground_Temperature=16,
            August_Ground_Temperature=16,
            September_Ground_Temperature=14,
            October_Ground_Temperature=11,
            November_Ground_Temperature=8,
            December_Ground_Temperature=5,
        )

        # GlobalGeometryRules
        self._newidfobject(idf, 
            "GlobalGeometryRules",
            Starting_Vertex_Position="UpperLeftCorner",
            Vertex_Entry_Direction="Counterclockwise",
            Coordinate_System="Relative",
        )

        return idf

    def _set_location(self, idf: IDF, latitude: float, longitude: float):
        """Set site location."""
        self._newidfobject(idf, 
            "Site:Location",
            Name="Building_Location",
            Latitude=latitude,
            Longitude=longitude,
            Time_Zone=1.0,
            Elevation=44,
        )

    def _add_materials(self, idf: IDF, archetype: SwedishArchetype):
        """Add materials and constructions from archetype."""
        wall_u = archetype.envelope.wall_u_value
        roof_u = archetype.envelope.roof_u_value
        floor_u = archetype.envelope.floor_u_value
        window_u = archetype.envelope.window_u_value
        window_shgc = archetype.envelope.window_shgc

        # Calculate insulation thicknesses
        k_insulation = 0.035
        r_surface = 0.21
        r_concrete = 0.2

        def calc_ins_thick(u_val):
            r_req = 1.0 / u_val
            r_ins = max(0.05, r_req - r_surface - r_concrete)
            return r_ins * k_insulation

        # Concrete
        self._newidfobject(idf, 
            "Material",
            Name="Concrete200",
            Roughness="MediumRough",
            Thickness=0.2,
            Conductivity=1.0,
            Density=2300,
            Specific_Heat=880,
        )

        # Wall insulation
        self._newidfobject(idf, 
            "Material",
            Name="WallInsulation",
            Roughness="MediumRough",
            Thickness=calc_ins_thick(wall_u),
            Conductivity=0.035,
            Density=30,
            Specific_Heat=840,
        )

        # Roof insulation
        self._newidfobject(idf, 
            "Material",
            Name="RoofInsulation",
            Roughness="MediumRough",
            Thickness=calc_ins_thick(roof_u),
            Conductivity=0.035,
            Density=30,
            Specific_Heat=840,
        )

        # Floor insulation
        self._newidfobject(idf, 
            "Material",
            Name="FloorInsulation",
            Roughness="MediumRough",
            Thickness=calc_ins_thick(floor_u),
            Conductivity=0.035,
            Density=30,
            Specific_Heat=840,
        )

        # Constructions
        self._newidfobject(idf, 
            "Construction",
            Name="ExteriorWall",
            Outside_Layer="Concrete200",
            Layer_2="WallInsulation",
        )

        self._newidfobject(idf, 
            "Construction",
            Name="Roof",
            Outside_Layer="Concrete200",
            Layer_2="RoofInsulation",
        )

        self._newidfobject(idf, 
            "Construction",
            Name="GroundFloor",
            Outside_Layer="Concrete200",
            Layer_2="FloorInsulation",
        )

        self._newidfobject(idf, 
            "Construction",
            Name="InteriorFloor",
            Outside_Layer="Concrete200",
        )

        # Window
        self._newidfobject(idf, 
            "WindowMaterial:SimpleGlazingSystem",
            Name="GlazingSystem",
            UFactor=window_u,
            Solar_Heat_Gain_Coefficient=window_shgc,
        )

        self._newidfobject(idf, 
            "Construction",
            Name="Window",
            Outside_Layer="GlazingSystem",
        )

    def _add_schedules(self, idf: IDF):
        """Add Sveby-based schedules."""
        # Schedule type limits (Numeric_Type required in EnergyPlus 25.1)
        self._newidfobject(idf,
            "ScheduleTypeLimits",
            Name="Fraction",
            Lower_Limit_Value=0,
            Upper_Limit_Value=1,
            Numeric_Type="Continuous",
        )

        self._newidfobject(idf,
            "ScheduleTypeLimits",
            Name="Temperature",
            Lower_Limit_Value=-50,
            Upper_Limit_Value=50,
            Numeric_Type="Continuous",
        )

        self._newidfobject(idf,
            "ScheduleTypeLimits",
            Name="Any Number",
            Numeric_Type="Continuous",
        )

        # Constant schedules
        for name, value, stype in [
            ("AlwaysOn", 1, "Fraction"),
            ("HeatSet", 21, "Temperature"),
            ("CoolSet", 50, "Temperature"),
            ("ThermType", 4, "Any Number"),
            ("ActivityLevel", 120, "Any Number"),
        ]:
            self._newidfobject(idf, 
                "Schedule:Constant",
                Name=name,
                Schedule_Type_Limits_Name=stype,
                Hourly_Value=value,
            )

        # OA spec
        self._newidfobject(idf, 
            "DesignSpecification:OutdoorAir",
            Name="OA_Spec",
            Outdoor_Air_Method="Flow/Area",
            Outdoor_Air_Flow_per_Zone_Floor_Area=0.00035,
        )

    def _build_geometry(
        self,
        idf: IDF,
        floor_plan: FloorPlan,
        floors: int,
        floor_height: float,
        archetype: SwedishArchetype,
    ):
        """Build zone and surface geometry using GeomEppy."""
        coords = list(floor_plan.polygon.exterior.coords)[:-1]  # Remove duplicate closing point

        for floor in range(1, floors + 1):
            zone_name = f"Floor{floor}"
            z_base = (floor - 1) * floor_height

            # Create zone
            self._newidfobject(idf, 
                "Zone",
                Name=zone_name,
                Direction_of_Relative_North=0,
                X_Origin=0,
                Y_Origin=0,
                Z_Origin=z_base,
                Type=1,
                Multiplier=1,
                Ceiling_Height=floor_height,
                Volume=floor_plan.area * floor_height,
            )

            # Create surfaces for this zone
            self._create_floor_surfaces(
                idf=idf,
                zone_name=zone_name,
                coords=coords,
                floor_height=floor_height,
                z_base=z_base,
                is_ground_floor=(floor == 1),
                is_top_floor=(floor == floors),
                floor_above=f"Floor{floor+1}" if floor < floors else None,
                floor_below=f"Floor{floor-1}" if floor > 1 else None,
            )

            # Add outdoor air node
            self._newidfobject(idf, "OutdoorAir:NodeList", Node_or_NodeList_Name_1=f"{zone_name}_OA")

    def _create_floor_surfaces(
        self,
        idf: IDF,
        zone_name: str,
        coords: List[Tuple[float, float]],
        floor_height: float,
        z_base: float,
        is_ground_floor: bool,
        is_top_floor: bool,
        floor_above: Optional[str],
        floor_below: Optional[str],
    ):
        """Create floor, ceiling, and wall surfaces for a zone."""
        z_top = z_base + floor_height
        n = len(coords)

        # Floor surface
        floor_vertices = [(x, y, z_base) for x, y in coords]
        floor_vertices.reverse()  # Floor faces down

        if is_ground_floor:
            boundary = "Ground"
            boundary_obj = ""
            construction = "GroundFloor"
        else:
            boundary = "Surface"
            boundary_obj = f"{floor_below}_Ceiling"
            construction = "InteriorFloor"

        self._add_surface(
            idf=idf,
            name=f"{zone_name}_Floor",
            surface_type="Floor",
            construction=construction,
            zone=zone_name,
            boundary=boundary,
            boundary_obj=boundary_obj,
            vertices=floor_vertices,
            sun_exposed=False,
        )

        # Ceiling surface
        ceiling_vertices = [(x, y, z_top) for x, y in coords]

        if is_top_floor:
            boundary = "Outdoors"
            boundary_obj = ""
            construction = "Roof"
            surface_type = "Roof"
            sun_exposed = True
        else:
            boundary = "Surface"
            boundary_obj = f"{floor_above}_Floor"
            construction = "InteriorFloor"
            surface_type = "Ceiling"
            sun_exposed = False

        self._add_surface(
            idf=idf,
            name=f"{zone_name}_Ceiling",
            surface_type=surface_type,
            construction=construction,
            zone=zone_name,
            boundary=boundary,
            boundary_obj=boundary_obj,
            vertices=ceiling_vertices,
            sun_exposed=sun_exposed,
        )

        # Wall surfaces
        for i in range(n):
            start = coords[i]
            end = coords[(i + 1) % n]

            # Wall vertices (counterclockwise when viewed from outside)
            wall_vertices = [
                (start[0], start[1], z_top),
                (start[0], start[1], z_base),
                (end[0], end[1], z_base),
                (end[0], end[1], z_top),
            ]

            # Determine cardinal direction
            dx = end[0] - start[0]
            dy = end[1] - start[1]
            azimuth = math.degrees(math.atan2(dy, -dx)) % 360
            cardinal = _azimuth_to_cardinal(azimuth)

            self._add_surface(
                idf=idf,
                name=f"{zone_name}_Wall_{cardinal}_{i}",
                surface_type="Wall",
                construction="ExteriorWall",
                zone=zone_name,
                boundary="Outdoors",
                boundary_obj="",
                vertices=wall_vertices,
                sun_exposed=True,
            )

    def _add_surface(
        self,
        idf: IDF,
        name: str,
        surface_type: str,
        construction: str,
        zone: str,
        boundary: str,
        boundary_obj: str,
        vertices: List[Tuple[float, float, float]],
        sun_exposed: bool = True,
    ):
        """Add a BuildingSurface:Detailed object."""
        obj = self._newidfobject(idf, 
            "BuildingSurface:Detailed",
            Name=name,
            Surface_Type=surface_type,
            Construction_Name=construction,
            Zone_Name=zone,
            Outside_Boundary_Condition=boundary,
            Outside_Boundary_Condition_Object=boundary_obj,
            Sun_Exposure="SunExposed" if sun_exposed else "NoSun",
            Wind_Exposure="WindExposed" if sun_exposed else "NoWind",
            Number_of_Vertices=len(vertices),
        )

        # Set vertices
        for i, (x, y, z) in enumerate(vertices):
            setattr(obj, f"Vertex_{i+1}_Xcoordinate", x)
            setattr(obj, f"Vertex_{i+1}_Ycoordinate", y)
            setattr(obj, f"Vertex_{i+1}_Zcoordinate", z)

    def _add_windows(
        self,
        idf: IDF,
        floor_plan: FloorPlan,
        floors: int,
        floor_height: float,
        wwr_per_orientation: Dict[str, float],
    ):
        """Add windows to walls based on orientation and WWR."""
        # Standard window dimensions
        window_height = min(1.4, floor_height - 1.0)
        sill_height = 0.9

        coords = list(floor_plan.polygon.exterior.coords)[:-1]
        n = len(coords)

        for floor in range(1, floors + 1):
            zone_name = f"Floor{floor}"
            z_base = (floor - 1) * floor_height

            for i in range(n):
                start = coords[i]
                end = coords[(i + 1) % n]

                # Wall properties
                dx = end[0] - start[0]
                dy = end[1] - start[1]
                length = math.sqrt(dx**2 + dy**2)

                if length < 1.0:  # Skip short walls
                    continue

                azimuth = math.degrees(math.atan2(dy, -dx)) % 360
                cardinal = _azimuth_to_cardinal(azimuth)

                # Get WWR for this orientation (simplify to 4 cardinals)
                simple_cardinal = cardinal[0] if len(cardinal) == 2 else cardinal
                if simple_cardinal not in wwr_per_orientation:
                    simple_cardinal = 'S'  # Default
                wwr = wwr_per_orientation.get(simple_cardinal, 0.2)

                # Calculate window dimensions
                wall_area = length * floor_height
                window_area = wall_area * wwr

                if window_area < 0.5:  # Skip tiny windows
                    continue

                window_width = min(window_area / window_height, length * 0.8)

                # Window center point
                t = 0.5  # Center of wall
                cx = start[0] + t * dx
                cy = start[1] + t * dy

                # Window vertices (perpendicular to wall)
                half_w = window_width / 2
                unit_dx = dx / length
                unit_dy = dy / length

                z_sill = z_base + sill_height
                z_head = z_sill + window_height

                # Window corners along wall direction
                window_vertices = [
                    (cx - half_w * unit_dx, cy - half_w * unit_dy, z_head),
                    (cx - half_w * unit_dx, cy - half_w * unit_dy, z_sill),
                    (cx + half_w * unit_dx, cy + half_w * unit_dy, z_sill),
                    (cx + half_w * unit_dx, cy + half_w * unit_dy, z_head),
                ]

                # Add window
                wall_name = f"{zone_name}_Wall_{cardinal}_{i}"
                self._add_window(
                    idf=idf,
                    name=f"{zone_name}_Win_{cardinal}_{i}",
                    wall_name=wall_name,
                    vertices=window_vertices,
                )

    def _add_window(
        self,
        idf: IDF,
        name: str,
        wall_name: str,
        vertices: List[Tuple[float, float, float]],
    ):
        """Add a FenestrationSurface:Detailed object."""
        obj = self._newidfobject(idf, 
            "FenestrationSurface:Detailed",
            Name=name,
            Surface_Type="Window",
            Construction_Name="Window",
            Building_Surface_Name=wall_name,
            Number_of_Vertices=len(vertices),
        )

        for i, (x, y, z) in enumerate(vertices):
            setattr(obj, f"Vertex_{i+1}_Xcoordinate", x)
            setattr(obj, f"Vertex_{i+1}_Ycoordinate", y)
            setattr(obj, f"Vertex_{i+1}_Zcoordinate", z)

    def _add_internal_loads(
        self,
        idf: IDF,
        zone_name: str,
        floor_area: float,
        archetype: SwedishArchetype,
    ):
        """Add People, Lights, Equipment, Infiltration."""
        # Sveby defaults
        num_people = max(1, floor_area / 25)
        lighting_w_m2 = 8
        equipment_w_m2 = 6
        infiltration_ach = archetype.envelope.infiltration_ach

        # People
        self._newidfobject(idf, 
            "People",
            Name=f"{zone_name}_People",
            Zone_or_ZoneList_or_Space_or_SpaceList_Name=zone_name,
            Number_of_People_Schedule_Name="AlwaysOn",
            Number_of_People_Calculation_Method="People",
            Number_of_People=num_people,
            Fraction_Radiant=0.3,
            Activity_Level_Schedule_Name="ActivityLevel",
        )

        # Lights
        self._newidfobject(idf, 
            "Lights",
            Name=f"{zone_name}_Lights",
            Zone_or_ZoneList_or_Space_or_SpaceList_Name=zone_name,
            Schedule_Name="AlwaysOn",
            Design_Level_Calculation_Method="Watts/Area",
            Watts_per_Floor_Area=lighting_w_m2,
            Return_Air_Fraction=0.2,
            Fraction_Radiant=0.6,
            Fraction_Visible=0.2,
        )

        # Equipment
        self._newidfobject(idf, 
            "ElectricEquipment",
            Name=f"{zone_name}_Equipment",
            Zone_or_ZoneList_or_Space_or_SpaceList_Name=zone_name,
            Schedule_Name="AlwaysOn",
            Design_Level_Calculation_Method="Watts/Area",
            Watts_per_Floor_Area=equipment_w_m2,
            Fraction_Latent=0,
            Fraction_Radiant=0.3,
            Fraction_Lost=0,
        )

        # Infiltration
        self._newidfobject(idf, 
            "ZoneInfiltration:DesignFlowRate",
            Name=f"{zone_name}_Infiltration",
            Zone_or_ZoneList_or_Space_or_SpaceList_Name=zone_name,
            Schedule_Name="AlwaysOn",
            Design_Flow_Rate_Calculation_Method="AirChanges/Hour",
            Air_Changes_per_Hour=infiltration_ach,
        )

    def _add_hvac(self, idf: IDF, zone_name: str, archetype: SwedishArchetype):
        """Add IdealLoadsAirSystem with heat recovery."""
        hr_eff = archetype.hvac.heat_recovery_efficiency
        hr_type = "Sensible" if hr_eff > 0 else "None"

        # IdealLoads (EnergyPlus 25.1 bug: Cooling_Sensible_Heat_Ratio must be BLANK
        # when using ConstantSupplyHumidityRatio, otherwise segfault)
        self._newidfobject(idf,
            "ZoneHVAC:IdealLoadsAirSystem",
            Name=f"{zone_name}_IdealLoads",
            Zone_Supply_Air_Node_Name=f"{zone_name}_Supply",
            Zone_Exhaust_Air_Node_Name=f"{zone_name}_Exhaust",
            Maximum_Heating_Supply_Air_Temperature=50,
            Minimum_Cooling_Supply_Air_Temperature=13,
            Maximum_Heating_Supply_Air_Humidity_Ratio=0.015,
            Minimum_Cooling_Supply_Air_Humidity_Ratio=0.009,
            Heating_Limit="NoLimit",
            Cooling_Limit="NoLimit",
            Dehumidification_Control_Type="ConstantSupplyHumidityRatio",
            Cooling_Sensible_Heat_Ratio="",
            Humidification_Control_Type="ConstantSupplyHumidityRatio",
            Design_Specification_Outdoor_Air_Object_Name="OA_Spec",
            Outdoor_Air_Inlet_Node_Name=f"{zone_name}_OA",
            Demand_Controlled_Ventilation_Type="None",
            Outdoor_Air_Economizer_Type="NoEconomizer",
            Heat_Recovery_Type=hr_type,
            Sensible_Heat_Recovery_Effectiveness=hr_eff,
            Latent_Heat_Recovery_Effectiveness=0,
        )

        # Equipment list
        self._newidfobject(idf, 
            "ZoneHVAC:EquipmentList",
            Name=f"{zone_name}_EquipList",
            Load_Distribution_Scheme="SequentialLoad",
            Zone_Equipment_1_Object_Type="ZoneHVAC:IdealLoadsAirSystem",
            Zone_Equipment_1_Name=f"{zone_name}_IdealLoads",
            Zone_Equipment_1_Cooling_Sequence=1,
            Zone_Equipment_1_Heating_or_NoLoad_Sequence=1,
        )

        # Equipment connections
        self._newidfobject(idf, 
            "ZoneHVAC:EquipmentConnections",
            Zone_Name=zone_name,
            Zone_Conditioning_Equipment_List_Name=f"{zone_name}_EquipList",
            Zone_Air_Inlet_Node_or_NodeList_Name=f"{zone_name}_Supply",
            Zone_Air_Exhaust_Node_or_NodeList_Name=f"{zone_name}_Exhaust",
            Zone_Air_Node_Name=f"{zone_name}_AirNode",
            Zone_Return_Air_Node_or_NodeList_Name=f"{zone_name}_Return",
        )

        # Thermostat
        self._newidfobject(idf,
            "ZoneControl:Thermostat",
            Name=f"{zone_name}_Thermostat",
            Zone_or_ZoneList_Name=zone_name,
            Control_Type_Schedule_Name="ThermType",
            Control_1_Object_Type="ThermostatSetpoint:DualSetpoint",
            Control_1_Name=f"{zone_name}_DualSetpoint",
        )

        self._newidfobject(idf, 
            "ThermostatSetpoint:DualSetpoint",
            Name=f"{zone_name}_DualSetpoint",
            Heating_Setpoint_Temperature_Schedule_Name="HeatSet",
            Cooling_Setpoint_Temperature_Schedule_Name="CoolSet",
        )

    def _add_outputs(self, idf: IDF):
        """Add output variables."""
        for var in [
            "Zone Ideal Loads Zone Total Heating Energy",
            "Zone Ideal Loads Zone Total Cooling Energy",
            "Zone Mean Air Temperature",
        ]:
            self._newidfobject(idf, 
                "Output:Variable",
                Key_Value="*",
                Variable_Name=var,
                Reporting_Frequency="Hourly",
            )

        self._newidfobject(idf, 
            "Output:Meter",
            Key_Name="Heating:EnergyTransfer",
            Reporting_Frequency="Hourly",
        )

        self._newidfobject(idf, 
            "OutputControl:Table:Style",
            Column_Separator="Comma",
            Unit_Conversion="JtoKWH",
        )

        self._newidfobject(idf, 
            "Output:Table:SummaryReports",
            Report_1_Name="AllSummary",
        )

    def _estimate_heating(
        self,
        floor_plan: FloorPlan,
        floors: int,
        archetype: SwedishArchetype,
    ) -> float:
        """Estimate annual heating demand."""
        hdd = 3500  # Stockholm HDD
        gross_area = floor_plan.area * floors

        # Calculate wall area by orientation
        wall_area = sum(w.length for w in floor_plan.walls) * floors * 2.8
        window_area = wall_area * 0.2  # Approximate
        roof_area = floor_plan.area
        floor_area = floor_plan.area

        # UA values
        wall_ua = wall_area * archetype.envelope.wall_u_value
        window_ua = window_area * archetype.envelope.window_u_value
        roof_ua = roof_area * archetype.envelope.roof_u_value
        floor_ua = floor_area * archetype.envelope.floor_u_value

        # Infiltration
        volume = floor_plan.area * floors * 2.8
        inf_ua = volume * archetype.envelope.infiltration_ach * 0.34

        # Ventilation (with HR)
        vent_rate = gross_area * 0.35 / 1000
        hr_eff = archetype.hvac.heat_recovery_efficiency
        vent_ua = vent_rate * 1200 * (1 - hr_eff)

        total_ua = wall_ua + window_ua + roof_ua + floor_ua + inf_ua + vent_ua
        heating_kwh = total_ua * hdd * 24 / 1000
        internal_gains = gross_area * 15

        return max(0, heating_kwh - internal_gains) / gross_area

    def _select_weather_file(self, latitude: float) -> str:
        """Select Swedish weather file based on latitude."""
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

    def generate_multizone(
        self,
        footprint_coords: List[Tuple[float, float]],
        floors: int,
        archetype: SwedishArchetype,
        output_dir: Path,
        zone_breakdown: Dict[str, float],
        model_name: str = "multizone_v2",
        floor_height: float = 2.8,
        wwr_per_orientation: Optional[Dict[str, float]] = None,
        latitude: float = 59.35,
        longitude: float = 17.95,
        has_ftx: bool = True,
        has_f_only: bool = False,
    ) -> BaselineModel:
        """
        Generate multi-zone IDF for mixed-use buildings using FLOOR-BASED zoning.

        Swedish building pattern:
        - Ground floor(s): Commercial (retail, restaurant, grocery) with F-only ventilation
        - Upper floors: Residential with FTX heat recovery

        This is CRITICAL for accurate energy modeling because:
        1. Restaurant ventilation (10 L/s·m², no HR) dominates heat loss
        2. Commercial zones have higher internal gains
        3. Floor-by-floor modeling captures thermal coupling

        Args:
            footprint_coords: List of (x, y) coordinates in meters
            floors: Number of floors
            archetype: Swedish building archetype (for envelope only)
            output_dir: Output directory
            zone_breakdown: Dict of zone_type -> fraction (0.0-1.0)
                           e.g., {'residential': 0.88, 'restaurant': 0.06, 'retail': 0.06}
            model_name: Model name
            floor_height: Height per floor (m)
            wwr_per_orientation: WWR dict like {'N': 0.15, 'S': 0.25}
            latitude: Site latitude
            longitude: Site longitude
            has_ftx: Whether residential zones have FTX heat recovery
            has_f_only: Whether building has F-only ventilation (older buildings)

        Returns:
            BaselineModel with generated IDF path
        """
        from ..ingest.zone_configs import ZONE_CONFIGS
        from .zone_assignment import assign_zones_to_floors, get_zone_layout_summary

        logger.info(f"Generating multi-zone model: {model_name}")
        logger.info(f"  Zone breakdown: {zone_breakdown}")

        # Analyze footprint
        floor_plan = analyze_footprint(footprint_coords)
        logger.info(f"  Footprint area: {floor_plan.area:.1f} m²")

        # CRITICAL: Use floor-based zone assignment (Swedish pattern)
        # Commercial on ground floor(s), residential on upper floors
        zone_layout = assign_zones_to_floors(
            total_floors=floors,
            footprint_area_m2=floor_plan.area,
            zone_breakdown=zone_breakdown,
            floor_height_m=floor_height,
            has_ftx=has_ftx,
            has_f_only=has_f_only,
        )

        # Log zone layout for debugging
        layout_summary = get_zone_layout_summary(zone_layout)
        logger.info(f"Zone Layout:\n{layout_summary}")

        # Default WWR if not provided
        if wwr_per_orientation is None:
            wwr_per_orientation = {'N': 0.15, 'S': 0.25, 'E': 0.20, 'W': 0.20}

        # Create IDF
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        idf_path = output_dir / f"{model_name}.idf"

        # Start with minimal IDF
        idf = self._create_minimal_idf()

        # Set site location
        self._set_location(idf, latitude, longitude)

        # Add materials and constructions from archetype (envelope)
        self._add_materials(idf, archetype)

        # Add schedules (including zone-specific schedules)
        self._add_schedules(idf)
        self._add_multizone_schedules(idf, zone_breakdown)

        # Build geometry using floor-based zone assignment
        self._build_floor_based_geometry(
            idf=idf,
            floor_plan=floor_plan,
            zone_layout=zone_layout,
            floor_height=floor_height,
            archetype=archetype,
        )

        # Add windows per floor zone
        self._add_floor_based_windows(
            idf=idf,
            floor_plan=floor_plan,
            zone_layout=zone_layout,
            floor_height=floor_height,
            wwr_per_orientation=wwr_per_orientation,
        )

        # Add internal loads and HVAC per floor zone
        for floor_zone in zone_layout.floor_zones:
            zone_config = ZONE_CONFIGS.get(floor_zone.zone_type, ZONE_CONFIGS['other'])

            self._add_multizone_internal_loads(
                idf=idf,
                zone_name=floor_zone.zone_name,
                zone_area=floor_zone.area_m2,
                zone_config=zone_config,
            )

            # Use zone-specific ventilation (from FloorZone)
            self._add_floor_zone_hvac(
                idf=idf,
                floor_zone=floor_zone,
            )

        # Add outputs
        self._add_outputs(idf)

        # Save
        idf.saveas(str(idf_path))
        logger.info(f"Saved multi-zone IDF to {idf_path}")

        # Estimate heating (weighted by zones)
        gross_area = floor_plan.area * floors
        estimated_heating = self._estimate_multizone_heating(
            floor_plan, floors, archetype, zone_breakdown
        )

        return BaselineModel(
            idf_path=idf_path,
            weather_file=self._select_weather_file(latitude),
            archetype_used=f"{archetype.name}_multizone",
            floor_area_m2=gross_area,
            predicted_heating_kwh_m2=estimated_heating,
        )

    def _add_multizone_schedules(self, idf: IDF, zone_breakdown: Dict[str, float]):
        """Add zone-specific occupancy and lighting schedules."""
        from ..ingest.zone_configs import ZONE_CONFIGS

        for zone_type in zone_breakdown.keys():
            config = ZONE_CONFIGS.get(zone_type, ZONE_CONFIGS['other'])
            hours = config.get('occupancy_hours_per_day', 10)

            # Occupancy schedule for this zone type
            schedule_name = f"Occ_{zone_type}"
            if zone_type == 'residential':
                # Residential: higher evening/night occupancy
                self._newidfobject(idf, 
                    "Schedule:Compact",
                    Name=schedule_name,
                    Schedule_Type_Limits_Name="Fraction",
                    Field_1="Through: 12/31",
                    Field_2="For: AllDays",
                    Field_3="Until: 07:00",
                    Field_4="1.0",
                    Field_5="Until: 09:00",
                    Field_6="0.5",
                    Field_7="Until: 17:00",
                    Field_8="0.3",
                    Field_9="Until: 22:00",
                    Field_10="0.8",
                    Field_11="Until: 24:00",
                    Field_12="1.0",
                )
            elif zone_type == 'restaurant':
                # Restaurant: lunch and dinner peaks
                self._newidfobject(idf, 
                    "Schedule:Compact",
                    Name=schedule_name,
                    Schedule_Type_Limits_Name="Fraction",
                    Field_1="Through: 12/31",
                    Field_2="For: AllDays",
                    Field_3="Until: 10:00",
                    Field_4="0.0",
                    Field_5="Until: 14:00",
                    Field_6="0.8",
                    Field_7="Until: 17:00",
                    Field_8="0.3",
                    Field_9="Until: 22:00",
                    Field_10="1.0",
                    Field_11="Until: 24:00",
                    Field_12="0.0",
                )
            elif zone_type in ('retail', 'grocery'):
                # Retail: daytime only
                self._newidfobject(idf, 
                    "Schedule:Compact",
                    Name=schedule_name,
                    Schedule_Type_Limits_Name="Fraction",
                    Field_1="Through: 12/31",
                    Field_2="For: AllDays",
                    Field_3="Until: 09:00",
                    Field_4="0.0",
                    Field_5="Until: 20:00",
                    Field_6="1.0",
                    Field_7="Until: 24:00",
                    Field_8="0.0",
                )
            else:
                # Default commercial: 10h/day
                self._newidfobject(idf, 
                    "Schedule:Compact",
                    Name=schedule_name,
                    Schedule_Type_Limits_Name="Fraction",
                    Field_1="Through: 12/31",
                    Field_2="For: AllDays",
                    Field_3="Until: 08:00",
                    Field_4="0.0",
                    Field_5="Until: 18:00",
                    Field_6="1.0",
                    Field_7="Until: 24:00",
                    Field_8="0.0",
                )

    def _build_floor_based_geometry(
        self,
        idf: IDF,
        floor_plan: FloorPlan,
        zone_layout: 'BuildingZoneLayout',
        floor_height: float,
        archetype: SwedishArchetype,
    ):
        """
        Build geometry with ONE zone per floor based on zone assignment.

        Swedish pattern:
        - Floor 1: Restaurant zone (F-only ventilation)
        - Floors 2-N: Residential zones (FTX with heat recovery)

        This replaces the old approach of creating zones for all use types
        on all floors, which doesn't match reality.
        """
        from .zone_assignment import FloorZone

        coords = list(floor_plan.polygon.exterior.coords)[:-1]
        total_floors = zone_layout.total_floors

        for floor_zone in zone_layout.floor_zones:
            zone_name = floor_zone.zone_name
            floor_num = floor_zone.floor
            z_base = (floor_num - 1) * floor_height
            z_top = z_base + floor_height

            # Create zone object
            self._newidfobject(idf, 
                "Zone",
                Name=zone_name,
                Direction_of_Relative_North=0,
                X_Origin=0,
                Y_Origin=0,
                Z_Origin=z_base,
                Type=1,
                Multiplier=1,
                Ceiling_Height=floor_height,
                Volume=floor_plan.area * floor_height,
            )

            # Floor surface
            floor_vertices = [(x, y, z_base) for x, y in coords]
            floor_vertices.reverse()  # Floor faces down

            if floor_zone.is_ground_floor:
                floor_bc = "Ground"
                floor_bc_obj = ""
                floor_construction = "GroundFloor"
            else:
                # Find zone below (previous floor)
                zone_below = None
                for fz in zone_layout.floor_zones:
                    if fz.floor == floor_num - 1:
                        zone_below = fz
                        break
                floor_bc = "Surface"
                floor_bc_obj = f"{zone_below.zone_name}_Ceiling" if zone_below else ""
                floor_construction = "InteriorFloor"

            self._add_floor_surface(
                idf=idf,
                name=f"{zone_name}_Floor",
                zone=zone_name,
                construction=floor_construction,
                boundary=floor_bc,
                boundary_obj=floor_bc_obj,
                vertices=floor_vertices,
            )

            # Ceiling/Roof surface
            ceiling_vertices = [(x, y, z_top) for x, y in coords]

            if floor_zone.is_top_floor:
                ceiling_bc = "Outdoors"
                ceiling_bc_obj = ""
                ceiling_construction = "Roof"
                surface_type = "Roof"
                sun_exposed = True
            else:
                # Find zone above (next floor)
                zone_above = None
                for fz in zone_layout.floor_zones:
                    if fz.floor == floor_num + 1:
                        zone_above = fz
                        break
                ceiling_bc = "Surface"
                ceiling_bc_obj = f"{zone_above.zone_name}_Floor" if zone_above else ""
                ceiling_construction = "InteriorFloor"
                surface_type = "Ceiling"
                sun_exposed = False

            self._add_ceiling_surface(
                idf=idf,
                name=f"{zone_name}_Ceiling",
                zone=zone_name,
                surface_type=surface_type,
                construction=ceiling_construction,
                boundary=ceiling_bc,
                boundary_obj=ceiling_bc_obj,
                vertices=ceiling_vertices,
                sun_exposed=sun_exposed,
            )

            # Wall surfaces
            n = len(coords)
            for i in range(n):
                start = coords[i]
                end = coords[(i + 1) % n]

                wall_vertices = [
                    (start[0], start[1], z_top),
                    (start[0], start[1], z_base),
                    (end[0], end[1], z_base),
                    (end[0], end[1], z_top),
                ]

                dx = end[0] - start[0]
                dy = end[1] - start[1]
                azimuth = math.degrees(math.atan2(dy, -dx)) % 360
                cardinal = _azimuth_to_cardinal(azimuth)

                self._add_surface(
                    idf=idf,
                    name=f"{zone_name}_Wall_{cardinal}_{i}",
                    surface_type="Wall",
                    construction="ExteriorWall",
                    zone=zone_name,
                    boundary="Outdoors",
                    boundary_obj="",
                    vertices=wall_vertices,
                    sun_exposed=True,
                )

            # Add outdoor air node for this zone
            self._newidfobject(idf, "OutdoorAir:NodeList", Node_or_NodeList_Name_1=f"{zone_name}_OA")

    def _add_floor_surface(
        self,
        idf: IDF,
        name: str,
        zone: str,
        construction: str,
        boundary: str,
        boundary_obj: str,
        vertices: List[Tuple[float, float, float]],
    ):
        """Add a floor surface."""
        obj = self._newidfobject(idf, 
            "BuildingSurface:Detailed",
            Name=name,
            Surface_Type="Floor",
            Construction_Name=construction,
            Zone_Name=zone,
            Outside_Boundary_Condition=boundary,
            Outside_Boundary_Condition_Object=boundary_obj,
            Sun_Exposure="NoSun",
            Wind_Exposure="NoWind",
            Number_of_Vertices=len(vertices),
        )
        for i, (x, y, z) in enumerate(vertices):
            setattr(obj, f"Vertex_{i+1}_Xcoordinate", x)
            setattr(obj, f"Vertex_{i+1}_Ycoordinate", y)
            setattr(obj, f"Vertex_{i+1}_Zcoordinate", z)

    def _add_ceiling_surface(
        self,
        idf: IDF,
        name: str,
        zone: str,
        surface_type: str,
        construction: str,
        boundary: str,
        boundary_obj: str,
        vertices: List[Tuple[float, float, float]],
        sun_exposed: bool,
    ):
        """Add a ceiling or roof surface."""
        obj = self._newidfobject(idf, 
            "BuildingSurface:Detailed",
            Name=name,
            Surface_Type=surface_type,
            Construction_Name=construction,
            Zone_Name=zone,
            Outside_Boundary_Condition=boundary,
            Outside_Boundary_Condition_Object=boundary_obj,
            Sun_Exposure="SunExposed" if sun_exposed else "NoSun",
            Wind_Exposure="WindExposed" if sun_exposed else "NoWind",
            Number_of_Vertices=len(vertices),
        )
        for i, (x, y, z) in enumerate(vertices):
            setattr(obj, f"Vertex_{i+1}_Xcoordinate", x)
            setattr(obj, f"Vertex_{i+1}_Ycoordinate", y)
            setattr(obj, f"Vertex_{i+1}_Zcoordinate", z)

    def _add_floor_based_windows(
        self,
        idf: IDF,
        floor_plan: FloorPlan,
        zone_layout: 'BuildingZoneLayout',
        floor_height: float,
        wwr_per_orientation: Dict[str, float],
    ):
        """Add windows to each floor zone based on orientation and WWR."""
        window_height = min(1.4, floor_height - 1.0)
        sill_height = 0.9

        coords = list(floor_plan.polygon.exterior.coords)[:-1]
        n = len(coords)

        for floor_zone in zone_layout.floor_zones:
            zone_name = floor_zone.zone_name
            floor_num = floor_zone.floor
            z_base = (floor_num - 1) * floor_height

            for i in range(n):
                start = coords[i]
                end = coords[(i + 1) % n]

                dx = end[0] - start[0]
                dy = end[1] - start[1]
                length = math.sqrt(dx**2 + dy**2)

                if length < 1.0:
                    continue

                azimuth = math.degrees(math.atan2(dy, -dx)) % 360
                cardinal = _azimuth_to_cardinal(azimuth)

                simple_cardinal = cardinal[0] if len(cardinal) == 2 else cardinal
                if simple_cardinal not in wwr_per_orientation:
                    simple_cardinal = 'S'
                wwr = wwr_per_orientation.get(simple_cardinal, 0.2)

                wall_area = length * floor_height
                window_area = wall_area * wwr

                if window_area < 0.5:
                    continue

                window_width = min(window_area / window_height, length * 0.8)

                t = 0.5
                cx = start[0] + t * dx
                cy = start[1] + t * dy

                half_w = window_width / 2
                unit_dx = dx / length
                unit_dy = dy / length

                z_sill = z_base + sill_height
                z_head = z_sill + window_height

                window_vertices = [
                    (cx - half_w * unit_dx, cy - half_w * unit_dy, z_head),
                    (cx - half_w * unit_dx, cy - half_w * unit_dy, z_sill),
                    (cx + half_w * unit_dx, cy + half_w * unit_dy, z_sill),
                    (cx + half_w * unit_dx, cy + half_w * unit_dy, z_head),
                ]

                wall_name = f"{zone_name}_Wall_{cardinal}_{i}"
                self._add_window(
                    idf=idf,
                    name=f"{zone_name}_Win_{cardinal}_{i}",
                    wall_name=wall_name,
                    vertices=window_vertices,
                )

    def _add_floor_zone_hvac(
        self,
        idf: IDF,
        floor_zone: 'FloorZone',
    ):
        """
        Add HVAC for a specific floor zone with its ventilation settings.

        Uses FloorZone properties directly instead of looking up in ZONE_CONFIGS,
        ensuring correct heat recovery and airflow per floor.
        """
        zone_name = floor_zone.zone_name
        airflow_l_s_m2 = floor_zone.airflow_l_s_m2
        hr_eff = floor_zone.heat_recovery_eff
        setpoint_heat = 21.0  # Swedish default
        setpoint_cool = 26.0

        # Commercial zones may have different setpoints
        if floor_zone.zone_type in ('restaurant', 'retail', 'grocery'):
            setpoint_heat = 20.0
            setpoint_cool = 24.0

        hr_type = "Sensible" if hr_eff > 0 else "None"

        # Outdoor air specification
        oa_spec_name = f"{zone_name}_OA_Spec"
        self._newidfobject(idf, 
            "DesignSpecification:OutdoorAir",
            Name=oa_spec_name,
            Outdoor_Air_Method="Flow/Area",
            Outdoor_Air_Flow_per_Zone_Floor_Area=airflow_l_s_m2 / 1000,  # m³/s per m²
        )

        # Setpoint schedules
        heat_set_name = f"{zone_name}_HeatSet"
        cool_set_name = f"{zone_name}_CoolSet"

        self._newidfobject(idf, 
            "Schedule:Compact",
            Name=heat_set_name,
            Schedule_Type_Limits_Name="Temperature",
            Field_1="Through: 12/31",
            Field_2="For: AllDays",
            Field_3="Until: 24:00",
            Field_4=str(setpoint_heat),
        )

        self._newidfobject(idf, 
            "Schedule:Compact",
            Name=cool_set_name,
            Schedule_Type_Limits_Name="Temperature",
            Field_1="Through: 12/31",
            Field_2="For: AllDays",
            Field_3="Until: 24:00",
            Field_4=str(setpoint_cool),
        )

        # IdealLoads with floor-specific heat recovery (EnergyPlus 25.1 bug: Cooling_Sensible_Heat_Ratio must be BLANK)
        self._newidfobject(idf,
            "ZoneHVAC:IdealLoadsAirSystem",
            Name=f"{zone_name}_IdealLoads",
            Zone_Supply_Air_Node_Name=f"{zone_name}_Supply",
            Zone_Exhaust_Air_Node_Name=f"{zone_name}_Exhaust",
            Maximum_Heating_Supply_Air_Temperature=50,
            Minimum_Cooling_Supply_Air_Temperature=13,
            Maximum_Heating_Supply_Air_Humidity_Ratio=0.015,
            Minimum_Cooling_Supply_Air_Humidity_Ratio=0.009,
            Heating_Limit="NoLimit",
            Cooling_Limit="NoLimit",
            Dehumidification_Control_Type="ConstantSupplyHumidityRatio",
            Cooling_Sensible_Heat_Ratio="",
            Humidification_Control_Type="ConstantSupplyHumidityRatio",
            Design_Specification_Outdoor_Air_Object_Name=oa_spec_name,
            Outdoor_Air_Inlet_Node_Name=f"{zone_name}_OA",
            Demand_Controlled_Ventilation_Type="None",
            Outdoor_Air_Economizer_Type="NoEconomizer",
            Heat_Recovery_Type=hr_type,
            Sensible_Heat_Recovery_Effectiveness=hr_eff,
            Latent_Heat_Recovery_Effectiveness=0,
        )

        # Equipment list
        self._newidfobject(idf, 
            "ZoneHVAC:EquipmentList",
            Name=f"{zone_name}_EquipList",
            Load_Distribution_Scheme="SequentialLoad",
            Zone_Equipment_1_Object_Type="ZoneHVAC:IdealLoadsAirSystem",
            Zone_Equipment_1_Name=f"{zone_name}_IdealLoads",
            Zone_Equipment_1_Cooling_Sequence=1,
            Zone_Equipment_1_Heating_or_NoLoad_Sequence=1,
        )

        # Equipment connections
        self._newidfobject(idf, 
            "ZoneHVAC:EquipmentConnections",
            Zone_Name=zone_name,
            Zone_Conditioning_Equipment_List_Name=f"{zone_name}_EquipList",
            Zone_Air_Inlet_Node_or_NodeList_Name=f"{zone_name}_Supply",
            Zone_Air_Exhaust_Node_or_NodeList_Name=f"{zone_name}_Exhaust",
            Zone_Air_Node_Name=f"{zone_name}_AirNode",
            Zone_Return_Air_Node_or_NodeList_Name=f"{zone_name}_Return",
        )

        # Thermostat
        self._newidfobject(idf,
            "ZoneControl:Thermostat",
            Name=f"{zone_name}_Thermostat",
            Zone_or_ZoneList_Name=zone_name,
            Control_Type_Schedule_Name="ThermType",
            Control_1_Object_Type="ThermostatSetpoint:DualSetpoint",
            Control_1_Name=f"{zone_name}_DualSetpoint",
        )

        self._newidfobject(idf, 
            "ThermostatSetpoint:DualSetpoint",
            Name=f"{zone_name}_DualSetpoint",
            Heating_Setpoint_Temperature_Schedule_Name=heat_set_name,
            Cooling_Setpoint_Temperature_Schedule_Name=cool_set_name,
        )

    def _build_multizone_geometry(
        self,
        idf: IDF,
        floor_plan: FloorPlan,
        floors: int,
        floor_height: float,
        zone_breakdown: Dict[str, float],
        archetype: SwedishArchetype,
    ):
        """
        Build multi-zone geometry.

        For simplicity, we stack zones vertically (ground floor = commercial).
        More sophisticated approach would split horizontally but requires
        complex geometry operations.
        """
        # For now, create zones as percentage of floor area
        # All zones share the same footprint but are modeled separately
        # (EnergyPlus will handle thermal coupling through floors)

        for floor in range(1, floors + 1):
            z_floor = (floor - 1) * floor_height
            z_ceiling = floor * floor_height

            for zone_type, fraction in zone_breakdown.items():
                if fraction < 0.01:
                    continue

                zone_name = f"Floor{floor}_{zone_type.capitalize()}"

                # Create zone
                self._newidfobject(idf, "Zone", Name=zone_name)

                # Create floor surface
                floor_coords = [
                    (p[0], p[1], z_floor) for p in floor_plan.polygon.exterior.coords[:-1]
                ]
                self._create_surface(
                    idf=idf,
                    name=f"{zone_name}_Floor",
                    zone=zone_name,
                    surface_type="Floor",
                    construction="FloorConstruction",
                    coords=floor_coords,
                    outside_bc="Ground" if floor == 1 else "Surface",
                )

                # Create ceiling/roof surface
                ceiling_coords = [
                    (p[0], p[1], z_ceiling) for p in reversed(list(floor_plan.polygon.exterior.coords[:-1]))
                ]
                self._create_surface(
                    idf=idf,
                    name=f"{zone_name}_Ceiling",
                    zone=zone_name,
                    surface_type="Ceiling" if floor < floors else "Roof",
                    construction="RoofConstruction" if floor == floors else "IntFloor",
                    coords=ceiling_coords,
                    outside_bc="Outdoors" if floor == floors else "Surface",
                )

                # Create wall surfaces
                for i, wall in enumerate(floor_plan.walls):
                    wall_name = f"{zone_name}_Wall{i}"
                    wall_coords = [
                        (wall.start[0], wall.start[1], z_ceiling),
                        (wall.start[0], wall.start[1], z_floor),
                        (wall.end[0], wall.end[1], z_floor),
                        (wall.end[0], wall.end[1], z_ceiling),
                    ]
                    self._create_surface(
                        idf=idf,
                        name=wall_name,
                        zone=zone_name,
                        surface_type="Wall",
                        construction="WallConstruction",
                        coords=wall_coords,
                        outside_bc="Outdoors",
                    )

    def _create_surface(
        self,
        idf: IDF,
        name: str,
        zone: str,
        surface_type: str,
        construction: str,
        coords: List[Tuple[float, float, float]],
        outside_bc: str = "Outdoors",
    ):
        """Create a surface with specified vertices."""
        surface = self._newidfobject(idf, 
            "BuildingSurface:Detailed",
            Name=name,
            Surface_Type=surface_type,
            Construction_Name=construction,
            Zone_Name=zone,
            Outside_Boundary_Condition=outside_bc,
            Sun_Exposure="SunExposed" if outside_bc == "Outdoors" else "NoSun",
            Wind_Exposure="WindExposed" if outside_bc == "Outdoors" else "NoWind",
            Number_of_Vertices=len(coords),
        )

        # Add vertices
        for i, (x, y, z) in enumerate(coords):
            setattr(surface, f"Vertex_{i+1}_Xcoordinate", x)
            setattr(surface, f"Vertex_{i+1}_Ycoordinate", y)
            setattr(surface, f"Vertex_{i+1}_Zcoordinate", z)

    def _add_multizone_internal_loads(
        self,
        idf: IDF,
        zone_name: str,
        zone_area: float,
        zone_config: Dict[str, Any],
    ):
        """Add internal loads for a specific zone type."""
        zone_type = zone_name.split('_')[-1].lower()
        schedule_name = f"Occ_{zone_type}"

        # People (simplified - use equipment gains instead)
        lighting_w_m2 = zone_config.get('lighting_w_m2', 10.0)
        equipment_w_m2 = zone_config.get('equipment_w_m2', 10.0)

        # Lighting
        self._newidfobject(idf, 
            "Lights",
            Name=f"{zone_name}_Lights",
            Zone_or_ZoneList_or_Space_or_SpaceList_Name=zone_name,
            Schedule_Name=schedule_name,
            Design_Level_Calculation_Method="Watts/Area",
            Watts_per_Floor_Area=lighting_w_m2,
        )

        # Equipment
        self._newidfobject(idf, 
            "ElectricEquipment",
            Name=f"{zone_name}_Equip",
            Zone_or_ZoneList_or_Space_or_SpaceList_Name=zone_name,
            Schedule_Name=schedule_name,
            Design_Level_Calculation_Method="Watts/Area",
            Watts_per_Floor_Area=equipment_w_m2,
        )

        # Infiltration (same for all zones, comes from archetype)
        self._newidfobject(idf, 
            "ZoneInfiltration:DesignFlowRate",
            Name=f"{zone_name}_Infiltration",
            Zone_or_ZoneList_or_Space_or_SpaceList_Name=zone_name,
            Schedule_Name="Always1",
            Design_Flow_Rate_Calculation_Method="AirChanges/Hour",
            Air_Changes_per_Hour=0.1,  # Lower for mixed-use (typically better sealed)
        )

    def _add_multizone_hvac(
        self,
        idf: IDF,
        zone_name: str,
        zone_config: Dict[str, Any],
    ):
        """Add HVAC with zone-specific heat recovery and ventilation."""
        # Get zone-specific parameters from config
        airflow_l_s_m2 = zone_config.get('airflow_l_s_m2', 0.35)
        hr_eff = zone_config.get('heat_recovery_eff', 0.0)
        vent_type = zone_config.get('ventilation_type', 'F')
        setpoint_heat = zone_config.get('setpoint_heating_c', 21.0)
        setpoint_cool = zone_config.get('setpoint_cooling_c', 24.0)

        # HR type
        hr_type = "Sensible" if hr_eff > 0 else "None"

        # Create outdoor air specification for this zone
        oa_spec_name = f"{zone_name}_OA_Spec"
        self._newidfobject(idf, 
            "DesignSpecification:OutdoorAir",
            Name=oa_spec_name,
            Outdoor_Air_Method="Flow/Area",
            Outdoor_Air_Flow_per_Zone_Floor_Area=airflow_l_s_m2 / 1000,  # m³/s per m²
        )

        # Zone-specific setpoint schedules
        heat_set_name = f"{zone_name}_HeatSet"
        cool_set_name = f"{zone_name}_CoolSet"

        self._newidfobject(idf, 
            "Schedule:Compact",
            Name=heat_set_name,
            Schedule_Type_Limits_Name="Temperature",
            Field_1="Through: 12/31",
            Field_2="For: AllDays",
            Field_3="Until: 24:00",
            Field_4=str(setpoint_heat),
        )

        self._newidfobject(idf, 
            "Schedule:Compact",
            Name=cool_set_name,
            Schedule_Type_Limits_Name="Temperature",
            Field_1="Through: 12/31",
            Field_2="For: AllDays",
            Field_3="Until: 24:00",
            Field_4=str(setpoint_cool),
        )

        # IdealLoads with zone-specific parameters (EnergyPlus 25.1 bug: Cooling_Sensible_Heat_Ratio must be BLANK)
        self._newidfobject(idf,
            "ZoneHVAC:IdealLoadsAirSystem",
            Name=f"{zone_name}_IdealLoads",
            Zone_Supply_Air_Node_Name=f"{zone_name}_Supply",
            Zone_Exhaust_Air_Node_Name=f"{zone_name}_Exhaust",
            Maximum_Heating_Supply_Air_Temperature=50,
            Minimum_Cooling_Supply_Air_Temperature=13,
            Maximum_Heating_Supply_Air_Humidity_Ratio=0.015,
            Minimum_Cooling_Supply_Air_Humidity_Ratio=0.009,
            Heating_Limit="NoLimit",
            Cooling_Limit="NoLimit",
            Dehumidification_Control_Type="ConstantSupplyHumidityRatio",
            Cooling_Sensible_Heat_Ratio="",
            Humidification_Control_Type="ConstantSupplyHumidityRatio",
            Design_Specification_Outdoor_Air_Object_Name=oa_spec_name,
            Outdoor_Air_Inlet_Node_Name=f"{zone_name}_OA",
            Demand_Controlled_Ventilation_Type="None",
            Outdoor_Air_Economizer_Type="NoEconomizer",
            Heat_Recovery_Type=hr_type,
            Sensible_Heat_Recovery_Effectiveness=hr_eff,
            Latent_Heat_Recovery_Effectiveness=0,
        )

        # Equipment list
        self._newidfobject(idf, 
            "ZoneHVAC:EquipmentList",
            Name=f"{zone_name}_EquipList",
            Load_Distribution_Scheme="SequentialLoad",
            Zone_Equipment_1_Object_Type="ZoneHVAC:IdealLoadsAirSystem",
            Zone_Equipment_1_Name=f"{zone_name}_IdealLoads",
            Zone_Equipment_1_Cooling_Sequence=1,
            Zone_Equipment_1_Heating_or_NoLoad_Sequence=1,
        )

        # Equipment connections
        self._newidfobject(idf, 
            "ZoneHVAC:EquipmentConnections",
            Zone_Name=zone_name,
            Zone_Conditioning_Equipment_List_Name=f"{zone_name}_EquipList",
            Zone_Air_Inlet_Node_or_NodeList_Name=f"{zone_name}_Supply",
            Zone_Air_Exhaust_Node_or_NodeList_Name=f"{zone_name}_Exhaust",
            Zone_Air_Node_Name=f"{zone_name}_AirNode",
            Zone_Return_Air_Node_or_NodeList_Name=f"{zone_name}_Return",
        )

        # Thermostat
        self._newidfobject(idf,
            "ZoneControl:Thermostat",
            Name=f"{zone_name}_Thermostat",
            Zone_or_ZoneList_Name=zone_name,
            Control_Type_Schedule_Name="ThermType",
            Control_1_Object_Type="ThermostatSetpoint:DualSetpoint",
            Control_1_Name=f"{zone_name}_DualSetpoint",
        )

        self._newidfobject(idf, 
            "ThermostatSetpoint:DualSetpoint",
            Name=f"{zone_name}_DualSetpoint",
            Heating_Setpoint_Temperature_Schedule_Name=heat_set_name,
            Cooling_Setpoint_Temperature_Schedule_Name=cool_set_name,
        )

    def _estimate_multizone_heating(
        self,
        floor_plan: FloorPlan,
        floors: int,
        archetype: SwedishArchetype,
        zone_breakdown: Dict[str, float],
    ) -> float:
        """Estimate annual heating for multi-zone building."""
        from ..ingest.zone_configs import ZONE_CONFIGS

        hdd = 3500  # Stockholm HDD
        gross_area = floor_plan.area * floors

        # Calculate envelope losses (same for all zones)
        wall_area = sum(w.length for w in floor_plan.walls) * floors * 2.8
        window_area = wall_area * 0.2
        roof_area = floor_plan.area
        floor_area = floor_plan.area

        wall_ua = wall_area * archetype.envelope.wall_u_value
        window_ua = window_area * archetype.envelope.window_u_value
        roof_ua = roof_area * archetype.envelope.roof_u_value
        floor_ua = floor_area * archetype.envelope.floor_u_value

        # Infiltration
        volume = floor_plan.area * floors * 2.8
        inf_ua = volume * archetype.envelope.infiltration_ach * 0.34

        # Ventilation - weighted by zones
        total_vent_loss = 0.0
        total_internal_gains = 0.0

        for zone_type, fraction in zone_breakdown.items():
            config = ZONE_CONFIGS.get(zone_type, ZONE_CONFIGS['other'])
            zone_area = gross_area * fraction

            # Ventilation rate and HR for this zone
            airflow = config['airflow_l_s_m2'] / 1000  # m³/s per m²
            hr = config['heat_recovery_eff']

            # Ventilation heat loss (W/K)
            vent_ua_zone = zone_area * airflow * 1200 * (1 - hr)
            total_vent_loss += vent_ua_zone

            # Internal gains (W)
            internal_w_m2 = config.get('internal_gains_w_m2', 5.0)
            operating_fraction = (
                config.get('occupancy_hours_per_day', 10) *
                config.get('operating_days_per_year', 300) / (24 * 365)
            )
            total_internal_gains += zone_area * internal_w_m2 * operating_fraction

        total_ua = wall_ua + window_ua + roof_ua + floor_ua + inf_ua + total_vent_loss
        heating_kwh = total_ua * hdd * 24 / 1000
        internal_gains_kwh = total_internal_gains * 8760 / 1000

        return max(0, heating_kwh - internal_gains_kwh) / gross_area


def generate_from_footprint(
    footprint_coords: List[Tuple[float, float]],
    floors: int,
    archetype: SwedishArchetype,
    output_dir: Path,
    **kwargs,
) -> BaselineModel:
    """
    Convenience function to generate IDF from footprint.

    Args:
        footprint_coords: List of (x, y) coordinates
        floors: Number of floors
        archetype: Swedish archetype
        output_dir: Output directory
        **kwargs: Additional arguments for GeomEppyGenerator.generate()

    Returns:
        BaselineModel
    """
    generator = GeomEppyGenerator()
    return generator.generate(
        footprint_coords=footprint_coords,
        floors=floors,
        archetype=archetype,
        output_dir=output_dir,
        **kwargs,
    )
