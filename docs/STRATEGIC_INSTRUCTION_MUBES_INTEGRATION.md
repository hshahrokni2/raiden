# Strategic Instruction: MUBES-Inspired Architecture Upgrade

**Priority:** HIGH
**Estimated Effort:** 2-3 weeks
**Author:** Architecture Review
**Date:** 2025-12-19
**Last Updated:** 2025-12-19 (Claude Opus 4.5 Analysis)

---

## 🎯 ULTRATHINK EXECUTIVE SUMMARY

### Current State (Raiden v0.2.0)
- **35,800 LOC** across 71 Python files
- **40 Swedish archetypes** (TABULA/EPISCOPE) with construction-era detection
- **22 ECMs** with constraint-aware filtering
- **37,489 Stockholm buildings** in GeoJSON with 167 properties each
- **209 tests passing** with EnergyPlus 25.1.0 integration
- Deterministic calibration achieving 10% gap to energy declarations

### Strategic Gap Analysis

| Capability | Current | Target | Gap Severity |
|------------|---------|--------|--------------|
| Footprint geometry | Rectangle approximation | Actual polygon | **CRITICAL** - loses L/star/courtyard shapes |
| Calibration | Deterministic (~30s) | Bayesian with CI (~5s) | **HIGH** - no uncertainty quantification |
| ECM count | 22 | 50+ | **MEDIUM** - missing Swedish-specific (FVP, DH opt) |
| Cost data | Hardcoded estimates | BeBo/Wikells sourced | **MEDIUM** - affects ROI accuracy |
| Scalability | Single building | 37k batch | **LOW** - architecture supports, needs optimization |

### Three Transformational Upgrades

This document provides implementation instructions for upgrading Raiden's building simulation pipeline based on research from KTH's MUBES (Massive Urban Building Energy Simulations) project:

#### 1. **GeomEppy Integration** (Week 1) - CRITICAL
Replace manual IDF generation (~1,000 lines) with geometry-preserving library (~200 lines).
- **Why**: Current `generator.py:148-168` converts ALL footprints to rectangles, losing lamellhus variants, stjärnhus, slutet kvarter
- **Impact**: Accurate surface areas → accurate heat loss → accurate savings predictions
- **Risk**: GeomEppy uses forked eppy - verify compatibility with existing `idf_parser.py`

#### 2. **Surrogate-Based Bayesian Calibration** (Week 2-3) - HIGH
Replace deterministic calibration with uncertainty-quantified approach.
- **Why**: Current calibration gives point estimates; BRF boards need confidence intervals for investment decisions
- **Impact**: "Roof insulation saves 5-12% (90% CI)" vs "saves 8%"
- **Compute**: One-time ~8h training (40 archetypes × 100 samples); per-building <5s

#### 3. **ECM Catalog Expansion** (Ongoing) - MEDIUM
Expand from 22 to 50+ ECMs with Swedish-specific measures.
- **Why**: Missing high-impact Swedish measures: frånluftsvärmepump (40-60% savings), district heating optimization
- **Impact**: More relevant recommendations, especially for F-ventilated buildings

### ROI of This Investment

| Metric | Before | After | Business Value |
|--------|--------|-------|----------------|
| Geometry accuracy | ~70% (rectangles) | ~95% (actual) | Correct wall/window areas |
| Prediction uncertainty | None | 90% CI on all outputs | Board-level confidence |
| Calibration speed | 30s (E+ run) | 5s (surrogate) | 37k buildings/day possible |
| ECM coverage | 22 (generic) | 50+ (Swedish-specific) | FVP alone = major upgrade |

### Critical Path

```
Week 1: GeomEppy
├── Add dependency, create generator_v2.py
├── Create minimal IDF template
├── Test against existing generator (5% tolerance)
└── Update address_pipeline to use v2

Week 2: Surrogate Training (can run overnight)
├── Implement surrogate.py (Gaussian Process)
├── Generate Latin Hypercube samples (100 per archetype)
├── Run batch E+ simulations (~8h with 8 workers)
└── Train and validate GP models (R² > 0.95)

Week 3: Bayesian Calibration
├── Implement bayesian.py (ABC-SMC)
├── Create calibrator_v2.py interface
├── Add uncertainty to ECM predictions
└── Update CLI/API with --uncertainty flag
```

These changes will:
- Preserve actual building footprint geometry (not rectangle approximation)
- Provide confidence intervals on all energy predictions
- Enable hierarchical learning across 37,489 buildings
- Reduce calibration time from minutes to seconds per building

---

## Part 1: GeomEppy Universal Adoption

### 1.1 Objective

Replace `src/baseline/generator.py` (~1,000 lines of manual IDF string generation) with GeomEppy-based generation (~200 lines) that preserves actual building footprints from Microsoft Building Footprints and Sweden Buildings GeoJSON.

### 1.2 Why This Matters

Current Raiden approach (generator.py:148-168):
```python
# PROBLEM: Converts ANY footprint to equivalent rectangle
half_perim = perimeter / 2
discriminant = half_perim**2 - 4 * footprint_area
length = (half_perim + math.sqrt(discriminant)) / 2
width = footprint_area / length
```

This loses:
- L-shaped buildings (lamellhus variants)
- Star-shaped buildings (stjärnhus)
- Courtyard buildings (slutet kvarter)
- Any irregular footprint

GeomEppy preserves actual geometry and handles surface intersections automatically.

### 1.3 Implementation Steps

#### Step 1: Add GeomEppy Dependency

```bash
# pyproject.toml
[project.dependencies]
geomeppy = ">=0.11.8"
```

Note: GeomEppy uses a forked version of eppy. Verify compatibility with existing eppy usage in `src/core/idf_parser.py`.

#### Step 2: Create New Generator Module

Create `src/baseline/generator_v2.py`:

```python
"""
GeomEppy-based IDF Generator.

Replaces manual IDF string generation with geometry-preserving approach.
Compatible with Microsoft Building Footprints and Sweden Buildings GeoJSON.
"""

from pathlib import Path
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import logging

from geomeppy import IDF
from shapely.geometry import Polygon
from shapely.ops import transform
import pyproj

from ..geometry.building_geometry import BuildingGeometry
from .archetypes import SwedishArchetype

logger = logging.getLogger(__name__)


@dataclass
class BaselineModelV2:
    """Generated baseline model with geometry metadata."""
    idf_path: Path
    weather_file: str
    archetype_used: str
    floor_area_m2: float
    footprint_preserved: bool  # True if actual footprint used
    num_zones: int
    predicted_heating_kwh_m2: float


class GeomEppyGenerator:
    """
    Generate EnergyPlus IDF from actual building footprints.

    Key improvements over v1:
    - Preserves actual footprint geometry (no rectangle approximation)
    - Automatic surface intersection handling
    - Built-in WWR per orientation
    - Optional core-perimeter zoning
    """

    # Path to minimal IDF template (E+ version specific)
    TEMPLATE_IDF = Path(__file__).parent / "templates" / "minimal_v25.1.idf"

    def __init__(self, energyplus_idd_path: Optional[str] = None):
        """
        Initialize generator.

        Args:
            energyplus_idd_path: Path to Energy+.idd (auto-detect if None)
        """
        if energyplus_idd_path:
            IDF.setiddname(energyplus_idd_path)

    def generate(
        self,
        footprint_coords: List[Tuple[float, float]],
        height_m: float,
        num_stories: int,
        archetype: SwedishArchetype,
        wwr_by_orientation: Dict[str, float],
        output_path: Path,
        model_name: str = "baseline",
        latitude: float = 59.35,
        longitude: float = 17.95,
        zoning: str = "by_storey",  # or "core_perimeter"
    ) -> BaselineModelV2:
        """
        Generate IDF from actual building footprint.

        Args:
            footprint_coords: List of (x, y) tuples in meters (local coords)
                              OR (lon, lat) tuples (will be converted)
            height_m: Total building height
            num_stories: Number of floors
            archetype: Swedish archetype with envelope/HVAC properties
            wwr_by_orientation: WWR per cardinal direction {'N': 0.15, 'S': 0.25, ...}
            output_path: Where to save IDF
            model_name: Model identifier
            latitude: Site latitude
            longitude: Site longitude
            zoning: "by_storey" (1 zone/floor) or "core_perimeter" (5 zones/floor)

        Returns:
            BaselineModelV2 with path and metadata
        """
        logger.info(f"Generating IDF: {model_name}")
        logger.info(f"  Footprint vertices: {len(footprint_coords)}")
        logger.info(f"  Height: {height_m}m, Stories: {num_stories}")
        logger.info(f"  Archetype: {archetype.name}")

        # Convert coordinates if needed (WGS84 -> local meters)
        local_coords = self._ensure_local_coords(footprint_coords, latitude, longitude)

        # Validate and clean footprint
        local_coords = self._clean_footprint(local_coords)

        # Create IDF from template
        idf = IDF(str(self.TEMPLATE_IDF))

        # Add building block with actual footprint
        idf.add_block(
            name=model_name,
            coordinates=local_coords,
            height=height_m,
            num_stories=num_stories,
            below_ground_stories=0,
            below_ground_storey_height=0,
            zoning=zoning,
        )

        # Set WWR per orientation
        wwr_map = self._convert_wwr_to_azimuth(wwr_by_orientation)
        idf.set_wwr(wwr_map, construction="Window")

        # Handle surface intersections and boundary conditions
        idf.intersect_match()

        # Apply archetype constructions
        self._apply_constructions(idf, archetype)

        # Apply archetype HVAC (IdealLoadsAirSystem with heat recovery)
        self._apply_hvac(idf, archetype)

        # Apply internal loads (Sveby schedules)
        self._apply_internal_loads(idf, archetype)

        # Apply E+ 25.1.0 bug workaround
        self._apply_ep_bug_workaround(idf)

        # Set simulation parameters
        self._set_simulation_params(idf, latitude, longitude)

        # Add output variables
        self._add_outputs(idf)

        # Save IDF
        output_path.parent.mkdir(parents=True, exist_ok=True)
        idf.saveas(str(output_path))

        # Calculate metadata
        footprint_area = Polygon(local_coords).area
        floor_area = footprint_area * num_stories
        num_zones = num_stories if zoning == "by_storey" else num_stories * 5

        return BaselineModelV2(
            idf_path=output_path,
            weather_file=self._select_weather_file(latitude),
            archetype_used=archetype.name,
            floor_area_m2=floor_area,
            footprint_preserved=True,
            num_zones=num_zones,
            predicted_heating_kwh_m2=self._estimate_heating(floor_area, archetype),
        )

    def _ensure_local_coords(
        self,
        coords: List[Tuple[float, float]],
        center_lat: float,
        center_lon: float,
    ) -> List[Tuple[float, float]]:
        """Convert WGS84 (lon, lat) to local meters if needed."""
        # Check if coords look like WGS84 (small numbers near expected range)
        sample = coords[0]
        if -180 <= sample[0] <= 180 and -90 <= sample[1] <= 90:
            # Likely WGS84, convert to local meters
            proj_wgs84 = pyproj.CRS("EPSG:4326")
            proj_local = pyproj.CRS("EPSG:3006")  # SWEREF99 TM for Sweden
            transformer = pyproj.Transformer.from_crs(proj_wgs84, proj_local, always_xy=True)

            converted = [transformer.transform(lon, lat) for lon, lat in coords]

            # Center around origin for numerical stability
            min_x = min(c[0] for c in converted)
            min_y = min(c[1] for c in converted)
            return [(x - min_x, y - min_y) for x, y in converted]

        # Already in local coords
        return coords

    def _clean_footprint(self, coords: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """Clean and validate footprint polygon."""
        poly = Polygon(coords)

        if not poly.is_valid:
            # Try to fix self-intersections
            poly = poly.buffer(0)
            logger.warning("Fixed invalid footprint geometry")

        if not poly.exterior.is_ccw:
            # GeomEppy expects counter-clockwise
            coords = list(reversed(coords))

        # Remove closing point if present (GeomEppy adds it)
        if coords[0] == coords[-1]:
            coords = coords[:-1]

        return coords

    def _convert_wwr_to_azimuth(self, wwr_by_orientation: Dict[str, float]) -> Dict[int, float]:
        """Convert cardinal directions to azimuth angles."""
        direction_to_azimuth = {
            'N': 0, 'NE': 45, 'E': 90, 'SE': 135,
            'S': 180, 'SW': 225, 'W': 270, 'NW': 315,
        }
        return {
            direction_to_azimuth[d]: wwr
            for d, wwr in wwr_by_orientation.items()
            if d in direction_to_azimuth
        }

    def _apply_constructions(self, idf: IDF, archetype: SwedishArchetype) -> None:
        """Add Material and Construction objects from archetype U-values."""
        env = archetype.envelope

        # Calculate insulation thicknesses from U-values
        k_insulation = 0.035  # W/mK mineral wool
        r_surface = 0.17 + 0.04  # Internal + external
        r_concrete = 0.2  # 200mm concrete

        def calc_ins_thickness(u_value: float) -> float:
            r_required = 1.0 / u_value
            r_insulation = max(0.05, r_required - r_surface - r_concrete)
            return r_insulation * k_insulation

        # Add materials
        idf.newidfobject(
            "MATERIAL",
            Name="Concrete200",
            Roughness="MediumRough",
            Thickness=0.2,
            Conductivity=1.0,
            Density=2300,
            Specific_Heat=880,
        )

        idf.newidfobject(
            "MATERIAL",
            Name="WallInsulation",
            Roughness="MediumRough",
            Thickness=calc_ins_thickness(env.wall_u_value),
            Conductivity=0.035,
            Density=30,
            Specific_Heat=840,
        )

        idf.newidfobject(
            "MATERIAL",
            Name="RoofInsulation",
            Roughness="MediumRough",
            Thickness=calc_ins_thickness(env.roof_u_value),
            Conductivity=0.035,
            Density=30,
            Specific_Heat=840,
        )

        idf.newidfobject(
            "MATERIAL",
            Name="FloorInsulation",
            Roughness="MediumRough",
            Thickness=calc_ins_thickness(env.floor_u_value),
            Conductivity=0.035,
            Density=30,
            Specific_Heat=840,
        )

        # Add constructions
        idf.newidfobject(
            "CONSTRUCTION",
            Name="ExteriorWall",
            Outside_Layer="Concrete200",
            Layer_2="WallInsulation",
        )

        idf.newidfobject(
            "CONSTRUCTION",
            Name="Roof",
            Outside_Layer="Concrete200",
            Layer_2="RoofInsulation",
        )

        idf.newidfobject(
            "CONSTRUCTION",
            Name="GroundFloor",
            Outside_Layer="Concrete200",
            Layer_2="FloorInsulation",
        )

        idf.newidfobject(
            "CONSTRUCTION",
            Name="InteriorFloor",
            Outside_Layer="Concrete200",
        )

        # Window (SimpleGlazingSystem)
        idf.newidfobject(
            "WINDOWMATERIAL:SIMPLEGLAZINGSYSTEM",
            Name="GlazingSystem",
            UFactor=env.window_u_value,
            Solar_Heat_Gain_Coefficient=env.window_shgc,
        )

        idf.newidfobject(
            "CONSTRUCTION",
            Name="Window",
            Outside_Layer="GlazingSystem",
        )

        # Assign constructions to surfaces
        for surface in idf.idfobjects["BUILDINGSURFACE:DETAILED"]:
            if surface.Surface_Type == "Wall":
                surface.Construction_Name = "ExteriorWall"
            elif surface.Surface_Type == "Roof":
                surface.Construction_Name = "Roof"
            elif surface.Surface_Type == "Floor":
                if surface.Outside_Boundary_Condition == "Ground":
                    surface.Construction_Name = "GroundFloor"
                else:
                    surface.Construction_Name = "InteriorFloor"
            elif surface.Surface_Type == "Ceiling":
                surface.Construction_Name = "InteriorFloor"

        for window in idf.idfobjects["FENESTRATIONSURFACE:DETAILED"]:
            window.Construction_Name = "Window"

    def _apply_hvac(self, idf: IDF, archetype: SwedishArchetype) -> None:
        """Add IdealLoadsAirSystem with heat recovery for each zone."""
        hvac = archetype.hvac
        hr_eff = hvac.heat_recovery_efficiency
        hr_type = "Sensible" if hr_eff > 0 else "None"

        for zone in idf.idfobjects["ZONE"]:
            zone_name = zone.Name

            # Outdoor air node
            idf.newidfobject(
                "OUTDOORAIR:NODELIST",
                Node_or_NodeList_Name_1=f"{zone_name}_OA",
            )

            # Outdoor air spec (BBR 6:251 = 0.35 l/s/m²)
            idf.newidfobject(
                "DESIGNSPECIFICATION:OUTDOORAIR",
                Name=f"{zone_name}_OA_Spec",
                Outdoor_Air_Method="Flow/Area",
                Outdoor_Air_Flow_per_Zone_Floor_Area=0.00035,
            )

            # IdealLoadsAirSystem
            idf.newidfobject(
                "ZONEHVAC:IDEALLOADSAIRSYSTEM",
                Name=f"{zone_name}_IdealLoads",
                Zone_Supply_Air_Node_Name=f"{zone_name}_Supply",
                Zone_Exhaust_Air_Node_Name=f"{zone_name}_Exhaust",
                Maximum_Heating_Supply_Air_Temperature=50,
                Minimum_Cooling_Supply_Air_Temperature=13,
                Maximum_Heating_Supply_Air_Humidity_Ratio=0.015,
                Minimum_Cooling_Supply_Air_Humidity_Ratio=0.009,
                Heating_Limit="NoLimit",
                Cooling_Limit="NoLimit",
                Dehumidification_Control_Type="ConstantSupplyHumidityRatio",  # E+ 25.1 bug!
                Humidification_Control_Type="ConstantSupplyHumidityRatio",    # E+ 25.1 bug!
                Design_Specification_Outdoor_Air_Object_Name=f"{zone_name}_OA_Spec",
                Outdoor_Air_Inlet_Node_Name=f"{zone_name}_OA",
                Heat_Recovery_Type=hr_type,
                Sensible_Heat_Recovery_Effectiveness=hr_eff,
                Latent_Heat_Recovery_Effectiveness=0.0,
            )

            # Equipment list
            idf.newidfobject(
                "ZONEHVAC:EQUIPMENTLIST",
                Name=f"{zone_name}_EquipList",
                Load_Distribution_Scheme="SequentialLoad",
                Zone_Equipment_1_Object_Type="ZoneHVAC:IdealLoadsAirSystem",
                Zone_Equipment_1_Name=f"{zone_name}_IdealLoads",
                Zone_Equipment_1_Cooling_Sequence=1,
                Zone_Equipment_1_Heating_or_NoLoad_Sequence=1,
            )

            # Equipment connections
            idf.newidfobject(
                "ZONEHVAC:EQUIPMENTCONNECTIONS",
                Zone_Name=zone_name,
                Zone_Conditioning_Equipment_List_Name=f"{zone_name}_EquipList",
                Zone_Air_Inlet_Node_or_NodeList_Name=f"{zone_name}_Supply",
                Zone_Air_Exhaust_Node_or_NodeList_Name=f"{zone_name}_Exhaust",
                Zone_Air_Node_Name=f"{zone_name}_AirNode",
                Zone_Return_Air_Node_or_NodeList_Name=f"{zone_name}_Return",
            )

            # Thermostat
            idf.newidfobject(
                "ZONECONTROL:THERMOSTAT",
                Name=f"{zone_name}_Thermostat",
                Zone_or_ZoneList_Name=zone_name,
                Control_Type_Schedule_Name="ThermType",
                Control_1_Object_Type="ThermostatSetpoint:DualSetpoint",
                Control_1_Name=f"{zone_name}_DualSetpoint",
            )

            idf.newidfobject(
                "THERMOSTATSETPOINT:DUALSETPOINT",
                Name=f"{zone_name}_DualSetpoint",
                Heating_Setpoint_Temperature_Schedule_Name="HeatSet",
                Cooling_Setpoint_Temperature_Schedule_Name="CoolSet",
            )

    def _apply_internal_loads(self, idf: IDF, archetype: SwedishArchetype) -> None:
        """Add People, Lights, Equipment, Infiltration per zone (Sveby schedules)."""
        env = archetype.envelope

        for zone in idf.idfobjects["ZONE"]:
            zone_name = zone.Name
            floor_area = zone.Floor_Area if hasattr(zone, 'Floor_Area') else 100  # Estimate

            # People (~25 m²/person, Sveby)
            idf.newidfobject(
                "PEOPLE",
                Name=f"{zone_name}_People",
                Zone_or_ZoneList_Name=zone_name,
                Number_of_People_Schedule_Name="OccupancySchedule",
                Number_of_People_Calculation_Method="People",
                Number_of_People=max(1, floor_area / 25),
                Fraction_Radiant=0.3,
                Activity_Level_Schedule_Name="ActivityLevel",
            )

            # Lights (8 W/m², Sveby)
            idf.newidfobject(
                "LIGHTS",
                Name=f"{zone_name}_Lights",
                Zone_or_ZoneList_Name=zone_name,
                Schedule_Name="LightingSchedule",
                Design_Level_Calculation_Method="Watts/Area",
                Watts_per_Zone_Floor_Area=8,
                Return_Air_Fraction=0.2,
                Fraction_Radiant=0.6,
                Fraction_Visible=0.2,
            )

            # Equipment (6 W/m², Sveby)
            idf.newidfobject(
                "ELECTRICEQUIPMENT",
                Name=f"{zone_name}_Equipment",
                Zone_or_ZoneList_Name=zone_name,
                Schedule_Name="EquipmentSchedule",
                Design_Level_Calculation_Method="Watts/Area",
                Watts_per_Zone_Floor_Area=6,
                Fraction_Latent=0,
                Fraction_Radiant=0.3,
                Fraction_Lost=0,
            )

            # Infiltration
            idf.newidfobject(
                "ZONEINFILTRATION:DESIGNFLOWRATE",
                Name=f"{zone_name}_Infiltration",
                Zone_or_ZoneList_Name=zone_name,
                Schedule_Name="AlwaysOn",
                Design_Flow_Rate_Calculation_Method="AirChanges/Hour",
                Air_Changes_per_Hour=env.infiltration_ach,
            )

    def _apply_ep_bug_workaround(self, idf: IDF) -> None:
        """
        Apply EnergyPlus 25.1.0 bug workaround.

        Bug: Using 'None' for Dehumidification/Humidification Control Type
        causes segmentation fault. Must use 'ConstantSupplyHumidityRatio'.

        This is already handled in _apply_hvac, but verify here.
        """
        for ideal_loads in idf.idfobjects["ZONEHVAC:IDEALLOADSAIRSYSTEM"]:
            if ideal_loads.Dehumidification_Control_Type == "None":
                ideal_loads.Dehumidification_Control_Type = "ConstantSupplyHumidityRatio"
            if ideal_loads.Humidification_Control_Type == "None":
                ideal_loads.Humidification_Control_Type = "ConstantSupplyHumidityRatio"

    def _set_simulation_params(self, idf: IDF, latitude: float, longitude: float) -> None:
        """Set simulation control parameters."""
        # These should already exist in template, but ensure correct values

        # Update Site:Location
        for loc in idf.idfobjects["SITE:LOCATION"]:
            loc.Latitude = latitude
            loc.Longitude = longitude
            loc.Time_Zone = 1.0  # CET
            loc.Elevation = 44  # Stockholm average

    def _add_outputs(self, idf: IDF) -> None:
        """Add output variables for results parsing."""
        outputs = [
            ("Zone Ideal Loads Zone Total Heating Energy", "Hourly"),
            ("Zone Ideal Loads Zone Total Cooling Energy", "Hourly"),
            ("Zone Mean Air Temperature", "Hourly"),
        ]

        for var, freq in outputs:
            idf.newidfobject(
                "OUTPUT:VARIABLE",
                Key_Value="*",
                Variable_Name=var,
                Reporting_Frequency=freq,
            )

        idf.newidfobject(
            "OUTPUT:METER",
            Key_Name="Heating:EnergyTransfer",
            Reporting_Frequency="Hourly",
        )

        idf.newidfobject(
            "OUTPUTCONTROL:TABLE:STYLE",
            Column_Separator="Comma",
            Unit_Conversion="JtoKWH",
        )

        idf.newidfobject(
            "OUTPUT:TABLE:SUMMARYREPORTS",
            Report_1_Name="AllSummary",
        )

    def _select_weather_file(self, latitude: float) -> str:
        """Select Swedish weather file by latitude."""
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

    def _estimate_heating(self, floor_area_m2: float, archetype: SwedishArchetype) -> float:
        """Rough estimate of heating demand (placeholder for surrogate)."""
        # Simple era-based estimate
        era_estimates = {
            "pre_1945": 120,
            "1945_1960": 100,
            "1961_1975": 90,
            "1976_1985": 70,
            "1986_1995": 55,
            "1996_2010": 45,
            "2011_plus": 30,
        }
        return era_estimates.get(archetype.era_id, 60)
```

#### Step 3: Create Minimal IDF Template

Create `src/baseline/templates/minimal_v25.1.idf`:

```
!- Minimal template for GeomEppy (EnergyPlus 25.1)
!- Only includes simulation settings and schedules
!- Geometry, constructions, HVAC added by generator

Version,25.1;

Building,
    Building,
    0,
    City,
    0.04,
    0.4,
    FullExterior,
    25,
    6;

Timestep,4;

HeatBalanceAlgorithm,ConductionTransferFunction;
SurfaceConvectionAlgorithm:Inside,TARP;
SurfaceConvectionAlgorithm:Outside,DOE-2;

SimulationControl,
    No,No,No,No,Yes;

RunPeriod,
    Annual,1,1,,12,31,,Sunday,No,No,No,Yes,Yes;

Site:Location,
    Stockholm,59.35,17.95,1.0,44;

GlobalGeometryRules,
    UpperLeftCorner,Counterclockwise,Relative;

Site:GroundTemperature:BuildingSurface,
    5,5,6,8,11,14,16,16,14,11,8,5;

!- Schedules (Sveby)
ScheduleTypeLimits,Fraction,0,1,Continuous;
ScheduleTypeLimits,Temperature,-50,50,Continuous;
ScheduleTypeLimits,Any Number,,,Continuous;

Schedule:Constant,AlwaysOn,Fraction,1;
Schedule:Constant,HeatSet,Temperature,21;
Schedule:Constant,CoolSet,Temperature,50;
Schedule:Constant,ThermType,Any Number,4;
Schedule:Constant,ActivityLevel,Any Number,120;

Schedule:Compact,
    OccupancySchedule,Fraction,
    Through: 12/31,For: AllDays,
    Until: 07:00,0.9,Until: 09:00,0.5,Until: 17:00,0.2,
    Until: 22:00,0.9,Until: 24:00,0.9;

Schedule:Compact,
    LightingSchedule,Fraction,
    Through: 12/31,For: AllDays,
    Until: 06:00,0.05,Until: 08:00,0.5,Until: 17:00,0.1,
    Until: 22:00,0.7,Until: 24:00,0.3;

Schedule:Compact,
    EquipmentSchedule,Fraction,
    Through: 12/31,For: AllDays,
    Until: 07:00,0.3,Until: 09:00,0.6,Until: 17:00,0.3,
    Until: 22:00,0.7,Until: 24:00,0.4;
```

#### Step 4: Update Integration Points

Modify `src/baseline/__init__.py` to export new generator:

```python
from .generator_v2 import GeomEppyGenerator, BaselineModelV2
from .generator import BaselineGenerator, BaselineModel  # Keep for backwards compat

# Default to v2
generate_baseline = GeomEppyGenerator().generate
```

#### Step 5: Update BuildingGeometry to Preserve Footprint

Modify `src/geometry/building_geometry.py` to store raw footprint coordinates:

```python
@dataclass
class BuildingGeometry:
    # Existing fields...

    # NEW: Raw footprint for GeomEppy
    footprint_coords: List[Tuple[float, float]] = None  # (x, y) in meters or (lon, lat)
    footprint_crs: str = "EPSG:4326"  # Coordinate reference system
```

#### Step 6: Migration and Testing

1. Create tests comparing v1 and v2 output for same inputs
2. Verify E+ simulation results are within 5% for simple rectangular buildings
3. Test with actual MS Building Footprints irregular polygons
4. Verify E+ 25.1.0 bug workaround is applied correctly

---

## Part 2: Surrogate-Based Bayesian Calibration

### 2.1 Objective

Replace deterministic calibration (`src/baseline/calibrator.py`) with uncertainty-quantified Bayesian approach that:
- Provides confidence intervals on all predictions
- Learns from archetype-level data
- Enables fast calibration via surrogate models (not repeated E+ runs)

### 2.2 Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    CALIBRATION PIPELINE                          │
│                                                                  │
│  ONE-TIME (per archetype):                                       │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ 1. DESIGN OF EXPERIMENTS                                    │ │
│  │    - Latin Hypercube sampling of parameter space            │ │
│  │    - 100 samples per archetype                              │ │
│  │    - Parameters: infiltration, wall_u, roof_u, floor_u,     │ │
│  │                  window_u, heat_recovery, setpoint          │ │
│  └────────────────────────────────────────────────────────────┘ │
│                          ↓                                       │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ 2. E+ SIMULATION BATCH                                      │ │
│  │    - Run all 100 samples (~2 hours per archetype)           │ │
│  │    - Extract heating_kwh_m2 for each                        │ │
│  │    - Store results in database                              │ │
│  └────────────────────────────────────────────────────────────┘ │
│                          ↓                                       │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ 3. TRAIN SURROGATE MODEL                                    │ │
│  │    - Gaussian Process Regressor                             │ │
│  │    - Input: 7 parameters                                    │ │
│  │    - Output: heating_kwh_m2                                 │ │
│  │    - Save model per archetype                               │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  PER-BUILDING (fast):                                            │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ 4. BAYESIAN INFERENCE                                       │ │
│  │    - Load archetype surrogate                               │ │
│  │    - Define prior from archetype defaults                   │ │
│  │    - ABC-SMC with measured energy as target                 │ │
│  │    - Output: posterior distributions for all parameters     │ │
│  └────────────────────────────────────────────────────────────┘ │
│                          ↓                                       │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ 5. UNCERTAINTY PROPAGATION                                  │ │
│  │    - Sample from posteriors                                 │ │
│  │    - Predict ECM savings for each sample                    │ │
│  │    - Output: savings with confidence intervals              │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 2.3 Implementation Steps

#### Step 1: Add Dependencies

```toml
# pyproject.toml
[project.dependencies]
scikit-learn = ">=1.3.0"  # Gaussian Process
pyabc = ">=0.12.0"        # ABC-SMC implementation
# OR
pymc = ">=5.0.0"          # Full Bayesian (heavier)
```

#### Step 2: Create Surrogate Training Module

Create `src/calibration/surrogate.py`:

```python
"""
Surrogate model training for fast Bayesian calibration.

Trains Gaussian Process models on E+ simulation results,
enabling instant predictions for parameter combinations.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
import logging

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
from sklearn.preprocessing import StandardScaler
from scipy.stats import qmc
import joblib

logger = logging.getLogger(__name__)


@dataclass
class SurrogateConfig:
    """Configuration for surrogate model training."""
    n_samples: int = 100  # Latin Hypercube samples
    random_state: int = 42

    # Parameter bounds (Swedish buildings)
    param_bounds: Dict[str, Tuple[float, float]] = None

    def __post_init__(self):
        if self.param_bounds is None:
            self.param_bounds = {
                'infiltration_ach': (0.02, 0.20),
                'wall_u_value': (0.15, 1.50),
                'roof_u_value': (0.10, 0.60),
                'floor_u_value': (0.15, 0.80),
                'window_u_value': (0.70, 2.50),
                'heat_recovery_eff': (0.0, 0.90),
                'heating_setpoint': (18.0, 23.0),
            }


@dataclass
class TrainedSurrogate:
    """Trained surrogate model with metadata."""
    archetype_id: str
    gp_model: GaussianProcessRegressor
    scaler_X: StandardScaler
    scaler_y: StandardScaler
    param_names: List[str]
    param_bounds: Dict[str, Tuple[float, float]]
    training_r2: float
    training_rmse: float
    n_training_samples: int


class SurrogateTrainer:
    """
    Train Gaussian Process surrogate models for each archetype.

    Workflow:
    1. Generate Latin Hypercube samples of parameter space
    2. Run E+ simulations for all samples
    3. Train GP regressor on results
    4. Save model for fast inference
    """

    def __init__(self, config: SurrogateConfig = None):
        self.config = config or SurrogateConfig()
        self.param_names = list(self.config.param_bounds.keys())

    def generate_samples(self) -> np.ndarray:
        """Generate Latin Hypercube samples of parameter space."""
        n_params = len(self.param_names)

        # Latin Hypercube sampling (better coverage than random)
        sampler = qmc.LatinHypercube(d=n_params, seed=self.config.random_state)
        samples_unit = sampler.random(n=self.config.n_samples)

        # Scale to parameter bounds
        lower = np.array([self.config.param_bounds[p][0] for p in self.param_names])
        upper = np.array([self.config.param_bounds[p][1] for p in self.param_names])
        samples = qmc.scale(samples_unit, lower, upper)

        return samples

    def samples_to_dicts(self, samples: np.ndarray) -> List[Dict[str, float]]:
        """Convert sample array to list of parameter dicts."""
        return [
            {name: samples[i, j] for j, name in enumerate(self.param_names)}
            for i in range(samples.shape[0])
        ]

    def train(
        self,
        archetype_id: str,
        X: np.ndarray,  # Parameter samples (n_samples, n_params)
        y: np.ndarray,  # E+ results: heating_kwh_m2 (n_samples,)
    ) -> TrainedSurrogate:
        """
        Train Gaussian Process surrogate on simulation results.

        Args:
            archetype_id: Identifier for this archetype
            X: Parameter samples (n_samples, n_params)
            y: Corresponding heating_kwh_m2 results

        Returns:
            TrainedSurrogate ready for inference
        """
        logger.info(f"Training surrogate for {archetype_id} with {len(X)} samples")

        # Standardize inputs and outputs
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()

        X_scaled = scaler_X.fit_transform(X)
        y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()

        # Define GP kernel (Matern is robust for physical systems)
        kernel = (
            ConstantKernel(1.0, (1e-3, 1e3)) *
            Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e2), nu=2.5) +
            WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-5, 1e1))
        )

        # Train GP
        gp = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=10,
            normalize_y=False,  # Already normalized
            random_state=self.config.random_state,
        )
        gp.fit(X_scaled, y_scaled)

        # Evaluate training performance
        y_pred_scaled = gp.predict(X_scaled)
        y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()

        r2 = 1 - np.sum((y - y_pred)**2) / np.sum((y - y.mean())**2)
        rmse = np.sqrt(np.mean((y - y_pred)**2))

        logger.info(f"Training R²: {r2:.4f}, RMSE: {rmse:.2f} kWh/m²")

        return TrainedSurrogate(
            archetype_id=archetype_id,
            gp_model=gp,
            scaler_X=scaler_X,
            scaler_y=scaler_y,
            param_names=self.param_names,
            param_bounds=self.config.param_bounds,
            training_r2=r2,
            training_rmse=rmse,
            n_training_samples=len(X),
        )

    def save(self, surrogate: TrainedSurrogate, output_dir: Path) -> Path:
        """Save trained surrogate to disk."""
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / f"surrogate_{surrogate.archetype_id}.joblib"
        joblib.dump(surrogate, path)
        logger.info(f"Saved surrogate to {path}")
        return path

    @staticmethod
    def load(path: Path) -> TrainedSurrogate:
        """Load trained surrogate from disk."""
        return joblib.load(path)


class SurrogatePredictor:
    """
    Fast predictions using trained surrogate.

    Provides both point predictions and uncertainty estimates.
    """

    def __init__(self, surrogate: TrainedSurrogate):
        self.surrogate = surrogate

    def predict(
        self,
        params: Dict[str, float],
        return_std: bool = False,
    ) -> Tuple[float, Optional[float]]:
        """
        Predict heating_kwh_m2 for given parameters.

        Args:
            params: Parameter dict
            return_std: Whether to return uncertainty estimate

        Returns:
            (prediction, std) if return_std else prediction
        """
        # Convert to array in correct order
        X = np.array([[params[name] for name in self.surrogate.param_names]])
        X_scaled = self.surrogate.scaler_X.transform(X)

        if return_std:
            y_scaled, std_scaled = self.surrogate.gp_model.predict(X_scaled, return_std=True)
            y = self.surrogate.scaler_y.inverse_transform(y_scaled.reshape(-1, 1)).ravel()[0]
            # Approximate std transformation
            std = std_scaled[0] * self.surrogate.scaler_y.scale_[0]
            return y, std
        else:
            y_scaled = self.surrogate.gp_model.predict(X_scaled)
            y = self.surrogate.scaler_y.inverse_transform(y_scaled.reshape(-1, 1)).ravel()[0]
            return y

    def predict_batch(
        self,
        params_list: List[Dict[str, float]],
    ) -> np.ndarray:
        """Predict for multiple parameter sets (vectorized)."""
        X = np.array([
            [params[name] for name in self.surrogate.param_names]
            for params in params_list
        ])
        X_scaled = self.surrogate.scaler_X.transform(X)
        y_scaled = self.surrogate.gp_model.predict(X_scaled)
        y = self.surrogate.scaler_y.inverse_transform(y_scaled.reshape(-1, 1)).ravel()
        return y
```

#### Step 3: Create Bayesian Calibration Module

Create `src/calibration/bayesian.py`:

```python
"""
Bayesian calibration using ABC-SMC.

Provides posterior distributions over building parameters
given measured energy consumption.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Callable
import logging

import numpy as np
from scipy import stats

from .surrogate import TrainedSurrogate, SurrogatePredictor

logger = logging.getLogger(__name__)


@dataclass
class Prior:
    """Prior distribution for a parameter."""
    name: str
    distribution: str  # 'normal', 'uniform', 'truncnorm'
    params: Dict[str, float]  # Distribution parameters

    def sample(self, n: int = 1) -> np.ndarray:
        """Sample from prior."""
        if self.distribution == 'normal':
            return np.random.normal(self.params['loc'], self.params['scale'], n)
        elif self.distribution == 'uniform':
            return np.random.uniform(self.params['low'], self.params['high'], n)
        elif self.distribution == 'truncnorm':
            a = (self.params['low'] - self.params['loc']) / self.params['scale']
            b = (self.params['high'] - self.params['loc']) / self.params['scale']
            return stats.truncnorm.rvs(a, b, self.params['loc'], self.params['scale'], n)
        else:
            raise ValueError(f"Unknown distribution: {self.distribution}")

    def pdf(self, x: float) -> float:
        """Probability density at x."""
        if self.distribution == 'normal':
            return stats.norm.pdf(x, self.params['loc'], self.params['scale'])
        elif self.distribution == 'uniform':
            return stats.uniform.pdf(x, self.params['low'],
                                     self.params['high'] - self.params['low'])
        elif self.distribution == 'truncnorm':
            a = (self.params['low'] - self.params['loc']) / self.params['scale']
            b = (self.params['high'] - self.params['loc']) / self.params['scale']
            return stats.truncnorm.pdf(x, a, b, self.params['loc'], self.params['scale'])


@dataclass
class CalibrationPriors:
    """Prior distributions for all calibration parameters."""
    priors: Dict[str, Prior]

    @classmethod
    def from_archetype(cls, archetype_id: str, archetype_defaults: Dict[str, float]):
        """Create priors centered on archetype defaults."""
        # Standard deviations as fraction of range
        param_stds = {
            'infiltration_ach': 0.03,
            'wall_u_value': 0.20,
            'roof_u_value': 0.10,
            'floor_u_value': 0.15,
            'window_u_value': 0.30,
            'heat_recovery_eff': 0.10,
            'heating_setpoint': 1.0,
        }

        param_bounds = {
            'infiltration_ach': (0.02, 0.20),
            'wall_u_value': (0.15, 1.50),
            'roof_u_value': (0.10, 0.60),
            'floor_u_value': (0.15, 0.80),
            'window_u_value': (0.70, 2.50),
            'heat_recovery_eff': (0.0, 0.90),
            'heating_setpoint': (18.0, 23.0),
        }

        priors = {}
        for param, default in archetype_defaults.items():
            if param in param_stds:
                bounds = param_bounds[param]
                priors[param] = Prior(
                    name=param,
                    distribution='truncnorm',
                    params={
                        'loc': default,
                        'scale': param_stds[param],
                        'low': bounds[0],
                        'high': bounds[1],
                    }
                )

        return cls(priors=priors)

    def sample(self) -> Dict[str, float]:
        """Sample all parameters from priors."""
        return {name: prior.sample(1)[0] for name, prior in self.priors.items()}

    def sample_batch(self, n: int) -> List[Dict[str, float]]:
        """Sample n parameter sets."""
        samples = {name: prior.sample(n) for name, prior in self.priors.items()}
        return [
            {name: samples[name][i] for name in self.priors}
            for i in range(n)
        ]


@dataclass
class PosteriorSample:
    """A single sample from the posterior."""
    params: Dict[str, float]
    weight: float = 1.0


@dataclass
class CalibrationPosterior:
    """Posterior distribution from Bayesian calibration."""
    archetype_id: str
    measured_kwh_m2: float
    samples: List[PosteriorSample]

    # Summary statistics (computed lazily)
    _means: Dict[str, float] = field(default_factory=dict)
    _stds: Dict[str, float] = field(default_factory=dict)
    _ci_90: Dict[str, Tuple[float, float]] = field(default_factory=dict)

    def __post_init__(self):
        self._compute_summaries()

    def _compute_summaries(self):
        """Compute posterior summary statistics."""
        param_names = list(self.samples[0].params.keys())
        weights = np.array([s.weight for s in self.samples])
        weights = weights / weights.sum()

        for name in param_names:
            values = np.array([s.params[name] for s in self.samples])
            self._means[name] = np.average(values, weights=weights)
            self._stds[name] = np.sqrt(np.average((values - self._means[name])**2, weights=weights))
            self._ci_90[name] = (
                np.percentile(values, 5),
                np.percentile(values, 95),
            )

    @property
    def means(self) -> Dict[str, float]:
        return self._means

    @property
    def stds(self) -> Dict[str, float]:
        return self._stds

    @property
    def ci_90(self) -> Dict[str, Tuple[float, float]]:
        return self._ci_90

    def sample(self, n: int = 1) -> List[Dict[str, float]]:
        """Sample from posterior (with replacement, weighted)."""
        weights = np.array([s.weight for s in self.samples])
        weights = weights / weights.sum()
        indices = np.random.choice(len(self.samples), size=n, p=weights)
        return [self.samples[i].params for i in indices]

    def to_dict(self) -> Dict:
        """Convert to serializable dict."""
        return {
            'archetype_id': self.archetype_id,
            'measured_kwh_m2': self.measured_kwh_m2,
            'means': self._means,
            'stds': self._stds,
            'ci_90': {k: list(v) for k, v in self._ci_90.items()},
            'n_samples': len(self.samples),
        }


class ABCSMCCalibrator:
    """
    Approximate Bayesian Computation with Sequential Monte Carlo.

    Fast calibration using surrogate model instead of E+ runs.
    """

    def __init__(
        self,
        surrogate: TrainedSurrogate,
        priors: CalibrationPriors,
        measurement_error_kwh_m2: float = 3.0,  # Uncertainty in declaration
    ):
        self.predictor = SurrogatePredictor(surrogate)
        self.priors = priors
        self.measurement_error = measurement_error_kwh_m2

    def calibrate(
        self,
        measured_kwh_m2: float,
        n_particles: int = 1000,
        n_generations: int = 10,
        quantile_epsilon: float = 0.5,  # Keep top 50% each generation
    ) -> CalibrationPosterior:
        """
        Run ABC-SMC calibration.

        Args:
            measured_kwh_m2: Observed energy consumption
            n_particles: Number of parameter samples per generation
            n_generations: Number of SMC iterations
            quantile_epsilon: Fraction of particles to keep each generation

        Returns:
            CalibrationPosterior with parameter distributions
        """
        logger.info(f"Starting ABC-SMC calibration for {measured_kwh_m2:.1f} kWh/m²")

        # Initial generation: sample from prior
        particles = self.priors.sample_batch(n_particles)
        weights = np.ones(n_particles) / n_particles

        for gen in range(n_generations):
            # Predict for all particles
            predictions = self.predictor.predict_batch(particles)

            # Compute distances to observation
            distances = np.abs(predictions - measured_kwh_m2)

            # Determine epsilon (acceptance threshold)
            epsilon = np.quantile(distances, quantile_epsilon)

            # Accept particles below threshold
            accepted_mask = distances <= epsilon
            n_accepted = accepted_mask.sum()

            logger.debug(f"Generation {gen}: ε={epsilon:.2f}, accepted={n_accepted}/{n_particles}")

            if n_accepted < 10:
                logger.warning(f"Too few particles accepted, stopping early")
                break

            # Resample and perturb for next generation
            accepted_indices = np.where(accepted_mask)[0]

            if gen < n_generations - 1:
                # Resample with replacement
                resampled_indices = np.random.choice(
                    accepted_indices,
                    size=n_particles,
                    replace=True
                )

                # Perturb using kernel (Gaussian around accepted values)
                new_particles = []
                for idx in resampled_indices:
                    base = particles[idx]
                    perturbed = {}
                    for name, value in base.items():
                        std = self.priors.priors[name].params.get('scale', 0.1) * 0.5
                        bounds = (
                            self.priors.priors[name].params.get('low', -np.inf),
                            self.priors.priors[name].params.get('high', np.inf),
                        )
                        new_val = np.clip(
                            value + np.random.normal(0, std),
                            bounds[0], bounds[1]
                        )
                        perturbed[name] = new_val
                    new_particles.append(perturbed)

                particles = new_particles
                weights = np.ones(n_particles) / n_particles
            else:
                # Final generation: compute importance weights
                particles = [particles[i] for i in accepted_indices]

                # Weight by likelihood (Gaussian around observation)
                likelihoods = stats.norm.pdf(
                    predictions[accepted_mask],
                    loc=measured_kwh_m2,
                    scale=self.measurement_error
                )
                weights = likelihoods / likelihoods.sum()

        # Create posterior samples
        posterior_samples = [
            PosteriorSample(params=p, weight=w)
            for p, w in zip(particles, weights)
        ]

        posterior = CalibrationPosterior(
            archetype_id=self.predictor.surrogate.archetype_id,
            measured_kwh_m2=measured_kwh_m2,
            samples=posterior_samples,
        )

        logger.info(f"Calibration complete: {len(posterior_samples)} posterior samples")
        for name, mean in posterior.means.items():
            ci = posterior.ci_90[name]
            logger.info(f"  {name}: {mean:.3f} (90% CI: {ci[0]:.3f} - {ci[1]:.3f})")

        return posterior


class UncertaintyPropagator:
    """
    Propagate calibration uncertainty to ECM savings predictions.
    """

    def __init__(
        self,
        baseline_surrogate: TrainedSurrogate,
        ecm_surrogates: Dict[str, TrainedSurrogate],  # ECM ID -> surrogate
    ):
        self.baseline_predictor = SurrogatePredictor(baseline_surrogate)
        self.ecm_predictors = {
            ecm_id: SurrogatePredictor(surr)
            for ecm_id, surr in ecm_surrogates.items()
        }

    def predict_savings(
        self,
        posterior: CalibrationPosterior,
        ecm_id: str,
        n_samples: int = 1000,
    ) -> Dict:
        """
        Predict ECM savings with uncertainty.

        Args:
            posterior: Calibrated parameter posterior
            ecm_id: ECM to evaluate
            n_samples: Monte Carlo samples

        Returns:
            Dict with mean, std, ci_90 for savings (fraction and kWh/m²)
        """
        if ecm_id not in self.ecm_predictors:
            raise ValueError(f"No surrogate for ECM: {ecm_id}")

        ecm_predictor = self.ecm_predictors[ecm_id]

        # Sample from posterior
        param_samples = posterior.sample(n_samples)

        # Predict baseline and ECM for each sample
        baselines = self.baseline_predictor.predict_batch(param_samples)
        with_ecms = ecm_predictor.predict_batch(param_samples)

        # Compute savings
        savings_kwh_m2 = baselines - with_ecms
        savings_fraction = savings_kwh_m2 / baselines

        return {
            'ecm_id': ecm_id,
            'savings_fraction': {
                'mean': float(np.mean(savings_fraction)),
                'std': float(np.std(savings_fraction)),
                'ci_90': (float(np.percentile(savings_fraction, 5)),
                          float(np.percentile(savings_fraction, 95))),
            },
            'savings_kwh_m2': {
                'mean': float(np.mean(savings_kwh_m2)),
                'std': float(np.std(savings_kwh_m2)),
                'ci_90': (float(np.percentile(savings_kwh_m2, 5)),
                          float(np.percentile(savings_kwh_m2, 95))),
            },
            'baseline_kwh_m2': {
                'mean': float(np.mean(baselines)),
                'ci_90': (float(np.percentile(baselines, 5)),
                          float(np.percentile(baselines, 95))),
            },
        }
```

#### Step 4: Create Batch Training Pipeline

Create `src/calibration/train_surrogates.py`:

```python
"""
Batch training script for surrogate models.

Run once to generate surrogates for all archetypes.
Estimated time: ~3 days on 8-core machine for 40 archetypes × 100 samples.
"""

import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import logging
import json

from ..baseline.archetypes import get_all_archetypes
from ..baseline.generator_v2 import GeomEppyGenerator
from ..simulation import SimulationRunner, ResultsParser
from .surrogate import SurrogateTrainer, SurrogateConfig

logger = logging.getLogger(__name__)


def train_archetype_surrogate(
    archetype_id: str,
    output_dir: Path,
    weather_file: Path,
    n_samples: int = 100,
) -> Path:
    """
    Train surrogate for a single archetype.

    1. Generate parameter samples
    2. Run E+ simulations
    3. Train GP model
    4. Save to disk
    """
    logger.info(f"Training surrogate for {archetype_id}")

    archetype_dir = output_dir / archetype_id
    archetype_dir.mkdir(parents=True, exist_ok=True)

    # Initialize components
    trainer = SurrogateTrainer(SurrogateConfig(n_samples=n_samples))
    generator = GeomEppyGenerator()
    runner = SimulationRunner()
    parser = ResultsParser()

    # Get archetype defaults
    archetypes = get_all_archetypes()
    archetype = archetypes[archetype_id]

    # Generate parameter samples
    samples = trainer.generate_samples()
    param_dicts = trainer.samples_to_dicts(samples)

    # Reference geometry (simple rectangle for surrogate training)
    ref_footprint = [(0, 0), (20, 0), (20, 40), (0, 40)]
    ref_height = 12.0
    ref_stories = 4
    ref_wwr = {'N': 0.15, 'S': 0.25, 'E': 0.20, 'W': 0.20}

    # Run simulations
    results = []
    for i, params in enumerate(param_dicts):
        logger.info(f"  Sample {i+1}/{n_samples}")

        # Update archetype with sampled parameters
        modified_archetype = archetype.copy()
        modified_archetype.envelope.infiltration_ach = params['infiltration_ach']
        modified_archetype.envelope.wall_u_value = params['wall_u_value']
        modified_archetype.envelope.roof_u_value = params['roof_u_value']
        modified_archetype.envelope.floor_u_value = params['floor_u_value']
        modified_archetype.envelope.window_u_value = params['window_u_value']
        modified_archetype.hvac.heat_recovery_efficiency = params['heat_recovery_eff']
        # setpoint handled separately in IDF

        # Generate IDF
        idf_path = archetype_dir / f"sample_{i:03d}.idf"
        model = generator.generate(
            footprint_coords=ref_footprint,
            height_m=ref_height,
            num_stories=ref_stories,
            archetype=modified_archetype,
            wwr_by_orientation=ref_wwr,
            output_path=idf_path,
            model_name=f"sample_{i:03d}",
        )

        # Run simulation
        sim_output = archetype_dir / f"sim_{i:03d}"
        sim_result = runner.run(idf_path, weather_file, sim_output)

        if sim_result.success:
            parsed = parser.parse(sim_output)
            results.append(parsed.heating_kwh_m2)
        else:
            logger.warning(f"  Simulation failed for sample {i}")
            results.append(np.nan)

    # Remove failed simulations
    valid_mask = ~np.isnan(results)
    X_valid = samples[valid_mask]
    y_valid = np.array(results)[valid_mask]

    logger.info(f"  Valid simulations: {len(y_valid)}/{n_samples}")

    # Train surrogate
    surrogate = trainer.train(archetype_id, X_valid, y_valid)

    # Save
    model_path = trainer.save(surrogate, output_dir / "models")

    # Save training metadata
    metadata = {
        'archetype_id': archetype_id,
        'n_samples': n_samples,
        'n_valid': len(y_valid),
        'training_r2': surrogate.training_r2,
        'training_rmse': surrogate.training_rmse,
    }
    with open(archetype_dir / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)

    return model_path


def main():
    parser = argparse.ArgumentParser(description="Train surrogate models for all archetypes")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--weather-file", type=Path, required=True)
    parser.add_argument("--n-samples", type=int, default=100)
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--archetypes", nargs="*", help="Specific archetypes (default: all)")

    args = parser.parse_args()

    archetypes = get_all_archetypes()
    archetype_ids = args.archetypes or list(archetypes.keys())

    logger.info(f"Training surrogates for {len(archetype_ids)} archetypes")

    with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {
            executor.submit(
                train_archetype_surrogate,
                arch_id,
                args.output_dir,
                args.weather_file,
                args.n_samples,
            ): arch_id
            for arch_id in archetype_ids
        }

        for future in futures:
            arch_id = futures[future]
            try:
                path = future.result()
                logger.info(f"Completed {arch_id}: {path}")
            except Exception as e:
                logger.error(f"Failed {arch_id}: {e}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
```

#### Step 5: Update Calibrator Interface

Create `src/calibration/calibrator_v2.py`:

```python
"""
Unified calibration interface with Bayesian support.

Provides drop-in replacement for BaselineCalibrator with uncertainty.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple
import logging

from .surrogate import SurrogateTrainer, TrainedSurrogate
from .bayesian import ABCSMCCalibrator, CalibrationPriors, CalibrationPosterior, UncertaintyPropagator

logger = logging.getLogger(__name__)


@dataclass
class CalibrationResultV2:
    """Enhanced calibration result with uncertainty."""
    success: bool

    # Point estimates (for backwards compatibility)
    calibrated_kwh_m2: float
    adjusted_infiltration_ach: float
    adjusted_heat_recovery: float
    adjusted_window_u: float

    # Uncertainty quantification (NEW)
    posterior: Optional[CalibrationPosterior] = None
    calibrated_kwh_m2_ci_90: Tuple[float, float] = None
    parameter_uncertainties: Dict[str, Dict] = None  # {param: {mean, std, ci_90}}

    # Metadata
    measured_kwh_m2: float = 0.0
    archetype_id: str = ""
    calibration_method: str = "bayesian"  # or "deterministic"


class BayesianCalibrator:
    """
    Bayesian calibration using pre-trained surrogates.

    Usage:
        calibrator = BayesianCalibrator(surrogate_dir=Path("./surrogates"))
        result = calibrator.calibrate(
            archetype_id="mfh_1961_1975",
            measured_kwh_m2=85.0,
        )
        print(f"Infiltration: {result.posterior.means['infiltration_ach']:.3f}")
        print(f"90% CI: {result.posterior.ci_90['infiltration_ach']}")
    """

    def __init__(self, surrogate_dir: Path):
        """
        Initialize with directory of pre-trained surrogates.

        Expected structure:
        surrogate_dir/
            models/
                surrogate_mfh_1961_1975.joblib
                surrogate_mfh_1976_1985.joblib
                ...
        """
        self.surrogate_dir = Path(surrogate_dir)
        self.surrogates: Dict[str, TrainedSurrogate] = {}
        self._load_surrogates()

    def _load_surrogates(self):
        """Load all available surrogates."""
        models_dir = self.surrogate_dir / "models"
        if not models_dir.exists():
            logger.warning(f"Surrogate directory not found: {models_dir}")
            return

        for path in models_dir.glob("surrogate_*.joblib"):
            surrogate = SurrogateTrainer.load(path)
            self.surrogates[surrogate.archetype_id] = surrogate
            logger.info(f"Loaded surrogate: {surrogate.archetype_id}")

    def calibrate(
        self,
        archetype_id: str,
        measured_kwh_m2: float,
        archetype_defaults: Optional[Dict[str, float]] = None,
        n_particles: int = 1000,
    ) -> CalibrationResultV2:
        """
        Calibrate building parameters given measured energy.

        Args:
            archetype_id: Archetype identifier
            measured_kwh_m2: Observed energy consumption
            archetype_defaults: Prior centers (uses archetype if None)
            n_particles: ABC-SMC particles (more = slower but more accurate)

        Returns:
            CalibrationResultV2 with posterior distributions
        """
        if archetype_id not in self.surrogates:
            available = list(self.surrogates.keys())
            raise ValueError(f"No surrogate for {archetype_id}. Available: {available}")

        surrogate = self.surrogates[archetype_id]

        # Create priors from archetype defaults
        if archetype_defaults is None:
            # Use surrogate's training center (TODO: load from archetype)
            archetype_defaults = {
                'infiltration_ach': 0.06,
                'wall_u_value': 0.60,
                'roof_u_value': 0.30,
                'floor_u_value': 0.40,
                'window_u_value': 1.20,
                'heat_recovery_eff': 0.70,
                'heating_setpoint': 21.0,
            }

        priors = CalibrationPriors.from_archetype(archetype_id, archetype_defaults)

        # Run ABC-SMC
        calibrator = ABCSMCCalibrator(surrogate, priors)
        posterior = calibrator.calibrate(
            measured_kwh_m2=measured_kwh_m2,
            n_particles=n_particles,
        )

        # Extract point estimates for backwards compatibility
        means = posterior.means

        return CalibrationResultV2(
            success=True,
            calibrated_kwh_m2=measured_kwh_m2,  # Calibrated to match measurement
            adjusted_infiltration_ach=means.get('infiltration_ach', 0.06),
            adjusted_heat_recovery=means.get('heat_recovery_eff', 0.70),
            adjusted_window_u=means.get('window_u_value', 1.0),
            posterior=posterior,
            calibrated_kwh_m2_ci_90=posterior.ci_90.get('heating_kwh_m2', (measured_kwh_m2, measured_kwh_m2)),
            parameter_uncertainties={
                name: {
                    'mean': posterior.means[name],
                    'std': posterior.stds[name],
                    'ci_90': posterior.ci_90[name],
                }
                for name in posterior.means
            },
            measured_kwh_m2=measured_kwh_m2,
            archetype_id=archetype_id,
            calibration_method="bayesian",
        )
```

---

## Part 3: Integration and Migration

### 3.1 Module Structure

After implementation:

```
src/
├── baseline/
│   ├── __init__.py
│   ├── archetypes.py          # Existing
│   ├── generator.py           # DEPRECATED (keep for backwards compat)
│   ├── generator_v2.py        # NEW: GeomEppy-based
│   ├── calibrator.py          # DEPRECATED
│   └── templates/
│       └── minimal_v25.1.idf  # NEW: GeomEppy template
│
├── calibration/               # NEW MODULE
│   ├── __init__.py
│   ├── surrogate.py           # GP surrogate training
│   ├── bayesian.py            # ABC-SMC calibration
│   ├── calibrator_v2.py       # Unified interface
│   └── train_surrogates.py    # Batch training script
│
└── ... (existing modules)
```

### 3.2 Migration Path

**Phase 1: GeomEppy (Week 1)**
1. Add geomeppy dependency
2. Implement generator_v2.py
3. Create minimal template IDF
4. Update BuildingGeometry to preserve footprint
5. Test against existing generator
6. Update address_pipeline to use v2

**Phase 2: Surrogate Training (Week 2)**
1. Add scikit-learn, pyabc dependencies
2. Implement surrogate.py
3. Run batch training for all 40 archetypes
4. Validate surrogate accuracy (R² > 0.95 target)
5. Store trained models

**Phase 3: Bayesian Calibration (Week 3)**
1. Implement bayesian.py
2. Implement calibrator_v2.py
3. Add uncertainty propagation to ECM analysis
4. Update reporting to show confidence intervals
5. Update CLI and API

### 3.3 Testing Requirements

1. **GeomEppy Tests**
   - Compare v1 and v2 output for rectangular buildings (should be within 5%)
   - Test with actual MS Building Footprints polygons
   - Verify E+ 25.1.0 bug workaround is applied

2. **Surrogate Tests**
   - Training R² > 0.95 for each archetype
   - Cross-validation RMSE < 5 kWh/m²
   - Prediction speed > 10,000 samples/second

3. **Bayesian Tests**
   - Posterior covers true value in synthetic tests
   - Confidence intervals have correct coverage (90% CI should contain true value 90% of time)
   - Convergence diagnostics (effective sample size, acceptance rate)

### 3.4 Compute Resources

**Surrogate Training (one-time):**
- 40 archetypes × 100 samples × 60s/run = ~67 hours
- With 8 parallel workers: ~8 hours
- Storage: ~4GB for all models

**Per-Building Calibration (ongoing):**
- ABC-SMC with 1000 particles: <5 seconds
- Uncertainty propagation for 22 ECMs: <1 second

---

## Part 4: Success Criteria

### 4.1 Quantitative Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Surrogate R² | > 0.95 | Cross-validation on held-out samples |
| Surrogate RMSE | < 5 kWh/m² | Cross-validation |
| Calibration time | < 10s | Wall clock per building |
| CI coverage | 85-95% | Synthetic test with known parameters |
| Code reduction | > 50% | Lines in generator module |

### 4.2 Qualitative Criteria

1. **All existing tests pass** after migration
2. **CLI and API** produce identical point estimates (within 5%)
3. **Reports** include confidence intervals for all predictions
4. **Documentation** updated with new calibration workflow

### 4.3 Deliverables

1. `src/baseline/generator_v2.py` - GeomEppy-based IDF generation
2. `src/calibration/` - Complete Bayesian calibration module
3. Pre-trained surrogates for all 40 archetypes
4. Updated CLI with `--uncertainty` flag
5. Updated API with uncertainty fields in response
6. Test suite for new modules
7. Migration guide for existing users

---

---

## Part 5: Expanded ECM Catalog (CityBES-Inspired)

### 5.1 Current State

Raiden has **22 ECMs** in 6 categories:
- Envelope (5): wall_external_insulation, wall_internal_insulation, roof_insulation, window_replacement, air_sealing
- HVAC (4): ftx_upgrade, ftx_installation, demand_controlled_ventilation, heat_pump_integration
- Renewable (1): solar_pv
- Controls (2): smart_thermostats, led_lighting
- Operational (10): duc_calibration, effektvakt_optimization, heating_curve_adjustment, ventilation_schedule_optimization, radiator_balancing, night_setback, summer_bypass, hot_water_temperature, pump_optimization, bms_optimization

### 5.2 Target State

Expand to **50+ ECMs** based on:
- [CityBES/CBES](https://cbes.lbl.gov/) (80+ ECMs for US commercial buildings)
- [TABULA/EPISCOPE](https://episcope.eu/) (European residential archetypes)
- Swedish building regulations (BBR, Boverket)
- District heating optimization (Swedish-specific)

### 5.3 New ECMs to Add

#### ENVELOPE (Add 8 → Total 13)

| ID | Name | Name (SV) | Description | Typical Savings |
|----|------|-----------|-------------|-----------------|
| `basement_insulation` | Basement Insulation | Källarisolering | Insulate basement ceiling or walls to reduce ground losses | 5-10% heating |
| `door_replacement` | Entrance Door Replacement | Dörrbytte | Replace entrance doors with insulated, airtight doors | 2-5% heating |
| `window_film` | Window Film | Fönsterfilm | Apply low-e film to existing windows (low-cost alternative to replacement) | 5-10% heating |
| `thermal_bridge_remediation` | Thermal Bridge Fix | Köldbryggsåtgärd | Address thermal bridges at balconies, connections | 5-15% heating |
| `attic_hatch_insulation` | Attic Hatch Insulation | Vindslucksisolering | Insulate and seal attic access hatches | 1-2% heating |
| `pipe_insulation` | Pipe Insulation | Rörisolering | Insulate heating/DHW pipes in unheated spaces | 2-5% heating |
| `floor_insulation` | Ground Floor Insulation | Golvisolering | Add insulation to ground floor (crawl space or slab edge) | 5-10% heating |
| `facade_renovation` | Facade Renovation Package | Fasadrenovering | Complete facade renovation with insulation + windows + air sealing | 30-50% heating |

#### HVAC (Add 12 → Total 16)

| ID | Name | Name (SV) | Description | Typical Savings |
|----|------|-----------|-------------|-----------------|
| `exhaust_air_heat_pump` | Exhaust Air Heat Pump | Frånluftsvärmepump | Extract heat from exhaust air for heating/DHW (FVP) | 40-60% heating |
| `ground_source_heat_pump` | Ground Source Heat Pump | Bergvärmepump | Install GSHP with borehole(s) | 60-70% heating |
| `air_source_heat_pump` | Air Source Heat Pump | Luftvärmepump | Install ASHP for heating supplement | 40-50% heating |
| `district_heating_optimization` | DH Substation Optimization | Fjärrvärmeoptimering | Optimize district heating substation (return temp, delta-T) | 5-10% cost |
| `radiator_fans` | Radiator Fans | Radiatorfläktar | Add fans behind radiators to improve heat transfer, enable lower supply temps | 5-10% heating |
| `underfloor_heating_conversion` | Underfloor Heating | Golvvärmesystem | Convert to low-temp underfloor heating (enables heat pump) | 10-20% heating |
| `heat_recovery_dhw` | DHW Heat Recovery | Spillvåttenvärmeåtervinning | Recover heat from shower/bath drain water | 5-10% DHW |
| `vrf_system` | VRF HVAC System | VRF-system | Variable Refrigerant Flow system for heating/cooling | 20-40% HVAC |
| `economizer` | Air Economizer | Ekonomiser | Use outdoor air for free cooling when conditions allow | 10-30% cooling |
| `energy_recovery_ventilation` | ERV Installation | ERV-installation | Energy Recovery Ventilation with latent heat recovery | 25-40% heating |
| `fan_coil_units` | Fan Coil Units | Fläktkonvektorer | Replace radiators with fan coils for faster response | 5-10% heating |
| `chiller_upgrade` | Chiller Efficiency Upgrade | Kylmaskinsuppgradering | Replace old chiller with high-efficiency unit (for cooled buildings) | 20-40% cooling |

#### RENEWABLE (Add 4 → Total 5)

| ID | Name | Name (SV) | Description | Typical Savings |
|----|------|-----------|-------------|-----------------|
| `solar_thermal` | Solar Thermal Collectors | Solfångare | Install solar thermal for DHW preheating | 30-50% DHW |
| `solar_pv_facade` | Facade-Integrated PV | Fasadsolceller | Building-integrated PV on facades (lower yield but more area) | 10-15% electricity |
| `battery_storage` | Battery Storage | Batterilagring | Add battery storage for PV self-consumption and peak shaving | Cost reduction |
| `wind_turbine_small` | Small Wind Turbine | Småskalig vindkraft | Small rooftop or building-mounted wind turbine | 5-15% electricity |

#### CONTROLS (Add 8 → Total 10)

| ID | Name | Name (SV) | Description | Typical Savings |
|----|------|-----------|-------------|-----------------|
| `occupancy_sensors` | Occupancy Sensors | Närvarosensorer | Motion/occupancy sensors for lighting and HVAC | 10-30% lighting |
| `daylight_sensors` | Daylight Harvesting | Dagsljusstyrning | Photocell-based lighting dimming based on daylight | 20-40% lighting |
| `co2_sensors` | CO2 Sensors | CO2-sensorer | CO2-based demand controlled ventilation | 10-20% heating |
| `humidity_sensors` | Humidity Sensors | Fuktsensorer | Humidity-based ventilation control (kitchens, bathrooms) | 5-15% heating |
| `smart_meter_feedback` | Energy Display/Feedback | Energivisning | Real-time energy display to occupants | 5-10% all |
| `automated_blinds` | Automated Blinds/Shades | Automatiska persienner | Motorized blinds with solar/occupancy control | 5-15% cooling/heating |
| `predictive_control` | Predictive HVAC Control | Prediktiv styrning | Weather-predictive heating optimization | 5-10% heating |
| `fault_detection` | Fault Detection & Diagnostics | Feldetektering (FDD) | Automated detection of HVAC faults and inefficiencies | 5-15% HVAC |

#### LIGHTING (Add 4 → Total 5)

| ID | Name | Name (SV) | Description | Typical Savings |
|----|------|-----------|-------------|-----------------|
| `led_common_areas` | LED Common Areas | LED trapphus/källare | Replace lighting in stairwells, basement, garage | 50-70% common lighting |
| `led_outdoor` | LED Outdoor Lighting | LED utomhusbelysning | Replace outdoor/parking lighting with LED | 50-70% outdoor lighting |
| `task_lighting` | Task Lighting | Arbetsplatsbelysning | Localized task lighting to reduce ambient levels | 20-30% lighting |
| `delamping` | Delamping/Reduced Fixtures | Borttagning av armaturer | Remove unnecessary lighting fixtures | 10-30% lighting |

#### SERVICE HOT WATER (New Category - Add 6)

| ID | Name | Name (SV) | Description | Typical Savings |
|----|------|-----------|-------------|-----------------|
| `low_flow_fixtures` | Low-Flow Fixtures | Snålspolande armaturer | Install low-flow showerheads and faucets | 20-40% DHW |
| `dhw_circulation_optimization` | DHW Circulation Optimization | VVC-optimering | Optimize circulation pump schedules and temperature | 10-20% DHW |
| `heat_pump_water_heater` | Heat Pump Water Heater | Varmvattenpump | Replace electric/gas water heater with heat pump | 50-70% DHW |
| `solar_preheat_dhw` | Solar DHW Preheat | Solförvärmning VV | Solar thermal system to preheat domestic hot water | 30-50% DHW |
| `dhw_tank_insulation` | DHW Tank Insulation | Ackumulatorisolering | Add insulation jacket to hot water tank | 5-10% DHW |
| `instantaneous_water_heater` | Instantaneous Water Heater | Genomströmningsberedare | Replace storage tank with on-demand heater (eliminates standby loss) | 10-20% DHW |

#### PLUG LOADS (New Category - Add 4)

| ID | Name | Name (SV) | Description | Typical Savings |
|----|------|-----------|-------------|-----------------|
| `plug_load_controls` | Smart Plug Controls | Smartpluggar | Timer/occupancy-controlled outlets for equipment | 10-20% plug loads |
| `efficient_appliances` | Efficient Appliances | Energieffektiva vitvaror | Replace old appliances with A+++ rated | 30-50% appliances |
| `computer_power_management` | PC Power Management | Datorströmsparning | Enable sleep/hibernate on computers | 20-40% IT equipment |
| `elevator_modernization` | Elevator Modernization | Hissmodernisering | Replace old elevator with regenerative drive | 30-50% elevator energy |

#### OPERATIONAL (Add 5 → Total 15)

| ID | Name | Name (SV) | Description | Typical Savings |
|----|------|-----------|-------------|-----------------|
| `commissioning` | Recommissioning | Ominjustering | Comprehensive system tune-up and optimization | 5-15% all |
| `energy_monitoring` | Energy Monitoring System | Energiövervakningssystem | Install submetering and monitoring platform | 5-10% all (via awareness) |
| `maintenance_optimization` | Preventive Maintenance | Förebyggande underhåll | Optimize maintenance schedules (filter changes, etc.) | 3-5% HVAC |
| `setpoint_adjustment` | Setpoint Optimization | Börvärdesoptimering | Review and optimize all temperature setpoints | 3-5% heating/cooling |
| `weather_compensation` | Weather Compensation | Utetemperaturkompensering | Adjust heating based on weather forecast (not just outdoor temp) | 3-5% heating |

### 5.4 ECM Dependency Matrix

Some ECMs should be bundled or have dependencies:

```
┌─────────────────────────────────────────────────────────────────┐
│                    ECM DEPENDENCIES                              │
│                                                                  │
│  ground_source_heat_pump                                        │
│    └── REQUIRES: underfloor_heating_conversion OR radiator_fans │
│         (for low supply temperature compatibility)              │
│                                                                  │
│  exhaust_air_heat_pump                                          │
│    └── CONFLICTS: ftx_installation (uses same exhaust air)     │
│                                                                  │
│  solar_pv + battery_storage                                     │
│    └── SYNERGY: battery improves PV economics significantly    │
│                                                                  │
│  facade_renovation                                               │
│    └── INCLUDES: wall_external_insulation + window_replacement │
│                  + air_sealing (bundled package)                │
│                                                                  │
│  demand_controlled_ventilation                                   │
│    └── REQUIRES: co2_sensors OR humidity_sensors               │
│                                                                  │
│  predictive_control                                              │
│    └── REQUIRES: smart_thermostats + energy_monitoring         │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 5.5 Swedish-Specific Considerations

#### District Heating Optimization
Sweden has 50%+ district heating penetration. Specific measures:

```python
"district_heating_optimization": ECM(
    id="district_heating_optimization",
    name="DH Substation Optimization",
    name_sv="Fjärrvärmeundercentralsoptimering",
    category=ECMCategory.HVAC,
    description="""
    Optimize district heating substation for:
    - Lower return temperature (fjärrvärmebolag often penalize high return)
    - Higher delta-T (more efficient heat transfer)
    - Two-stage DHW preheating (use return water)
    - Correct sizing of heat exchangers

    Target: Return temp < 35°C, Delta-T > 40°C
    """,
    parameters=[
        ECMParameter("target_return_temp_c", [30, 35, 40], "°C", "Target return temperature"),
    ],
    constraints=[
        ECMConstraint("heating_system", "eq", "district", "Requires district heating"),
    ],
    cost_per_unit=0,
    cost_unit="building",
    fixed_cost=15000,  # Substation adjustment + consultant
    typical_savings_percent=10,  # Cost savings from better tariff
    affected_end_use="heating",
    disruption_level="low",
    typical_lifetime_years=10,
)
```

#### Exhaust Air Heat Pump (Frånluftsvärmepump - FVP)
Very common in Swedish multi-family buildings:

```python
"exhaust_air_heat_pump": ECM(
    id="exhaust_air_heat_pump",
    name="Exhaust Air Heat Pump",
    name_sv="Frånluftsvärmepump (FVP)",
    category=ECMCategory.HVAC,
    description="""
    Extract heat from exhaust ventilation air for heating and DHW.
    Very common retrofit for F-ventilated Swedish buildings.

    NIBE F470, F750 or similar. COP 3.0-3.5.
    Cannot combine with FTX (competing for same heat source).
    """,
    parameters=[
        ECMParameter("capacity_kw", [5, 8, 12, 16], "kW", "Heat pump capacity"),
        ECMParameter("cop", [3.0, 3.2, 3.5], "", "Coefficient of Performance"),
    ],
    constraints=[
        ECMConstraint("ventilation_type", "eq", "f", "Requires F-ventilation"),
        ECMConstraint("has_ftx", "eq", False, "Cannot combine with FTX"),
    ],
    cost_per_unit=15000,  # SEK per kW
    cost_unit="kW",
    typical_savings_percent=50,
    affected_end_use="heating",
    disruption_level="medium",
    typical_lifetime_years=20,
)
```

### 5.6 Implementation Steps

1. **Add new ECM definitions** to `src/ecm/catalog.py`
2. **Update constraint fields** in `BuildingContext` to support new checks
3. **Create IDF modifiers** for each new ECM in `src/ecm/idf_modifier.py`
4. **Add Swedish cost data** to `src/roi/costs_sweden.py`
5. **Update surrogate training** to include new ECMs (more E+ runs)
6. **Update reporting** to show new ECM categories

### 5.7 Priority Order for Implementation

**Phase 1: High-Impact Swedish ECMs (Week 1)**
- `exhaust_air_heat_pump` - Very common Swedish retrofit
- `district_heating_optimization` - Unique to Swedish market
- `ground_source_heat_pump` - High savings potential
- `solar_thermal` - Good DHW supplement

**Phase 2: Envelope Upgrades (Week 2)**
- `basement_insulation`
- `thermal_bridge_remediation`
- `facade_renovation` (package)
- `window_film`

**Phase 3: Controls & Monitoring (Week 3)**
- `occupancy_sensors`
- `daylight_sensors`
- `predictive_control`
- `fault_detection`

**Phase 4: DHW & Plug Loads (Week 4)**
- `low_flow_fixtures`
- `heat_pump_water_heater`
- `plug_load_controls`
- `efficient_appliances`

---

## Part 6: Swedish Cost Databases

### 6.1 Overview of Available Sources

Sweden has several robust cost databases for building renovation and energy efficiency measures. Here's a comprehensive overview:

### 6.2 Primary Cost Databases

#### 1. **Wikells Sektionsfakta** (Commercial, Industry Standard)
**URL:** [wikells.se](https://wikells.se/)

The de-facto standard for Swedish construction cost estimation.

| Product | Coverage | Update Frequency |
|---------|----------|------------------|
| NYB (Nybyggnad) | New construction, ~7,000 articles | Every 2 years (book), 2x/year (digital) |
| ROT (Renovering) | Renovation/retrofit | Every 2 years |
| VS (VVS) | Plumbing, heating | Every 2 years |
| Ventilation | Ventilation systems | Every 2 years |
| EL | Electrical | Every 2 years |

**Access:** Subscription-based (approximately 5,000-15,000 SEK/year depending on package)

**Data format:** PDF/book + proprietary software. No API.

**Raiden integration:** Would require manual extraction or license negotiation.

---

#### 2. **BeBo Lönsamhetskalkyl** (Free, Multi-Family Focused)
**URL:** [bebostad.se/verktyg/bebolonsamhetskalkyl](https://www.bebostad.se/verktyg/bebolonsamhetskalkyl)

Excel-based profitability calculator specifically for multi-family buildings.

**Downloads available:**
- `BeBos lönsamhetskalkyl 1.5.xls` - Main calculator
- User manual (v1.5)
- Worked examples

**ECMs covered with cost data:**
- Window replacement (U-value 1.1 → 0.8 W/m²K)
- FTX heat recovery ventilation
- Facade insulation
- Combined renovation packages

**Cost methodology:** Net present value with initial investment + O&M over lifetime

**Raiden integration:** **HIGH PRIORITY** - Free, Excel format easily parseable, Swedish-specific

---

#### 3. **BeBo Vägledning Typkostnader** (Free, Detailed)
**URL:** [bebostad.se - Vägledning lönsamhet och kostnader](https://www.bebostad.se/projekt/avslutade-projekt/2023/2023-11-vagledning-lonsamhet-och-kostnader-for-energieffektivisering-typkostnader-lonsamhetsexempel)

Detailed cost guidance with example prices from WSP's cost estimation department.

**Contents:**
- Type costs for common ECMs
- Three worked examples with profitability calculations
- Comparison of simplified vs. detailed methods

**Raiden integration:** **HIGH PRIORITY** - Free, authoritative source

---

#### 4. **Svensk Byggtjänst - BK 2025** (Commercial)
**URL:** [byggtjanst.se/bokhandel/bk-2025](https://byggtjanst.se/bokhandel/bk-2025)

Byggmästarnas Kostnadskalkylator - structured according to BSAB 96 (same as AMA Hus 24).

**Also available:**
- Afry's lilla prisbok (small price book)
- Kalkyl GRÅ (calculation software)

**Access:** Purchase required (~2,000-3,000 SEK)

---

#### 5. **Energimyndigheten - Energikalkylen** (Free, Småhus)
**URL:** [energieffektivasmahus.se](https://energieffektivasmahus.se/)

Calculator for single-family homes (småhus) with cost comparisons for heating systems.

**Coverage:**
- Heat pumps (bergvärme, luftvärmepump, frånluftsvärmepump)
- District heating connection
- Pellets/biofuel
- Electric heating

**Note:** Being updated with new cost data and improved interface.

---

#### 6. **Sveriges Allmännytta (formerly SABO)** (Member Access)
**URL:** [sverigesallmannytta.se](https://www.sverigesallmannytta.se/)

Public housing association with extensive renovation cost data.

**Key reports:**
- "Hem för miljoner" - Million Programme renovation
- "Effektiv renovering" - Cost reduction strategies
- Annual renovation cost benchmarks

**Access:** Some reports public, detailed data requires membership

---

### 6.3 ECM-Specific Cost Data (2024-2025 SEK)

Based on research, here are current Swedish market prices:

#### Heat Pumps

| Type | Total Cost (incl. installation) | Cost per kW | ROT Deduction |
|------|--------------------------------|-------------|---------------|
| Bergvärmepump (GSHP) | 120,000 - 220,000 SEK | 7,000-15,000 SEK/kW | 50% labor |
| Frånluftsvärmepump (EAHP) | 80,000 - 150,000 SEK | 10,000-15,000 SEK/kW | 50% labor |
| Luftvärmepump (ASHP) | 50,000 - 120,000 SEK | 5,000-10,000 SEK/kW | 50% labor |

**Borehole drilling:** ~300 SEK/meter + casing 500-900 SEK/meter

---

#### Solar PV

| System Size | Total Cost (before deduction) | Per kWp | Green Deduction |
|-------------|------------------------------|---------|-----------------|
| 5 kWp (small) | 60,000 - 80,000 SEK | 12,000-16,000 | 15% (from July 2025) |
| 10 kWp (typical villa) | 100,000 - 150,000 SEK | 10,000-15,000 | 15% |
| 50 kWp (flerbostadshus) | 400,000 - 600,000 SEK | 8,000-12,000 | 15% |

**Battery storage:** 50,000-100,000 SEK for 10 kWh, 50% deduction

---

#### Ventilation (FTX)

| Building Type | Cost per m² | Cost per Apartment | Heat Recovery |
|---------------|-------------|-------------------|---------------|
| Flerbostadshus (central) | 300-500 SEK/m² | 50,000-100,000 SEK | 70-90% |
| Flerbostadshus (per-unit) | - | 30,000-60,000 SEK | 70-85% |
| Villa/småhus | - | 70,000-150,000 SEK total | 80-90% |

**F→FTX conversion (flerbostadshus):**
- With existing ducts: ~900,000 SEK
- New installation: ~2,300,000 SEK

---

#### Envelope Measures

| Measure | Cost (SEK/m²) | Typical Savings |
|---------|---------------|-----------------|
| External wall insulation (100mm) | 1,200-2,000 SEK/m² wall | 15-25% heating |
| Internal wall insulation (50mm) | 600-1,000 SEK/m² wall | 5-15% heating |
| Roof insulation (200mm) | 300-600 SEK/m² roof | 5-15% heating |
| Window replacement (triple) | 5,000-8,000 SEK/m² window | 10-20% heating |
| Air sealing | 30-80 SEK/m² floor | 5-10% heating |
| Entrance door replacement | 15,000-40,000 SEK/door | 1-3% heating |

---

#### District Heating Optimization

| Measure | Cost | Typical Savings |
|---------|------|-----------------|
| Substation optimization | 10,000-30,000 SEK | 5-15% cost |
| Return temp reduction | 5,000-15,000 SEK | Tariff improvement |
| Heating curve adjustment | 0-5,000 SEK | 3-8% energy |

---

### 6.4 Cost Index and Inflation

Swedish construction costs have increased significantly:

| Period | BKI Increase | Material Costs |
|--------|--------------|----------------|
| 2000-2023 | +132% total | +176% total |
| Average/year | +3.5%/year | +4.6%/year |
| 2023-2024 | +4-6% | +3-5% |

**Recommendation:** Apply 4% annual escalation to historical cost data.

---

### 6.5 Raiden Cost Database Implementation

#### Recommended Structure

```python
# src/roi/costs_sweden_v2.py

from dataclasses import dataclass
from typing import Dict, Optional
from datetime import date
from enum import Enum


class CostSource(Enum):
    """Source of cost data for traceability."""
    WIKELLS_2024 = "wikells_sektionsfakta_2024"
    BEBO_2023 = "bebo_lonsamhetskalkyl_2023"
    BEBO_TYPKOSTNADER_2023 = "bebo_typkostnader_2023"
    MARKET_RESEARCH_2025 = "market_research_2025"
    ENERGIMYNDIGHETEN = "energimyndigheten"
    USER_INPUT = "user_input"


@dataclass
class CostEntry:
    """A single cost data point with metadata."""
    value_sek: float
    unit: str  # "SEK/m²", "SEK/kW", "SEK/unit", etc.
    source: CostSource
    year: int
    confidence: float  # 0-1, how reliable is this data
    notes: Optional[str] = None

    def inflate_to(self, target_year: int, annual_rate: float = 0.04) -> float:
        """Inflate cost to target year."""
        years = target_year - self.year
        return self.value_sek * (1 + annual_rate) ** years


@dataclass
class ECMCostData:
    """Complete cost data for an ECM."""
    ecm_id: str
    material_cost: CostEntry
    labor_cost: CostEntry
    fixed_cost: Optional[CostEntry] = None
    annual_maintenance: Optional[CostEntry] = None
    lifetime_years: int = 25

    # Swedish-specific deductions
    rot_eligible: bool = False  # 50% labor deduction
    green_tech_eligible: bool = False  # 15-20% deduction
    energy_efficiency_grant_eligible: bool = False  # Boverket grant

    def total_cost(self, quantity: float, year: int = 2025) -> float:
        """Calculate total cost for given quantity."""
        material = self.material_cost.inflate_to(year) * quantity
        labor = self.labor_cost.inflate_to(year) * quantity
        fixed = self.fixed_cost.inflate_to(year) if self.fixed_cost else 0
        return material + labor + fixed

    def total_cost_after_deductions(self, quantity: float, year: int = 2025) -> float:
        """Calculate cost after applicable Swedish deductions."""
        material = self.material_cost.inflate_to(year) * quantity
        labor = self.labor_cost.inflate_to(year) * quantity
        fixed = self.fixed_cost.inflate_to(year) if self.fixed_cost else 0

        # Apply ROT (50% of labor, max 50,000 SEK/person/year)
        if self.rot_eligible:
            labor *= 0.5

        # Apply green tech deduction
        if self.green_tech_eligible:
            total = material + labor + fixed
            total *= 0.85  # 15% deduction

        return material + labor + fixed


# Example cost entries
SWEDISH_ECM_COSTS: Dict[str, ECMCostData] = {

    "ground_source_heat_pump": ECMCostData(
        ecm_id="ground_source_heat_pump",
        material_cost=CostEntry(
            value_sek=80000,  # Heat pump unit
            unit="SEK/unit",
            source=CostSource.MARKET_RESEARCH_2025,
            year=2025,
            confidence=0.8,
        ),
        labor_cost=CostEntry(
            value_sek=40000,  # Installation + borehole
            unit="SEK/unit",
            source=CostSource.MARKET_RESEARCH_2025,
            year=2025,
            confidence=0.7,
            notes="Excludes borehole drilling, add 300 SEK/m",
        ),
        fixed_cost=CostEntry(
            value_sek=50000,  # Typical borehole 150m @ 300 SEK/m
            unit="SEK/unit",
            source=CostSource.MARKET_RESEARCH_2025,
            year=2025,
            confidence=0.7,
            notes="Borehole drilling 150m typical",
        ),
        lifetime_years=20,
        rot_eligible=True,
        green_tech_eligible=True,
    ),

    "solar_pv": ECMCostData(
        ecm_id="solar_pv",
        material_cost=CostEntry(
            value_sek=8000,
            unit="SEK/kWp",
            source=CostSource.MARKET_RESEARCH_2025,
            year=2025,
            confidence=0.85,
            notes="Panels + inverter + mounting",
        ),
        labor_cost=CostEntry(
            value_sek=4000,
            unit="SEK/kWp",
            source=CostSource.MARKET_RESEARCH_2025,
            year=2025,
            confidence=0.8,
        ),
        lifetime_years=25,
        rot_eligible=True,
        green_tech_eligible=True,  # 15% from July 2025
    ),

    "ftx_installation": ECMCostData(
        ecm_id="ftx_installation",
        material_cost=CostEntry(
            value_sek=250,
            unit="SEK/m²",
            source=CostSource.BEBO_TYPKOSTNADER_2023,
            year=2023,
            confidence=0.75,
        ),
        labor_cost=CostEntry(
            value_sek=150,
            unit="SEK/m²",
            source=CostSource.BEBO_TYPKOSTNADER_2023,
            year=2023,
            confidence=0.75,
        ),
        annual_maintenance=CostEntry(
            value_sek=5,
            unit="SEK/m²/year",
            source=CostSource.BEBO_2023,
            year=2023,
            confidence=0.7,
            notes="Filter changes, inspections",
        ),
        lifetime_years=20,
        rot_eligible=True,
    ),

    "wall_external_insulation": ECMCostData(
        ecm_id="wall_external_insulation",
        material_cost=CostEntry(
            value_sek=800,
            unit="SEK/m² wall",
            source=CostSource.BEBO_TYPKOSTNADER_2023,
            year=2023,
            confidence=0.7,
            notes="100mm mineral wool + rendering",
        ),
        labor_cost=CostEntry(
            value_sek=700,
            unit="SEK/m² wall",
            source=CostSource.BEBO_TYPKOSTNADER_2023,
            year=2023,
            confidence=0.7,
        ),
        fixed_cost=CostEntry(
            value_sek=50000,
            unit="SEK/building",
            source=CostSource.BEBO_2023,
            year=2023,
            confidence=0.6,
            notes="Scaffolding, setup",
        ),
        lifetime_years=40,
        rot_eligible=True,
    ),

    # ... more ECMs
}
```

---

### 6.6 Data Collection Priority

| Source | Priority | Effort | Data Quality |
|--------|----------|--------|--------------|
| **BeBo Lönsamhetskalkyl** | HIGH | Low (free Excel) | Good |
| **BeBo Typkostnader 2023** | HIGH | Low (free PDF) | Very Good |
| **Market research (web)** | MEDIUM | Medium | Variable |
| **Wikells Sektionsfakta** | LOW | High (paid, manual) | Excellent |
| **SABO/Sveriges Allmännytta** | MEDIUM | Medium (membership) | Very Good |

---

### 6.7 Implementation Steps

1. **Download BeBo tools** from bebostad.se
   - Parse `BeBos lönsamhetskalkyl 1.5.xls`
   - Extract cost parameters from worked examples

2. **Extract Typkostnader data**
   - Parse 2023 cost guidance PDF
   - Create structured database entries

3. **Web scrape current market prices**
   - Heat pump installers (NIBE, Thermia, etc.)
   - Solar installers (GreenMatch, Otovo, etc.)
   - Ventilation companies

4. **Build cost update pipeline**
   - Annual update from BeBo releases
   - BKI index application for inflation
   - User feedback mechanism for local prices

5. **Add uncertainty quantification**
   - Confidence intervals from source variability
   - Regional price adjustments (Stockholm +15%, Norrland -10%)

---

## References

### MUBES & Calibration
- [MUBES GitHub](https://github.com/KTH-UrbanT/mubes-ubem)
- [GeomEppy Documentation](https://geomeppy.readthedocs.io/)
- [Copula-based calibration paper (2024)](https://www.sciencedirect.com/science/article/pii/S0378778824002500)
- [Hierarchical calibration paper (2018)](https://www.sciencedirect.com/science/article/abs/pii/S0378778818312532)
- [ABC-SMC tutorial](https://pyabc.readthedocs.io/)

### ECM Catalogs
- [CityBES/CBES - LBNL](https://cbes.lbl.gov/) - 80+ ECMs for commercial buildings
- [CityBES ECM List](https://cbes.lbl.gov/preliminary_retrofits/ecm_list) - Complete measure inventory
- [TABULA/EPISCOPE](https://episcope.eu/) - European residential building typology
- [ASHRAE Energy Efficiency Measures](https://xp20.ashrae.org/pcbea/files/eems-to-consider-2011-09-15.pdf) - Comprehensive EEM guidance

### Swedish Sources
- [Boverket Building Regulations (BBR)](https://www.boverket.se/)
- [Swedish District Heating Best Practice](https://www.shcbysweden.se/wp-content/uploads/2019/11/Best-Practice-Guide_District-Energy-by-Sweden-EN.pdf)
- [Swedish Long-Term Renovation Strategy](https://bpie.eu/wp-content/uploads/2018/01/iBROAD_CountryFactsheet_SWEDEN-2018.pdf)
- [Energiföretagen District Heating Substations](https://www.energiforetagen.se/)

### Swedish Cost Databases
- [BeBo Lönsamhetskalkyl](https://www.bebostad.se/verktyg/bebolonsamhetskalkyl) - Free Excel tool for multi-family buildings
- [BeBo Vägledning Typkostnader](https://www.bebostad.se/projekt/avslutade-projekt/2023/2023-11-vagledning-lonsamhet-och-kostnader-for-energieffektivisering-typkostnader-lonsamhetsexempel) - WSP cost data
- [Wikells Sektionsfakta](https://wikells.se/) - Industry standard (commercial)
- [Svensk Byggtjänst BK 2025](https://byggtjanst.se/bokhandel/bk-2025) - Construction cost calculator
- [Sveriges Allmännytta](https://www.sverigesallmannytta.se/) - Public housing association
- [Energieffektiva småhus](https://energieffektivasmahus.se/) - Energikalkylen for single-family
- [Byggföretagen BKI](https://byggforetagen.se/statistik/byggkostnader/) - Construction cost index

---

## Appendix: Quick Reference

### GeomEppy Cheat Sheet

```python
from geomeppy import IDF

# Create from footprint
idf = IDF("template.idf")
idf.add_block(coordinates=[(0,0), (10,0), (10,20), (0,20)], height=15, num_stories=5)
idf.set_wwr({180: 0.25, 0: 0.15})  # S=25%, N=15%
idf.intersect_match()
idf.saveas("output.idf")
```

### Bayesian Calibration Cheat Sheet

```python
from src.calibration import BayesianCalibrator

calibrator = BayesianCalibrator(surrogate_dir=Path("./surrogates"))
result = calibrator.calibrate(archetype_id="mfh_1961_1975", measured_kwh_m2=85.0)

# Point estimate
print(f"Infiltration: {result.adjusted_infiltration_ach:.3f}")

# Uncertainty
print(f"90% CI: {result.posterior.ci_90['infiltration_ach']}")

# ECM savings with uncertainty
savings = result.predict_ecm_savings("wall_external_insulation")
print(f"Savings: {savings['savings_fraction']['mean']:.1%}")
print(f"90% CI: {savings['savings_fraction']['ci_90']}")
```

---

## 🧠 ULTRATHINK: Strategic Prioritization Matrix

### Decision Framework: What to Build First

Given limited resources, here's the prioritization based on **impact × feasibility**:

```
                    HIGH IMPACT
                         │
    ┌────────────────────┼────────────────────┐
    │                    │                    │
    │  [2] BAYESIAN      │  [1] GEOMEPPY     │
    │  CALIBRATION       │  INTEGRATION       │
    │                    │                    │
    │  Impact: 9/10      │  Impact: 10/10    │
    │  Effort: 8/10      │  Effort: 5/10     │
    │  Risk: Medium      │  Risk: Low        │
    │                    │                    │
LOW ├────────────────────┼────────────────────┤ HIGH
FEAS│                    │                    │ FEAS
    │  [4] COST DB       │  [3] ECM EXPAND   │
    │  INTEGRATION       │  (Phase 1)        │
    │                    │                    │
    │  Impact: 6/10      │  Impact: 7/10     │
    │  Effort: 6/10      │  Effort: 4/10     │
    │  Risk: Low         │  Risk: Low        │
    │                    │                    │
    └────────────────────┼────────────────────┘
                         │
                    LOW IMPACT
```

### Recommended Execution Order

| Priority | Component | Why First | Dependencies |
|----------|-----------|-----------|--------------|
| **P0** | GeomEppy Integration | Foundation for everything; unblocks accurate ECM simulation | None |
| **P1** | ECM Expansion (Phase 1) | FVP + DH optimization = immediate Swedish market value | P0 (for accurate savings) |
| **P2** | Bayesian Calibration | Differentiated value prop, but requires surrogate training infrastructure | P0 |
| **P3** | Cost Database | Nice-to-have; current estimates are reasonable for MVP | None |

### Quick Wins (Can Do Now)

These can be implemented immediately without the full architecture upgrade:

1. **Add `exhaust_air_heat_pump` ECM** (~2 hours)
   - Very common Swedish retrofit
   - Simple IDF modifier (add heat recovery to exhaust)
   - 40-60% savings potential

2. **Add `district_heating_optimization` ECM** (~2 hours)
   - Zero/low cost operational measure
   - Just thermostat adjustments in IDF
   - 5-10% cost savings

3. **Download BeBo Lönsamhetskalkyl** (~1 hour)
   - Free Excel with real Swedish costs
   - Parse and update `costs_sweden.py`
   - Immediate ROI accuracy improvement

### Risk Mitigation

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| GeomEppy incompatible with eppy | Medium | High | Test early; fallback to manual geometry |
| Surrogate training fails (R² < 0.90) | Low | High | Add more samples; try neural network surrogate |
| E+ 25.1 bug breaks GeomEppy IDF | Medium | Medium | Apply same workaround (ConstantSupplyHumidityRatio) |
| BeBo cost data outdated | Low | Low | Apply BKI inflation index |

### Success Metrics

After full implementation, Raiden should achieve:

| Metric | Target | Measurement |
|--------|--------|-------------|
| Geometry preservation | >95% of footprint vertices retained | Compare input/output polygon |
| Surrogate accuracy | R² > 0.95, RMSE < 5 kWh/m² | Cross-validation |
| Calibration speed | <10s per building | Wall clock time |
| CI coverage | 85-95% | Synthetic tests with known parameters |
| ECM coverage | 50+ measures | Catalog count |
| Swedish-specific ECMs | 10+ unique to Swedish market | Manual review |

### What NOT to Do

1. **Don't build full PostgreSQL backend yet** - Current GeoJSON + file-based approach scales to 37k buildings fine
2. **Don't pursue real-time Boverket API** - PDF scraping is complex; GeoJSON already has the data
3. **Don't add non-Swedish ECMs** - Focus on Swedish market value (FVP, DH, BBR compliance)
4. **Don't over-engineer cost uncertainty** - Point estimates with ±20% range are sufficient for BRF decisions

---

*Analysis by Claude Opus 4.5 | 2025-12-19*
