"""
Pytest configuration and fixtures for Raiden tests.

Provides reusable test fixtures for:
- Building geometry
- Swedish archetypes
- Sample IDF content
- Mock simulation results
"""

import pytest
from pathlib import Path
import tempfile
import shutil

# Add src to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.geometry.building_geometry import BuildingGeometry, WallSegment
from src.baseline.archetypes import SwedishArchetype, ArchetypeMatcher, BuildingType


# =============================================================================
# PATH FIXTURES
# =============================================================================

@pytest.fixture
def project_root() -> Path:
    """Project root directory."""
    return Path(__file__).parent.parent


@pytest.fixture
def data_dir(project_root) -> Path:
    """Data directory."""
    return project_root / "data"


@pytest.fixture
def output_dir(project_root) -> Path:
    """Output directory with actual simulation results."""
    return project_root / "output_final"


@pytest.fixture
def temp_dir():
    """Temporary directory for test outputs, cleaned up after test."""
    tmp = tempfile.mkdtemp(prefix="raiden_test_")
    yield Path(tmp)
    shutil.rmtree(tmp, ignore_errors=True)


# =============================================================================
# GEOMETRY FIXTURES
# =============================================================================

@pytest.fixture
def simple_geometry() -> BuildingGeometry:
    """Simple rectangular 7-floor building geometry (Sjostaden-like)."""
    # Simple rectangular footprint: 20m x 16m = 320 m²
    footprint = [
        (0, 0), (20, 0), (20, 16), (0, 16), (0, 0)
    ]

    return BuildingGeometry(
        footprint_coords_local=footprint,
        footprint_area_m2=320.0,
        perimeter_m=72.0,  # 2*(20+16)
        height_m=21.0,
        floors=7,
        floor_height_m=3.0,
        gross_floor_area_m2=2240.0,  # 320 * 7
        volume_m3=6720.0,  # 320 * 21
        wall_segments=[
            WallSegment(orientation='S', azimuth=180, area_m2=420, length_m=20, height_m=21),
            WallSegment(orientation='E', azimuth=90, area_m2=336, length_m=16, height_m=21),
            WallSegment(orientation='N', azimuth=0, area_m2=420, length_m=20, height_m=21),
            WallSegment(orientation='W', azimuth=270, area_m2=336, length_m=16, height_m=21),
        ],
        total_wall_area_m2=1512.0,  # 72 * 21
        total_window_area_m2=378.0,  # ~25% WWR
        window_area_by_orientation={'N': 84, 'S': 126, 'E': 84, 'W': 84},
        roof_area_m2=320.0,
        centroid_local=(10, 8),
    )


@pytest.fixture
def small_geometry() -> BuildingGeometry:
    """Small 3-floor building for quick tests."""
    footprint = [(0, 0), (10, 0), (10, 10), (0, 10), (0, 0)]

    return BuildingGeometry(
        footprint_coords_local=footprint,
        footprint_area_m2=100.0,
        perimeter_m=40.0,
        height_m=9.0,
        floors=3,
        floor_height_m=3.0,
        gross_floor_area_m2=300.0,
        volume_m3=900.0,
        wall_segments=[
            WallSegment(orientation='S', azimuth=180, area_m2=90, length_m=10, height_m=9),
            WallSegment(orientation='E', azimuth=90, area_m2=90, length_m=10, height_m=9),
            WallSegment(orientation='N', azimuth=0, area_m2=90, length_m=10, height_m=9),
            WallSegment(orientation='W', azimuth=270, area_m2=90, length_m=10, height_m=9),
        ],
        total_wall_area_m2=360.0,
        total_window_area_m2=90.0,
        window_area_by_orientation={'N': 22.5, 'S': 22.5, 'E': 22.5, 'W': 22.5},
        roof_area_m2=100.0,
        centroid_local=(5, 5),
    )


# =============================================================================
# ARCHETYPE FIXTURES
# =============================================================================

@pytest.fixture
def modern_archetype() -> SwedishArchetype:
    """Modern Swedish multi-family archetype (2000+)."""
    matcher = ArchetypeMatcher()
    return matcher.match(
        construction_year=2003,
        building_type=BuildingType.MULTI_FAMILY,
        facade_material="render",
    )


@pytest.fixture
def older_archetype() -> SwedishArchetype:
    """Older Swedish multi-family archetype (1960s)."""
    matcher = ArchetypeMatcher()
    return matcher.match(
        construction_year=1965,
        building_type=BuildingType.MULTI_FAMILY,
        facade_material="brick",
    )


# =============================================================================
# IDF CONTENT FIXTURES
# =============================================================================

@pytest.fixture
def sample_idf_content() -> str:
    """Minimal IDF content for testing modifications."""
    return '''!-Generator IDFEditor 1.50
!-Option SortedOrder

Version,
    25.1;                    !- Version Identifier

Building,
    Test Building,           !- Name
    0,                       !- North Axis {deg}
    City,                    !- Terrain
    0.04,                    !- Loads Convergence Tolerance Value
    0.4,                     !- Temperature Convergence Tolerance Value {deltaC}
    FullInteriorAndExterior, !- Solar Distribution
    25,                      !- Maximum Number of Warmup Days
    6;                       !- Minimum Number of Warmup Days

Timestep,
    4;                       !- Number of Timesteps per Hour

Material,
    Wall_Insulation,         !- Name
    Rough,                   !- Roughness
    0.250,                   !- Thickness {m}
    0.040,                   !- Conductivity {W/m-K}
    30,                      !- Density {kg/m3}
    840;                     !- Specific Heat {J/kg-K}

Material,
    Roof_Insulation,         !- Name
    Rough,                   !- Roughness
    0.350,                   !- Thickness {m}
    0.040,                   !- Conductivity {W/m-K}
    30,                      !- Density {kg/m3}
    840;                     !- Specific Heat {J/kg-K}

WindowMaterial:SimpleGlazingSystem,
    Triple_Glazing,          !- Name
    1.0,                     !- U-Factor {W/m2-K}
    0.45;                    !- Solar Heat Gain Coefficient

ZoneInfiltration:DesignFlowRate,
    Zone1_Infiltration,      !- Name
    Zone1,                   !- Zone Name
    Always On,               !- Schedule Name
    AirChanges/Hour,         !- Design Flow Rate Calculation Method
    ,                        !- Design Flow Rate {m3/s}
    ,                        !- Flow per Zone Floor Area {m3/s-m2}
    ,                        !- Flow per Exterior Surface Area {m3/s-m2}
    0.06;                    !- Air Changes per Hour

ZoneHVAC:IdealLoadsAirSystem,
    Zone1_IdealLoads,        !- Name
    ,                        !- Availability Schedule Name
    Zone1_Supply,            !- Zone Supply Air Node Name
    ,                        !- Zone Exhaust Air Node Name
    ,                        !- System Inlet Air Node Name
    50,                      !- Maximum Heating Supply Air Temperature {C}
    13,                      !- Minimum Cooling Supply Air Temperature {C}
    0.0156,                  !- Maximum Heating Supply Air Humidity Ratio {kgWater/kgDryAir}
    0.0077,                  !- Minimum Cooling Supply Air Humidity Ratio {kgWater/kgDryAir}
    NoLimit,                 !- Heating Limit
    ,                        !- Maximum Heating Air Flow Rate {m3/s}
    ,                        !- Maximum Sensible Heating Capacity {W}
    NoLimit,                 !- Cooling Limit
    ,                        !- Maximum Cooling Air Flow Rate {m3/s}
    ,                        !- Maximum Total Cooling Capacity {W}
    Heating_Setpoint,        !- Heating Availability Schedule Name
    Cooling_Setpoint,        !- Cooling Availability Schedule Name
    ConstantSupplyHumidityRatio,  !- Dehumidification Control Type
    ,                        !- Cooling Sensible Heat Ratio
    ConstantSupplyHumidityRatio,  !- Humidification Control Type
    ,                        !- Design Specification Outdoor Air Object Name
    ,                        !- Outdoor Air Inlet Node Name
    ,                        !- Demand Controlled Ventilation Type
    ,                        !- Outdoor Air Economizer Type
    Sensible,                !- Heat Recovery Type
    0.75,                    !- Sensible Heat Recovery Effectiveness
    0.0;                     !- Latent Heat Recovery Effectiveness

Lights,
    Zone1_Lights,            !- Name
    Zone1,                   !- Zone Name
    Lighting_Schedule,       !- Schedule Name
    Watts/Area,              !- Design Level Calculation Method
    ,                        !- Lighting Level {W}
    8,                       !- Watts per Zone Floor Area {W/m2}
    ,                        !- Watts per Person {W/person}
    0,                       !- Return Air Fraction
    0.42,                    !- Fraction Radiant
    0.18;                    !- Fraction Visible
'''


# =============================================================================
# SIMULATION RESULTS FIXTURES
# =============================================================================

@pytest.fixture
def sample_csv_content() -> str:
    """Sample eplustbl.csv content for testing parser."""
    return '''Program Version:,EnergyPlus, Version 25.1.0-c4000000
,,
REPORT:,Annual Building Utility Performance Summary
,,
,Total Building Area,2240.00
,Net Conditioned Building Area,2240.00
,Unconditioned Building Area,0.00
,,
End Uses
,,Electricity [kWh],Natural Gas [kWh],Gasoline [kWh],Diesel [kWh],Coal [kWh],Fuel Oil No 1 [kWh],Fuel Oil No 2 [kWh],Propane [kWh],Other Fuel 1 [kWh],Other Fuel 2 [kWh],District Cooling [kWh],District Heating Water [kWh],District Heating Steam [kWh],Water [m3]
,Heating,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,93765.00,0.00,0.00
,Cooling,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
,Interior Lighting,41207.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
,Interior Equipment,59358.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
,Fans,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
'''


@pytest.fixture
def sample_csv_file(temp_dir, sample_csv_content) -> Path:
    """Write sample CSV to temp file and return path."""
    csv_path = temp_dir / "eplustbl.csv"
    csv_path.write_text(sample_csv_content)
    return temp_dir  # Return directory, not file (parser expects directory)


# =============================================================================
# ECM FIXTURES
# =============================================================================

@pytest.fixture
def ecm_params_wall_insulation() -> dict:
    """Parameters for wall insulation ECM."""
    return {
        'thickness_mm': 100,
        'material': 'mineral_wool',
    }


@pytest.fixture
def ecm_params_window_replacement() -> dict:
    """Parameters for window replacement ECM."""
    return {
        'u_value': 0.8,
        'shgc': 0.5,
    }


@pytest.fixture
def ecm_params_air_sealing() -> dict:
    """Parameters for air sealing ECM."""
    return {
        'reduction_factor': 0.5,
    }


@pytest.fixture
def ecm_params_ftx_upgrade() -> dict:
    """Parameters for FTX upgrade ECM."""
    return {
        'effectiveness': 0.85,
    }


@pytest.fixture
def ecm_params_led_lighting() -> dict:
    """Parameters for LED lighting ECM."""
    return {
        'power_density': 4,
    }
