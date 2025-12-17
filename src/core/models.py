"""
Pydantic models for BRF building data.

Covers both the input schema (from energy declarations) and enriched output schema
(with WWR, materials, shading, solar potential for EnergyPlus simulation).
"""

from __future__ import annotations

from datetime import date
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


# =============================================================================
# ENUMS
# =============================================================================


class EnergyClass(str, Enum):
    A = "A"
    B = "B"
    C = "C"
    D = "D"
    E = "E"
    F = "F"
    G = "G"


class BuildingPurpose(str, Enum):
    FLERBOSTADSHUS = "Flerbostadshus"
    KONTOR = "Kontor"
    HANDEL = "Handel"
    INDUSTRI = "Industri"
    SAMHALLSFUNKTION = "Samhällsfunktion"


class FacadeMaterial(str, Enum):
    BRICK = "brick"
    CONCRETE = "concrete"
    PLASTER = "plaster"
    GLASS = "glass"
    METAL = "metal"
    WOOD = "wood"
    STONE = "stone"
    UNKNOWN = "unknown"


class HeatingType(str, Enum):
    DISTRICT = "district_heating"
    GROUND_SOURCE_HEAT_PUMP = "ground_source_heat_pump"
    EXHAUST_AIR_HEAT_PUMP = "exhaust_air_heat_pump"
    ELECTRIC = "electric"
    GAS = "gas"
    OIL = "oil"


class VentilationType(str, Enum):
    F = "exhaust_only"
    FT = "supply_exhaust"
    FTX = "supply_exhaust_heat_recovery"
    NATURAL = "natural"


# =============================================================================
# INPUT SCHEMA (matches BRF JSON from energy declarations)
# =============================================================================


class Geometry(BaseModel):
    """3D geometry of building in SWEREF99 TM (EPSG:3006)."""

    type: Literal["MultiPolygon"] = "MultiPolygon"
    height_meters: float
    coordinates_3d: list[list[float]] = Field(
        description="List of [x, y, z] coordinates at roof level"
    )
    ground_footprint: list[list[float]] = Field(
        description="List of [x, y, z] coordinates at ground level"
    )


class Ownership(BaseModel):
    owner: str
    owner_type_simple: str
    owner_type_detailed: str | None = None


class BuildingInfo(BaseModel):
    purpose: str
    category: str
    building_type: str | None = None
    uuid: str
    construction_year: int
    tax_value_year: int | None = None
    last_renovation_year: int | None = None


class Dimensions(BaseModel):
    heated_area_sqm: float
    footprint_area_sqm: float
    building_height_m: float
    floors_above_ground: int
    basement_floors: int = 0
    staircases: int | None = None
    apartments: int | None = None
    unheated_garage_sqm: float | None = None


class SpaceUsage(BaseModel):
    """Space usage breakdown in percent."""

    residential: float = 0
    office: float = 0
    grocery: float = 0
    restaurant: float = 0
    industrial: float = 0
    other: float = 0
    total: float = 100


class HeatingEnergy(BaseModel):
    exhaust_air_heat_pump_kwh: float = 0
    ground_source_heat_pump_kwh: float = 0
    district_heating_kwh: float = 0
    oil_kwh: float = 0
    gas_kwh: float = 0
    wood_kwh: float = 0
    biofuel_kwh: float = 0


class SolarInfo(BaseModel):
    has_solar_cells: bool = False
    solar_cell_output_kwh: float = 0
    has_solar_thermal: bool = False


class MeasurementPeriod(BaseModel):
    start: str
    end: str


class ReferenceValues(BaseModel):
    reference_value_1: float | None = None
    reference_value_2_max: float | None = None


class EnergyData(BaseModel):
    """Energy performance data from energy declaration."""

    energy_class: EnergyClass
    energy_performance_kwh_per_sqm: float
    total_energy_consumption_kwh: float
    primary_energy_consumption_kwh: float | None = None
    property_electricity_kwh: float | None = None
    hot_water_electricity_kwh: float | None = None
    estimated_electricity_production_kwh: float | None = None
    heating: HeatingEnergy
    solar: SolarInfo
    measurement_period: MeasurementPeriod | None = None
    reference_values: ReferenceValues | None = None


class VentilationTypes(BaseModel):
    exhaust_only_F: bool = False
    exhaust_with_heat_recovery_Fmed: bool = False
    supply_exhaust_FT: bool = False
    supply_exhaust_heat_recovery_FTX: bool = False
    natural_ventilation: bool = False


class Ventilation(BaseModel):
    approved: bool
    requirement_met: bool
    types: VentilationTypes
    designed_airflow_ls_sqm: float | None = None  # From PDF: l/s,m²
    measured_airflow_ls_sqm: float | None = None


class RadonMeasurement(BaseModel):
    """Radon measurement from energy declaration PDF."""

    value_bq_m3: int | None = None  # Bq/m³
    measurement_date: date | None = None
    status: Literal["GOOD", "WARNING", "HIGH"] | None = None  # <200, 200-400, >400


class EnergyRecommendation(BaseModel):
    """Energy saving recommendation from declaration."""

    description: str
    annual_savings_kwh: int | None = None
    cost_savings_kr: float | None = None
    implementation_cost_kr: int | None = None
    payback_years: float | None = None


class Location(BaseModel):
    address: str
    additional_addresses: list[str] = Field(default_factory=list)  # From PDF: multiple addresses
    property_designation: str | None = None
    house_number: int | None = None
    is_main_address: bool = True
    municipality: str
    municipality_code: int | None = None
    county: str | None = None
    county_code: int | None = None
    postal_code: str | None = None
    postal_area: str | None = None


class Administrative(BaseModel):
    fnr: int | None = None
    form_id: int | None = None
    approved_date: str | None = None
    declaration_version: str | None = None
    energy_declaration_version: int | None = None


class BuildingProperties(BaseModel):
    """All properties of a building from energy declaration."""

    ownership: Ownership
    building_info: BuildingInfo
    dimensions: Dimensions
    space_usage_percent: SpaceUsage
    energy: EnergyData
    ventilation: Ventilation
    location: Location
    administrative: Administrative | None = None

    # PDF-extracted data
    radon: RadonMeasurement | None = None
    recommendations: list[EnergyRecommendation] = Field(default_factory=list)
    specific_energy_kwh_sqm: float | None = None  # Actual (not primary) from PDF
    reference_value_2: float | None = None  # Similar buildings benchmark


class BRFBuilding(BaseModel):
    """Single building within a BRF property."""

    building_id: int
    geometry: Geometry
    properties: BuildingProperties


class BRFSummary(BaseModel):
    """Summary of the entire BRF property."""

    total_buildings: int
    total_apartments: int | None = None
    total_heated_area_sqm: float
    construction_year: int
    energy_class: EnergyClass
    energy_performance_kwh_per_sqm: float
    heating_system: str | None = None
    has_solar_panels: bool = False
    location: str | None = None


class BRFProperty(BaseModel):
    """
    Root model for a BRF property.
    This matches the input JSON structure.
    """

    brf_name: str
    source_file: str | None = None
    coordinate_system: str = "EPSG:3006 (SWEREF99 TM)"
    buildings: list[BRFBuilding]
    summary: BRFSummary


# =============================================================================
# ENRICHED OUTPUT SCHEMA (for EnergyPlus)
# =============================================================================


class WindowToWallRatio(BaseModel):
    """Window-to-wall ratio by facade orientation."""

    north: float = Field(ge=0, le=1, description="WWR for north facade")
    south: float = Field(ge=0, le=1, description="WWR for south facade")
    east: float = Field(ge=0, le=1, description="WWR for east facade")
    west: float = Field(ge=0, le=1, description="WWR for west facade")
    average: float = Field(ge=0, le=1, description="Average WWR")
    source: str = Field(description="Detection method used")
    confidence: float = Field(ge=0, le=1, description="Detection confidence")


class UValues(BaseModel):
    """Thermal transmittance values in W/(m2K)."""

    walls: float = Field(description="Exterior wall U-value")
    roof: float = Field(description="Roof U-value")
    floor: float = Field(description="Ground floor U-value")
    windows: float = Field(description="Window U-value")
    doors: float = Field(description="Door U-value", default=1.5)


class EnvelopeData(BaseModel):
    """Building envelope characteristics."""

    window_to_wall_ratio: WindowToWallRatio | None = None
    facade_material: FacadeMaterial = FacadeMaterial.UNKNOWN
    facade_material_confidence: float = 0.0
    roof_material: str | None = None
    insulation_thickness_cm: float | None = None
    u_values: UValues | None = None
    airtightness_n50: float | None = Field(
        None, description="Air changes per hour at 50Pa pressure difference"
    )


class ShadingObstruction(BaseModel):
    """External shading obstruction."""

    type: Literal["building", "tree", "terrain"]
    direction: Literal["north", "south", "east", "west", "northeast", "northwest", "southeast", "southwest"]
    distance_m: float
    height_m: float | None = None
    width_m: float | None = None
    coverage_pct: float | None = None


class ShadingAnalysis(BaseModel):
    """Shading analysis results."""

    annual_shadow_hours: dict[str, float] = Field(
        default_factory=dict, description="Shadow hours by surface (roof, facades)"
    )
    solar_access_factor: float = Field(
        ge=0, le=1, description="Fraction of potential solar radiation received"
    )
    primary_shading_sources: list[str] = Field(default_factory=list)
    obstructions: list[ShadingObstruction] = Field(default_factory=list)


class SolarPotential(BaseModel):
    """Remaining solar PV potential analysis."""

    suitable_roof_area_sqm: float = 0
    remaining_capacity_kwp: float = 0
    annual_yield_potential_kwh: float = 0
    optimal_tilt_deg: float = 35
    optimal_azimuth_deg: float = 180  # South
    shading_loss_pct: float = 0
    existing_pv_area_sqm: float = 0
    source: str = "not_analyzed"


class Landscaping(BaseModel):
    """Green space and vegetation data."""

    green_area_sqm: float = 0
    tree_canopy_coverage_pct: float = 0
    ndvi_average: float | None = None
    permeable_surface_pct: float = 0
    tree_count: int | None = None


class EnergyPlusConstruction(BaseModel):
    """Construction type mapping for EnergyPlus."""

    exterior_wall: str = "default_exterior_wall"
    roof: str = "default_roof"
    floor: str = "default_floor"
    interior_wall: str = "default_interior_wall"
    window: str = "default_window"
    door: str = "default_door"


class EnergyPlusZone(BaseModel):
    """Zone definition for EnergyPlus."""

    name: str
    floor_area_sqm: float
    volume_m3: float
    floor_number: int
    zone_type: Literal["residential", "commercial", "common", "parking"]


class EnergyPlusReady(BaseModel):
    """
    All data prepared for EnergyPlus IDF generation.
    """

    # Geometry (transformed to WGS84 for visualization, kept in local for IDF)
    footprint_coords_wgs84: list[tuple[float, float]] = Field(default_factory=list)
    footprint_coords_local: list[tuple[float, float]] = Field(default_factory=list)
    height_m: float
    num_stories: int
    floor_to_floor_height_m: float = 3.0

    # Envelope
    envelope: EnvelopeData = Field(default_factory=EnvelopeData)
    constructions: EnergyPlusConstruction = Field(default_factory=EnergyPlusConstruction)

    # Zones
    zones: list[EnergyPlusZone] = Field(default_factory=list)

    # Shading
    shading: ShadingAnalysis = Field(default_factory=ShadingAnalysis)

    # Solar
    solar_potential: SolarPotential = Field(default_factory=SolarPotential)

    # Landscaping
    landscaping: Landscaping = Field(default_factory=Landscaping)

    # Internal loads (defaults based on Swedish building standards)
    occupant_density_m2_per_person: float = 30
    lighting_power_density_w_per_m2: float = 8
    equipment_power_density_w_per_m2: float = 12

    # Infiltration
    infiltration_ach: float = 0.5

    # Schedules reference
    occupancy_schedule: str = "residential_occupancy"
    lighting_schedule: str = "residential_lighting"
    equipment_schedule: str = "residential_equipment"


class ImageryCapture(BaseModel):
    """Reference to captured imagery for a facade."""

    facade_direction: str
    image_path: str
    source: Literal["google_streetview", "mapillary", "manual", "generated"]
    capture_date: date | None = None
    latitude: float | None = None
    longitude: float | None = None
    heading: float | None = None


class EnrichmentMetadata(BaseModel):
    """Metadata about the enrichment process."""

    enrichment_date: date
    toolkit_version: str
    data_sources: list[str] = Field(default_factory=list)
    imagery_captures: list[ImageryCapture] = Field(default_factory=list)
    ai_models_used: list[str] = Field(default_factory=list)
    manual_overrides: list[str] = Field(default_factory=list)
    validation_status: Literal["pending", "validated", "rejected"] = "pending"


class EnrichedBuilding(BaseModel):
    """
    Enriched building model combining original BRF data
    with extracted metadata ready for EnergyPlus.
    """

    # Original data
    building_id: int
    original_geometry: Geometry
    original_properties: BuildingProperties

    # Enriched data for EnergyPlus
    energyplus_ready: EnergyPlusReady

    # Process metadata
    enrichment_metadata: EnrichmentMetadata


class EnrichedBRFProperty(BaseModel):
    """
    Complete enriched BRF property with all buildings.
    Ready for visualization and EnergyPlus export.
    """

    brf_name: str
    source_file: str | None = None
    coordinate_system: str = "EPSG:3006 (SWEREF99 TM)"

    # Original summary
    original_summary: BRFSummary

    # Enriched buildings
    buildings: list[EnrichedBuilding]

    # 3D visualization data (generated)
    visualization: dict | None = Field(
        None, description="Three.js compatible geometry data"
    )

    # Aggregated metrics
    total_remaining_pv_potential_kwh: float = 0
    average_wwr: float | None = None
    predominant_facade_material: FacadeMaterial | None = None


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================


def estimate_u_values(
    construction_year: int,
    facade_material: FacadeMaterial,
    renovation_year: int | None = None,
) -> UValues:
    """
    Estimate U-values based on construction year and material.
    Uses Swedish building regulation standards (BBR).
    """
    reference_year = renovation_year if renovation_year else construction_year

    # Swedish BBR requirements by era
    if reference_year >= 2020:
        # BBR29+
        return UValues(walls=0.18, roof=0.13, floor=0.15, windows=1.2)
    elif reference_year >= 2010:
        # BBR16-28
        return UValues(walls=0.20, roof=0.13, floor=0.15, windows=1.3)
    elif reference_year >= 2000:
        # Around 2003 (Sjöstaden era)
        return UValues(walls=0.25, roof=0.15, floor=0.20, windows=1.6)
    elif reference_year >= 1990:
        return UValues(walls=0.30, roof=0.20, floor=0.25, windows=2.0)
    elif reference_year >= 1975:
        # Post oil crisis improvements
        return UValues(walls=0.40, roof=0.25, floor=0.30, windows=2.5)
    elif reference_year >= 1960:
        # Million program era
        return UValues(walls=0.50, roof=0.35, floor=0.40, windows=2.8)
    else:
        # Pre-1960
        return UValues(walls=0.80, roof=0.50, floor=0.50, windows=3.0)


def estimate_infiltration(construction_year: int, renovation_year: int | None = None) -> float:
    """Estimate air infiltration rate (ACH at 50Pa) based on construction era."""
    reference_year = renovation_year if renovation_year else construction_year

    if reference_year >= 2010:
        return 0.6  # Modern airtight construction
    elif reference_year >= 2000:
        return 1.0
    elif reference_year >= 1990:
        return 2.0
    elif reference_year >= 1975:
        return 4.0
    else:
        return 6.0
