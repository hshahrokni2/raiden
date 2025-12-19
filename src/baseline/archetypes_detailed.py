"""
Detailed Swedish Building Archetypes Database.

Comprehensive archetype data based on:
- TABULA/EPISCOPE Swedish building typology (episcope.eu)
- Boverket BETSI project (2010)
- Boverket Energiguiden
- Swedish building regulations history (SBN 1967, 1975, 1980, BBR)
- MDPI research on Swedish post-war construction
- Sveby standards
- Historical construction documentation

Sources:
- https://episcope.eu/building-typology/country/se.html
- https://www.boverket.se/sv/energiguiden/
- https://www.mdpi.com/2075-5309/9/4/99
- https://historia.vattenfall.se/stories/fran-vattenkraft-till-solceller/fjarrvarmens-historia

Each archetype includes:
- Construction period and typical year range
- Wall construction types and U-values
- Roof construction and U-values
- Floor/foundation types and U-values
- Window types and U-values
- Infiltration rates (ACH natural, n50)
- Ventilation system type and efficiency
- Heating system distribution
- Typical building forms
- Common issues and renovation needs
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum


# =============================================================================
# ENUMS
# =============================================================================

class BuildingEra(Enum):
    """Swedish building construction eras - expanded with historical periods."""
    # Historical eras (pre-1930)
    MEDIEVAL_PRE_1700 = "pre_1700"           # Gamla Stan, medieval
    STORMAKTSTID_1700_1800 = "1700_1800"     # Stormaktstiden, 1700s townhouses
    INDUSTRIALISM_1800_1880 = "1800_1880"    # Early industrial, pre-stenstaden
    STENSTADEN_1880_1900 = "1880_1900"       # Stone city era, brick apartments
    JUGEND_1900_1910 = "1900_1910"           # Art Nouveau/Jugend
    NATIONALROMANTIK_1910_1920 = "1910_1920" # National Romanticism
    TJUGOTAL_1920_1930 = "1920_1930"         # 20-talsklassicism

    # Modern eras (1930+)
    PRE_1930 = "pre_1930"           # Generic pre-functionalism (for compatibility)
    FUNKIS_1930_1945 = "1930_1945"  # Functionalism
    FOLKHEM_1946_1960 = "1946_1960" # Folkhemmet
    REKORD_1961_1975 = "1961_1975"  # Miljonprogrammet/Rekordåren
    ENERGI_1976_1985 = "1976_1985"  # Post oil crisis
    MODERN_1986_1995 = "1986_1995"  # Modern well-insulated
    LAGENERGI_1996_2010 = "1996_2010"  # Low-energy transition
    NARA_NOLL_2011_PLUS = "2011_plus"  # Near-zero energy


class WallConstructionType(Enum):
    """Swedish wall construction types - expanded with historical types."""
    # Historical Masonry (pre-1900)
    MEDIEVAL_STONE = "medieval_stone"                # Medeltida sten (600-1000mm)
    RUBBLE_STONE = "rubble_stone"                    # Natursten/gråsten
    SOLID_BRICK_2_STONE = "solid_brick_2_stone"      # 2-stens tegel (~480mm)
    SOLID_BRICK_2_5_STONE = "solid_brick_2_5_stone"  # 2.5-stens tegel (~600mm)

    # Standard Brick/Masonry
    SOLID_BRICK_1_STONE = "solid_brick_1_stone"      # 1-stens tegel (~240mm)
    SOLID_BRICK_1_5_STONE = "solid_brick_1_5_stone"  # 1.5-stens tegel (~360mm)
    CAVITY_BRICK = "cavity_brick"                     # Hålmur med luftspalt
    BRICK_LIGHT_CONCRETE = "brick_light_concrete"    # Tegel + lättbetong

    # Hybrid (Landshövdingehus style)
    BRICK_WOOD_HYBRID = "brick_wood_hybrid"          # Tegel + trä (landshövdingehus)

    # Concrete
    CONCRETE_SANDWICH = "concrete_sandwich"           # Betongsandwich element
    LIGHT_CONCRETE_BLOCK = "light_concrete_block"    # Lättbetongblock
    CAST_IN_PLACE = "cast_in_place"                  # Platsgjuten betong

    # Wood - Historical
    LOG_TIMBER = "log_timber"                        # Liggande timmer/knuttimmer
    STANDING_TIMBER = "standing_timber"              # Stående timmer/resvirke
    STANDING_PLANK = "standing_plank"                # Stående plank (1900s)

    # Wood - Modern
    STUD_FRAME_SAWDUST = "stud_frame_sawdust"        # Regelstomme + sågspån
    STUD_FRAME_MINERAL = "stud_frame_mineral"        # Regelstomme + mineralull
    CLT = "clt"                                       # Korslimmat trä

    # Composite
    PREFAB_ELEMENT = "prefab_element"                # Prefab element


class WindowType(Enum):
    """Swedish window types by era."""
    SINGLE_PANE = "single"              # Enkelglas (pre-1950)
    COUPLED_2_PANE = "coupled_2"        # Kopplade 2-glas (1950-1970)
    SEALED_2_PANE = "sealed_2"          # Förseglade 2-glas (1970-1985)
    TRIPLE_PANE = "triple_3"            # 3-glas (1980+)
    LOW_E_TRIPLE = "low_e_triple"       # Lågemissions 3-glas (2000+)
    PASSIVE_HOUSE = "passive"           # Passivhus standard (2010+)


class VentilationType(Enum):
    """Swedish ventilation system types."""
    NATURAL = "S"           # Självdrag (natural)
    EXHAUST = "F"           # Mekanisk frånluft
    BALANCED = "FT"         # Mekanisk till- och frånluft
    HEAT_RECOVERY = "FTX"   # Med värmeåtervinning


class HeatingSystemType(Enum):
    """Swedish heating system types."""
    COAL_COKE = "coal_coke"              # Kol/koks (pre-1960)
    OIL_BOILER = "oil_boiler"            # Oljepanna
    DISTRICT_OIL = "district_oil"        # Fjärrvärme (oljebaserad)
    DISTRICT_BIOMASS = "district_bio"    # Fjärrvärme (biobaserad)
    ELECTRIC_DIRECT = "electric_direct"  # Direktverkande el
    ELECTRIC_WATERBORNE = "electric_water"  # Vattenburet el
    HEAT_PUMP_AIR = "hp_air"             # Luftvärmepump
    HEAT_PUMP_GROUND = "hp_ground"       # Bergvärme
    HEAT_PUMP_EXHAUST = "hp_exhaust"     # Frånluftsvärmepump


# =============================================================================
# ARCHETYPE DESCRIPTOR ENUMS
# =============================================================================
# These enums enable deterministic or AI-based archetype matching from
# visual inspection, address data, or building metadata

class BalconyType(Enum):
    """Balcony types for visual identification."""
    NONE = "none"                        # Inga balkonger
    RECESSED = "recessed"                # Indragna (loggia)
    LOGGIA = "loggia"                    # Loggia (similar to recessed)
    PROJECTING = "projecting"            # Utskjutande
    GLAZED = "glazed"                    # Inglasade
    GALLERY = "gallery"                  # Loftgång/loftgang
    FRENCH = "french"                    # Franska balkonger (doors only)
    CORNER = "corner"                    # Hörnbalkonger


class RoofProfile(Enum):
    """Roof profile for visual identification."""
    FLAT = "flat"                        # Platt tak
    LOW_SLOPE = "low_slope"              # Låglutande (<15°)
    LOW_PITCHED = "low_pitched"          # Alias for low_slope
    PITCHED = "pitched"                  # Sadeltak (>15°)
    MANSARD = "mansard"                  # Brutet tak/Mansardtak
    HIP = "hip"                          # Valmat tak
    PYRAMID = "pyramid"                  # Pyramidtak (punkthus)
    GREEN = "green"                      # Grönt tak/sedumtak


class FacadePattern(Enum):
    """Facade pattern for visual identification."""
    REGULAR_PUNCHED = "regular_punched"  # Regelbundna hål i fasad
    IRREGULAR_PUNCHED = "irregular"      # Oregelbundna fönster
    RIBBON_WINDOWS = "ribbon"            # Bandformade fönster (funkis)
    CURTAIN_WALL = "curtain_wall"        # Glasfasad
    MIXED = "mixed"                      # Blandad
    HORIZONTAL_BANDS = "horizontal"      # Horisontella band
    VERTICAL_EMPHASIS = "vertical"       # Vertikal betoning
    GRID_UNIFORM = "grid_uniform"        # Jämt rutnät (miljonprogram)
    LARGE_GLAZING = "large_glazing"      # Stora glaspartier


class PlanShape(Enum):
    """Building plan shape for identification."""
    RECTANGULAR = "rectangular"          # Rektangulär
    L_SHAPE = "l_shape"                  # L-form
    U_SHAPE = "u_shape"                  # U-form (slutet kvarter)
    T_SHAPE = "t_shape"                  # T-form
    H_SHAPE = "h_shape"                  # H-form
    STAR = "star"                        # Stjärnform
    POINT = "point"                      # Punkthus (kompakt)
    SLAB = "slab"                        # Skiva (lång, smal)
    COURTYARD = "courtyard"              # Med innergård
    IRREGULAR = "irregular"              # Oregelbunden


class UrbanSetting(Enum):
    """Urban context for archetype selection."""
    INNER_CITY = "inner_city"            # Innerstad
    INNER_SUBURB = "inner_suburb"        # Närförort (1920s-1950s)
    OUTER_SUBURB = "outer_suburb"        # Ytterstaden (miljonprogrammet)
    SATELLITE_TOWN = "satellite"         # Förort/ABC-stad
    SMALL_TOWN = "small_town"            # Småstad
    RURAL = "rural"                      # Landsbygd
    WATERFRONT = "waterfront"            # Sjöstad/vattennära


class OwnershipType(Enum):
    """Original ownership type for historical identification."""
    MUNICIPAL = "municipal"              # Kommunalt (allmännytta)
    COOPERATIVE = "cooperative"          # Bostadsrättsförening
    BRF = "brf"                          # Bostadsrättsförening (specific)
    PRIVATE_RENTAL = "private_rental"    # Privat hyresrätt
    PRIVATE_OWNER = "private_owner"      # Privatägd (småhus)
    HSB = "hsb"                          # HSB cooperative
    RIKSBYGGEN = "riksbyggen"            # Riksbyggen cooperative
    HOUSING_COMPANY = "housing_company"  # Bostadsbolag (other)
    INSTITUTIONAL = "institutional"      # Institution (skola → bostad)
    CHURCH = "church"                    # Kyrklig


class EnergyCertification(Enum):
    """Swedish energy certification levels."""
    NONE = "none"
    ENERGY_CLASS_G = "G"
    ENERGY_CLASS_F = "F"
    ENERGY_CLASS_E = "E"
    ENERGY_CLASS_D = "D"
    ENERGY_CLASS_C = "C"
    ENERGY_CLASS_B = "B"
    ENERGY_CLASS_A = "A"
    MILJOBYGGNAD_BRONZE = "mb_bronze"
    MILJOBYGGNAD_SILVER = "mb_silver"
    MILJOBYGGNAD_GOLD = "mb_gold"
    FEBY_SILVER = "feby_silver"
    FEBY_GOLD = "feby_gold"
    PASSIVE_HOUSE = "passive"
    PLUS_ENERGY = "plus_energy"
    GREEN_BUILDING = "green_building"    # Green Building certification
    SVANEN = "svanen"                    # Nordic Ecolabel (Svanen)


# =============================================================================
# ARCHETYPE DESCRIPTORS DATACLASS
# =============================================================================

@dataclass
class ArchetypeDescriptors:
    """
    Comprehensive descriptors for archetype matching.

    Used for:
    1. Deterministic matching from public data sources
    2. AI/ML-based matching from street view images
    3. Calibration parameter selection

    Each descriptor has a confidence weight for probabilistic matching.
    """

    # === GEOMETRIC DESCRIPTORS ===
    # Building dimensions (typical ranges)
    building_depth_m: Tuple[float, float] = (10.0, 14.0)  # (min, max) depth in meters
    floor_to_floor_m: Tuple[float, float] = (2.7, 3.0)    # (min, max) height per floor
    building_length_m: Tuple[float, float] = (20.0, 60.0) # (min, max) typical length

    # Plan characteristics
    plan_shape: List[PlanShape] = field(default_factory=list)  # Possible plan shapes
    stairwell_apartments: Tuple[int, int] = (2, 4)  # Apartments per stairwell (min, max)

    # === VISUAL DESCRIPTORS (for image recognition) ===
    balcony_types: List[BalconyType] = field(default_factory=list)  # Common balcony types
    roof_profiles: List[RoofProfile] = field(default_factory=list)  # Possible roof types
    facade_patterns: List[FacadePattern] = field(default_factory=list)  # Window/facade patterns

    # Facade details
    typical_colors: List[str] = field(default_factory=list)  # Common facade colors
    window_proportions: str = "portrait"        # "portrait", "square", "landscape"
    has_bay_windows: bool = False               # Burspråk
    has_corner_windows: bool = False            # Hörnfönster
    has_roof_terrace: bool = False              # Takterrass

    # === CONTEXTUAL DESCRIPTORS ===
    urban_settings: List[UrbanSetting] = field(default_factory=list)  # Typical locations
    typical_neighborhoods: List[str] = field(default_factory=list)  # Example areas (Södermalm, Gröndal...)
    typical_cities: List[str] = field(default_factory=list)  # Where most common

    # === OWNERSHIP & PROGRAM MARKERS ===
    original_ownership: List[OwnershipType] = field(default_factory=list)  # Original ownership type
    housing_programs: List[str] = field(default_factory=list)  # Barnrikehus, Miljonprogrammet, etc.
    notable_developers: List[str] = field(default_factory=list)  # HSB, Riksbyggen, Svenska Bostäder...
    notable_architects: List[str] = field(default_factory=list)  # Backström & Reinius, Markelius...

    # === CERTIFICATION & PERFORMANCE ===
    typical_certifications: List[EnergyCertification] = field(default_factory=list)
    has_solar_pv: bool = False
    has_battery_storage: bool = False
    has_ev_charging: bool = False

    # === MATCHING KEYWORDS ===
    # Keywords that might appear in property listings, building records, etc.
    keywords_sv: List[str] = field(default_factory=list)  # Swedish keywords
    keywords_en: List[str] = field(default_factory=list)  # English keywords

    # === CALIBRATION HINTS ===
    # Typical calibration adjustments needed from archetype defaults
    infiltration_variability: str = "medium"    # low, medium, high
    u_value_variability: str = "medium"         # How much U-values vary
    occupancy_pattern: str = "residential"      # residential, elderly, student, mixed

    # === RENOVATION STATE INDICATORS ===
    likely_renovated_if: List[str] = field(default_factory=list)  # Conditions suggesting renovation
    renovation_era_signs: Dict[str, str] = field(default_factory=dict)  # Visual signs by era


# =============================================================================
# DETAILED CONSTRUCTION DATA
# =============================================================================

@dataclass
class WallConstruction:
    """Detailed wall construction specification."""
    type: WallConstructionType
    name_sv: str
    name_en: str
    total_thickness_mm: int
    insulation_thickness_mm: int
    insulation_type: str  # mineralull, cellplast, lättbetong, etc.
    u_value: float  # W/m²K
    thermal_bridge_factor: float = 1.0  # Multiplier for thermal bridges
    description: str = ""


@dataclass
class WindowConstruction:
    """Detailed window specification."""
    type: WindowType
    name_sv: str
    name_en: str
    u_value_glass: float  # W/m²K (glass only)
    u_value_installed: float  # W/m²K (including frame)
    shgc: float  # Solar heat gain coefficient
    num_panes: int
    gas_fill: str = "air"  # air, argon, krypton
    coating: str = ""  # low-e coating type


@dataclass
class RoofConstruction:
    """Detailed roof construction specification."""
    name_sv: str
    insulation_thickness_mm: int
    insulation_type: str
    u_value: float  # W/m²K
    roof_type: str  # flat, pitched, cold_attic


@dataclass
class FloorConstruction:
    """Detailed floor/foundation specification."""
    name_sv: str
    type: str  # slab_on_grade, basement, crawlspace
    insulation_thickness_mm: int
    u_value: float  # W/m²K


# =============================================================================
# COMPREHENSIVE ARCHETYPE DATABASE
# =============================================================================

@dataclass
class DetailedArchetype:
    """
    Comprehensive Swedish building archetype with full technical details.

    Based on TABULA/EPISCOPE methodology and Swedish research.
    """
    # Identification
    id: str
    name_sv: str
    name_en: str
    era: BuildingEra
    year_start: int
    year_end: int

    # Building stock statistics
    stock_share_percent: float  # Share of total Swedish housing stock
    typical_atemp_m2: Tuple[int, int]  # (min, max) typical size range
    typical_floors: Tuple[int, int]  # (min, max) typical floor range

    # Wall constructions (can have multiple common types)
    wall_constructions: List[WallConstruction]

    # Roof construction
    roof_construction: RoofConstruction

    # Floor construction
    floor_construction: FloorConstruction

    # Window specifications
    window_construction: WindowConstruction
    typical_wwr: float  # Window-to-wall ratio

    # Air tightness
    infiltration_ach: float  # Natural ACH at normal conditions
    n50_ach: float  # ACH at 50 Pa pressure

    # Ventilation
    ventilation_type: VentilationType
    ventilation_rate_l_s_m2: float
    heat_recovery_efficiency: float  # 0 for non-FTX
    sfp_kw_per_m3s: float  # Specific fan power

    # Heating systems (distribution %)
    heating_systems: Dict[HeatingSystemType, float]
    typical_heating_kwh_m2: float  # Before renovation

    # Hot water
    dhw_kwh_m2: float

    # Internal loads (Sveby defaults)
    occupancy_w_per_m2: float
    lighting_w_m2: float
    equipment_w_m2: float

    # Building forms typical for this era
    typical_forms: List[str]  # lamellhus, skivhus, etc.

    # Facade materials
    typical_facades: List[str]

    # Common issues and renovation potential
    common_issues: List[str]
    renovation_potential_kwh_m2: float  # Typical savings possible
    typical_ecms: List[str]

    # Notes (with defaults)
    wwr_by_orientation: Dict[str, float] = field(default_factory=dict)
    description: str = ""
    sources: List[str] = field(default_factory=list)

    # Comprehensive descriptors for matching (optional - for enhanced archetypes)
    descriptors: Optional[ArchetypeDescriptors] = None


# =============================================================================
# SWEDISH PLUS-ENERGY & PASSIVE HOUSE ARCHETYPES (HIGH PERFORMANCE)
# =============================================================================
# Modern high-performance buildings meeting strict Swedish standards:
# - FEBY Passive House (15-17 W/m²/year heating)
# - Miljöbyggnad Gold (65% of BBR energy requirements)
# - Plus Energy (produces more energy than consumed)
# Sources: FEBY, Riksbyggen, Portvakten Söder research, Brf Viva

SWEDISH_HIGH_PERFORMANCE_ARCHETYPES: Dict[str, DetailedArchetype] = {

    # =========================================================================
    # FEBY PASSIVE HOUSE MULTI-FAMILY (2009+)
    # Swedish adaptation of passive house standard
    # Examples: Portvakten Söder (Växjö), various HSB/Riksbyggen projects
    # =========================================================================
    "passive_house_mfh_2009_plus": DetailedArchetype(
        id="passive_house_mfh_2009_plus",
        name_sv="Passivhus flerbostadshus",
        name_en="Passive House Multi-family (FEBY)",
        era=BuildingEra.NARA_NOLL_2011_PLUS,
        year_start=2009,
        year_end=2030,

        stock_share_percent=0.3,  # Growing but still small share
        typical_atemp_m2=(1500, 5000),
        typical_floors=(4, 8),

        wall_constructions=[
            WallConstruction(
                type=WallConstructionType.CLT,
                name_sv="KL-trä passivhusvägg",
                name_en="CLT passive house wall",
                total_thickness_mm=530,
                insulation_thickness_mm=300,
                insulation_type="mineralull/träfiberisolering",
                u_value=0.11,
                thermal_bridge_factor=1.03,
                description="Portvakten Söder type: 53cm thick wood frame walls"
            ),
            WallConstruction(
                type=WallConstructionType.CONCRETE_SANDWICH,
                name_sv="Betongsandwich passivhus",
                name_en="Concrete sandwich passive house",
                total_thickness_mm=500,
                insulation_thickness_mm=300,
                insulation_type="mineralull/cellplast",
                u_value=0.10,
                thermal_bridge_factor=1.05,
                description="Concrete passive house with thermal bridge minimization"
            ),
        ],
        roof_construction=RoofConstruction(
            name_sv="Passivhustak",
            insulation_thickness_mm=500,
            insulation_type="mineralull",
            u_value=0.075,
            roof_type="flat"
        ),
        floor_construction=FloorConstruction(
            name_sv="Välisolerad platta",
            type="slab_on_grade",
            insulation_thickness_mm=300,
            u_value=0.12
        ),
        window_construction=WindowConstruction(
            type=WindowType.PASSIVE_HOUSE,
            name_sv="Passivhusfönster",
            name_en="Passive house certified windows",
            u_value_glass=0.5,
            u_value_installed=0.85,
            shgc=0.50,
            num_panes=3,
            gas_fill="argon/krypton",
            coating="triple low-e"
        ),
        typical_wwr=0.20,

        infiltration_ach=0.03,
        n50_ach=0.3,  # FEBY requirement: ≤0.3 ACH @ 50Pa

        ventilation_type=VentilationType.HEAT_RECOVERY,
        ventilation_rate_l_s_m2=0.35,
        heat_recovery_efficiency=0.85,  # FEBY: ≥80%
        sfp_kw_per_m3s=1.5,  # FEBY: ≤1.5 kW/(m³/s)

        heating_systems={
            HeatingSystemType.DISTRICT_BIOMASS: 0.50,
            HeatingSystemType.HEAT_PUMP_GROUND: 0.35,
            HeatingSystemType.HEAT_PUMP_AIR: 0.15,
        },
        typical_heating_kwh_m2=15,  # FEBY: 15-17 W/m²/year

        dhw_kwh_m2=20,  # FEBY: ≤20 kWh/m²Atemp/year

        occupancy_w_per_m2=1.8,
        lighting_w_m2=4,
        equipment_w_m2=3,

        typical_forms=["lamellhus", "punkthus"],
        typical_facades=["tra", "puts", "fibercement"],

        common_issues=[
            "No traditional heating system required",
            "Completely airtight construction",
            "Super-insulated windows (U<1.0)",
            "Heat from occupants, appliances, lighting retained",
            "Mechanical ventilation with heat recovery essential",
            "Portvakten Söder: 40.2 kWh/m²Atemp/year actual",
        ],
        renovation_potential_kwh_m2=5,  # Already optimal
        typical_ecms=["solar_pv"],

        description="FEBY Passive House: Swedish adaptation requiring ≤15-17 W/m²/year heating, "
                    "n50 ≤0.3, and heat recovery ≥80%. Portvakten Söder (Växjö) is iconic example.",

        descriptors=ArchetypeDescriptors(
            # Geometric
            building_depth_m=(12.0, 18.0),
            floor_to_floor_m=(2.7, 3.0),
            building_length_m=(25.0, 80.0),
            plan_shape=[PlanShape.RECTANGULAR, PlanShape.SLAB],
            stairwell_apartments=(4, 8),

            # Visual
            balcony_types=[BalconyType.RECESSED, BalconyType.GLAZED],
            roof_profiles=[RoofProfile.FLAT, RoofProfile.LOW_SLOPE],
            facade_patterns=[FacadePattern.REGULAR_PUNCHED, FacadePattern.HORIZONTAL_BANDS],
            typical_colors=["white", "light_gray", "wood_natural", "dark_gray"],
            window_proportions="landscape",
            has_roof_terrace=True,

            # Context
            urban_settings=[UrbanSetting.INNER_SUBURB, UrbanSetting.SATELLITE_TOWN],
            typical_neighborhoods=["Portvakten", "Vallastaden", "Norra Djurgårdsstaden"],
            typical_cities=["Växjö", "Linköping", "Stockholm", "Malmö"],

            # Ownership
            original_ownership=[OwnershipType.MUNICIPAL, OwnershipType.HOUSING_COMPANY],
            housing_programs=["EU Concerto", "Passivhusprogrammet"],
            notable_developers=["Hyresbostäder i Växjö", "HSB", "Riksbyggen"],
            notable_architects=["Arkitektbolaget", "White Arkitekter"],

            # Certification
            typical_certifications=[
                EnergyCertification.PASSIVE_HOUSE,
                EnergyCertification.FEBY_GOLD,
                EnergyCertification.ENERGY_CLASS_A,
            ],
            has_solar_pv=True,

            # Keywords
            keywords_sv=["passivhus", "FEBY", "lågenergihus", "nollenergihus", "superissolering"],
            keywords_en=["passive house", "FEBY", "low energy", "zero energy", "super insulation"],

            # Calibration
            infiltration_variability="low",  # Very consistent construction
            u_value_variability="low",
            occupancy_pattern="residential",

            # Renovation indicators
            likely_renovated_if=["N/A - new construction"],
            renovation_era_signs={},
        ),
    ),

    # =========================================================================
    # PLUS-ENERGY MULTI-FAMILY (2015+)
    # Buildings that produce more energy than they consume annually
    # Example: Riksbyggen Brf Viva (Göteborg)
    # =========================================================================
    "plus_energy_mfh_2015_plus": DetailedArchetype(
        id="plus_energy_mfh_2015_plus",
        name_sv="Plusenergihus flerbostadshus",
        name_en="Plus-Energy Multi-family",
        era=BuildingEra.NARA_NOLL_2011_PLUS,
        year_start=2015,
        year_end=2030,

        stock_share_percent=0.1,  # Very rare, pioneering
        typical_atemp_m2=(2000, 6000),
        typical_floors=(4, 8),

        wall_constructions=[
            WallConstruction(
                type=WallConstructionType.CLT,
                name_sv="KL-trä plusenergivägg",
                name_en="CLT plus-energy wall",
                total_thickness_mm=450,
                insulation_thickness_mm=250,
                insulation_type="träfiberisolering/mineralull",
                u_value=0.10,
                thermal_bridge_factor=1.02,
                description="Plus-energy: optimized for minimal thermal bridges"
            ),
        ],
        roof_construction=RoofConstruction(
            name_sv="Plusenergitak med solceller",
            insulation_thickness_mm=450,
            insulation_type="mineralull",
            u_value=0.08,
            roof_type="low_slope"  # For solar optimization
        ),
        floor_construction=FloorConstruction(
            name_sv="Välisolerad platta",
            type="slab_on_grade",
            insulation_thickness_mm=300,
            u_value=0.12
        ),
        window_construction=WindowConstruction(
            type=WindowType.PASSIVE_HOUSE,
            name_sv="Passivhusfönster med solskydd",
            name_en="Passive house windows with solar control",
            u_value_glass=0.5,
            u_value_installed=0.8,
            shgc=0.35,  # Lower for cooling load management
            num_panes=3,
            gas_fill="krypton",
            coating="triple low-e + solar control"
        ),
        typical_wwr=0.22,

        infiltration_ach=0.03,
        n50_ach=0.3,

        ventilation_type=VentilationType.HEAT_RECOVERY,
        ventilation_rate_l_s_m2=0.35,
        heat_recovery_efficiency=0.88,
        sfp_kw_per_m3s=1.2,

        heating_systems={
            HeatingSystemType.HEAT_PUMP_GROUND: 0.60,
            HeatingSystemType.DISTRICT_BIOMASS: 0.30,
            HeatingSystemType.HEAT_PUMP_AIR: 0.10,
        },
        typical_heating_kwh_m2=12,  # Very low

        dhw_kwh_m2=18,

        occupancy_w_per_m2=1.8,
        lighting_w_m2=3,  # All LED
        equipment_w_m2=3,

        typical_forms=["lamellhus", "kvarter"],
        typical_facades=["tra", "puts", "solceller"],

        common_issues=[
            "Produces more energy than consumed annually",
            "Large rooftop solar PV installation (100+ kW)",
            "Battery storage from recycled bus batteries (Brf Viva)",
            "Smart energy management system",
            "Miljöbyggnad Gold certified (65% of BBR)",
            "Climate-improved concrete (30% lower CO2)",
            "Vacuum waste collection (Envac)",
            "34% heating from wastewater (Hammarby model)",
        ],
        renovation_potential_kwh_m2=0,  # Already optimal
        typical_ecms=[],  # No improvements needed

        description="Plus-energy buildings produce more energy than consumed. "
                    "Brf Viva (Göteborg, 2018): 132 apartments, solar+battery storage from bus batteries, "
                    "Miljöbyggnad Gold, climate concrete, vacuum waste.",

        descriptors=ArchetypeDescriptors(
            # Geometric
            building_depth_m=(14.0, 20.0),
            floor_to_floor_m=(2.8, 3.2),
            building_length_m=(30.0, 100.0),
            plan_shape=[PlanShape.RECTANGULAR, PlanShape.COURTYARD],
            stairwell_apartments=(4, 10),

            # Visual
            balcony_types=[BalconyType.RECESSED, BalconyType.GLAZED],
            roof_profiles=[RoofProfile.LOW_SLOPE],  # For solar panels
            facade_patterns=[FacadePattern.REGULAR_PUNCHED, FacadePattern.MIXED],
            typical_colors=["wood_natural", "dark_gray", "white", "green_accents"],
            window_proportions="landscape",
            has_roof_terrace=True,

            # Context
            urban_settings=[UrbanSetting.INNER_SUBURB, UrbanSetting.SATELLITE_TOWN],
            typical_neighborhoods=["Guldheden", "Norra Djurgårdsstaden", "Vallastaden"],
            typical_cities=["Göteborg", "Stockholm", "Malmö", "Uppsala"],

            # Ownership
            original_ownership=[OwnershipType.COOPERATIVE, OwnershipType.HOUSING_COMPANY],
            housing_programs=["Positive Footprint Housing", "IRIS Smart Cities"],
            notable_developers=["Riksbyggen", "HSB", "Skanska"],
            notable_architects=["Malmström Edström", "White Arkitekter"],

            # Certification
            typical_certifications=[
                EnergyCertification.PLUS_ENERGY,
                EnergyCertification.MILJOBYGGNAD_GOLD,
                EnergyCertification.ENERGY_CLASS_A,
            ],
            has_solar_pv=True,
            has_battery_storage=True,
            has_ev_charging=True,

            # Keywords
            keywords_sv=["plusenergihus", "plusenergi", "solceller", "batterilager",
                        "miljöbyggnad guld", "hållbart", "klimatsmart"],
            keywords_en=["plus energy", "net positive", "solar panels", "battery storage",
                        "green building gold", "sustainable", "climate smart"],

            # Calibration
            infiltration_variability="low",
            u_value_variability="low",
            occupancy_pattern="residential",

            # Renovation indicators
            likely_renovated_if=["N/A - new construction"],
            renovation_era_signs={},
        ),
    ),

    # =========================================================================
    # MILJÖBYGGNAD GOLD MULTI-FAMILY (2015+)
    # Not quite passive house but high performance (65% of BBR)
    # =========================================================================
    "miljobyggnad_gold_mfh": DetailedArchetype(
        id="miljobyggnad_gold_mfh",
        name_sv="Miljöbyggnad Guld flerbostadshus",
        name_en="Miljöbyggnad Gold Multi-family",
        era=BuildingEra.NARA_NOLL_2011_PLUS,
        year_start=2012,
        year_end=2030,

        stock_share_percent=0.5,  # Growing category
        typical_atemp_m2=(1500, 6000),
        typical_floors=(4, 10),

        wall_constructions=[
            WallConstruction(
                type=WallConstructionType.CONCRETE_SANDWICH,
                name_sv="Betongsandwich MB Guld",
                name_en="Concrete sandwich MB Gold",
                total_thickness_mm=450,
                insulation_thickness_mm=250,
                insulation_type="mineralull/PIR",
                u_value=0.13,
                thermal_bridge_factor=1.08,
                description="Miljöbyggnad Gold: 65% of BBR energy requirements"
            ),
        ],
        roof_construction=RoofConstruction(
            name_sv="Välisolerat tak",
            insulation_thickness_mm=400,
            insulation_type="mineralull",
            u_value=0.09,
            roof_type="flat"
        ),
        floor_construction=FloorConstruction(
            name_sv="Välisolerad platta",
            type="slab_on_grade",
            insulation_thickness_mm=250,
            u_value=0.15
        ),
        window_construction=WindowConstruction(
            type=WindowType.LOW_E_TRIPLE,
            name_sv="Energieffektiva treglasfönster",
            name_en="Energy efficient triple glazing",
            u_value_glass=0.7,
            u_value_installed=0.9,
            shgc=0.45,
            num_panes=3,
            gas_fill="argon",
            coating="low-e"
        ),
        typical_wwr=0.22,

        infiltration_ach=0.05,
        n50_ach=0.6,

        ventilation_type=VentilationType.HEAT_RECOVERY,
        ventilation_rate_l_s_m2=0.35,
        heat_recovery_efficiency=0.82,
        sfp_kw_per_m3s=1.5,

        heating_systems={
            HeatingSystemType.DISTRICT_BIOMASS: 0.70,
            HeatingSystemType.HEAT_PUMP_GROUND: 0.20,
            HeatingSystemType.HEAT_PUMP_AIR: 0.10,
        },
        typical_heating_kwh_m2=28,  # 65% of BBR

        dhw_kwh_m2=20,

        occupancy_w_per_m2=1.8,
        lighting_w_m2=5,
        equipment_w_m2=4,

        typical_forms=["lamellhus", "punkthus", "kvarter"],
        typical_facades=["puts", "tegel", "tra", "metall"],

        common_issues=[
            "Meets Miljöbyggnad Gold (SGBC certification)",
            "65% of BBR energy requirements",
            "High indoor environment requirements",
            "Material selection requirements (chemical content)",
            "Third-party verified",
            "15 indicators across Energy/Materials/Indoor Environment",
        ],
        renovation_potential_kwh_m2=8,
        typical_ecms=["solar_pv"],

        description="Miljöbyggnad Gold: Sweden's leading certification requiring 65% of BBR energy use, "
                    "strict indoor climate requirements, and sustainable material choices.",

        descriptors=ArchetypeDescriptors(
            # Geometric
            building_depth_m=(12.0, 22.0),
            floor_to_floor_m=(2.7, 3.2),
            building_length_m=(20.0, 100.0),
            plan_shape=[PlanShape.RECTANGULAR, PlanShape.L_SHAPE, PlanShape.COURTYARD],
            stairwell_apartments=(4, 8),

            # Visual
            balcony_types=[BalconyType.RECESSED, BalconyType.PROJECTING, BalconyType.GLAZED],
            roof_profiles=[RoofProfile.FLAT, RoofProfile.LOW_SLOPE],
            facade_patterns=[FacadePattern.REGULAR_PUNCHED, FacadePattern.MIXED],
            typical_colors=["white", "gray", "brick_red", "wood_natural"],
            window_proportions="square",

            # Context
            urban_settings=[UrbanSetting.INNER_CITY, UrbanSetting.INNER_SUBURB, UrbanSetting.SATELLITE_TOWN],
            typical_neighborhoods=["Norra Djurgårdsstaden", "Hammarby Sjöstad", "Hyllie"],
            typical_cities=["Stockholm", "Göteborg", "Malmö", "Uppsala", "Linköping"],

            # Ownership
            original_ownership=[OwnershipType.COOPERATIVE, OwnershipType.HOUSING_COMPANY, OwnershipType.MUNICIPAL],
            housing_programs=["Miljöbyggnad"],
            notable_developers=["JM", "NCC", "Skanska", "Riksbyggen", "HSB"],
            notable_architects=["Various major firms"],

            # Certification
            typical_certifications=[
                EnergyCertification.MILJOBYGGNAD_GOLD,
                EnergyCertification.ENERGY_CLASS_A,
                EnergyCertification.ENERGY_CLASS_B,
            ],
            has_solar_pv=True,
            has_ev_charging=True,

            # Keywords
            keywords_sv=["miljöbyggnad", "guld", "hållbart", "certifierat", "energieffektivt"],
            keywords_en=["environmental building", "gold certified", "sustainable", "energy efficient"],

            # Calibration
            infiltration_variability="low",
            u_value_variability="low",
            occupancy_pattern="residential",

            # Renovation indicators
            likely_renovated_if=["N/A - new construction"],
            renovation_era_signs={},
        ),
    ),
}


# =============================================================================
# SWEDISH HISTORICAL ARCHETYPES (PRE-1930)
# =============================================================================
# Detailed historical building types for older Swedish urban buildings
# Sources: Stockholmskällan, Swedish building history, Visit Sweden, Stadshem
# Research on stenstaden, landshövdingehus, and historical construction

SWEDISH_HISTORICAL_ARCHETYPES: Dict[str, DetailedArchetype] = {

    # =========================================================================
    # MEDIEVAL/OLD TOWN (Pre-1700) - Gamla Stan style
    # =========================================================================
    "hist_medieval_pre1700": DetailedArchetype(
        id="hist_medieval_pre1700",
        name_sv="Medeltida stenhus (Gamla Stan)",
        name_en="Medieval stone building (Old Town)",
        era=BuildingEra.MEDIEVAL_PRE_1700,
        year_start=1300,
        year_end=1699,

        stock_share_percent=0.3,  # Very rare, heritage protected
        typical_atemp_m2=(200, 800),
        typical_floors=(3, 5),

        wall_constructions=[
            WallConstruction(
                type=WallConstructionType.MEDIEVAL_STONE,
                name_sv="Medeltida stenmur",
                name_en="Medieval stone masonry",
                total_thickness_mm=800,
                insulation_thickness_mm=0,
                insulation_type="none",
                u_value=1.6,
                thermal_bridge_factor=1.0,
                description="Thick stone walls with rubble core, lime mortar. "
                            "Original material from city wall demolitions (1600s)."
            ),
            WallConstruction(
                type=WallConstructionType.SOLID_BRICK_2_STONE,
                name_sv="Tjock tegelmur",
                name_en="Thick brick masonry",
                total_thickness_mm=500,
                insulation_thickness_mm=0,
                insulation_type="none",
                u_value=1.4,
                thermal_bridge_factor=1.0,
                description="Unusual brick format from 1700s, similar to monastery bricks"
            ),
        ],

        roof_construction=RoofConstruction(
            name_sv="Kallvind med torv/lera",
            insulation_thickness_mm=0,
            insulation_type="torv/lera",
            u_value=1.0,
            roof_type="cold_attic"
        ),

        floor_construction=FloorConstruction(
            name_sv="Stenvalv eller träbjälklag på murad grund",
            type="basement",
            insulation_thickness_mm=0,
            u_value=1.2
        ),

        window_construction=WindowConstruction(
            type=WindowType.SINGLE_PANE,
            name_sv="Blyinfattade fönster/enkelglas",
            name_en="Leaded glass/single pane",
            u_value_glass=5.7,
            u_value_installed=5.0,
            shgc=0.85,
            num_panes=1,
            gas_fill="air"
        ),
        typical_wwr=0.10,  # Small windows
        wwr_by_orientation={"N": 0.08, "S": 0.12, "E": 0.10, "W": 0.10},

        infiltration_ach=0.80,
        n50_ach=20.0,

        ventilation_type=VentilationType.NATURAL,
        ventilation_rate_l_s_m2=0.25,
        heat_recovery_efficiency=0.0,
        sfp_kw_per_m3s=0.0,

        heating_systems={
            HeatingSystemType.DISTRICT_BIOMASS: 0.85,
            HeatingSystemType.ELECTRIC_WATERBORNE: 0.10,
            HeatingSystemType.OIL_BOILER: 0.05,
        },
        typical_heating_kwh_m2=250,

        dhw_kwh_m2=30,

        occupancy_w_per_m2=2.0,
        lighting_w_m2=8,
        equipment_w_m2=5,

        typical_forms=["slutet_kvarter", "gathus"],
        typical_facades=["sten", "puts"],

        common_issues=[
            "K-märkt/skyddat - begränsade åtgärder",
            "Fuktproblem i tjocka murar",
            "Otillräcklig ventilation",
            "Höga takhöjder (3-4m) ger stora volymer",
            "Komplexa planlösningar",
        ],
        renovation_potential_kwh_m2=50,  # Limited due to heritage
        typical_ecms=[
            "window_secondary_glazing",
            "attic_insulation_internal",
            "improved_heating_controls",
        ],

        description="Medieval and early modern buildings in Old Town (Gamla Stan). "
                    "Thick stone/brick walls (500-1000mm), heritage protected. "
                    "North German architectural influence. Most from 1600s-1700s.",
        sources=["Stockholmskällan", "Wikipedia Gamla Stan", "Stockholms museum"],
        descriptors=ArchetypeDescriptors(
            building_depth_m=(8.0, 15.0),
            floor_to_floor_m=(3.0, 4.0),
            building_length_m=(10.0, 25.0),
            plan_shape=[PlanShape.RECTANGULAR, PlanShape.IRREGULAR],
            stairwell_apartments=(1, 3),
            balcony_types=[BalconyType.NONE],
            roof_profiles=[RoofProfile.PITCHED, RoofProfile.MANSARD],
            facade_patterns=[FacadePattern.IRREGULAR_PUNCHED],
            typical_colors=["gul", "röd", "orange", "terrakotta"],
            window_proportions="portrait",
            has_bay_windows=False,
            has_corner_windows=False,
            urban_settings=[UrbanSetting.INNER_CITY],
            typical_neighborhoods=["Gamla Stan", "Riddarholmen"],
            typical_cities=["Stockholm", "Visby"],
            original_ownership=[OwnershipType.PRIVATE_RENTAL, OwnershipType.INSTITUTIONAL],
            housing_programs=[],
            notable_developers=[],
            notable_architects=[],
            typical_certifications=[EnergyCertification.ENERGY_CLASS_G],
            keywords_sv=["gamla stan", "medeltida", "stenhus", "kulturminne", "k-märkt",
                        "riksintresse", "1600-tal", "1700-tal"],
            keywords_en=["old town", "medieval", "stone building", "heritage", "listed"],
            infiltration_variability="high",
            u_value_variability="high",
            occupancy_pattern="residential",
            likely_renovated_if=["modern fönster", "tilläggsisolering vind"],
            renovation_era_signs={"1970s": "inre modernisering", "1990s": "varsam renovering"},
        ),
    ),

    # =========================================================================
    # 1700s TOWNHOUSES (Stormaktstiden)
    # =========================================================================
    "hist_1700s_townhouse": DetailedArchetype(
        id="hist_1700s_townhouse",
        name_sv="1700-tals stadshus",
        name_en="18th century townhouse",
        era=BuildingEra.STORMAKTSTID_1700_1800,
        year_start=1700,
        year_end=1799,

        stock_share_percent=0.4,
        typical_atemp_m2=(300, 1200),
        typical_floors=(2, 4),

        wall_constructions=[
            WallConstruction(
                type=WallConstructionType.SOLID_BRICK_2_STONE,
                name_sv="2-stens tegelmur med puts",
                name_en="2-brick wall rendered",
                total_thickness_mm=480,
                insulation_thickness_mm=0,
                insulation_type="none",
                u_value=1.3,
                thermal_bridge_factor=1.0,
                description="Thick brick typical of 1700s burgher houses"
            ),
            WallConstruction(
                type=WallConstructionType.LOG_TIMBER,
                name_sv="Knuttimmer med panel",
                name_en="Log timber with cladding",
                total_thickness_mm=200,
                insulation_thickness_mm=0,
                insulation_type="air_gap",
                u_value=1.2,
                thermal_bridge_factor=1.0,
                description="Traditional log construction for smaller towns"
            ),
        ],

        roof_construction=RoofConstruction(
            name_sv="Kallvind med sågspån",
            insulation_thickness_mm=50,
            insulation_type="sågspån",
            u_value=0.7,
            roof_type="cold_attic"
        ),

        floor_construction=FloorConstruction(
            name_sv="Träbjälklag på murad källare",
            type="basement",
            insulation_thickness_mm=0,
            u_value=0.9
        ),

        window_construction=WindowConstruction(
            type=WindowType.SINGLE_PANE,
            name_sv="Spröjsade enkelglasfönster",
            name_en="Divided single pane windows",
            u_value_glass=5.5,
            u_value_installed=4.8,
            shgc=0.85,
            num_panes=1,
            gas_fill="air"
        ),
        typical_wwr=0.12,
        wwr_by_orientation={"N": 0.08, "S": 0.15, "E": 0.10, "W": 0.10},

        infiltration_ach=0.70,
        n50_ach=18.0,

        ventilation_type=VentilationType.NATURAL,
        ventilation_rate_l_s_m2=0.25,
        heat_recovery_efficiency=0.0,
        sfp_kw_per_m3s=0.0,

        heating_systems={
            HeatingSystemType.DISTRICT_BIOMASS: 0.70,
            HeatingSystemType.ELECTRIC_WATERBORNE: 0.20,
            HeatingSystemType.OIL_BOILER: 0.10,
        },
        typical_heating_kwh_m2=230,

        dhw_kwh_m2=30,

        occupancy_w_per_m2=2.0,
        lighting_w_m2=7,
        equipment_w_m2=5,

        typical_forms=["gathus", "herrgård"],
        typical_facades=["puts", "tegel", "trä"],

        common_issues=[
            "Ofta kulturskyddat",
            "Dragiga fönster",
            "Fuktproblem",
            "Kalla golv",
        ],
        renovation_potential_kwh_m2=60,
        typical_ecms=[
            "window_secondary_glazing",
            "attic_insulation",
            "air_sealing",
        ],

        description="18th century townhouses and burgher houses. "
                    "Stormaktstiden architecture with thick masonry or log construction.",
        sources=["Swedish architectural history", "Boverket"],
        descriptors=ArchetypeDescriptors(
            building_depth_m=(10.0, 18.0),
            floor_to_floor_m=(2.8, 3.5),
            building_length_m=(12.0, 30.0),
            plan_shape=[PlanShape.RECTANGULAR, PlanShape.L_SHAPE],
            stairwell_apartments=(1, 4),
            balcony_types=[BalconyType.NONE],
            roof_profiles=[RoofProfile.PITCHED, RoofProfile.MANSARD, RoofProfile.HIP],
            facade_patterns=[FacadePattern.REGULAR_PUNCHED],
            typical_colors=["gul", "vit", "röd", "grå"],
            window_proportions="portrait",
            has_bay_windows=False,
            has_corner_windows=False,
            urban_settings=[UrbanSetting.INNER_CITY, UrbanSetting.SMALL_TOWN],
            typical_neighborhoods=["Östermalm", "Norrmalm", "Södermalm"],
            typical_cities=["Stockholm", "Göteborg", "Uppsala", "Karlskrona"],
            original_ownership=[OwnershipType.PRIVATE_RENTAL],
            housing_programs=[],
            notable_developers=[],
            notable_architects=[],
            typical_certifications=[EnergyCertification.ENERGY_CLASS_G, EnergyCertification.ENERGY_CLASS_F],
            keywords_sv=["1700-tal", "borgarhus", "stadshus", "stormaktstiden", "puts",
                        "tegel", "herrgård", "kulturminne"],
            keywords_en=["18th century", "townhouse", "burgher house", "Georgian"],
            infiltration_variability="high",
            u_value_variability="medium",
            occupancy_pattern="residential",
            likely_renovated_if=["moderna fönster", "moderniserat kök/bad"],
            renovation_era_signs={"1950s": "modernisering", "1980s": "varsam renovering"},
        ),
    ),

    # =========================================================================
    # EARLY INDUSTRIAL (1800-1880)
    # =========================================================================
    "hist_1800_1880_industrial": DetailedArchetype(
        id="hist_1800_1880_industrial",
        name_sv="Industrialismens hus 1800-1880",
        name_en="Early industrial era 1800-1880",
        era=BuildingEra.INDUSTRIALISM_1800_1880,
        year_start=1800,
        year_end=1879,

        stock_share_percent=1.0,
        typical_atemp_m2=(400, 1500),
        typical_floors=(2, 4),

        wall_constructions=[
            WallConstruction(
                type=WallConstructionType.SOLID_BRICK_1_5_STONE,
                name_sv="1.5-stens tegelmur",
                name_en="1.5-brick wall",
                total_thickness_mm=360,
                insulation_thickness_mm=0,
                insulation_type="none",
                u_value=1.5,
                thermal_bridge_factor=1.0,
                description="Standard brick construction pre-building codes"
            ),
            WallConstruction(
                type=WallConstructionType.STANDING_TIMBER,
                name_sv="Stående timmer/resvirke",
                name_en="Standing timber frame",
                total_thickness_mm=150,
                insulation_thickness_mm=0,
                insulation_type="air_gap",
                u_value=1.3,
                thermal_bridge_factor=1.0,
                description="Precursor to modern stud frame"
            ),
        ],

        roof_construction=RoofConstruction(
            name_sv="Kallvind",
            insulation_thickness_mm=30,
            insulation_type="sågspån",
            u_value=0.8,
            roof_type="cold_attic"
        ),

        floor_construction=FloorConstruction(
            name_sv="Träbjälklag på torpargrund",
            type="crawlspace",
            insulation_thickness_mm=0,
            u_value=0.9
        ),

        window_construction=WindowConstruction(
            type=WindowType.SINGLE_PANE,
            name_sv="Enkelglasfönster",
            name_en="Single pane windows",
            u_value_glass=5.5,
            u_value_installed=4.5,
            shgc=0.85,
            num_panes=1,
            gas_fill="air"
        ),
        typical_wwr=0.14,
        wwr_by_orientation={"N": 0.10, "S": 0.16, "E": 0.12, "W": 0.12},

        infiltration_ach=0.60,
        n50_ach=15.0,

        ventilation_type=VentilationType.NATURAL,
        ventilation_rate_l_s_m2=0.30,
        heat_recovery_efficiency=0.0,
        sfp_kw_per_m3s=0.0,

        heating_systems={
            HeatingSystemType.DISTRICT_BIOMASS: 0.60,
            HeatingSystemType.ELECTRIC_WATERBORNE: 0.25,
            HeatingSystemType.OIL_BOILER: 0.15,
        },
        typical_heating_kwh_m2=210,

        dhw_kwh_m2=28,

        occupancy_w_per_m2=2.2,
        lighting_w_m2=7,
        equipment_w_m2=5,

        typical_forms=["gathus", "hyreshus"],
        typical_facades=["tegel", "puts", "trä"],

        common_issues=[
            "Pre-byggnadsstadga (1874) - varierande kvalitet",
            "Stora dragproblem",
            "Fukt i grunder",
            "Enkla fönster",
        ],
        renovation_potential_kwh_m2=90,
        typical_ecms=[
            "window_replacement",
            "wall_internal_insulation",
            "attic_insulation",
            "air_sealing",
        ],

        description="Early industrial era before first building codes (1874). "
                    "Variable construction quality, often workers' housing.",
        sources=["Boverket", "Swedish building history"],
        descriptors=ArchetypeDescriptors(
            building_depth_m=(10.0, 16.0),
            floor_to_floor_m=(2.6, 3.2),
            building_length_m=(15.0, 40.0),
            plan_shape=[PlanShape.RECTANGULAR],
            stairwell_apartments=(2, 6),
            balcony_types=[BalconyType.NONE],
            roof_profiles=[RoofProfile.PITCHED, RoofProfile.HIP],
            facade_patterns=[FacadePattern.REGULAR_PUNCHED],
            typical_colors=["gul", "röd", "vit", "brun"],
            window_proportions="portrait",
            has_bay_windows=False,
            has_corner_windows=False,
            urban_settings=[UrbanSetting.INNER_CITY, UrbanSetting.SMALL_TOWN],
            typical_neighborhoods=["Södermalm", "Vasastan", "Majorna"],
            typical_cities=["Stockholm", "Göteborg", "Norrköping", "Sundsvall"],
            original_ownership=[OwnershipType.PRIVATE_RENTAL],
            housing_programs=["Arbetarbostad"],
            notable_developers=[],
            notable_architects=[],
            typical_certifications=[EnergyCertification.ENERGY_CLASS_G],
            keywords_sv=["1800-tal", "industrialism", "arbetarbostad", "före byggnadsstadga",
                        "tegel", "trä", "torpargrund"],
            keywords_en=["19th century", "industrial era", "workers housing", "pre-code"],
            infiltration_variability="high",
            u_value_variability="high",
            occupancy_pattern="residential",
            likely_renovated_if=["nya fönster", "moderniserat kök/bad", "tilläggsisolering"],
            renovation_era_signs={"1950s": "standardhöjning", "1970s": "stambyte"},
        ),
    ),

    # =========================================================================
    # STENSTADEN (1880-1915) - Stone City Era
    # =========================================================================
    "hist_stenstaden": DetailedArchetype(
        id="hist_stenstaden",
        name_sv="Stenstaden (1880-1915)",
        name_en="Stone City Era (1880-1915)",
        era=BuildingEra.STENSTADEN_1880_1900,
        year_start=1880,
        year_end=1915,

        stock_share_percent=4.5,
        typical_atemp_m2=(1000, 4000),
        typical_floors=(4, 6),

        wall_constructions=[
            WallConstruction(
                type=WallConstructionType.SOLID_BRICK_1_5_STONE,
                name_sv="1.5-stens tegel med puts",
                name_en="1.5-brick rendered",
                total_thickness_mm=380,
                insulation_thickness_mm=0,
                insulation_type="none",
                u_value=1.3,
                thermal_bridge_factor=1.0,
                description="Standard stenstaden wall per 1874 Byggnadsstadga"
            ),
            WallConstruction(
                type=WallConstructionType.SOLID_BRICK_2_STONE,
                name_sv="2-stens tegel (bottenvåning)",
                name_en="2-brick (ground floor)",
                total_thickness_mm=480,
                insulation_thickness_mm=0,
                insulation_type="none",
                u_value=1.2,
                thermal_bridge_factor=1.0,
                description="Thicker walls on ground floors per regulations"
            ),
        ],

        roof_construction=RoofConstruction(
            name_sv="Kallvind med sågspån",
            insulation_thickness_mm=80,
            insulation_type="sågspån/kutterspån",
            u_value=0.5,
            roof_type="cold_attic"
        ),

        floor_construction=FloorConstruction(
            name_sv="Träbjälklag på murad grund",
            type="basement",
            insulation_thickness_mm=0,
            u_value=0.7
        ),

        window_construction=WindowConstruction(
            type=WindowType.COUPLED_2_PANE,
            name_sv="Kopplade tvåglasfönster",
            name_en="Coupled double windows",
            u_value_glass=3.0,
            u_value_installed=2.8,
            shgc=0.75,
            num_panes=2,
            gas_fill="air"
        ),
        typical_wwr=0.20,
        wwr_by_orientation={"N": 0.15, "S": 0.25, "E": 0.18, "W": 0.18},

        infiltration_ach=0.35,
        n50_ach=9.0,

        ventilation_type=VentilationType.NATURAL,
        ventilation_rate_l_s_m2=0.35,
        heat_recovery_efficiency=0.0,
        sfp_kw_per_m3s=0.0,

        heating_systems={
            HeatingSystemType.DISTRICT_BIOMASS: 0.80,
            HeatingSystemType.OIL_BOILER: 0.10,
            HeatingSystemType.ELECTRIC_WATERBORNE: 0.10,
        },
        typical_heating_kwh_m2=170,

        dhw_kwh_m2=25,

        occupancy_w_per_m2=2.5,
        lighting_w_m2=8,
        equipment_w_m2=6,

        typical_forms=["slutet_kvarter", "kvartersstad"],
        typical_facades=["tegel", "puts"],

        common_issues=[
            "Köldbryggor vid balkonger och burspråk",
            "Självdragsventilation med skorstenar",
            "Höga rumshöjder (3+ meter)",
            "Ornament och detaljer försvårar isolering",
        ],
        renovation_potential_kwh_m2=70,
        typical_ecms=[
            "window_replacement",
            "attic_insulation",
            "ftx_installation",
            "air_sealing",
        ],

        description="Stenstaden era (1880-1915). 5-story brick blocks built per "
                    "1874 Byggnadsstadga and 1866 Lindhagen Plan. Grid pattern streets, "
                    "enclosed blocks. Exists in Stockholm, Gothenburg, Malmö, Sundsvall.",
        sources=["Stockholmskällan", "Visit Sweden", "Wikipedia Stenstaden"],
        descriptors=ArchetypeDescriptors(
            building_depth_m=(12.0, 18.0),
            floor_to_floor_m=(3.0, 3.8),
            building_length_m=(20.0, 60.0),
            plan_shape=[PlanShape.COURTYARD, PlanShape.RECTANGULAR],
            stairwell_apartments=(2, 6),
            balcony_types=[BalconyType.NONE, BalconyType.FRENCH],
            roof_profiles=[RoofProfile.PITCHED, RoofProfile.MANSARD],
            facade_patterns=[FacadePattern.REGULAR_PUNCHED, FacadePattern.VERTICAL_EMPHASIS],
            typical_colors=["gul", "röd", "vit", "terrakotta", "grå"],
            window_proportions="portrait",
            has_bay_windows=True,
            has_corner_windows=False,
            urban_settings=[UrbanSetting.INNER_CITY],
            typical_neighborhoods=["Östermalm", "Vasastan", "Kungsholmen", "Linnéstaden", "Limhamn"],
            typical_cities=["Stockholm", "Göteborg", "Malmö", "Sundsvall", "Gävle"],
            original_ownership=[OwnershipType.PRIVATE_RENTAL],
            housing_programs=["Stenstaden"],
            notable_developers=[],
            notable_architects=["Isak Gustaf Clason", "Ferdinand Boberg"],
            typical_certifications=[EnergyCertification.ENERGY_CLASS_F, EnergyCertification.ENERGY_CLASS_G],
            keywords_sv=["stenstaden", "sekelskifte", "jugend", "nationalromantik", "burspråk",
                        "slutet kvarter", "innergård", "1880-tal", "1890-tal", "1900-tal"],
            keywords_en=["stone city", "turn of century", "art nouveau", "enclosed block", "courtyard"],
            infiltration_variability="medium",
            u_value_variability="medium",
            occupancy_pattern="residential",
            likely_renovated_if=["stambytt", "nya fönster", "vindsinredning"],
            renovation_era_signs={"1960s": "balkongbygge", "1980s": "fönsterbyte", "2000s": "vindsinredning"},
        ),
    ),

    # =========================================================================
    # LANDSHÖVDINGEHUS (1875-1945) - Gothenburg specific
    # =========================================================================
    "hist_landshovdingehus": DetailedArchetype(
        id="hist_landshovdingehus",
        name_sv="Landshövdingehus (Göteborg)",
        name_en="Governor's House (Gothenburg)",
        era=BuildingEra.STENSTADEN_1880_1900,
        year_start=1875,
        year_end=1945,

        stock_share_percent=0.8,  # Gothenburg specific
        typical_atemp_m2=(400, 1200),
        typical_floors=(3, 3),  # Always 3 floors

        wall_constructions=[
            WallConstruction(
                type=WallConstructionType.BRICK_WOOD_HYBRID,
                name_sv="Tegel + trä hybrid",
                name_en="Brick + wood hybrid",
                total_thickness_mm=250,
                insulation_thickness_mm=0,
                insulation_type="air_gap",
                u_value=1.1,
                thermal_bridge_factor=1.10,
                description="Ground floor brick (fire safety), upper floors wood. "
                            "Unique Gothenburg invention from 1875."
            ),
        ],

        roof_construction=RoofConstruction(
            name_sv="Kallvind med sågspån",
            insulation_thickness_mm=60,
            insulation_type="sågspån",
            u_value=0.55,
            roof_type="cold_attic"
        ),

        floor_construction=FloorConstruction(
            name_sv="Träbjälklag på murad källare",
            type="basement",
            insulation_thickness_mm=0,
            u_value=0.8
        ),

        window_construction=WindowConstruction(
            type=WindowType.COUPLED_2_PANE,
            name_sv="Kopplade 2-glas",
            name_en="Coupled double windows",
            u_value_glass=3.0,
            u_value_installed=2.8,
            shgc=0.75,
            num_panes=2,
            gas_fill="air"
        ),
        typical_wwr=0.18,
        wwr_by_orientation={"N": 0.14, "S": 0.22, "E": 0.16, "W": 0.16},

        infiltration_ach=0.40,
        n50_ach=10.0,

        ventilation_type=VentilationType.NATURAL,
        ventilation_rate_l_s_m2=0.35,
        heat_recovery_efficiency=0.0,
        sfp_kw_per_m3s=0.0,

        heating_systems={
            HeatingSystemType.DISTRICT_BIOMASS: 0.85,
            HeatingSystemType.ELECTRIC_WATERBORNE: 0.10,
            HeatingSystemType.OIL_BOILER: 0.05,
        },
        typical_heating_kwh_m2=165,

        dhw_kwh_m2=25,

        occupancy_w_per_m2=2.5,
        lighting_w_m2=7,
        equipment_w_m2=6,

        typical_forms=["landshovdingehus"],
        typical_facades=["tegel_trä"],

        common_issues=[
            "Trästomme känslig för fukt",
            "Köldbrygga mellan tegel och trä",
            "Självdragsventilation",
            "Brand- och skadedjursrisk i trädelar",
        ],
        renovation_potential_kwh_m2=65,
        typical_ecms=[
            "wall_internal_insulation",
            "window_replacement",
            "attic_insulation",
            "air_sealing",
        ],

        description="Landshövdingehus (Governor's House) - unique to Gothenburg. "
                    "3 floors: brick ground floor + 2 wooden upper floors. "
                    "Built 1875-1945 as workers' housing. ~1000 remain today. "
                    "Per 1875 fire regulations requiring masonry at ground level.",
        sources=["Wikipedia Landshövdingehus", "Goteborg.com", "Planning Perspectives"],
        descriptors=ArchetypeDescriptors(
            building_depth_m=(10.0, 14.0),
            floor_to_floor_m=(2.6, 3.0),
            building_length_m=(15.0, 40.0),
            plan_shape=[PlanShape.RECTANGULAR],
            stairwell_apartments=(2, 4),
            balcony_types=[BalconyType.NONE, BalconyType.PROJECTING],
            roof_profiles=[RoofProfile.PITCHED],
            facade_patterns=[FacadePattern.REGULAR_PUNCHED],
            typical_colors=["gul", "vit", "röd", "grön"],
            window_proportions="portrait",
            has_bay_windows=False,
            has_corner_windows=False,
            urban_settings=[UrbanSetting.INNER_CITY, UrbanSetting.INNER_SUBURB],
            typical_neighborhoods=["Haga", "Majorna", "Linnéstaden", "Masthugget", "Olskroken"],
            typical_cities=["Göteborg"],
            original_ownership=[OwnershipType.PRIVATE_RENTAL],
            housing_programs=["Arbetarbostad"],
            notable_developers=[],
            notable_architects=[],
            typical_certifications=[EnergyCertification.ENERGY_CLASS_F, EnergyCertification.ENERGY_CLASS_E],
            keywords_sv=["landshövdingehus", "göteborg", "tegel och trä", "arbetarbostad",
                        "tre våningar", "1800-tal", "1900-tal"],
            keywords_en=["governor's house", "gothenburg", "brick and wood", "workers housing"],
            infiltration_variability="high",
            u_value_variability="medium",
            occupancy_pattern="residential",
            likely_renovated_if=["fasadrenovering", "nya fönster", "tilläggsisolering"],
            renovation_era_signs={"1970s": "fönsterbyte", "1990s": "fasadisolering"},
        ),
    ),

    # =========================================================================
    # JUGEND/ART NOUVEAU (1900-1910)
    # =========================================================================
    "hist_jugend": DetailedArchetype(
        id="hist_jugend",
        name_sv="Jugend/Art Nouveau (1900-1910)",
        name_en="Art Nouveau (1900-1910)",
        era=BuildingEra.JUGEND_1900_1910,
        year_start=1900,
        year_end=1910,

        stock_share_percent=1.5,
        typical_atemp_m2=(800, 3000),
        typical_floors=(4, 6),

        wall_constructions=[
            WallConstruction(
                type=WallConstructionType.SOLID_BRICK_1_5_STONE,
                name_sv="1.5-stens tegel med putsad fasad",
                name_en="1.5-brick rendered (Art Nouveau)",
                total_thickness_mm=380,
                insulation_thickness_mm=0,
                insulation_type="none",
                u_value=1.2,
                thermal_bridge_factor=1.0,
                description="Smooth plaster in yellows/oranges, sparse decoration, "
                            "narrow relief bands, sometimes glazed tiles"
            ),
        ],

        roof_construction=RoofConstruction(
            name_sv="Brutet tak eller sadeltak",
            insulation_thickness_mm=80,
            insulation_type="sågspån/mineralull",
            u_value=0.45,
            roof_type="cold_attic"
        ),

        floor_construction=FloorConstruction(
            name_sv="Träbjälklag på murad grund",
            type="basement",
            insulation_thickness_mm=0,
            u_value=0.7
        ),

        window_construction=WindowConstruction(
            type=WindowType.COUPLED_2_PANE,
            name_sv="Kopplade tvåglasfönster med spröjs",
            name_en="Coupled windows with muntins",
            u_value_glass=2.9,
            u_value_installed=2.7,
            shgc=0.72,
            num_panes=2,
            gas_fill="air"
        ),
        typical_wwr=0.22,
        wwr_by_orientation={"N": 0.16, "S": 0.28, "E": 0.20, "W": 0.20},

        infiltration_ach=0.32,
        n50_ach=8.0,

        ventilation_type=VentilationType.NATURAL,
        ventilation_rate_l_s_m2=0.35,
        heat_recovery_efficiency=0.0,
        sfp_kw_per_m3s=0.0,

        heating_systems={
            HeatingSystemType.DISTRICT_BIOMASS: 0.80,
            HeatingSystemType.OIL_BOILER: 0.10,
            HeatingSystemType.ELECTRIC_WATERBORNE: 0.10,
        },
        typical_heating_kwh_m2=160,

        dhw_kwh_m2=25,

        occupancy_w_per_m2=2.5,
        lighting_w_m2=8,
        equipment_w_m2=6,

        typical_forms=["slutet_kvarter", "villa"],
        typical_facades=["puts", "tegel_glaserat"],

        common_issues=[
            "Ornament och burspråk ger köldbryggor",
            "Stora fönsterytor",
            "Kulturvärden begränsar åtgärder",
        ],
        renovation_potential_kwh_m2=65,
        typical_ecms=[
            "window_upgrade",
            "attic_insulation",
            "ftx_installation",
            "improved_controls",
        ],

        description="Art Nouveau / Jugend style (1900-1910). "
                    "Organic forms inspired by nature, Swedish variant more restrained. "
                    "Started with 1897 Stockholm exhibition.",
        sources=["Sekelskifte.com", "Stockholms läns museum"],
        descriptors=ArchetypeDescriptors(
            building_depth_m=(12.0, 18.0),
            floor_to_floor_m=(3.0, 3.6),
            building_length_m=(20.0, 50.0),
            plan_shape=[PlanShape.COURTYARD, PlanShape.RECTANGULAR],
            stairwell_apartments=(2, 4),
            balcony_types=[BalconyType.FRENCH, BalconyType.NONE],
            roof_profiles=[RoofProfile.PITCHED, RoofProfile.MANSARD],
            facade_patterns=[FacadePattern.REGULAR_PUNCHED, FacadePattern.IRREGULAR_PUNCHED],
            typical_colors=["gul", "orange", "terrakotta", "grön"],
            window_proportions="portrait",
            has_bay_windows=True,
            has_corner_windows=False,
            urban_settings=[UrbanSetting.INNER_CITY],
            typical_neighborhoods=["Lärkstaden", "Diplomatstaden", "Vasastan", "Linnéstaden"],
            typical_cities=["Stockholm", "Göteborg", "Malmö"],
            original_ownership=[OwnershipType.PRIVATE_RENTAL],
            housing_programs=[],
            notable_developers=[],
            notable_architects=["Ferdinand Boberg", "Lars Israel Wahlman"],
            typical_certifications=[EnergyCertification.ENERGY_CLASS_F, EnergyCertification.ENERGY_CLASS_E],
            keywords_sv=["jugend", "art nouveau", "sekelskifte", "organiska former",
                        "burspråk", "reliefband", "glaserade plattor", "1900-tal"],
            keywords_en=["art nouveau", "jugend", "turn of century", "organic forms"],
            infiltration_variability="medium",
            u_value_variability="medium",
            occupancy_pattern="residential",
            likely_renovated_if=["stambytt", "nya fönster"],
            renovation_era_signs={"1970s": "modernisering", "1990s": "varsam renovering"},
        ),
    ),

    # =========================================================================
    # NATIONALROMANTIK (1910-1920)
    # =========================================================================
    "hist_nationalromantik": DetailedArchetype(
        id="hist_nationalromantik",
        name_sv="Nationalromantik (1910-1920)",
        name_en="National Romanticism (1910-1920)",
        era=BuildingEra.NATIONALROMANTIK_1910_1920,
        year_start=1910,
        year_end=1920,

        stock_share_percent=1.8,
        typical_atemp_m2=(800, 3500),
        typical_floors=(3, 5),

        wall_constructions=[
            WallConstruction(
                type=WallConstructionType.SOLID_BRICK_1_5_STONE,
                name_sv="Mörkt tegel eller puts",
                name_en="Dark brick or render",
                total_thickness_mm=380,
                insulation_thickness_mm=0,
                insulation_type="none",
                u_value=1.2,
                thermal_bridge_factor=1.0,
                description="Heavy, closed character. Dark brick common. "
                            "Inspired by Old Town, Visby, Vasa castles."
            ),
            WallConstruction(
                type=WallConstructionType.LOG_TIMBER,
                name_sv="Tjärstruket timmer",
                name_en="Tar-coated timber",
                total_thickness_mm=180,
                insulation_thickness_mm=0,
                insulation_type="air_gap",
                u_value=1.0,
                thermal_bridge_factor=1.0,
                description="Timber facades, small-paned windows, "
                            "projecting upper floors (utkragning)"
            ),
        ],

        roof_construction=RoofConstruction(
            name_sv="Sadeltak med lösull",
            insulation_thickness_mm=100,
            insulation_type="mineralull",
            u_value=0.38,
            roof_type="cold_attic"
        ),

        floor_construction=FloorConstruction(
            name_sv="Träbjälklag på murad grund",
            type="basement",
            insulation_thickness_mm=0,
            u_value=0.65
        ),

        window_construction=WindowConstruction(
            type=WindowType.COUPLED_2_PANE,
            name_sv="Tätspröjsade kopplade fönster",
            name_en="Small-paned coupled windows",
            u_value_glass=2.9,
            u_value_installed=2.7,
            shgc=0.70,
            num_panes=2,
            gas_fill="air"
        ),
        typical_wwr=0.18,
        wwr_by_orientation={"N": 0.12, "S": 0.22, "E": 0.16, "W": 0.16},

        infiltration_ach=0.30,
        n50_ach=7.5,

        ventilation_type=VentilationType.NATURAL,
        ventilation_rate_l_s_m2=0.35,
        heat_recovery_efficiency=0.0,
        sfp_kw_per_m3s=0.0,

        heating_systems={
            HeatingSystemType.DISTRICT_BIOMASS: 0.75,
            HeatingSystemType.OIL_BOILER: 0.15,
            HeatingSystemType.ELECTRIC_WATERBORNE: 0.10,
        },
        typical_heating_kwh_m2=155,

        dhw_kwh_m2=25,

        occupancy_w_per_m2=2.5,
        lighting_w_m2=8,
        equipment_w_m2=6,

        typical_forms=["villa", "slutet_kvarter"],
        typical_facades=["tegel_mörk", "trä_tjärstruket", "puts"],

        common_issues=[
            "Tungt och slutet = begränsat dagsljus",
            "Komplexa takvinklar",
            "Kulturskydd begränsar åtgärder",
        ],
        renovation_potential_kwh_m2=60,
        typical_ecms=[
            "window_upgrade",
            "attic_insulation",
            "ftx_installation",
        ],

        description="National Romanticism (1910-1920). Reaction against Art Nouveau, "
                    "return to 'authentic' Swedish building traditions. "
                    "Inspired by medieval buildings, Visby, Vasa castles.",
        sources=["Sekelskifte.com", "Stockholms läns museum", "Gaveldekor.se"],
        descriptors=ArchetypeDescriptors(
            building_depth_m=(12.0, 18.0),
            floor_to_floor_m=(2.8, 3.4),
            building_length_m=(20.0, 50.0),
            plan_shape=[PlanShape.RECTANGULAR, PlanShape.COURTYARD],
            stairwell_apartments=(2, 4),
            balcony_types=[BalconyType.NONE, BalconyType.PROJECTING],
            roof_profiles=[RoofProfile.PITCHED, RoofProfile.HIP],
            facade_patterns=[FacadePattern.REGULAR_PUNCHED, FacadePattern.VERTICAL_EMPHASIS],
            typical_colors=["mörk tegel", "brun", "röd", "svart"],
            window_proportions="portrait",
            has_bay_windows=False,
            has_corner_windows=False,
            urban_settings=[UrbanSetting.INNER_CITY, UrbanSetting.INNER_SUBURB],
            typical_neighborhoods=["Röda Bergen", "Enskede", "Bromma"],
            typical_cities=["Stockholm", "Göteborg"],
            original_ownership=[OwnershipType.COOPERATIVE, OwnershipType.PRIVATE_RENTAL],
            housing_programs=["HSB-rörelsen"],
            notable_developers=["HSB"],
            notable_architects=["Ragnar Östberg", "Carl Westman", "Ivar Tengbom"],
            typical_certifications=[EnergyCertification.ENERGY_CLASS_F, EnergyCertification.ENERGY_CLASS_E],
            keywords_sv=["nationalromantik", "1910-tal", "mörkt tegel", "tjärstruken",
                        "medeltidsinspirerat", "slutet", "tungt"],
            keywords_en=["national romanticism", "1910s", "dark brick", "medieval inspired"],
            infiltration_variability="medium",
            u_value_variability="medium",
            occupancy_pattern="residential",
            likely_renovated_if=["stambytt", "fönsterbyte"],
            renovation_era_signs={"1970s": "stambyte", "2000s": "varsam renovering"},
        ),
    ),

    # =========================================================================
    # 20-TALSKLASSICISM (1920-1930)
    # =========================================================================
    "hist_20tal": DetailedArchetype(
        id="hist_20tal",
        name_sv="20-talsklassicism (1920-1930)",
        name_en="1920s Classicism",
        era=BuildingEra.TJUGOTAL_1920_1930,
        year_start=1920,
        year_end=1930,

        stock_share_percent=2.5,
        typical_atemp_m2=(800, 3000),
        typical_floors=(3, 5),

        wall_constructions=[
            WallConstruction(
                type=WallConstructionType.CAVITY_BRICK,
                name_sv="Hålmur med luftspalt",
                name_en="Cavity brick wall",
                total_thickness_mm=340,
                insulation_thickness_mm=0,
                insulation_type="air_cavity",
                u_value=1.0,
                thermal_bridge_factor=1.05,
                description="Introduction of cavity walls with air gap"
            ),
            WallConstruction(
                type=WallConstructionType.SOLID_BRICK_1_5_STONE,
                name_sv="1.5-stens tegel",
                name_en="1.5-brick solid",
                total_thickness_mm=360,
                insulation_thickness_mm=0,
                insulation_type="none",
                u_value=1.15,
                thermal_bridge_factor=1.0,
            ),
        ],

        roof_construction=RoofConstruction(
            name_sv="Kallvind med mineralull",
            insulation_thickness_mm=120,
            insulation_type="mineralull",
            u_value=0.32,
            roof_type="cold_attic"
        ),

        floor_construction=FloorConstruction(
            name_sv="Betong eller träbjälklag",
            type="basement",
            insulation_thickness_mm=0,
            u_value=0.6
        ),

        window_construction=WindowConstruction(
            type=WindowType.COUPLED_2_PANE,
            name_sv="Kopplade tvåglasfönster",
            name_en="Coupled double windows",
            u_value_glass=2.8,
            u_value_installed=2.6,
            shgc=0.72,
            num_panes=2,
            gas_fill="air"
        ),
        typical_wwr=0.18,
        wwr_by_orientation={"N": 0.14, "S": 0.22, "E": 0.16, "W": 0.16},

        infiltration_ach=0.28,
        n50_ach=7.0,

        ventilation_type=VentilationType.NATURAL,
        ventilation_rate_l_s_m2=0.35,
        heat_recovery_efficiency=0.0,
        sfp_kw_per_m3s=0.0,

        heating_systems={
            HeatingSystemType.DISTRICT_BIOMASS: 0.70,
            HeatingSystemType.OIL_BOILER: 0.20,
            HeatingSystemType.ELECTRIC_WATERBORNE: 0.10,
        },
        typical_heating_kwh_m2=150,

        dhw_kwh_m2=25,

        occupancy_w_per_m2=2.5,
        lighting_w_m2=8,
        equipment_w_m2=6,

        typical_forms=["lamellhus", "slutet_kvarter"],
        typical_facades=["puts", "tegel"],

        common_issues=[
            "Hålmur med begränsad isolerförmåga",
            "Självdragsventilation",
            "Höga rumshöjder (2.7-3m)",
        ],
        renovation_potential_kwh_m2=55,
        typical_ecms=[
            "wall_cavity_fill",
            "window_replacement",
            "attic_insulation",
            "ftx_installation",
        ],

        description="1920s Swedish Classicism. Transition to functionalism. "
                    "Introduction of cavity walls. HSB founded 1923. "
                    "Per Capita housing program started.",
        sources=["Swedish architectural history", "Boverket"],
        descriptors=ArchetypeDescriptors(
            building_depth_m=(11.0, 15.0),
            floor_to_floor_m=(2.7, 3.2),
            building_length_m=(20.0, 50.0),
            plan_shape=[PlanShape.SLAB, PlanShape.COURTYARD],
            stairwell_apartments=(2, 4),
            balcony_types=[BalconyType.NONE, BalconyType.RECESSED],
            roof_profiles=[RoofProfile.PITCHED, RoofProfile.HIP],
            facade_patterns=[FacadePattern.REGULAR_PUNCHED],
            typical_colors=["gul", "vit", "grå", "rosa"],
            window_proportions="portrait",
            has_bay_windows=False,
            has_corner_windows=False,
            urban_settings=[UrbanSetting.INNER_SUBURB],
            typical_neighborhoods=["Röda Bergen", "Kungsholmen", "Aspudden"],
            typical_cities=["Stockholm", "Göteborg", "Malmö"],
            original_ownership=[OwnershipType.COOPERATIVE],
            housing_programs=["HSB-rörelsen", "Per Capita"],
            notable_developers=["HSB"],
            notable_architects=["Sven Wallander"],
            typical_certifications=[EnergyCertification.ENERGY_CLASS_E, EnergyCertification.ENERGY_CLASS_F],
            keywords_sv=["20-tal", "klassicism", "HSB", "hålmur", "putsad fasad",
                        "1920-tal", "kooperativ"],
            keywords_en=["1920s classicism", "HSB", "cavity wall", "cooperative"],
            infiltration_variability="medium",
            u_value_variability="low",
            occupancy_pattern="residential",
            likely_renovated_if=["nya fönster", "tilläggsisolering vind"],
            renovation_era_signs={"1970s": "fönsterbyte", "1990s": "balkongbygge"},
        ),
    ),

    # =========================================================================
    # EGNAHEM (1900-1930) - Workers' Own Homes
    # =========================================================================
    "hist_egnahem": DetailedArchetype(
        id="hist_egnahem",
        name_sv="Egnahem (1900-1930)",
        name_en="Workers' Own Home (1900-1930)",
        era=BuildingEra.NATIONALROMANTIK_1910_1920,
        year_start=1904,
        year_end=1948,

        stock_share_percent=3.0,
        typical_atemp_m2=(60, 120),
        typical_floors=(1, 2),

        wall_constructions=[
            WallConstruction(
                type=WallConstructionType.STUD_FRAME_SAWDUST,
                name_sv="Regelstomme med sågspån",
                name_en="Stud frame with sawdust",
                total_thickness_mm=150,
                insulation_thickness_mm=100,
                insulation_type="sågspån",
                u_value=0.8,
                thermal_bridge_factor=1.10,
                description="Early stud frame with sawdust/wood chip insulation"
            ),
            WallConstruction(
                type=WallConstructionType.LOG_TIMBER,
                name_sv="Timmer med panel",
                name_en="Log with cladding",
                total_thickness_mm=180,
                insulation_thickness_mm=0,
                insulation_type="air_gap",
                u_value=1.0,
                thermal_bridge_factor=1.0,
            ),
        ],

        roof_construction=RoofConstruction(
            name_sv="Kallvind med sågspån",
            insulation_thickness_mm=80,
            insulation_type="sågspån",
            u_value=0.5,
            roof_type="cold_attic"
        ),

        floor_construction=FloorConstruction(
            name_sv="Träbjälklag på torpargrund",
            type="crawlspace",
            insulation_thickness_mm=0,
            u_value=0.8
        ),

        window_construction=WindowConstruction(
            type=WindowType.COUPLED_2_PANE,
            name_sv="Kopplade 2-glas",
            name_en="Coupled double windows",
            u_value_glass=3.0,
            u_value_installed=2.8,
            shgc=0.75,
            num_panes=2,
            gas_fill="air"
        ),
        typical_wwr=0.14,
        wwr_by_orientation={"N": 0.10, "S": 0.18, "E": 0.12, "W": 0.12},

        infiltration_ach=0.35,
        n50_ach=8.0,

        ventilation_type=VentilationType.NATURAL,
        ventilation_rate_l_s_m2=0.30,
        heat_recovery_efficiency=0.0,
        sfp_kw_per_m3s=0.0,

        heating_systems={
            HeatingSystemType.ELECTRIC_DIRECT: 0.35,
            HeatingSystemType.HEAT_PUMP_AIR: 0.30,
            HeatingSystemType.OIL_BOILER: 0.20,
            HeatingSystemType.DISTRICT_BIOMASS: 0.15,
        },
        typical_heating_kwh_m2=175,

        dhw_kwh_m2=28,

        occupancy_w_per_m2=2.2,
        lighting_w_m2=6,
        equipment_w_m2=5,

        typical_forms=["egnahem", "villa_1_plan", "parhus"],
        typical_facades=["trä", "puts"],

        common_issues=[
            "Bristfällig isolering",
            "Fuktproblem i krypgrund",
            "Dragiga fönster och dörrar",
            "Små rum med låga i tak",
        ],
        renovation_potential_kwh_m2=80,
        typical_ecms=[
            "wall_external_insulation",
            "window_replacement",
            "attic_insulation",
            "crawlspace_insulation",
            "heat_pump_air",
        ],

        description="Egnahem (workers' own home) movement 1904-1948. "
                    "State-funded program for working-class home ownership. "
                    "Small wooden houses (~100 m²) with gardens. "
                    "Examples: Landala Egnahem (Gothenburg), Eneborg (Helsingborg).",
        sources=["Wikipedia Egnahemsrörelsen", "World Garden Cities"],
        descriptors=ArchetypeDescriptors(
            building_depth_m=(7.0, 10.0),
            floor_to_floor_m=(2.4, 2.7),
            building_length_m=(8.0, 14.0),
            plan_shape=[PlanShape.RECTANGULAR],
            stairwell_apartments=(1, 1),
            balcony_types=[BalconyType.NONE],
            roof_profiles=[RoofProfile.PITCHED, RoofProfile.HIP],
            facade_patterns=[FacadePattern.REGULAR_PUNCHED],
            typical_colors=["vit", "gul", "röd", "grön"],
            window_proportions="portrait",
            has_bay_windows=False,
            has_corner_windows=False,
            urban_settings=[UrbanSetting.OUTER_SUBURB, UrbanSetting.SMALL_TOWN],
            typical_neighborhoods=["Landala", "Eneborg", "Enskede", "Stureby"],
            typical_cities=["Göteborg", "Helsingborg", "Stockholm", "Malmö"],
            original_ownership=[OwnershipType.PRIVATE_RENTAL],
            housing_programs=["Egnahemsrörelsen"],
            notable_developers=["Statens järnvägar"],
            notable_architects=[],
            typical_certifications=[EnergyCertification.ENERGY_CLASS_G, EnergyCertification.ENERGY_CLASS_F],
            keywords_sv=["egnahem", "arbetarbostad", "småhus", "trähus", "1900-tal",
                        "trädgårdsstad", "villa", "småstugor"],
            keywords_en=["workers home", "garden city", "wooden house", "1900s"],
            infiltration_variability="high",
            u_value_variability="high",
            occupancy_pattern="residential",
            likely_renovated_if=["tillbyggnad", "nya fönster", "tilläggsisolering"],
            renovation_era_signs={"1970s": "tillbyggd", "1990s": "nya fönster", "2010s": "bergvärme"},
        ),
    ),
}


# =============================================================================
# SWEDISH STOCKHOLM-SPECIFIC & SPECIAL FORM ARCHETYPES
# =============================================================================
# These archetypes represent unique building forms common in Stockholm region
# that have distinct construction characteristics not covered by era-based archetypes

SWEDISH_SPECIAL_FORM_ARCHETYPES: Dict[str, DetailedArchetype] = {

    # =========================================================================
    # BARNRIKEHUS (Large Family Housing) 1935-1948
    # Social housing for low-income large families, heavily subsidized
    # Narrow depth (~12m) for TB prevention, simple functionalist design
    # =========================================================================
    "barnrikehus_1935_1948": DetailedArchetype(
        id="barnrikehus_1935_1948",
        name_sv="Barnrikehus",
        name_en="Large Family Housing (Barnrikehus)",
        era=BuildingEra.FUNKIS_1930_1945,
        year_start=1935,
        year_end=1948,

        stock_share_percent=1.2,
        typical_atemp_m2=(1500, 4000),
        typical_floors=(4, 5),

        wall_constructions=[
            WallConstruction(
                type=WallConstructionType.SOLID_BRICK_1_STONE,
                name_sv="1-stens tegelmur putsad",
                name_en="Rendered single brick wall",
                total_thickness_mm=340,
                insulation_thickness_mm=0,
                insulation_type="ingen (luftspalt)",
                u_value=1.1,
                thermal_bridge_factor=1.15,
                description="Barnrikehus: simple rendered brick, often pastel colors"
            ),
        ],
        roof_construction=RoofConstruction(
            name_sv="Sadeltak med vindsutrymme",
            insulation_thickness_mm=100,
            insulation_type="kutterspån/sågspån",
            u_value=0.45,
            roof_type="cold_attic"
        ),
        floor_construction=FloorConstruction(
            name_sv="Källare under hela huset",
            type="basement",
            insulation_thickness_mm=0,
            u_value=0.50
        ),
        window_construction=WindowConstruction(
            type=WindowType.COUPLED_2_PANE,
            name_sv="Kopplade tvåglasfönster",
            name_en="Coupled double-pane",
            u_value_glass=2.9,
            u_value_installed=2.6,
            shgc=0.76,
            num_panes=2,
            gas_fill="air"
        ),
        typical_wwr=0.20,

        infiltration_ach=0.35,
        n50_ach=4.0,

        ventilation_type=VentilationType.NATURAL,
        ventilation_rate_l_s_m2=0.35,
        heat_recovery_efficiency=0.0,
        sfp_kw_per_m3s=0.0,

        heating_systems={
            HeatingSystemType.DISTRICT_OIL: 0.60,
            HeatingSystemType.DISTRICT_BIOMASS: 0.30,
            HeatingSystemType.ELECTRIC_DIRECT: 0.10,
        },
        typical_heating_kwh_m2=145,

        dhw_kwh_m2=25,

        occupancy_w_per_m2=2.0,
        lighting_w_m2=8,
        equipment_w_m2=4,

        typical_forms=["lamellhus"],
        typical_facades=["puts", "tegel"],

        common_issues=[
            "Narrow building depth (10-12m) for natural ventilation",
            "Minimal thermal insulation",
            "Small apartments designed for large families",
            "Now highly sought-after due to rent control",
            "Often heritage protected",
        ],
        renovation_potential_kwh_m2=40,
        typical_ecms=["window_replacement", "roof_insulation", "air_sealing"],

        description="Barnrikehus: Social housing built 1935-1948 for large low-income families. "
                    "Narrow depth for TB prevention and natural ventilation. "
                    "Simple functionalist design with pastel-colored rendered facades.",
        descriptors=ArchetypeDescriptors(
            building_depth_m=(10.0, 12.0),
            floor_to_floor_m=(2.7, 2.9),
            building_length_m=(30.0, 80.0),
            plan_shape=[PlanShape.SLAB, PlanShape.RECTANGULAR],
            stairwell_apartments=(4, 6),
            balcony_types=[BalconyType.RECESSED, BalconyType.NONE],
            roof_profiles=[RoofProfile.LOW_SLOPE, RoofProfile.PITCHED],
            facade_patterns=[FacadePattern.REGULAR_PUNCHED],
            typical_colors=["gul", "rosa", "ljusblå", "grå", "vit"],
            window_proportions="portrait",
            has_bay_windows=False,
            has_corner_windows=False,
            urban_settings=[UrbanSetting.INNER_SUBURB],
            typical_neighborhoods=["Traneberg", "Gröndal", "Hammarbyhöjden", "Midsommarkransen"],
            typical_cities=["Stockholm", "Göteborg", "Malmö"],
            original_ownership=[OwnershipType.MUNICIPAL],
            housing_programs=["Barnrikehusreformen", "Socialbostäder"],
            notable_developers=["Svenska Bostäder", "Stockholmshem"],
            notable_architects=["Kooperativa Förbundets Arkitektkontor"],
            typical_certifications=[EnergyCertification.ENERGY_CLASS_F, EnergyCertification.ENERGY_CLASS_E],
            keywords_sv=["barnrikehus", "barnrikhus", "familjehus", "socialbostäder", "folkhem",
                        "1930-tal", "1940-tal", "funkis", "putsad fasad", "pastellfärg"],
            keywords_en=["large family housing", "social housing", "functionalist", "1930s", "1940s"],
            infiltration_variability="medium",
            u_value_variability="low",
            occupancy_pattern="residential",
            likely_renovated_if=["new windows", "tilläggsisolering vind", "FTX installerat"],
            renovation_era_signs={"1970s": "ytterskikt puts", "1990s": "nya fönster", "2010s": "FTX"},
        ),
    ),

    # =========================================================================
    # SMALHUS (Narrow Houses) 1935-1955
    # HSB specialty, optimized for daylighting and natural ventilation
    # 8-10m depth, 3-4 stories, often in curved arrangements following terrain
    # =========================================================================
    "smalhus_1935_1955": DetailedArchetype(
        id="smalhus_1935_1955",
        name_sv="Smalhus",
        name_en="Narrow Apartment Houses (Smalhus)",
        era=BuildingEra.FUNKIS_1930_1945,
        year_start=1935,
        year_end=1955,

        stock_share_percent=2.5,
        typical_atemp_m2=(800, 2500),
        typical_floors=(3, 4),

        wall_constructions=[
            WallConstruction(
                type=WallConstructionType.CAVITY_BRICK,
                name_sv="Hålmur med luftspalt",
                name_en="Cavity brick wall",
                total_thickness_mm=350,
                insulation_thickness_mm=50,
                insulation_type="luftspalt (ibland kutterspån)",
                u_value=0.95,
                thermal_bridge_factor=1.10,
                description="Smalhus: cavity wall construction, often yellow brick"
            ),
        ],
        roof_construction=RoofConstruction(
            name_sv="Sadeltak eller platt tak",
            insulation_thickness_mm=120,
            insulation_type="kutterspån/mineralull",
            u_value=0.40,
            roof_type="cold_attic"
        ),
        floor_construction=FloorConstruction(
            name_sv="Källare",
            type="basement",
            insulation_thickness_mm=0,
            u_value=0.45
        ),
        window_construction=WindowConstruction(
            type=WindowType.COUPLED_2_PANE,
            name_sv="Kopplade tvåglasfönster",
            name_en="Coupled double-pane",
            u_value_glass=2.9,
            u_value_installed=2.5,
            shgc=0.76,
            num_panes=2,
            gas_fill="air"
        ),
        typical_wwr=0.22,

        infiltration_ach=0.30,
        n50_ach=3.5,

        ventilation_type=VentilationType.NATURAL,
        ventilation_rate_l_s_m2=0.35,
        heat_recovery_efficiency=0.0,
        sfp_kw_per_m3s=0.0,

        heating_systems={
            HeatingSystemType.DISTRICT_BIOMASS: 0.70,
            HeatingSystemType.DISTRICT_OIL: 0.20,
            HeatingSystemType.HEAT_PUMP_EXHAUST: 0.10,
        },
        typical_heating_kwh_m2=135,

        dhw_kwh_m2=25,

        occupancy_w_per_m2=2.0,
        lighting_w_m2=8,
        equipment_w_m2=4,

        typical_forms=["lamellhus", "smal_lamell"],
        typical_facades=["tegel", "puts"],

        common_issues=[
            "Excellent daylighting due to narrow depth",
            "Often curved following terrain (stighuslängor)",
            "Good natural ventilation but drafty",
            "HSB cooperative heritage",
            "Well-maintained common areas typically",
        ],
        renovation_potential_kwh_m2=35,
        typical_ecms=["window_replacement", "roof_insulation", "air_sealing"],

        description="Smalhus: HSB specialty narrow apartment blocks (8-10m depth) "
                    "optimized for daylighting and natural ventilation.",
        descriptors=ArchetypeDescriptors(
            building_depth_m=(8.0, 10.0),
            floor_to_floor_m=(2.6, 2.8),
            building_length_m=(20.0, 60.0),
            plan_shape=[PlanShape.SLAB, PlanShape.RECTANGULAR],
            stairwell_apartments=(2, 4),
            balcony_types=[BalconyType.RECESSED, BalconyType.FRENCH],
            roof_profiles=[RoofProfile.LOW_SLOPE, RoofProfile.FLAT],
            facade_patterns=[FacadePattern.REGULAR_PUNCHED, FacadePattern.HORIZONTAL_BANDS],
            typical_colors=["gul", "ljusgrå", "vit", "beige"],
            window_proportions="portrait",
            has_bay_windows=False,
            has_corner_windows=True,
            urban_settings=[UrbanSetting.INNER_SUBURB],
            typical_neighborhoods=["Gröndal", "Aspudden", "Enskede", "Mälarhöjden", "Hammarbyhöjden"],
            typical_cities=["Stockholm", "Göteborg"],
            original_ownership=[OwnershipType.COOPERATIVE],
            housing_programs=["HSB-rörelsen", "Kooperativ hyresrätt"],
            notable_developers=["HSB"],
            notable_architects=["Sven Wallander", "Eskil Sundahl"],
            typical_certifications=[EnergyCertification.ENERGY_CLASS_E, EnergyCertification.ENERGY_CLASS_D],
            keywords_sv=["smalhus", "smal", "HSB", "kooperativ", "terrängföljande",
                        "stighuslängor", "1940-tal", "1950-tal"],
            keywords_en=["narrow houses", "HSB", "cooperative", "terrain-following", "1940s", "1950s"],
            infiltration_variability="medium",
            u_value_variability="low",
            occupancy_pattern="residential",
            likely_renovated_if=["new windows", "FTX", "added balconies"],
            renovation_era_signs={"1980s": "tilläggsisolering", "2000s": "nya fönster"},
        ),
    ),

    # =========================================================================
    # STJÄRNHUS (Star Houses) 1944-1962
    # Backström & Reinius design, Y-shaped/star floor plan
    # 3-6 stories, rough plaster in strong colors, honeycomb arrangements
    # =========================================================================
    "stjarnhus_1944_1962": DetailedArchetype(
        id="stjarnhus_1944_1962",
        name_sv="Stjärnhus",
        name_en="Star Houses (Stjärnhus)",
        era=BuildingEra.FOLKHEM_1946_1960,
        year_start=1944,
        year_end=1962,

        stock_share_percent=1.0,
        typical_atemp_m2=(1200, 3500),
        typical_floors=(3, 6),

        wall_constructions=[
            WallConstruction(
                type=WallConstructionType.LIGHT_CONCRETE_BLOCK,
                name_sv="Lättbetongblock putsad",
                name_en="Rendered lightweight concrete block",
                total_thickness_mm=300,
                insulation_thickness_mm=0,
                insulation_type="lättbetong (självbärande isolering)",
                u_value=0.60,
                thermal_bridge_factor=1.15,
                description="Stjärnhus: rough plaster in strong colors (red, yellow, green)"
            ),
        ],
        roof_construction=RoofConstruction(
            name_sv="Platt tak eller låglutande",
            insulation_thickness_mm=150,
            insulation_type="mineralull/lättbetong",
            u_value=0.35,
            roof_type="flat"
        ),
        floor_construction=FloorConstruction(
            name_sv="Platta på mark eller souterräng",
            type="slab_on_grade",
            insulation_thickness_mm=50,
            u_value=0.50
        ),
        window_construction=WindowConstruction(
            type=WindowType.COUPLED_2_PANE,
            name_sv="Kopplade tvåglasfönster",
            name_en="Coupled double-pane",
            u_value_glass=2.8,
            u_value_installed=2.4,
            shgc=0.75,
            num_panes=2,
            gas_fill="air"
        ),
        typical_wwr=0.25,

        infiltration_ach=0.25,
        n50_ach=3.0,

        ventilation_type=VentilationType.NATURAL,
        ventilation_rate_l_s_m2=0.35,
        heat_recovery_efficiency=0.0,
        sfp_kw_per_m3s=0.0,

        heating_systems={
            HeatingSystemType.DISTRICT_BIOMASS: 0.75,
            HeatingSystemType.DISTRICT_OIL: 0.15,
            HeatingSystemType.HEAT_PUMP_EXHAUST: 0.10,
        },
        typical_heating_kwh_m2=120,

        dhw_kwh_m2=25,

        occupancy_w_per_m2=2.0,
        lighting_w_m2=8,
        equipment_w_m2=4,

        typical_forms=["stjarnhus"],
        typical_facades=["puts"],

        common_issues=[
            "Three-pointed star plan with shared stairwell",
            "Light from three directions per apartment",
            "Often in honeycomb patterns around hexagonal courtyards",
            "Nationally significant cultural heritage (Gröndal, Västertorp)",
            "Balcony for every apartment",
        ],
        renovation_potential_kwh_m2=25,  # Limited due to heritage
        typical_ecms=["window_secondary_glazing", "roof_insulation"],

        description="Stjärnhus: Backström & Reinius design Y-shaped floor plan (1944-1962). "
                    "Rough plaster in strong colors (red, yellow, green). Riksintresse.",
        descriptors=ArchetypeDescriptors(
            building_depth_m=(14.0, 18.0),
            floor_to_floor_m=(2.7, 2.9),
            building_length_m=(25.0, 35.0),
            plan_shape=[PlanShape.STAR],
            stairwell_apartments=(3, 6),
            balcony_types=[BalconyType.RECESSED, BalconyType.PROJECTING],
            roof_profiles=[RoofProfile.FLAT, RoofProfile.LOW_SLOPE],
            facade_patterns=[FacadePattern.REGULAR_PUNCHED, FacadePattern.HORIZONTAL_BANDS],
            typical_colors=["röd", "gul", "grön", "blå", "orange"],
            window_proportions="square",
            has_bay_windows=False,
            has_corner_windows=True,
            urban_settings=[UrbanSetting.INNER_SUBURB],
            typical_neighborhoods=["Gröndal", "Västertorp", "Årsta", "Björkhagen"],
            typical_cities=["Stockholm"],
            original_ownership=[OwnershipType.COOPERATIVE, OwnershipType.MUNICIPAL],
            housing_programs=["HSB-rörelsen", "Kooperativ hyresrätt"],
            notable_developers=["HSB", "Svenska Bostäder"],
            notable_architects=["Sven Backström", "Leif Reinius"],
            typical_certifications=[EnergyCertification.ENERGY_CLASS_D, EnergyCertification.ENERGY_CLASS_E],
            keywords_sv=["stjärnhus", "trepunktshus", "Y-hus", "bikakeplan", "Backström och Reinius",
                        "riksintresse", "1950-tal", "kulturarv"],
            keywords_en=["star house", "Y-shaped", "honeycomb", "1950s", "heritage"],
            infiltration_variability="low",
            u_value_variability="medium",
            occupancy_pattern="residential",
            likely_renovated_if=["ändrad fasadfärg", "inglasade balkonger"],
            renovation_era_signs={"1980s": "omputsning", "2000s": "fönsterbyte"},
        ),
    ),

    # =========================================================================
    # PUNKTHUS (Tower Blocks) 1950-1970
    # 8-11 story tower blocks, often marking neighborhood centers
    # Elevator buildings, typically one per stairwell
    # =========================================================================
    "punkthus_1950_1970": DetailedArchetype(
        id="punkthus_1950_1970",
        name_sv="Punkthus",
        name_en="Tower Blocks (Punkthus)",
        era=BuildingEra.FOLKHEM_1946_1960,
        year_start=1950,
        year_end=1970,

        stock_share_percent=2.0,
        typical_atemp_m2=(2000, 5000),
        typical_floors=(8, 11),

        wall_constructions=[
            WallConstruction(
                type=WallConstructionType.LIGHT_CONCRETE_BLOCK,
                name_sv="Lättbetongblock",
                name_en="Lightweight concrete block",
                total_thickness_mm=350,
                insulation_thickness_mm=50,
                insulation_type="lättbetong + mineralull",
                u_value=0.55,
                thermal_bridge_factor=1.20,
                description="Punkthus: often brick veneer or rendered"
            ),
            WallConstruction(
                type=WallConstructionType.CONCRETE_SANDWICH,
                name_sv="Betongsandwich (senare)",
                name_en="Concrete sandwich (later)",
                total_thickness_mm=300,
                insulation_thickness_mm=80,
                insulation_type="mineralull/cellplast",
                u_value=0.45,
                thermal_bridge_factor=1.25,
                description="Later punkthus with prefab sandwich panels"
            ),
        ],
        roof_construction=RoofConstruction(
            name_sv="Platt tak",
            insulation_thickness_mm=150,
            insulation_type="mineralull",
            u_value=0.30,
            roof_type="flat"
        ),
        floor_construction=FloorConstruction(
            name_sv="Källare med garage",
            type="basement",
            insulation_thickness_mm=50,
            u_value=0.45
        ),
        window_construction=WindowConstruction(
            type=WindowType.COUPLED_2_PANE,
            name_sv="Kopplade tvåglasfönster",
            name_en="Coupled double-pane",
            u_value_glass=2.8,
            u_value_installed=2.4,
            shgc=0.75,
            num_panes=2,
            gas_fill="air"
        ),
        typical_wwr=0.22,

        infiltration_ach=0.22,
        n50_ach=2.5,

        ventilation_type=VentilationType.EXHAUST,
        ventilation_rate_l_s_m2=0.35,
        heat_recovery_efficiency=0.0,
        sfp_kw_per_m3s=1.0,

        heating_systems={
            HeatingSystemType.DISTRICT_BIOMASS: 0.80,
            HeatingSystemType.DISTRICT_OIL: 0.15,
            HeatingSystemType.HEAT_PUMP_EXHAUST: 0.05,
        },
        typical_heating_kwh_m2=115,

        dhw_kwh_m2=25,

        occupancy_w_per_m2=2.0,
        lighting_w_m2=8,
        equipment_w_m2=5,

        typical_forms=["punkthus"],
        typical_facades=["tegel", "puts", "betong"],

        common_issues=[
            "First Swedish high-rise residential buildings",
            "Elevator buildings (1 per stairwell)",
            "Often landmark buildings in neighborhood centers",
            "Concrete construction with brick or render facade",
            "Extensive views but wind exposure issues",
        ],
        renovation_potential_kwh_m2=40,
        typical_ecms=["window_replacement", "roof_insulation", "ftx_installation"],

        description="Punkthus: Tower blocks (8-11 stories) common 1950-1970. "
                    "Often landmark buildings marking neighborhood centers.",
        descriptors=ArchetypeDescriptors(
            building_depth_m=(16.0, 22.0),
            floor_to_floor_m=(2.7, 3.0),
            building_length_m=(16.0, 22.0),
            plan_shape=[PlanShape.POINT, PlanShape.RECTANGULAR],
            stairwell_apartments=(4, 6),
            balcony_types=[BalconyType.RECESSED, BalconyType.PROJECTING],
            roof_profiles=[RoofProfile.FLAT],
            facade_patterns=[FacadePattern.REGULAR_PUNCHED, FacadePattern.VERTICAL_EMPHASIS],
            typical_colors=["vit", "ljusgrå", "beige", "gul"],
            window_proportions="square",
            has_bay_windows=False,
            has_corner_windows=True,
            has_roof_terrace=True,
            urban_settings=[UrbanSetting.INNER_SUBURB, UrbanSetting.OUTER_SUBURB],
            typical_neighborhoods=["Hässelby", "Vällingby", "Farsta", "Hökarängen", "Blackeberg"],
            typical_cities=["Stockholm", "Göteborg", "Malmö"],
            original_ownership=[OwnershipType.COOPERATIVE, OwnershipType.MUNICIPAL],
            housing_programs=["ABC-städer", "Grannskapsplanering"],
            notable_developers=["Svenska Bostäder", "HSB", "Stockholmshem"],
            notable_architects=["Sven Markelius", "Nils Tesch"],
            typical_certifications=[EnergyCertification.ENERGY_CLASS_D, EnergyCertification.ENERGY_CLASS_E],
            keywords_sv=["punkthus", "höghus", "tornhus", "1950-tal", "1960-tal",
                        "hiss", "utsikt", "centrumbyggnad"],
            keywords_en=["tower block", "point block", "high-rise", "1950s", "1960s"],
            infiltration_variability="low",
            u_value_variability="medium",
            occupancy_pattern="residential",
            likely_renovated_if=["fasadrenovering", "nya fönster", "hissrenovering"],
            renovation_era_signs={"1990s": "fasadisolering", "2010s": "FTX installation"},
        ),
    ),

    # =========================================================================
    # KOLLEKTIVHUS (Collective Housing) 1935-1980
    # Shared services (kitchen, daycare, laundry)
    # Designed for working families, feminist housing concept
    # =========================================================================
    "kollektivhus_1935_1980": DetailedArchetype(
        id="kollektivhus_1935_1980",
        name_sv="Kollektivhus",
        name_en="Collective Housing (Kollektivhus)",
        era=BuildingEra.FUNKIS_1930_1945,
        year_start=1935,
        year_end=1980,

        stock_share_percent=0.3,
        typical_atemp_m2=(2000, 8000),
        typical_floors=(5, 8),

        wall_constructions=[
            WallConstruction(
                type=WallConstructionType.CAVITY_BRICK,
                name_sv="Tegelmur med luftspalt",
                name_en="Cavity brick wall",
                total_thickness_mm=400,
                insulation_thickness_mm=50,
                insulation_type="luftspalt/mineralull",
                u_value=0.85,
                thermal_bridge_factor=1.15,
                description="Kollektivhus: quality construction, often yellow brick"
            ),
        ],
        roof_construction=RoofConstruction(
            name_sv="Platt tak",
            insulation_thickness_mm=150,
            insulation_type="mineralull",
            u_value=0.35,
            roof_type="flat"
        ),
        floor_construction=FloorConstruction(
            name_sv="Källare med gemensamma utrymmen",
            type="basement",
            insulation_thickness_mm=0,
            u_value=0.45
        ),
        window_construction=WindowConstruction(
            type=WindowType.COUPLED_2_PANE,
            name_sv="Kopplade tvåglasfönster",
            name_en="Coupled double-pane",
            u_value_glass=2.8,
            u_value_installed=2.5,
            shgc=0.75,
            num_panes=2,
            gas_fill="air"
        ),
        typical_wwr=0.25,

        infiltration_ach=0.25,
        n50_ach=3.0,

        ventilation_type=VentilationType.EXHAUST,
        ventilation_rate_l_s_m2=0.40,
        heat_recovery_efficiency=0.0,
        sfp_kw_per_m3s=1.0,

        heating_systems={
            HeatingSystemType.DISTRICT_BIOMASS: 0.85,
            HeatingSystemType.DISTRICT_OIL: 0.10,
            HeatingSystemType.ELECTRIC_WATERBORNE: 0.05,
        },
        typical_heating_kwh_m2=125,

        dhw_kwh_m2=30,  # Higher due to shared services

        occupancy_w_per_m2=2.0,
        lighting_w_m2=8,
        equipment_w_m2=5,

        typical_forms=["lamellhus", "slutet_kvarter"],
        typical_facades=["tegel", "puts"],

        common_issues=[
            "Shared restaurant/central kitchen with food elevators",
            "Integrated daycare facilities",
            "Smaller private kitchens, larger common spaces",
            "Markeliushuset (1935) is iconic example",
            "Designed for women's emancipation",
            "About 40 true kollektivhus exist in Sweden today",
        ],
        renovation_potential_kwh_m2=30,  # Limited due to heritage
        typical_ecms=["window_replacement", "roof_insulation"],

        description="Kollektivhus: Collective housing with shared services (kitchen, daycare). "
                    "Feminist housing concept for working families. About 40 exist in Sweden.",
        descriptors=ArchetypeDescriptors(
            building_depth_m=(14.0, 20.0),
            floor_to_floor_m=(2.8, 3.2),
            building_length_m=(40.0, 100.0),
            plan_shape=[PlanShape.RECTANGULAR, PlanShape.SLAB, PlanShape.COURTYARD],
            stairwell_apartments=(4, 8),
            balcony_types=[BalconyType.PROJECTING, BalconyType.RECESSED],
            roof_profiles=[RoofProfile.FLAT, RoofProfile.LOW_SLOPE],
            facade_patterns=[FacadePattern.REGULAR_PUNCHED, FacadePattern.RIBBON_WINDOWS],
            typical_colors=["vit", "ljusgrå", "gul", "röd"],
            window_proportions="landscape",
            has_bay_windows=False,
            has_corner_windows=True,
            urban_settings=[UrbanSetting.INNER_CITY, UrbanSetting.INNER_SUBURB],
            typical_neighborhoods=["Södermalm", "Kungsholmen", "Vasastan", "Marieberg"],
            typical_cities=["Stockholm", "Göteborg"],
            original_ownership=[OwnershipType.COOPERATIVE],
            housing_programs=["Kooperativ hyresrätt", "Kollektivboende"],
            notable_developers=["HSB", "SKB"],
            notable_architects=["Sven Markelius", "Albin Stark"],
            typical_certifications=[EnergyCertification.ENERGY_CLASS_E, EnergyCertification.ENERGY_CLASS_D],
            keywords_sv=["kollektivhus", "servicehus", "matsal", "daghem", "gemensam kök",
                        "Markeliushuset", "feminism", "arbetande familjer"],
            keywords_en=["collective housing", "service house", "communal dining", "shared kitchen"],
            infiltration_variability="medium",
            u_value_variability="medium",
            occupancy_pattern="residential",
            likely_renovated_if=["stängd matsal", "ombyggda lägenheter"],
            renovation_era_signs={"1980s": "stängda gemensamma ytor", "2000s": "återöppnade tjänster"},
        ),
    ),

    # =========================================================================
    # RENOVATED MILJONPROGRAMMET (ROT 1985-2000)
    # 1961-1975 buildings upgraded with new windows, insulation, balcony glazing
    # Significantly improved U-values from original construction
    # =========================================================================
    "miljonprogrammet_renovated": DetailedArchetype(
        id="miljonprogrammet_renovated",
        name_sv="Renoverat Miljonprogramshus",
        name_en="Renovated Million Programme Housing",
        era=BuildingEra.REKORD_1961_1975,
        year_start=1961,
        year_end=1975,  # Original construction

        stock_share_percent=5.0,  # Many have been renovated
        typical_atemp_m2=(2000, 8000),
        typical_floors=(3, 8),

        wall_constructions=[
            WallConstruction(
                type=WallConstructionType.CONCRETE_SANDWICH,
                name_sv="Betongsandwich tilläggsisolerad",
                name_en="Additional insulated concrete sandwich",
                total_thickness_mm=400,
                insulation_thickness_mm=180,  # Original 80 + added 100
                insulation_type="mineralull (original + tillägg)",
                u_value=0.25,  # Improved from ~0.45
                thermal_bridge_factor=1.15,
                description="ROT-renovated: additional facade insulation"
            ),
        ],
        roof_construction=RoofConstruction(
            name_sv="Tilläggsisolerat tak",
            insulation_thickness_mm=350,  # Upgraded
            insulation_type="mineralull",
            u_value=0.15,
            roof_type="flat"
        ),
        floor_construction=FloorConstruction(
            name_sv="Platta på mark",
            type="slab_on_grade",
            insulation_thickness_mm=100,
            u_value=0.35
        ),
        window_construction=WindowConstruction(
            type=WindowType.TRIPLE_PANE,
            name_sv="Nya treglasfönster",
            name_en="New triple-pane windows",
            u_value_glass=1.1,
            u_value_installed=1.3,
            shgc=0.50,
            num_panes=3,
            gas_fill="argon"
        ),
        typical_wwr=0.20,

        infiltration_ach=0.12,  # Improved from 0.20
        n50_ach=1.5,  # Improved from 3.0

        ventilation_type=VentilationType.EXHAUST,  # Often upgraded to FTX
        ventilation_rate_l_s_m2=0.35,
        heat_recovery_efficiency=0.0,  # Unless FTX installed
        sfp_kw_per_m3s=1.0,

        heating_systems={
            HeatingSystemType.DISTRICT_BIOMASS: 0.85,
            HeatingSystemType.HEAT_PUMP_EXHAUST: 0.10,
            HeatingSystemType.HEAT_PUMP_GROUND: 0.05,
        },
        typical_heating_kwh_m2=85,  # Down from ~130

        dhw_kwh_m2=25,

        occupancy_w_per_m2=2.0,
        lighting_w_m2=7,
        equipment_w_m2=4,

        typical_forms=["lamellhus", "skivhus"],
        typical_facades=["puts", "betong"],

        common_issues=[
            "Underwent ROT renovation 1985-2000",
            "New windows (triple-pane)",
            "Additional facade insulation (often rendered)",
            "Upgraded ventilation (sometimes FTX)",
            "Balcony glazing common",
            "8% energy reduction typical without envelope work",
            "30%+ reduction with deep renovation",
        ],
        renovation_potential_kwh_m2=20,  # Already renovated
        typical_ecms=["ftx_installation", "solar_pv"],

        description="Renovated Miljonprogrammet: 1961-1975 buildings upgraded through ROT program. "
                    "New windows, additional insulation, improved airtightness.",
        descriptors=ArchetypeDescriptors(
            building_depth_m=(11.0, 14.0),
            floor_to_floor_m=(2.5, 2.7),
            building_length_m=(30.0, 80.0),
            plan_shape=[PlanShape.SLAB, PlanShape.RECTANGULAR],
            stairwell_apartments=(2, 4),
            balcony_types=[BalconyType.GLAZED, BalconyType.PROJECTING],
            roof_profiles=[RoofProfile.FLAT, RoofProfile.LOW_SLOPE],
            facade_patterns=[FacadePattern.REGULAR_PUNCHED, FacadePattern.HORIZONTAL_BANDS],
            typical_colors=["vit", "grå", "beige", "ljusgul"],
            window_proportions="landscape",
            has_bay_windows=False,
            has_corner_windows=False,
            urban_settings=[UrbanSetting.OUTER_SUBURB, UrbanSetting.SATELLITE_TOWN],
            typical_neighborhoods=["Tensta", "Rinkeby", "Rosengård", "Bergsjön", "Skärholmen"],
            typical_cities=["Stockholm", "Göteborg", "Malmö"],
            original_ownership=[OwnershipType.MUNICIPAL],
            housing_programs=["Miljonprogrammet", "ROT-programmet"],
            notable_developers=["Svenska Bostäder", "Stockholmshem", "MKB"],
            notable_architects=[],
            typical_certifications=[EnergyCertification.ENERGY_CLASS_D, EnergyCertification.ENERGY_CLASS_C],
            keywords_sv=["miljonprogrammet", "renoverat", "ROT", "tilläggsisolerat",
                        "nya fönster", "inglasad balkong", "1960-tal", "1970-tal"],
            keywords_en=["million programme", "renovated", "additional insulation", "1960s", "1970s"],
            infiltration_variability="low",
            u_value_variability="low",
            occupancy_pattern="residential",
            likely_renovated_if=["always - this archetype is post-renovation"],
            renovation_era_signs={"1990s": "ny puts + fönster", "2000s": "FTX + balkongglasning"},
        ),
    ),

    # =========================================================================
    # SUSTAINABLE DISTRICT (Hammarby Sjöstad type) 1998-2016
    # Integrated environmental systems, district heating/cooling
    # High environmental standards, vacuum waste, wastewater heat recovery
    # =========================================================================
    "sustainable_district_2000s": DetailedArchetype(
        id="sustainable_district_2000s",
        name_sv="Hållbar stadsdel (Hammarby-typ)",
        name_en="Sustainable District (Hammarby Sjöstad type)",
        era=BuildingEra.LAGENERGI_1996_2010,
        year_start=1998,
        year_end=2016,

        stock_share_percent=1.5,
        typical_atemp_m2=(1500, 4000),
        typical_floors=(4, 8),

        wall_constructions=[
            WallConstruction(
                type=WallConstructionType.CONCRETE_SANDWICH,
                name_sv="Betongsandwich välisolerad",
                name_en="Well-insulated concrete sandwich",
                total_thickness_mm=400,
                insulation_thickness_mm=200,
                insulation_type="mineralull/cellplast",
                u_value=0.18,
                thermal_bridge_factor=1.10,
                description="Hammarby type: high insulation, discontinuous block form"
            ),
        ],
        roof_construction=RoofConstruction(
            name_sv="Platt tak välisolerat",
            insulation_thickness_mm=400,
            insulation_type="mineralull",
            u_value=0.10,
            roof_type="flat"
        ),
        floor_construction=FloorConstruction(
            name_sv="Platta på mark välisolerad",
            type="slab_on_grade",
            insulation_thickness_mm=200,
            u_value=0.20
        ),
        window_construction=WindowConstruction(
            type=WindowType.LOW_E_TRIPLE,
            name_sv="Lågemissions treglas",
            name_en="Low-E triple glazing",
            u_value_glass=0.9,
            u_value_installed=1.1,
            shgc=0.45,
            num_panes=3,
            gas_fill="argon",
            coating="low-e"
        ),
        typical_wwr=0.25,

        infiltration_ach=0.06,
        n50_ach=0.8,

        ventilation_type=VentilationType.HEAT_RECOVERY,
        ventilation_rate_l_s_m2=0.35,
        heat_recovery_efficiency=0.80,
        sfp_kw_per_m3s=1.5,

        heating_systems={
            HeatingSystemType.DISTRICT_BIOMASS: 0.95,  # 34% from wastewater
            HeatingSystemType.HEAT_PUMP_GROUND: 0.05,
        },
        typical_heating_kwh_m2=55,

        dhw_kwh_m2=20,

        occupancy_w_per_m2=2.0,
        lighting_w_m2=6,
        equipment_w_m2=4,

        typical_forms=["lamellhus", "kvarter"],
        typical_facades=["puts", "betong", "tra"],

        common_issues=[
            "Integrated environmental system (Hammarby Model)",
            "District heating from wastewater (34%)",
            "Vacuum waste collection (Envac)",
            "Solar panels common",
            "Biogas from wastewater for cooking",
            "30-40% lower environmental impact vs 1990s standard",
            "Water use 150 L/person/day (vs 200 normal)",
        ],
        renovation_potential_kwh_m2=10,  # Already highly efficient
        typical_ecms=["solar_pv"],

        description="Sustainable district buildings (Hammarby Sjöstad type): Integrated environmental "
                    "systems with wastewater heat recovery, vacuum waste, and biogas production.",
        descriptors=ArchetypeDescriptors(
            building_depth_m=(12.0, 16.0),
            floor_to_floor_m=(2.7, 3.0),
            building_length_m=(25.0, 60.0),
            plan_shape=[PlanShape.RECTANGULAR, PlanShape.COURTYARD, PlanShape.SLAB],
            stairwell_apartments=(2, 4),
            balcony_types=[BalconyType.GLAZED, BalconyType.PROJECTING, BalconyType.RECESSED],
            roof_profiles=[RoofProfile.FLAT, RoofProfile.LOW_SLOPE],
            facade_patterns=[FacadePattern.CURTAIN_WALL, FacadePattern.MIXED, FacadePattern.REGULAR_PUNCHED],
            typical_colors=["vit", "trä", "grå", "terrakotta"],
            window_proportions="portrait",
            has_bay_windows=False,
            has_corner_windows=True,
            has_roof_terrace=True,
            urban_settings=[UrbanSetting.INNER_CITY, UrbanSetting.INNER_SUBURB],
            typical_neighborhoods=["Hammarby Sjöstad", "Norra Djurgårdsstaden", "Årstaberg", "Henriksdal"],
            typical_cities=["Stockholm", "Malmö", "Göteborg"],
            original_ownership=[OwnershipType.COOPERATIVE, OwnershipType.PRIVATE_RENTAL],
            housing_programs=["Miljöstadsdel", "Hållbar stad"],
            notable_developers=["JM", "Skanska", "NCC", "Riksbyggen"],
            notable_architects=["White Arkitekter", "Wingårdhs", "Tengbom"],
            typical_certifications=[EnergyCertification.MILJOBYGGNAD_SILVER, EnergyCertification.ENERGY_CLASS_B],
            has_solar_pv=True,
            keywords_sv=["hammarby", "sjöstad", "miljöstadsdel", "hållbar", "vakuumsopor",
                        "biogas", "återvinning", "fjärrkyla", "2000-tal", "2010-tal"],
            keywords_en=["sustainable district", "eco district", "Hammarby Model", "vacuum waste"],
            infiltration_variability="low",
            u_value_variability="low",
            occupancy_pattern="residential",
            likely_renovated_if=["rarely - buildings are new"],
            renovation_era_signs={},
        ),
    ),

    # =========================================================================
    # CLT MULTI-FAMILY (2015+)
    # Cross-laminated timber construction
    # Carbon storage, different thermal mass, fire-protected
    # =========================================================================
    "clt_multifamily_2015_plus": DetailedArchetype(
        id="clt_multifamily_2015_plus",
        name_sv="KL-trä flerbostadshus",
        name_en="CLT Multi-family Housing",
        era=BuildingEra.NARA_NOLL_2011_PLUS,
        year_start=2015,
        year_end=2030,

        stock_share_percent=0.5,  # Growing rapidly
        typical_atemp_m2=(1000, 4000),
        typical_floors=(4, 9),

        wall_constructions=[
            WallConstruction(
                type=WallConstructionType.CLT,
                name_sv="Korslimmat trä (KL-trä)",
                name_en="Cross-laminated timber (CLT)",
                total_thickness_mm=400,
                insulation_thickness_mm=200,
                insulation_type="träfiberisolering/mineralull",
                u_value=0.12,
                thermal_bridge_factor=1.05,
                description="CLT: carbon-negative construction, stores 45+ tonnes CO2"
            ),
        ],
        roof_construction=RoofConstruction(
            name_sv="KL-trä tak",
            insulation_thickness_mm=450,
            insulation_type="träfiberisolering",
            u_value=0.08,
            roof_type="flat"
        ),
        floor_construction=FloorConstruction(
            name_sv="Betongplatta med träbjälklag ovan",
            type="slab_on_grade",
            insulation_thickness_mm=250,
            u_value=0.15
        ),
        window_construction=WindowConstruction(
            type=WindowType.PASSIVE_HOUSE,
            name_sv="Passivhusfönster",
            name_en="Passive house windows",
            u_value_glass=0.5,
            u_value_installed=0.8,
            shgc=0.40,
            num_panes=3,
            gas_fill="krypton",
            coating="triple low-e"
        ),
        typical_wwr=0.22,

        infiltration_ach=0.04,
        n50_ach=0.5,

        ventilation_type=VentilationType.HEAT_RECOVERY,
        ventilation_rate_l_s_m2=0.35,
        heat_recovery_efficiency=0.85,
        sfp_kw_per_m3s=1.5,

        heating_systems={
            HeatingSystemType.DISTRICT_BIOMASS: 0.60,
            HeatingSystemType.HEAT_PUMP_GROUND: 0.30,
            HeatingSystemType.HEAT_PUMP_AIR: 0.10,
        },
        typical_heating_kwh_m2=35,

        dhw_kwh_m2=20,

        occupancy_w_per_m2=2.0,
        lighting_w_m2=5,
        equipment_w_m2=4,

        typical_forms=["lamellhus", "punkthus"],
        typical_facades=["tra", "puts"],

        common_issues=[
            "Swedish CLT production (Martinsons since 2003)",
            "Carbon storage: 60m³ CLT = 45+ tonnes CO2",
            "Fire-protected with gypsum cladding",
            "Lower thermal mass than concrete (faster heating/cooling)",
            "Acoustic separation requires attention",
            "Stockholm planning 31 CLT towers (Anders Berensson)",
            "Near-passive house performance typical",
        ],
        renovation_potential_kwh_m2=5,  # Already near-passive
        typical_ecms=["solar_pv"],

        description="CLT multi-family housing: Cross-laminated timber construction (2015+). "
                    "Carbon-negative, stores 45+ tonnes CO2 per building. Near-passive house performance.",
        descriptors=ArchetypeDescriptors(
            building_depth_m=(12.0, 15.0),
            floor_to_floor_m=(2.8, 3.0),
            building_length_m=(20.0, 50.0),
            plan_shape=[PlanShape.RECTANGULAR, PlanShape.SLAB, PlanShape.POINT],
            stairwell_apartments=(2, 4),
            balcony_types=[BalconyType.PROJECTING, BalconyType.RECESSED],
            roof_profiles=[RoofProfile.FLAT, RoofProfile.PITCHED],
            facade_patterns=[FacadePattern.REGULAR_PUNCHED, FacadePattern.MIXED],
            typical_colors=["trä", "svart", "vit", "naturträ"],
            window_proportions="portrait",
            has_bay_windows=False,
            has_corner_windows=True,
            urban_settings=[UrbanSetting.INNER_CITY, UrbanSetting.INNER_SUBURB],
            typical_neighborhoods=["Norra Djurgårdsstaden", "Strandudden", "Linköping", "Skellefteå"],
            typical_cities=["Stockholm", "Göteborg", "Växjö", "Skellefteå", "Linköping"],
            original_ownership=[OwnershipType.COOPERATIVE, OwnershipType.PRIVATE_RENTAL],
            housing_programs=["Trästad", "Miljöbyggnad"],
            notable_developers=["Folkhem", "Lindbäcks", "Derome", "Moelven"],
            notable_architects=["Tengbom", "White Arkitekter", "C.F. Møller"],
            typical_certifications=[EnergyCertification.MILJOBYGGNAD_GOLD, EnergyCertification.FEBY_SILVER, EnergyCertification.PASSIVE_HOUSE],
            has_solar_pv=True,
            keywords_sv=["CLT", "KL-trä", "korslimmat", "träbyggnad", "massivträ",
                        "trähus", "koldioxidlagring", "klimatneutralt", "2010-tal", "2020-tal"],
            keywords_en=["CLT", "cross-laminated timber", "mass timber", "carbon negative", "wood building"],
            infiltration_variability="low",
            u_value_variability="low",
            occupancy_pattern="residential",
            likely_renovated_if=["rarely - buildings are new"],
            renovation_era_signs={},
        ),
    ),
}


# =============================================================================
# SWEDISH MULTI-FAMILY ARCHETYPES (FLERBOSTADSHUS)
# =============================================================================

SWEDISH_MFH_ARCHETYPES: Dict[str, DetailedArchetype] = {

    # =========================================================================
    # PRE-1930: Traditional construction (compatibility alias)
    # =========================================================================
    "mfh_pre_1930": DetailedArchetype(
        id="mfh_pre_1930",
        name_sv="Flerbostadshus före 1930",
        name_en="Multi-family pre-1930",
        era=BuildingEra.PRE_1930,
        year_start=1880,
        year_end=1929,

        stock_share_percent=8.5,
        typical_atemp_m2=(800, 3000),
        typical_floors=(3, 6),

        wall_constructions=[
            WallConstruction(
                type=WallConstructionType.SOLID_BRICK_1_5_STONE,
                name_sv="1.5-stens tegelmur",
                name_en="1.5 brick wall (360mm)",
                total_thickness_mm=360,
                insulation_thickness_mm=0,
                insulation_type="none",
                u_value=1.2,
                thermal_bridge_factor=1.0,
                description="Solid brick with lime mortar, no insulation"
            ),
            WallConstruction(
                type=WallConstructionType.SOLID_BRICK_1_STONE,
                name_sv="1-stens tegelmur med puts",
                name_en="1 brick wall with render",
                total_thickness_mm=280,
                insulation_thickness_mm=0,
                insulation_type="none",
                u_value=1.5,
                thermal_bridge_factor=1.0,
                description="Single brick with exterior render"
            ),
        ],

        roof_construction=RoofConstruction(
            name_sv="Kallvind med träbjälklag",
            insulation_thickness_mm=50,
            insulation_type="sågspån/kutterspån",
            u_value=0.6,
            roof_type="cold_attic"
        ),

        floor_construction=FloorConstruction(
            name_sv="Träbjälklag på murad grund",
            type="crawlspace",
            insulation_thickness_mm=0,
            u_value=0.8
        ),

        window_construction=WindowConstruction(
            type=WindowType.COUPLED_2_PANE,
            name_sv="Kopplade tvåglasfönster",
            name_en="Coupled double windows",
            u_value_glass=3.0,
            u_value_installed=2.8,
            shgc=0.75,
            num_panes=2,
            gas_fill="air"
        ),
        typical_wwr=0.18,
        wwr_by_orientation={"N": 0.15, "S": 0.20, "E": 0.18, "W": 0.18},

        infiltration_ach=0.40,
        n50_ach=10.0,

        ventilation_type=VentilationType.NATURAL,
        ventilation_rate_l_s_m2=0.35,
        heat_recovery_efficiency=0.0,
        sfp_kw_per_m3s=0.0,

        heating_systems={
            HeatingSystemType.DISTRICT_BIOMASS: 0.60,
            HeatingSystemType.OIL_BOILER: 0.25,
            HeatingSystemType.ELECTRIC_WATERBORNE: 0.15,
        },
        typical_heating_kwh_m2=180,

        dhw_kwh_m2=25,

        occupancy_w_per_m2=2.5,
        lighting_w_m2=8,
        equipment_w_m2=6,

        typical_forms=["slutet_kvarter", "trapphus"],
        typical_facades=["tegel", "puts"],

        common_issues=[
            "Hög infiltration",
            "Köldbryggor vid balkonger",
            "Dragiga fönster",
            "Fuktproblem i grund",
            "Dålig täthet vid fönster",
        ],
        renovation_potential_kwh_m2=80,
        typical_ecms=[
            "window_replacement",
            "attic_insulation",
            "air_sealing",
            "ftx_installation",
        ],

        description="Pre-war brick buildings, often with ornate facades and high ceilings",
        sources=["TABULA SE", "BETSI 2010", "Boverket Energiguiden"],
        descriptors=ArchetypeDescriptors(
            building_depth_m=(12.0, 18.0),
            floor_to_floor_m=(3.0, 3.8),
            building_length_m=(20.0, 60.0),
            plan_shape=[PlanShape.COURTYARD, PlanShape.RECTANGULAR],
            stairwell_apartments=(2, 6),
            balcony_types=[BalconyType.NONE, BalconyType.FRENCH],
            roof_profiles=[RoofProfile.PITCHED, RoofProfile.MANSARD],
            facade_patterns=[FacadePattern.REGULAR_PUNCHED, FacadePattern.VERTICAL_EMPHASIS],
            typical_colors=["gul", "röd", "terrakotta", "vit"],
            window_proportions="portrait",
            has_bay_windows=True,
            has_corner_windows=False,
            urban_settings=[UrbanSetting.INNER_CITY],
            typical_neighborhoods=["Östermalm", "Vasastan", "Södermalm", "Kungsholmen"],
            typical_cities=["Stockholm", "Göteborg", "Malmö"],
            original_ownership=[OwnershipType.PRIVATE_RENTAL],
            housing_programs=[],
            notable_developers=[],
            notable_architects=[],
            typical_certifications=[EnergyCertification.ENERGY_CLASS_F, EnergyCertification.ENERGY_CLASS_G],
            keywords_sv=["sekelskifte", "stenstaden", "tegel", "innergård", "före 1930"],
            keywords_en=["pre-war", "brick", "courtyard", "turn of century"],
            infiltration_variability="high",
            u_value_variability="medium",
            occupancy_pattern="residential",
            likely_renovated_if=["stambytt", "nya fönster", "vindsinredning"],
            renovation_era_signs={"1960s": "balkongbygge", "1980s": "fönsterbyte"},
        ),
    ),

    # =========================================================================
    # 1930-1945: Functionalism
    # =========================================================================
    "mfh_1930_1945": DetailedArchetype(
        id="mfh_1930_1945",
        name_sv="Flerbostadshus 1930-1945 (Funkis)",
        name_en="Multi-family 1930-1945 (Functionalism)",
        era=BuildingEra.FUNKIS_1930_1945,
        year_start=1930,
        year_end=1945,

        stock_share_percent=7.2,
        typical_atemp_m2=(800, 2500),
        typical_floors=(3, 5),

        wall_constructions=[
            WallConstruction(
                type=WallConstructionType.CAVITY_BRICK,
                name_sv="Hålmur med luftspalt",
                name_en="Cavity brick wall",
                total_thickness_mm=340,
                insulation_thickness_mm=0,
                insulation_type="air_cavity",
                u_value=1.0,
                thermal_bridge_factor=1.05,
                description="Cavity brick with air gap, characteristic of functionalism"
            ),
            WallConstruction(
                type=WallConstructionType.BRICK_LIGHT_CONCRETE,
                name_sv="Tegel + lättbetong",
                name_en="Brick + light concrete",
                total_thickness_mm=350,
                insulation_thickness_mm=100,
                insulation_type="lättbetong",
                u_value=0.8,
                thermal_bridge_factor=1.05,
            ),
        ],

        roof_construction=RoofConstruction(
            name_sv="Kallvind med mineralull",
            insulation_thickness_mm=100,
            insulation_type="mineralull",
            u_value=0.35,
            roof_type="cold_attic"
        ),

        floor_construction=FloorConstruction(
            name_sv="Betongbjälklag på krypgrund",
            type="crawlspace",
            insulation_thickness_mm=30,
            u_value=0.6
        ),

        window_construction=WindowConstruction(
            type=WindowType.COUPLED_2_PANE,
            name_sv="Kopplade tvåglasfönster",
            name_en="Coupled double windows",
            u_value_glass=2.9,
            u_value_installed=2.7,
            shgc=0.72,
            num_panes=2,
            gas_fill="air"
        ),
        typical_wwr=0.20,
        wwr_by_orientation={"N": 0.15, "S": 0.22, "E": 0.18, "W": 0.18},

        infiltration_ach=0.30,
        n50_ach=8.0,

        ventilation_type=VentilationType.NATURAL,
        ventilation_rate_l_s_m2=0.35,
        heat_recovery_efficiency=0.0,
        sfp_kw_per_m3s=0.0,

        heating_systems={
            HeatingSystemType.DISTRICT_BIOMASS: 0.70,
            HeatingSystemType.OIL_BOILER: 0.20,
            HeatingSystemType.ELECTRIC_WATERBORNE: 0.10,
        },
        typical_heating_kwh_m2=160,

        dhw_kwh_m2=25,

        occupancy_w_per_m2=2.5,
        lighting_w_m2=8,
        equipment_w_m2=6,

        typical_forms=["lamellhus", "slutet_kvarter"],
        typical_facades=["tegel", "puts"],

        common_issues=[
            "Luftspalt ger begränsad isolering",
            "Köldbryggor",
            "Självdragsventilation fungerar dåligt",
            "Låg täthet",
        ],
        renovation_potential_kwh_m2=70,
        typical_ecms=[
            "window_replacement",
            "attic_insulation",
            "air_sealing",
            "ftx_installation",
        ],

        description="Functionalist buildings with cleaner lines, cavity walls introduced",
        sources=["TABULA SE", "BETSI 2010"],
        descriptors=ArchetypeDescriptors(
            building_depth_m=(10.0, 14.0),
            floor_to_floor_m=(2.7, 3.0),
            building_length_m=(30.0, 80.0),
            plan_shape=[PlanShape.RECTANGULAR, PlanShape.L_SHAPE],
            stairwell_apartments=(2, 4),
            balcony_types=[BalconyType.RECESSED, BalconyType.PROJECTING],
            roof_profiles=[RoofProfile.FLAT, RoofProfile.LOW_PITCHED],
            facade_patterns=[FacadePattern.REGULAR_PUNCHED, FacadePattern.HORIZONTAL_BANDS],
            typical_colors=["vit", "ljusgrå", "gul", "ljusrosa"],
            window_proportions="square",
            has_bay_windows=False,
            has_corner_windows=True,
            urban_settings=[UrbanSetting.INNER_CITY, UrbanSetting.INNER_SUBURB],
            typical_neighborhoods=["Gärdet", "Fredhäll", "Hammarbyhöjden", "Örgryte"],
            typical_cities=["Stockholm", "Göteborg", "Malmö", "Uppsala"],
            original_ownership=[OwnershipType.HSB, OwnershipType.RIKSBYGGEN],
            housing_programs=["Funkis"],
            notable_developers=["HSB", "Riksbyggen"],
            notable_architects=["Sven Markelius", "Gunnar Asplund", "Sigurd Lewerentz"],
            typical_certifications=[EnergyCertification.ENERGY_CLASS_E, EnergyCertification.ENERGY_CLASS_F],
            keywords_sv=["funkis", "funktionalism", "1930-tal", "1940-tal", "ljusa fasader",
                        "platta tak", "bandade fönster", "lamellhus"],
            keywords_en=["functionalism", "bauhaus", "modernist", "flat roof", "ribbon windows"],
            infiltration_variability="medium",
            u_value_variability="medium",
            occupancy_pattern="residential",
            likely_renovated_if=["fasadrenovering", "fönsterbyte", "FTX installerat"],
            renovation_era_signs={"1970s": "tilläggsisolering", "1990s": "FTX", "2010s": "energirenovering"},
        ),
    ),

    # =========================================================================
    # 1946-1960: Folkhemmet
    # =========================================================================
    "mfh_1946_1960": DetailedArchetype(
        id="mfh_1946_1960",
        name_sv="Flerbostadshus 1946-1960 (Folkhemmet)",
        name_en="Multi-family 1946-1960 (People's Home)",
        era=BuildingEra.FOLKHEM_1946_1960,
        year_start=1946,
        year_end=1960,

        stock_share_percent=12.5,
        typical_atemp_m2=(1000, 3500),
        typical_floors=(3, 5),

        wall_constructions=[
            WallConstruction(
                type=WallConstructionType.LIGHT_CONCRETE_BLOCK,
                name_sv="Lättbetongblock med puts",
                name_en="Aerated concrete blocks rendered",
                total_thickness_mm=300,
                insulation_thickness_mm=200,  # Lättbetong acts as insulation
                insulation_type="lättbetong",
                u_value=0.65,
                thermal_bridge_factor=1.10,
                description="Common Folkhemmet construction"
            ),
            WallConstruction(
                type=WallConstructionType.CAVITY_BRICK,
                name_sv="Hålmur med mineralull",
                name_en="Cavity wall with mineral wool",
                total_thickness_mm=350,
                insulation_thickness_mm=75,
                insulation_type="mineralull",
                u_value=0.55,
                thermal_bridge_factor=1.10,
            ),
        ],

        roof_construction=RoofConstruction(
            name_sv="Kallvind med mineralull",
            insulation_thickness_mm=150,
            insulation_type="mineralull",
            u_value=0.25,
            roof_type="cold_attic"
        ),

        floor_construction=FloorConstruction(
            name_sv="Betongplatta på mark",
            type="slab_on_grade",
            insulation_thickness_mm=50,
            u_value=0.45
        ),

        window_construction=WindowConstruction(
            type=WindowType.COUPLED_2_PANE,
            name_sv="Kopplade tvåglasfönster",
            name_en="Coupled double windows",
            u_value_glass=2.8,
            u_value_installed=2.6,
            shgc=0.70,
            num_panes=2,
            gas_fill="air"
        ),
        typical_wwr=0.18,
        wwr_by_orientation={"N": 0.14, "S": 0.20, "E": 0.16, "W": 0.16},

        infiltration_ach=0.20,
        n50_ach=6.0,

        ventilation_type=VentilationType.NATURAL,
        ventilation_rate_l_s_m2=0.35,
        heat_recovery_efficiency=0.0,
        sfp_kw_per_m3s=0.0,

        heating_systems={
            HeatingSystemType.DISTRICT_BIOMASS: 0.75,
            HeatingSystemType.OIL_BOILER: 0.15,
            HeatingSystemType.ELECTRIC_WATERBORNE: 0.10,
        },
        typical_heating_kwh_m2=145,

        dhw_kwh_m2=25,

        occupancy_w_per_m2=2.7,
        lighting_w_m2=8,
        equipment_w_m2=8,

        typical_forms=["lamellhus", "stjarnhus", "punkthus"],
        typical_facades=["puts", "tegel"],

        common_issues=[
            "Köldbryggor vid balkonger",
            "Bristfällig isolering",
            "Självdragsventilation",
            "Gamla rörinstallationer",
        ],
        renovation_potential_kwh_m2=60,
        typical_ecms=[
            "window_replacement",
            "roof_insulation",
            "ftx_installation",
            "air_sealing",
        ],

        description="Standardized post-war construction, introduction of mineral wool",
        sources=["TABULA SE", "BETSI 2010", "Energimyndigheten"],
        descriptors=ArchetypeDescriptors(
            building_depth_m=(10.0, 12.0),
            floor_to_floor_m=(2.7, 3.0),
            building_length_m=(30.0, 100.0),
            plan_shape=[PlanShape.RECTANGULAR, PlanShape.STAR, PlanShape.T_SHAPE],
            stairwell_apartments=(2, 4),
            balcony_types=[BalconyType.PROJECTING, BalconyType.RECESSED],
            roof_profiles=[RoofProfile.FLAT, RoofProfile.LOW_PITCHED],
            facade_patterns=[FacadePattern.REGULAR_PUNCHED, FacadePattern.HORIZONTAL_BANDS],
            typical_colors=["ljusgul", "vit", "ljusgrå", "rosa"],
            window_proportions="square",
            has_bay_windows=False,
            has_corner_windows=True,
            urban_settings=[UrbanSetting.INNER_SUBURB, UrbanSetting.OUTER_SUBURB],
            typical_neighborhoods=["Vällingby", "Farsta", "Årsta", "Högsbo", "Rosengård"],
            typical_cities=["Stockholm", "Göteborg", "Malmö", "Örebro", "Västerås"],
            original_ownership=[OwnershipType.HSB, OwnershipType.RIKSBYGGEN, OwnershipType.MUNICIPAL],
            housing_programs=["Folkhemmet"],
            notable_developers=["HSB", "Riksbyggen", "Svenska Bostäder"],
            notable_architects=[],
            typical_certifications=[EnergyCertification.ENERGY_CLASS_E, EnergyCertification.ENERGY_CLASS_F],
            keywords_sv=["folkhemmet", "1950-tal", "efterkrigstid", "lamellhus", "stjärnhus",
                        "lättbetong", "punkthus"],
            keywords_en=["post-war", "people's home", "welfare state", "slab block", "tower block"],
            infiltration_variability="medium",
            u_value_variability="low",
            occupancy_pattern="residential",
            likely_renovated_if=["stambyte", "fasadrenovering", "fönsterbyte"],
            renovation_era_signs={"1980s": "fönsterbyte", "2000s": "stambyte", "2010s": "FTX"},
        ),
    ),

    # =========================================================================
    # 1961-1975: Miljonprogrammet
    # =========================================================================
    "mfh_1961_1975": DetailedArchetype(
        id="mfh_1961_1975",
        name_sv="Flerbostadshus 1961-1975 (Miljonprogrammet)",
        name_en="Multi-family 1961-1975 (Million Programme)",
        era=BuildingEra.REKORD_1961_1975,
        year_start=1961,
        year_end=1975,

        stock_share_percent=25.0,  # Largest share!
        typical_atemp_m2=(1500, 8000),
        typical_floors=(3, 12),

        wall_constructions=[
            WallConstruction(
                type=WallConstructionType.CONCRETE_SANDWICH,
                name_sv="Betongsandwichelement",
                name_en="Concrete sandwich panel",
                total_thickness_mm=280,
                insulation_thickness_mm=100,
                insulation_type="mineralull/cellplast",
                u_value=0.45,
                thermal_bridge_factor=1.15,
                description="Prefab concrete sandwich elements, characteristic of Miljonprogrammet"
            ),
            WallConstruction(
                type=WallConstructionType.LIGHT_CONCRETE_BLOCK,
                name_sv="Lättbetongblock med puts",
                name_en="Aerated concrete rendered",
                total_thickness_mm=300,
                insulation_thickness_mm=200,
                insulation_type="lättbetong",
                u_value=0.55,
                thermal_bridge_factor=1.10,
            ),
        ],

        roof_construction=RoofConstruction(
            name_sv="Platt tak med cellplast",
            insulation_thickness_mm=150,
            insulation_type="cellplast",
            u_value=0.22,
            roof_type="flat"
        ),

        floor_construction=FloorConstruction(
            name_sv="Betongplatta på mark",
            type="slab_on_grade",
            insulation_thickness_mm=80,
            u_value=0.35
        ),

        window_construction=WindowConstruction(
            type=WindowType.SEALED_2_PANE,
            name_sv="Förseglade tvåglasfönster",
            name_en="Sealed double-glazed units",
            u_value_glass=2.5,
            u_value_installed=2.3,
            shgc=0.65,
            num_panes=2,
            gas_fill="air"
        ),
        typical_wwr=0.22,
        wwr_by_orientation={"N": 0.18, "S": 0.25, "E": 0.20, "W": 0.20},

        infiltration_ach=0.15,
        n50_ach=5.0,

        ventilation_type=VentilationType.EXHAUST,  # F-system introduced
        ventilation_rate_l_s_m2=0.35,
        heat_recovery_efficiency=0.0,
        sfp_kw_per_m3s=1.0,

        heating_systems={
            HeatingSystemType.DISTRICT_BIOMASS: 0.85,
            HeatingSystemType.OIL_BOILER: 0.10,
            HeatingSystemType.ELECTRIC_WATERBORNE: 0.05,
        },
        typical_heating_kwh_m2=135,

        dhw_kwh_m2=25,

        occupancy_w_per_m2=2.7,
        lighting_w_m2=10,
        equipment_w_m2=10,

        typical_forms=["skivhus", "lamellhus", "loftgangshus", "punkthus"],
        typical_facades=["betong", "puts", "tegel"],

        common_issues=[
            "Kraftiga köldbryggor vid balkonginfästningar",
            "Fasadskador på betongelement",
            "F-ventilation utan värmeåtervinning",
            "Kulvertsystem med värmeförluster",
            "Stammar och rör i behov av renovering",
            "Dålig ljudisolering",
        ],
        renovation_potential_kwh_m2=55,
        typical_ecms=[
            "wall_external_insulation",
            "ftx_installation",
            "window_replacement",
            "roof_insulation",
            "air_sealing",
        ],

        description="Industrialized prefab construction, concrete sandwich panels dominant",
        sources=["TABULA SE", "BETSI 2010", "NCC miljonprogramrapport", "MDPI 2019"],
        descriptors=ArchetypeDescriptors(
            building_depth_m=(11.0, 13.0),
            floor_to_floor_m=(2.6, 2.8),
            building_length_m=(50.0, 150.0),
            plan_shape=[PlanShape.RECTANGULAR, PlanShape.L_SHAPE, PlanShape.H_SHAPE],
            stairwell_apartments=(2, 4),
            balcony_types=[BalconyType.PROJECTING, BalconyType.LOGGIA],
            roof_profiles=[RoofProfile.FLAT],
            facade_patterns=[FacadePattern.GRID_UNIFORM, FacadePattern.HORIZONTAL_BANDS],
            typical_colors=["grå", "vit", "beige", "gul", "brun"],
            window_proportions="landscape",
            has_bay_windows=False,
            has_corner_windows=False,
            urban_settings=[UrbanSetting.OUTER_SUBURB, UrbanSetting.SATELLITE_TOWN],
            typical_neighborhoods=["Rinkeby", "Tensta", "Rosengård", "Hammarkullen", "Bergsjön",
                                  "Skärholmen", "Fittja", "Jordbro", "Norsborg"],
            typical_cities=["Stockholm", "Göteborg", "Malmö", "Uppsala", "Linköping", "Västerås"],
            original_ownership=[OwnershipType.MUNICIPAL, OwnershipType.HSB, OwnershipType.RIKSBYGGEN],
            housing_programs=["Miljonprogrammet"],
            notable_developers=["Svenska Bostäder", "Stockholmshem", "Familjebostäder", "Poseidon"],
            notable_architects=[],
            typical_certifications=[EnergyCertification.ENERGY_CLASS_D, EnergyCertification.ENERGY_CLASS_E],
            keywords_sv=["miljonprogrammet", "1960-tal", "1970-tal", "betonghus", "skivhus",
                        "loftgångshus", "elementhus", "prefab", "förort"],
            keywords_en=["million programme", "prefab", "concrete", "slab block", "high-rise",
                        "suburban", "brutalist"],
            infiltration_variability="low",
            u_value_variability="low",
            occupancy_pattern="residential",
            likely_renovated_if=["fasadrenovering", "tilläggsisolering", "FTX installerat", "stambytt"],
            renovation_era_signs={"1990s": "fasadrenovering", "2000s": "stamrenovering",
                                 "2010s": "FTX + tilläggsisolering"},
        ),
    ),

    # =========================================================================
    # 1976-1985: Post oil crisis
    # =========================================================================
    "mfh_1976_1985": DetailedArchetype(
        id="mfh_1976_1985",
        name_sv="Flerbostadshus 1976-1985 (Efter oljekrisen)",
        name_en="Multi-family 1976-1985 (Post oil crisis)",
        era=BuildingEra.ENERGI_1976_1985,
        year_start=1976,
        year_end=1985,

        stock_share_percent=10.0,
        typical_atemp_m2=(1200, 4000),
        typical_floors=(2, 6),

        wall_constructions=[
            WallConstruction(
                type=WallConstructionType.CONCRETE_SANDWICH,
                name_sv="Betongsandwich med förbättrad isolering",
                name_en="Improved concrete sandwich",
                total_thickness_mm=320,
                insulation_thickness_mm=150,
                insulation_type="mineralull",
                u_value=0.28,
                thermal_bridge_factor=1.10,
                description="SBN 1975 requirements, better insulation"
            ),
            WallConstruction(
                type=WallConstructionType.STUD_FRAME_MINERAL,
                name_sv="Regelstomme med mineralull",
                name_en="Stud frame with mineral wool",
                total_thickness_mm=200,
                insulation_thickness_mm=170,
                insulation_type="mineralull",
                u_value=0.25,
                thermal_bridge_factor=1.05,
            ),
        ],

        roof_construction=RoofConstruction(
            name_sv="Isolerat vindsbjälklag",
            insulation_thickness_mm=250,
            insulation_type="mineralull",
            u_value=0.15,
            roof_type="cold_attic"
        ),

        floor_construction=FloorConstruction(
            name_sv="Isolerad platta på mark",
            type="slab_on_grade",
            insulation_thickness_mm=100,
            u_value=0.28
        ),

        window_construction=WindowConstruction(
            type=WindowType.TRIPLE_PANE,
            name_sv="Treglasfönster",
            name_en="Triple-glazed windows",
            u_value_glass=1.9,
            u_value_installed=1.8,
            shgc=0.60,
            num_panes=3,
            gas_fill="air"
        ),
        typical_wwr=0.20,
        wwr_by_orientation={"N": 0.15, "S": 0.22, "E": 0.18, "W": 0.18},

        infiltration_ach=0.10,
        n50_ach=3.5,

        ventilation_type=VentilationType.EXHAUST,
        ventilation_rate_l_s_m2=0.35,
        heat_recovery_efficiency=0.0,
        sfp_kw_per_m3s=1.2,

        heating_systems={
            HeatingSystemType.DISTRICT_BIOMASS: 0.80,
            HeatingSystemType.ELECTRIC_WATERBORNE: 0.12,
            HeatingSystemType.HEAT_PUMP_EXHAUST: 0.08,
        },
        typical_heating_kwh_m2=105,

        dhw_kwh_m2=25,

        occupancy_w_per_m2=2.7,
        lighting_w_m2=10,
        equipment_w_m2=10,

        typical_forms=["lamellhus", "radhus", "vinkelbyggnad"],
        typical_facades=["tegel", "puts", "trä"],

        common_issues=[
            "F-ventilation utan värmeåtervinning",
            "Vissa köldbryggor kvarstår",
            "Balkonginfästningar",
        ],
        renovation_potential_kwh_m2=35,
        typical_ecms=[
            "ftx_installation",
            "air_sealing",
            "window_upgrade",
            "demand_controlled_ventilation",
        ],

        description="Improved insulation after oil crisis, SBN 1975/1980 standards",
        sources=["TABULA SE", "BETSI 2010", "SBN 1975"],
        descriptors=ArchetypeDescriptors(
            building_depth_m=(11.0, 14.0),
            floor_to_floor_m=(2.6, 2.8),
            building_length_m=(30.0, 80.0),
            plan_shape=[PlanShape.RECTANGULAR, PlanShape.L_SHAPE],
            stairwell_apartments=(2, 4),
            balcony_types=[BalconyType.PROJECTING, BalconyType.LOGGIA, BalconyType.RECESSED],
            roof_profiles=[RoofProfile.LOW_PITCHED, RoofProfile.FLAT],
            facade_patterns=[FacadePattern.REGULAR_PUNCHED],
            typical_colors=["tegel", "brun", "röd", "vit", "grå"],
            window_proportions="square",
            has_bay_windows=False,
            has_corner_windows=False,
            urban_settings=[UrbanSetting.INNER_SUBURB, UrbanSetting.OUTER_SUBURB],
            typical_neighborhoods=["Norra Ängby", "Bromsten", "Bagarmossen", "Stureby"],
            typical_cities=["Stockholm", "Göteborg", "Malmö", "Uppsala"],
            original_ownership=[OwnershipType.HSB, OwnershipType.RIKSBYGGEN, OwnershipType.BRF],
            housing_programs=[],
            notable_developers=["JM", "NCC", "PEAB", "Skanska"],
            notable_architects=[],
            typical_certifications=[EnergyCertification.ENERGY_CLASS_D],
            keywords_sv=["1980-tal", "efter oljekrisen", "bättre isolering", "SBN 75", "SBN 80",
                        "tegel", "lamellhus"],
            keywords_en=["post-oil crisis", "1980s", "brick", "improved insulation"],
            infiltration_variability="low",
            u_value_variability="low",
            occupancy_pattern="residential",
            likely_renovated_if=["FTX installerat", "balkongglasning"],
            renovation_era_signs={"2010s": "FTX", "2020s": "solceller"},
        ),
    ),

    # =========================================================================
    # 1986-1995: Modern well-insulated
    # =========================================================================
    "mfh_1986_1995": DetailedArchetype(
        id="mfh_1986_1995",
        name_sv="Flerbostadshus 1986-1995",
        name_en="Multi-family 1986-1995",
        era=BuildingEra.MODERN_1986_1995,
        year_start=1986,
        year_end=1995,

        stock_share_percent=8.0,
        typical_atemp_m2=(1000, 3000),
        typical_floors=(2, 5),

        wall_constructions=[
            WallConstruction(
                type=WallConstructionType.STUD_FRAME_MINERAL,
                name_sv="Regelstomme 195mm + tilläggsisolering",
                name_en="195mm stud + additional insulation",
                total_thickness_mm=250,
                insulation_thickness_mm=220,
                insulation_type="mineralull",
                u_value=0.20,
                thermal_bridge_factor=1.05,
            ),
            WallConstruction(
                type=WallConstructionType.PREFAB_ELEMENT,
                name_sv="Prefab element med bättre isolering",
                name_en="Prefab panels improved",
                total_thickness_mm=280,
                insulation_thickness_mm=180,
                insulation_type="mineralull",
                u_value=0.22,
                thermal_bridge_factor=1.05,
            ),
        ],

        roof_construction=RoofConstruction(
            name_sv="Välisolerat vindsbjälklag",
            insulation_thickness_mm=350,
            insulation_type="mineralull/lösull",
            u_value=0.12,
            roof_type="cold_attic"
        ),

        floor_construction=FloorConstruction(
            name_sv="Välisolerad platta på mark",
            type="slab_on_grade",
            insulation_thickness_mm=150,
            u_value=0.22
        ),

        window_construction=WindowConstruction(
            type=WindowType.TRIPLE_PANE,
            name_sv="Treglasfönster med lågemissionsglas",
            name_en="Triple-glazed with low-e",
            u_value_glass=1.5,
            u_value_installed=1.4,
            shgc=0.55,
            num_panes=3,
            gas_fill="air",
            coating="low-e"
        ),
        typical_wwr=0.18,
        wwr_by_orientation={"N": 0.14, "S": 0.20, "E": 0.16, "W": 0.16},

        infiltration_ach=0.08,
        n50_ach=2.5,

        ventilation_type=VentilationType.HEAT_RECOVERY,
        ventilation_rate_l_s_m2=0.35,
        heat_recovery_efficiency=0.70,
        sfp_kw_per_m3s=1.5,

        heating_systems={
            HeatingSystemType.DISTRICT_BIOMASS: 0.75,
            HeatingSystemType.HEAT_PUMP_EXHAUST: 0.15,
            HeatingSystemType.ELECTRIC_WATERBORNE: 0.10,
        },
        typical_heating_kwh_m2=85,

        dhw_kwh_m2=25,

        occupancy_w_per_m2=2.7,
        lighting_w_m2=8,
        equipment_w_m2=10,

        typical_forms=["lamellhus", "punkthus", "radhus"],
        typical_facades=["tegel", "puts", "trä"],

        common_issues=[
            "FTX-aggregat åldrande",
            "OVK-krav ej alltid uppfyllda",
        ],
        renovation_potential_kwh_m2=25,
        typical_ecms=[
            "ftx_upgrade",
            "led_lighting",
            "solar_pv",
            "demand_controlled_ventilation",
        ],

        description="FTX becoming standard, good insulation, OVK system introduced",
        sources=["TABULA SE", "BETSI 2010"],
        descriptors=ArchetypeDescriptors(
            building_depth_m=(11.0, 14.0),
            floor_to_floor_m=(2.6, 2.9),
            building_length_m=(25.0, 60.0),
            plan_shape=[PlanShape.RECTANGULAR, PlanShape.L_SHAPE],
            stairwell_apartments=(2, 4),
            balcony_types=[BalconyType.PROJECTING, BalconyType.RECESSED, BalconyType.FRENCH],
            roof_profiles=[RoofProfile.LOW_PITCHED, RoofProfile.FLAT, RoofProfile.PITCHED],
            facade_patterns=[FacadePattern.REGULAR_PUNCHED, FacadePattern.MIXED],
            typical_colors=["tegel", "vit", "ljusgrå", "gul"],
            window_proportions="square",
            has_bay_windows=False,
            has_corner_windows=False,
            urban_settings=[UrbanSetting.INNER_SUBURB, UrbanSetting.OUTER_SUBURB, UrbanSetting.SMALL_TOWN],
            typical_neighborhoods=["Rissne", "Segeltorp", "Eriksberg"],
            typical_cities=["Stockholm", "Göteborg", "Malmö", "Uppsala", "Örebro"],
            original_ownership=[OwnershipType.BRF, OwnershipType.HSB, OwnershipType.RIKSBYGGEN],
            housing_programs=[],
            notable_developers=["JM", "NCC", "Skanska", "PEAB"],
            notable_architects=[],
            typical_certifications=[EnergyCertification.ENERGY_CLASS_C, EnergyCertification.ENERGY_CLASS_D],
            keywords_sv=["1990-tal", "FTX", "OVK", "energieffektiv"],
            keywords_en=["1990s", "heat recovery", "modern construction"],
            infiltration_variability="low",
            u_value_variability="low",
            occupancy_pattern="residential",
            likely_renovated_if=["FTX uppgraderat", "solceller installerade"],
            renovation_era_signs={"2020s": "solceller"},
        ),
    ),

    # =========================================================================
    # 1996-2010: Low-energy transition
    # =========================================================================
    "mfh_1996_2010": DetailedArchetype(
        id="mfh_1996_2010",
        name_sv="Flerbostadshus 1996-2010",
        name_en="Multi-family 1996-2010",
        era=BuildingEra.LAGENERGI_1996_2010,
        year_start=1996,
        year_end=2010,

        stock_share_percent=6.0,
        typical_atemp_m2=(800, 3500),
        typical_floors=(2, 8),

        wall_constructions=[
            WallConstruction(
                type=WallConstructionType.STUD_FRAME_MINERAL,
                name_sv="Regelstomme 195mm + utvändig isolering",
                name_en="195mm stud + external insulation",
                total_thickness_mm=300,
                insulation_thickness_mm=270,
                insulation_type="mineralull",
                u_value=0.17,
                thermal_bridge_factor=1.03,
            ),
        ],

        roof_construction=RoofConstruction(
            name_sv="Välisolerat tak",
            insulation_thickness_mm=400,
            insulation_type="lösull",
            u_value=0.10,
            roof_type="cold_attic"
        ),

        floor_construction=FloorConstruction(
            name_sv="Isolerad platta",
            type="slab_on_grade",
            insulation_thickness_mm=200,
            u_value=0.18
        ),

        window_construction=WindowConstruction(
            type=WindowType.LOW_E_TRIPLE,
            name_sv="Energifönster 3-glas",
            name_en="Energy-rated triple glazed",
            u_value_glass=1.1,
            u_value_installed=1.2,
            shgc=0.50,
            num_panes=3,
            gas_fill="argon",
            coating="low-e"
        ),
        typical_wwr=0.20,
        wwr_by_orientation={"N": 0.15, "S": 0.25, "E": 0.18, "W": 0.18},

        infiltration_ach=0.06,
        n50_ach=1.5,

        ventilation_type=VentilationType.HEAT_RECOVERY,
        ventilation_rate_l_s_m2=0.35,
        heat_recovery_efficiency=0.75,
        sfp_kw_per_m3s=1.5,

        heating_systems={
            HeatingSystemType.DISTRICT_BIOMASS: 0.70,
            HeatingSystemType.HEAT_PUMP_GROUND: 0.15,
            HeatingSystemType.HEAT_PUMP_EXHAUST: 0.10,
            HeatingSystemType.ELECTRIC_WATERBORNE: 0.05,
        },
        typical_heating_kwh_m2=65,

        dhw_kwh_m2=25,

        occupancy_w_per_m2=2.7,
        lighting_w_m2=8,
        equipment_w_m2=12,

        typical_forms=["lamellhus", "punkthus", "generic"],
        typical_facades=["puts", "tegel", "glas"],

        common_issues=[
            "God baseline, begränsad renoveringspotential",
        ],
        renovation_potential_kwh_m2=15,
        typical_ecms=[
            "solar_pv",
            "demand_controlled_ventilation",
            "led_lighting",
        ],

        description="Modern BBR standards, efficient systems",
        sources=["TABULA SE", "BBR"],
        descriptors=ArchetypeDescriptors(
            building_depth_m=(12.0, 15.0),
            floor_to_floor_m=(2.7, 3.0),
            building_length_m=(20.0, 60.0),
            plan_shape=[PlanShape.RECTANGULAR, PlanShape.L_SHAPE, PlanShape.U_SHAPE],
            stairwell_apartments=(2, 6),
            balcony_types=[BalconyType.PROJECTING, BalconyType.RECESSED, BalconyType.GLAZED],
            roof_profiles=[RoofProfile.FLAT, RoofProfile.LOW_PITCHED],
            facade_patterns=[FacadePattern.REGULAR_PUNCHED, FacadePattern.LARGE_GLAZING, FacadePattern.MIXED],
            typical_colors=["vit", "grå", "tegel", "trä", "svart"],
            window_proportions="square",
            has_bay_windows=False,
            has_corner_windows=True,
            urban_settings=[UrbanSetting.INNER_CITY, UrbanSetting.INNER_SUBURB, UrbanSetting.WATERFRONT],
            typical_neighborhoods=["Hammarby Sjöstad", "Norra Djurgårdsstaden", "Västra Hamnen", "Lindholmen"],
            typical_cities=["Stockholm", "Göteborg", "Malmö", "Uppsala", "Lund"],
            original_ownership=[OwnershipType.BRF, OwnershipType.PRIVATE_RENTAL],
            housing_programs=[],
            notable_developers=["JM", "Skanska", "NCC", "Bonava", "HSB", "Riksbyggen"],
            notable_architects=["Tham & Videgård", "Wingårdhs"],
            typical_certifications=[EnergyCertification.ENERGY_CLASS_B, EnergyCertification.ENERGY_CLASS_C,
                                   EnergyCertification.GREEN_BUILDING],
            keywords_sv=["2000-tal", "modern", "lågenergihus", "energieffektiv", "BBR",
                        "hammarby sjöstad", "sjöstaden"],
            keywords_en=["2000s", "modern", "low energy", "energy efficient", "BBR compliant"],
            infiltration_variability="low",
            u_value_variability="low",
            occupancy_pattern="residential",
            likely_renovated_if=[],
            renovation_era_signs={},
        ),
    ),

    # =========================================================================
    # 2011+: Near-zero energy
    # =========================================================================
    "mfh_2011_plus": DetailedArchetype(
        id="mfh_2011_plus",
        name_sv="Flerbostadshus 2011+ (Lågenergihus)",
        name_en="Multi-family 2011+ (Low-energy)",
        era=BuildingEra.NARA_NOLL_2011_PLUS,
        year_start=2011,
        year_end=2030,

        stock_share_percent=4.0,
        typical_atemp_m2=(500, 4000),
        typical_floors=(2, 10),

        wall_constructions=[
            WallConstruction(
                type=WallConstructionType.STUD_FRAME_MINERAL,
                name_sv="Tjock regelstomme med köldbryggebrytning",
                name_en="Deep stud frame with thermal break",
                total_thickness_mm=350,
                insulation_thickness_mm=320,
                insulation_type="mineralull",
                u_value=0.12,
                thermal_bridge_factor=1.02,
            ),
            WallConstruction(
                type=WallConstructionType.CLT,
                name_sv="CLT med utvändig isolering",
                name_en="CLT with external insulation",
                total_thickness_mm=400,
                insulation_thickness_mm=250,
                insulation_type="mineralull",
                u_value=0.11,
                thermal_bridge_factor=1.02,
            ),
        ],

        roof_construction=RoofConstruction(
            name_sv="Passivinspirerat tak",
            insulation_thickness_mm=500,
            insulation_type="lösull",
            u_value=0.08,
            roof_type="cold_attic"
        ),

        floor_construction=FloorConstruction(
            name_sv="Högsisolerad platta",
            type="slab_on_grade",
            insulation_thickness_mm=300,
            u_value=0.12
        ),

        window_construction=WindowConstruction(
            type=WindowType.PASSIVE_HOUSE,
            name_sv="Passivhusfönster",
            name_en="Passive house windows",
            u_value_glass=0.6,
            u_value_installed=0.9,
            shgc=0.45,
            num_panes=3,
            gas_fill="argon",
            coating="triple-low-e"
        ),
        typical_wwr=0.22,
        wwr_by_orientation={"N": 0.15, "S": 0.30, "E": 0.18, "W": 0.18},

        infiltration_ach=0.04,
        n50_ach=0.8,

        ventilation_type=VentilationType.HEAT_RECOVERY,
        ventilation_rate_l_s_m2=0.35,
        heat_recovery_efficiency=0.85,
        sfp_kw_per_m3s=1.5,

        heating_systems={
            HeatingSystemType.DISTRICT_BIOMASS: 0.60,
            HeatingSystemType.HEAT_PUMP_GROUND: 0.25,
            HeatingSystemType.HEAT_PUMP_EXHAUST: 0.10,
            HeatingSystemType.ELECTRIC_WATERBORNE: 0.05,
        },
        typical_heating_kwh_m2=40,

        dhw_kwh_m2=20,  # Heat pump DHW

        occupancy_w_per_m2=2.7,
        lighting_w_m2=6,  # LED standard
        equipment_w_m2=12,

        typical_forms=["lamellhus", "punkthus", "generic"],
        typical_facades=["puts", "trä", "glas"],

        common_issues=[
            "Mycket effektiv, minimal renoveringspotential",
        ],
        renovation_potential_kwh_m2=5,
        typical_ecms=[
            "solar_pv",
            "battery_storage",
        ],

        description="BBR 2011+ requirements, approaching passive house levels",
        sources=["BBR", "FEBY"],
        descriptors=ArchetypeDescriptors(
            building_depth_m=(12.0, 16.0),
            floor_to_floor_m=(2.7, 3.1),
            building_length_m=(20.0, 80.0),
            plan_shape=[PlanShape.RECTANGULAR, PlanShape.L_SHAPE, PlanShape.U_SHAPE],
            stairwell_apartments=(2, 6),
            balcony_types=[BalconyType.GLAZED, BalconyType.PROJECTING, BalconyType.RECESSED],
            roof_profiles=[RoofProfile.FLAT, RoofProfile.GREEN, RoofProfile.LOW_PITCHED],
            facade_patterns=[FacadePattern.LARGE_GLAZING, FacadePattern.MIXED, FacadePattern.CURTAIN_WALL],
            typical_colors=["vit", "grå", "svart", "trä", "corten"],
            window_proportions="square",
            has_bay_windows=False,
            has_corner_windows=True,
            urban_settings=[UrbanSetting.INNER_CITY, UrbanSetting.WATERFRONT, UrbanSetting.INNER_SUBURB],
            typical_neighborhoods=["Norra Djurgårdsstaden", "Hagastaden", "Barkarby",
                                  "Frihamnen", "Vallastaden"],
            typical_cities=["Stockholm", "Göteborg", "Malmö", "Uppsala", "Linköping"],
            original_ownership=[OwnershipType.BRF, OwnershipType.PRIVATE_RENTAL],
            housing_programs=[],
            notable_developers=["JM", "Skanska", "NCC", "Bonava", "HSB", "Riksbyggen", "Veidekke"],
            notable_architects=["White", "Wingårdhs", "Tham & Videgård", "Sandellsandberg"],
            typical_certifications=[EnergyCertification.ENERGY_CLASS_A, EnergyCertification.ENERGY_CLASS_B,
                                   EnergyCertification.SVANEN, EnergyCertification.MILJOBYGGNAD_SILVER,
                                   EnergyCertification.GREEN_BUILDING],
            keywords_sv=["2010-tal", "2020-tal", "lågenergihus", "nära-noll", "BBR", "passivhus",
                        "miljöbyggnad", "hållbart", "CLT", "trähus"],
            keywords_en=["2010s", "2020s", "low energy", "near-zero", "passive", "sustainable", "CLT", "timber"],
            infiltration_variability="low",
            u_value_variability="low",
            occupancy_pattern="residential",
            likely_renovated_if=[],
            renovation_era_signs={},
        ),
    ),
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_archetype_by_year(year: int, building_type: str = "mfh") -> DetailedArchetype:
    """
    Get the appropriate archetype for a given construction year.

    Args:
        year: Construction year
        building_type: Building type - "mfh" (multi-family), "sfh" (single-family),
                       "terraced" (radhus/kedjehus/parhus)

    Returns:
        Matching DetailedArchetype
    """
    building_type_lower = building_type.lower()

    # Select appropriate archetype dictionary
    if building_type_lower in ("sfh", "single_family", "villa", "småhus", "smahus"):
        archetypes = SWEDISH_SFH_ARCHETYPES
        default_key = "sfh_2011_plus"
    elif building_type_lower in ("terraced", "radhus", "kedjehus", "parhus", "row_house"):
        archetypes = SWEDISH_TERRACED_ARCHETYPES
        default_key = "terraced_2011_plus"
    else:
        archetypes = SWEDISH_MFH_ARCHETYPES
        default_key = "mfh_2011_plus"

    for archetype in archetypes.values():
        if archetype.year_start <= year <= archetype.year_end:
            return archetype

    # Default to newest if year is beyond range
    return archetypes[default_key]


def get_smart_archetype(
    construction_year: int,
    num_apartments: Optional[int] = None,
    num_floors: Optional[int] = None,
    building_form: Optional[str] = None,
    facade_material: Optional[str] = None,
    city: Optional[str] = None,
    atemp_m2: Optional[float] = None,
    is_heritage: bool = False,
    # New parameters for high-performance detection
    energy_class: Optional[str] = None,
    certification: Optional[str] = None,
    declared_energy_kwh_m2: Optional[float] = None,
    has_solar: bool = False,
    has_ftx: bool = False,
    keywords: Optional[List[str]] = None,
) -> Tuple[DetailedArchetype, str, float]:
    """
    Smart archetype selection using all available building data.

    This is the recommended function for the pipeline to use. It considers
    multiple inputs to find the best matching archetype.

    Args:
        construction_year: Year of construction (required)
        num_apartments: Number of apartments (helps distinguish MFH vs SFH)
        num_floors: Number of floors (helps identify building type)
        building_form: Detected form (lamellhus, skivhus, punkthus, villa, etc.)
        facade_material: Facade material (brick, concrete, wood, etc.)
        city: City name (for regional archetypes like landshövdingehus)
        atemp_m2: Heated floor area in m²
        is_heritage: Whether building is heritage protected
        energy_class: Energy performance class (A-G)
        certification: Building certification (miljöbyggnad_gold, feby, passive, plus_energy)
        declared_energy_kwh_m2: Declared energy use from energy declaration
        has_solar: Whether building has solar PV
        has_ftx: Whether building has FTX ventilation
        keywords: List of keywords from property listing or records

    Returns:
        Tuple of (archetype, selection_reason, confidence_score)
        - archetype: The selected DetailedArchetype
        - selection_reason: Human-readable explanation of selection
        - confidence_score: 0.0-1.0 indicating match confidence
    """
    reasons = []
    confidence = 0.5  # Base confidence

    # =================================================================
    # 0. CHECK FOR HIGH-PERFORMANCE BUILDINGS FIRST
    # =================================================================
    # These take priority as they have very specific characteristics

    # Plus-energy detection
    if certification and "plus" in certification.lower():
        selected = SWEDISH_HIGH_PERFORMANCE_ARCHETYPES.get("plus_energy_mfh_2015_plus")
        if selected:
            return selected, "Certification indicates plus-energy building", 0.95

    # Passive house detection
    if certification and ("passiv" in certification.lower() or "feby" in certification.lower()):
        selected = SWEDISH_HIGH_PERFORMANCE_ARCHETYPES.get("passive_house_mfh_2009_plus")
        if selected:
            return selected, "Certification indicates FEBY passive house", 0.90

    # Miljöbyggnad Gold detection
    if certification and "guld" in certification.lower():
        selected = SWEDISH_HIGH_PERFORMANCE_ARCHETYPES.get("miljobyggnad_gold_mfh")
        if selected:
            return selected, "Miljöbyggnad Gold certified", 0.85

    # Very low declared energy suggests high-performance
    if declared_energy_kwh_m2 is not None and construction_year >= 2009:
        if declared_energy_kwh_m2 <= 20:
            selected = SWEDISH_HIGH_PERFORMANCE_ARCHETYPES.get("passive_house_mfh_2009_plus")
            if selected:
                return selected, f"Very low energy ({declared_energy_kwh_m2} kWh/m²) → passive house", 0.80
        elif declared_energy_kwh_m2 <= 35:
            selected = SWEDISH_HIGH_PERFORMANCE_ARCHETYPES.get("miljobyggnad_gold_mfh")
            if selected:
                return selected, f"Low energy ({declared_energy_kwh_m2} kWh/m²) → high-performance", 0.70

    # Energy class A with modern construction
    if energy_class and energy_class.upper() == "A" and construction_year >= 2012:
        selected = SWEDISH_HIGH_PERFORMANCE_ARCHETYPES.get("miljobyggnad_gold_mfh")
        if selected:
            reasons.append("Energy class A + modern → high-performance")
            confidence = 0.75

    # Keyword matching for high-performance
    if keywords:
        kw_lower = [k.lower() for k in keywords]
        if any(k in kw_lower for k in ["plusenergi", "plus energy", "nollenergihus"]):
            selected = SWEDISH_HIGH_PERFORMANCE_ARCHETYPES.get("plus_energy_mfh_2015_plus")
            if selected:
                return selected, "Keywords indicate plus-energy building", 0.85
        elif any(k in kw_lower for k in ["passivhus", "passive house", "feby"]):
            selected = SWEDISH_HIGH_PERFORMANCE_ARCHETYPES.get("passive_house_mfh_2009_plus")
            if selected:
                return selected, "Keywords indicate passive house", 0.85
        elif any(k in kw_lower for k in ["miljöbyggnad guld", "gold certified"]):
            selected = SWEDISH_HIGH_PERFORMANCE_ARCHETYPES.get("miljobyggnad_gold_mfh")
            if selected:
                return selected, "Keywords indicate Miljöbyggnad Gold", 0.80

    # Determine building category
    building_category = "mfh"  # Default

    # 1. Use num_apartments to distinguish SFH vs MFH
    if num_apartments is not None:
        if num_apartments == 1:
            building_category = "sfh"
            reasons.append(f"1 apartment → single-family")
            confidence += 0.2
        elif num_apartments <= 4 and num_floors and num_floors <= 2:
            # Could be terraced or small MFH
            if atemp_m2 and atemp_m2 < 200:
                building_category = "terraced"
                reasons.append(f"{num_apartments} apts, {num_floors} floors, small area → terraced")
                confidence += 0.15
            else:
                building_category = "mfh"
                reasons.append(f"{num_apartments} apartments → multi-family")
        else:
            building_category = "mfh"
            reasons.append(f"{num_apartments} apartments → multi-family")
            confidence += 0.1

    # 2. Use building_form for more specific matching
    if building_form:
        form_lower = building_form.lower()

        # Terraced house forms
        if form_lower in ("radhus", "kedjehus", "parhus", "row_house", "terraced"):
            building_category = "terraced"
            reasons.append(f"form={building_form} → terraced")
            confidence += 0.2

        # Single-family forms
        elif form_lower in ("villa", "villa_1_plan", "villa_1_5_plan", "villa_2_plan",
                            "egnahem", "fritidshus"):
            building_category = "sfh"
            reasons.append(f"form={building_form} → single-family")
            confidence += 0.2

        # Multi-family forms
        elif form_lower in ("lamellhus", "skivhus", "punkthus", "loftgangshus",
                            "stjarnhus", "slutet_kvarter"):
            building_category = "mfh"
            reasons.append(f"form={building_form} → multi-family")
            confidence += 0.2

    # 3. Check for specific historical and special form archetypes
    selected_archetype = None

    # Gothenburg landshövdingehus
    if city and ("göteborg" in city.lower() or "gothenburg" in city.lower()):
        if construction_year >= 1875 and construction_year <= 1945:
            if num_floors == 3:
                selected_archetype = SWEDISH_HISTORICAL_ARCHETYPES.get("hist_landshovdingehus")
                reasons.append("Gothenburg + 3 floors + era → landshövdingehus")
                confidence += 0.3

    # Old town / medieval
    if is_heritage and construction_year < 1700:
        selected_archetype = SWEDISH_HISTORICAL_ARCHETYPES.get("hist_medieval_pre1700")
        reasons.append("Heritage + pre-1700 → medieval")
        confidence += 0.2

    # =================================================================
    # SPECIAL FORM ARCHETYPES (Stockholm-specific and unique forms)
    # =================================================================

    # Punkthus (Tower Blocks) - 8+ floors, 1950-1970
    if selected_archetype is None and building_form:
        if building_form.lower() == "punkthus":
            if construction_year >= 1950 and construction_year <= 1970:
                selected_archetype = SWEDISH_SPECIAL_FORM_ARCHETYPES.get("punkthus_1950_1970")
                reasons.append("form=punkthus + era → punkthus archetype")
                confidence += 0.3
    if selected_archetype is None and num_floors and num_floors >= 8:
        if construction_year >= 1950 and construction_year <= 1970:
            if building_category == "mfh":
                selected_archetype = SWEDISH_SPECIAL_FORM_ARCHETYPES.get("punkthus_1950_1970")
                reasons.append("8+ floors + 1950-1970 → punkthus")
                confidence += 0.2

    # Stjärnhus (Star Houses) - unique Y-shape, 1944-1962, Stockholm
    if selected_archetype is None and building_form:
        if building_form.lower() in ("stjarnhus", "stjärnhus", "star_house"):
            selected_archetype = SWEDISH_SPECIAL_FORM_ARCHETYPES.get("stjarnhus_1944_1962")
            reasons.append("form=stjärnhus → star house archetype")
            confidence += 0.35
    if selected_archetype is None and city:
        city_lower = city.lower()
        # Gröndal and Västertorp are famous for stjärnhus
        if any(area in city_lower for area in ("gröndal", "västertorp", "årsta")):
            if construction_year >= 1944 and construction_year <= 1962:
                selected_archetype = SWEDISH_SPECIAL_FORM_ARCHETYPES.get("stjarnhus_1944_1962")
                reasons.append(f"{city} + era → likely stjärnhus")
                confidence += 0.25

    # CLT Multi-family (2015+) - timber construction
    if selected_archetype is None and facade_material:
        material_lower = facade_material.lower()
        if any(m in material_lower for m in ("clt", "kl-trä", "korslimmat", "massivträ", "cross-laminated")):
            if construction_year >= 2015:
                selected_archetype = SWEDISH_SPECIAL_FORM_ARCHETYPES.get("clt_multifamily_2015_plus")
                reasons.append("CLT/timber construction + 2015+ → CLT archetype")
                confidence += 0.35

    # Sustainable District (Hammarby/Stockholm Waterfront type) 1998-2016
    if selected_archetype is None and city:
        city_lower = city.lower()
        if any(area in city_lower for area in ("hammarby", "sjöstaden", "norra djurgårdsstaden",
                                                "royal seaport", "hagastaden")):
            if construction_year >= 1998 and construction_year <= 2020:
                selected_archetype = SWEDISH_SPECIAL_FORM_ARCHETYPES.get("sustainable_district_2000s")
                reasons.append(f"{city} sustainable district → Hammarby-type archetype")
                confidence += 0.3

    # Smalhus (Narrow Houses) - HSB specialty, 1935-1955, narrow depth
    if selected_archetype is None:
        if construction_year >= 1935 and construction_year <= 1955:
            if building_category == "mfh" and num_floors and num_floors in (3, 4):
                # Could be smalhus - check for HSB or Stockholm
                if city and "stockholm" in city.lower():
                    selected_archetype = SWEDISH_SPECIAL_FORM_ARCHETYPES.get("smalhus_1935_1955")
                    reasons.append("Stockholm + 3-4 floors + 1935-1955 → likely smalhus")
                    confidence += 0.15

    # Barnrikehus (Large Family Housing) - 1935-1948, subsidized social housing
    # Hard to detect without specific markers - this is a fallback for the era
    if selected_archetype is None:
        if construction_year >= 1935 and construction_year <= 1948:
            if building_category == "mfh" and num_floors and num_floors in (4, 5):
                # Edge areas of Stockholm/other cities
                if city and not any(area in city.lower() for area in ("gröndal", "västertorp")):
                    # Could be barnrikehus - give it lower confidence
                    pass  # Let it fall through to standard mfh_1930_1945

    # =================================================================
    # HISTORICAL ARCHETYPES
    # =================================================================

    # Stenstaden (stone city)
    if selected_archetype is None:
        if construction_year >= 1880 and construction_year <= 1915:
            if building_category == "mfh" and num_floors and num_floors >= 4:
                if facade_material and facade_material.lower() in ("tegel", "brick", "puts", "render"):
                    selected_archetype = SWEDISH_HISTORICAL_ARCHETYPES.get("hist_stenstaden")
                    reasons.append("1880-1915 + MFH + 4+ floors + brick → stenstaden")
                    confidence += 0.2

    # Jugend
    if construction_year >= 1900 and construction_year <= 1910:
        if building_category == "mfh" and selected_archetype is None:
            selected_archetype = SWEDISH_HISTORICAL_ARCHETYPES.get("hist_jugend")
            reasons.append("1900-1910 + MFH → Jugend")
            confidence += 0.1

    # Nationalromantik
    if construction_year >= 1910 and construction_year <= 1920:
        if building_category == "mfh" and selected_archetype is None:
            selected_archetype = SWEDISH_HISTORICAL_ARCHETYPES.get("hist_nationalromantik")
            reasons.append("1910-1920 + MFH → Nationalromantik")
            confidence += 0.1

    # 20-talsklassicism
    if construction_year >= 1920 and construction_year <= 1930:
        if building_category == "mfh" and selected_archetype is None:
            selected_archetype = SWEDISH_HISTORICAL_ARCHETYPES.get("hist_20tal")
            reasons.append("1920-1930 + MFH → 20-talsklassicism")
            confidence += 0.1

    # Egnahem (workers' homes)
    if construction_year >= 1904 and construction_year <= 1948:
        if building_category == "sfh" and atemp_m2 and atemp_m2 < 150:
            if selected_archetype is None:
                selected_archetype = SWEDISH_HISTORICAL_ARCHETYPES.get("hist_egnahem")
                reasons.append("1904-1948 + SFH + small area → egnahem")
                confidence += 0.15

    # 4. Fall back to standard lookup if no specific match
    if selected_archetype is None:
        selected_archetype = get_archetype_by_year(construction_year, building_category)
        reasons.append(f"Standard {building_category} lookup for {construction_year}")

    # Cap confidence at 1.0
    confidence = min(confidence, 1.0)

    reason_str = "; ".join(reasons)
    return selected_archetype, reason_str, confidence


def classify_building_type(
    num_apartments: Optional[int] = None,
    num_floors: Optional[int] = None,
    atemp_m2: Optional[float] = None,
    building_form: Optional[str] = None,
) -> str:
    """
    Classify building type based on available data.

    Returns one of: "mfh", "sfh", "terraced", "unknown"
    """
    # Form-based classification (highest confidence)
    if building_form:
        form_lower = building_form.lower()
        if form_lower in ("radhus", "kedjehus", "parhus", "row_house", "terraced"):
            return "terraced"
        elif form_lower in ("villa", "villa_1_plan", "villa_1_5_plan", "egnahem"):
            return "sfh"
        elif form_lower in ("lamellhus", "skivhus", "punkthus", "loftgangshus"):
            return "mfh"

    # Apartment-based classification
    if num_apartments is not None:
        if num_apartments == 1:
            return "sfh"
        elif num_apartments <= 6 and num_floors and num_floors <= 2:
            return "terraced"
        else:
            return "mfh"

    # Area-based heuristic
    if atemp_m2 is not None:
        if atemp_m2 < 250:
            return "sfh"
        elif atemp_m2 < 500:
            return "terraced"
        else:
            return "mfh"

    return "unknown"


def get_u_value_for_year(year: int, component: str = "wall") -> float:
    """
    Get typical U-value for a building component by construction year.

    Args:
        year: Construction year
        component: "wall", "roof", "window", "floor"

    Returns:
        Typical U-value in W/m²K
    """
    archetype = get_archetype_by_year(year)

    if component == "wall":
        return archetype.wall_constructions[0].u_value
    elif component == "roof":
        return archetype.roof_construction.u_value
    elif component == "window":
        return archetype.window_construction.u_value_installed
    elif component == "floor":
        return archetype.floor_construction.u_value
    else:
        raise ValueError(f"Unknown component: {component}")


def get_heating_kwh_by_year(year: int) -> float:
    """
    Get typical heating energy consumption by construction year.

    Args:
        year: Construction year

    Returns:
        Typical heating in kWh/m²/year
    """
    archetype = get_archetype_by_year(year)
    return archetype.typical_heating_kwh_m2


def list_archetypes() -> List[str]:
    """List all available archetype IDs."""
    return list(SWEDISH_MFH_ARCHETYPES.keys())


def get_archetype_summary() -> str:
    """Get a summary of all archetypes for documentation."""
    lines = ["Swedish Multi-Family Building Archetypes", "=" * 45, ""]

    for archetype in SWEDISH_MFH_ARCHETYPES.values():
        wall_u = archetype.wall_constructions[0].u_value
        lines.append(f"{archetype.year_start}-{archetype.year_end}: {archetype.name_en}")
        lines.append(f"  Wall U: {wall_u:.2f}, Window U: {archetype.window_construction.u_value_installed:.1f}")
        lines.append(f"  Heating: {archetype.typical_heating_kwh_m2} kWh/m²")
        lines.append(f"  Ventilation: {archetype.ventilation_type.value}")
        lines.append(f"  Stock share: {archetype.stock_share_percent}%")
        lines.append("")

    return "\n".join(lines)


# =============================================================================
# SWEDISH SINGLE-FAMILY ARCHETYPES (SMÅHUS)
# =============================================================================
# Source: TABULA/EPISCOPE SE, Boverket BETSI, Energimyndigheten 2009
# Key finding: ~715,000 houses built 1961-1980, largely homogeneous
# Pre/post SBN75 distinction is critical for this era

SWEDISH_SFH_ARCHETYPES: Dict[str, DetailedArchetype] = {

    # =========================================================================
    # PRE-1930: Traditional timber construction
    # =========================================================================
    "sfh_pre_1930": DetailedArchetype(
        id="sfh_pre_1930",
        name_sv="Småhus före 1930",
        name_en="Single-family pre-1930",
        era=BuildingEra.PRE_1930,
        year_start=1880,
        year_end=1929,

        stock_share_percent=4.5,
        typical_atemp_m2=(100, 200),
        typical_floors=(1, 2),

        wall_constructions=[
            WallConstruction(
                type=WallConstructionType.STANDING_PLANK,
                name_sv="Stående timmer/plank",
                name_en="Standing timber/plank",
                total_thickness_mm=150,
                insulation_thickness_mm=0,
                insulation_type="air_gap",
                u_value=1.4,
                thermal_bridge_factor=1.0,
                description="Traditional timber with no insulation, air gaps"
            ),
            WallConstruction(
                type=WallConstructionType.SOLID_BRICK_1_STONE,
                name_sv="1-stens tegel",
                name_en="Solid brick",
                total_thickness_mm=240,
                insulation_thickness_mm=0,
                insulation_type="none",
                u_value=1.6,
                thermal_bridge_factor=1.0,
            ),
        ],

        roof_construction=RoofConstruction(
            name_sv="Kallvind med sågspån",
            insulation_thickness_mm=50,
            insulation_type="sågspån",
            u_value=0.7,
            roof_type="cold_attic"
        ),

        floor_construction=FloorConstruction(
            name_sv="Träbjälklag på torpargrund",
            type="crawlspace",
            insulation_thickness_mm=0,
            u_value=0.9
        ),

        window_construction=WindowConstruction(
            type=WindowType.SINGLE_PANE,
            name_sv="Enkelglasfönster",
            name_en="Single pane windows",
            u_value_glass=5.7,
            u_value_installed=4.5,
            shgc=0.85,
            num_panes=1,
            gas_fill="air"
        ),
        typical_wwr=0.12,
        wwr_by_orientation={"N": 0.08, "S": 0.15, "E": 0.10, "W": 0.10},

        infiltration_ach=0.60,
        n50_ach=15.0,

        ventilation_type=VentilationType.NATURAL,
        ventilation_rate_l_s_m2=0.30,
        heat_recovery_efficiency=0.0,
        sfp_kw_per_m3s=0.0,

        heating_systems={
            HeatingSystemType.OIL_BOILER: 0.30,
            HeatingSystemType.ELECTRIC_DIRECT: 0.35,
            HeatingSystemType.HEAT_PUMP_AIR: 0.20,
            HeatingSystemType.DISTRICT_BIOMASS: 0.15,
        },
        typical_heating_kwh_m2=220,

        dhw_kwh_m2=30,

        occupancy_w_per_m2=2.0,
        lighting_w_m2=6,
        equipment_w_m2=5,

        typical_forms=["villa_1_5_plan", "villa_1_plan"],
        typical_facades=["trä", "tegel", "puts"],

        common_issues=[
            "Mycket hög infiltration",
            "Inget isolering",
            "Fuktproblem i grund",
            "Enkla fönster",
            "Bristfällig ventilation",
        ],
        renovation_potential_kwh_m2=120,
        typical_ecms=[
            "wall_internal_insulation",
            "window_replacement",
            "attic_insulation",
            "air_sealing",
            "heat_pump_air",
        ],

        description="Traditional Swedish timber houses, no modern insulation",
        sources=["TABULA SE", "BETSI 2010", "Boverket"],
        descriptors=ArchetypeDescriptors(
            building_depth_m=(7.0, 10.0),
            floor_to_floor_m=(2.4, 2.7),
            building_length_m=(8.0, 14.0),
            plan_shape=[PlanShape.RECTANGULAR, PlanShape.L_SHAPE],
            stairwell_apartments=(1, 1),
            balcony_types=[BalconyType.NONE],
            roof_profiles=[RoofProfile.PITCHED, RoofProfile.MANSARD],
            facade_patterns=[FacadePattern.REGULAR_PUNCHED],
            typical_colors=["faluröd", "vit", "gul", "grå"],
            window_proportions="portrait",
            has_bay_windows=False,
            has_corner_windows=False,
            urban_settings=[UrbanSetting.RURAL, UrbanSetting.SMALL_TOWN, UrbanSetting.OUTER_SUBURB],
            typical_neighborhoods=["Landsbygd", "Villaområde", "Äldre trädgårdsstad"],
            typical_cities=["Hela Sverige"],
            original_ownership=[OwnershipType.PRIVATE_OWNER],
            housing_programs=["Egnahem"],
            notable_developers=[],
            notable_architects=[],
            typical_certifications=[EnergyCertification.ENERGY_CLASS_G],
            keywords_sv=["timmerhus", "torparstugan", "sekelskiftevilla", "trähus", "faluröd",
                        "arbetarbostad", "bondgård"],
            keywords_en=["timber house", "wooden house", "farmhouse", "cottage", "traditional"],
            infiltration_variability="high",
            u_value_variability="high",
            occupancy_pattern="residential",
            likely_renovated_if=["tilläggsisolerat", "fönsterbyte", "värmepump installerad"],
            renovation_era_signs={"1970s": "tilläggsisolering", "1990s": "fönsterbyte",
                                 "2010s": "luftvärmepump"},
        ),
    ),

    # =========================================================================
    # 1930-1945: Funkis era
    # =========================================================================
    "sfh_1930_1945": DetailedArchetype(
        id="sfh_1930_1945",
        name_sv="Småhus 1930-1945 (Funkis)",
        name_en="Single-family 1930-1945 (Functionalism)",
        era=BuildingEra.FUNKIS_1930_1945,
        year_start=1930,
        year_end=1945,

        stock_share_percent=3.5,
        typical_atemp_m2=(100, 180),
        typical_floors=(1, 2),

        wall_constructions=[
            WallConstruction(
                type=WallConstructionType.STUD_FRAME_MINERAL,
                name_sv="Regelstomme med kutterspån",
                name_en="Stud frame with wood shavings",
                total_thickness_mm=150,
                insulation_thickness_mm=100,
                insulation_type="kutterspån",
                u_value=0.9,
                thermal_bridge_factor=1.10,
                description="Early insulated frame construction"
            ),
        ],

        roof_construction=RoofConstruction(
            name_sv="Kallvind med mineralull",
            insulation_thickness_mm=100,
            insulation_type="mineralull",
            u_value=0.40,
            roof_type="cold_attic"
        ),

        floor_construction=FloorConstruction(
            name_sv="Träbjälklag på krypgrund",
            type="crawlspace",
            insulation_thickness_mm=50,
            u_value=0.65
        ),

        window_construction=WindowConstruction(
            type=WindowType.COUPLED_2_PANE,
            name_sv="Kopplade 2-glas",
            name_en="Coupled double windows",
            u_value_glass=3.0,
            u_value_installed=2.8,
            shgc=0.75,
            num_panes=2,
            gas_fill="air"
        ),
        typical_wwr=0.14,
        wwr_by_orientation={"N": 0.10, "S": 0.18, "E": 0.12, "W": 0.12},

        infiltration_ach=0.40,
        n50_ach=10.0,

        ventilation_type=VentilationType.NATURAL,
        ventilation_rate_l_s_m2=0.30,
        heat_recovery_efficiency=0.0,
        sfp_kw_per_m3s=0.0,

        heating_systems={
            HeatingSystemType.OIL_BOILER: 0.25,
            HeatingSystemType.ELECTRIC_DIRECT: 0.30,
            HeatingSystemType.HEAT_PUMP_AIR: 0.30,
            HeatingSystemType.DISTRICT_BIOMASS: 0.15,
        },
        typical_heating_kwh_m2=190,

        dhw_kwh_m2=28,

        occupancy_w_per_m2=2.0,
        lighting_w_m2=6,
        equipment_w_m2=6,

        typical_forms=["villa_1_5_plan"],
        typical_facades=["trä", "puts"],

        common_issues=[
            "Bristfällig isolering",
            "Luftläckage",
            "Gamla fönster",
        ],
        renovation_potential_kwh_m2=90,
        typical_ecms=[
            "wall_external_insulation",
            "window_replacement",
            "attic_insulation",
            "heat_pump_air",
        ],

        description="Functionalist villas with early insulation attempts",
        sources=["TABULA SE", "BETSI 2010"],
        descriptors=ArchetypeDescriptors(
            building_depth_m=(8.0, 10.0),
            floor_to_floor_m=(2.5, 2.8),
            building_length_m=(10.0, 14.0),
            plan_shape=[PlanShape.RECTANGULAR, PlanShape.L_SHAPE],
            stairwell_apartments=(1, 1),
            balcony_types=[BalconyType.NONE, BalconyType.PROJECTING],
            roof_profiles=[RoofProfile.LOW_PITCHED, RoofProfile.FLAT],
            facade_patterns=[FacadePattern.REGULAR_PUNCHED],
            typical_colors=["vit", "ljusgrå", "gul"],
            window_proportions="square",
            has_bay_windows=False,
            has_corner_windows=True,
            urban_settings=[UrbanSetting.INNER_SUBURB, UrbanSetting.OUTER_SUBURB, UrbanSetting.SMALL_TOWN],
            typical_neighborhoods=["Villastad", "Trädgårdsstad"],
            typical_cities=["Stockholm", "Göteborg", "Malmö"],
            original_ownership=[OwnershipType.PRIVATE_OWNER],
            housing_programs=["Funkis-villa"],
            notable_developers=[],
            notable_architects=[],
            typical_certifications=[EnergyCertification.ENERGY_CLASS_F, EnergyCertification.ENERGY_CLASS_G],
            keywords_sv=["funkisvilla", "30-talsvilla", "40-talsvilla", "platt tak", "putsad"],
            keywords_en=["functionalist villa", "1930s villa", "flat roof"],
            infiltration_variability="medium",
            u_value_variability="medium",
            occupancy_pattern="residential",
            likely_renovated_if=["tilläggsisolering", "fönsterbyte", "värmepump"],
            renovation_era_signs={"1970s": "tilläggsisolering", "2000s": "fönsterbyte"},
        ),
    ),

    # =========================================================================
    # 1946-1960: Folkhemmet villas
    # =========================================================================
    "sfh_1946_1960": DetailedArchetype(
        id="sfh_1946_1960",
        name_sv="Småhus 1946-1960 (Folkhemmet)",
        name_en="Single-family 1946-1960 (People's Home)",
        era=BuildingEra.FOLKHEM_1946_1960,
        year_start=1946,
        year_end=1960,

        stock_share_percent=8.0,
        typical_atemp_m2=(100, 160),
        typical_floors=(1, 2),

        wall_constructions=[
            WallConstruction(
                type=WallConstructionType.LIGHT_CONCRETE_BLOCK,
                name_sv="Lättbetongblock",
                name_en="Lightweight concrete (AAC)",
                total_thickness_mm=250,
                insulation_thickness_mm=200,  # Lättbetong itself
                insulation_type="lättbetong",
                u_value=0.70,
                thermal_bridge_factor=1.05,
                description="Common Folkhemmet construction, λ≈0.16 W/(m·K)"
            ),
            WallConstruction(
                type=WallConstructionType.STUD_FRAME_MINERAL,
                name_sv="Regelstomme med mineralull",
                name_en="Stud frame with mineral wool",
                total_thickness_mm=150,
                insulation_thickness_mm=100,
                insulation_type="mineralull",
                u_value=0.55,
                thermal_bridge_factor=1.10,
            ),
        ],

        roof_construction=RoofConstruction(
            name_sv="Kallvind med mineralull",
            insulation_thickness_mm=125,
            insulation_type="mineralull",
            u_value=0.30,
            roof_type="cold_attic"
        ),

        floor_construction=FloorConstruction(
            name_sv="Betongplatta eller krypgrund",
            type="slab_on_grade",
            insulation_thickness_mm=50,
            u_value=0.50
        ),

        window_construction=WindowConstruction(
            type=WindowType.COUPLED_2_PANE,
            name_sv="Kopplade 2-glas",
            name_en="Coupled double windows",
            u_value_glass=2.8,
            u_value_installed=2.6,
            shgc=0.72,
            num_panes=2,
            gas_fill="air"
        ),
        typical_wwr=0.15,
        wwr_by_orientation={"N": 0.10, "S": 0.18, "E": 0.14, "W": 0.14},

        infiltration_ach=0.25,
        n50_ach=7.0,

        ventilation_type=VentilationType.NATURAL,
        ventilation_rate_l_s_m2=0.35,
        heat_recovery_efficiency=0.0,
        sfp_kw_per_m3s=0.0,

        heating_systems={
            HeatingSystemType.OIL_BOILER: 0.20,
            HeatingSystemType.ELECTRIC_DIRECT: 0.25,
            HeatingSystemType.HEAT_PUMP_AIR: 0.35,
            HeatingSystemType.DISTRICT_BIOMASS: 0.15,
            HeatingSystemType.HEAT_PUMP_GROUND: 0.05,
        },
        typical_heating_kwh_m2=165,

        dhw_kwh_m2=28,

        occupancy_w_per_m2=2.2,
        lighting_w_m2=7,
        equipment_w_m2=7,

        typical_forms=["villa_1_5_plan", "villa_1_plan"],
        typical_facades=["puts", "trä", "tegel"],

        common_issues=[
            "Lättbetong kan ha radonfrågor (alunhaltig)",
            "Bristfällig isolering vs. moderna krav",
            "Självdragsventilation",
        ],
        renovation_potential_kwh_m2=70,
        typical_ecms=[
            "wall_external_insulation",
            "attic_insulation",
            "window_replacement",
            "heat_pump_air",
        ],

        description="Post-war villas, lightweight concrete (lättbetong) common",
        sources=["TABULA SE", "BETSI 2010", "Energimyndigheten 2009"],
        descriptors=ArchetypeDescriptors(
            building_depth_m=(8.0, 11.0),
            floor_to_floor_m=(2.5, 2.7),
            building_length_m=(10.0, 16.0),
            plan_shape=[PlanShape.RECTANGULAR, PlanShape.L_SHAPE],
            stairwell_apartments=(1, 1),
            balcony_types=[BalconyType.NONE, BalconyType.PROJECTING],
            roof_profiles=[RoofProfile.LOW_PITCHED, RoofProfile.PITCHED],
            facade_patterns=[FacadePattern.REGULAR_PUNCHED],
            typical_colors=["vit", "gul", "ljusgrå", "rosa"],
            window_proportions="square",
            has_bay_windows=False,
            has_corner_windows=False,
            urban_settings=[UrbanSetting.INNER_SUBURB, UrbanSetting.OUTER_SUBURB, UrbanSetting.SMALL_TOWN],
            typical_neighborhoods=["50-talsområde", "Villaområde"],
            typical_cities=["Hela Sverige"],
            original_ownership=[OwnershipType.PRIVATE_OWNER],
            housing_programs=["Folkhemsvilla"],
            notable_developers=["Myresjöhus", "Borohus"],
            notable_architects=[],
            typical_certifications=[EnergyCertification.ENERGY_CLASS_E, EnergyCertification.ENERGY_CLASS_F],
            keywords_sv=["50-talsvilla", "folkhemsvilla", "lättbetong", "ytong", "siporex"],
            keywords_en=["1950s villa", "post-war", "lightweight concrete", "AAC"],
            infiltration_variability="low",
            u_value_variability="medium",
            occupancy_pattern="residential",
            likely_renovated_if=["tilläggsisolering", "fönsterbyte", "värmepump installerad"],
            renovation_era_signs={"1980s": "tilläggsisolering", "2000s": "luftvärmepump"},
        ),
    ),

    # =========================================================================
    # 1961-1975: Pre-SBN75 (before oil crisis regulations)
    # =========================================================================
    "sfh_1961_1975_pre_sbn75": DetailedArchetype(
        id="sfh_1961_1975_pre_sbn75",
        name_sv="Småhus 1961-1975 (Före SBN75)",
        name_en="Single-family 1961-1975 (Pre-SBN75)",
        era=BuildingEra.REKORD_1961_1975,
        year_start=1961,
        year_end=1975,

        stock_share_percent=18.0,  # ~714,000 houses from this era
        typical_atemp_m2=(100, 180),
        typical_floors=(1, 2),

        wall_constructions=[
            WallConstruction(
                type=WallConstructionType.STUD_FRAME_MINERAL,
                name_sv="Regelstomme 95mm med mineralull",
                name_en="95mm stud frame with mineral wool",
                total_thickness_mm=150,
                insulation_thickness_mm=95,
                insulation_type="mineralull",
                u_value=0.50,
                thermal_bridge_factor=1.15,
                description="Pre-SBN75: 95mm studs, minimal insulation"
            ),
            WallConstruction(
                type=WallConstructionType.LIGHT_CONCRETE_BLOCK,
                name_sv="Lättbetongblock 250mm",
                name_en="Lightweight concrete 250mm",
                total_thickness_mm=250,
                insulation_thickness_mm=200,
                insulation_type="lättbetong",
                u_value=0.55,
                thermal_bridge_factor=1.05,
            ),
        ],

        roof_construction=RoofConstruction(
            name_sv="Takstol med mineralull 125mm",
            insulation_thickness_mm=125,
            insulation_type="mineralull",
            u_value=0.30,
            roof_type="cold_attic"
        ),

        floor_construction=FloorConstruction(
            name_sv="Platta på mark med EPS",
            type="slab_on_grade",
            insulation_thickness_mm=50,
            u_value=0.45
        ),

        window_construction=WindowConstruction(
            type=WindowType.SEALED_2_PANE,
            name_sv="Förseglade 2-glas",
            name_en="Sealed double glazed",
            u_value_glass=2.8,
            u_value_installed=2.6,
            shgc=0.70,
            num_panes=2,
            gas_fill="air"
        ),
        typical_wwr=0.18,
        wwr_by_orientation={"N": 0.12, "S": 0.22, "E": 0.16, "W": 0.16},

        infiltration_ach=0.20,
        n50_ach=5.0,

        ventilation_type=VentilationType.NATURAL,
        ventilation_rate_l_s_m2=0.35,
        heat_recovery_efficiency=0.0,
        sfp_kw_per_m3s=0.0,

        heating_systems={
            HeatingSystemType.OIL_BOILER: 0.15,
            HeatingSystemType.ELECTRIC_DIRECT: 0.25,
            HeatingSystemType.HEAT_PUMP_AIR: 0.40,
            HeatingSystemType.HEAT_PUMP_GROUND: 0.10,
            HeatingSystemType.DISTRICT_BIOMASS: 0.10,
        },
        typical_heating_kwh_m2=145,

        dhw_kwh_m2=28,

        occupancy_w_per_m2=2.3,
        lighting_w_m2=8,
        equipment_w_m2=8,

        typical_forms=["villa_1_5_plan", "villa_1_plan", "kedjehus"],
        typical_facades=["trä", "tegel", "puts"],

        common_issues=[
            "Pre-SBN75: underdimensionerad isolering",
            "Självdragsventilation utan värmeåtervinning",
            "Kallras från fönster",
            "Gammal oljepanna ofta ersatt",
        ],
        renovation_potential_kwh_m2=60,
        typical_ecms=[
            "wall_external_insulation",
            "attic_insulation",
            "window_replacement",
            "ftx_installation",
            "heat_pump_ground",
        ],

        description="Pre-oil crisis construction, 95mm studs typical, minimal insulation by modern standards. About 40% higher energy use than post-2011 houses.",
        sources=["TABULA SE", "BETSI 2010", "Licentiate thesis Ekström", "Energimyndigheten 2009"],
        descriptors=ArchetypeDescriptors(
            building_depth_m=(9.0, 12.0),
            floor_to_floor_m=(2.4, 2.6),
            building_length_m=(10.0, 18.0),
            plan_shape=[PlanShape.RECTANGULAR, PlanShape.L_SHAPE],
            stairwell_apartments=(1, 1),
            balcony_types=[BalconyType.NONE, BalconyType.PROJECTING],
            roof_profiles=[RoofProfile.LOW_PITCHED],
            facade_patterns=[FacadePattern.REGULAR_PUNCHED],
            typical_colors=["gul", "vit", "brun", "grå"],
            window_proportions="landscape",
            has_bay_windows=False,
            has_corner_windows=False,
            urban_settings=[UrbanSetting.OUTER_SUBURB, UrbanSetting.SMALL_TOWN, UrbanSetting.RURAL],
            typical_neighborhoods=["60-talsområde", "70-talsområde", "Villaområde"],
            typical_cities=["Hela Sverige"],
            original_ownership=[OwnershipType.PRIVATE_OWNER],
            housing_programs=[],
            notable_developers=["Myresjöhus", "Borohus", "Hjältevadshus", "Smålandsvillan"],
            notable_architects=[],
            typical_certifications=[EnergyCertification.ENERGY_CLASS_E, EnergyCertification.ENERGY_CLASS_F],
            keywords_sv=["60-talsvilla", "70-talsvilla", "enplansvilla", "katalogvilla",
                        "rekordåren", "pre-SBN75"],
            keywords_en=["1960s villa", "1970s villa", "catalog house", "pre-oil crisis"],
            infiltration_variability="medium",
            u_value_variability="medium",
            occupancy_pattern="residential",
            likely_renovated_if=["tilläggsisolering", "fönsterbyte", "bergvärme", "luftvärmepump"],
            renovation_era_signs={"1990s": "tilläggsisolering", "2010s": "bergvärme"},
        ),
    ),

    # =========================================================================
    # 1976-1985: Post-SBN75 with improved insulation
    # =========================================================================
    "sfh_1976_1985": DetailedArchetype(
        id="sfh_1976_1985",
        name_sv="Småhus 1976-1985 (Efter SBN75)",
        name_en="Single-family 1976-1985 (Post-SBN75)",
        era=BuildingEra.ENERGI_1976_1985,
        year_start=1976,
        year_end=1985,

        stock_share_percent=12.0,
        typical_atemp_m2=(120, 200),
        typical_floors=(1, 2),

        wall_constructions=[
            WallConstruction(
                type=WallConstructionType.STUD_FRAME_MINERAL,
                name_sv="Regelstomme 170-195mm med mineralull",
                name_en="170-195mm stud frame with mineral wool",
                total_thickness_mm=220,
                insulation_thickness_mm=170,
                insulation_type="mineralull",
                u_value=0.25,
                thermal_bridge_factor=1.08,
                description="SBN75/80: Increased insulation thickness per regulations"
            ),
        ],

        roof_construction=RoofConstruction(
            name_sv="Isolerat vindsbjälklag",
            insulation_thickness_mm=250,
            insulation_type="mineralull",
            u_value=0.15,
            roof_type="cold_attic"
        ),

        floor_construction=FloorConstruction(
            name_sv="Isolerad platta på mark",
            type="slab_on_grade",
            insulation_thickness_mm=100,
            u_value=0.30
        ),

        window_construction=WindowConstruction(
            type=WindowType.TRIPLE_PANE,
            name_sv="3-glasfönster",
            name_en="Triple glazed windows",
            u_value_glass=2.0,
            u_value_installed=1.9,
            shgc=0.62,
            num_panes=3,
            gas_fill="air"
        ),
        typical_wwr=0.18,
        wwr_by_orientation={"N": 0.12, "S": 0.22, "E": 0.16, "W": 0.16},

        infiltration_ach=0.12,
        n50_ach=3.0,  # SBN 1980 introduced 3 ACH50 requirement

        ventilation_type=VentilationType.EXHAUST,
        ventilation_rate_l_s_m2=0.35,
        heat_recovery_efficiency=0.0,
        sfp_kw_per_m3s=1.0,

        heating_systems={
            HeatingSystemType.ELECTRIC_DIRECT: 0.25,
            HeatingSystemType.HEAT_PUMP_AIR: 0.35,
            HeatingSystemType.HEAT_PUMP_GROUND: 0.20,
            HeatingSystemType.DISTRICT_BIOMASS: 0.15,
            HeatingSystemType.OIL_BOILER: 0.05,
        },
        typical_heating_kwh_m2=110,

        dhw_kwh_m2=26,

        occupancy_w_per_m2=2.3,
        lighting_w_m2=8,
        equipment_w_m2=10,

        typical_forms=["villa_1_5_plan", "villa_1_plan", "kedjehus", "radhus"],
        typical_facades=["trä", "tegel"],

        common_issues=[
            "F-ventilation utan värmeåtervinning",
            "Direktel vanligt (höga elkostnader)",
            "Kataloghusera med bra grund men uppgraderbar",
        ],
        renovation_potential_kwh_m2=35,
        typical_ecms=[
            "ftx_installation",
            "heat_pump_ground",
            "attic_insulation",
            "window_upgrade",
        ],

        description="Post oil-crisis houses with significantly improved insulation per SBN75/80. First airtightness codes (3 ACH50). Catalog houses (kataloghus) with k-value wall ~0.17, ceiling ~0.09.",
        sources=["TABULA SE", "SBN 1975", "SBN 1980", "Borohus catalog 1985"],
        descriptors=ArchetypeDescriptors(
            building_depth_m=(9.0, 12.0),
            floor_to_floor_m=(2.4, 2.6),
            building_length_m=(12.0, 20.0),
            plan_shape=[PlanShape.RECTANGULAR, PlanShape.L_SHAPE],
            stairwell_apartments=(1, 1),
            balcony_types=[BalconyType.NONE, BalconyType.PROJECTING],
            roof_profiles=[RoofProfile.LOW_PITCHED, RoofProfile.PITCHED],
            facade_patterns=[FacadePattern.REGULAR_PUNCHED],
            typical_colors=["röd tegel", "gul tegel", "träpanel"],
            window_proportions="square",
            has_bay_windows=False,
            has_corner_windows=False,
            urban_settings=[UrbanSetting.OUTER_SUBURB, UrbanSetting.SMALL_TOWN, UrbanSetting.RURAL],
            typical_neighborhoods=["80-talsområde", "Villaområde"],
            typical_cities=["Hela Sverige"],
            original_ownership=[OwnershipType.PRIVATE_OWNER],
            housing_programs=[],
            notable_developers=["Myresjöhus", "Borohus", "Fiskarhedenvillan", "Trivselhus"],
            notable_architects=[],
            typical_certifications=[EnergyCertification.ENERGY_CLASS_D, EnergyCertification.ENERGY_CLASS_E],
            keywords_sv=["80-talsvilla", "kataloghus", "SBN75", "SBN80", "bra isolering"],
            keywords_en=["1980s villa", "catalog house", "post-oil crisis", "well insulated"],
            infiltration_variability="low",
            u_value_variability="low",
            occupancy_pattern="residential",
            likely_renovated_if=["FTX installerat", "bergvärme"],
            renovation_era_signs={"2010s": "bergvärme", "2020s": "solceller"},
        ),
    ),

    # =========================================================================
    # 1986-1995: Well-insulated with FTX becoming common
    # =========================================================================
    "sfh_1986_1995": DetailedArchetype(
        id="sfh_1986_1995",
        name_sv="Småhus 1986-1995",
        name_en="Single-family 1986-1995",
        era=BuildingEra.MODERN_1986_1995,
        year_start=1986,
        year_end=1995,

        stock_share_percent=7.0,
        typical_atemp_m2=(120, 200),
        typical_floors=(1, 2),

        wall_constructions=[
            WallConstruction(
                type=WallConstructionType.STUD_FRAME_MINERAL,
                name_sv="Regelstomme 195mm + tillägg 45mm",
                name_en="195mm stud + 45mm furring",
                total_thickness_mm=280,
                insulation_thickness_mm=240,
                insulation_type="mineralull",
                u_value=0.18,
                thermal_bridge_factor=1.05,
                description="Deep studs with horizontal furring to break thermal bridges"
            ),
        ],

        roof_construction=RoofConstruction(
            name_sv="Välisolerat vindsbjälklag",
            insulation_thickness_mm=350,
            insulation_type="mineralull/lösull",
            u_value=0.12,
            roof_type="cold_attic"
        ),

        floor_construction=FloorConstruction(
            name_sv="Välisolerad platta",
            type="slab_on_grade",
            insulation_thickness_mm=150,
            u_value=0.22
        ),

        window_construction=WindowConstruction(
            type=WindowType.TRIPLE_PANE,
            name_sv="3-glas med lågemission",
            name_en="Triple with low-e coating",
            u_value_glass=1.6,
            u_value_installed=1.5,
            shgc=0.55,
            num_panes=3,
            gas_fill="air",
            coating="low-e"
        ),
        typical_wwr=0.18,
        wwr_by_orientation={"N": 0.12, "S": 0.22, "E": 0.16, "W": 0.16},

        infiltration_ach=0.08,
        n50_ach=2.0,

        ventilation_type=VentilationType.HEAT_RECOVERY,
        ventilation_rate_l_s_m2=0.35,
        heat_recovery_efficiency=0.70,
        sfp_kw_per_m3s=1.5,

        heating_systems={
            HeatingSystemType.HEAT_PUMP_GROUND: 0.30,
            HeatingSystemType.HEAT_PUMP_AIR: 0.25,
            HeatingSystemType.ELECTRIC_WATERBORNE: 0.20,
            HeatingSystemType.DISTRICT_BIOMASS: 0.15,
            HeatingSystemType.HEAT_PUMP_EXHAUST: 0.10,
        },
        typical_heating_kwh_m2=85,

        dhw_kwh_m2=25,

        occupancy_w_per_m2=2.3,
        lighting_w_m2=7,
        equipment_w_m2=10,

        typical_forms=["villa_1_5_plan", "villa_1_plan", "radhus"],
        typical_facades=["trä", "puts", "tegel"],

        common_issues=[
            "FTX-aggregat kan behöva bytas",
            "God baseline, begränsat uppgraderingsbehov",
        ],
        renovation_potential_kwh_m2=25,
        typical_ecms=[
            "ftx_upgrade",
            "solar_pv",
            "window_upgrade",
        ],

        description="Well-insulated houses with FTX ventilation becoming standard. Swedish wall R-50 typical (over 1 foot thick).",
        sources=["TABULA SE", "Paul Kando survey 1985"],
        descriptors=ArchetypeDescriptors(
            building_depth_m=(10.0, 13.0),
            floor_to_floor_m=(2.5, 2.7),
            building_length_m=(12.0, 20.0),
            plan_shape=[PlanShape.RECTANGULAR, PlanShape.L_SHAPE],
            stairwell_apartments=(1, 1),
            balcony_types=[BalconyType.NONE, BalconyType.PROJECTING],
            roof_profiles=[RoofProfile.PITCHED, RoofProfile.LOW_PITCHED],
            facade_patterns=[FacadePattern.REGULAR_PUNCHED],
            typical_colors=["vit puts", "träpanel", "tegel"],
            window_proportions="square",
            has_bay_windows=False,
            has_corner_windows=False,
            urban_settings=[UrbanSetting.OUTER_SUBURB, UrbanSetting.SMALL_TOWN],
            typical_neighborhoods=["90-talsområde", "Villaområde"],
            typical_cities=["Hela Sverige"],
            original_ownership=[OwnershipType.PRIVATE_OWNER],
            housing_programs=[],
            notable_developers=["Myresjöhus", "Trivselhus", "A-hus", "Eksjöhus"],
            notable_architects=[],
            typical_certifications=[EnergyCertification.ENERGY_CLASS_C, EnergyCertification.ENERGY_CLASS_D],
            keywords_sv=["90-talsvilla", "FTX", "välisolerat", "kataloghus"],
            keywords_en=["1990s villa", "heat recovery", "well insulated"],
            infiltration_variability="low",
            u_value_variability="low",
            occupancy_pattern="residential",
            likely_renovated_if=["FTX-aggregat bytt", "solceller"],
            renovation_era_signs={"2020s": "solceller"},
        ),
    ),

    # =========================================================================
    # 1996-2010: Modern BBR
    # =========================================================================
    "sfh_1996_2010": DetailedArchetype(
        id="sfh_1996_2010",
        name_sv="Småhus 1996-2010",
        name_en="Single-family 1996-2010",
        era=BuildingEra.LAGENERGI_1996_2010,
        year_start=1996,
        year_end=2010,

        stock_share_percent=5.0,
        typical_atemp_m2=(130, 220),
        typical_floors=(1, 2),

        wall_constructions=[
            WallConstruction(
                type=WallConstructionType.STUD_FRAME_MINERAL,
                name_sv="Regelstomme 200mm + utvändig 50mm",
                name_en="200mm stud + 50mm external",
                total_thickness_mm=300,
                insulation_thickness_mm=270,
                insulation_type="mineralull",
                u_value=0.15,
                thermal_bridge_factor=1.03,
            ),
        ],

        roof_construction=RoofConstruction(
            name_sv="Välisolerat tak",
            insulation_thickness_mm=450,
            insulation_type="lösull",
            u_value=0.09,
            roof_type="cold_attic"
        ),

        floor_construction=FloorConstruction(
            name_sv="Välisolerad platta",
            type="slab_on_grade",
            insulation_thickness_mm=200,
            u_value=0.18
        ),

        window_construction=WindowConstruction(
            type=WindowType.LOW_E_TRIPLE,
            name_sv="Energifönster 3-glas argon",
            name_en="Energy triple with argon",
            u_value_glass=1.1,
            u_value_installed=1.2,
            shgc=0.50,
            num_panes=3,
            gas_fill="argon",
            coating="low-e"
        ),
        typical_wwr=0.20,
        wwr_by_orientation={"N": 0.12, "S": 0.28, "E": 0.18, "W": 0.18},

        infiltration_ach=0.05,
        n50_ach=1.2,

        ventilation_type=VentilationType.HEAT_RECOVERY,
        ventilation_rate_l_s_m2=0.35,
        heat_recovery_efficiency=0.80,
        sfp_kw_per_m3s=1.5,

        heating_systems={
            HeatingSystemType.HEAT_PUMP_GROUND: 0.40,
            HeatingSystemType.HEAT_PUMP_AIR: 0.25,
            HeatingSystemType.DISTRICT_BIOMASS: 0.20,
            HeatingSystemType.HEAT_PUMP_EXHAUST: 0.10,
            HeatingSystemType.ELECTRIC_WATERBORNE: 0.05,
        },
        typical_heating_kwh_m2=60,

        dhw_kwh_m2=25,

        occupancy_w_per_m2=2.3,
        lighting_w_m2=6,
        equipment_w_m2=12,

        typical_forms=["villa_1_plan", "villa_1_5_plan", "radhus"],
        typical_facades=["puts", "trä"],

        common_issues=[
            "God baseline, minimal renoveringspotential",
        ],
        renovation_potential_kwh_m2=12,
        typical_ecms=[
            "solar_pv",
            "demand_controlled_ventilation",
        ],

        description="Modern BBR requirements, ground source heat pumps common",
        sources=["BBR", "TABULA SE"],
        descriptors=ArchetypeDescriptors(
            building_depth_m=(10.0, 14.0),
            floor_to_floor_m=(2.5, 2.8),
            building_length_m=(12.0, 22.0),
            plan_shape=[PlanShape.RECTANGULAR, PlanShape.L_SHAPE, PlanShape.U_SHAPE],
            stairwell_apartments=(1, 1),
            balcony_types=[BalconyType.NONE, BalconyType.PROJECTING],
            roof_profiles=[RoofProfile.PITCHED, RoofProfile.LOW_PITCHED, RoofProfile.FLAT],
            facade_patterns=[FacadePattern.REGULAR_PUNCHED, FacadePattern.MIXED],
            typical_colors=["vit puts", "grå", "träpanel", "svart"],
            window_proportions="square",
            has_bay_windows=False,
            has_corner_windows=True,
            urban_settings=[UrbanSetting.OUTER_SUBURB, UrbanSetting.SMALL_TOWN],
            typical_neighborhoods=["2000-talsområde", "Villaområde"],
            typical_cities=["Hela Sverige"],
            original_ownership=[OwnershipType.PRIVATE_OWNER],
            housing_programs=[],
            notable_developers=["Fiskarhedenvillan", "A-hus", "Trivselhus", "Myresjöhus", "Skanska BoKlok"],
            notable_architects=[],
            typical_certifications=[EnergyCertification.ENERGY_CLASS_B, EnergyCertification.ENERGY_CLASS_C],
            keywords_sv=["2000-talsvilla", "bergvärme", "lågenergihus", "BBR"],
            keywords_en=["2000s villa", "ground source heat pump", "low energy", "modern"],
            infiltration_variability="low",
            u_value_variability="low",
            occupancy_pattern="residential",
            likely_renovated_if=["solceller installerade"],
            renovation_era_signs={},
        ),
    ),

    # =========================================================================
    # 2011+: Near-passive house
    # =========================================================================
    "sfh_2011_plus": DetailedArchetype(
        id="sfh_2011_plus",
        name_sv="Småhus 2011+ (Lågenergihus)",
        name_en="Single-family 2011+ (Low-energy)",
        era=BuildingEra.NARA_NOLL_2011_PLUS,
        year_start=2011,
        year_end=2030,

        stock_share_percent=3.0,
        typical_atemp_m2=(140, 250),
        typical_floors=(1, 2),

        wall_constructions=[
            WallConstruction(
                type=WallConstructionType.STUD_FRAME_MINERAL,
                name_sv="Passivinspirerad vägg 350mm",
                name_en="Passive-inspired wall 350mm",
                total_thickness_mm=400,
                insulation_thickness_mm=350,
                insulation_type="mineralull",
                u_value=0.10,
                thermal_bridge_factor=1.02,
            ),
            WallConstruction(
                type=WallConstructionType.CLT,
                name_sv="CLT med utvändig isolering",
                name_en="CLT with external insulation",
                total_thickness_mm=400,
                insulation_thickness_mm=250,
                insulation_type="mineralull",
                u_value=0.11,
                thermal_bridge_factor=1.02,
            ),
        ],

        roof_construction=RoofConstruction(
            name_sv="Passivhusnivå tak",
            insulation_thickness_mm=550,
            insulation_type="lösull",
            u_value=0.07,
            roof_type="cold_attic"
        ),

        floor_construction=FloorConstruction(
            name_sv="Passivhusnivå platta",
            type="slab_on_grade",
            insulation_thickness_mm=300,
            u_value=0.12
        ),

        window_construction=WindowConstruction(
            type=WindowType.PASSIVE_HOUSE,
            name_sv="Passivhusfönster",
            name_en="Passive house certified windows",
            u_value_glass=0.5,
            u_value_installed=0.8,
            shgc=0.45,
            num_panes=3,
            gas_fill="argon",
            coating="triple-low-e"
        ),
        typical_wwr=0.22,
        wwr_by_orientation={"N": 0.12, "S": 0.35, "E": 0.18, "W": 0.18},

        infiltration_ach=0.03,
        n50_ach=0.6,

        ventilation_type=VentilationType.HEAT_RECOVERY,
        ventilation_rate_l_s_m2=0.35,
        heat_recovery_efficiency=0.88,
        sfp_kw_per_m3s=1.2,

        heating_systems={
            HeatingSystemType.HEAT_PUMP_GROUND: 0.45,
            HeatingSystemType.HEAT_PUMP_AIR: 0.30,
            HeatingSystemType.DISTRICT_BIOMASS: 0.15,
            HeatingSystemType.HEAT_PUMP_EXHAUST: 0.10,
        },
        typical_heating_kwh_m2=35,

        dhw_kwh_m2=20,

        occupancy_w_per_m2=2.3,
        lighting_w_m2=5,  # LED standard
        equipment_w_m2=12,

        typical_forms=["villa_1_plan", "villa_2_plan"],
        typical_facades=["trä", "puts"],

        common_issues=[
            "Passivhusnivå, inget renoveringsbehov",
        ],
        renovation_potential_kwh_m2=5,
        typical_ecms=[
            "solar_pv",
            "battery_storage",
        ],

        description="Near-passive house standards, BBR 2011+ requirements",
        sources=["BBR", "FEBY Passivhus"],
        descriptors=ArchetypeDescriptors(
            building_depth_m=(10.0, 15.0),
            floor_to_floor_m=(2.5, 2.9),
            building_length_m=(12.0, 25.0),
            plan_shape=[PlanShape.RECTANGULAR, PlanShape.L_SHAPE],
            stairwell_apartments=(1, 1),
            balcony_types=[BalconyType.NONE, BalconyType.PROJECTING],
            roof_profiles=[RoofProfile.PITCHED, RoofProfile.FLAT, RoofProfile.GREEN],
            facade_patterns=[FacadePattern.REGULAR_PUNCHED, FacadePattern.LARGE_GLAZING],
            typical_colors=["svart träpanel", "vit puts", "grå", "corten"],
            window_proportions="square",
            has_bay_windows=False,
            has_corner_windows=True,
            urban_settings=[UrbanSetting.OUTER_SUBURB, UrbanSetting.SMALL_TOWN, UrbanSetting.RURAL],
            typical_neighborhoods=["Nytt villaområde", "Hållbart bostadsområde"],
            typical_cities=["Hela Sverige"],
            original_ownership=[OwnershipType.PRIVATE_OWNER],
            housing_programs=[],
            notable_developers=["A-hus", "Fiskarhedenvillan", "Trivselhus", "Villa Verde"],
            notable_architects=["Tham & Videgård"],
            typical_certifications=[EnergyCertification.ENERGY_CLASS_A, EnergyCertification.ENERGY_CLASS_B,
                                   EnergyCertification.SVANEN, EnergyCertification.MILJOBYGGNAD_SILVER],
            keywords_sv=["passivhus", "lågenergihus", "nära-noll", "miljöcertifierat", "CLT", "trähus"],
            keywords_en=["passive house", "near-zero", "sustainable", "certified", "timber frame"],
            infiltration_variability="low",
            u_value_variability="low",
            occupancy_pattern="residential",
            likely_renovated_if=[],
            renovation_era_signs={},
        ),
    ),
}


# =============================================================================
# SWEDISH TERRACED HOUSE ARCHETYPES (RADHUS/KEDJEHUS/PARHUS)
# =============================================================================
# Source: Swedish Wood, Boverket, TABULA/EPISCOPE
# ~90% of Swedish low-rise housing is wood-framed

SWEDISH_TERRACED_ARCHETYPES: Dict[str, DetailedArchetype] = {

    # =========================================================================
    # 1960s-1970s RADHUS (Terraced houses)
    # =========================================================================
    "terraced_1960_1975": DetailedArchetype(
        id="terraced_1960_1975",
        name_sv="Radhus 1960-1975",
        name_en="Terraced house 1960-1975",
        era=BuildingEra.REKORD_1961_1975,
        year_start=1960,
        year_end=1975,

        stock_share_percent=4.0,
        typical_atemp_m2=(80, 140),
        typical_floors=(1, 2),

        wall_constructions=[
            WallConstruction(
                type=WallConstructionType.STUD_FRAME_MINERAL,
                name_sv="Regelstomme 95mm med mineralull",
                name_en="95mm stud frame with mineral wool",
                total_thickness_mm=145,
                insulation_thickness_mm=95,
                insulation_type="mineralull",
                u_value=0.50,
                thermal_bridge_factor=1.12,
                description="Pre-SBN75 standard framing, shared party walls"
            ),
        ],

        roof_construction=RoofConstruction(
            name_sv="Pulpettak eller sadeltak",
            insulation_thickness_mm=125,
            insulation_type="mineralull",
            u_value=0.30,
            roof_type="cold_attic"
        ),

        floor_construction=FloorConstruction(
            name_sv="Platta på mark",
            type="slab_on_grade",
            insulation_thickness_mm=50,
            u_value=0.45
        ),

        window_construction=WindowConstruction(
            type=WindowType.SEALED_2_PANE,
            name_sv="Förseglade 2-glas",
            name_en="Sealed double glazed",
            u_value_glass=2.8,
            u_value_installed=2.6,
            shgc=0.70,
            num_panes=2,
            gas_fill="air"
        ),
        typical_wwr=0.20,
        wwr_by_orientation={"N": 0.12, "S": 0.28, "E": 0.18, "W": 0.18},

        infiltration_ach=0.18,
        n50_ach=5.0,

        ventilation_type=VentilationType.NATURAL,
        ventilation_rate_l_s_m2=0.35,
        heat_recovery_efficiency=0.0,
        sfp_kw_per_m3s=0.0,

        heating_systems={
            HeatingSystemType.ELECTRIC_DIRECT: 0.30,
            HeatingSystemType.HEAT_PUMP_AIR: 0.35,
            HeatingSystemType.DISTRICT_BIOMASS: 0.20,
            HeatingSystemType.HEAT_PUMP_GROUND: 0.10,
            HeatingSystemType.OIL_BOILER: 0.05,
        },
        typical_heating_kwh_m2=135,

        dhw_kwh_m2=28,

        occupancy_w_per_m2=2.5,
        lighting_w_m2=8,
        equipment_w_m2=8,

        typical_forms=["radhus", "kedjehus"],
        typical_facades=["tegel", "trä"],

        common_issues=[
            "Gemensamma brandväggar med grannar",
            "Platta tak = risk för läckage",
            "Självdragsventilation",
            "95mm isolering otillräckligt",
        ],
        renovation_potential_kwh_m2=55,
        typical_ecms=[
            "wall_external_insulation",
            "attic_insulation",
            "ftx_installation",
            "heat_pump_air",
        ],

        description="1960s-70s terraced houses. Often built as Miljonprogrammet "
                    "counterparts for families. Shared party walls reduce heat loss.",
        sources=["TABULA SE", "Swedish Wood"],
        descriptors=ArchetypeDescriptors(
            building_depth_m=(8.0, 10.0),
            floor_to_floor_m=(2.4, 2.6),
            building_length_m=(5.0, 8.0),  # Per unit
            plan_shape=[PlanShape.RECTANGULAR],
            stairwell_apartments=(1, 1),
            balcony_types=[BalconyType.NONE, BalconyType.PROJECTING],
            roof_profiles=[RoofProfile.FLAT, RoofProfile.LOW_PITCHED],
            facade_patterns=[FacadePattern.REGULAR_PUNCHED],
            typical_colors=["röd tegel", "gul tegel", "träpanel"],
            window_proportions="landscape",
            has_bay_windows=False,
            has_corner_windows=False,
            urban_settings=[UrbanSetting.OUTER_SUBURB, UrbanSetting.SATELLITE_TOWN],
            typical_neighborhoods=["Radhusområde", "Miljonprogramområde"],
            typical_cities=["Stockholm", "Göteborg", "Malmö", "Uppsala", "Västerås"],
            original_ownership=[OwnershipType.BRF, OwnershipType.PRIVATE_OWNER],
            housing_programs=["Miljonprogrammet"],
            notable_developers=["HSB", "Riksbyggen", "JM"],
            notable_architects=[],
            typical_certifications=[EnergyCertification.ENERGY_CLASS_E, EnergyCertification.ENERGY_CLASS_F],
            keywords_sv=["60-talsradhus", "70-talsradhus", "miljonprogrammet", "kedjehus"],
            keywords_en=["1960s terraced", "1970s terraced", "row house", "townhouse"],
            infiltration_variability="medium",
            u_value_variability="medium",
            occupancy_pattern="residential",
            likely_renovated_if=["tilläggsisolering", "fönsterbyte", "luftvärmepump"],
            renovation_era_signs={"1990s": "tilläggsisolering", "2010s": "luftvärmepump"},
        ),
    ),

    # =========================================================================
    # 1976-1995 RADHUS (Post oil crisis)
    # =========================================================================
    "terraced_1976_1995": DetailedArchetype(
        id="terraced_1976_1995",
        name_sv="Radhus 1976-1995",
        name_en="Terraced house 1976-1995",
        era=BuildingEra.ENERGI_1976_1985,
        year_start=1976,
        year_end=1995,

        stock_share_percent=3.5,
        typical_atemp_m2=(100, 160),
        typical_floors=(1, 2),

        wall_constructions=[
            WallConstruction(
                type=WallConstructionType.STUD_FRAME_MINERAL,
                name_sv="Regelstomme 170mm med mineralull",
                name_en="170mm stud frame with mineral wool",
                total_thickness_mm=210,
                insulation_thickness_mm=170,
                insulation_type="mineralull",
                u_value=0.25,
                thermal_bridge_factor=1.08,
                description="Post-SBN75 improved insulation"
            ),
        ],

        roof_construction=RoofConstruction(
            name_sv="Isolerat vindsbjälklag",
            insulation_thickness_mm=250,
            insulation_type="mineralull",
            u_value=0.15,
            roof_type="cold_attic"
        ),

        floor_construction=FloorConstruction(
            name_sv="Isolerad platta på mark",
            type="slab_on_grade",
            insulation_thickness_mm=100,
            u_value=0.30
        ),

        window_construction=WindowConstruction(
            type=WindowType.TRIPLE_PANE,
            name_sv="3-glasfönster",
            name_en="Triple glazed windows",
            u_value_glass=2.0,
            u_value_installed=1.9,
            shgc=0.62,
            num_panes=3,
            gas_fill="air"
        ),
        typical_wwr=0.20,
        wwr_by_orientation={"N": 0.12, "S": 0.28, "E": 0.18, "W": 0.18},

        infiltration_ach=0.10,
        n50_ach=3.0,

        ventilation_type=VentilationType.EXHAUST,
        ventilation_rate_l_s_m2=0.35,
        heat_recovery_efficiency=0.0,
        sfp_kw_per_m3s=1.0,

        heating_systems={
            HeatingSystemType.ELECTRIC_DIRECT: 0.25,
            HeatingSystemType.HEAT_PUMP_AIR: 0.35,
            HeatingSystemType.HEAT_PUMP_GROUND: 0.20,
            HeatingSystemType.DISTRICT_BIOMASS: 0.15,
            HeatingSystemType.ELECTRIC_WATERBORNE: 0.05,
        },
        typical_heating_kwh_m2=100,

        dhw_kwh_m2=26,

        occupancy_w_per_m2=2.5,
        lighting_w_m2=7,
        equipment_w_m2=10,

        typical_forms=["radhus", "kedjehus", "parhus"],
        typical_facades=["tegel", "trä", "puts"],

        common_issues=[
            "F-ventilation utan värmeåtervinning",
            "Direktel vanligt",
        ],
        renovation_potential_kwh_m2=30,
        typical_ecms=[
            "ftx_installation",
            "heat_pump_ground",
            "solar_pv",
        ],

        description="Post oil-crisis terraced houses with improved insulation. "
                    "SBN75/80 requirements. First airtightness codes.",
        sources=["TABULA SE", "SBN 1975", "SBN 1980"],
        descriptors=ArchetypeDescriptors(
            building_depth_m=(9.0, 11.0),
            floor_to_floor_m=(2.4, 2.6),
            building_length_m=(5.0, 8.0),  # Per unit
            plan_shape=[PlanShape.RECTANGULAR],
            stairwell_apartments=(1, 1),
            balcony_types=[BalconyType.NONE, BalconyType.PROJECTING],
            roof_profiles=[RoofProfile.PITCHED, RoofProfile.LOW_PITCHED],
            facade_patterns=[FacadePattern.REGULAR_PUNCHED],
            typical_colors=["röd tegel", "gul tegel", "träpanel", "puts"],
            window_proportions="square",
            has_bay_windows=False,
            has_corner_windows=False,
            urban_settings=[UrbanSetting.OUTER_SUBURB, UrbanSetting.SMALL_TOWN],
            typical_neighborhoods=["80-tals radhusområde", "90-tals radhusområde"],
            typical_cities=["Hela Sverige"],
            original_ownership=[OwnershipType.BRF, OwnershipType.PRIVATE_OWNER],
            housing_programs=[],
            notable_developers=["JM", "NCC", "Skanska", "PEAB"],
            notable_architects=[],
            typical_certifications=[EnergyCertification.ENERGY_CLASS_D, EnergyCertification.ENERGY_CLASS_E],
            keywords_sv=["80-talsradhus", "90-talsradhus", "SBN75", "kedjehus", "parhus"],
            keywords_en=["1980s terraced", "1990s terraced", "post-oil crisis"],
            infiltration_variability="low",
            u_value_variability="low",
            occupancy_pattern="residential",
            likely_renovated_if=["FTX installerat", "bergvärme"],
            renovation_era_signs={"2010s": "bergvärme", "2020s": "solceller"},
        ),
    ),

    # =========================================================================
    # 1996-2010 RADHUS (Modern)
    # =========================================================================
    "terraced_1996_2010": DetailedArchetype(
        id="terraced_1996_2010",
        name_sv="Radhus 1996-2010",
        name_en="Terraced house 1996-2010",
        era=BuildingEra.LAGENERGI_1996_2010,
        year_start=1996,
        year_end=2010,

        stock_share_percent=2.0,
        typical_atemp_m2=(110, 180),
        typical_floors=(2, 2),

        wall_constructions=[
            WallConstruction(
                type=WallConstructionType.STUD_FRAME_MINERAL,
                name_sv="Regelstomme 200mm + utvändig 50mm",
                name_en="200mm stud + 50mm external",
                total_thickness_mm=300,
                insulation_thickness_mm=250,
                insulation_type="mineralull",
                u_value=0.16,
                thermal_bridge_factor=1.04,
            ),
        ],

        roof_construction=RoofConstruction(
            name_sv="Välisolerat tak",
            insulation_thickness_mm=400,
            insulation_type="lösull",
            u_value=0.10,
            roof_type="cold_attic"
        ),

        floor_construction=FloorConstruction(
            name_sv="Välisolerad platta",
            type="slab_on_grade",
            insulation_thickness_mm=200,
            u_value=0.18
        ),

        window_construction=WindowConstruction(
            type=WindowType.LOW_E_TRIPLE,
            name_sv="Energifönster 3-glas argon",
            name_en="Energy triple with argon",
            u_value_glass=1.1,
            u_value_installed=1.2,
            shgc=0.50,
            num_panes=3,
            gas_fill="argon",
            coating="low-e"
        ),
        typical_wwr=0.22,
        wwr_by_orientation={"N": 0.14, "S": 0.30, "E": 0.20, "W": 0.20},

        infiltration_ach=0.05,
        n50_ach=1.2,

        ventilation_type=VentilationType.HEAT_RECOVERY,
        ventilation_rate_l_s_m2=0.35,
        heat_recovery_efficiency=0.80,
        sfp_kw_per_m3s=1.5,

        heating_systems={
            HeatingSystemType.HEAT_PUMP_GROUND: 0.40,
            HeatingSystemType.HEAT_PUMP_AIR: 0.25,
            HeatingSystemType.DISTRICT_BIOMASS: 0.20,
            HeatingSystemType.HEAT_PUMP_EXHAUST: 0.10,
            HeatingSystemType.ELECTRIC_WATERBORNE: 0.05,
        },
        typical_heating_kwh_m2=55,

        dhw_kwh_m2=25,

        occupancy_w_per_m2=2.5,
        lighting_w_m2=6,
        equipment_w_m2=12,

        typical_forms=["radhus", "parhus"],
        typical_facades=["puts", "trä"],

        common_issues=[
            "God baseline, begränsad renoveringspotential",
        ],
        renovation_potential_kwh_m2=12,
        typical_ecms=[
            "solar_pv",
            "demand_controlled_ventilation",
        ],

        description="Modern BBR terraced houses with FTX ventilation.",
        sources=["BBR", "TABULA SE"],
        descriptors=ArchetypeDescriptors(
            building_depth_m=(10.0, 12.0),
            floor_to_floor_m=(2.5, 2.7),
            building_length_m=(6.0, 9.0),  # Per unit
            plan_shape=[PlanShape.RECTANGULAR],
            stairwell_apartments=(1, 1),
            balcony_types=[BalconyType.NONE, BalconyType.PROJECTING],
            roof_profiles=[RoofProfile.PITCHED, RoofProfile.FLAT],
            facade_patterns=[FacadePattern.REGULAR_PUNCHED, FacadePattern.MIXED],
            typical_colors=["vit puts", "grå", "träpanel"],
            window_proportions="square",
            has_bay_windows=False,
            has_corner_windows=True,
            urban_settings=[UrbanSetting.OUTER_SUBURB, UrbanSetting.SMALL_TOWN, UrbanSetting.WATERFRONT],
            typical_neighborhoods=["2000-tals radhusområde", "Nybyggt område"],
            typical_cities=["Hela Sverige"],
            original_ownership=[OwnershipType.BRF, OwnershipType.PRIVATE_OWNER],
            housing_programs=[],
            notable_developers=["JM", "NCC", "Skanska", "Bonava", "HSB"],
            notable_architects=[],
            typical_certifications=[EnergyCertification.ENERGY_CLASS_B, EnergyCertification.ENERGY_CLASS_C],
            keywords_sv=["2000-talsradhus", "FTX", "bergvärme", "modernt radhus"],
            keywords_en=["2000s terraced", "modern townhouse", "heat pump"],
            infiltration_variability="low",
            u_value_variability="low",
            occupancy_pattern="residential",
            likely_renovated_if=["solceller installerade"],
            renovation_era_signs={},
        ),
    ),

    # =========================================================================
    # 2011+ RADHUS (Near-passive)
    # =========================================================================
    "terraced_2011_plus": DetailedArchetype(
        id="terraced_2011_plus",
        name_sv="Radhus 2011+ (Lågenergihus)",
        name_en="Terraced house 2011+ (Low-energy)",
        era=BuildingEra.NARA_NOLL_2011_PLUS,
        year_start=2011,
        year_end=2030,

        stock_share_percent=1.5,
        typical_atemp_m2=(120, 200),
        typical_floors=(2, 2),

        wall_constructions=[
            WallConstruction(
                type=WallConstructionType.STUD_FRAME_MINERAL,
                name_sv="Passivinspirerad vägg 300mm",
                name_en="Passive-inspired wall 300mm",
                total_thickness_mm=350,
                insulation_thickness_mm=300,
                insulation_type="mineralull",
                u_value=0.12,
                thermal_bridge_factor=1.02,
            ),
        ],

        roof_construction=RoofConstruction(
            name_sv="Passivhusnivå tak",
            insulation_thickness_mm=500,
            insulation_type="lösull",
            u_value=0.08,
            roof_type="cold_attic"
        ),

        floor_construction=FloorConstruction(
            name_sv="Passivhusnivå platta",
            type="slab_on_grade",
            insulation_thickness_mm=300,
            u_value=0.12
        ),

        window_construction=WindowConstruction(
            type=WindowType.PASSIVE_HOUSE,
            name_sv="Passivhusfönster",
            name_en="Passive house certified windows",
            u_value_glass=0.5,
            u_value_installed=0.8,
            shgc=0.45,
            num_panes=3,
            gas_fill="argon",
            coating="triple-low-e"
        ),
        typical_wwr=0.24,
        wwr_by_orientation={"N": 0.14, "S": 0.35, "E": 0.22, "W": 0.20},

        infiltration_ach=0.03,
        n50_ach=0.6,

        ventilation_type=VentilationType.HEAT_RECOVERY,
        ventilation_rate_l_s_m2=0.35,
        heat_recovery_efficiency=0.88,
        sfp_kw_per_m3s=1.2,

        heating_systems={
            HeatingSystemType.HEAT_PUMP_GROUND: 0.45,
            HeatingSystemType.HEAT_PUMP_AIR: 0.30,
            HeatingSystemType.DISTRICT_BIOMASS: 0.15,
            HeatingSystemType.HEAT_PUMP_EXHAUST: 0.10,
        },
        typical_heating_kwh_m2=30,

        dhw_kwh_m2=20,

        occupancy_w_per_m2=2.5,
        lighting_w_m2=5,
        equipment_w_m2=12,

        typical_forms=["radhus"],
        typical_facades=["trä", "puts"],

        common_issues=[
            "Passivhusnivå, inget renoveringsbehov",
        ],
        renovation_potential_kwh_m2=5,
        typical_ecms=[
            "solar_pv",
            "battery_storage",
        ],

        description="Near-passive terraced houses, BBR 2011+ requirements.",
        sources=["BBR", "FEBY Passivhus"],
        descriptors=ArchetypeDescriptors(
            building_depth_m=(10.0, 13.0),
            floor_to_floor_m=(2.5, 2.8),
            building_length_m=(6.0, 10.0),  # Per unit
            plan_shape=[PlanShape.RECTANGULAR],
            stairwell_apartments=(1, 1),
            balcony_types=[BalconyType.NONE, BalconyType.PROJECTING],
            roof_profiles=[RoofProfile.PITCHED, RoofProfile.FLAT, RoofProfile.GREEN],
            facade_patterns=[FacadePattern.REGULAR_PUNCHED, FacadePattern.MIXED, FacadePattern.LARGE_GLAZING],
            typical_colors=["svart träpanel", "vit puts", "grå", "corten"],
            window_proportions="square",
            has_bay_windows=False,
            has_corner_windows=True,
            urban_settings=[UrbanSetting.OUTER_SUBURB, UrbanSetting.SMALL_TOWN, UrbanSetting.WATERFRONT],
            typical_neighborhoods=["Hållbart bostadsområde", "Nybyggt område"],
            typical_cities=["Hela Sverige"],
            original_ownership=[OwnershipType.BRF, OwnershipType.PRIVATE_OWNER],
            housing_programs=[],
            notable_developers=["JM", "Skanska", "NCC", "Bonava", "HSB", "Veidekke"],
            notable_architects=[],
            typical_certifications=[EnergyCertification.ENERGY_CLASS_A, EnergyCertification.ENERGY_CLASS_B,
                                   EnergyCertification.SVANEN, EnergyCertification.MILJOBYGGNAD_SILVER],
            keywords_sv=["passivhus radhus", "lågenergiradhus", "nära-noll", "miljöcertifierat"],
            keywords_en=["passive terraced", "near-zero townhouse", "sustainable"],
            infiltration_variability="low",
            u_value_variability="low",
            occupancy_pattern="residential",
            likely_renovated_if=[],
            renovation_era_signs={},
        ),
    ),
}


# =============================================================================
# MILJONPROGRAMMET SUB-TYPES
# =============================================================================
# Source: KTH Byggvetenskap, SBUF rapport, Karlstad kommun
# The Miljonprogrammet (1965-1974) used 4 main building types

@dataclass
class MiljonprogrammetSubtype:
    """
    Detailed sub-type within Miljonprogrammet (1961-1975).

    The Miljonprogrammet used industrialized prefab construction with
    4 main building types, each with specific characteristics.
    """
    id: str
    name_sv: str
    name_en: str
    typical_floors: Tuple[int, int]
    typical_apartments_per_entrance: int
    typical_atemp_m2: Tuple[int, int]
    has_elevator: bool
    description: str

    # Building form factors
    facade_length_m: Tuple[int, int]  # min, max typical
    building_depth_m: float
    floor_height_m: float

    # Specific construction details
    wall_construction: str
    balcony_type: str
    thermal_bridges_factor: float  # Higher = more thermal bridges

    # Common renovation needs
    typical_issues: List[str]
    renovation_priority: List[str]


MILJONPROGRAMMET_SUBTYPES: Dict[str, MiljonprogrammetSubtype] = {

    "lamellhus": MiljonprogrammetSubtype(
        id="lamellhus",
        name_sv="Lamellhus",
        name_en="Slab block (low-rise)",
        typical_floors=(3, 4),
        typical_apartments_per_entrance=12,
        typical_atemp_m2=(800, 2000),
        has_elevator=False,

        description="Low-rise slab blocks, 3-4 floors without elevator. "
                    "Built using 3M modular system. ~300,000 apartments built in this form. "
                    "Often placed in parallel rows or right angles around courtyards.",

        facade_length_m=(30, 60),
        building_depth_m=12.0,
        floor_height_m=2.7,

        wall_construction="betongsandwich_100mm_mineralull",
        balcony_type="ingjuten_balkong",
        thermal_bridges_factor=1.15,

        typical_issues=[
            "Betongsandwich med 100mm isolering (U≈0.45)",
            "Ingjutna balkonger = kraftiga köldbryggor",
            "Platta tak med invändig avvattning",
            "F-ventilation utan värmeåtervinning",
        ],
        renovation_priority=[
            "wall_external_insulation",
            "ftx_installation",
            "balcony_thermal_break",
        ],
    ),

    "skivhus": MiljonprogrammetSubtype(
        id="skivhus",
        name_sv="Skivhus",
        name_en="Slab block (high-rise)",
        typical_floors=(8, 12),
        typical_apartments_per_entrance=48,
        typical_atemp_m2=(3000, 8000),
        has_elevator=True,

        description="High-rise slab blocks, 8-12 floors with elevators. "
                    "Similar structure to lamellhus but taller. "
                    "Long continuous facade, often with visible thermal bridges at floor slabs. "
                    "Cranes on rails used for mounting prefab elements.",

        facade_length_m=(50, 120),
        building_depth_m=12.0,
        floor_height_m=2.7,

        wall_construction="betongsandwich_100mm_cellplast",
        balcony_type="ingjuten_balkong",
        thermal_bridges_factor=1.20,

        typical_issues=[
            "Stora fasadytor med köldbryggor vid bjälklagskanter",
            "Hissystem i behov av modernisering",
            "Kulvertsystem med värmeförluster",
            "Platta tak med läckagerisker",
            "Dålig ljudisolering mellan våningar",
        ],
        renovation_priority=[
            "wall_external_insulation",
            "ftx_installation",
            "window_replacement",
            "elevator_modernization",
        ],
    ),

    "punkthus": MiljonprogrammetSubtype(
        id="punkthus",
        name_sv="Punkthus",
        name_en="Point block tower",
        typical_floors=(8, 16),
        typical_apartments_per_entrance=32,
        typical_atemp_m2=(2500, 6000),
        has_elevator=True,

        description="Tower blocks with compact footprint. "
                    "Developed from earlier stjärnhus (star-shaped) forms. "
                    "Central stairwell/elevator core with apartments around. "
                    "Higher surface-to-volume ratio than lamellhus.",

        facade_length_m=(20, 30),
        building_depth_m=20.0,  # Square-ish footprint
        floor_height_m=2.7,

        wall_construction="lattbetong_150mm_puts",
        balcony_type="balkong_med_plåtfront",
        thermal_bridges_factor=1.15,

        typical_issues=[
            "Hög ytaandel ger större värmeförluster",
            "Lättbetong kan ge radonfrågor",
            "Centralt trapphus kan ge ljudproblem",
        ],
        renovation_priority=[
            "wall_external_insulation",
            "ftx_installation",
            "radon_mitigation",
        ],
    ),

    "loftgangshus": MiljonprogrammetSubtype(
        id="loftgangshus",
        name_sv="Loftgångshus",
        name_en="Gallery access block",
        typical_floors=(4, 8),
        typical_apartments_per_entrance=24,
        typical_atemp_m2=(1500, 4000),
        has_elevator=True,

        description="Access via external gallery (loftgång). "
                    "Single-loaded corridor on exterior of building. "
                    "Apartments have windows on both sides (through-ventilation possible). "
                    "Gallery creates additional thermal bridge and weather exposure issues.",

        facade_length_m=(40, 80),
        building_depth_m=10.0,  # Thinner because single-loaded
        floor_height_m=2.7,

        wall_construction="betongsandwich_100mm_mineralull",
        balcony_type="loftgang_öppen",
        thermal_bridges_factor=1.25,  # Higher due to gallery connections

        typical_issues=[
            "Öppna loftgångar ger väderexponering",
            "Kraftiga köldbryggor vid loftgångsinfästningar",
            "Entréer exponerade för väder",
            "Genomgående lägenheter kan ge övertemperatur",
        ],
        renovation_priority=[
            "loftgang_inglasning",
            "wall_external_insulation",
            "ftx_installation",
        ],
    ),
}


# =============================================================================
# SWEDISH CLIMATE ZONES
# =============================================================================
# Source: BBR, Boverket
# Sweden uses 3-4 climate zones for building regulations

@dataclass
class SwedishClimateZone:
    """Swedish climate zone for BBR requirements."""
    id: str
    name_sv: str
    name_en: str
    regions: List[str]  # Counties/regions in this zone

    # Climate data (Sveby typical)
    hdd_base_17: int  # Heating degree days base 17°C
    winter_design_temp_c: float  # Dimensionerande utetemperatur (DUT)
    annual_solar_kwh_m2: int  # Horizontal global radiation

    # BBR energy requirements (kWh/m²/year, non-electric heating)
    bbr_target_mfh_pre2020: float  # Multi-family pre-2020
    bbr_target_sfh_pre2020: float  # Single-family pre-2020
    bbr_target_mfh_2020: float     # Multi-family BBR 2020+
    bbr_target_sfh_2020: float     # Single-family BBR 2020+

    # Typical weather file
    epw_city: str  # EnergyPlus weather city


SWEDISH_CLIMATE_ZONES: Dict[str, SwedishClimateZone] = {

    "zone_I_north": SwedishClimateZone(
        id="zone_I_north",
        name_sv="Klimatzon I (Norrland)",
        name_en="Climate Zone I (Northern Sweden)",
        regions=["Norrbotten", "Västerbotten", "Jämtland"],

        hdd_base_17=5800,
        winter_design_temp_c=-35.0,
        annual_solar_kwh_m2=850,

        bbr_target_mfh_pre2020=130,
        bbr_target_sfh_pre2020=130,
        bbr_target_mfh_2020=115,
        bbr_target_sfh_2020=115,

        epw_city="Lulea",
    ),

    "zone_II_central": SwedishClimateZone(
        id="zone_II_central",
        name_sv="Klimatzon II (Mellansverige)",
        name_en="Climate Zone II (Central Sweden)",
        regions=["Dalarna", "Gävleborg", "Västernorrland", "Uppsala", "Västmanland"],

        hdd_base_17=4500,
        winter_design_temp_c=-22.0,
        annual_solar_kwh_m2=950,

        bbr_target_mfh_pre2020=110,
        bbr_target_sfh_pre2020=110,
        bbr_target_mfh_2020=100,
        bbr_target_sfh_2020=100,

        epw_city="Sundsvall",
    ),

    "zone_III_south": SwedishClimateZone(
        id="zone_III_south",
        name_sv="Klimatzon III (Södra Sverige)",
        name_en="Climate Zone III (Southern Sweden)",
        regions=["Stockholm", "Södermanland", "Östergötland", "Västra Götaland",
                 "Jönköping", "Kronoberg", "Kalmar", "Gotland", "Blekinge",
                 "Skåne", "Halland", "Örebro", "Värmland"],

        hdd_base_17=3500,
        winter_design_temp_c=-16.0,
        annual_solar_kwh_m2=1000,

        bbr_target_mfh_pre2020=90,
        bbr_target_sfh_pre2020=90,
        bbr_target_mfh_2020=80,
        bbr_target_sfh_2020=80,

        epw_city="Stockholm",
    ),
}


def get_climate_zone_for_region(region: str) -> Optional[SwedishClimateZone]:
    """
    Get climate zone for a Swedish region/county.

    Args:
        region: Swedish county or region name

    Returns:
        SwedishClimateZone or None if not found
    """
    region_lower = region.lower()
    for zone in SWEDISH_CLIMATE_ZONES.values():
        for r in zone.regions:
            if r.lower() in region_lower or region_lower in r.lower():
                return zone
    return None


def get_climate_zone_for_city(city: str) -> SwedishClimateZone:
    """
    Get climate zone for a Swedish city (simplified mapping).

    Args:
        city: City name

    Returns:
        SwedishClimateZone (defaults to zone III if unknown)
    """
    city_lower = city.lower()

    # Zone I cities
    zone_I_cities = ["luleå", "lulea", "kiruna", "gällivare", "boden", "umeå", "umea",
                     "skellefteå", "skelleftea", "östersund", "ostersund", "sundsvall"]
    if any(c in city_lower for c in zone_I_cities):
        return SWEDISH_CLIMATE_ZONES["zone_I_north"]

    # Zone II cities
    zone_II_cities = ["gävle", "gavle", "falun", "borlänge", "borlange", "mora",
                      "hudiksvall", "söderhamn", "soderhamn", "örnsköldsvik", "ornskoldsvik"]
    if any(c in city_lower for c in zone_II_cities):
        return SWEDISH_CLIMATE_ZONES["zone_II_central"]

    # Default to Zone III (most of population)
    return SWEDISH_CLIMATE_ZONES["zone_III_south"]


# =============================================================================
# COMBINED ARCHETYPE ACCESS
# =============================================================================

def get_all_archetypes() -> Dict[str, DetailedArchetype]:
    """Get all archetypes (High-Performance + Historical + Special Form + MFH + SFH + Terraced combined)."""
    combined = {}
    combined.update(SWEDISH_HIGH_PERFORMANCE_ARCHETYPES)
    combined.update(SWEDISH_HISTORICAL_ARCHETYPES)
    combined.update(SWEDISH_SPECIAL_FORM_ARCHETYPES)
    combined.update(SWEDISH_MFH_ARCHETYPES)
    combined.update(SWEDISH_SFH_ARCHETYPES)
    combined.update(SWEDISH_TERRACED_ARCHETYPES)
    return combined


def get_archetype(archetype_id: str) -> Optional[DetailedArchetype]:
    """Get a specific archetype by ID."""
    all_archetypes = get_all_archetypes()
    return all_archetypes.get(archetype_id)


def get_archetype_for_building(
    construction_year: int,
    building_type: str = "mfh",
    region: Optional[str] = None
) -> Tuple[DetailedArchetype, Optional[SwedishClimateZone]]:
    """
    Get the most appropriate archetype for a building.

    Args:
        construction_year: Year of construction
        building_type: "mfh" (multi-family) or "sfh" (single-family)
        region: Optional region for climate zone lookup

    Returns:
        Tuple of (archetype, climate_zone)
    """
    archetype = get_archetype_by_year(construction_year, building_type)

    climate_zone = None
    if region:
        climate_zone = get_climate_zone_for_region(region)

    return archetype, climate_zone


# =============================================================================
# DESCRIPTOR-BASED MATCHING
# =============================================================================

@dataclass
class MatchScore:
    """Result from descriptor-based matching."""
    archetype: DetailedArchetype
    total_score: float
    component_scores: Dict[str, float]
    matched_on: List[str]
    notes: List[str]


def match_by_descriptors(
    # Basic building info
    construction_year: int,
    building_type: str = "mfh",

    # Geometric descriptors
    building_depth_m: Optional[float] = None,
    building_length_m: Optional[float] = None,
    floor_height_m: Optional[float] = None,
    plan_shape: Optional[str] = None,

    # Visual descriptors (from street view or inspection)
    balcony_type: Optional[str] = None,
    roof_profile: Optional[str] = None,
    facade_pattern: Optional[str] = None,
    facade_colors: Optional[List[str]] = None,
    has_bay_windows: Optional[bool] = None,
    has_corner_windows: Optional[bool] = None,

    # Contextual descriptors
    neighborhood: Optional[str] = None,
    city: Optional[str] = None,
    urban_setting: Optional[str] = None,

    # Ownership/program
    ownership_type: Optional[str] = None,
    housing_program: Optional[str] = None,
    developer: Optional[str] = None,
    architect: Optional[str] = None,

    # Energy/certification
    energy_class: Optional[str] = None,
    certification: Optional[str] = None,
    declared_energy_kwh_m2: Optional[float] = None,
    has_solar: bool = False,
    has_ftx: bool = False,

    # Keywords from text sources
    keywords: Optional[List[str]] = None,

    # Options
    require_descriptors: bool = False,
    top_n: int = 3,
) -> List[MatchScore]:
    """
    Match building to archetypes using comprehensive descriptors.

    This function enables both deterministic matching from public data
    and AI/ML-based matching from street view images or text analysis.

    Args:
        construction_year: Year of construction (required)
        building_type: "mfh" (multi-family) or "sfh" (single-family)

        Geometric descriptors (from GIS/CAD):
        - building_depth_m: Building depth in meters
        - building_length_m: Building length in meters
        - floor_height_m: Floor-to-floor height in meters
        - plan_shape: Plan shape (rectangular, star, point, slab, etc.)

        Visual descriptors (from street view/inspection):
        - balcony_type: Balcony type (none, recessed, projecting, glazed, gallery)
        - roof_profile: Roof profile (flat, low_slope, pitched, mansard)
        - facade_pattern: Facade pattern (regular_punched, ribbon, curtain_wall)
        - facade_colors: List of facade colors in Swedish
        - has_bay_windows: Has bay windows (burspråk)
        - has_corner_windows: Has corner windows (hörnfönster)

        Contextual descriptors:
        - neighborhood: Neighborhood name
        - city: City name
        - urban_setting: Urban setting (inner_city, inner_suburb, outer_suburb)

        Ownership/program:
        - ownership_type: Ownership type (municipal, cooperative, private_rental)
        - housing_program: Housing program (Miljonprogrammet, Barnrikehus, etc.)
        - developer: Developer name (HSB, Riksbyggen, Svenska Bostäder, etc.)
        - architect: Architect name

        Energy/certification:
        - energy_class: Energy class (A-G)
        - certification: Certification (Miljöbyggnad, FEBY, etc.)
        - declared_energy_kwh_m2: Declared energy use
        - has_solar: Has solar PV
        - has_ftx: Has FTX ventilation

        Keywords:
        - keywords: Keywords from text sources (property listings, records)

        Options:
        - require_descriptors: Only consider archetypes with descriptors
        - top_n: Number of top matches to return

    Returns:
        List of MatchScore objects, sorted by score descending
    """
    all_archetypes = get_all_archetypes()
    results: List[MatchScore] = []

    for arch_id, archetype in all_archetypes.items():
        # Skip if we require descriptors and archetype doesn't have them
        if require_descriptors and not archetype.descriptors:
            continue

        # Skip if building type doesn't match
        if building_type == "sfh" and "sfh" not in arch_id.lower() and "villa" not in arch_id.lower():
            if "mfh" in arch_id.lower() or "flerbostadshus" in archetype.name_sv.lower():
                continue
        if building_type == "mfh" and "sfh" in arch_id.lower():
            continue

        # Start scoring
        scores: Dict[str, float] = {}
        matched_on: List[str] = []
        notes: List[str] = []

        # === YEAR MATCHING (always applies) ===
        if archetype.year_start <= construction_year <= archetype.year_end:
            scores["year"] = 1.0
            matched_on.append("construction_year")
        else:
            # Partial score for near misses
            year_distance = min(
                abs(construction_year - archetype.year_start),
                abs(construction_year - archetype.year_end)
            )
            if year_distance <= 5:
                scores["year"] = 0.8
            elif year_distance <= 10:
                scores["year"] = 0.5
            elif year_distance <= 20:
                scores["year"] = 0.2
            else:
                scores["year"] = 0.0

        # === DESCRIPTOR-BASED MATCHING ===
        if archetype.descriptors:
            desc = archetype.descriptors

            # Geometric matching
            if building_depth_m:
                min_d, max_d = desc.building_depth_m
                if min_d <= building_depth_m <= max_d:
                    scores["depth"] = 1.0
                    matched_on.append("building_depth")
                elif abs(building_depth_m - min_d) <= 2 or abs(building_depth_m - max_d) <= 2:
                    scores["depth"] = 0.7
                else:
                    scores["depth"] = 0.0

            if building_length_m:
                min_l, max_l = desc.building_length_m
                if min_l <= building_length_m <= max_l:
                    scores["length"] = 1.0
                    matched_on.append("building_length")
                elif abs(building_length_m - min_l) <= 5 or abs(building_length_m - max_l) <= 5:
                    scores["length"] = 0.7
                else:
                    scores["length"] = 0.0

            if floor_height_m:
                min_h, max_h = desc.floor_to_floor_m
                if min_h <= floor_height_m <= max_h:
                    scores["floor_height"] = 1.0
                    matched_on.append("floor_height")
                else:
                    scores["floor_height"] = 0.5 if abs(floor_height_m - (min_h + max_h)/2) <= 0.3 else 0.0

            if plan_shape and desc.plan_shape:
                shape_lower = plan_shape.lower()
                plan_names = [p.value.lower() for p in desc.plan_shape]
                if shape_lower in plan_names or any(shape_lower in p for p in plan_names):
                    scores["plan_shape"] = 1.0
                    matched_on.append("plan_shape")
                else:
                    scores["plan_shape"] = 0.0

            # Visual matching
            if balcony_type and desc.balcony_types:
                bt_lower = balcony_type.lower()
                balcony_names = [b.value.lower() for b in desc.balcony_types]
                if bt_lower in balcony_names or any(bt_lower in b for b in balcony_names):
                    scores["balcony"] = 1.0
                    matched_on.append("balcony_type")
                else:
                    scores["balcony"] = 0.0

            if roof_profile and desc.roof_profiles:
                rf_lower = roof_profile.lower()
                roof_names = [r.value.lower() for r in desc.roof_profiles]
                if rf_lower in roof_names or any(rf_lower in r for r in roof_names):
                    scores["roof"] = 1.0
                    matched_on.append("roof_profile")
                else:
                    scores["roof"] = 0.0

            if facade_pattern and desc.facade_patterns:
                fp_lower = facade_pattern.lower()
                pattern_names = [p.value.lower() for p in desc.facade_patterns]
                if fp_lower in pattern_names or any(fp_lower in p for p in pattern_names):
                    scores["facade_pattern"] = 1.0
                    matched_on.append("facade_pattern")
                else:
                    scores["facade_pattern"] = 0.0

            if facade_colors and desc.typical_colors:
                matching_colors = sum(1 for c in facade_colors if c.lower() in [tc.lower() for tc in desc.typical_colors])
                if matching_colors:
                    scores["colors"] = min(1.0, matching_colors / 2)
                    matched_on.append("facade_colors")

            if has_bay_windows is not None and desc.has_bay_windows == has_bay_windows:
                scores["bay_windows"] = 1.0
                matched_on.append("bay_windows")

            if has_corner_windows is not None and desc.has_corner_windows == has_corner_windows:
                scores["corner_windows"] = 1.0
                matched_on.append("corner_windows")

            # Contextual matching
            if neighborhood and desc.typical_neighborhoods:
                if any(neighborhood.lower() in n.lower() for n in desc.typical_neighborhoods):
                    scores["neighborhood"] = 1.0
                    matched_on.append("neighborhood")

            if city and desc.typical_cities:
                if any(city.lower() in c.lower() for c in desc.typical_cities):
                    scores["city"] = 1.0
                    matched_on.append("city")

            if urban_setting and desc.urban_settings:
                us_lower = urban_setting.lower()
                setting_names = [u.value.lower() for u in desc.urban_settings]
                if us_lower in setting_names or any(us_lower in s for s in setting_names):
                    scores["urban_setting"] = 1.0
                    matched_on.append("urban_setting")

            # Ownership matching
            if ownership_type and desc.original_ownership:
                ot_lower = ownership_type.lower()
                owner_names = [o.value.lower() for o in desc.original_ownership]
                if ot_lower in owner_names or any(ot_lower in o for o in owner_names):
                    scores["ownership"] = 1.0
                    matched_on.append("ownership_type")

            if housing_program and desc.housing_programs:
                if any(housing_program.lower() in p.lower() for p in desc.housing_programs):
                    scores["program"] = 1.0
                    matched_on.append("housing_program")

            if developer and desc.notable_developers:
                if any(developer.lower() in d.lower() for d in desc.notable_developers):
                    scores["developer"] = 1.0
                    matched_on.append("developer")

            if architect and desc.notable_architects:
                if any(architect.lower() in a.lower() for a in desc.notable_architects):
                    scores["architect"] = 1.0
                    matched_on.append("architect")

            # Certification matching
            if certification and desc.typical_certifications:
                cert_lower = certification.lower()
                cert_names = [c.value.lower() for c in desc.typical_certifications]
                if cert_lower in cert_names or any(cert_lower in c for c in cert_names):
                    scores["certification"] = 1.0
                    matched_on.append("certification")

            if has_solar and desc.has_solar_pv:
                scores["solar"] = 1.0
                matched_on.append("solar_pv")

            # Keyword matching
            if keywords and (desc.keywords_sv or desc.keywords_en):
                all_keywords = [k.lower() for k in desc.keywords_sv + desc.keywords_en]
                search_keywords = [k.lower() for k in keywords]
                matching = sum(1 for k in search_keywords if any(k in ak or ak in k for ak in all_keywords))
                if matching:
                    scores["keywords"] = min(1.0, matching / 2)
                    matched_on.append(f"keywords({matching})")
                    notes.append(f"Matched keywords: {matching}")

        # Calculate total score with weights
        weights = {
            "year": 3.0,
            "keywords": 2.5,
            "plan_shape": 2.0,
            "neighborhood": 2.0,
            "program": 2.0,
            "architect": 2.0,
            "depth": 1.5,
            "length": 1.0,
            "floor_height": 1.0,
            "balcony": 1.5,
            "roof": 1.0,
            "facade_pattern": 1.0,
            "colors": 0.5,
            "bay_windows": 0.5,
            "corner_windows": 0.5,
            "city": 0.5,
            "urban_setting": 1.0,
            "ownership": 1.0,
            "developer": 1.0,
            "certification": 1.5,
            "solar": 0.5,
        }

        total_weighted = sum(scores.get(k, 0) * weights.get(k, 1.0) for k in scores)
        max_possible = sum(weights.get(k, 1.0) for k in scores)
        total_score = total_weighted / max_possible if max_possible > 0 else 0

        if len(matched_on) >= 5:
            total_score = min(1.0, total_score * 1.1)
            notes.append("Multi-descriptor match boost")

        results.append(MatchScore(
            archetype=archetype,
            total_score=total_score,
            component_scores=scores,
            matched_on=matched_on,
            notes=notes,
        ))

    results.sort(key=lambda x: x.total_score, reverse=True)
    return results[:top_n]


def match_by_keywords(
    keywords: List[str],
    construction_year: Optional[int] = None,
    require_descriptors: bool = True,
) -> List[MatchScore]:
    """
    Simple keyword-based matching for text analysis.

    Useful for matching based on property listings, records, or AI-extracted text.
    """
    return match_by_descriptors(
        construction_year=construction_year or 1970,
        keywords=keywords,
        require_descriptors=require_descriptors,
    )


def match_by_visual(
    construction_year: int,
    balcony_type: Optional[str] = None,
    roof_profile: Optional[str] = None,
    facade_pattern: Optional[str] = None,
    facade_colors: Optional[List[str]] = None,
    has_bay_windows: Optional[bool] = None,
    has_corner_windows: Optional[bool] = None,
    plan_shape: Optional[str] = None,
) -> List[MatchScore]:
    """
    Visual-based matching for street view image analysis.

    Useful for AI/ML models analyzing building images.
    """
    return match_by_descriptors(
        construction_year=construction_year,
        balcony_type=balcony_type,
        roof_profile=roof_profile,
        facade_pattern=facade_pattern,
        facade_colors=facade_colors,
        has_bay_windows=has_bay_windows,
        has_corner_windows=has_corner_windows,
        plan_shape=plan_shape,
        require_descriptors=True,
    )


# Print summary when run directly
if __name__ == "__main__":
    print(get_archetype_summary())
