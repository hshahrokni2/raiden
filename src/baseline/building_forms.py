"""
Swedish Multi-Family Building Forms and Construction Types.

Comprehensive classification of Swedish residential buildings for
accurate energy modeling. Based on:
- Boverket building classification
- TABULA/EPISCOPE Swedish typology
- Swedish Statistics (SCB) building categories

Building forms significantly impact:
- Surface-to-volume ratio (compactness)
- Thermal bridging patterns
- Ventilation system requirements
- Solar exposure per orientation
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum


class BuildingForm(Enum):
    """
    Swedish multi-family building forms (hustyper).

    Each form has distinct geometric and energy characteristics.
    """
    # Slab blocks (long, narrow)
    LAMELLHUS = "lamellhus"          # 3-4 stories, common post-1945
    SKIVHUS = "skivhus"              # Large slab, 8+ stories, miljonprogrammet

    # Point blocks (compact footprint)
    PUNKTHUS = "punkthus"            # Tower, 8+ stories
    STJARNHUS = "stjarnhus"          # Star-shaped, 3+ wings

    # Access types
    LOFTGANGSHUS = "loftgangshus"    # Gallery access (external corridors)
    TRAPPHUS = "trapphus"            # Stairwell access (internal)

    # Urban forms
    SLUTET_KVARTER = "slutet_kvarter"  # Closed perimeter block
    OPPET_KVARTER = "oppet_kvarter"    # Open perimeter block
    VINKELBYGGNAD = "vinkelbyggnad"    # L-shaped

    # Row house forms (for completeness)
    RADHUS = "radhus"                # Terraced houses
    KEDJEHUS = "kedjehus"            # Linked houses

    # Generic/unknown
    GENERIC = "generic"


class ConstructionMethod(Enum):
    """Swedish construction methods."""
    MURAD_TEGEL = "murad_tegel"           # Brick masonry (pre-1960)
    MURAD_LÄTTBETONG = "murad_lattbetong" # Light concrete blocks
    BETONGELEMENT = "betongelement"        # Precast concrete panels
    PLATSGJUTEN = "platsgjuten"            # Cast-in-place concrete
    TRASTOMME = "trastomme"                # Timber frame
    STÅLSTOMME = "stalstomme"              # Steel frame
    PREFAB_TRA = "prefab_tra"              # Prefab timber (modern)
    CLT = "clt"                            # Cross-laminated timber


class FacadeMaterial(Enum):
    """Facade cladding materials."""
    TEGEL = "tegel"              # Brick
    PUTS = "puts"                # Render/stucco
    BETONG = "betong"            # Exposed concrete
    SKIVFASAD = "skivfasad"      # Panel cladding (eternit, etc.)
    TRAPANEL = "trapanel"        # Wood panels
    GLAS = "glas"                # Curtain wall
    PLÅT = "plat"                # Metal cladding


@dataclass
class BuildingFormProperties:
    """
    Geometric and energy properties specific to building form.

    These properties modify the base archetype values.
    """
    form: BuildingForm
    name_sv: str
    name_en: str
    description: str

    # Typical geometry
    typical_stories_min: int
    typical_stories_max: int
    typical_width_m: float         # Building depth
    typical_length_m: float        # Building length (0 = varies)
    typical_apartments_per_floor: int

    # Compactness (surface-to-volume ratio relative to cube)
    # Lower = more compact = less heat loss
    compactness_factor: float      # 1.0 = cube, 1.5 = typical, 2.0 = very spread

    # Thermal bridge multiplier (vs reference building)
    thermal_bridge_factor: float   # 1.0 = reference, 1.2 = more bridges

    # WWR by orientation (N, E, S, W)
    typical_wwr_by_orientation: Dict[str, float] = field(default_factory=dict)

    # Common construction methods for this form
    common_construction: List[ConstructionMethod] = field(default_factory=list)

    # Common issues for this form
    common_issues: List[str] = field(default_factory=list)

    # Typical era range
    common_era_start: int = 1900
    common_era_end: int = 2030


# =============================================================================
# SWEDISH BUILDING FORM DATABASE
# =============================================================================

BUILDING_FORMS: Dict[BuildingForm, BuildingFormProperties] = {

    BuildingForm.LAMELLHUS: BuildingFormProperties(
        form=BuildingForm.LAMELLHUS,
        name_sv="Lamellhus",
        name_en="Slab Block (narrow)",
        description="Long, narrow slab buildings, typically 3-4 stories. Very common in Swedish suburbs built 1945-1975.",
        typical_stories_min=3,
        typical_stories_max=5,
        typical_width_m=11,
        typical_length_m=50,
        typical_apartments_per_floor=6,
        compactness_factor=1.4,
        thermal_bridge_factor=1.1,
        typical_wwr_by_orientation={"N": 0.15, "E": 0.20, "S": 0.25, "W": 0.20},
        common_construction=[
            ConstructionMethod.MURAD_TEGEL,
            ConstructionMethod.BETONGELEMENT,
        ],
        common_issues=[
            "End walls often poorly insulated",
            "Thermal bridges at balconies",
            "Stairwell ventilation losses",
        ],
        common_era_start=1945,
        common_era_end=1985,
    ),

    BuildingForm.SKIVHUS: BuildingFormProperties(
        form=BuildingForm.SKIVHUS,
        name_sv="Skivhus",
        name_en="Large Slab Block",
        description="Wide, tall slab buildings, typically 8+ stories. Common in miljonprogrammet areas.",
        typical_stories_min=8,
        typical_stories_max=16,
        typical_width_m=15,
        typical_length_m=80,
        typical_apartments_per_floor=12,
        compactness_factor=1.2,  # More compact due to size
        thermal_bridge_factor=1.15,
        typical_wwr_by_orientation={"N": 0.20, "E": 0.22, "S": 0.25, "W": 0.22},
        common_construction=[
            ConstructionMethod.BETONGELEMENT,
            ConstructionMethod.PLATSGJUTEN,
        ],
        common_issues=[
            "Facade element joints",
            "Window frame thermal bridges",
            "High wind exposure",
            "Elevator shaft heat losses",
        ],
        common_era_start=1965,
        common_era_end=1975,
    ),

    BuildingForm.PUNKTHUS: BuildingFormProperties(
        form=BuildingForm.PUNKTHUS,
        name_sv="Punkthus",
        name_en="Point Block (Tower)",
        description="Compact tower buildings with central core. Good compactness ratio.",
        typical_stories_min=8,
        typical_stories_max=20,
        typical_width_m=18,
        typical_length_m=18,
        typical_apartments_per_floor=4,
        compactness_factor=1.1,  # Very compact
        thermal_bridge_factor=1.0,
        typical_wwr_by_orientation={"N": 0.18, "E": 0.22, "S": 0.25, "W": 0.22},
        common_construction=[
            ConstructionMethod.BETONGELEMENT,
            ConstructionMethod.PLATSGJUTEN,
        ],
        common_issues=[
            "High wind exposure at upper floors",
            "Central core ventilation",
            "Equal solar exposure all sides",
        ],
        common_era_start=1960,
        common_era_end=2020,
    ),

    BuildingForm.STJARNHUS: BuildingFormProperties(
        form=BuildingForm.STJARNHUS,
        name_sv="Stjärnhus",
        name_en="Star Building",
        description="Star-shaped buildings with 3+ wings. Common in 1950s-60s.",
        typical_stories_min=4,
        typical_stories_max=12,
        typical_width_m=12,
        typical_length_m=0,  # Varies by wing count
        typical_apartments_per_floor=6,
        compactness_factor=1.5,  # Less compact due to wings
        thermal_bridge_factor=1.25,  # More corners
        typical_wwr_by_orientation={"N": 0.18, "E": 0.20, "S": 0.22, "W": 0.20},
        common_construction=[
            ConstructionMethod.MURAD_TEGEL,
            ConstructionMethod.BETONGELEMENT,
        ],
        common_issues=[
            "Many external corners",
            "Complex roof geometry",
            "Wind turbulence between wings",
        ],
        common_era_start=1950,
        common_era_end=1970,
    ),

    BuildingForm.LOFTGANGSHUS: BuildingFormProperties(
        form=BuildingForm.LOFTGANGSHUS,
        name_sv="Loftgångshus",
        name_en="Gallery Access Building",
        description="Buildings with external access galleries (corridors). Single-loaded.",
        typical_stories_min=3,
        typical_stories_max=8,
        typical_width_m=8,
        typical_length_m=60,
        typical_apartments_per_floor=10,
        compactness_factor=1.6,  # Gallery adds surface area
        thermal_bridge_factor=1.3,  # Gallery connections
        typical_wwr_by_orientation={"N": 0.10, "E": 0.15, "S": 0.30, "W": 0.15},  # Gallery side has less
        common_construction=[
            ConstructionMethod.BETONGELEMENT,
        ],
        common_issues=[
            "Gallery thermal bridges",
            "Single-sided apartments (through-ventilation issues)",
            "Exposed gallery in winter",
            "Privacy on gallery side",
        ],
        common_era_start=1965,
        common_era_end=1975,
    ),

    BuildingForm.SLUTET_KVARTER: BuildingFormProperties(
        form=BuildingForm.SLUTET_KVARTER,
        name_sv="Slutet kvarter (Stadskvarter)",
        name_en="Closed Perimeter Block",
        description="Traditional urban blocks with courtyard. Common in city centers.",
        typical_stories_min=4,
        typical_stories_max=7,
        typical_width_m=15,
        typical_length_m=0,  # Follows plot boundary
        typical_apartments_per_floor=8,
        compactness_factor=1.3,
        thermal_bridge_factor=1.0,  # Party walls reduce losses
        typical_wwr_by_orientation={"N": 0.15, "E": 0.18, "S": 0.20, "W": 0.18},
        common_construction=[
            ConstructionMethod.MURAD_TEGEL,
            ConstructionMethod.PLATSGJUTEN,
        ],
        common_issues=[
            "Mixed envelope conditions (street vs courtyard)",
            "Heritage constraints",
            "Limited roof access for solar",
        ],
        common_era_start=1880,
        common_era_end=1940,
    ),

    BuildingForm.VINKELBYGGNAD: BuildingFormProperties(
        form=BuildingForm.VINKELBYGGNAD,
        name_sv="Vinkelbyggnad",
        name_en="L-Shaped Building",
        description="L-shaped buildings, often oriented to optimize solar exposure.",
        typical_stories_min=3,
        typical_stories_max=6,
        typical_width_m=12,
        typical_length_m=40,
        typical_apartments_per_floor=6,
        compactness_factor=1.35,
        thermal_bridge_factor=1.15,
        typical_wwr_by_orientation={"N": 0.15, "E": 0.18, "S": 0.25, "W": 0.18},
        common_construction=[
            ConstructionMethod.MURAD_TEGEL,
            ConstructionMethod.BETONGELEMENT,
        ],
        common_issues=[
            "Corner thermal bridge",
            "Shading between wings",
        ],
        common_era_start=1945,
        common_era_end=1990,
    ),

    BuildingForm.RADHUS: BuildingFormProperties(
        form=BuildingForm.RADHUS,
        name_sv="Radhus",
        name_en="Row House / Terraced House",
        description="Terraced houses sharing party walls. Individual entrances.",
        typical_stories_min=2,
        typical_stories_max=3,
        typical_width_m=6,
        typical_length_m=12,
        typical_apartments_per_floor=1,  # Per unit
        compactness_factor=1.5,
        thermal_bridge_factor=1.0,  # Party walls help
        typical_wwr_by_orientation={"N": 0.12, "E": 0.15, "S": 0.25, "W": 0.15},
        common_construction=[
            ConstructionMethod.MURAD_TEGEL,
            ConstructionMethod.TRASTOMME,
            ConstructionMethod.PREFAB_TRA,
        ],
        common_issues=[
            "End units have more heat loss",
            "Basement/crawl space insulation",
            "Attic insulation continuity",
        ],
        common_era_start=1960,
        common_era_end=2030,
    ),

    BuildingForm.GENERIC: BuildingFormProperties(
        form=BuildingForm.GENERIC,
        name_sv="Generisk flerbostadshus",
        name_en="Generic Multi-Family",
        description="Generic multi-family when specific form is unknown.",
        typical_stories_min=3,
        typical_stories_max=6,
        typical_width_m=12,
        typical_length_m=30,
        typical_apartments_per_floor=4,
        compactness_factor=1.3,
        thermal_bridge_factor=1.1,
        typical_wwr_by_orientation={"N": 0.18, "E": 0.20, "S": 0.22, "W": 0.20},
        common_construction=[
            ConstructionMethod.BETONGELEMENT,
            ConstructionMethod.MURAD_TEGEL,
        ],
        common_issues=[],
        common_era_start=1900,
        common_era_end=2030,
    ),
}


# =============================================================================
# CONSTRUCTION METHOD PROPERTIES
# =============================================================================

@dataclass
class ConstructionProperties:
    """Properties of construction methods."""
    method: ConstructionMethod
    name_sv: str
    name_en: str

    # Typical thermal mass (kJ/m²K of floor area)
    thermal_mass_kj_m2k: float

    # Wall thickness range (mm)
    typical_wall_thickness_mm: Tuple[int, int]

    # Airtightness (typical ACH at 50Pa)
    typical_ach50: float

    # Common in which eras
    common_era_start: int
    common_era_end: int


CONSTRUCTION_METHODS: Dict[ConstructionMethod, ConstructionProperties] = {

    ConstructionMethod.MURAD_TEGEL: ConstructionProperties(
        method=ConstructionMethod.MURAD_TEGEL,
        name_sv="Murad tegelvägg",
        name_en="Brick Masonry",
        thermal_mass_kj_m2k=400,  # High thermal mass
        typical_wall_thickness_mm=(380, 510),
        typical_ach50=4.0,  # Leaky
        common_era_start=1900,
        common_era_end=1965,
    ),

    ConstructionMethod.BETONGELEMENT: ConstructionProperties(
        method=ConstructionMethod.BETONGELEMENT,
        name_sv="Betongelement (sandwichelement)",
        name_en="Precast Concrete Panels",
        thermal_mass_kj_m2k=350,
        typical_wall_thickness_mm=(200, 300),
        typical_ach50=2.5,
        common_era_start=1960,
        common_era_end=1990,
    ),

    ConstructionMethod.PLATSGJUTEN: ConstructionProperties(
        method=ConstructionMethod.PLATSGJUTEN,
        name_sv="Platsgjuten betong",
        name_en="Cast-in-Place Concrete",
        thermal_mass_kj_m2k=380,
        typical_wall_thickness_mm=(200, 350),
        typical_ach50=2.0,
        common_era_start=1950,
        common_era_end=2030,
    ),

    ConstructionMethod.TRASTOMME: ConstructionProperties(
        method=ConstructionMethod.TRASTOMME,
        name_sv="Träregelstomme",
        name_en="Timber Frame",
        thermal_mass_kj_m2k=150,  # Low thermal mass
        typical_wall_thickness_mm=(150, 250),
        typical_ach50=3.0,
        common_era_start=1970,
        common_era_end=2030,
    ),

    ConstructionMethod.CLT: ConstructionProperties(
        method=ConstructionMethod.CLT,
        name_sv="KL-trä (korslimmat trä)",
        name_en="Cross-Laminated Timber",
        thermal_mass_kj_m2k=250,  # Moderate thermal mass
        typical_wall_thickness_mm=(100, 200),
        typical_ach50=1.0,  # Very tight
        common_era_start=2010,
        common_era_end=2030,
    ),

    ConstructionMethod.PREFAB_TRA: ConstructionProperties(
        method=ConstructionMethod.PREFAB_TRA,
        name_sv="Prefabricerat trä",
        name_en="Prefab Timber",
        thermal_mass_kj_m2k=180,
        typical_wall_thickness_mm=(200, 350),
        typical_ach50=1.5,
        common_era_start=2000,
        common_era_end=2030,
    ),
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_form_properties(form: BuildingForm) -> BuildingFormProperties:
    """Get properties for a building form."""
    return BUILDING_FORMS.get(form, BUILDING_FORMS[BuildingForm.GENERIC])


def detect_building_form(
    stories: int,
    width_m: float,
    length_m: float,
    construction_year: int,
    has_gallery: bool = False,
) -> BuildingForm:
    """
    Detect building form from basic geometry.

    Args:
        stories: Number of stories
        width_m: Building width/depth
        length_m: Building length
        construction_year: Year built
        has_gallery: If external gallery access is known

    Returns:
        Most likely BuildingForm
    """
    aspect_ratio = length_m / width_m if width_m > 0 else 1

    # Gallery access building
    if has_gallery:
        return BuildingForm.LOFTGANGSHUS

    # Point block (compact, tall)
    if stories >= 8 and aspect_ratio < 1.3:
        return BuildingForm.PUNKTHUS

    # Large slab (tall, long)
    if stories >= 8 and aspect_ratio > 3:
        return BuildingForm.SKIVHUS

    # Narrow slab (medium height, long)
    if 3 <= stories <= 5 and aspect_ratio > 3:
        return BuildingForm.LAMELLHUS

    # Low-rise row house
    if stories <= 3 and width_m < 8:
        return BuildingForm.RADHUS

    # Pre-war city block
    if construction_year < 1940:
        return BuildingForm.SLUTET_KVARTER

    return BuildingForm.GENERIC


def get_form_modifier(
    form: BuildingForm,
    base_u_value: float,
    component: str = "wall",
) -> float:
    """
    Get modified U-value based on building form.

    Form affects heat loss through thermal bridges and geometry.

    Args:
        form: Building form
        base_u_value: Base U-value from archetype
        component: Building component ('wall', 'roof', 'window')

    Returns:
        Modified U-value
    """
    props = get_form_properties(form)

    # Thermal bridge factor applies mainly to walls
    if component == "wall":
        return base_u_value * props.thermal_bridge_factor

    # Compactness affects overall but less directly
    return base_u_value


def estimate_surface_area(
    form: BuildingForm,
    atemp_m2: float,
    stories: int,
) -> Dict[str, float]:
    """
    Estimate building surface areas by orientation.

    Args:
        form: Building form
        atemp_m2: Heated floor area
        stories: Number of stories

    Returns:
        Dict with wall_N, wall_E, wall_S, wall_W, roof, floor areas
    """
    props = get_form_properties(form)

    floor_area_per_story = atemp_m2 / stories
    floor_height = 2.7  # Typical

    # Estimate footprint dimensions
    if props.typical_length_m > 0:
        # Use typical proportions
        length = props.typical_length_m
        width = floor_area_per_story / length
    else:
        # Assume square-ish
        width = (floor_area_per_story ** 0.5)
        length = width

    wall_height = stories * floor_height

    # Wall areas by orientation
    # Assuming long side faces E-W
    wall_n = length * wall_height
    wall_s = length * wall_height
    wall_e = width * wall_height
    wall_w = width * wall_height

    return {
        "wall_N": wall_n,
        "wall_E": wall_e,
        "wall_S": wall_s,
        "wall_W": wall_w,
        "roof": floor_area_per_story,
        "floor": floor_area_per_story,
    }
