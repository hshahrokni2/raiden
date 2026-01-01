"""
Baseline Module - Auto-generate calibrated building energy models.

Takes building data from public sources and creates:
- Archetype-based initial model (by era + form)
- Auto-calibrated to energy declaration
- Ready for ECM analysis

Key principle: Maximum accuracy from minimum owner input.

Swedish Building Eras (TABULA/EPISCOPE):
- PRE_1930: Traditional brick (U_wall ~1.2)
- 1930-1945: Functionalism/Funkis (U_wall ~1.0)
- 1946-1960: Folkhemmet (U_wall ~0.65)
- 1961-1975: Miljonprogrammet (U_wall ~0.45)
- 1976-1985: Post oil crisis (U_wall ~0.28)
- 1986-1995: FTX standard (U_wall ~0.20)
- 1996-2010: Low-energy transition (U_wall ~0.17)
- 2011+: Near-zero energy (U_wall ~0.12)

Building Forms:
- LAMELLHUS: Slab block (3-4 stories)
- SKIVHUS: Large slab (8+ stories, miljonprogrammet)
- PUNKTHUS: Point block tower
- STJARNHUS: Star-shaped
- LOFTGANGSHUS: Gallery access
- SLUTET_KVARTER: Closed perimeter block

Sources: TABULA/EPISCOPE, Boverket BETSI, SBN 1967/1975/1980, BBR
"""

from .archetypes import (
    SwedishArchetype, ArchetypeMatcher, BuildingType,
    MatchResult, EnvelopeProperties, HVACProperties,
    HeatingSystem, VentilationType,
)
from .building_forms import (
    BuildingForm, BuildingFormProperties, ConstructionMethod,
    ConstructionProperties, FacadeMaterial,
    BUILDING_FORMS, CONSTRUCTION_METHODS,
    get_form_properties, detect_building_form,
    get_form_modifier, estimate_surface_area,
)
from .archetypes_detailed import (
    # Core archetype dataclasses
    DetailedArchetype,
    WallConstruction,
    WindowConstruction,
    RoofConstruction,
    FloorConstruction,
    ArchetypeDescriptors,
    # Enums
    BuildingEra,
    WallConstructionType,
    WindowType,
    VentilationType as DetailedVentilationType,
    HeatingSystemType,
    # Descriptor enums (for matching)
    BalconyType,
    RoofProfile,
    FacadePattern,
    PlanShape,
    UrbanSetting,
    OwnershipType,
    EnergyCertification,
    # High-performance archetypes (passive house, plus-energy)
    SWEDISH_HIGH_PERFORMANCE_ARCHETYPES,
    # Historical archetypes (pre-1930)
    SWEDISH_HISTORICAL_ARCHETYPES,
    # Special form archetypes (Stockholm-specific, unique forms)
    SWEDISH_SPECIAL_FORM_ARCHETYPES,
    # Multi-family archetypes
    SWEDISH_MFH_ARCHETYPES,
    # Single-family archetypes
    SWEDISH_SFH_ARCHETYPES,
    # Terraced house archetypes
    SWEDISH_TERRACED_ARCHETYPES,
    # Miljonprogrammet sub-types
    MiljonprogrammetSubtype,
    MILJONPROGRAMMET_SUBTYPES,
    # Climate zones
    SwedishClimateZone,
    SWEDISH_CLIMATE_ZONES,
    get_climate_zone_for_region,
    get_climate_zone_for_city,
    # Helper functions
    get_archetype_by_year,
    get_u_value_for_year,
    get_heating_kwh_by_year,
    list_archetypes as list_detailed_archetypes,
    get_archetype_summary,
    get_all_archetypes,
    get_archetype,
    get_archetype_for_building,
    # Smart selection functions
    get_smart_archetype,
    classify_building_type,
    # Descriptor-based matching
    MatchScore,
    match_by_descriptors,
    match_by_keywords,
    match_by_visual,
)
from .generator import BaselineGenerator, BaselineModel, generate_baseline
from .generator_v2 import (
    GeomEppyGenerator,
    analyze_footprint,
    generate_from_footprint,
    FloorPlan,
    WallSegment,
    GEOMEPPY_AVAILABLE,
)
from .zone_assignment import (
    FloorZone,
    BuildingZoneLayout,
    assign_zones_to_floors,
    get_zone_layout_summary,
)
from .calibrator import BaselineCalibrator, CalibrationResult, calibrate_baseline
from .archetype_matcher_v2 import (
    ArchetypeMatcherV2,
    ArchetypeMatchResult,
    ScoredCandidate,
    DataSourceScores,
    AIVisualAnalysis,
    WWRAnalysisResult,
    MaterialAnalysisResult,
    PVAnalysisResult,
    ScoringWeights,
    match_archetype_from_context,
    match_archetype_from_building_data,
)
from .llm_archetype_reasoner import (
    LLMArchetypeReasoner,
    LLMReasoningResult,
    RenovationAnalysis,
    VisualEraAnalysis,
    enhance_archetype_match,
)

__all__ = [
    # Original Archetypes (simplified)
    'SwedishArchetype',
    'ArchetypeMatcher',
    'BuildingType',
    'MatchResult',
    'EnvelopeProperties',
    'HVACProperties',
    'HeatingSystem',
    'VentilationType',
    # Detailed Archetypes (comprehensive TABULA-based)
    'DetailedArchetype',
    'ArchetypeDescriptors',
    'SWEDISH_HIGH_PERFORMANCE_ARCHETYPES',
    'SWEDISH_HISTORICAL_ARCHETYPES',
    'SWEDISH_SPECIAL_FORM_ARCHETYPES',
    'SWEDISH_MFH_ARCHETYPES',
    'SWEDISH_SFH_ARCHETYPES',
    'SWEDISH_TERRACED_ARCHETYPES',
    'BuildingEra',
    'WallConstructionType',
    'WindowType',
    'DetailedVentilationType',
    'HeatingSystemType',
    'WallConstruction',
    'WindowConstruction',
    'RoofConstruction',
    'FloorConstruction',
    # Descriptor enums
    'BalconyType',
    'RoofProfile',
    'FacadePattern',
    'PlanShape',
    'UrbanSetting',
    'OwnershipType',
    'EnergyCertification',
    'get_archetype_by_year',
    'get_u_value_for_year',
    'get_heating_kwh_by_year',
    'list_detailed_archetypes',
    'get_archetype_summary',
    'get_all_archetypes',
    'get_archetype',
    'get_archetype_for_building',
    # Smart selection functions
    'get_smart_archetype',
    'classify_building_type',
    # Descriptor-based matching
    'MatchScore',
    'match_by_descriptors',
    'match_by_keywords',
    'match_by_visual',
    # Miljonprogrammet sub-types
    'MiljonprogrammetSubtype',
    'MILJONPROGRAMMET_SUBTYPES',
    # Climate zones
    'SwedishClimateZone',
    'SWEDISH_CLIMATE_ZONES',
    'get_climate_zone_for_region',
    'get_climate_zone_for_city',
    # Building Forms
    'BuildingForm',
    'BuildingFormProperties',
    'ConstructionMethod',
    'ConstructionProperties',
    'FacadeMaterial',
    'BUILDING_FORMS',
    'CONSTRUCTION_METHODS',
    'get_form_properties',
    'detect_building_form',
    'get_form_modifier',
    'estimate_surface_area',
    # Generator
    'BaselineGenerator',
    'BaselineModel',
    'generate_baseline',
    # Generator V2 (GeomEppy)
    'GeomEppyGenerator',
    'analyze_footprint',
    'generate_from_footprint',
    'FloorPlan',
    'WallSegment',
    'GEOMEPPY_AVAILABLE',
    # Zone Assignment (Multi-zone floor-based)
    'FloorZone',
    'BuildingZoneLayout',
    'assign_zones_to_floors',
    'get_zone_layout_summary',
    # Calibrator
    'BaselineCalibrator',
    'CalibrationResult',
    'calibrate_baseline',
    # V2 Archetype Matcher (integrated with data pipeline)
    'ArchetypeMatcherV2',
    'ArchetypeMatchResult',
    'ScoredCandidate',
    'DataSourceScores',
    'AIVisualAnalysis',
    'WWRAnalysisResult',
    'MaterialAnalysisResult',
    'PVAnalysisResult',
    'ScoringWeights',
    'match_archetype_from_context',
    'match_archetype_from_building_data',
    # LLM-Enhanced Archetype Reasoner
    'LLMArchetypeReasoner',
    'LLMReasoningResult',
    'RenovationAnalysis',
    'VisualEraAnalysis',
    'enhance_archetype_match',
]
