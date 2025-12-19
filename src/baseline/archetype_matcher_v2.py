"""
Hybrid Archetype Matcher V2 - Integrated with Raiden Data Pipeline.

This matcher uses REAL DATA from:
1. Energy declarations (construction year, energy class, ventilation, heating system)
2. Mapillary facade images (via existing FacadeImageFetcher)
3. OSM/Overture building footprints (form detection)
4. Google Solar API (roof characteristics)

It does NOT infer things that are already available in the data sources.

Usage:
    from src.baseline.archetype_matcher_v2 import ArchetypeMatcherV2
    from src.core.building_context import EnhancedBuildingContext

    # Match from EnhancedBuildingContext (has all real data)
    matcher = ArchetypeMatcherV2()
    result = matcher.match_from_context(building_context)

    # Or match from AddressPipeline BuildingData
    result = matcher.match_from_building_data(building_data, facade_images)
"""

from __future__ import annotations

import base64
import json
import logging
import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

from .archetypes_detailed import (
    DetailedArchetype,
    ArchetypeDescriptors,
    get_all_archetypes,
    BuildingEra,
    PlanShape,
    BalconyType,
    RoofProfile,
    FacadePattern,
    UrbanSetting,
    EnergyCertification,
    WallConstructionType,
)

if TYPE_CHECKING:
    from ..core.building_context import EnhancedBuildingContext
    from ..core.address_pipeline import BuildingData
    from ..ingest.image_fetcher import FacadeImage
    from ..ingest.energidek_parser import EnergyDeclarationData
    from ..core.models import WindowToWallRatio, FacadeMaterial
    from ..ai.wwr_detector import WWRDetector
    from ..ai.material_classifier import MaterialClassifier
    from ..analysis.roof_analyzer import RoofAnalyzer, RoofAnalysis
    from ..geometry.building_geometry import BuildingGeometry

logger = logging.getLogger(__name__)


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class DataSourceScores:
    """Scores from each data source."""
    energy_declaration: float = 0.0  # From actual declaration data
    osm_geometry: float = 0.0        # From footprint/form detection
    mapillary_visual: float = 0.0    # From facade image analysis (material + patterns)
    wwr_analysis: float = 0.0        # From WWR detection (AI window detection)
    google_solar: float = 0.0        # From roof characteristics / PV potential
    building_geometry: float = 0.0   # From building geometry calculations
    location: float = 0.0            # From address/neighborhood

    @property
    def total(self) -> float:
        return (
            self.energy_declaration +
            self.osm_geometry +
            self.mapillary_visual +
            self.wwr_analysis +
            self.google_solar +
            self.building_geometry +
            self.location
        )


@dataclass
class ScoredCandidate:
    """Archetype candidate with scoring details."""
    archetype: DetailedArchetype
    score: float
    source_scores: DataSourceScores = field(default_factory=DataSourceScores)
    match_reasons: List[str] = field(default_factory=list)
    mismatch_reasons: List[str] = field(default_factory=list)


@dataclass
class WWRAnalysisResult:
    """Result from Window-to-Wall Ratio analysis."""
    north: float = 0.0
    south: float = 0.0
    east: float = 0.0
    west: float = 0.0
    average: float = 0.0
    source: str = "unknown"  # "ai_opencv", "ai_sam", "era_estimation"
    confidence: float = 0.0


@dataclass
class MaterialAnalysisResult:
    """Result from facade material classification."""
    primary_material: Optional[str] = None
    secondary_material: Optional[str] = None
    material_scores: Dict[str, float] = field(default_factory=dict)
    source: str = "unknown"  # "ai_dino", "ai_heuristic", "osm"
    confidence: float = 0.0


@dataclass
class PVAnalysisResult:
    """Result from PV/roof analysis."""
    roof_type: str = "unknown"  # flat, pitched, gabled, hipped
    total_roof_area_m2: float = 0.0
    usable_pv_area_m2: float = 0.0
    max_capacity_kwp: float = 0.0
    annual_generation_kwh: float = 0.0
    existing_pv_kwp: float = 0.0  # Already installed PV
    source: str = "unknown"  # "google_solar", "osm", "estimation"
    confidence: float = 0.0


@dataclass
class AIVisualAnalysis:
    """Result from AI visual analysis of facade images."""
    facade_material: Optional[str] = None
    facade_pattern: Optional[FacadePattern] = None
    balcony_type: Optional[BalconyType] = None
    roof_profile: Optional[RoofProfile] = None
    facade_colors: List[str] = field(default_factory=list)
    window_pattern: str = ""  # "portrait", "landscape", "square"
    has_bay_windows: bool = False
    has_ornament: bool = False
    estimated_era: Optional[str] = None
    confidence: float = 0.0
    evidence: List[str] = field(default_factory=list)

    # Integrated analysis results from dedicated AI modules
    wwr_result: Optional[WWRAnalysisResult] = None
    material_result: Optional[MaterialAnalysisResult] = None
    pv_result: Optional[PVAnalysisResult] = None


@dataclass
class ArchetypeMatchResult:
    """Complete result of archetype matching."""
    archetype: DetailedArchetype
    confidence: float = 0.0  # 0-1

    # Data sources used
    data_sources_used: List[str] = field(default_factory=list)

    # Scoring breakdown
    source_scores: DataSourceScores = field(default_factory=DataSourceScores)

    # Explanation
    match_reasons: List[str] = field(default_factory=list)
    mismatch_reasons: List[str] = field(default_factory=list)

    # Visual analysis (if Mapillary images were used)
    visual_analysis: Optional[AIVisualAnalysis] = None

    # Alternative candidates
    alternatives: List[Tuple[DetailedArchetype, float]] = field(default_factory=list)

    # Calibration hints from archetype descriptors
    calibration_hints: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# SCORING WEIGHTS
# =============================================================================

class ScoringWeights:
    """Weights for different data sources."""

    # Energy Declaration (highest weight - most reliable)
    DECLARATION_YEAR_MATCH = 30
    DECLARATION_ENERGY_CLASS = 10
    DECLARATION_VENTILATION = 8
    DECLARATION_HEATING = 7

    # OSM/Overture Geometry
    GEOMETRY_FORM_MATCH = 12
    GEOMETRY_FLOORS_MATCH = 5

    # AI WWR Analysis (from wwr_detector.py)
    WWR_MATCH = 10
    WWR_ORIENTATION_MATCH = 5  # South-facing WWR typical for era

    # AI Material Classification (from material_classifier.py)
    MATERIAL_MATCH = 12
    MATERIAL_CONFIDENCE_BONUS = 3  # High confidence from AI

    # Google Solar / PV Analysis (from roof_analyzer.py)
    PV_ROOF_TYPE_MATCH = 5
    PV_EXISTING_MATCH = 5  # Has existing PV = modern building

    # Building Geometry (from building_geometry.py)
    GEOMETRY_AREA_MATCH = 5
    GEOMETRY_COMPACTNESS = 3  # Shape factor typical for era

    # Mapillary Visual (AI analysis)
    VISUAL_MATERIAL_MATCH = 15
    VISUAL_PATTERN_MATCH = 10
    VISUAL_BALCONY_MATCH = 5
    VISUAL_ERA_MATCH = 10

    # Location
    LOCATION_CITY_MATCH = 5
    LOCATION_NEIGHBORHOOD = 10


# =============================================================================
# ERA MAPPING
# =============================================================================

# Map energy class to expected eras (buildings rarely outperform their era)
ENERGY_CLASS_ERA_EXPECTATIONS = {
    "A": [BuildingEra.NARA_NOLL_2011_PLUS],
    "B": [BuildingEra.NARA_NOLL_2011_PLUS, BuildingEra.LAGENERGI_1996_2010],
    "C": [BuildingEra.LAGENERGI_1996_2010, BuildingEra.MODERN_1986_1995],
    "D": [BuildingEra.MODERN_1986_1995, BuildingEra.ENERGI_1976_1985],
    "E": [BuildingEra.ENERGI_1976_1985, BuildingEra.REKORD_1961_1975],
    "F": [BuildingEra.REKORD_1961_1975, BuildingEra.FOLKHEM_1946_1960],
    "G": [BuildingEra.FOLKHEM_1946_1960, BuildingEra.FUNKIS_1930_1945, BuildingEra.PRE_1930],
}

# Map ventilation types to typical eras
VENTILATION_ERA_MAP = {
    "FTX": [BuildingEra.MODERN_1986_1995, BuildingEra.LAGENERGI_1996_2010, BuildingEra.NARA_NOLL_2011_PLUS],
    "FT": [BuildingEra.ENERGI_1976_1985, BuildingEra.MODERN_1986_1995],
    "F": [BuildingEra.REKORD_1961_1975, BuildingEra.ENERGI_1976_1985],
    "S": [BuildingEra.PRE_1930, BuildingEra.FUNKIS_1930_1945, BuildingEra.FOLKHEM_1946_1960],
}


# =============================================================================
# MAIN MATCHER CLASS
# =============================================================================

class ArchetypeMatcherV2:
    """
    Hybrid archetype matcher integrated with Raiden data pipeline.

    Uses REAL DATA from:
    - Energy declarations (via EnergyDeclarationData)
    - Mapillary images (via FacadeImageFetcher)
    - OSM/Overture (via BuildingDataFetcher)
    - Google Solar API (via RoofAnalyzer)

    Pipeline:
    1. Score from energy declaration (construction year, energy class, ventilation)
    2. Score from geometry (building form from OSM footprint)
    3. Score from visual analysis (Mapillary facade images via AI)
    4. Score from location (city/neighborhood matching)
    5. Combine scores and return best match with confidence
    """

    def __init__(
        self,
        archetypes: Optional[Dict[str, DetailedArchetype]] = None,
        ai_provider: str = "claude",
        anthropic_api_key: Optional[str] = None,
        use_ai_modules: bool = True,
        ai_device: str = "cpu",
        wwr_backend: str = "opencv",
    ):
        """
        Initialize the matcher with integrated AI modules.

        Args:
            archetypes: Custom archetype dictionary (uses default if None)
            ai_provider: AI provider for visual analysis ("claude", "openai")
            anthropic_api_key: API key for Claude (defaults to env var)
            use_ai_modules: Whether to use dedicated AI modules (WWR, Material)
            ai_device: Device for AI inference ("cpu", "cuda", "mps")
            wwr_backend: Backend for WWR detection ("opencv", "sam", "lang_sam")
        """
        self.archetypes = archetypes or get_all_archetypes()
        self.ai_provider = ai_provider
        self.anthropic_api_key = anthropic_api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.use_ai_modules = use_ai_modules
        self.ai_device = ai_device
        self.wwr_backend = wwr_backend

        # Lazy-initialized AI modules
        self._wwr_detector = None
        self._material_classifier = None
        self._roof_analyzer = None

        logger.info(f"ArchetypeMatcherV2 initialized with {len(self.archetypes)} archetypes")
        if use_ai_modules:
            logger.info(f"AI modules enabled: device={ai_device}, wwr_backend={wwr_backend}")

    def _get_wwr_detector(self) -> Optional["WWRDetector"]:
        """Lazy-load WWR detector."""
        if self._wwr_detector is None and self.use_ai_modules:
            try:
                from ..ai.wwr_detector import WWRDetector
                self._wwr_detector = WWRDetector(
                    backend=self.wwr_backend,
                    device=self.ai_device,
                )
                logger.debug("WWRDetector initialized")
            except ImportError as e:
                logger.warning(f"WWRDetector not available: {e}")
        return self._wwr_detector

    def _get_material_classifier(self) -> Optional["MaterialClassifier"]:
        """Lazy-load material classifier."""
        if self._material_classifier is None and self.use_ai_modules:
            try:
                from ..ai.material_classifier import MaterialClassifier
                self._material_classifier = MaterialClassifier(
                    device=self.ai_device,
                )
                logger.debug("MaterialClassifier initialized")
            except ImportError as e:
                logger.warning(f"MaterialClassifier not available: {e}")
        return self._material_classifier

    def _get_roof_analyzer(self) -> Optional["RoofAnalyzer"]:
        """Lazy-load roof analyzer (for Google Solar API)."""
        if self._roof_analyzer is None and self.use_ai_modules:
            try:
                from ..analysis.roof_analyzer import RoofAnalyzer
                self._roof_analyzer = RoofAnalyzer()
                logger.debug("RoofAnalyzer initialized")
            except ImportError as e:
                logger.warning(f"RoofAnalyzer not available: {e}")
        return self._roof_analyzer

    def match_from_context(
        self,
        context: "EnhancedBuildingContext",
        facade_images: Optional[List["FacadeImage"]] = None,
        use_ai_visual: bool = True,
    ) -> ArchetypeMatchResult:
        """
        Match archetype from EnhancedBuildingContext.

        This is the PRIMARY entry point - uses real data from energy declarations.

        Args:
            context: EnhancedBuildingContext with real building data
            facade_images: Optional Mapillary images for visual analysis
            use_ai_visual: Whether to use AI for visual analysis

        Returns:
            ArchetypeMatchResult with matched archetype and confidence
        """
        data_sources = []
        candidates: List[ScoredCandidate] = []

        # Get real data from context
        construction_year = context.construction_year
        energy_class = context.energy_declaration.energy_class if context.energy_declaration else None
        ventilation_type = context.ventilation_type.value if context.ventilation_type else None
        heating_system = context.heating_system.value if context.heating_system else None
        facade_material = context.facade_material
        building_type = context.building_type.value if context.building_type else "multi_family"

        logger.info(f"Matching archetype: year={construction_year}, energy={energy_class}, "
                   f"vent={ventilation_type}, facade={facade_material}")

        # Score all archetypes
        for arch_id, archetype in self.archetypes.items():
            # Filter by building type first
            if not self._matches_building_type(archetype, building_type):
                continue

            source_scores = DataSourceScores()
            match_reasons = []
            mismatch_reasons = []

            # 1. Score from energy declaration (REAL DATA)
            if context.energy_declaration:
                data_sources.append("energy_declaration")
                decl_score, reasons, mismatches = self._score_from_declaration(
                    archetype,
                    construction_year=construction_year,
                    energy_class=energy_class,
                    ventilation_type=ventilation_type,
                    heating_system=heating_system,
                )
                source_scores.energy_declaration = decl_score
                match_reasons.extend(reasons)
                mismatch_reasons.extend(mismatches)

            # 2. Score from geometry (OSM/Overture)
            if context.floors > 0:
                data_sources.append("osm_geometry")
                geo_score, reasons = self._score_from_geometry(
                    archetype,
                    floors=context.floors,
                    wall_area_m2=context.wall_area_m2,
                    window_to_wall_ratio=context.window_to_wall_ratio,
                )
                source_scores.osm_geometry = geo_score
                match_reasons.extend(reasons)

            # 3. Score from location
            if context.address:
                data_sources.append("location")
                loc_score, reasons = self._score_from_location(
                    archetype,
                    address=context.address,
                )
                source_scores.location = loc_score
                match_reasons.extend(reasons)

            candidates.append(ScoredCandidate(
                archetype=archetype,
                score=source_scores.total,
                source_scores=source_scores,
                match_reasons=match_reasons,
                mismatch_reasons=mismatch_reasons,
            ))

        # Sort by score
        candidates.sort(key=lambda c: c.score, reverse=True)

        if not candidates:
            return self._default_result(building_type, construction_year)

        # 4. Visual analysis with Mapillary (if available and enabled)
        visual_analysis = None
        if use_ai_visual and facade_images and self.anthropic_api_key:
            data_sources.append("mapillary_visual")
            visual_analysis = self._analyze_facade_images(facade_images, candidates[:5])

            if visual_analysis and visual_analysis.confidence > 0.5:
                # Re-score top candidates with visual data
                for candidate in candidates[:5]:
                    visual_score = self._score_from_visual(
                        candidate.archetype,
                        visual_analysis,
                    )
                    candidate.source_scores.mapillary_visual = visual_score
                    candidate.score = candidate.source_scores.total

                # Re-sort
                candidates.sort(key=lambda c: c.score, reverse=True)

        # Build result
        winner = candidates[0]
        max_possible = 100  # Maximum possible score

        return ArchetypeMatchResult(
            archetype=winner.archetype,
            confidence=min(winner.score / max_possible, 1.0),
            data_sources_used=list(set(data_sources)),
            source_scores=winner.source_scores,
            match_reasons=winner.match_reasons,
            mismatch_reasons=winner.mismatch_reasons,
            visual_analysis=visual_analysis,
            alternatives=[(c.archetype, c.score / max_possible) for c in candidates[1:4]],
            calibration_hints=self._get_calibration_hints(winner.archetype, winner),
        )

    def match_from_building_data(
        self,
        building_data: "BuildingData",
        use_ai_visual: bool = True,
    ) -> ArchetypeMatchResult:
        """
        Match archetype from AddressPipeline BuildingData.

        This is for when you don't have an energy declaration but have
        fetched data from the address pipeline.

        Args:
            building_data: BuildingData from AddressPipeline
            use_ai_visual: Whether to use AI for visual analysis

        Returns:
            ArchetypeMatchResult
        """
        data_sources = building_data.data_sources.copy()
        candidates: List[ScoredCandidate] = []

        # Extract data
        construction_year = building_data.construction_year
        facade_material = building_data.facade_material
        building_type = building_data.building_type
        building_form = building_data.building_form
        num_floors = building_data.num_floors
        energy_class = building_data.energy_class if building_data.energy_class != "Unknown" else None
        address = building_data.address

        # Extract ventilation/heating (from Sweden Buildings GeoJSON)
        ventilation_type = "FTX" if building_data.has_ftx else None
        heating_system = building_data.heating_system if hasattr(building_data, 'heating_system') else None
        has_heat_pump = building_data.has_heat_pump if hasattr(building_data, 'has_heat_pump') else False
        has_solar = building_data.has_solar if hasattr(building_data, 'has_solar') else False
        wwr = building_data.wwr if hasattr(building_data, 'wwr') else 0.20

        logger.info(f"Matching from BuildingData: year={construction_year}, "
                   f"facade={facade_material}, form={building_form}, "
                   f"FTX={ventilation_type}, heating={heating_system}, WWR={wwr:.0%}")

        # Score all archetypes
        for arch_id, archetype in self.archetypes.items():
            if not self._matches_building_type(archetype, building_type):
                continue

            source_scores = DataSourceScores()
            match_reasons = []
            mismatch_reasons = []

            # Score from available data (year + energy class + ventilation)
            decl_score, reasons, mismatches = self._score_from_declaration(
                archetype,
                construction_year,
                energy_class,
                ventilation_type,
                heating_system,
            )
            source_scores.energy_declaration = decl_score
            match_reasons.extend(reasons)
            mismatch_reasons.extend(mismatches)

            # Score from geometry/form
            if building_form and building_form != "generic":
                form_score, reasons = self._score_form_match(archetype, building_form)
                source_scores.osm_geometry = form_score
                match_reasons.extend(reasons)

            # Score floors
            if num_floors > 0:
                floor_score, reasons = self._score_floors_match(archetype, num_floors)
                source_scores.osm_geometry += floor_score
                match_reasons.extend(reasons)

            # Score from facade material (from Mapillary AI or user)
            if facade_material and facade_material != "unknown":
                mat_score, reasons = self._score_material_match(archetype, facade_material)
                source_scores.mapillary_visual = mat_score
                match_reasons.extend(reasons)

            # Score from location
            if address:
                loc_score, reasons = self._score_from_location(archetype, address)
                source_scores.location = loc_score
                match_reasons.extend(reasons)

            # Score from WWR (from Mapillary AI detection)
            if hasattr(building_data, 'wwr') and building_data.wwr > 0:
                wwr_result = WWRAnalysisResult(
                    north=building_data.wwr_by_direction.get('N', 0.0),
                    south=building_data.wwr_by_direction.get('S', 0.0),
                    east=building_data.wwr_by_direction.get('E', 0.0),
                    west=building_data.wwr_by_direction.get('W', 0.0),
                    average=building_data.wwr,
                    source="ai_opencv",
                    confidence=0.7,
                )
                wwr_score, reasons = self._score_from_wwr(archetype, wwr_result)
                source_scores.wwr_analysis = wwr_score
                match_reasons.extend(reasons)

            candidates.append(ScoredCandidate(
                archetype=archetype,
                score=source_scores.total,
                source_scores=source_scores,
                match_reasons=match_reasons,
                mismatch_reasons=mismatch_reasons,
            ))

        # Sort and prepare facade images for visual analysis
        candidates.sort(key=lambda c: c.score, reverse=True)

        if not candidates:
            return self._default_result(building_type, construction_year)

        # Visual analysis from Mapillary images
        visual_analysis = None
        if use_ai_visual and building_data.facade_images and self.anthropic_api_key:
            # Convert image URLs to FacadeImage objects
            facade_images = self._urls_to_facade_images(building_data.facade_images)
            if facade_images:
                data_sources.append("mapillary_ai")
                visual_analysis = self._analyze_facade_images(facade_images, candidates[:5])

                if visual_analysis and visual_analysis.confidence > 0.5:
                    for candidate in candidates[:5]:
                        visual_score = self._score_from_visual(
                            candidate.archetype,
                            visual_analysis,
                        )
                        candidate.source_scores.mapillary_visual = visual_score
                        candidate.score = candidate.source_scores.total

                    candidates.sort(key=lambda c: c.score, reverse=True)

        winner = candidates[0]
        max_possible = 100

        return ArchetypeMatchResult(
            archetype=winner.archetype,
            confidence=min(winner.score / max_possible, 1.0),
            data_sources_used=data_sources,
            source_scores=winner.source_scores,
            match_reasons=winner.match_reasons,
            mismatch_reasons=winner.mismatch_reasons,
            visual_analysis=visual_analysis,
            alternatives=[(c.archetype, c.score / max_possible) for c in candidates[1:4]],
            calibration_hints=self._get_calibration_hints(winner.archetype, winner),
        )

    # =========================================================================
    # SCORING METHODS
    # =========================================================================

    def _score_from_declaration(
        self,
        archetype: DetailedArchetype,
        construction_year: int,
        energy_class: Optional[str],
        ventilation_type: Optional[str],
        heating_system: Optional[str],
    ) -> Tuple[float, List[str], List[str]]:
        """Score archetype based on energy declaration data."""
        score = 0.0
        reasons = []
        mismatches = []

        # Year match (35 pts)
        year_score, year_reasons, year_mismatches = self._score_year_match(
            archetype, construction_year
        )
        score += year_score
        reasons.extend(year_reasons)
        mismatches.extend(year_mismatches)

        # Energy class match (15 pts)
        if energy_class:
            if energy_class.upper() in ENERGY_CLASS_ERA_EXPECTATIONS:
                expected_eras = ENERGY_CLASS_ERA_EXPECTATIONS[energy_class.upper()]
                if archetype.era in expected_eras:
                    score += ScoringWeights.DECLARATION_ENERGY_CLASS
                    reasons.append(f"Energy class {energy_class} typical for {archetype.era.value} era")
                else:
                    # Penalty for mismatch
                    mismatches.append(f"Energy class {energy_class} unusual for {archetype.era.value}")

        # Ventilation match (10 pts)
        if ventilation_type:
            vent_upper = ventilation_type.upper()
            if vent_upper in VENTILATION_ERA_MAP:
                if archetype.era in VENTILATION_ERA_MAP[vent_upper]:
                    score += ScoringWeights.DECLARATION_VENTILATION
                    reasons.append(f"Ventilation type {vent_upper} matches era")

        # Heating system (10 pts) - district heating common in all eras
        if heating_system:
            # Most Swedish MFH use district heating, so this is less discriminating
            score += 5  # Base score for having heating data

        return score, reasons, mismatches

    def _score_year_match(
        self,
        archetype: DetailedArchetype,
        year: int,
    ) -> Tuple[float, List[str], List[str]]:
        """Score construction year match."""
        reasons = []
        mismatches = []

        if archetype.year_start <= year <= archetype.year_end:
            reasons.append(f"Year {year} within {archetype.year_start}-{archetype.year_end}")
            return (ScoringWeights.DECLARATION_YEAR_MATCH, reasons, mismatches)

        # Boundary tolerance (3 years)
        dist_to_start = abs(year - archetype.year_start)
        dist_to_end = abs(year - archetype.year_end)
        min_dist = min(dist_to_start, dist_to_end)

        if min_dist <= 3:
            reasons.append(f"Year {year} near era boundary")
            return (ScoringWeights.DECLARATION_YEAR_MATCH * 0.7, reasons, mismatches)

        if min_dist <= 10:
            return (ScoringWeights.DECLARATION_YEAR_MATCH * 0.3, [], mismatches)

        mismatches.append(f"Year {year} outside {archetype.year_start}-{archetype.year_end}")
        return (0, reasons, mismatches)

    def _score_from_geometry(
        self,
        archetype: DetailedArchetype,
        floors: int,
        wall_area_m2: float,
        window_to_wall_ratio: float,
    ) -> Tuple[float, List[str]]:
        """Score from building geometry."""
        score = 0.0
        reasons = []

        # Floor match
        if archetype.typical_floors:
            min_floors, max_floors = archetype.typical_floors
            if min_floors <= floors <= max_floors:
                score += ScoringWeights.GEOMETRY_FLOORS_MATCH
                reasons.append(f"Floor count {floors} matches typical range")

        return score, reasons

    def _score_form_match(
        self,
        archetype: DetailedArchetype,
        building_form: str,
    ) -> Tuple[float, List[str]]:
        """Score building form match."""
        score = 0.0
        reasons = []

        if not archetype.descriptors:
            return score, reasons

        form_lower = building_form.lower()

        # Check plan shapes
        for shape in archetype.descriptors.plan_shape:
            if form_lower in shape.value.lower():
                score += ScoringWeights.GEOMETRY_FORM_MATCH
                reasons.append(f"Building form '{building_form}' matches archetype")
                return score, reasons

        # Check archetype ID
        if form_lower in archetype.id.lower():
            score += ScoringWeights.GEOMETRY_FORM_MATCH
            reasons.append(f"Building form matches archetype ID")
            return score, reasons

        # Partial matches
        form_keywords = {
            "lamell": ["rectangular", "l_shape"],
            "skiv": ["rectangular"],
            "punkt": ["square", "circular"],
            "stjarn": ["y_shape"],
        }

        for key, shapes in form_keywords.items():
            if key in form_lower:
                for shape in archetype.descriptors.plan_shape:
                    if shape.value in shapes:
                        score += ScoringWeights.GEOMETRY_FORM_MATCH * 0.7
                        reasons.append(f"Form '{building_form}' partially matches")
                        return score, reasons

        return score, reasons

    def _score_floors_match(
        self,
        archetype: DetailedArchetype,
        num_floors: int,
    ) -> Tuple[float, List[str]]:
        """Score floor count match."""
        if not archetype.typical_floors:
            return 0, []

        min_f, max_f = archetype.typical_floors
        if min_f <= num_floors <= max_f:
            return ScoringWeights.GEOMETRY_FLOORS_MATCH, [f"Floor count {num_floors} typical"]

        return 0, []

    def _score_material_match(
        self,
        archetype: DetailedArchetype,
        material: str,
    ) -> Tuple[float, List[str]]:
        """Score facade material match."""
        score = 0.0
        reasons = []
        material_lower = material.lower()

        if archetype.wall_constructions:
            for wall in archetype.wall_constructions:
                wall_type = wall.type.value.lower()

                if material_lower == "concrete" and "concrete" in wall_type:
                    score += ScoringWeights.VISUAL_MATERIAL_MATCH
                    reasons.append(f"Concrete facade matches {wall.name_en}")
                    return score, reasons

                if material_lower == "brick" and ("brick" in wall_type or "tegel" in wall_type):
                    score += ScoringWeights.VISUAL_MATERIAL_MATCH
                    reasons.append(f"Brick facade matches {wall.name_en}")
                    return score, reasons

                if material_lower == "wood" and "wood" in wall_type:
                    score += ScoringWeights.VISUAL_MATERIAL_MATCH
                    reasons.append(f"Wood facade matches {wall.name_en}")
                    return score, reasons

                if material_lower in ["render", "plaster", "stucco"]:
                    if "stud" in wall_type or "frame" in wall_type:
                        score += ScoringWeights.VISUAL_MATERIAL_MATCH * 0.7
                        reasons.append(f"Rendered facade likely matches")
                        return score, reasons

        return score, reasons

    def _score_from_location(
        self,
        archetype: DetailedArchetype,
        address: str,
    ) -> Tuple[float, List[str]]:
        """Score from location (city/neighborhood)."""
        score = 0.0
        reasons = []

        if not archetype.descriptors:
            return score, reasons

        address_lower = address.lower()

        # Check city match
        for city in archetype.descriptors.typical_cities:
            if city.lower() in address_lower:
                score += ScoringWeights.LOCATION_CITY_MATCH
                reasons.append(f"Located in typical city: {city}")
                break

        # Check neighborhood match (higher value)
        for neighborhood in archetype.descriptors.typical_neighborhoods:
            if neighborhood.lower() in address_lower:
                score += ScoringWeights.LOCATION_NEIGHBORHOOD
                reasons.append(f"Located in typical neighborhood: {neighborhood}")
                break

        # Check keywords
        for keyword in archetype.descriptors.keywords_sv:
            if keyword.lower() in address_lower:
                score += 3
                reasons.append(f"Address contains keyword: {keyword}")
                break

        return score, reasons

    def _score_from_visual(
        self,
        archetype: DetailedArchetype,
        visual: AIVisualAnalysis,
    ) -> float:
        """Score archetype based on AI visual analysis."""
        score = 0.0

        if not archetype.descriptors:
            return score

        # Material match
        if visual.facade_material:
            mat_score, _ = self._score_material_match(archetype, visual.facade_material)
            score += mat_score

        # Balcony match
        if visual.balcony_type and visual.balcony_type in archetype.descriptors.balcony_types:
            score += ScoringWeights.VISUAL_BALCONY_MATCH

        # Facade pattern match
        if visual.facade_pattern and visual.facade_pattern in archetype.descriptors.facade_patterns:
            score += ScoringWeights.VISUAL_PATTERN_MATCH

        # Color match
        if visual.facade_colors and archetype.descriptors.typical_colors:
            for color in visual.facade_colors:
                if color.lower() in [c.lower() for c in archetype.descriptors.typical_colors]:
                    score += 3
                    break

        return score

    # =========================================================================
    # AI VISUAL ANALYSIS
    # =========================================================================

    def _analyze_facade_images(
        self,
        images: List["FacadeImage"],
        top_candidates: List[ScoredCandidate],
    ) -> Optional[AIVisualAnalysis]:
        """
        Use AI to analyze facade images and extract visual features.

        Uses Claude's vision capabilities to identify:
        - Facade material (concrete, brick, render, wood)
        - Balcony type (projecting, recessed, French, loggia)
        - Window pattern and proportions
        - Facade pattern (grid, horizontal bands, etc.)
        - Approximate era based on architectural style
        """
        if not images or not self.anthropic_api_key:
            return None

        # Select best image (prefer front-facing, recent)
        best_image = self._select_best_image(images)
        if not best_image:
            return None

        try:
            # Download image if needed
            image_data = self._get_image_data(best_image)
            if not image_data:
                return None

            # Build prompt with candidate information
            prompt = self._build_visual_analysis_prompt(top_candidates)

            # Call Claude Vision API
            response = self._call_claude_vision(image_data, prompt)
            if not response:
                return None

            # Parse response
            return self._parse_visual_response(response)

        except Exception as e:
            logger.warning(f"Visual analysis failed: {e}")
            return None

    def _select_best_image(self, images: List["FacadeImage"]) -> Optional["FacadeImage"]:
        """Select the best image for analysis."""
        # Prefer front-facing (S/SW orientation in Sweden)
        for img in images:
            if img.facade_direction in ["S", "SW"]:
                return img

        # Fall back to any image with URL
        for img in images:
            if img.url or img.local_path:
                return img

        return None

    def _get_image_data(self, image: "FacadeImage") -> Optional[str]:
        """Get base64 image data."""
        import requests

        try:
            if image.local_path and image.local_path.exists():
                with open(image.local_path, "rb") as f:
                    return base64.b64encode(f.read()).decode()

            if image.url:
                response = requests.get(image.url, timeout=30)
                if response.ok:
                    return base64.b64encode(response.content).decode()

        except Exception as e:
            logger.warning(f"Failed to get image data: {e}")

        return None

    def _build_visual_analysis_prompt(
        self,
        candidates: List[ScoredCandidate],
    ) -> str:
        """Build prompt for visual analysis."""
        prompt = """Analyze this Swedish building facade image. Extract the following features:

1. FACADE MATERIAL: What is the main facade material?
   Options: concrete, brick, wood, render/plaster, glass, stone

2. BALCONY TYPE: What type of balconies (if any)?
   Options: projecting (utstickande), recessed (indragna), french (franska), loggia, none

3. FACADE PATTERN: How are windows and elements arranged?
   Options: grid_uniform, horizontal_bands, vertical_emphasis, irregular, large_glazing

4. WINDOW PROPORTIONS: Are windows portrait, landscape, or square?

5. COLORS: What are the main facade colors? (Swedish terms if possible)

6. ESTIMATED ERA: Based on architectural style, estimate the construction era:
   - pre_1930 (ornate, traditional)
   - 1930_1945 (functionalism, simple)
   - 1946_1960 (folkhem, moderate)
   - 1961_1975 (miljonprogram, repetitive)
   - 1976_1985 (post oil crisis)
   - 1986_1995 (modern, FTX era)
   - 1996_2010 (low energy)
   - 2011_plus (near zero, sustainable)

7. SPECIAL FEATURES:
   - Has bay windows (burspråk)? yes/no
   - Has ornamental details? yes/no
   - Has visible heat recovery units on roof? yes/no

TOP CANDIDATES TO CONSIDER:
"""
        for i, candidate in enumerate(candidates[:3], 1):
            arch = candidate.archetype
            prompt += f"\n{i}. {arch.name_en} ({arch.year_start}-{arch.year_end})"
            if arch.descriptors and arch.descriptors.typical_colors:
                prompt += f"\n   Typical colors: {', '.join(arch.descriptors.typical_colors[:3])}"
            if arch.descriptors and arch.descriptors.balcony_types:
                prompt += f"\n   Typical balconies: {', '.join(b.value for b in arch.descriptors.balcony_types[:2])}"

        prompt += """

Respond in JSON format:
{
    "facade_material": "concrete|brick|wood|render|glass|stone",
    "balcony_type": "projecting|recessed|french|loggia|none",
    "facade_pattern": "grid_uniform|horizontal_bands|vertical_emphasis|irregular|large_glazing",
    "window_proportions": "portrait|landscape|square",
    "colors": ["color1", "color2"],
    "estimated_era": "1961_1975",
    "has_bay_windows": false,
    "has_ornament": false,
    "confidence": 0.8,
    "evidence": ["key observation 1", "key observation 2"]
}
"""
        return prompt

    def _call_claude_vision(self, image_base64: str, prompt: str) -> Optional[str]:
        """Call Claude Vision API."""
        try:
            import anthropic

            client = anthropic.Anthropic(api_key=self.anthropic_api_key)

            message = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=1024,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/jpeg",
                                    "data": image_base64,
                                },
                            },
                            {
                                "type": "text",
                                "text": prompt,
                            },
                        ],
                    }
                ],
            )

            return message.content[0].text

        except Exception as e:
            logger.warning(f"Claude Vision API call failed: {e}")
            return None

    def _parse_visual_response(self, response: str) -> Optional[AIVisualAnalysis]:
        """Parse AI visual analysis response."""
        try:
            # Extract JSON from response
            import re
            json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
            if not json_match:
                return None

            data = json.loads(json_match.group())

            # Map strings to enums
            balcony_map = {
                "projecting": BalconyType.PROJECTING,
                "recessed": BalconyType.RECESSED,
                "french": BalconyType.FRENCH,
                "loggia": BalconyType.LOGGIA,
            }

            pattern_map = {
                "grid_uniform": FacadePattern.GRID_UNIFORM,
                "horizontal_bands": FacadePattern.HORIZONTAL_BANDS,
                "vertical_emphasis": FacadePattern.VERTICAL_EMPHASIS,
                "irregular": FacadePattern.IRREGULAR,
                "large_glazing": FacadePattern.LARGE_GLAZING,
            }

            return AIVisualAnalysis(
                facade_material=data.get("facade_material"),
                facade_pattern=pattern_map.get(data.get("facade_pattern")),
                balcony_type=balcony_map.get(data.get("balcony_type")),
                facade_colors=data.get("colors", []),
                window_pattern=data.get("window_proportions", ""),
                has_bay_windows=data.get("has_bay_windows", False),
                has_ornament=data.get("has_ornament", False),
                estimated_era=data.get("estimated_era"),
                confidence=float(data.get("confidence", 0.5)),
                evidence=data.get("evidence", []),
            )

        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Failed to parse visual response: {e}")
            return None

    # =========================================================================
    # DEDICATED AI MODULE ANALYSIS
    # =========================================================================

    def analyze_wwr_from_images(
        self,
        images: List["FacadeImage"],
    ) -> Optional[WWRAnalysisResult]:
        """
        Analyze Window-to-Wall Ratio using dedicated WWRDetector module.

        Uses the existing AI infrastructure in src/ai/wwr_detector.py with
        OpenCV, SAM, or LangSAM backends.
        """
        detector = self._get_wwr_detector()
        if not detector or not images:
            return None

        try:
            from PIL import Image
            import requests
            from io import BytesIO

            wwr_by_direction: Dict[str, float] = {}
            confidences = []

            for img in images:
                direction = getattr(img, 'facade_direction', None) or 'unclassified'

                # Get image data
                pil_image = None
                if hasattr(img, 'local_path') and img.local_path and Path(img.local_path).exists():
                    pil_image = Image.open(img.local_path)
                elif hasattr(img, 'url') and img.url:
                    response = requests.get(img.url, timeout=30)
                    if response.ok:
                        pil_image = Image.open(BytesIO(response.content))

                if pil_image is None:
                    continue

                # Analyze with WWR detector
                wwr_result = detector.calculate_wwr(pil_image)
                if wwr_result:
                    wwr_value = wwr_result.average if hasattr(wwr_result, 'average') else wwr_result
                    if isinstance(wwr_value, (int, float)) and 0 < wwr_value < 1:
                        wwr_by_direction[direction] = float(wwr_value)
                        if hasattr(wwr_result, 'confidence'):
                            confidences.append(wwr_result.confidence)

            if not wwr_by_direction:
                return None

            # Compute averages
            avg_wwr = sum(wwr_by_direction.values()) / len(wwr_by_direction)
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0.6

            return WWRAnalysisResult(
                north=wwr_by_direction.get('N', 0.0),
                south=wwr_by_direction.get('S', 0.0),
                east=wwr_by_direction.get('E', 0.0),
                west=wwr_by_direction.get('W', 0.0),
                average=avg_wwr,
                source=f"ai_{self.wwr_backend}",
                confidence=avg_confidence,
            )

        except Exception as e:
            logger.warning(f"WWR analysis failed: {e}")
            return None

    def analyze_material_from_images(
        self,
        images: List["FacadeImage"],
    ) -> Optional[MaterialAnalysisResult]:
        """
        Analyze facade material using dedicated MaterialClassifier module.

        Uses the existing AI infrastructure in src/ai/material_classifier.py
        with DINOv2 or heuristic backends.
        """
        classifier = self._get_material_classifier()
        if not classifier or not images:
            return None

        try:
            from PIL import Image
            import requests
            from io import BytesIO

            all_predictions: List[Dict[str, float]] = []

            for img in images:
                # Get image data
                pil_image = None
                if hasattr(img, 'local_path') and img.local_path and Path(img.local_path).exists():
                    pil_image = Image.open(img.local_path)
                elif hasattr(img, 'url') and img.url:
                    response = requests.get(img.url, timeout=30)
                    if response.ok:
                        pil_image = Image.open(BytesIO(response.content))

                if pil_image is None:
                    continue

                # Classify material
                prediction = classifier.classify(pil_image)
                if prediction and hasattr(prediction, 'all_scores'):
                    all_predictions.append(prediction.all_scores)

            if not all_predictions:
                return None

            # Aggregate predictions across images
            aggregated: Dict[str, float] = {}
            for pred in all_predictions:
                for material, score in pred.items():
                    aggregated[material] = aggregated.get(material, 0) + score

            # Normalize
            total = sum(aggregated.values()) or 1
            for mat in aggregated:
                aggregated[mat] /= total

            # Get top materials
            sorted_materials = sorted(aggregated.items(), key=lambda x: x[1], reverse=True)
            primary = sorted_materials[0][0] if sorted_materials else None
            secondary = sorted_materials[1][0] if len(sorted_materials) > 1 else None

            # Get confidence (score of primary material)
            confidence = sorted_materials[0][1] if sorted_materials else 0.0

            return MaterialAnalysisResult(
                primary_material=primary,
                secondary_material=secondary,
                material_scores=aggregated,
                source="ai_dino",
                confidence=confidence,
            )

        except Exception as e:
            logger.warning(f"Material analysis failed: {e}")
            return None

    def analyze_roof_and_pv(
        self,
        latitude: float,
        longitude: float,
        construction_year: Optional[int] = None,
        footprint_area_m2: Optional[float] = None,
    ) -> Optional[PVAnalysisResult]:
        """
        Analyze roof and PV potential using RoofAnalyzer.

        Uses Google Solar API when available, falls back to estimation.
        """
        analyzer = self._get_roof_analyzer()
        if not analyzer:
            return None

        try:
            # Analyze roof
            roof_analysis = analyzer.analyze(
                latitude=latitude,
                longitude=longitude,
                construction_year=construction_year,
                footprint_area_m2=footprint_area_m2,
            )

            if not roof_analysis:
                return None

            return PVAnalysisResult(
                roof_type=roof_analysis.roof_type.value if hasattr(roof_analysis.roof_type, 'value') else str(roof_analysis.roof_type),
                total_roof_area_m2=roof_analysis.total_area_m2,
                usable_pv_area_m2=roof_analysis.net_available_m2,
                max_capacity_kwp=roof_analysis.optimal_capacity_kwp,
                annual_generation_kwh=roof_analysis.annual_generation_potential_kwh,
                existing_pv_kwp=roof_analysis.existing_solar.capacity_kwp if roof_analysis.existing_solar else 0.0,
                source=roof_analysis.data_source if hasattr(roof_analysis, 'data_source') else "estimation",
                confidence=roof_analysis.confidence if hasattr(roof_analysis, 'confidence') else 0.5,
            )

        except Exception as e:
            logger.warning(f"Roof/PV analysis failed: {e}")
            return None

    def _score_from_wwr(
        self,
        archetype: DetailedArchetype,
        wwr_result: WWRAnalysisResult,
    ) -> Tuple[float, List[str]]:
        """Score archetype based on WWR analysis from AI."""
        score = 0.0
        reasons = []

        if not archetype.typical_wwr:
            return score, reasons

        # Check if average WWR is typical for this archetype
        expected_wwr = archetype.typical_wwr
        actual_wwr = wwr_result.average

        # Allow 30% tolerance
        if abs(actual_wwr - expected_wwr) / expected_wwr < 0.3:
            score += ScoringWeights.WWR_MATCH
            reasons.append(f"WWR {actual_wwr:.0%} matches expected {expected_wwr:.0%}")

        # Check south-facing orientation (typical for energy-conscious eras)
        if wwr_result.south > wwr_result.north * 1.2:
            # South-facing WWR > North by 20%
            if archetype.era in [
                BuildingEra.ENERGI_1976_1985,
                BuildingEra.MODERN_1986_1995,
                BuildingEra.LAGENERGI_1996_2010,
                BuildingEra.NARA_NOLL_2011_PLUS,
            ]:
                score += ScoringWeights.WWR_ORIENTATION_MATCH
                reasons.append("South-facing windows typical for energy-conscious era")

        return score, reasons

    def _score_from_material_ai(
        self,
        archetype: DetailedArchetype,
        material_result: MaterialAnalysisResult,
    ) -> Tuple[float, List[str]]:
        """Score archetype based on AI material classification."""
        score = 0.0
        reasons = []

        if not material_result.primary_material:
            return score, reasons

        primary = material_result.primary_material.lower()

        # Use existing material scoring
        mat_score, mat_reasons = self._score_material_match(archetype, primary)
        score += mat_score
        reasons.extend(mat_reasons)

        # Confidence bonus
        if material_result.confidence > 0.7:
            score += ScoringWeights.MATERIAL_CONFIDENCE_BONUS
            reasons.append(f"High confidence AI material detection ({material_result.confidence:.0%})")

        return score, reasons

    def _score_from_pv(
        self,
        archetype: DetailedArchetype,
        pv_result: PVAnalysisResult,
    ) -> Tuple[float, List[str]]:
        """Score archetype based on PV/roof analysis."""
        score = 0.0
        reasons = []

        # Roof type match
        if archetype.descriptors:
            for roof_profile in archetype.descriptors.roof_profiles:
                roof_type_lower = pv_result.roof_type.lower()
                profile_lower = roof_profile.value.lower()

                if roof_type_lower == profile_lower or (
                    roof_type_lower == "flat" and profile_lower == "flat"
                ) or (
                    roof_type_lower in ["gabled", "pitched"] and profile_lower in ["gabled", "pitched", "hip"]
                ):
                    score += ScoringWeights.PV_ROOF_TYPE_MATCH
                    reasons.append(f"Roof type {pv_result.roof_type} matches archetype")
                    break

        # Existing PV = modern building
        if pv_result.existing_pv_kwp > 0:
            if archetype.era in [
                BuildingEra.LAGENERGI_1996_2010,
                BuildingEra.NARA_NOLL_2011_PLUS,
            ]:
                score += ScoringWeights.PV_EXISTING_MATCH
                reasons.append("Existing PV suggests modern/renovated building")

        return score, reasons

    # =========================================================================
    # HELPER METHODS
    # =========================================================================

    def _matches_building_type(self, archetype: DetailedArchetype, building_type: str) -> bool:
        """Check if archetype matches building type."""
        arch_id = archetype.id.lower()
        bt = building_type.lower()

        if "multi" in bt or bt == "mfh":
            return "mfh" in arch_id or "skivhus" in arch_id or "punkthus" in arch_id or "passive" in arch_id or "plus" in arch_id
        elif "single" in bt or bt == "sfh":
            return "sfh" in arch_id
        elif "terrace" in bt or "row" in bt:
            return "terraced" in arch_id or "radhus" in arch_id

        return True

    def _urls_to_facade_images(
        self,
        facade_images: Dict[str, List[str]],
    ) -> List["FacadeImage"]:
        """Convert URL dict to FacadeImage objects."""
        from ..ingest.image_fetcher import FacadeImage

        images = []
        direction_to_angle = {"N": 0, "E": 90, "S": 180, "W": 270}

        for direction, urls in facade_images.items():
            if direction == "unclassified":
                continue
            angle = direction_to_angle.get(direction, 0)
            for url in urls[:2]:  # Max 2 per direction
                images.append(FacadeImage(
                    image_id=f"{direction}_{hash(url) % 10000}",
                    source="mapillary",
                    latitude=0,
                    longitude=0,
                    compass_angle=angle,
                    url=url,
                    facade_direction=direction,
                ))

        return images

    def _get_calibration_hints(
        self,
        archetype: DetailedArchetype,
        candidate: ScoredCandidate,
    ) -> Dict[str, Any]:
        """Get calibration hints from matched archetype."""
        hints = {}

        if archetype.descriptors:
            desc = archetype.descriptors

            # Infiltration variability
            if desc.infiltration_variability == "high":
                hints["infiltration_note"] = "High variability - recommend measurement"
            elif desc.infiltration_variability == "low":
                hints["infiltration_note"] = "Low variability - archetype default reliable"

            # U-value variability
            if desc.u_value_variability == "high":
                hints["u_value_note"] = "High variability - inspect insulation state"

            # Renovation signs
            if desc.likely_renovated_if:
                hints["renovation_indicators"] = desc.likely_renovated_if

        # Add U-values from archetype
        if archetype.wall_constructions:
            hints["wall_u_value"] = archetype.wall_constructions[0].u_value
        if archetype.roof_construction:
            hints["roof_u_value"] = archetype.roof_construction.u_value
        if archetype.window_construction:
            hints["window_u_value"] = archetype.window_construction.u_value_installed

        return hints

    def _default_result(
        self,
        building_type: str,
        construction_year: Optional[int],
    ) -> ArchetypeMatchResult:
        """Return default result when no match found."""
        # Find a generic archetype for the year
        for arch in self.archetypes.values():
            if construction_year and arch.year_start <= construction_year <= arch.year_end:
                if self._matches_building_type(arch, building_type):
                    return ArchetypeMatchResult(
                        archetype=arch,
                        confidence=0.3,
                        data_sources_used=["fallback"],
                        match_reasons=["Default match based on year"],
                    )

        # Last resort
        arch = next(iter(self.archetypes.values()))
        return ArchetypeMatchResult(
            archetype=arch,
            confidence=0.2,
            data_sources_used=["fallback"],
            match_reasons=["No specific match - using default"],
        )


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def match_archetype_from_context(
    context: "EnhancedBuildingContext",
    facade_images: Optional[List["FacadeImage"]] = None,
) -> ArchetypeMatchResult:
    """
    Match archetype from EnhancedBuildingContext.

    Usage:
        from src.baseline import match_archetype_from_context

        result = match_archetype_from_context(building_context)
        print(result.archetype.name_en)
    """
    matcher = ArchetypeMatcherV2()
    return matcher.match_from_context(context, facade_images)


def match_archetype_from_building_data(
    building_data: "BuildingData",
) -> ArchetypeMatchResult:
    """
    Match archetype from AddressPipeline BuildingData.

    Usage:
        from src.baseline import match_archetype_from_building_data

        result = match_archetype_from_building_data(building_data)
        print(result.archetype.name_en)
    """
    matcher = ArchetypeMatcherV2()
    return matcher.match_from_building_data(building_data)
