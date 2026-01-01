"""
Enhanced Building Context - Smart data aggregation for Swedish buildings.

Combines data from multiple sources:
- Energy declaration (PDF)
- Public map data (OSM, Overture)
- Archetype matching
- Facade analysis (AI/vision)

And intelligently infers:
- What measures are ALREADY implemented
- What ECMs are applicable
- Baseline parameters adjusted for existing measures
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set
from enum import Enum
import logging

from ..baseline.archetypes import (
    SwedishArchetype, ArchetypeMatcher, SWEDISH_ARCHETYPES,
    BuildingType, HeatingSystem, VentilationType
)
from ..baseline.archetype_matcher_v2 import ArchetypeMatcherV2, ArchetypeMatchResult
from ..ingest.energidek_parser import EnergyDeclarationData

logger = logging.getLogger(__name__)


class ExistingMeasure(Enum):
    """Measures that may already be implemented."""
    # Envelope
    EXTERNAL_WALL_INSULATION = "external_wall_insulation"
    INTERNAL_WALL_INSULATION = "internal_wall_insulation"
    ROOF_INSULATION = "roof_insulation"
    WINDOW_REPLACEMENT = "window_replacement"
    AIR_SEALING = "air_sealing"
    BASEMENT_INSULATION = "basement_insulation"

    # HVAC - Ventilation
    FTX_SYSTEM = "ftx_system"  # Full heat recovery ventilation
    F_SYSTEM = "f_system"  # Mechanical exhaust only (no heat recovery)
    HEAT_RECOVERY = "heat_recovery"  # Generic heat recovery (FTX or exhaust HP)
    DCV_SYSTEM = "dcv_system"  # Demand-controlled ventilation

    # HVAC - Heat Pumps (mutually exclusive for main heating)
    HEAT_PUMP_GROUND = "heat_pump_ground"  # Ground source / geothermal
    HEAT_PUMP_EXHAUST = "heat_pump_exhaust"  # Exhaust air (frånluftsvärmepump)
    HEAT_PUMP_AIR = "heat_pump_air"  # Air source (luft-luft/luft-vatten)
    HEAT_PUMP_WATER = "heat_pump_water"  # DHW heat pump

    # Renewables
    SOLAR_PV = "solar_pv"
    SOLAR_THERMAL = "solar_thermal"

    # Controls & Lighting
    LED_LIGHTING = "led_lighting"  # Any LED (general, common, outdoor)
    SMART_THERMOSTATS = "smart_thermostats"
    BMS_SYSTEM = "bms_system"  # Building management system


@dataclass
class InferredCharacteristics:
    """Characteristics inferred from available data."""
    # Source tracking
    source: str  # 'energy_declaration', 'osm', 'archetype', 'facade_analysis'
    confidence: float  # 0.0 to 1.0

    # Inferred values
    construction_year: Optional[int] = None
    facade_material: Optional[str] = None
    building_type: Optional[str] = None
    heating_system: Optional[str] = None
    ventilation_type: Optional[str] = None
    roof_type: Optional[str] = None
    window_u_value: Optional[float] = None
    heat_recovery_efficiency: Optional[float] = None


@dataclass
class EnhancedBuildingContext:
    """
    Complete building context with all available data.

    This is the central data structure that:
    1. Aggregates data from all sources
    2. Identifies existing measures
    3. Provides inputs for ECM filtering
    4. Provides inputs for baseline generation
    """
    # Identification
    address: str = ""
    property_id: str = ""

    # Core characteristics (required for analysis)
    construction_year: int = 0
    building_type: BuildingType = BuildingType.MULTI_FAMILY
    facade_material: str = "unknown"
    heating_system: HeatingSystem = HeatingSystem.DISTRICT
    ventilation_type: VentilationType = VentilationType.EXHAUST

    # Geometry
    atemp_m2: float = 0.0
    floors: int = 1
    floor_height_m: float = 2.7
    wall_area_m2: float = 0.0
    window_area_m2: float = 0.0
    roof_area_m2: float = 0.0
    available_pv_area_m2: float = 0.0
    window_to_wall_ratio: float = 0.15

    # Current performance (from energy declaration or archetype)
    current_heating_kwh_m2: float = 0.0
    current_lighting_w_m2: float = 8.0
    current_window_u: float = 2.0
    current_infiltration_ach: float = 0.1
    current_heat_recovery: float = 0.0

    # Constraints
    heritage_listed: bool = False
    has_hydronic_distribution: bool = True
    roof_type: str = "flat"
    shading_factor: float = 0.1

    # Financial constraints
    max_investment_sek: Optional[float] = None

    # Matched archetype
    archetype_id: Optional[str] = None
    archetype: Optional[SwedishArchetype] = None

    # Existing measures (THE KEY INSIGHT!)
    existing_measures: Set[ExistingMeasure] = field(default_factory=set)

    # Data sources used
    data_sources: List[InferredCharacteristics] = field(default_factory=list)

    # Raw energy declaration data
    energy_declaration: Optional[EnergyDeclarationData] = None

    # Calibration hints from LLM archetype reasoner (renovation detection)
    calibration_hints: Dict[str, Any] = field(default_factory=dict)

    def to_constraint_context(self) -> Dict:
        """Convert to dict for ConstraintEngine."""
        return {
            'construction_year': self.construction_year,
            'building_type': self.building_type.value if isinstance(self.building_type, BuildingType) else self.building_type,
            'facade_material': self.facade_material,
            'heating_system': self.heating_system.value if isinstance(self.heating_system, HeatingSystem) else self.heating_system,
            'ventilation_type': self.ventilation_type.value if isinstance(self.ventilation_type, VentilationType) else self.ventilation_type,
            'heritage_listed': self.heritage_listed,
            'current_window_u': self.current_window_u,
            'current_infiltration_ach': self.current_infiltration_ach,
            'current_heat_recovery': self.current_heat_recovery,
            'has_hydronic_distribution': self.has_hydronic_distribution,
            'floor_area_m2': self.atemp_m2,
            'wall_area_m2': self.wall_area_m2,
            'window_area_m2': self.window_area_m2,
            'roof_area_m2': self.roof_area_m2,
            'available_pv_area_m2': self.available_pv_area_m2,
            'roof_type': self.roof_type,
            'shading_factor': self.shading_factor,
            'current_lighting_w_m2': self.current_lighting_w_m2,
            'max_investment_sek': self.max_investment_sek,
        }


class ExistingMeasuresDetector:
    """
    Detect existing measures from energy declaration data.

    This is CRITICAL - we don't want to recommend measures
    that are already implemented!
    """

    # Keywords that indicate measure is already done
    MEASURE_KEYWORDS = {
        # Ventilation
        ExistingMeasure.FTX_SYSTEM: [
            'ftx', 'värmeåtervinning', 'heat recovery',
            'roterande värmeväxlare', 'plattvärmeväxlare',
            'motströmsvärmeväxlare', 'fläkt med återvinning'
        ],
        ExistingMeasure.F_SYSTEM: [
            'mekanisk frånluft', 'fläktventilation', 'f-system'
        ],
        ExistingMeasure.DCV_SYSTEM: [
            'behovsstyrd ventilation', 'dcv', 'co2-styrd',
            'demand controlled', 'variabelt luftflöde'
        ],
        # Heat pumps
        ExistingMeasure.HEAT_PUMP_GROUND: [
            'bergvärme', 'jordvärme', 'geotermi', 'ground source',
            'borrhål', 'geotermisk'
        ],
        ExistingMeasure.HEAT_PUMP_EXHAUST: [
            'frånluftsvärmepump', 'exhaust air heat pump', 'fvp',
            'frånluftspump'
        ],
        ExistingMeasure.HEAT_PUMP_AIR: [
            'luft-luft', 'luft-vatten', 'luftvärmepump',
            'air source', 'air-to-air', 'air-to-water'
        ],
        ExistingMeasure.HEAT_PUMP_WATER: [
            'varmvattenpump', 'dhw heat pump', 'varmvattenvärmepump'
        ],
        # Solar
        ExistingMeasure.SOLAR_PV: [
            'solcell', 'solel', 'pv', 'photovoltaic', 'solpanel'
        ],
        ExistingMeasure.SOLAR_THERMAL: [
            'solfångare', 'solar thermal', 'solvärme'
        ],
        # Envelope
        ExistingMeasure.WINDOW_REPLACEMENT: [
            'fönsterbyte', 'nya fönster', 'treglas', 'triple glazing',
            'energifönster', 'lågemissionsglas'
        ],
        ExistingMeasure.EXTERNAL_WALL_INSULATION: [
            'tilläggsisolering', 'fasadisolering', 'etics',
            'utvändig isolering', 'tilläggsisolerad'
        ],
        ExistingMeasure.ROOF_INSULATION: [
            'vindsisolering', 'takisolering', 'attic insulation',
            'blåst isolering'
        ],
        ExistingMeasure.BASEMENT_INSULATION: [
            'källarisolering', 'golvssiolering', 'basement insulation'
        ],
        # Controls
        ExistingMeasure.LED_LIGHTING: [
            'led', 'led-belysning', 'energieffektiv belysning'
        ],
        ExistingMeasure.SMART_THERMOSTATS: [
            'smart termostat', 'rumstermostat', 'iot termostat'
        ],
        ExistingMeasure.BMS_SYSTEM: [
            'styr- och övervakning', 'bms', 'fastighetssystem',
            'ddc', 'building automation'
        ],
    }

    def detect_from_declaration(
        self,
        declaration: EnergyDeclarationData
    ) -> Set[ExistingMeasure]:
        """
        Detect existing measures from energy declaration.

        Sources:
        - Ventilation system type (FTX = heat recovery exists)
        - Heat sources (ground source HP, exhaust air HP, air source HP)
        - Solar generation (PV/thermal exists)
        - Recommendations (if NOT recommended, might be done)
        - Raw text analysis

        CRITICAL: This is the ROCK SOLID logic for filtering ECMs.
        If any measure is detected here, corresponding ECMs will be excluded.
        """
        existing = set()

        # ════════════════════════════════════════════════════════════════
        # 1. VENTILATION TYPE - from energy declaration
        # ════════════════════════════════════════════════════════════════
        vent_type = declaration.ventilation.system_type.upper() if declaration.ventilation else ''
        if vent_type == 'FTX':
            existing.add(ExistingMeasure.FTX_SYSTEM)
            existing.add(ExistingMeasure.HEAT_RECOVERY)
            logger.info("✓ Detected FTX system (heat recovery ventilation)")
        elif vent_type in ('F', 'FT'):
            existing.add(ExistingMeasure.F_SYSTEM)
            logger.info(f"✓ Detected {vent_type} system (mechanical ventilation)")

        # Check for heat recovery efficiency (indicates FTX even if not labeled)
        if hasattr(declaration.ventilation, 'heat_recovery_efficiency'):
            hr_eff = declaration.ventilation.heat_recovery_efficiency or 0
            if hr_eff > 0.4:  # Any meaningful heat recovery
                existing.add(ExistingMeasure.HEAT_RECOVERY)
                if hr_eff > 0.6:  # Good FTX system
                    existing.add(ExistingMeasure.FTX_SYSTEM)
                    logger.info(f"✓ Detected FTX (heat recovery efficiency: {hr_eff:.0%})")

        # ════════════════════════════════════════════════════════════════
        # 2. HEAT PUMPS - from energy kWh data
        # NOTE: These are MUTUALLY EXCLUSIVE for main heating
        # ════════════════════════════════════════════════════════════════
        if declaration.ground_source_heat_pump_kwh and declaration.ground_source_heat_pump_kwh > 0:
            existing.add(ExistingMeasure.HEAT_PUMP_GROUND)
            logger.info(f"✓ Ground source HP: {declaration.ground_source_heat_pump_kwh:,.0f} kWh")

        if declaration.exhaust_air_heat_pump_kwh and declaration.exhaust_air_heat_pump_kwh > 0:
            existing.add(ExistingMeasure.HEAT_PUMP_EXHAUST)
            existing.add(ExistingMeasure.HEAT_RECOVERY)  # FVP provides heat recovery too
            logger.info(f"✓ Exhaust air HP: {declaration.exhaust_air_heat_pump_kwh:,.0f} kWh")

        # Check for air-source heat pump (if field exists)
        if hasattr(declaration, 'air_source_heat_pump_kwh'):
            if declaration.air_source_heat_pump_kwh and declaration.air_source_heat_pump_kwh > 0:
                existing.add(ExistingMeasure.HEAT_PUMP_AIR)
                logger.info(f"✓ Air source HP: {declaration.air_source_heat_pump_kwh:,.0f} kWh")

        # Check for generic heat pump kWh (some declarations bundle all HPs)
        if hasattr(declaration, 'heat_pump_kwh'):
            if declaration.heat_pump_kwh and declaration.heat_pump_kwh > 0:
                # We have SOME heat pump, but don't know which type
                # Flag all HP ECMs as existing to be safe
                if not any(m in existing for m in [
                    ExistingMeasure.HEAT_PUMP_GROUND,
                    ExistingMeasure.HEAT_PUMP_EXHAUST,
                    ExistingMeasure.HEAT_PUMP_AIR
                ]):
                    logger.warning(f"⚠ Unknown HP type with {declaration.heat_pump_kwh:,.0f} kWh - blocking all HP ECMs")
                    existing.add(ExistingMeasure.HEAT_PUMP_GROUND)
                    existing.add(ExistingMeasure.HEAT_PUMP_EXHAUST)
                    existing.add(ExistingMeasure.HEAT_PUMP_AIR)

        # ════════════════════════════════════════════════════════════════
        # 3. SOLAR - from energy kWh data
        # ════════════════════════════════════════════════════════════════
        if declaration.solar_pv_kwh and declaration.solar_pv_kwh > 0:
            existing.add(ExistingMeasure.SOLAR_PV)
            logger.info(f"✓ Solar PV: {declaration.solar_pv_kwh:,.0f} kWh")

        if declaration.solar_thermal_kwh and declaration.solar_thermal_kwh > 0:
            existing.add(ExistingMeasure.SOLAR_THERMAL)
            logger.info(f"✓ Solar thermal: {declaration.solar_thermal_kwh:,.0f} kWh")

        # 4. Analyze raw text for keywords
        if declaration.raw_text:
            text_lower = declaration.raw_text.lower()
            for measure, keywords in self.MEASURE_KEYWORDS.items():
                for keyword in keywords:
                    if keyword in text_lower:
                        # Double-check it's not a recommendation
                        if not self._is_recommendation_context(text_lower, keyword):
                            existing.add(measure)
                            logger.debug(f"Detected existing from text: {measure.value}")
                            break

        # 5. Infer from energy performance
        # If building has very good energy class (A/B) and is old, likely has measures
        if declaration.energy_class in ['A', 'B'] and declaration.construction_year:
            if declaration.construction_year < 2010:
                logger.info("High energy class for older building suggests measures implemented")
                # Can't know which, but useful for context

        return existing

    def _is_recommendation_context(self, text: str, keyword: str) -> bool:
        """Check if keyword appears in recommendation context."""
        # Find keyword position
        pos = text.find(keyword)
        if pos == -1:
            return False

        # Check surrounding context (100 chars before)
        context_start = max(0, pos - 100)
        context = text[context_start:pos]

        recommendation_words = [
            'rekommend', 'föreslås', 'bör', 'kan',
            'åtgärd', 'potentiell', 'möjlig'
        ]

        for word in recommendation_words:
            if word in context:
                return True

        return False


class BuildingContextBuilder:
    """
    Build EnhancedBuildingContext from multiple data sources.

    Orchestrates:
    1. Energy declaration parsing
    2. Archetype matching (using ArchetypeMatcherV2 with detailed archetypes)
    3. Geometry data (OSM/Overture)
    4. Existing measures detection
    5. Baseline parameter adjustment
    """

    def __init__(self, use_v2_matcher: bool = True):
        self.use_v2_matcher = use_v2_matcher
        self.archetype_matcher = ArchetypeMatcher()  # Legacy fallback
        self.archetype_matcher_v2 = ArchetypeMatcherV2() if use_v2_matcher else None
        self.measures_detector = ExistingMeasuresDetector()

    def build_from_declaration(
        self,
        declaration: EnergyDeclarationData,
        geometry_data: Optional[Dict] = None,
        facade_material: Optional[str] = None,
    ) -> EnhancedBuildingContext:
        """
        Build complete context from energy declaration.

        Args:
            declaration: Parsed energy declaration
            geometry_data: Optional geometry from OSM/Overture
            facade_material: Optional facade material from analysis

        Returns:
            Complete EnhancedBuildingContext
        """
        ctx = EnhancedBuildingContext()

        # Basic info from declaration
        ctx.address = ', '.join(declaration.addresses) if declaration.addresses else ''
        ctx.property_id = declaration.declaration_id
        ctx.energy_declaration = declaration

        # Construction year
        if declaration.construction_year:
            ctx.construction_year = declaration.construction_year
            ctx.data_sources.append(InferredCharacteristics(
                source='energy_declaration',
                confidence=0.95,
                construction_year=declaration.construction_year
            ))

        # Area
        if declaration.atemp_sqm:
            ctx.atemp_m2 = declaration.atemp_sqm

        # Current energy performance
        if declaration.specific_energy_kwh_sqm:
            ctx.current_heating_kwh_m2 = declaration.specific_energy_kwh_sqm

        # Ventilation type
        ctx.ventilation_type = self._infer_ventilation_type(declaration)

        # Heating system
        ctx.heating_system = self._infer_heating_system(declaration)

        # Facade material (from analysis or default)
        if facade_material:
            ctx.facade_material = facade_material
        else:
            ctx.facade_material = self._infer_facade_material(declaration)

        # Match archetype using V2 matcher (with detailed archetypes and real data)
        if self.use_v2_matcher and self.archetype_matcher_v2:
            match_result = self.archetype_matcher_v2.match_from_context(ctx)
            ctx.archetype_id = match_result.archetype.id
            # Store detailed archetype info via data sources
            ctx.data_sources.append(InferredCharacteristics(
                source='archetype_v2',
                confidence=match_result.confidence,
                construction_year=ctx.construction_year,
                facade_material=ctx.facade_material,
            ))
            # Apply detailed archetype defaults
            self._apply_detailed_archetype_defaults(ctx, match_result)
            logger.info(f"Matched archetype: {match_result.archetype.name_en} "
                       f"(confidence: {match_result.confidence:.0%})")
            logger.debug(f"Match reasons: {match_result.match_reasons}")
        else:
            # Legacy fallback
            archetype = self.archetype_matcher.match(
                construction_year=ctx.construction_year,
                building_type='multi_family',
                facade_material=ctx.facade_material
            )
            if archetype:
                ctx.archetype = archetype
                ctx.archetype_id = self._get_archetype_id(archetype)
                self._apply_archetype_defaults(ctx, archetype)

        # Detect existing measures (CRITICAL!)
        ctx.existing_measures = self.measures_detector.detect_from_declaration(declaration)

        # Adjust baseline for existing measures
        self._adjust_for_existing_measures(ctx)

        # Add geometry if provided
        if geometry_data:
            self._apply_geometry_data(ctx, geometry_data)

        return ctx

    def _infer_ventilation_type(self, declaration: EnergyDeclarationData) -> VentilationType:
        """Infer ventilation type from declaration."""
        vent = declaration.ventilation.system_type.upper()

        if 'FTX' in vent:
            return VentilationType.BALANCED
        elif 'FT' in vent:
            return VentilationType.BALANCED_NO_HR
        elif 'F' in vent:
            return VentilationType.EXHAUST
        elif 'S' in vent or 'SJÄLV' in vent:
            return VentilationType.NATURAL

        # Default based on year
        if declaration.construction_year:
            if declaration.construction_year < 1970:
                return VentilationType.NATURAL
            elif declaration.construction_year < 2000:
                return VentilationType.EXHAUST
            else:
                return VentilationType.BALANCED

        return VentilationType.EXHAUST

    def _infer_heating_system(self, declaration: EnergyDeclarationData) -> HeatingSystem:
        """Infer heating system from declaration."""
        # Check which heat source is dominant
        sources = {
            'district': declaration.district_heating_kwh or 0,
            'ground_hp': declaration.ground_source_heat_pump_kwh or 0,
            'exhaust_hp': declaration.exhaust_air_heat_pump_kwh or 0,
            'electric': declaration.electric_heating_kwh or 0,
            'gas': declaration.gas_kwh or 0,
            'oil': declaration.oil_kwh or 0,
        }

        dominant = max(sources, key=sources.get)

        if dominant == 'district':
            return HeatingSystem.DISTRICT
        elif dominant == 'ground_hp':
            return HeatingSystem.HEAT_PUMP_GROUND
        elif dominant == 'exhaust_hp':
            return HeatingSystem.HEAT_PUMP_AIR  # Exhaust air HP
        elif dominant == 'electric':
            return HeatingSystem.ELECTRIC
        elif dominant == 'gas':
            return HeatingSystem.GAS
        elif dominant == 'oil':
            return HeatingSystem.OIL

        return HeatingSystem.DISTRICT  # Default for Swedish buildings

    def _infer_facade_material(self, declaration: EnergyDeclarationData) -> str:
        """Infer facade material from construction year if not provided."""
        year = declaration.construction_year or 1970

        # Swedish building stock patterns
        if year < 1945:
            return 'brick'  # Pre-war mostly brick
        elif year < 1970:
            return 'concrete'  # Folkhem era
        elif year < 1990:
            return 'concrete'  # Million program
        else:
            return 'render'  # Modern tends to be rendered

    def _get_archetype_id(self, archetype: SwedishArchetype) -> str:
        """Find archetype ID from database."""
        for arch_id, arch in SWEDISH_ARCHETYPES.items():
            if arch.name == archetype.name:
                return arch_id
        return 'unknown'

    def _apply_detailed_archetype_defaults(
        self,
        ctx: EnhancedBuildingContext,
        match_result: ArchetypeMatchResult
    ) -> None:
        """Apply detailed archetype defaults from V2 matcher."""
        arch = match_result.archetype

        # Store calibration hints for later use by BayesianCalibrationPipeline
        hints = match_result.calibration_hints
        if hints:
            ctx.calibration_hints = hints
            logger.info(f"Stored calibration hints: {list(hints.keys())}")

        if ctx.current_window_u == 2.0 and 'window_u_value' in hints:
            ctx.current_window_u = hints['window_u_value']

        # Apply floor height from descriptors
        if arch.descriptors and arch.descriptors.floor_to_floor_m:
            ctx.floor_height_m = sum(arch.descriptors.floor_to_floor_m) / 2

        # Apply WWR
        if arch.typical_wwr:
            ctx.window_to_wall_ratio = arch.typical_wwr

        # Apply infiltration based on era/construction
        if ctx.current_infiltration_ach == 0.1:
            # Use era-based defaults
            era_infiltration = {
                'pre_1930': 0.15,
                '1930_1945': 0.12,
                '1946_1960': 0.10,
                '1961_1975': 0.08,  # Concrete panel - tighter
                '1976_1985': 0.06,
                '1986_1995': 0.05,
                '1996_2010': 0.04,
                '2011_plus': 0.03,
            }
            ctx.current_infiltration_ach = era_infiltration.get(arch.era.value, 0.08)

        # Apply heat recovery from ventilation type
        if arch.ventilation:
            ctx.current_heat_recovery = arch.ventilation.heat_recovery_efficiency

        logger.debug(f"Applied archetype defaults: window_u={ctx.current_window_u}, "
                    f"infiltration={ctx.current_infiltration_ach}")

    def _apply_archetype_defaults(
        self,
        ctx: EnhancedBuildingContext,
        archetype: SwedishArchetype
    ) -> None:
        """Apply archetype defaults to context (legacy method)."""
        # Only apply if not already set from declaration
        if ctx.current_window_u == 2.0:  # Default
            ctx.current_window_u = archetype.envelope.window_u_value

        if ctx.current_infiltration_ach == 0.1:  # Default
            ctx.current_infiltration_ach = archetype.envelope.infiltration_ach

        if ctx.current_heat_recovery == 0.0:  # Default
            ctx.current_heat_recovery = archetype.hvac.heat_recovery_efficiency

        ctx.floor_height_m = archetype.typical_floor_height_m
        ctx.window_to_wall_ratio = archetype.typical_wwr

    def _adjust_for_existing_measures(self, ctx: EnhancedBuildingContext) -> None:
        """
        CRITICAL: Adjust baseline parameters for existing measures.

        If the building already has FTX, the baseline should reflect
        the CURRENT state, not the archetype default.
        """
        # FTX already installed → update heat recovery
        if ExistingMeasure.FTX_SYSTEM in ctx.existing_measures:
            ctx.current_heat_recovery = 0.75  # Typical FTX efficiency
            ctx.ventilation_type = VentilationType.BALANCED
            logger.info("Adjusted baseline: FTX already installed (HR=75%)")

        # Ground source HP → different heating system
        if ExistingMeasure.HEAT_PUMP_GROUND in ctx.existing_measures:
            ctx.heating_system = HeatingSystem.HEAT_PUMP_GROUND
            logger.info("Adjusted baseline: Ground source HP installed")

        # Window replacement → better U-value
        if ExistingMeasure.WINDOW_REPLACEMENT in ctx.existing_measures:
            ctx.current_window_u = min(ctx.current_window_u, 1.2)  # Assume decent windows
            logger.info("Adjusted baseline: Windows already replaced (U≤1.2)")

        # Air sealing → better infiltration
        if ExistingMeasure.AIR_SEALING in ctx.existing_measures:
            ctx.current_infiltration_ach = min(ctx.current_infiltration_ach, 0.06)
            logger.info("Adjusted baseline: Air sealing done (ACH≤0.06)")

        # LED lighting
        if ExistingMeasure.LED_LIGHTING in ctx.existing_measures:
            ctx.current_lighting_w_m2 = 5.0  # LED level
            logger.info("Adjusted baseline: LED lighting installed")

    def _apply_geometry_data(self, ctx: EnhancedBuildingContext, data: Dict) -> None:
        """Apply geometry data from OSM/Overture."""
        if 'floors' in data:
            ctx.floors = data['floors']
        if 'wall_area_m2' in data:
            ctx.wall_area_m2 = data['wall_area_m2']
        if 'window_area_m2' in data:
            ctx.window_area_m2 = data['window_area_m2']
        if 'roof_area_m2' in data:
            ctx.roof_area_m2 = data['roof_area_m2']
        if 'available_pv_area_m2' in data:
            ctx.available_pv_area_m2 = data['available_pv_area_m2']
        if 'roof_type' in data:
            ctx.roof_type = data['roof_type']


class SmartECMFilter:
    """
    Filter ECMs based on building context AND existing measures.

    ROCK SOLID two-stage filtering:
    1. ConstraintEngine - technical applicability (facade, heating system, etc.)
    2. Existing measures - don't recommend what's already done

    CRITICAL RULES:
    - If FTX exists → no ftx_installation, only ftx_upgrade/overhaul
    - If heat pump exists → no other heat pump ECMs (they're mutually exclusive)
    - If solar PV exists → no solar_pv ECM
    - If heat recovery exists → reduced savings for exhaust_air_heat_pump
    """

    # ════════════════════════════════════════════════════════════════
    # COMPREHENSIVE ECM → MEASURE MAPPING
    # ════════════════════════════════════════════════════════════════
    ECM_TO_MEASURE = {
        # Envelope ECMs
        'wall_external_insulation': ExistingMeasure.EXTERNAL_WALL_INSULATION,
        'wall_internal_insulation': ExistingMeasure.INTERNAL_WALL_INSULATION,
        'facade_renovation': ExistingMeasure.EXTERNAL_WALL_INSULATION,  # Implies insulation
        'roof_insulation': ExistingMeasure.ROOF_INSULATION,
        'window_replacement': ExistingMeasure.WINDOW_REPLACEMENT,
        'air_sealing': ExistingMeasure.AIR_SEALING,
        'basement_insulation': ExistingMeasure.BASEMENT_INSULATION,

        # Ventilation ECMs
        'ftx_installation': ExistingMeasure.FTX_SYSTEM,  # Blocked if FTX exists
        # ftx_upgrade and ftx_overhaul REQUIRE existing FTX (handled in special cases)

        # Solar ECMs
        'solar_pv': ExistingMeasure.SOLAR_PV,
        'solar_thermal': ExistingMeasure.SOLAR_THERMAL,

        # Lighting ECMs (all map to same measure)
        'led_lighting': ExistingMeasure.LED_LIGHTING,
        'led_common_areas': ExistingMeasure.LED_LIGHTING,
        'led_outdoor': ExistingMeasure.LED_LIGHTING,

        # Controls
        'smart_thermostats': ExistingMeasure.SMART_THERMOSTATS,
        'building_automation_system': ExistingMeasure.BMS_SYSTEM,
    }

    # Heat pump ECMs that should be blocked if ANY heat pump exists
    HEAT_PUMP_ECMS = {
        'ground_source_heat_pump': ExistingMeasure.HEAT_PUMP_GROUND,
        'exhaust_air_heat_pump': ExistingMeasure.HEAT_PUMP_EXHAUST,
        'air_source_heat_pump': ExistingMeasure.HEAT_PUMP_AIR,
        'heat_pump_water_heater': ExistingMeasure.HEAT_PUMP_WATER,
        'heat_pump_integration': None,  # Generic - blocked if any HP
    }

    # ECMs that require mechanical ventilation (F, FT, or FTX)
    REQUIRES_MECH_VENT = {
        'demand_controlled_ventilation',
        'ftx_upgrade',
        'ftx_overhaul',
        'ventilation_schedule_optimization',
    }

    def filter_ecms(
        self,
        all_ecms: List,
        context: EnhancedBuildingContext,
        constraint_engine
    ) -> Dict[str, List]:
        """
        Filter ECMs and categorize by applicability.

        Returns:
            {
                'applicable': [...],  # Can be recommended
                'already_done': [...],  # Already implemented
                'not_applicable': [...],  # Technical constraints prevent
            }
        """
        from ..ecm.constraints import BuildingContext as ConstraintContext

        # Convert to constraint context
        constraint_ctx = ConstraintContext(**context.to_constraint_context())

        applicable = []
        already_done = []
        not_applicable = []

        # Pre-compute flags for efficiency
        has_ftx = ExistingMeasure.FTX_SYSTEM in context.existing_measures
        has_f_system = ExistingMeasure.F_SYSTEM in context.existing_measures
        has_mech_vent = has_ftx or has_f_system
        has_heat_recovery = ExistingMeasure.HEAT_RECOVERY in context.existing_measures

        # Check for ANY heat pump
        has_any_heat_pump = any(m in context.existing_measures for m in [
            ExistingMeasure.HEAT_PUMP_GROUND,
            ExistingMeasure.HEAT_PUMP_EXHAUST,
            ExistingMeasure.HEAT_PUMP_AIR,
        ])

        for ecm in all_ecms:
            # Stage 1: Technical constraint check
            result = constraint_engine.evaluate_ecm(ecm, constraint_ctx)

            if not result.is_valid:
                not_applicable.append({
                    'ecm': ecm,
                    'reasons': result.failed_constraints
                })
                continue

            # ════════════════════════════════════════════════════════════════
            # Stage 2: Existing measure check (simple mapping)
            # ════════════════════════════════════════════════════════════════
            ecm_measure = self.ECM_TO_MEASURE.get(ecm.id)
            if ecm_measure and ecm_measure in context.existing_measures:
                already_done.append({
                    'ecm': ecm,
                    'reason': f"Already implemented: {ecm_measure.value}"
                })
                continue

            # ════════════════════════════════════════════════════════════════
            # Stage 3: Special case - FTX logic
            # ════════════════════════════════════════════════════════════════
            if ecm.id == 'ftx_installation':
                if has_ftx:
                    already_done.append({
                        'ecm': ecm,
                        'reason': 'FTX system already installed'
                    })
                    continue

            if ecm.id in ('ftx_upgrade', 'ftx_overhaul'):
                if not has_ftx:
                    not_applicable.append({
                        'ecm': ecm,
                        'reasons': [('ventilation', 'No FTX system to upgrade')]
                    })
                    continue

            # ════════════════════════════════════════════════════════════════
            # Stage 4: Special case - Heat pump mutual exclusivity
            # ════════════════════════════════════════════════════════════════
            if ecm.id in self.HEAT_PUMP_ECMS:
                if has_any_heat_pump:
                    # Check if this SPECIFIC type is installed
                    specific_measure = self.HEAT_PUMP_ECMS.get(ecm.id)
                    if specific_measure and specific_measure in context.existing_measures:
                        already_done.append({
                            'ecm': ecm,
                            'reason': f'Already has: {specific_measure.value}'
                        })
                    else:
                        # Different HP type exists - still not applicable
                        not_applicable.append({
                            'ecm': ecm,
                            'reasons': [('heating', 'Building already has a heat pump system')]
                        })
                    continue

            # Special case: heat_pump_integration requires existing HP
            if ecm.id == 'heat_pump_integration':
                if not has_any_heat_pump:
                    not_applicable.append({
                        'ecm': ecm,
                        'reasons': [('heating', 'No heat pump to integrate')]
                    })
                    continue

            # ════════════════════════════════════════════════════════════════
            # Stage 5: Special case - DCV and vent optimization need mech vent
            # ════════════════════════════════════════════════════════════════
            if ecm.id in self.REQUIRES_MECH_VENT:
                if not has_mech_vent:
                    not_applicable.append({
                        'ecm': ecm,
                        'reasons': [('ventilation', 'Requires mechanical ventilation (F/FT/FTX)')]
                    })
                    continue

            # ════════════════════════════════════════════════════════════════
            # Stage 6: Special case - Exhaust air HP less effective with FTX
            # ════════════════════════════════════════════════════════════════
            if ecm.id == 'exhaust_air_heat_pump' and has_ftx:
                # FTX already recovers heat from exhaust - exhaust HP adds little
                not_applicable.append({
                    'ecm': ecm,
                    'reasons': [('heating', 'FTX already recovers exhaust heat - exhaust HP not beneficial')]
                })
                continue

            # Passed all filters!
            applicable.append(ecm)

        return {
            'applicable': applicable,
            'already_done': already_done,
            'not_applicable': not_applicable
        }

    def explain_filtering(
        self,
        filter_result: Dict[str, List],
        context: EnhancedBuildingContext
    ) -> str:
        """Generate human-readable explanation of ECM filtering."""
        lines = []
        lines.append(f"ECM Analysis for: {context.address}")
        lines.append(f"Year: {context.construction_year}, Facade: {context.facade_material}")
        lines.append("")

        # Existing measures
        if context.existing_measures:
            lines.append("EXISTING MEASURES DETECTED:")
            for measure in context.existing_measures:
                lines.append(f"  ✓ {measure.value}")
            lines.append("")

        # Applicable ECMs
        lines.append(f"APPLICABLE ECMs ({len(filter_result['applicable'])}):")
        for ecm in filter_result['applicable']:
            lines.append(f"  → {ecm.name}")

        # Already done
        if filter_result['already_done']:
            lines.append("")
            lines.append(f"ALREADY IMPLEMENTED ({len(filter_result['already_done'])}):")
            for item in filter_result['already_done']:
                lines.append(f"  ✓ {item['ecm'].name} - {item['reason']}")

        # Not applicable
        if filter_result['not_applicable']:
            lines.append("")
            lines.append(f"NOT APPLICABLE ({len(filter_result['not_applicable'])}):")
            for item in filter_result['not_applicable']:
                lines.append(f"  ✗ {item['ecm'].name}")
                for field, reason in item['reasons']:
                    lines.append(f"      {reason}")

        return "\n".join(lines)
