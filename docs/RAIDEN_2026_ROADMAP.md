# RAIDEN 2026: Agentic Building Intelligence Roadmap

> **Goal:** Achieve 90% accuracy in automated building energy simulation using state-of-the-art agentic AI, physics-informed modeling, and learned occupancy patterns.

**Created:** 2026-01-03
**Based on:** Literature review of 2025-2026 BEM research + Raiden codebase analysis

---

## Executive Summary

Raiden already has a strong foundation:
- GeomEppyGenerator for actual polygon footprints
- Zone configs for 20+ Swedish building types (BBR-compliant)
- AI facade analysis (WWR, materials)
- Bayesian calibration with GP surrogates
- Multi-source data fusion (Sweden GeoJSON, Gripen, Microsoft Buildings)

**This roadmap focuses on:**
1. Replacing IdealLoadsAirSystem with realistic Swedish HVAC
2. Learning occupancy patterns from data
3. Improving calibration accuracy (target: CVRMSE < 5%)
4. Adding physics-informed neural networks for speed
5. Agentic orchestration for portfolio-scale analysis

---

## Phase 1: Realistic HVAC Systems (P0 - Critical)

### Current State
```
All buildings use: ZoneHVAC:IdealLoadsAirSystem
- No actual heating/cooling equipment
- No district heating connection
- No heat pump modeling
- Heat recovery via simple efficiency parameter
```

### Target State
```
Swedish HVAC templates:
- District Heating (fjärrvärme) - 70% of MFH
- Air Source Heat Pump (ASHP)
- Ground Source Heat Pump (GSHP)
- Exhaust Air Heat Pump (FTX-VP)
- Direct Electric (direktel)
- Hybrid systems (DH + HP)
```

### Implementation

#### 1.1 Create HVAC Template Library

**File:** `src/hvac/swedish_systems.py`

```python
"""
Swedish HVAC System Templates for EnergyPlus.

Based on:
- BeBo (Beställargruppen Bostäder) documentation
- Sveby 2.0 system definitions
- Actual Swedish installations (SIS standards)
"""

from enum import Enum
from dataclasses import dataclass
from typing import Dict, List, Optional

class SwedishHVACSystem(Enum):
    """Swedish HVAC system types mapped to EnergyPlus objects."""

    # District Heating (70% of Swedish MFH)
    DISTRICT_HEATING = "district_heating"

    # Heat Pumps
    EXHAUST_AIR_HP = "exhaust_air_hp"      # FTX-VP (Nibe F730, etc.)
    GROUND_SOURCE_HP = "ground_source_hp"  # Bergvärme
    AIR_SOURCE_HP = "air_source_hp"        # Luft-vatten

    # Ventilation
    FTX = "ftx"              # Balanced with HR (80-90% efficiency)
    F_SYSTEM = "f_system"    # Exhaust only
    FT_SYSTEM = "ft_system"  # Balanced without HR
    NATURAL = "natural"      # Självdrag (pre-1970)

    # Electric
    DIRECT_ELECTRIC = "direct_electric"

    # Hybrid
    DISTRICT_PLUS_HP = "district_plus_hp"  # DH + supplemental HP


@dataclass
class HVACTemplate:
    """EnergyPlus IDF snippet for a Swedish HVAC system."""
    system_type: SwedishHVACSystem

    # EnergyPlus objects as strings
    plant_loop: Optional[str] = None           # Hot water loop
    air_loop: Optional[str] = None             # Ventilation system
    zone_equipment: str = ""                   # Per-zone equipment

    # Performance parameters
    heating_cop: float = 1.0                   # COP at design conditions
    cooling_cop: float = 3.0
    heat_recovery_effectiveness: float = 0.0

    # Swedish-specific
    district_heating_substation: bool = False
    radiator_system: bool = True               # Most Swedish buildings
    underfloor_heating: bool = False           # Newer buildings


# District Heating Template (most common in Sweden)
DISTRICT_HEATING_TEMPLATE = '''
!- ============================================================
!- DISTRICT HEATING (Fjärrvärme)
!- Swedish standard: 80/60°C supply/return (older: 90/70)
!- Substation with heat exchanger
!- ============================================================

Sizing:Plant,
    {plant_name},
    Heating,
    {design_load_w},       !- Design Load (from E+ sizing or input)
    1.0;                   !- Loop Design Temperature Difference

PlantLoop,
    {plant_name},
    Water,
    ,                      !- Fluid Type
    ,                      !- User Defined Fluid Type
    {plant_name} Operation,
    {plant_name} Supply Outlet,
    {plant_name} Supply Inlet,
    {plant_name} Demand Outlet,
    {plant_name} Demand Inlet,
    autosize,              !- Maximum Loop Flow Rate
    0,                     !- Minimum Loop Flow Rate
    autosize,              !- Plant Loop Volume
    {plant_name} Supply Setpoint Nodes,
    ;                      !- CommonPipe

!- District Heating as infinite source
DistrictHeating,
    {plant_name} District Heating,
    {plant_name} DH Inlet,
    {plant_name} DH Outlet,
    autosize;              !- Nominal Capacity

!- Radiator in each zone
ZoneHVAC:Baseboard:RadiantConvective:Water,
    {zone_name} Radiator,
    AlwaysOn,
    {zone_name} Radiator Inlet,
    {zone_name} Radiator Outlet,
    HeatingDesignCapacity,
    autosize,              !- Design Heating Capacity
    ,                      !- Heating Design Capacity Per Floor Area
    ,                      !- Fraction of Autosized Heating Design Capacity
    0.3,                   !- Fraction Radiant
    0.0,                   !- Fraction of Radiant Energy to Surface 1
    {surface_list};        !- Surface names
'''

# Exhaust Air Heat Pump (FTX-VP) Template
EXHAUST_AIR_HP_TEMPLATE = '''
!- ============================================================
!- EXHAUST AIR HEAT PUMP (FTX-VP)
!- Extracts heat from exhaust air, heats supply + DHW
!- Common: Nibe F730, F750, Thermia Mega
!- ============================================================

!- Air-to-Water Heat Pump (exhaust air source)
HeatPump:WaterToWater:EquationFit:Heating,
    {hp_name},
    {hp_inlet_node},
    {hp_outlet_node},
    {source_inlet_node},   !- Exhaust air as source
    {source_outlet_node},
    autosize,              !- Reference Heating Capacity
    autosize,              !- Reference Heating Power Consumption
    {cop_heating},         !- Reference COP (typically 3.0-4.0 for FTX-VP)
    autosize,              !- Reference Load Side Flow Rate
    autosize,              !- Reference Source Side Flow Rate
    {hp_name} Heating CAPFTemp,
    {hp_name} Heating CAPFFlow,
    {hp_name} Heating EIRFTemp,
    {hp_name} Heating EIRFFlow,
    1.0,                   !- Compressor Minimum Part Load Ratio
    30;                    !- Maximum Cycling Rate

!- Performance curves for Swedish conditions
Curve:Biquadratic,
    {hp_name} Heating CAPFTemp,
    1.0,                   !- Coefficient1 Constant
    0.0,                   !- Coefficient2 x (exhaust temp)
    0.0,                   !- Coefficient3 x**2
    0.0,                   !- Coefficient4 y (water temp)
    0.0,                   !- Coefficient5 y**2
    0.0,                   !- Coefficient6 x*y
    -20, 25,               !- Min/Max X (exhaust air temp)
    30, 60;                !- Min/Max Y (water supply temp)
'''

# Ground Source Heat Pump (Bergvärme) Template
GSHP_TEMPLATE = '''
!- ============================================================
!- GROUND SOURCE HEAT PUMP (Bergvärme)
!- Borehole field + water-to-water HP
!- Swedish standard: 100-150m boreholes
!- ============================================================

GroundHeatExchanger:System,
    {ghx_name},
    {ghx_inlet_node},
    {ghx_outlet_node},
    autosize,              !- Design Flow Rate
    Site:GroundTemperature:Undisturbed:KusudaAchenbach,
    {borehole_data};       !- Number of boreholes, depth, etc.

HeatPump:WaterToWater:EquationFit:Heating,
    {hp_name},
    {hp_hot_inlet},
    {hp_hot_outlet},
    {hp_source_inlet},     !- From GHX
    {hp_source_outlet},    !- To GHX
    autosize,              !- Reference Heating Capacity
    autosize,              !- Reference Heating Power Consumption
    {cop_heating},         !- COP (typically 4.0-5.0 for GSHP)
    autosize,              !- Reference Load Side Flow Rate
    autosize,              !- Reference Source Side Flow Rate
    {performance_curves};

!- Swedish ground temperature profile
Site:GroundTemperature:Undisturbed:KusudaAchenbach,
    {ground_temp_name},
    1.8,                   !- Soil Thermal Conductivity (Swedish granite/clay)
    2400,                  !- Soil Density
    1000,                  !- Soil Specific Heat
    8.5,                   !- Average Ground Temperature (Stockholm)
    3.5,                   !- Amplitude (seasonal variation)
    15;                    !- Phase Shift (days from Jan 1 to minimum)
'''


def generate_hvac_idf(
    system_type: SwedishHVACSystem,
    zone_names: List[str],
    design_heating_load_w: float,
    archetype_era: str,
) -> str:
    """
    Generate EnergyPlus HVAC objects for a Swedish building.

    Args:
        system_type: Swedish HVAC system type
        zone_names: List of thermal zone names
        design_heating_load_w: Total design heating load
        archetype_era: Building era for typical system selection

    Returns:
        IDF string with all HVAC objects
    """
    # Implementation based on system type
    ...
```

#### 1.2 Add HVAC Selection Logic

**File:** `src/baseline/hvac_selector.py`

```python
"""
Automatic HVAC system selection based on building data.

Uses:
- Energy declaration (Gripen) heating type
- Building age (era)
- Location (DH availability)
- Size (HP sizing limits)
"""

from dataclasses import dataclass
from typing import Optional
from ..ingest.sweden_buildings import SwedishBuilding
from ..ingest.gripen_loader import GripenBuilding
from .archetypes import SwedishArchetype

@dataclass
class HVACSelection:
    """Selected HVAC system with parameters."""
    primary_heating: SwedishHVACSystem
    ventilation: SwedishHVACSystem

    # Sizing parameters
    heating_capacity_kw: Optional[float] = None
    hp_cop: float = 3.5
    heat_recovery_eff: float = 0.80

    # DHW
    dhw_system: str = "integrated"  # or "separate_tank"
    dhw_hp_fraction: float = 1.0    # Fraction from HP vs electric backup

    # Source data
    detected_from: str = "archetype"  # or "gripen", "sweden_geojson"
    confidence: float = 0.7


def select_hvac_system(
    building: Optional[SwedishBuilding] = None,
    gripen: Optional[GripenBuilding] = None,
    archetype: Optional[SwedishArchetype] = None,
    construction_year: Optional[int] = None,
    atemp_m2: Optional[float] = None,
) -> HVACSelection:
    """
    Select appropriate HVAC system from available data.

    Priority:
    1. Sweden Buildings GeoJSON (actual data)
    2. Gripen energy declaration (actual data)
    3. Archetype defaults (inferred)
    """

    # 1. Check Sweden Buildings GeoJSON (best source)
    if building:
        primary = _detect_from_sweden_building(building)
        if primary:
            return HVACSelection(
                primary_heating=primary,
                ventilation=_detect_ventilation(building),
                detected_from="sweden_geojson",
                confidence=0.90,
            )

    # 2. Check Gripen (nationwide data)
    if gripen:
        primary = _detect_from_gripen(gripen)
        if primary:
            return HVACSelection(
                primary_heating=primary,
                ventilation=_detect_ventilation_gripen(gripen),
                detected_from="gripen",
                confidence=0.85,
            )

    # 3. Fall back to archetype
    return _hvac_from_archetype(archetype, construction_year)


def _detect_from_sweden_building(b: SwedishBuilding) -> Optional[SwedishHVACSystem]:
    """Detect heating system from Sweden Buildings GeoJSON."""
    # Check actual energy consumption by source
    total = (
        (b.district_heating_kwh or 0) +
        (b.ground_source_hp_kwh or 0) +
        (b.exhaust_air_hp_kwh or 0) +
        (b.electric_heating_kwh or 0) +
        (b.oil_kwh or 0) +
        (b.gas_kwh or 0)
    )

    if total == 0:
        return None

    # Return dominant heating source
    if (b.district_heating_kwh or 0) / total > 0.5:
        return SwedishHVACSystem.DISTRICT_HEATING
    if (b.ground_source_hp_kwh or 0) / total > 0.5:
        return SwedishHVACSystem.GROUND_SOURCE_HP
    if (b.exhaust_air_hp_kwh or 0) / total > 0.5:
        return SwedishHVACSystem.EXHAUST_AIR_HP
    # ... etc

    return SwedishHVACSystem.DISTRICT_HEATING  # Default for Swedish MFH
```

#### 1.3 Integrate into Generator

**Modify:** `src/baseline/generator_v2.py`

```python
# Replace IdealLoadsAirSystem with actual HVAC
def _generate_hvac(self, zone_name: str, hvac_selection: HVACSelection) -> str:
    """Generate realistic HVAC instead of IdealLoads."""

    if hvac_selection.primary_heating == SwedishHVACSystem.DISTRICT_HEATING:
        return self._generate_district_heating(zone_name)
    elif hvac_selection.primary_heating == SwedishHVACSystem.EXHAUST_AIR_HP:
        return self._generate_ftx_hp(zone_name, hvac_selection.hp_cop)
    elif hvac_selection.primary_heating == SwedishHVACSystem.GROUND_SOURCE_HP:
        return self._generate_gshp(zone_name, hvac_selection.hp_cop)
    else:
        # Fall back to IdealLoads for unsupported systems
        return self._generate_ideal_loads(zone_name)
```

### Deliverables Phase 1
- [ ] `src/hvac/swedish_systems.py` - HVAC template library
- [ ] `src/hvac/hvac_selector.py` - Auto-selection from data
- [ ] Update `generator_v2.py` - Use realistic HVAC
- [ ] Test cases for each HVAC type
- [ ] Validation against Gripen declared values

---

## Phase 2: Learned Occupancy Patterns (P1 - High Impact)

### Current State
```python
# Fixed Sveby schedules for ALL buildings:
Schedule:Compact,
    OccupancySchedule,
    Through: 12/31, For: AllDays,
    Until: 07:00, 0.9,
    Until: 09:00, 0.5,
    Until: 17:00, 0.2,  # Everyone at work
    Until: 22:00, 0.9,
    Until: 24:00, 0.9;
```

### Target State
```
Building-specific schedules based on:
- Building type (residential, office, retail, restaurant)
- Location (city center vs suburb)
- Demographics (families vs students vs elderly)
- Actual meter data (if available)
- Swedish national patterns by region
```

### Implementation

#### 2.1 Swedish Occupancy Pattern Library

**File:** `src/schedules/swedish_patterns.py`

```python
"""
Swedish Occupancy Patterns by Building Type.

Sources:
- Sveby 2.0 (2022) - National energy calculation conventions
- SCB (Statistics Sweden) - Time use surveys
- Energimyndigheten - Building sector statistics
- BRF operational data (anonymized)
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np

@dataclass
class OccupancyPattern:
    """24-hour occupancy pattern with metadata."""
    name: str
    name_sv: str

    # Hourly fractions (0.0 - 1.0) for typical weekday
    weekday: List[float]  # 24 values

    # Hourly fractions for weekend
    weekend: List[float]  # 24 values

    # Seasonal variation (multiplier by month, 1-12)
    seasonal_factors: List[float]  # 12 values

    # Source confidence
    source: str
    confidence: float


# RESIDENTIAL PATTERNS
# Based on SCB Time Use Survey + BRF meter data

RESIDENTIAL_FAMILIES = OccupancyPattern(
    name="residential_families",
    name_sv="Flerbostadshus - barnfamiljer",
    weekday=[
        0.95, 0.95, 0.95, 0.95, 0.95, 0.90,  # 00-06: Night
        0.70, 0.40, 0.20,                     # 06-09: Morning departure
        0.15, 0.15, 0.15, 0.15, 0.15,        # 09-14: Work/school
        0.30, 0.50, 0.70,                     # 14-17: Return
        0.85, 0.90, 0.90, 0.90, 0.90, 0.95, 0.95  # 17-24: Evening
    ],
    weekend=[
        0.95, 0.95, 0.95, 0.95, 0.95, 0.95,  # 00-06
        0.90, 0.85, 0.80, 0.70, 0.60, 0.60,  # 06-12
        0.55, 0.55, 0.60, 0.65, 0.70, 0.80,  # 12-18
        0.85, 0.90, 0.90, 0.95, 0.95, 0.95   # 18-24
    ],
    seasonal_factors=[
        1.05, 1.05, 1.00, 0.95, 0.90, 0.85,  # Jan-Jun (summer away)
        0.70, 0.75, 0.90, 1.00, 1.00, 1.05   # Jul-Dec
    ],
    source="SCB + Sveby",
    confidence=0.85,
)

RESIDENTIAL_ELDERLY = OccupancyPattern(
    name="residential_elderly",
    name_sv="Seniorbostad",
    weekday=[
        0.98, 0.98, 0.98, 0.98, 0.98, 0.95,  # Very high night occupancy
        0.90, 0.85, 0.80, 0.75, 0.70, 0.70,  # Morning activities
        0.70, 0.75, 0.75, 0.80, 0.85, 0.90,  # Afternoon
        0.95, 0.98, 0.98, 0.98, 0.98, 0.98   # Evening
    ],
    weekend=[...],  # Similar to weekday for elderly
    seasonal_factors=[1.0] * 12,  # Less seasonal variation
    source="Sveby + Senior housing operators",
    confidence=0.80,
)

RESIDENTIAL_STUDENTS = OccupancyPattern(
    name="residential_students",
    name_sv="Studentbostad",
    weekday=[
        0.85, 0.85, 0.80, 0.75, 0.70, 0.60,  # Night owls!
        0.50, 0.45, 0.40, 0.35, 0.30, 0.30,  # Late starts
        0.30, 0.35, 0.40, 0.45, 0.50, 0.60,  # Afternoon
        0.70, 0.80, 0.85, 0.90, 0.90, 0.90   # Evening
    ],
    weekend=[...],
    seasonal_factors=[
        1.0, 1.0, 1.0, 1.0, 1.0, 0.5,  # Summer break!
        0.3, 0.6, 0.9, 1.0, 1.0, 0.8   # Exam periods
    ],
    source="Student housing surveys",
    confidence=0.75,
)


# COMMERCIAL PATTERNS

OFFICE_STANDARD = OccupancyPattern(
    name="office_standard",
    name_sv="Kontor - normal",
    weekday=[
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0,        # Night: empty
        0.05, 0.20, 0.70, 0.90, 0.95, 0.90,  # Morning arrival
        0.60, 0.85, 0.95, 0.90, 0.70, 0.30,  # Lunch dip, afternoon
        0.10, 0.05, 0.02, 0.0, 0.0, 0.0      # Evening: empty
    ],
    weekend=[0.0] * 24,  # Empty
    seasonal_factors=[
        1.0, 1.0, 1.0, 1.0, 0.9, 0.7,  # Summer holiday
        0.5, 0.7, 0.9, 1.0, 1.0, 0.8   # Christmas
    ],
    source="Sveby Kontor",
    confidence=0.90,
)

RETAIL_SHOPPING = OccupancyPattern(
    name="retail_shopping",
    name_sv="Butik",
    weekday=[
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.05, 0.20, 0.40, 0.60, 0.70,   # Opens 10:00
        0.65, 0.70, 0.75, 0.80, 0.90, 0.95,  # Peak afternoon
        0.70, 0.30, 0.0, 0.0, 0.0, 0.0       # Closes ~19:00
    ],
    weekend=[
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.10, 0.30, 0.50, 0.70,    # Opens later
        0.80, 0.90, 0.95, 0.85, 0.60, 0.20,  # Peak early afternoon
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0         # Closes ~17:00
    ],
    seasonal_factors=[
        0.8, 0.8, 0.9, 1.0, 1.0, 0.9,
        0.8, 0.9, 1.0, 1.0, 1.2, 1.5  # Christmas peak!
    ],
    source="Retail industry data",
    confidence=0.85,
)


def get_occupancy_pattern(
    building_type: str,
    demographics: str = "mixed",
    location_type: str = "urban",
) -> OccupancyPattern:
    """
    Get appropriate occupancy pattern for building.

    Args:
        building_type: residential, office, retail, restaurant, etc.
        demographics: families, elderly, students, mixed
        location_type: urban, suburban, rural

    Returns:
        OccupancyPattern for schedule generation
    """
    ...


def generate_idf_schedules(
    pattern: OccupancyPattern,
    schedule_name: str = "OccupancySchedule",
) -> str:
    """Convert OccupancyPattern to EnergyPlus Schedule:Compact."""
    ...
```

#### 2.2 Occupancy Learning from Meter Data

**File:** `src/schedules/pattern_learner.py`

```python
"""
Learn occupancy patterns from smart meter data.

When real meter data is available (e.g., from BRF or utility),
we can learn building-specific patterns that improve simulation accuracy.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

@dataclass
class LearnedPattern:
    """Occupancy pattern learned from meter data."""
    hourly_profile: np.ndarray  # 24 x 7 (hours x days)
    confidence: float
    num_samples: int
    cluster_id: int  # Which pattern cluster this belongs to


class OccupancyPatternLearner:
    """
    Learn occupancy patterns from electricity/heating meter data.

    Method:
    1. Normalize meter readings to 0-1 scale
    2. Extract daily load shapes
    3. Cluster similar days
    4. Identify weekday/weekend patterns
    5. Detect seasonal variation
    """

    def __init__(self, n_clusters: int = 5):
        self.n_clusters = n_clusters
        self.kmeans = None
        self.cluster_profiles = None

    def fit(
        self,
        meter_data: np.ndarray,  # Shape: (n_days, 24)
        timestamps: List,         # Date for each day
    ) -> 'OccupancyPatternLearner':
        """
        Fit pattern learner to meter data.

        Args:
            meter_data: Daily load profiles (n_days x 24 hours)
            timestamps: Date/time for each day (for weekday detection)

        Returns:
            Self (fitted)
        """
        # Normalize each day to 0-1
        normalized = self._normalize_daily(meter_data)

        # Cluster similar load shapes
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)
        clusters = self.kmeans.fit_predict(normalized)

        # Extract representative profile for each cluster
        self.cluster_profiles = {}
        for i in range(self.n_clusters):
            mask = clusters == i
            self.cluster_profiles[i] = normalized[mask].mean(axis=0)

        return self

    def predict_pattern(
        self,
        building_type: str,
        historical_meter: Optional[np.ndarray] = None,
    ) -> LearnedPattern:
        """
        Predict occupancy pattern for a building.

        If historical meter data is provided, match to nearest cluster.
        Otherwise, use building type heuristics.
        """
        ...

    def to_idf_schedule(self, pattern: LearnedPattern) -> str:
        """Convert learned pattern to EnergyPlus schedule."""
        ...


class SwedishLoadShapeLibrary:
    """
    Pre-trained load shape clusters for Swedish buildings.

    Based on aggregated smart meter data from Swedish utilities
    and Energiföretagen statistics.
    """

    # Pre-computed cluster centroids
    RESIDENTIAL_CLUSTERS = {
        0: "daytime_away",     # Families, working hours empty
        1: "home_all_day",     # Elderly, WFH
        2: "evening_heavy",    # Young professionals
        3: "irregular",        # Shift workers
        4: "summer_vacation",  # July pattern
    }

    COMMERCIAL_CLUSTERS = {
        0: "office_9to5",
        1: "retail_10to19",
        2: "restaurant_lunch_dinner",
        3: "24h_operation",
        4: "weekend_only",
    }

    @classmethod
    def get_default_profile(cls, building_type: str, cluster_name: str) -> np.ndarray:
        """Get pre-computed load profile."""
        ...
```

### Deliverables Phase 2
- [ ] `src/schedules/swedish_patterns.py` - Pattern library
- [ ] `src/schedules/pattern_learner.py` - ML-based learning
- [ ] Integration with zone_configs.py
- [ ] Validation against actual BRF meter data
- [ ] A/B test: fixed schedules vs learned patterns

---

## Phase 3: Calibration Enhancement (P1 - High Impact)

### Current State
- 80 LHS samples (should be 150-200)
- No train/test split (can't detect overfitting)
- Fixed priors (not context-aware)
- Simple RMSE metric (not ASHRAE-compliant)

### Target State
- 200+ LHS samples with stratification
- 80/20 train/test with R² validation
- Context-aware priors (FTX detected → tight HR prior)
- ASHRAE Guideline 14 metrics (NMBE, CVRMSE)
- Morris screening to reduce parameter dimensions

### Implementation

#### 3.1 Enhanced Calibration Pipeline

**Update:** `src/calibration/bayesian.py`

```python
# Already partially implemented from earlier sessions
# Key additions:

class EnhancedBayesianCalibrator:
    """
    Improved Bayesian calibration with:
    - Context-aware priors
    - Morris screening
    - ASHRAE metrics
    - Train/test validation
    """

    def __init__(
        self,
        n_samples: int = 200,          # Increased from 80
        train_test_split: float = 0.8,  # 80% train, 20% test
        use_morris_screening: bool = True,
        ashrae_compliance: bool = True,
    ):
        ...

    @classmethod
    def from_building_context(
        cls,
        context: EnhancedBuildingContext,
        archetype_id: str,
    ) -> 'EnhancedBayesianCalibrator':
        """
        Create calibrator with context-aware priors.

        If FTX detected → heat_recovery_eff ~ Beta(8, 2) on [0.65, 0.90]
        If air sealing done → infiltration_ach ~ N(0.04, 0.01)
        If renovation detected → wider priors for envelope
        """
        priors = {}

        # Ventilation system detection
        if context.existing_measures.has_ftx:
            priors['heat_recovery_eff'] = BetaPrior(
                alpha=8, beta=2,
                low=0.65, high=0.90,
                source="FTX detected in energy declaration"
            )

        # LLM calibration hints
        if context.llm_calibration_hints:
            for param, hint in context.llm_calibration_hints.items():
                priors[param] = hint.to_prior()

        return cls(priors=priors)

    def calibrate(
        self,
        baseline_idf: Path,
        target_energy_kwh_m2: float,
        weather_file: Path,
    ) -> CalibrationResult:
        """
        Run enhanced calibration.

        Steps:
        1. Morris screening (if enabled) → reduce to 3-5 params
        2. LHS sampling (200 points)
        3. Train GP surrogate (80% data)
        4. Validate on test set (20% data)
        5. Run MCMC on surrogate
        6. Report ASHRAE metrics
        """
        ...
```

#### 3.2 ASHRAE Guideline 14 Metrics

**File:** `src/calibration/ashrae_metrics.py`

```python
"""
ASHRAE Guideline 14-2014 Calibration Metrics.

Compliance requirements:
- Monthly: NMBE within ±5%, CVRMSE ≤ 15%
- Hourly: NMBE within ±10%, CVRMSE ≤ 30%
"""

from dataclasses import dataclass
from typing import List, Tuple
import numpy as np

@dataclass
class ASHRAEMetrics:
    """ASHRAE Guideline 14 calibration metrics."""
    nmbe: float          # Normalized Mean Bias Error (%)
    cvrmse: float        # Coefficient of Variation of RMSE (%)
    r_squared: float     # Coefficient of determination

    resolution: str      # 'hourly' or 'monthly'
    compliant: bool      # Meets ASHRAE thresholds

    def __str__(self) -> str:
        status = "✓ COMPLIANT" if self.compliant else "✗ NOT COMPLIANT"
        return (
            f"ASHRAE Guideline 14 ({self.resolution}):\n"
            f"  NMBE: {self.nmbe:+.1f}% (limit: ±{self._nmbe_limit()}%)\n"
            f"  CVRMSE: {self.cvrmse:.1f}% (limit: {self._cvrmse_limit()}%)\n"
            f"  R²: {self.r_squared:.2f}\n"
            f"  Status: {status}"
        )

    def _nmbe_limit(self) -> int:
        return 10 if self.resolution == 'hourly' else 5

    def _cvrmse_limit(self) -> int:
        return 30 if self.resolution == 'hourly' else 15


def calculate_ashrae_metrics(
    measured: np.ndarray,
    simulated: np.ndarray,
    resolution: str = 'monthly',
) -> ASHRAEMetrics:
    """
    Calculate ASHRAE Guideline 14 metrics.

    Args:
        measured: Actual energy consumption
        simulated: Simulated energy consumption
        resolution: 'hourly' or 'monthly'

    Returns:
        ASHRAEMetrics with compliance check
    """
    n = len(measured)
    mean_measured = np.mean(measured)

    # NMBE: Normalized Mean Bias Error
    nmbe = 100 * np.sum(simulated - measured) / (n * mean_measured)

    # CVRMSE: Coefficient of Variation of RMSE
    rmse = np.sqrt(np.sum((simulated - measured)**2) / n)
    cvrmse = 100 * rmse / mean_measured

    # R-squared
    ss_res = np.sum((measured - simulated)**2)
    ss_tot = np.sum((measured - mean_measured)**2)
    r_squared = 1 - (ss_res / ss_tot)

    # Compliance check
    if resolution == 'hourly':
        compliant = abs(nmbe) <= 10 and cvrmse <= 30
    else:  # monthly
        compliant = abs(nmbe) <= 5 and cvrmse <= 15

    return ASHRAEMetrics(
        nmbe=nmbe,
        cvrmse=cvrmse,
        r_squared=r_squared,
        resolution=resolution,
        compliant=compliant,
    )
```

### Deliverables Phase 3
- [ ] Enhanced `bayesian.py` with context-aware priors
- [ ] `ashrae_metrics.py` - Compliance metrics
- [ ] Morris screening integration
- [ ] Train/test validation in surrogate builder
- [ ] Documentation of calibration requirements

---

## Phase 4: Physics-Informed Neural Networks (P2 - Future)

### Goal
Replace GP surrogates with PINNs for:
- Better extrapolation
- Physical constraints (energy balance)
- Faster inference for real-time digital twins

### Implementation

**File:** `src/calibration/pinn_surrogate.py`

```python
"""
Physics-Informed Neural Network for Building Thermal Modeling.

Embeds RC thermal network equations as loss terms,
providing physical constraints that improve generalization.

Based on:
- Gokhale et al. (2022) "Physics informed neural networks for
  control oriented thermal modeling of buildings"
- Applied Energy 314:118852
"""

import torch
import torch.nn as nn
from typing import List, Tuple

class RCThermalNetwork(nn.Module):
    """
    RC (Resistance-Capacitance) thermal network with learnable parameters.

    Standard 2R2C model:
    - C_air * dT_air/dt = Q_hvac + Q_internal + (T_wall - T_air)/R_in
    - C_wall * dT_wall/dt = (T_air - T_wall)/R_in + (T_out - T_wall)/R_out

    Parameters R_in, R_out, C_air, C_wall are learned from data
    while respecting physical constraints.
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        n_layers: int = 3,
    ):
        super().__init__()

        # Learnable thermal parameters (constrained positive)
        self.log_R_in = nn.Parameter(torch.tensor(0.0))   # Internal resistance
        self.log_R_out = nn.Parameter(torch.tensor(0.0))  # External resistance
        self.log_C_air = nn.Parameter(torch.tensor(0.0))  # Air capacitance
        self.log_C_wall = nn.Parameter(torch.tensor(0.0)) # Wall capacitance

        # Neural network for residual correction
        layers = []
        input_dim = 5  # T_air, T_wall, T_out, Q_hvac, Q_internal
        for i in range(n_layers):
            layers.append(nn.Linear(
                input_dim if i == 0 else hidden_dim,
                hidden_dim
            ))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim, 2))  # dT_air, dT_wall

        self.residual_net = nn.Sequential(*layers)

    @property
    def R_in(self):
        return torch.exp(self.log_R_in)

    @property
    def R_out(self):
        return torch.exp(self.log_R_out)

    @property
    def C_air(self):
        return torch.exp(self.log_C_air)

    @property
    def C_wall(self):
        return torch.exp(self.log_C_wall)

    def physics_forward(
        self,
        T_air: torch.Tensor,
        T_wall: torch.Tensor,
        T_out: torch.Tensor,
        Q_hvac: torch.Tensor,
        Q_internal: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute temperature derivatives from RC model.

        Returns:
            dT_air_dt, dT_wall_dt
        """
        dT_air_dt = (
            Q_hvac + Q_internal + (T_wall - T_air) / self.R_in
        ) / self.C_air

        dT_wall_dt = (
            (T_air - T_wall) / self.R_in + (T_out - T_wall) / self.R_out
        ) / self.C_wall

        return dT_air_dt, dT_wall_dt

    def forward(
        self,
        T_air: torch.Tensor,
        T_wall: torch.Tensor,
        T_out: torch.Tensor,
        Q_hvac: torch.Tensor,
        Q_internal: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Physics + Neural Network hybrid forward pass.
        """
        # Physics-based prediction
        dT_air_phys, dT_wall_phys = self.physics_forward(
            T_air, T_wall, T_out, Q_hvac, Q_internal
        )

        # Neural network correction (for unmodeled effects)
        x = torch.stack([T_air, T_wall, T_out, Q_hvac, Q_internal], dim=-1)
        correction = self.residual_net(x)
        dT_air_nn, dT_wall_nn = correction[..., 0], correction[..., 1]

        # Combine (physics dominant, NN for correction)
        dT_air = dT_air_phys + 0.1 * dT_air_nn
        dT_wall = dT_wall_phys + 0.1 * dT_wall_nn

        return dT_air, dT_wall


class BuildingPINN:
    """
    PINN-based building energy surrogate.

    Replaces GP surrogate for:
    - Multi-step prediction (better than GP)
    - Physical constraints (energy balance)
    - Real-time inference (ms instead of seconds)
    """

    def __init__(self, zones: int = 1):
        self.model = RCThermalNetwork()
        self.zones = zones

    def train(
        self,
        training_data: List[Tuple],  # (inputs, outputs) from E+
        physics_weight: float = 0.1,  # Weight for physics loss
        epochs: int = 1000,
    ) -> 'BuildingPINN':
        """
        Train PINN with combined data + physics loss.

        Loss = MSE(predicted, actual) + physics_weight * physics_loss

        Physics loss enforces:
        - Energy balance (Q_in = Q_out + dU)
        - Non-negative thermal masses
        - Reasonable time constants
        """
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)

        for epoch in range(epochs):
            total_loss = 0

            for inputs, targets in training_data:
                # Data loss
                predictions = self.model(*inputs)
                data_loss = nn.MSELoss()(predictions, targets)

                # Physics loss (energy balance)
                physics_loss = self._compute_physics_loss(inputs, predictions)

                loss = data_loss + physics_weight * physics_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

        return self

    def _compute_physics_loss(self, inputs, predictions) -> torch.Tensor:
        """Compute physics constraint violation loss."""
        # Energy balance: sum of heat flows = rate of energy change
        # This is automatically satisfied by the RC model structure
        # Additional constraints can be added here
        return torch.tensor(0.0)

    def predict(
        self,
        initial_state: torch.Tensor,
        weather: torch.Tensor,
        hvac_schedule: torch.Tensor,
        horizon_hours: int = 24,
    ) -> torch.Tensor:
        """
        Predict building temperature over time horizon.

        Much faster than GP for multi-step prediction.
        """
        ...
```

### Deliverables Phase 4
- [ ] `pinn_surrogate.py` - RC thermal network PINN
- [ ] Integration with calibration pipeline
- [ ] Comparison study: GP vs PINN accuracy/speed
- [ ] GPU training scripts

---

## Phase 5: Agentic Orchestration (P2 - Scale)

### Goal
Portfolio-scale analysis (1000+ buildings) with autonomous agents that:
- Triage buildings by potential
- Handle errors autonomously
- Re-analyze low-confidence results
- Generate actionable reports

### Implementation

Already partially implemented in `src/orchestrator/`. Key additions:

```python
# src/orchestrator/agentic_pipeline.py

class RaidenAgentOrchestrator:
    """
    Multi-agent orchestrator for portfolio analysis.

    Agents:
    1. DataAgent - Fetches and validates building data
    2. AnalysisAgent - Runs calibration and simulation
    3. QCAgent - Validates results, triggers re-analysis
    4. ReportAgent - Generates recommendations
    """

    async def analyze_portfolio(
        self,
        addresses: List[str],
        parallel_workers: int = 50,
        quality_threshold: float = 0.70,
    ) -> PortfolioResult:
        """
        Analyze portfolio with autonomous agents.

        Flow:
        1. DataAgent fetches all building data in parallel
        2. TriageAgent prioritizes by potential savings
        3. AnalysisAgent runs calibration (surrogates for most)
        4. QCAgent checks results, flags anomalies
        5. DeepAnalysisAgent re-runs flagged buildings with E+
        6. ReportAgent aggregates and ranks
        """
        ...
```

### Deliverables Phase 5
- [ ] Enhance existing orchestrator with agent patterns
- [ ] Add retry/fallback logic for data fetching
- [ ] Parallel E+ execution (cloud burst)
- [ ] Portfolio dashboard (web UI)

---

## Implementation Timeline (Parallel Tracks)

```
Track A: HVAC (2-3 weeks implementation)
─────────────────────────────────────────────────────────────────
Week 1: District heating template + integration
Week 2: Heat pump templates (GSHP, ASHP, FTX-VP)
Week 3: Validation against Gripen declared values

Track B: Schedules (2 weeks implementation)
─────────────────────────────────────────────────────────────────
Week 1: Swedish pattern library + integration
Week 2: Pattern learner from meter data (if available)

Track C: Calibration (1-2 weeks implementation)
─────────────────────────────────────────────────────────────────
Week 1: ASHRAE metrics + context-aware priors
Week 2: Morris screening + train/test validation

Track D: Future (research phase)
─────────────────────────────────────────────────────────────────
Ongoing: PINN development, portfolio orchestration
```

---

## Success Metrics

| Metric | Current | Target | How to Measure |
|--------|---------|--------|----------------|
| Energy prediction accuracy | ±10-15% | ±5% | CVRMSE vs Gripen |
| Calibration CVRMSE | ~10% | <5% | ASHRAE metrics |
| HVAC system accuracy | N/A (IdealLoads) | 85% match | Gripen comparison |
| Schedule realism | Fixed | Building-specific | Load shape correlation |
| Portfolio throughput | ~10/hr | 100+/hr | Parallel orchestration |
| ECM ranking accuracy | Unknown | Top-3 in 80% | Expert validation |

---

## Files to Create/Modify

### New Files
```
src/hvac/
├── __init__.py
├── swedish_systems.py       # HVAC templates
├── hvac_selector.py         # Auto-selection
└── performance_curves.py    # HP COPs, etc.

src/schedules/
├── __init__.py
├── swedish_patterns.py      # Occupancy library
├── pattern_learner.py       # ML learning
└── schedule_generator.py    # IDF generation

src/calibration/
├── ashrae_metrics.py        # ASHRAE Guideline 14
├── pinn_surrogate.py        # PINN (future)
└── morris_screening.py      # Already exists, enhance
```

### Modified Files
```
src/baseline/generator_v2.py     # Add HVAC integration
src/baseline/archetypes.py       # Add HVAC defaults per era
src/calibration/bayesian.py      # Context-aware priors
src/calibration/surrogate.py     # Train/test validation
src/analysis/full_pipeline.py    # Wire everything together
```

---

## Quick Wins (Do First)

1. **Increase LHS samples** - Change 80 → 200 in `surrogate.py`
2. **Add train/test split** - Detect surrogate overfitting
3. **Use actual footprint** - Wire GeomEppyGenerator into full_pipeline
4. **Context-aware priors** - Use existing FTX detection

These changes require minimal code and provide immediate accuracy improvement.

---

## References

1. BeBo - Beställargruppen Bostäder (Swedish building owner consortium)
2. Sveby 2.0 - Swedish energy calculation conventions
3. BBR 6:23-6:24 - Boverket ventilation requirements
4. ASHRAE Guideline 14-2014 - Calibration procedures
5. SCB Time Use Survey - Swedish occupancy patterns
6. Energimyndigheten - Building sector statistics
7. Gokhale et al. (2022) - PINN for building thermal modeling
8. EPlus-LLM (2025) - Automated IDF generation
