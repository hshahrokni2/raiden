# Archetype Matching Pipeline Design

## Executive Summary

This document describes the optimal pipeline for matching a Swedish building address to the correct archetype from our 40 detailed archetypes. The approach uses a **hybrid algorithmic + AI pipeline** that is fast for obvious cases and accurate for ambiguous ones.

## Available Data Sources

| Source | Data Provided | Status |
|--------|---------------|--------|
| **Energy Declaration** | Construction year, Atemp, energy class, HVAC, ventilation, current kWh/m² | ✅ Working |
| **OSM/Overture** | Footprint, floors, height, facade material (sometimes), roof type | ✅ Working |
| **Mapillary** | Street-level facade images with orientation (N/S/E/W) | ✅ Working |
| **Google Solar API** | Roof segments, pitch, azimuth, solar potential | ✅ Working |
| **Address Geocoding** | Lat/lon, neighborhood, city | ✅ Working |

## Archetype Descriptor Fields

Each of our 40 archetypes has rich descriptors:

```python
@dataclass
class ArchetypeDescriptors:
    # Geometric
    building_depth_m: Tuple[float, float]
    floor_to_floor_m: Tuple[float, float]
    building_length_m: Tuple[float, float]
    plan_shape: List[PlanShape]

    # Visual (for AI matching)
    balcony_types: List[BalconyType]
    roof_profiles: List[RoofProfile]
    facade_patterns: List[FacadePattern]
    typical_colors: List[str]
    window_proportions: str
    has_bay_windows: bool
    has_corner_windows: bool

    # Contextual (for location matching)
    urban_settings: List[UrbanSetting]
    typical_neighborhoods: List[str]
    typical_cities: List[str]

    # Keywords (for text matching)
    keywords_sv: List[str]
    keywords_en: List[str]

    # Energy (for calibration)
    typical_certifications: List[EnergyCertification]
    infiltration_variability: str
    u_value_variability: str
```

---

## Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        ADDRESS INPUT                             │
│                    "Rinkeby Allé 10, Stockholm"                  │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                 STAGE 0: DATA AGGREGATION                        │
│                                                                  │
│   ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│   │   Energy    │  │    OSM /    │  │  Mapillary  │             │
│   │ Declaration │  │  Overture   │  │   Images    │             │
│   │             │  │             │  │             │             │
│   │ • year:1972 │  │ • floors:8  │  │ • 4 images  │             │
│   │ • kWh:135   │  │ • form:slab │  │ • N/S/E/W   │             │
│   │ • class:E   │  │ • material: │  │   oriented  │             │
│   │ • FV:yes    │  │   concrete  │  │             │             │
│   └──────┬──────┘  └──────┬──────┘  └──────┬──────┘             │
│          └────────────────┼────────────────┘                     │
│                           ▼                                      │
│              ┌─────────────────────────┐                        │
│              │   UnifiedBuildingData   │                        │
│              │   confidence: 0.95      │                        │
│              └─────────────────────────┘                        │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│              STAGE 1: ALGORITHMIC PRE-FILTER                     │
│                                                                  │
│   Step 1.1: Year-based filtering                                │
│   ─────────────────────────────                                 │
│   year=1972 → Candidates: [mfh_1961_1975, mfh_1976_1985]        │
│                                                                  │
│   Step 1.2: Hard attribute scoring                              │
│   ───────────────────────────────                               │
│   For each candidate, score:                                    │
│     • year_in_range:     +40 pts (if exact match)               │
│     • facade_material:   +20 pts (concrete → miljonprogram)     │
│     • building_form:     +15 pts (slab → skivhus)               │
│     • energy_class:      +10 pts (E matches typical)            │
│     • location_match:    +15 pts (Rinkeby in neighborhoods)     │
│                                                                  │
│   Step 1.3: Keyword matching                                    │
│   ──────────────────────────                                    │
│   Address contains "Rinkeby" → matches miljonprogram keywords   │
│     • +10 pts bonus                                             │
│                                                                  │
│   Results:                                                      │
│   ┌────────────────────────────────────────────┐                │
│   │  mfh_1961_1975 (Miljonprogrammet): 95 pts  │ ← WINNER       │
│   │  mfh_1976_1985 (Post oil crisis):  45 pts  │                │
│   │  skivhus_1965_1975 (Slab block):   85 pts  │ ← Close!       │
│   └────────────────────────────────────────────┘                │
│                                                                  │
│   DECISION LOGIC:                                               │
│   ┌──────────────────────────────────────────────────────────┐  │
│   │ IF top_score > 90 AND gap_to_second > 25:                │  │
│   │    → FAST PATH: Return winner (skip AI)                  │  │
│   │ ELSE:                                                    │  │
│   │    → Continue to Stage 2 with top 3 candidates           │  │
│   └──────────────────────────────────────────────────────────┘  │
│                                                                  │
│   In this case: 95 pts, but gap only 10 pts → CONTINUE TO AI    │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│               STAGE 2: AI VISUAL REFINEMENT                      │
│                                                                  │
│   Input: Facade images + top 3 candidates                       │
│                                                                  │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │                    VISION AI PROMPT                      │   │
│   │                                                          │   │
│   │  "Analyze this Swedish building facade image.            │   │
│   │                                                          │   │
│   │   The building was constructed in 1972 in Rinkeby,      │   │
│   │   Stockholm. Based on visual features, determine        │   │
│   │   which archetype best matches:                         │   │
│   │                                                          │   │
│   │   CANDIDATE 1: mfh_1961_1975 (Miljonprogrammet)         │   │
│   │   - Typical: concrete sandwich, projecting balconies,   │   │
│   │     flat roof, gray/beige colors, grid window pattern   │   │
│   │                                                          │   │
│   │   CANDIDATE 2: skivhus_1965_1975 (Slab block)           │   │
│   │   - Typical: long slab form >50m, gallery access,       │   │
│   │     repeated stairwell rhythm, horizontal emphasis      │   │
│   │                                                          │   │
│   │   CANDIDATE 3: mfh_1976_1985 (Post oil crisis)          │   │
│   │   - Typical: brick veneer, better proportioned          │   │
│   │     windows, pitched roof elements                      │   │
│   │                                                          │   │
│   │   For each, score 0-100 based on visual match.          │   │
│   │   Return JSON with scores and visual evidence."         │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                  │
│   AI Response:                                                  │
│   {                                                             │
│     "mfh_1961_1975": {                                          │
│       "score": 85,                                              │
│       "evidence": [                                             │
│         "Concrete sandwich panels visible",                     │
│         "Projecting balconies with metal railings",            │
│         "Flat roof confirmed",                                  │
│         "Beige/gray color scheme typical of era"               │
│       ]                                                         │
│     },                                                          │
│     "skivhus_1965_1975": {                                      │
│       "score": 92,                                              │
│       "evidence": [                                             │
│         "Building length >80m (slab form)",                    │
│         "Gallery access visible on north facade",              │
│         "8+ stories matches skivhus profile",                  │
│         "Repeated stairwell modules"                           │
│       ]                                                         │
│     },                                                          │
│     "mfh_1976_1985": {                                          │
│       "score": 25,                                              │
│       "evidence": [                                             │
│         "No brick visible - concrete facade",                  │
│         "Flat roof, not pitched"                               │
│       ]                                                         │
│     }                                                           │
│   }                                                             │
│                                                                  │
│   REFINEMENT: skivhus wins on visual! (More specific subtype)   │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│            STAGE 3: FINAL SCORING & EXPLANATION                  │
│                                                                  │
│   Combined Score = Algorithmic (60%) + AI Visual (40%)          │
│                                                                  │
│   ┌───────────────────────────────────────────────────────────┐ │
│   │  Archetype          │ Algo  │  AI   │ Combined │ Final   │ │
│   ├───────────────────────────────────────────────────────────┤ │
│   │  skivhus_1965_1975  │  85   │  92   │   88     │ WINNER  │ │
│   │  mfh_1961_1975      │  95   │  85   │   91     │ 2nd     │ │
│   │  mfh_1976_1985      │  45   │  25   │   37     │ 3rd     │ │
│   └───────────────────────────────────────────────────────────┘ │
│                                                                  │
│   Wait - mfh_1961_1975 scores higher combined!                  │
│   But skivhus IS a subtype of mfh_1961_1975...                  │
│                                                                  │
│   SUBTYPE LOGIC:                                                │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │ IF specific_subtype (skivhus) scores within 5 pts of    │   │
│   │    parent (mfh_1961_1975):                              │   │
│   │    → Prefer the MORE SPECIFIC archetype                 │   │
│   │    → Inherit parent properties, apply subtype modifiers │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                  │
│   FINAL RESULT:                                                 │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │  ArchetypeMatch(                                        │   │
│   │    archetype = skivhus_1965_1975,                       │   │
│   │    parent_archetype = mfh_1961_1975,                    │   │
│   │    confidence = 0.88,                                   │   │
│   │    reasoning = [                                        │   │
│   │      "✓ Construction year 1972 within 1965-1975 range", │   │
│   │      "✓ Located in Rinkeby (known Miljonprogram area)", │   │
│   │      "✓ Concrete sandwich facade (visual confirmation)",│   │
│   │      "✓ Slab block form >80m length detected",          │   │
│   │      "✓ Gallery access (loftgång) visible",             │   │
│   │      "✓ Energy class E typical for unrenovated"         │   │
│   │    ],                                                   │   │
│   │    alternatives = [                                     │   │
│   │      ("mfh_1961_1975", 0.91, "Generic Miljonprogram"),  │   │
│   │    ]                                                    │   │
│   │  )                                                      │   │
│   └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

---

## Decision Logic: When to Use AI

| Condition | Action | Reason |
|-----------|--------|--------|
| Top score > 90 AND gap > 25 | Skip AI | Clear winner, no ambiguity |
| Year on era boundary (±3 years) | Use AI | Could belong to either era |
| Multiple archetypes within 15 pts | Use AI | Need visual disambiguation |
| Special form detected (star, point) | Use AI | Confirm rare building type |
| Missing construction year | Use AI | Must infer from facade |
| High-performance keywords | Use AI | Distinguish passive/plus-energy |
| Heritage indicators | Use AI | Detect ornamental features |

---

## Scoring Weights

### Stage 1: Algorithmic Scoring (100 points max)

| Factor | Weight | Description |
|--------|--------|-------------|
| Year match | 40 pts | Construction year within archetype range |
| Facade material | 20 pts | Concrete/brick/plaster/wood match |
| Building form | 15 pts | Slab/point/lamell detection |
| Energy class | 10 pts | Aligns with typical for era |
| Location | 10 pts | City/neighborhood in descriptor list |
| Keywords | 5 pts | Address/description keyword matches |

### Stage 2: AI Visual Scoring (100 points max)

| Factor | Weight | Description |
|--------|--------|-------------|
| Facade pattern | 25 pts | Grid/ribbon/punched window pattern |
| Balcony style | 20 pts | Recessed/projecting/glazed/none |
| Material texture | 20 pts | Visual concrete vs brick vs render |
| Roof profile | 15 pts | Flat/pitched/mansard visible |
| Color scheme | 10 pts | Era-typical colors |
| Ornament level | 10 pts | Plain vs decorated (era indicator) |

### Final Combined Score

```
final_score = (algorithmic_score * 0.60) + (ai_score * 0.40)
confidence = final_score / 100
```

---

## Implementation Classes

### 1. UnifiedBuildingData

```python
@dataclass
class UnifiedBuildingData:
    """Aggregated building data from all sources."""
    # Core identifiers
    address: str
    latitude: float
    longitude: float

    # From Energy Declaration
    construction_year: Optional[int]
    atemp_m2: Optional[float]
    energy_class: Optional[str]  # A-G
    current_kwh_m2: Optional[float]
    heating_system: Optional[str]  # district, heat_pump, electric
    ventilation_type: Optional[str]  # FTX, F, natural
    heat_recovery_efficiency: Optional[float]

    # From OSM/Overture
    num_floors: Optional[int]
    building_height_m: Optional[float]
    footprint_area_m2: Optional[float]
    building_form: Optional[str]  # lamell, skiva, punkt, etc.
    facade_material: Optional[str]  # detected from tags
    roof_type: Optional[str]

    # From Mapillary
    facade_images: List[FacadeImage]  # With orientation

    # From Google Solar
    roof_segments: Optional[List[RoofSegment]]
    solar_potential_kwh: Optional[float]

    # Metadata
    data_sources: List[str]  # Which sources contributed
    confidence: float  # Overall data quality (0-1)

    # Derived
    city: Optional[str]
    neighborhood: Optional[str]
    building_type: str  # multi_family, single_family, terraced
```

### 2. ArchetypeMatchResult

```python
@dataclass
class ArchetypeMatchResult:
    """Result of archetype matching."""
    archetype: DetailedArchetype
    parent_archetype: Optional[DetailedArchetype]  # If subtype
    confidence: float  # 0-1

    # Scoring breakdown
    algorithmic_score: float
    ai_score: Optional[float]
    component_scores: Dict[str, float]

    # Explanation
    reasoning: List[str]  # Human-readable bullet points
    visual_evidence: List[str]  # From AI analysis

    # Alternatives
    alternatives: List[Tuple[DetailedArchetype, float, str]]

    # Calibration hints
    recommended_adjustments: Dict[str, float]  # e.g., {"u_wall": -0.05}
```

### 3. ArchetypeMatcherV2

```python
class ArchetypeMatcherV2:
    """Hybrid algorithmic + AI archetype matcher."""

    def match(
        self,
        building_data: UnifiedBuildingData,
        use_ai: bool = True,  # Enable AI refinement
        ai_threshold: float = 0.15,  # Gap threshold to trigger AI
    ) -> ArchetypeMatchResult:
        """
        Match building to best archetype.

        Pipeline:
        1. Algorithmic pre-filter (always)
        2. AI visual refinement (if ambiguous or use_ai=True)
        3. Combined scoring and explanation
        """
        # Stage 1: Algorithmic
        candidates = self._algorithmic_prefilter(building_data)

        # Check if AI needed
        top_score = candidates[0].score
        gap = top_score - candidates[1].score if len(candidates) > 1 else 100

        if not use_ai or (top_score > 90 and gap > 25):
            # Fast path - clear winner
            return self._finalize_result(candidates[0], ai_used=False)

        # Stage 2: AI refinement
        ai_scores = self._ai_visual_analysis(
            building_data.facade_images,
            candidates[:3]
        )

        # Stage 3: Combine and explain
        return self._combine_scores(candidates, ai_scores, building_data)

    def _algorithmic_prefilter(
        self,
        data: UnifiedBuildingData
    ) -> List[ScoredCandidate]:
        """Score all archetypes algorithmically."""
        scores = []
        for archetype in get_all_archetypes().values():
            score = 0

            # Year match (40 pts)
            if data.construction_year:
                if archetype.year_start <= data.construction_year <= archetype.year_end:
                    score += 40
                elif abs(data.construction_year - archetype.year_start) <= 3:
                    score += 25  # Boundary case

            # Facade material (20 pts)
            if data.facade_material and archetype.descriptors:
                # Map material to typical archetypes
                score += self._score_material_match(
                    data.facade_material, archetype
                )

            # Building form (15 pts)
            if data.building_form and archetype.descriptors:
                if data.building_form in [s.value for s in archetype.descriptors.plan_shape]:
                    score += 15

            # Energy class (10 pts)
            if data.energy_class and archetype.descriptors:
                expected_classes = [c.value for c in archetype.descriptors.typical_certifications]
                if data.energy_class in expected_classes:
                    score += 10

            # Location (10 pts)
            if archetype.descriptors:
                if data.city in archetype.descriptors.typical_cities:
                    score += 5
                if data.neighborhood in archetype.descriptors.typical_neighborhoods:
                    score += 5

            # Keywords (5 pts)
            if archetype.descriptors:
                address_lower = data.address.lower()
                for kw in archetype.descriptors.keywords_sv:
                    if kw.lower() in address_lower:
                        score += 5
                        break

            scores.append(ScoredCandidate(archetype, score, {}))

        return sorted(scores, key=lambda x: x.score, reverse=True)

    def _ai_visual_analysis(
        self,
        images: List[FacadeImage],
        candidates: List[ScoredCandidate]
    ) -> Dict[str, AIScore]:
        """Use vision AI to score candidates based on facade images."""
        if not images:
            return {}

        # Build prompt with candidate descriptors
        prompt = self._build_vision_prompt(images, candidates)

        # Call vision API (Claude, GPT-4V, etc.)
        response = self._call_vision_api(images[0].url, prompt)

        return self._parse_ai_response(response)
```

---

## Configuration

```python
# src/core/config.py additions

ARCHETYPE_MATCHING = {
    # Scoring weights
    "algorithmic_weight": 0.60,
    "ai_weight": 0.40,

    # AI trigger thresholds
    "fast_path_score": 90,  # Skip AI if top score > this
    "fast_path_gap": 25,    # AND gap > this

    # AI provider
    "vision_api": "claude",  # or "openai", "google"
    "vision_model": "claude-3-5-sonnet-20241022",

    # Caching
    "cache_ai_results": True,
    "cache_ttl_hours": 24 * 30,  # 30 days
}
```

---

## Example Usage

```python
from src.core.building_data_aggregator import BuildingDataAggregator
from src.baseline.archetype_matcher_v2 import ArchetypeMatcherV2

# Aggregate all building data
aggregator = BuildingDataAggregator()
building_data = aggregator.fetch("Rinkeby Allé 10, Stockholm")

# Match to archetype
matcher = ArchetypeMatcherV2()
result = matcher.match(building_data)

print(f"Archetype: {result.archetype.name_en}")
print(f"Confidence: {result.confidence:.0%}")
print(f"Reasoning:")
for reason in result.reasoning:
    print(f"  {reason}")

# Output:
# Archetype: Slab Block (Skivhus) 1965-1975
# Confidence: 88%
# Reasoning:
#   ✓ Construction year 1972 within 1965-1975 range
#   ✓ Located in Rinkeby (known Miljonprogram area)
#   ✓ Concrete sandwich facade (visual confirmation)
#   ✓ Slab block form >80m length detected
#   ✓ Gallery access (loftgång) visible
#   ✓ Energy class E typical for unrenovated
```

---

## Performance Targets

| Metric | Target | Notes |
|--------|--------|-------|
| Algorithmic-only latency | < 100ms | No external API calls |
| With AI latency | < 3s | Vision API call included |
| Accuracy (obvious cases) | > 95% | Year clearly in single era |
| Accuracy (ambiguous) | > 85% | Boundary years, rare types |
| False positive rate | < 5% | Wrong era completely |

---

## Next Steps

1. **Implement `BuildingDataAggregator`** - Unified data fetching
2. **Implement `ArchetypeMatcherV2`** - Hybrid matching logic
3. **Create vision prompt templates** - Per archetype visual cues
4. **Add CLI command** - `raiden match-archetype ADDRESS`
5. **Build evaluation dataset** - 100+ buildings with known archetypes
6. **Tune weights** - Based on evaluation results
