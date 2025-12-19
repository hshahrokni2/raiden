# Raiden - Swedish Building ECM Simulator

## Mission

**Automated energy conservation measure (ECM) analysis for ANY Swedish building using only public data.**

Given just an address, automatically:
1. Fetch building data from public sources
2. Generate calibrated baseline energy model
3. Identify valid ECMs (constraint-aware)
4. Simulate all sensible combinations
5. Output ROI-ranked recommendations

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                          RAIDEN                                  │
│                                                                  │
│  INPUT: Address or Organization Number                          │
│         ▼                                                        │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ DATA FUSION (src/ingest/)                                 │   │
│  │ • OSM/Overture → footprint, height, floors               │   │
│  │ • Mapillary → facade material, WWR per orientation       │   │
│  │ • LiDAR (Lantmäteriet) → roof geometry, PV potential     │   │
│  │ • Energy declaration → actual kWh/m², heating system     │   │
│  └──────────────────────────────────────────────────────────┘   │
│         ▼                                                        │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ GEOMETRY (src/geometry/) - NEW                            │   │
│  │ • Wall areas per orientation (N/S/E/W)                   │   │
│  │ • Window areas from WWR                                   │   │
│  │ • PV potential (roof area, slope, shading)               │   │
│  │ • Thermal mass from materials                            │   │
│  └──────────────────────────────────────────────────────────┘   │
│         ▼                                                        │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ BASELINE (src/baseline/) - NEW                            │   │
│  │ • Archetype matching (7 Swedish eras defined)            │   │
│  │ • Auto-generate EnergyPlus IDF                           │   │
│  │ • Calibrate to energy declaration (±10%)                 │   │
│  └──────────────────────────────────────────────────────────┘   │
│         ▼                                                        │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ ECM ENGINE (src/ecm/) - NEW                               │   │
│  │ • 12 Swedish ECMs defined with constraints               │   │
│  │ • Constraint-aware: NO facade insulation on brick        │   │
│  │ • Combination generator (pruned, no dominated options)   │   │
│  │ • IDF modifier (apply ECMs to baseline)                  │   │
│  └──────────────────────────────────────────────────────────┘   │
│         ▼                                                        │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ SIMULATION (src/simulation/) - NEW                        │   │
│  │ • Parallel EnergyPlus execution                          │   │
│  │ • Results parsing                                         │   │
│  └──────────────────────────────────────────────────────────┘   │
│         ▼                                                        │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ ROI (src/roi/) - NEW                                      │   │
│  │ • Swedish cost database (2024 SEK)                       │   │
│  │ • Payback, NPV, IRR calculations                         │   │
│  │ • Ranked recommendations                                  │   │
│  └──────────────────────────────────────────────────────────┘   │
│         ▼                                                        │
│  OUTPUT: Ranked ECM list, packages (Basic/Standard/Premium)     │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Module Status

| Module | Status | Key Files |
|--------|--------|-----------|
| `src/ingest/` | ✅ COMPLETE | **sweden_buildings** ⭐, brf_parser, overture_fetcher, image_fetcher, energidek_parser |
| `src/geometry/` | ✅ COMPLETE | building_geometry, pv_potential, thermal_mass |
| `src/baseline/` | ✅ COMPLETE | archetypes, generator, calibrator, **llm_archetype_reasoner**, archetype_matcher_v2 |
| `src/ecm/` | ✅ COMPLETE | catalog, constraints, combinations, idf_modifier |
| `src/simulation/` | ✅ COMPLETE | runner, results |
| `src/roi/` | ✅ COMPLETE | costs_sweden, calculator |
| `src/core/` | ✅ COMPLETE | building_context, idf_parser, **address_pipeline** |
| `src/analysis/` | ✅ COMPLETE | building_analyzer, package_simulator, roof_analyzer |
| `src/planning/` | ✅ COMPLETE | **MaintenancePlan, CashFlowSimulator, ECMSequencer, EffektvaktOptimizer** |
| `src/reporting/` | ✅ COMPLETE | html_report (with maintenance plan + effektvakt sections) |
| `src/cli/` | ✅ COMPLETE | main.py (CLI with **analyze-address** command) |
| `src/api/` | ✅ COMPLETE | **FastAPI REST API** |

**✅ COMPLETE** = Fully implemented and tested
**🔄 IN PROGRESS** = Partially implemented

## Key Data Structures

### Swedish Archetypes (COMPLETE)
7 eras defined in `src/baseline/archetypes.py`:
- Pre-1945 Brick
- 1945-1960 Brick (Folkhemmet)
- 1961-1975 Concrete Panel (Miljonprogrammet)
- 1976-1985 Insulated
- 1986-1995 Well Insulated
- 1996-2010 Modern
- 2011+ Low Energy

### Swedish Building Forms (NEW)
Building forms in `src/baseline/building_forms.py`:
- **LAMELLHUS**: Slab block (3-4 stories, 1945-1985)
- **SKIVHUS**: Large slab (8+ stories, miljonprogrammet)
- **PUNKTHUS**: Point block tower (8+ stories, compact)
- **STJÄRNHUS**: Star-shaped (3+ wings, 1950s-60s)
- **LOFTGÅNGSHUS**: Gallery access (external corridors)
- **SLUTET_KVARTER**: Closed perimeter block (pre-1940)
- **VINKELBYGGNAD**: L-shaped buildings
- **RADHUS**: Row houses / terraced

Each form has:
- Compactness factor (surface-to-volume ratio)
- Thermal bridge factor (adjusted U-values)
- Typical WWR by orientation
- Common construction methods
- Era-specific characteristics

### ECM Catalog (COMPLETE)
22 ECMs defined in `src/ecm/catalog.py`:
- **Envelope** (5): wall_external_insulation, wall_internal_insulation, roof_insulation, window_replacement, air_sealing
- **HVAC** (4): ftx_upgrade, ftx_installation, demand_controlled_ventilation, heat_pump_integration
- **Renewable** (1): solar_pv
- **Controls** (2): smart_thermostats, led_lighting
- **Operational** (10): duc_calibration, effektvakt_optimization, heating_curve_adjustment, ventilation_schedule_optimization, radiator_balancing, night_setback, summer_bypass, hot_water_temperature, pump_optimization, bms_optimization

Each ECM has:
- Parameters (with ranges)
- Constraints (e.g., `facade_material not_in ['brick']`)
- Swedish costs (SEK)
- Typical savings

```python
# Quick access
from src.ecm import get_all_ecms, get_ecm, ECMCategory

ecms = get_all_ecms()  # 22 ECMs
ecm = get_ecm('wall_external_insulation')
```

### Swedish Costs (COMPLETE)
Defined in `src/roi/costs_sweden.py`:
- Energy prices (district heating, electricity, etc.)
- ECM costs per unit (SEK/m², SEK/kW, etc.)
- Carbon intensities

## Critical Implementation Notes

### EnergyPlus 25.1.0 Bug Workaround
When using `ZoneHVAC:IdealLoadsAirSystem` with heat recovery:
```
ConstantSupplyHumidityRatio,  !- Dehumidification Control Type (NOT 'None'!)
,                              !- Cooling Sensible Heat Ratio (BLANK!)
ConstantSupplyHumidityRatio,  !- Humidification Control Type (NOT 'None'!)
```
Using `None` causes segmentation fault. This is documented in `DEVELOPMENT_LOG.md`.

### Constraint Examples
```python
# Brick facade → NO external insulation
ECMConstraint("facade_material", "not_in", ["brick"],
              "Cannot add external insulation to brick facade")

# Heritage building → NO exterior changes
ECMConstraint("heritage_listed", "eq", False,
              "Heritage buildings cannot have exterior changes")

# Already efficient → NO upgrade
ECMConstraint("current_heat_recovery", "lt", 0.80,
              "Already high efficiency, limited benefit")
```

## Implementation Priority

1. **Geometry module** - Calculate areas from OSM footprint
2. **Baseline generator** - Auto-generate IDF from archetype
3. **IDF modifier** - Apply ECMs to baseline
4. **Simulation runner** - Parallel E+ execution
5. **Results parser** - Extract annual energy
6. **ROI calculator** - Financial metrics

## Example Usage (Target API)

```python
from raiden import analyze_building

results = analyze_building(
    address="Sjöstaden 2, Stockholm",
    # OR
    org_number="769612-1234"
)

# Results:
# - Baseline: 95 kWh/m²
# - Top ECMs ranked by ROI
# - Packages: Basic (7yr payback), Standard (9yr), Premium (12yr)
```

## Existing Working Example

The Sjostaden model (`sjostaden_7zone.idf`) is a calibrated 7-zone model:
- 2,240 m² multi-family (2003 construction)
- **Calibrated**: 36.3 kWh/m² heating (declared: 33 kWh/m², 10% gap)
- FTX with 82% heat recovery (calibrated from 75%)
- Infiltration: 0.04 ACH (calibrated from 0.06)
- Window U-value: 0.85 W/m²K (calibrated from 1.0)

### Latest ECM Results (2025-12-18)
```
Baseline:               36.3 kWh/m²
DCV:                   18.4 kWh/m² (−49%)
Wall Insulation:       31.4 kWh/m² (−14%)
Air Sealing:           32.5 kWh/m² (−10%)
Heat Pump Integration: 35.1 kWh/m² (−3%)
Smart Thermostats:     35.5 kWh/m² (−2%)
Roof Insulation:       35.5 kWh/m² (−2%)
LED Lighting:          40.8 kWh/m² (+12%)  # Explained in report
```

## Data Sources (All Free/Public)

| Source | Data | Access | Status |
|--------|------|--------|--------|
| **Sweden Buildings GeoJSON** ⭐ | 37,489 buildings with 167 properties each | Local file | ✅ PRIMARY |
| **Overture Maps** | Footprint, height, floors, material | CLI/DuckDB | ✅ Fallback |
| **Microsoft Building Footprints** | 1.4B buildings, Sweden (399 files) | Direct download | ✅ Fallback |
| **Lantmäteriet** | Official Swedish buildings, LiDAR | Open data | Partial |
| **Mapillary** | Street view facades | API key | ✅ Integrated |
| **Google Solar API** | Roof analysis, PV potential | API key | ✅ Integrated |
| **Energideklaration** | Energy use, year, heating | Boverket | ✅ Integrated |
| **Sveby** | Load defaults | Published | ✅ Used |

### Building Data Priority (for Sweden)
1. **Sweden Buildings GeoJSON** ⭐ - `data/sweden_buildings.geojson` (37,489 Stockholm buildings with energy data!)
2. **Overture Maps** (combines OSM + Microsoft) - `src/ingest/overture_fetcher.py`
3. **Microsoft Building Footprints** - Direct download for 20% with heights
4. **Lantmäteriet "Byggnad Nedladdning"** - Official Swedish data
5. ~~Google Open Buildings~~ - Does NOT cover Sweden/Europe

**Note:** The Sweden Buildings GeoJSON is THE RICHEST source with actual energy declarations, heating types, ventilation, solar, and more. Use it first! Fall back to Overture/Microsoft for addresses not in the dataset.

## Files to Read

1. `src/ingest/sweden_buildings.py` ⭐ - **37,489 buildings with 167 properties** (READ FIRST!)
2. `src/baseline/archetypes.py` - Swedish building archetypes (COMPLETE)
3. `src/ecm/catalog.py` - ECM definitions (COMPLETE)
4. `src/ecm/constraints.py` - Constraint engine
5. `src/roi/costs_sweden.py` - Swedish costs (COMPLETE)
6. `examples/sjostaden_2/energyplus/TECHNICAL_NOTES.md` - Model details
7. `examples/sjostaden_2/energyplus/DEVELOPMENT_LOG.md` - E+ debugging

## Next Steps

**Completed (2025-12-18):**
- ✅ Smart ECM filtering (existing measures detection)
- ✅ Full end-to-end pipeline with EnergyPlus simulation
- ✅ CLI tool with batch processing support
- ✅ All 12 ECM modifiers working
- ✅ HTML report generator with Swedish text
- ✅ Package generation (Steg 0-3: Nollkostnad → Premium)
- ✅ Swedish weather file auto-download (11 cities)
- ✅ Model calibration (27% gap → 10% gap)
- ✅ Heat pump ECM fixed (was showing 0%)
- ✅ LED heating interaction explained in report
- ✅ **Address-to-Report Pipeline** (`analyze-address` command!)
- ✅ **Maintenance Plan Builder** (BRF long-term planning)
- ✅ **Cash Flow Cascade** (zero-cost → fund larger investments)
- ✅ **Effektvakt Optimizer** (peak shaving with thermal mass)
- ✅ **FastAPI REST API** (web integration)
- ✅ **Geocoding with Nominatim** (address → coordinates)
- ✅ **Building Form Detection** (lamellhus, skivhus, punkthus, etc.)
- ✅ **Mapillary Integration** (facade images from street view)
- ✅ **Form-Aware Thermal Modeling** (adjusted U-values per form)
- ✅ **Sweden Buildings GeoJSON** (37,489 buildings with 167 properties)
- ✅ **Mapillary + AI Facade Analysis** (WWR detection, material classification)
- ✅ **LLM-Enhanced Archetype Reasoner** (renovation detection, anomaly detection with Komilion API)
- ✅ **40 Detailed Swedish Archetypes** (with descriptors for matching)

**Remaining:**
1. PostgreSQL/PostGIS database storage
2. Real PDF extraction (automated energy declaration parsing)
3. Boverket API integration (auto-fetch energy declarations)
4. Web frontend (React/Vue)
5. Mapillary ML facade analysis (detect material/WWR from images)

## Quick Start

```bash
# THE VISION: Analyze building by address!
python -m src.cli.main analyze-address "Aktergatan 11, Stockholm" --year 2003 --apartments 110

# Run analysis on a building JSON
python -m src.cli.main analyze examples/sjostaden_2/BRF_Sjostaden_2_enriched.json

# List available ECMs
python -m src.cli.main ecm-list

# Run full E2E demo with simulations
python examples/sjostaden_2/run_full_analysis.py

# Run maintenance plan demo
python examples/sjostaden_2/maintenance_plan_demo.py

# Start REST API
uvicorn src.api.main:app --reload
```

## New Features (2025-12-18)

### Address-to-Report Pipeline
```python
from src.core.address_pipeline import analyze_address

result = analyze_address("Aktergatan 11, Stockholm")
print(f"Report: {result.report_path}")
print(f"NPV: {result.maintenance_plan.net_present_value_sek:,.0f} SEK")
```

### Maintenance Plan with Cash Flow Cascade
```python
from src.planning import ECMSequencer, CashFlowSimulator, BRFFinancials

financials = BRFFinancials(
    current_fund_sek=2_500_000,
    annual_fund_contribution_sek=500_000,
    current_avgift_sek_month=4800,
    num_apartments=110,
)

sequencer = ECMSequencer()
plan = sequencer.create_optimal_plan(candidates, financials, renovations)

simulator = CashFlowSimulator()
plan = simulator.simulate(plan, start_year=2025)

# Results: Break-even, NPV, fund balance trajectory
```

### Effektvakt (Peak Shaving) Analysis
```python
from src.planning import analyze_effektvakt_potential

result = analyze_effektvakt_potential(
    atemp_m2=15350,
    construction_year=2003,
    heating_type="heat_pump",
    current_el_peak_kw=120,
)

print(f"Coast duration: {result.coast_duration_hours:.1f} hours")
print(f"Annual savings: {result.total_annual_savings_sek:,.0f} SEK")
```

### REST API
```bash
# Start server
uvicorn src.api.main:app --reload

# Analyze building
curl -X POST "http://localhost:8000/analyze/address" \
  -H "Content-Type: application/json" \
  -d '{"address": "Aktergatan 11, Stockholm", "construction_year": 2003}'

# Get report
curl "http://localhost:8000/report/{analysis_id}"
```

### Building Form Detection
```python
from src.baseline import (
    ArchetypeMatcher, BuildingType, BuildingForm,
    get_form_properties, detect_building_form
)

# Match archetype with form detection
matcher = ArchetypeMatcher()
result = matcher.match_with_form(
    construction_year=1970,
    building_type=BuildingType.MULTI_FAMILY,
    facade_material='concrete',
    stories=10,
    width_m=15,
    length_m=80
)

print(f"Archetype: {result.archetype.name}")
print(f"Building form: {result.building_form.value}")  # skivhus
print(f"Base wall U: {result.archetype.envelope.wall_u_value}")
print(f"Adjusted wall U: {result.adjusted_envelope.wall_u_value}")  # +15% for thermal bridges

# Form properties
props = get_form_properties(BuildingForm.LOFTGANGSHUS)
print(f"Compactness: {props.compactness_factor}")  # 1.6
print(f"Thermal bridge factor: {props.thermal_bridge_factor}")  # 1.3
```

### Mapillary Integration
```python
from src.ingest.image_fetcher import FacadeImageFetcher

# Requires MAPILLARY_TOKEN environment variable
fetcher = FacadeImageFetcher()

# Fetch facade images by direction
images = fetcher.fetch_for_building(
    building_coords=[(18.06, 59.30), ...],  # Footprint
    building_id="my_building",
    search_radius_m=100,
)

for direction, imgs in images.items():
    print(f"{direction}: {len(imgs)} images")
```

---

## COMPREHENSIVE DATA INFRASTRUCTURE (READ THIS FIRST!)

This section documents ALL existing data sources and AI modules. **DO NOT reinvent these - they already exist!**

### 1. AI MODULES (`src/ai/`)

#### WWR Detector (`src/ai/wwr_detector.py`)
**Status: FULLY IMPLEMENTED**
- Detects Window-to-Wall Ratio from facade images
- Backends: `opencv` (default), `sam`, `lang_sam`
- Returns per-orientation WWR (N/S/E/W)

```python
from src.ai.wwr_detector import WWRDetector

detector = WWRDetector(backend="opencv", device="cpu")
wwr_result = detector.calculate_wwr(pil_image)
# Returns: WindowToWallRatio(north=0.15, south=0.25, east=0.20, west=0.20, average=0.20)
```

#### Material Classifier (`src/ai/material_classifier.py`)
**Status: FULLY IMPLEMENTED**
- Classifies facade materials from images
- Uses DINOv2 features or heuristic color/texture analysis
- Materials: brick, concrete, plaster, glass, metal, wood, stone

```python
from src.ai.material_classifier import MaterialClassifier

classifier = MaterialClassifier(device="cpu")
prediction = classifier.classify(pil_image)
# Returns: MaterialPrediction(material=FacadeMaterial.CONCRETE, confidence=0.85, all_scores={...})
```

### 2. ROOF & PV ANALYSIS (`src/analysis/roof_analyzer.py`, `src/geometry/pv_potential.py`)

#### Roof Analyzer
**Status: FULLY IMPLEMENTED with Google Solar API**
- Analyzes roof type, segments, pitch, azimuth
- Detects obstructions (HVAC, skylights, chimneys)
- Detects existing solar installations
- **Uses Google Solar API when GOOGLE_API_KEY is set**

```python
from src.analysis.roof_analyzer import RoofAnalyzer

analyzer = RoofAnalyzer()
analysis = analyzer.analyze(latitude=59.30, longitude=18.10, footprint_area_m2=500)
# Returns: RoofAnalysis with total_area, roof_type, segments, obstructions, existing_solar, pv_potential
```

#### PV Potential Calculator
**Status: FULLY IMPLEMENTED**
- Swedish irradiance data for 5 climate zones
- Orientation & tilt optimization
- Shading loss calculations
- System losses (inverter, soiling, wiring)

```python
from src.geometry.pv_potential import calculate_pv_potential

result = calculate_pv_potential(
    roof_area_m2=300,
    roof_type="flat",
    roof_slope_deg=10,
    latitude=59.3,  # Stockholm
)
# Returns: max_capacity_kwp, annual_yield_kwh, optimal_tilt, shading_loss_factor
```

### 3. BUILDING GEOMETRY (`src/geometry/building_geometry.py`)

**Status: FULLY IMPLEMENTED**
- Calculates wall areas per orientation (N/S/E/W)
- Window areas from WWR
- Roof geometry (flat vs pitched)
- Volume and perimeter calculations
- GeoJSON/polygon parsing

```python
from src.geometry.building_geometry import BuildingGeometry

geom = BuildingGeometry.from_geojson(footprint_geojson, height_m=15, floors=5)
print(f"Total wall area: {geom.total_wall_area_m2} m²")
print(f"South facade: {geom.facades['S'].wall_area_m2} m²")
```

### 4. SWEDISH BUILDINGS GEOJSON (`src/ingest/sweden_buildings.py`)

**Status: FULLY IMPLEMENTED** ⭐ **RICHEST DATA SOURCE**

Local GeoJSON with **37,489 buildings** and **167 properties per building** including:
- Building footprints (EPSG:3006 SWEREF99 TM with 3D heights)
- Energy declarations (energy class A-G, kWh/m²)
- Heating systems (district heating, heat pumps, oil, gas, pellets)
- Ventilation type (F, FT, FTX, natural draft)
- Solar (PV, thermal)
- Construction year, Atemp, number of apartments/floors
- Full address data (street, postal code, city, municipality)

**File**: `data/sweden_buildings.geojson` (272MB)

**Dataset Statistics:**
```
Total buildings: 37,489
Energy Classes: A(66), B(740), C(2,629), D(5,798), E(9,917), F(6,613), G(3,055)
Building Types: Single-family(16,893), Multi-family(15,857), Commercial(4,739)
Ventilation: S(14,289), F(9,839), FTX(8,180), FT(1,051)
With Solar PV: 413
With Heat Pump: 5,067
```

```python
from src.ingest import SwedenBuildingsLoader, load_sweden_buildings, find_building_by_address

# Load all buildings
loader = load_sweden_buildings()
stats = loader.get_statistics()

# Find by address
buildings = loader.find_by_address("Bellmansgatan")

# Find by location (WGS84)
buildings = loader.find_by_location(lat=59.30, lon=18.10, radius_m=100)

# Access rich data
for b in buildings[:1]:
    print(f"Address: {b.address}, {b.city}")
    print(f"Year: {b.construction_year}, Atemp: {b.atemp_m2} m²")
    print(f"Energy class: {b.energy_class} ({b.energy_performance_kwh_m2} kWh/m²)")
    print(f"Heating: {b.get_primary_heating()}")
    print(f"Ventilation: {b.ventilation_type}")
    print(f"Has solar PV: {b.has_solar_pv}")
    print(f"District heating: {b.district_heating_kwh} kWh")
    print(f"Ground source HP: {b.ground_source_hp_kwh} kWh")
```

**SwedishBuilding Properties:**
- `building_id`, `uuid` - Identifiers
- `address`, `postal_code`, `city`, `municipality`, `county` - Location
- `footprint_coords`, `footprint_area_m2`, `height_m` - Geometry
- `construction_year`, `atemp_m2`, `num_apartments`, `num_floors` - Building
- `energy_class`, `energy_performance_kwh_m2` - Energy
- `district_heating_kwh`, `ground_source_hp_kwh`, `exhaust_air_hp_kwh`, etc. - Heating
- `ventilation_type` - F, FT, FTX, S (natural)
- `has_solar_pv`, `has_solar_thermal`, `solar_pv_kwh` - Solar
- `raw_properties` - Access all 167 original properties

**NOTE:** Currently Stockholm-focused. Use as primary source, fall back to Overture/OSM for other cities.

### 5. BUILDING FOOTPRINT FETCHERS

#### Overture Maps (`src/ingest/overture_fetcher.py`)
**Status: FULLY IMPLEMENTED** (Primary source for non-Stockholm)
- Combines OSM + Microsoft Building Footprints
- Returns: height, num_floors, facade_material, roof_material, roof_shape
- Uses: CLI or DuckDB S3 access

```python
from src.ingest import OvertureFetcher

fetcher = OvertureFetcher()
buildings = fetcher.get_buildings_in_bbox(min_lon, min_lat, max_lon, max_lat)
# Returns: List of GeoJSON features with footprint polygons
```

#### Microsoft Building Footprints (`src/ingest/microsoft_buildings.py`)
**Status: FULLY IMPLEMENTED** (Fallback)
- 1.4B buildings globally, Sweden has 399 files
- ~20% have height estimates
- Direct Azure blob storage access

```python
from src.ingest import get_microsoft_buildings, MicrosoftBuildingsFetcher

# Quick access
buildings = get_microsoft_buildings(lat=59.30, lon=18.10, radius_m=500)

# Or with fetcher instance
fetcher = MicrosoftBuildingsFetcher()
buildings = fetcher.get_buildings_for_location(59.30, 18.10, search_radius_m=500)
```

### 5. IMAGE FETCHING (`src/ingest/image_fetcher.py`)

**Status: FULLY IMPLEMENTED**
- **Mapillary** (primary): Street-level images with compass angle
- **WikiMedia Commons**: Fallback, no auth needed
- **KartaView**: Open alternative
- **Manual upload**: EXIF GPS extraction

```python
from src.ingest.image_fetcher import FacadeImageFetcher

fetcher = FacadeImageFetcher()  # Requires MAPILLARY_TOKEN
images = fetcher.fetch_for_building(building_coords, search_radius_m=100)
# Returns: Dict[str, List[FacadeImage]] by direction (N, S, E, W)
```

### 6. ADDRESS PIPELINE (`src/core/address_pipeline.py`)

**Status: FULLY IMPLEMENTED with smart fallback chain**

Data source priority (automatic fallback):
1. **Sweden Buildings GeoJSON** ⭐ - 37,489 Stockholm buildings with 167 properties
2. **OSM/Overture** - Footprint, height, floors, tags
3. **Mapillary** - Facade images
4. **Nominatim** - Geocoding fallback

```python
from src.core.address_pipeline import BuildingDataFetcher

fetcher = BuildingDataFetcher()

# If building is in GeoJSON → uses real energy data (83% confidence)
data = fetcher.fetch("Bellmansgatan 16")
# Returns: BuildingData with REAL energy class, heating type, ventilation, solar...

# If building NOT in GeoJSON → falls back to OSM (40% confidence)
data = fetcher.fetch("Some Address in Malmö")
# Returns: BuildingData with inferred values from OSM + geocoding
```

**What you get from Sweden Buildings GeoJSON:**
- ✓ Construction year, Atemp, floors, apartments
- ✓ Energy class (A-G) and kWh/m²
- ✓ Heating system (district, heat pumps, electric, oil, gas)
- ✓ Ventilation type (F, FT, FTX, natural)
- ✓ Solar (PV and thermal)
- ✓ Building footprint and height

**Fallback for non-Stockholm buildings:**
1. Nominatim geocoding → coordinates
2. OSM Overpass → footprint, height, floors, tags
3. **Mapillary + AI Analysis** → WWR detection, facade material classification
4. Building form detection → lamellhus/skivhus/etc.
5. Era-based inference → LAST RESORT only (if AI unavailable)

### 7. DETAILED ARCHETYPES (`src/baseline/archetypes_detailed.py`)

**Status: FULLY IMPLEMENTED - 40 ARCHETYPES**

Comprehensive Swedish building archetypes with:
- Wall/roof/floor U-values
- Ventilation systems
- Window constructions
- **ArchetypeDescriptors** (balcony types, roof profiles, facade patterns, colors, typical neighborhoods)

**Archetype Categories:**
- `SWEDISH_MFH_ARCHETYPES`: 15 multi-family by era
- `SWEDISH_SFH_ARCHETYPES`: Single-family
- `SWEDISH_TERRACED_ARCHETYPES`: Radhus/row houses
- `SWEDISH_HISTORICAL_ARCHETYPES`: Pre-1930
- `SWEDISH_HIGH_PERFORMANCE_ARCHETYPES`: Passive house, plus-energy
- `SWEDISH_SPECIAL_FORM_ARCHETYPES`: Stockholm-specific (skivhus, punkthus)
- `MILJONPROGRAMMET_SUBTYPES`: Variants within 1961-1975

```python
from src.baseline import get_all_archetypes, get_archetype

all_archs = get_all_archetypes()  # 40 archetypes
arch = get_archetype("mfh_1961_1975")  # Million Programme

print(f"Wall U: {arch.wall_constructions[0].u_value}")
print(f"Typical colors: {arch.descriptors.typical_colors}")
print(f"Typical neighborhoods: {arch.descriptors.typical_neighborhoods}")
```

### 8. ARCHETYPE MATCHER V2 (`src/baseline/archetype_matcher_v2.py`)

**Status: FULLY INTEGRATED with all data sources**

Scores 40 archetypes using REAL DATA from BuildingData:

| Data Source | Scoring Weight | Data Used |
|-------------|----------------|-----------|
| Sweden Buildings GeoJSON | 35 pts | year, energy_class, ventilation, heating |
| OSM/Overture | 15 pts | floors, building_form |
| Mapillary AI | 20 pts | WWR detection, material classification |
| Google Solar | 10 pts | roof type, existing PV |
| Location | 10 pts | neighborhood matching |

```python
from src.baseline import ArchetypeMatcherV2
from src.core.address_pipeline import BuildingDataFetcher

# Get building data (uses Sweden GeoJSON → fallback to OSM → AI)
fetcher = BuildingDataFetcher()
building_data = fetcher.fetch("Bellmansgatan 16")

# Match archetype using ALL real data
matcher = ArchetypeMatcherV2(use_ai_modules=True)
result = matcher.match_from_building_data(building_data)

print(f"Archetype: {result.archetype.name_en}")
print(f"Confidence: {result.confidence:.0%}")
print(f"Scores: {result.source_scores}")  # Shows per-source breakdown
```

**What's scored:**
- `construction_year` → matches era (1961-1975 = Miljonprogrammet)
- `energy_class` → matches expected for era (F/G = older buildings)
- `has_ftx` → FTX typical for 1986+ buildings
- `heating_system` → district/heat_pump/electric
- `facade_material` → brick/concrete/plaster/etc.
- `building_form` → lamellhus/skivhus/punkthus
- `num_floors` → matches typical for form
- `wwr` → window-to-wall ratio from AI detection
- `address` → neighborhood matching

### 9. RAIDEN LLM INTELLIGENCE (`src/baseline/llm_archetype_reasoner.py`)

**Status: FULLY IMPLEMENTED with Komilion Premium Mode**

Raiden's LLM-enhanced reasoning layer (using Komilion premium mode by default):
- **Renovation Detection**: Identifies when old buildings have been upgraded (e.g., 1905 building with FTX + energy class C → detected upgrades)
- **Anomaly Detection**: Flags unusual combinations (e.g., 2013 building with energy class D → possible construction issues)
- **Chain-of-Thought Reasoning**: Explains why a building matches or doesn't match expected patterns
- **Calibration Hints**: Suggests parameter adjustments based on detected anomalies

```python
from src.baseline import LLMArchetypeReasoner, enhance_archetype_match

# Raiden uses Komilion premium mode by default
raiden = LLMArchetypeReasoner(api_key="sk-komilion-...")

# Analyze building for renovations
result = raiden.reason_about_building(building_data, candidates)

# Result includes:
# - best_archetype: Selected archetype ID
# - confidence: Match confidence (0-1)
# - reasoning_chain: Step-by-step analysis
# - renovation_analysis: Detected upgrades (ventilation, windows, envelope)
# - anomalies: Unusual patterns found
# - calibration_hints: Parameter adjustment suggestions
```

**Renovation Analysis:**
```python
result.renovation_analysis:
  - is_renovated: True
  - renovation_confidence: 0.95
  - likely_upgrades: ['ventilation_upgrade', 'window_replacement', 'envelope_insulation']
  - original_era: 'PRE_1930'
  - estimated_renovation_year: 2015
```

**Example - Raiden Renovation Detection:**
```
Input: Birger Jarlsgatan 102A, Stockholm
- Construction year: 1905
- Energy class: C (good for such old building!)
- Ventilation: FTX (unusual for pre-1930)

Raiden Analysis (premium mode):
"A 1905 building with FTX ventilation and energy class C indicates
significant renovation. Pre-1930 buildings typically have energy class F/G
with natural ventilation. This building likely underwent:
1. Complete ventilation system upgrade (FTX)
2. Window replacement (triple-glazed)
3. Envelope improvements
Recommend using POST_OIL_CRISIS archetype (1976-1985) U-values."

Renovation confidence: 95%
Detected upgrades: ventilation_upgrade, window_replacement, envelope_insulation
```

### 10. BUILDING CONTEXT (`src/core/building_context.py`)

**Status: FULLY IMPLEMENTED**

Central data aggregation structure:
- `EnhancedBuildingContext`: All building data in one place
- `BuildingContextBuilder`: Orchestrates data from all sources
- `ExistingMeasuresDetector`: Detects what's already implemented (FTX, heat pumps, solar, etc.)
- `SmartECMFilter`: Two-stage filtering (technical + existing measures)

```python
from src.core.building_context import BuildingContextBuilder

builder = BuildingContextBuilder(use_v2_matcher=True)
context = builder.build_from_declaration(energy_declaration, geometry_data)
# Returns: EnhancedBuildingContext with existing_measures, applicable ECMs, etc.
```

---

## DATA SOURCE FLOW

```
ADDRESS
    │
    ▼
┌───────────────────────────────────────────────────────────────┐
│ Step 1: Sweden Buildings GeoJSON (TRY FIRST!)                │
│  ⭐ 37,489 Stockholm buildings with 167 properties            │
│  • If found: DONE! Have energy class, heating, ventilation,  │
│              solar, construction year, Atemp, footprint!     │
│  • Confidence: 83%                                            │
└───────────────────────────────────────────────────────────────┘
    │ Not found?
    ▼
┌───────────────────────────────────────────────────────────────┐
│ Step 2: OSM/Overture + Nominatim                              │
│  • Nominatim → coordinates                                    │
│  • OSM Overpass → footprint, height, floors, tags            │
│  • Building form detection → lamellhus/skivhus/punkthus      │
└───────────────────────────────────────────────────────────────┘
    │
    ▼
┌───────────────────────────────────────────────────────────────┐
│ Step 3: Mapillary + AI Analysis                               │
│  • Fetch street view images (N/S/E/W facades)                │
│  • WWRDetector → window-to-wall ratio from images            │
│  • MaterialClassifier → facade material (brick, concrete...) │
│  • Confidence: 60-70%                                         │
└───────────────────────────────────────────────────────────────┘
    │ AI unavailable?
    ▼
┌───────────────────────────────────────────────────────────────┐
│ Step 4: Era-Based Inference (LAST RESORT)                     │
│  • Construction year → estimated U-values                     │
│  • Era → facade material guess                                │
│  • Confidence: 40%                                            │
└───────────────────────────────────────────────────────────────┘
    │
    ▼
┌───────────────────────────────────────────────────────────────┐
│ Additional AI Analysis (always runs if available)            │
│  • RoofAnalyzer + Google Solar API → roof type, PV potential │
└───────────────────────────────────────────────────────────────┘
    │
    ▼
┌───────────────────────────────────────────────────────────────┐
│ ArchetypeMatcherV2                                            │
│  • Score from energy declaration (year, energy class)        │
│  • Score from geometry (form, floors)                        │
│  • Score from AI analysis (WWR, material, roof)              │
│  • Score from location (neighborhood matching)               │
│  → Best match from 40 archetypes                             │
└───────────────────────────────────────────────────────────────┘
    │
    ▼
┌───────────────────────────────────────────────────────────────┐
│ Raiden LLM Intelligence (Komilion Premium)                    │
│  • Renovation detection (old building + good energy class?)  │
│  • Anomaly detection (unusual combinations)                  │
│  • Chain-of-thought reasoning                                │
│  • Calibration hints for EnergyPlus parameter adjustment     │
│  → Enhanced match with renovation analysis (95% accuracy)    │
└───────────────────────────────────────────────────────────────┘
    │
    ▼
┌───────────────────────────────────────────────────────────────┐
│ BuildingContextBuilder                                        │
│  • Aggregate all data sources                                │
│  • Detect existing measures (FTX, heat pumps, solar)         │
│  • Apply archetype defaults                                   │
│  → EnhancedBuildingContext                                   │
└───────────────────────────────────────────────────────────────┘
    │
    ▼
┌───────────────────────────────────────────────────────────────┐
│ ECM Analysis                                                  │
│  • ConstraintEngine → technically applicable ECMs            │
│  • SmartECMFilter → exclude already-implemented measures     │
│  • Simulation → energy savings                               │
│  • ROI calculation → ranked recommendations                  │
└───────────────────────────────────────────────────────────────┘
```

---

## ENVIRONMENT VARIABLES

```bash
# Required for full functionality
MAPILLARY_TOKEN=your_token          # For facade images
GOOGLE_API_KEY=your_key             # For Google Solar API
ANTHROPIC_API_KEY=your_key          # For Claude Vision analysis
KOMILION_API_KEY=your_key           # For LLM archetype reasoning (optional)

# Optional
ENERGYPLUS_IDD_PATH=/path/to/Energy+.idd
WEATHER_FILE_DIR=/path/to/weather/files
```

---

## WHAT'S REMAINING

1. **PostgreSQL/PostGIS database** - Store results for multi-building portfolios
2. **Boverket API integration** - Auto-fetch energy declarations
3. **PDF extraction** - Parse energy declaration PDFs automatically
4. **Web frontend** - React/Vue dashboard
5. **Satellite imagery analysis** - Detect rooftop PV from aerial images

---

## KEY PRINCIPLE

**USE REAL DATA - DON'T INFER!**

We have actual data from:
- Energy declarations → real energy class, real heating system, real ventilation type
- Mapillary → real facade images for AI analysis
- OSM/Overture → real footprints and heights
- Google Solar API → real roof analysis

The archetype matcher scores based on this REAL DATA, not guesswork.
