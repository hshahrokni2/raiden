# Raiden - Swedish Building ECM Simulator

> **ARCHITECTURE REFERENCE:** Use `@PROJECT_INDEX.json` for codebase navigation - contains function signatures, call graphs, and file organization.
>
> **IMPORTANT:** See [`docs/SYSTEM_MAP.md`](docs/SYSTEM_MAP.md) for the authoritative map of all components, their connections, and integration status.

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
│  │ GEOMETRY (src/geometry/)                                  │   │
│  │ • Wall areas per orientation (N/S/E/W)                   │   │
│  │ • Window areas from WWR                                   │   │
│  │ • PV potential (roof area, slope, shading)               │   │
│  │ • Thermal mass from materials                            │   │
│  └──────────────────────────────────────────────────────────┘   │
│         ▼                                                        │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ BASELINE (src/baseline/)                                  │   │
│  │ • Archetype matching (7 Swedish eras defined)            │   │
│  │ • Auto-generate EnergyPlus IDF                           │   │
│  │ • Calibrate to energy declaration (±10%)                 │   │
│  └──────────────────────────────────────────────────────────┘   │
│         ▼                                                        │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ ECM ENGINE (src/ecm/)                                      │   │
│  │ • 51 Swedish ECMs defined with constraints               │   │
│  │ • Constraint-aware: NO facade insulation on brick        │   │
│  │ • Combination generator (pruned, no dominated options)   │   │
│  │ • IDF modifier (apply ECMs to baseline)                  │   │
│  └──────────────────────────────────────────────────────────┘   │
│         ▼                                                        │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ SIMULATION (src/simulation/)                              │   │
│  │ • Parallel EnergyPlus execution                          │   │
│  │ • Results parsing                                         │   │
│  └──────────────────────────────────────────────────────────┘   │
│         ▼                                                        │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ ROI (src/roi/)                                            │   │
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
| `src/calibration/` | ✅ COMPLETE | **bayesian**, surrogate, sensitivity, pipeline, metrics |
| `src/ecm/` | ✅ COMPLETE | catalog, constraints, combinations, idf_modifier |
| `src/simulation/` | ✅ COMPLETE | runner, results |
| `src/roi/` | ✅ COMPLETE | costs_sweden, calculator |
| `src/core/` | ✅ COMPLETE | building_context, idf_parser, **address_pipeline** |
| `src/analysis/` | ✅ COMPLETE | building_analyzer, package_simulator, roof_analyzer |
| `src/planning/` | ✅ COMPLETE | **MaintenancePlan, CashFlowSimulator, ECMSequencer, EffektvaktOptimizer** |
| `src/reporting/` | ✅ COMPLETE | html_report (with maintenance plan + effektvakt sections) |
| `src/orchestrator/` | ✅ COMPLETE | **RaidenOrchestrator**, prioritizer, qc_agent, surrogate_library, portfolio_report |
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
**51 ECMs** defined in `src/ecm/catalog.py` with **50 having thermal simulation effects**:
- **Envelope** (9): wall_external_insulation, wall_internal_insulation, roof_insulation, window_replacement, air_sealing, basement_insulation, thermal_bridge_remediation, facade_renovation, entrance_door_replacement
- **HVAC** (12): ftx_upgrade, ftx_installation, ftx_overhaul, demand_controlled_ventilation, heat_pump_integration, exhaust_air_heat_pump, ground_source_heat_pump, air_source_heat_pump, heat_pump_water_heater, vrf_system, radiator_fans, heat_recovery_dhw
- **Renewable** (3): solar_pv, solar_thermal, battery_storage
- **Controls** (7): smart_thermostats, occupancy_sensors, daylight_sensors, predictive_control, fault_detection, individual_metering, building_automation_system
- **Lighting** (3): led_lighting, led_common_areas, led_outdoor
- **Operational** (17): duc_calibration, effektvakt_optimization, heating_curve_adjustment, ventilation_schedule_optimization, radiator_balancing, night_setback, summer_bypass, hot_water_temperature, pump_optimization, bms_optimization, district_heating_optimization, energy_monitoring, recommissioning, dhw_circulation_optimization, dhw_tank_insulation, pipe_insulation, low_flow_fixtures

Each ECM has:
- Parameters (with ranges)
- Constraints (e.g., `facade_material not_in ['brick']`)
- Swedish costs (SEK) with V2 regional multipliers
- Typical savings

```python
# Quick access
from src.ecm import get_all_ecms, get_ecm, ECMCategory

ecms = get_all_ecms()  # 51 ECMs
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
| **Sweden Buildings GeoJSON** ⭐ | 37,489 buildings with 167 properties each | Local file | ✅ PRIMARY (Stockholm) |
| **Microsoft Building Footprints** | 1.4B buildings, Sweden (399 files) | Direct download | ✅ FALLBACK (footprints) |
| **Gripen** | Energy declarations nationwide (1.37M buildings) | Local file | ✅ IMPLEMENTED (fallback energy) |
| **Lantmäteriet** | Official Swedish buildings, LiDAR | Open data | Partial |
| **Mapillary** | Street view facades | API key | ✅ Integrated |
| **Google Solar API** | Roof analysis, PV potential | API key | ✅ Integrated |
| **Energideklaration** | Energy use, year, heating | Boverket | ✅ Integrated |
| **Sveby** | Load defaults | Published | ✅ Used |
| ~~**Overture Maps**~~ | ~~Footprint, height, floors, material~~ | ~~CLI/DuckDB~~ | ⚠️ DEPRECATED |

### Building Data Priority (for Sweden)
1. **Sweden Buildings GeoJSON** ⭐ - `data/sweden_buildings.geojson` (37,489 Stockholm buildings with energy data!)
2. **Microsoft Building Footprints + Gripen** - Primary fallback for buildings outside Stockholm
3. **Lantmäteriet "Byggnad Nedladdning"** - Official Swedish data
4. ~~Overture Maps~~ - DEPRECATED (OSM data quality inconsistent)
5. ~~Google Open Buildings~~ - Does NOT cover Sweden/Europe

**Note:** The Sweden Buildings GeoJSON is THE RICHEST source with actual energy declarations, heating types, ventilation, solar, and more. Use it first! For buildings outside Stockholm, use Microsoft Building Footprints (geometry) + Gripen (energy data).

## Files to Read

1. `src/ingest/sweden_buildings.py` ⭐ - **37,489 buildings with 167 properties** (READ FIRST!)
2. `src/baseline/archetypes.py` - Swedish building archetypes (COMPLETE)
3. `src/ecm/catalog.py` - ECM definitions (COMPLETE)
4. `src/ecm/constraints.py` - Constraint engine
5. `src/roi/costs_sweden.py` - Swedish costs (COMPLETE)
6. `docs/CALIBRATION_SYSTEM.md` ⭐ - **Full Bayesian calibration system architecture** (NEW!)
7. `examples/sjostaden_2/energyplus/TECHNICAL_NOTES.md` - Model details
8. `examples/sjostaden_2/energyplus/DEVELOPMENT_LOG.md` - E+ debugging

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
3. Boverket API integration (for cities OUTSIDE Stockholm - Stockholm already has energy data in GeoJSON!)
4. Web frontend (React/Vue)

**Note:** Mapillary facade analysis is ALREADY IMPLEMENTED via `WWRDetector` and `MaterialClassifier` in `src/ai/`.

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

**NOTE:** Currently Stockholm-focused. Use as primary source, fall back to Microsoft Building DB + Gripen for other cities.

### 5. BUILDING FOOTPRINT FETCHERS

#### Microsoft Building Footprints (`src/ingest/microsoft_buildings.py`)
**Status: FULLY IMPLEMENTED** (Primary fallback for non-Stockholm footprints)
- 1.4B buildings globally, Sweden has 399 files
- ~20% have height estimates
- Direct Azure blob storage access
- **USE THIS for buildings outside Stockholm**

```python
from src.ingest import get_microsoft_buildings, MicrosoftBuildingsFetcher

# Quick access
buildings = get_microsoft_buildings(lat=59.30, lon=18.10, radius_m=500)

# Or with fetcher instance
fetcher = MicrosoftBuildingsFetcher()
buildings = fetcher.get_buildings_for_location(59.30, 18.10, search_radius_m=500)
```

#### Gripen Energy Declarations (`src/ingest/gripen_loader.py`)
**Status: FULLY IMPLEMENTED** (Fallback for non-Stockholm energy data)
- ~1.37 million buildings nationwide (2019-2024)
- 205 fields per building including energy class, kWh/m², heating, ventilation
- Official Boverket energy declaration database
- **USE THIS for energy data outside Stockholm**

```python
from src.ingest import GripenLoader, load_gripen, find_gripen_building

# Load most recent year (2024)
loader = GripenLoader(years=[2024])

# Find by address
buildings = loader.find_by_address("Kungsgatan 1", city="Stockholm")

# Find by municipality
buildings = loader.find_by_municipality("Göteborg")

# Quick lookup
building = find_gripen_building("Vasagatan 10", city="Malmö")
if building:
    print(f"Energy class: {building.energy_class}")
    print(f"kWh/m²: {building.specific_energy_kwh_m2}")
    print(f"Heating: {building.get_primary_heating()}")
    print(f"Has FTX: {building.has_ftx}")
```

**GripenBuilding Properties:**
- `formular_id`, `property_designation` - Identifiers
- `address`, `postal_code`, `city`, `municipality_name`, `county_name` - Location
- `construction_year`, `atemp_m2`, `num_apartments`, `num_floors` - Building
- `energy_class`, `specific_energy_kwh_m2`, `total_energy_kwh` - Energy
- `has_ftx`, `has_f_only`, `has_natural_draft`, `ventilation_airflow_ls_m2` - Ventilation
- `district_heating_space_kwh`, `ground_source_hp_kwh`, etc. - Heating breakdown
- `has_solar_pv`, `solar_pv_production_kwh` - Solar
- `get_primary_heating()` - Determine dominant heating source
- `get_zone_breakdown()` - Get mixed-use area fractions
- `raw_properties` - Access all 205 original fields

### 6. IMAGE FETCHING (`src/ingest/image_fetcher.py`)

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

### 7. ADDRESS PIPELINE (`src/core/address_pipeline.py`)

**Status: FULLY IMPLEMENTED with smart fallback chain**

Data source priority (automatic fallback):
1. **Sweden Buildings GeoJSON** ⭐ - 37,489 Stockholm buildings with 167 properties
2. **Microsoft Building Footprints + Gripen** - Footprint + energy data for non-Stockholm
3. **Mapillary** - Facade images for AI analysis
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
2. **Microsoft Building Footprints** → footprint polygons (1.4B globally)
3. **Gripen** → energy declarations nationwide (✅ IMPLEMENTED - 1.37M buildings)
4. **Mapillary + AI Analysis** → WWR detection, facade material classification
5. Building form detection → lamellhus/skivhus/etc.
6. Era-based inference → LAST RESORT only (if AI unavailable)

### 8. DETAILED ARCHETYPES (`src/baseline/archetypes_detailed.py`)

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

### 9. ARCHETYPE MATCHER V2 (`src/baseline/archetype_matcher_v2.py`)

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

# Get building data (uses Sweden GeoJSON → Microsoft + Gripen → AI)
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

### 10. RAIDEN LLM INTELLIGENCE (`src/baseline/llm_archetype_reasoner.py`)

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

### 11. BUILDING CONTEXT (`src/core/building_context.py`)

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

## DATA SOURCE FLOW (VERIFIED 2025-12-28)

**All components below are WIRED INTO `full_pipeline.py` and tested.**

```
ADDRESS
    │
    ▼
┌───────────────────────────────────────────────────────────────┐
│ Step 1a: Sweden Buildings GeoJSON (TRY FIRST!)               │
│  ⭐ 37,489 Stockholm buildings with 167 properties            │
│  • If found: Have energy class, heating, ventilation,        │
│              solar, construction year, Atemp, footprint!     │
│  • Location: full_pipeline.py lines 1140-1224                │
│  • Confidence: 83%                                            │
└───────────────────────────────────────────────────────────────┘
    │ Not found in Stockholm GeoJSON?
    ▼
┌───────────────────────────────────────────────────────────────┐
│ Step 1b: Gripen Energy Declarations (NATIONWIDE FALLBACK)    │
│  ⭐ 830,610 buildings across ALL of Sweden                    │
│  • Energy class, heating system, ventilation, solar          │
│  • Renovation history tracking (changes over years!)         │
│  • Location: full_pipeline.py lines 1226-1290                │
│  • Confidence: 80%                                            │
└───────────────────────────────────────────────────────────────┘
    │ No footprint from GeoJSON?
    ▼
┌───────────────────────────────────────────────────────────────┐
│ Step 1c: Microsoft Building Footprints (IF NO FOOTPRINT)     │
│  ⭐ 1.4 billion buildings globally, high-quality ML polygons  │
│  • Building footprint when not in Stockholm GeoJSON          │
│  • Height estimates (~20% of buildings have this)            │
│  • Location: full_pipeline.py lines 1292-1325                │
│  • Confidence: 75%                                            │
└───────────────────────────────────────────────────────────────┘
    │
    ▼
┌───────────────────────────────────────────────────────────────┐
│ Step 2-4: Google Street View + AI Analysis (ALWAYS RUNS)     │
│  ⭐ 36+ multi-angle images (3 positions × 3 pitches × 4 dirs) │
│  • Historical imagery (3+ years back for renovation detect)  │
│  • WWRDetector → window-to-wall ratio per orientation        │
│  • MaterialClassifierV2 → facade material (CLIP + SAM)       │
│  • ImageQualityAssessor → filter blurry/occluded             │
│  • GroundFloorDetector → commercial detection                │
│  • Location: full_pipeline.py lines 1377-1640                │
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

## BAYESIAN CALIBRATION IMPROVEMENT PLAN (2025-12-21)

**Full documentation:** See `docs/CALIBRATION_SYSTEM.md` for complete architecture, data flows, and implementation details.

Based on literature review of ASHRAE Guideline 14, Kennedy & O'Hagan Bayesian framework, ABC-SMC methods, and GP surrogate modeling.

### Problem Statement
- FTX detected (80% efficiency) but calibrated to 41% → priors too broad
- Surrogate R² = 1.0 (overfitting) → no train/test validation
- 24/49 ECMs have no IDF implementation → silent failures
- Package interactions oversimplified → negative savings in some cases

### Phase 1: Fix Calibration Quality (CRITICAL)
**Files:** `src/calibration/bayesian.py`, `src/calibration/surrogate.py`

1. **Context-aware priors** - Use detected existing measures to constrain priors
   - FTX detected → heat_recovery_eff ∈ Beta(α=8, β=2) on [0.65, 0.90]
   - Air sealing done → infiltration_ach ∈ N(0.04, 0.01)
   - Add `CalibrationPriors.from_building_context(context, archetype_id)`

2. **Surrogate cross-validation** - Detect overfitting
   - 80/20 train/test split
   - Report test_r2 separately (expect 0.85-0.95, not 1.0)
   - Flag if train_r2 - test_r2 > 0.10

3. **ASHRAE metrics** - Report calibration quality
   - NMBE < ±10% (hourly), < ±5% (monthly)
   - CVRMSE < 30% (hourly), < 15% (monthly)

### Phase 2: Improve Surrogate Quality
**Files:** `src/calibration/surrogate.py`

1. **Kernel selection** - Switch to Matern 5/2 (physical systems not infinitely smooth)
2. **Hyperparameter tuning** - Increase n_restarts_optimizer to 30
3. **Sample size** - Increase default from 80 to 150
4. **Progressive LHS** - Adaptive sampling to avoid over/under-sampling

### Phase 3: Parameter Screening (Morris Method)
**Files:** `src/calibration/screening.py` (NEW)

1. **Morris screening** - Pre-calibration parameter importance
   - Run r(k+1) = 80 simulations for 7 parameters
   - Identify influential parameters (μ* > threshold)
   - Fix non-influential at archetype defaults

2. **Reduced calibration** - Only calibrate 3-5 identifiable parameters
   - Avoids over-parameterization (Kennedy & O'Hagan warning)
   - Improves posterior precision

### Phase 4: ECM Improvements
**Files:** `src/ecm/idf_modifier.py`, `src/analysis/package_generator.py`

1. **Missing ECM handlers** - Implement 24 missing ECMs or remove from catalog
   - wall_external_insulation regex fix
   - basement_insulation, facade_renovation, etc.

2. **ECM interaction matrix** - Replace fixed 0.70 factor
   - Positive synergies: (wall + window) = 1.15
   - Negative synergies: (FTX + DCV) = 0.75

3. **Catalog cost sync** - Use catalog costs, not hardcoded in package_generator

### ASHRAE Guideline 14 Calibration Criteria

| Resolution | NMBE Limit | CV(RMSE) Limit | R² Minimum |
|------------|------------|----------------|------------|
| Monthly    | ±5%        | 15%            | 75%        |
| Hourly     | ±10%       | 30%            | 75%        |

### Latin Hypercube Sampling Recommendations

| Parameters | Minimum | Recommended | Production |
|------------|---------|-------------|------------|
| 7          | 70      | 150-200     | 500-1000   |

### Key Literature Sources
- ASHRAE Guideline 14-2002: Calibration metrics
- Kennedy & O'Hagan (2001): Bayesian calibration framework
- Chong & Menberg (2018): Guidelines for building energy Bayesian calibration
- Morris (1991): Sensitivity screening method

---

## PIPELINE INTEGRITY CHECKLIST (FOR FUTURE SESSIONS)

**IMPORTANT:** Before making changes, verify these components are still wired:

### Data Source Chain (in `src/analysis/full_pipeline.py`)
| Step | Component | Lines | Import |
|------|-----------|-------|--------|
| 1a | Sweden GeoJSON | 1140-1224 | `from ..ingest.sweden_buildings import SwedenBuildingsLoader` |
| 1b | Gripen Energy | 1226-1290 | `from ..ingest.gripen_loader import GripenLoader` |
| 1c | Microsoft Footprints | 1292-1325 | `from ..ingest.microsoft_buildings import get_microsoft_buildings` |
| 2+ | Google Street View | 1377-1640 | `from ..ingest.streetview_fetcher import StreetViewFacadeFetcher` |
| 2+ | Google Solar API | via `_fetch_google_solar` | `from ..analysis.roof_analyzer import RoofAnalyzer` |

### AI Modules (in `src/ai/`)
| Module | File | Used In |
|--------|------|---------|
| WWR Detection | `wwr_detector.py` | `full_pipeline.py` |
| Material V2 | `material_classifier_v2.py` | `full_pipeline.py` |
| Image Quality | `image_quality.py` | `full_pipeline.py` |
| Ground Floor | `ground_floor_detector.py` | `full_pipeline.py` |

### ECM Modifiers (in `src/ecm/idf_modifier.py`)
- All 50 ECMs with thermal effects have handlers
- Roof insulation regex fixed 2025-12-28 (was not matching commented IDF format)

### Quick Verification Command
```bash
# Verify all imports work
python -c "from src.analysis.full_pipeline import FullPipelineAnalyzer; print('OK')"

# Verify data sources are imported
python -c "
from src.ingest.sweden_buildings import SwedenBuildingsLoader
from src.ingest.gripen_loader import GripenLoader
from src.ingest.microsoft_buildings import get_microsoft_buildings
print('All data sources importable')
"
```

---

## WHAT'S REMAINING

1. **PostgreSQL/PostGIS database** - Store results for multi-building portfolios
2. **Boverket API integration** - Only needed for cities OUTSIDE Stockholm (Stockholm has 37,489 buildings with real energy data in GeoJSON!)
3. **PDF extraction** - Parse energy declaration PDFs automatically
4. **Web frontend** - React/Vue dashboard

**Note:** Satellite/Google Solar API roof analysis is ALREADY IMPLEMENTED in `src/analysis/roof_analyzer.py`.

---

## 🗺️ BATTLE MAP (2025-12-22) - SESSION STATE

### ✅ COMPLETED THIS SESSION

**Session 1: Calibration Phase 1-4**

| Task | Status | Key Files | Notes |
|------|--------|-----------|-------|
| Context-aware priors | ✅ DONE | `bayesian.py:from_building_context()` | FTX → tight heat recovery prior |
| Calibration hints from LLM | ✅ DONE | `bayesian.py`, `pipeline.py`, `building_context.py` | Window/wall/roof adjustments |
| Morris sensitivity screening | ✅ DONE | `sensitivity.py` | MorrisScreening, run_morris_analysis |
| FixedParamPredictor | ✅ DONE | `surrogate.py` | Inject fixed params for reduced calibration |
| MC uncertainty propagation | ✅ DONE | `bayesian.py:ECMUncertaintyPropagator` | Replace sqrt(2) approximation |
| ECM parameter effects | ✅ DONE | `bayesian.py:ECM_PARAMETER_EFFECTS` | Maps ECMs to surrogate modifications |
| All 37 calibration tests | ✅ PASS | `tests/test_calibration.py` | Morris + MC + hints tests |

**Session 2: RaidenOrchestrator (Portfolio-Scale Analysis)**

| Task | Status | Key Files | Notes |
|------|--------|-----------|-------|
| RaidenOrchestrator class | ✅ DONE | `orchestrator.py` | Tiered processing (Fast/Standard/Deep) |
| BuildingPrioritizer | ✅ DONE | `prioritizer.py` | 8 prioritization strategies |
| ImageQCAgent | ✅ DONE | `qc_agent.py` | Re-analyze low-confidence images |
| ECMRefinerAgent | ✅ DONE | `qc_agent.py` | Handle negative savings, interactions |
| AnomalyAgent | ✅ DONE | `qc_agent.py` | Investigate unusual patterns |
| SurrogateLibrary | ✅ DONE | `surrogate_library.py` | Pre-trained GP per archetype |
| PortfolioAnalytics | ✅ DONE | `portfolio_report.py` | Aggregate metrics, top buildings |
| All 32 orchestrator tests | ✅ PASS | `tests/test_orchestrator.py` | Full coverage |

### 🎯 RAIDEN ORCHESTRATOR ARCHITECTURE (COMPLETE)

**Capability**: Analyze 1000+ buildings in parallel with agentic QC and refinement.

```
┌─────────────────────────────────────────────────────────────────┐
│                    RAIDEN ORCHESTRATOR                          │
│                                                                  │
│  INPUT: Portfolio (CSV/DB of addresses)                         │
│         ▼                                                        │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ TIER 1: Fast Triage (10 buildings/sec)                    │   │
│  │ • Sweden GeoJSON lookup → instant energy data             │   │
│  │ • Confidence scoring → route to appropriate tier          │   │
│  │ • Skip analysis if energy class A/B (already optimized)  │   │
│  └──────────────────────────────────────────────────────────┘   │
│         ▼                                                        │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ TIER 2: Standard Analysis (parallel, 50 concurrent)       │   │
│  │ • Archetype matching (ArchetypeMatcherV2)                 │   │
│  │ • Pre-trained surrogate lookup                           │   │
│  │ • ECM savings estimation (no E+ simulation)              │   │
│  │ • Flag for QC if confidence < 70%                        │   │
│  └──────────────────────────────────────────────────────────┘   │
│         ▼                                                        │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ TIER 3: Deep Analysis (agentic, 10 concurrent)           │   │
│  │ • Full Bayesian calibration                              │   │
│  │ • EnergyPlus simulation                                   │   │
│  │ • LLM reasoning for anomalies                            │   │
│  └──────────────────────────────────────────────────────────┘   │
│         ▼                                                        │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ AGENTIC QC (triggered by low confidence)                  │   │
│  │ A. Image QC: Re-analyze facades if WWR conf < 60%        │   │
│  │ B. ECM Refinement: Adjust packages per building context  │   │
│  │ C. Anomaly Resolution: LLM explains unusual patterns     │   │
│  └──────────────────────────────────────────────────────────┘   │
│         ▼                                                        │
│  OUTPUT: Portfolio report, priority rankings, budget optimizer  │
└─────────────────────────────────────────────────────────────────┘
```

### 📁 Files Created (ALL COMPLETE)

| File | Purpose | Status |
|------|---------|--------|
| `src/orchestrator/__init__.py` | Module exports | ✅ |
| `src/orchestrator/orchestrator.py` | RaidenOrchestrator core class | ✅ |
| `src/orchestrator/prioritizer.py` | Building prioritization strategies | ✅ |
| `src/orchestrator/qc_agent.py` | Agentic QC (image re-analysis, ECM refinement) | ✅ |
| `src/orchestrator/surrogate_library.py` | Pre-trained surrogates for 40 archetypes | ✅ |
| `src/orchestrator/portfolio_report.py` | Aggregate portfolio analytics | ✅ |
| `tests/test_orchestrator.py` | 32 tests (all passing) | ✅ |

### 🔑 Key Design Decisions

1. **Tiered Processing**: Fast triage (GeoJSON) → Standard (surrogate) → Deep (E+ sim)
2. **Pre-trained Surrogates**: Train once per archetype, reuse for all buildings
3. **Agentic QC**: Only triggered when confidence < threshold, not for every building
4. **Parallel Execution**: asyncio + ProcessPoolExecutor for CPU-bound E+ sims
5. **Uncertainty Propagation**: ECMUncertaintyPropagator for all recommendations

### 📊 Pre-trained Surrogate Library

For each of 40 archetypes, pre-train GP surrogates:
```python
SURROGATE_LIBRARY = {
    "mfh_1961_1975": {
        "surrogate_path": "surrogates/mfh_1961_1975_gp.pkl",
        "train_r2": 0.94,
        "test_r2": 0.89,
        "params": ["infiltration_ach", "wall_u_value", "heat_recovery_eff", ...],
        "trained_date": "2025-12-22",
    },
    # ... 39 more
}
```

### 🎯 Agentic QC Triggers

| Trigger | Action | Agent |
|---------|--------|-------|
| WWR confidence < 60% | Re-fetch Mapillary, try different angles | ImageQCAgent |
| Material confidence < 70% | Use LLM vision on multiple images | ImageQCAgent |
| Archetype score < 50 pts | Flag for human review or LLM reasoning | ArchetypeQCAgent |
| ECM savings negative | Check interaction matrix, adjust | ECMRefinerAgent |
| Energy class mismatch > 2 | Investigate renovation history | AnomalyAgent |

### 📈 Portfolio Analytics

```python
@dataclass
class PortfolioAnalytics:
    total_buildings: int
    analyzed: int
    skipped_already_optimized: int
    flagged_for_qc: int

    total_savings_potential_kwh: float
    total_investment_sek: float
    portfolio_npv_sek: float
    portfolio_payback_years: float

    top_10_buildings: List[BuildingResult]  # Highest ROI
    worst_10_buildings: List[BuildingResult]  # Highest current consumption

    ecm_frequency: Dict[str, int]  # Most recommended ECMs
    archetype_distribution: Dict[str, int]
```

### 🔜 REMAINING TASKS (Future Sessions)

| Priority | Task | Description |
|----------|------|-------------|
| P0 | Pre-train surrogate library | Train GP for all 40 archetypes (requires E+ runs) |
| P1 | CLI portfolio command | `raiden portfolio --input buildings.csv` |
| P1 | PostgreSQL integration | Store portfolio results for BRF portfolios |
| P2 | Parallel E+ execution | ProcessPoolExecutor for deep analysis tier |
| P2 | Web dashboard | React/Vue frontend for portfolio visualization |
| P3 | Boverket API | Auto-fetch energy declarations for non-Stockholm |
| P3 | PDF extraction | Parse energy declaration PDFs automatically |

### 📋 Usage Example (After Implementation)

```python
from src.orchestrator import RaidenOrchestrator, TierConfig

# Configure for portfolio analysis
config = TierConfig(
    standard_workers=100,      # Parallel surrogate predictions
    deep_workers=20,           # Parallel E+ simulations
    skip_energy_classes=("A", "B"),  # Skip already-optimized
)

orchestrator = RaidenOrchestrator(config=config)

# Analyze portfolio
addresses = ["Bellmansgatan 16", "Aktergatan 11", ...]  # 1000+ addresses
results = await orchestrator.analyze_portfolio(addresses)

# Generate report
from src.orchestrator import generate_portfolio_report
report = generate_portfolio_report(results.analytics, format="html")

print(f"Analyzed: {results.analyzed} buildings")
print(f"Total savings: {results.analytics.total_savings_potential_kwh:,.0f} kWh")
print(f"Portfolio NPV: {results.analytics.portfolio_npv_sek:,.0f} SEK")
```

---

## KEY PRINCIPLE

**USE REAL DATA - DON'T INFER!**

We have actual data from:
- Energy declarations → real energy class, real heating system, real ventilation type
- Mapillary → real facade images for AI analysis
- OSM/Overture → real footprints and heights
- Google Solar API → real roof analysis

The archetype matcher scores based on this REAL DATA, not guesswork.

---

## 🚀 RAIDEN 2026 ROADMAP (Active Development)

> **Full Details:** See [`docs/RAIDEN_2026_ROADMAP.md`](docs/RAIDEN_2026_ROADMAP.md)

### Current Focus: Phase 1-3 (Jan 2026)

#### Phase 1: Realistic HVAC Systems (P0 - CRITICAL)

**Problem:** All buildings use `IdealLoadsAirSystem` - no actual equipment modeling.

**Solution:** Swedish HVAC templates with real equipment:
- District heating (fjärrvärme) - 70% of MFH
- Heat pumps (GSHP, ASHP, FTX-VP) with real COPs
- Auto-select from Gripen/GeoJSON heating data

**Files:**
- `src/hvac/__init__.py` - Module
- `src/hvac/swedish_systems.py` - HVAC templates
- `src/hvac/hvac_selector.py` - Auto-selection from data

**Status:** 🔄 IN PROGRESS

#### Phase 2: Learned Occupancy Patterns (P1 - HIGH IMPACT)

**Problem:** Fixed Sveby schedules for all buildings (same pattern regardless of type).

**Solution:** Building-specific schedules:
- Residential variants (families, elderly, students)
- Commercial patterns (retail, restaurant, office)
- Swedish seasonal variation (summer holidays!)

**Files:**
- `src/schedules/__init__.py` - Module
- `src/schedules/swedish_patterns.py` - Pattern library
- `src/schedules/schedule_generator.py` - IDF generation

**Status:** 🔄 IN PROGRESS

#### Phase 3: Calibration Enhancement (P1 - QUICK WINS)

**Problem:** Suboptimal calibration (80 samples, no validation, no ASHRAE metrics).

**Solution:**
- Increase samples: 80 → 200
- Add train/test split (detect overfitting)
- ASHRAE Guideline 14 metrics (NMBE, CVRMSE)
- Context-aware priors (FTX detected → tight HR prior)

**Files:**
- `src/calibration/ashrae_metrics.py` - ASHRAE compliance
- Updated `src/calibration/surrogate.py` - Train/test split

**Status:** 🔄 IN PROGRESS

### Module Status (Updated 2026-01-03)

| Module | Status | Notes |
|--------|--------|-------|
| `src/hvac/` | 🔄 NEW | Swedish HVAC system templates |
| `src/schedules/` | 🔄 NEW | Occupancy pattern library |
| `src/calibration/` | 🔄 ENHANCING | ASHRAE metrics, better surrogates |
| `src/baseline/generator_v2.py` | 🔄 ENHANCING | Wiring realistic HVAC |

### Quick Commands

```bash
# Run tests for new modules
python -m pytest tests/test_hvac.py -v
python -m pytest tests/test_schedules.py -v
python -m pytest tests/test_calibration.py -v

# Check ASHRAE compliance
python -c "from src.calibration.ashrae_metrics import calculate_ashrae_metrics; print('OK')"
```

---

## SESSION CONTINUITY

When starting a new session, read these files first:
1. `CLAUDE.md` - This file (project overview)
2. `docs/RAIDEN_2026_ROADMAP.md` - Detailed implementation plan
3. `docs/BATTLE_PLAN.md` - Development history and decisions
4. Check TODO list with `/todos` command

### Current Implementation State (Last Updated: 2026-01-03)

```
PHASE 1 (HVAC):        [░░░░░░░░░░] 0% - Starting
PHASE 2 (SCHEDULES):   [░░░░░░░░░░] 0% - Starting
PHASE 3 (CALIBRATION): [░░░░░░░░░░] 0% - Starting
```
