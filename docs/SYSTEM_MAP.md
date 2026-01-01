# RAIDEN SYSTEM MAP

**Last Updated: 2025-12-23**

This document is the authoritative map of all Raiden components, their connections, and integration status.

---

## EXECUTIVE SUMMARY

| Category | Total | Connected | Orphaned |
|----------|-------|-----------|----------|
| Source Modules | 20 | 16 | 4 |
| Scripts | 20 | 12 | 8 |
| Total LOC | 56,550 | ~45,000 | ~11,550 |

**Main Pipelines:**
1. `run_production_pipeline.py` - Original V1 (working, basic)
2. `run_production_pipeline_v2.py` - V2 with full infrastructure (working, recommended)
3. `src/analysis/full_pipeline.py` - Library version (partially working)
4. `src/cli/main.py` - CLI interface (working)

---

## MODULE DEPENDENCY GRAPH

```
                          ┌─────────────────────────────────────────────────┐
                          │              ENTRY POINTS                        │
                          │  CLI: src/cli/main.py                           │
                          │  Scripts: run_production_pipeline_v2.py         │
                          │  API: src/api/main.py                           │
                          └─────────────────┬───────────────────────────────┘
                                            │
        ┌───────────────────────────────────┼───────────────────────────────┐
        │                                   │                               │
        ▼                                   ▼                               ▼
┌───────────────┐               ┌───────────────────┐            ┌──────────────────┐
│   INGEST      │               │    BASELINE       │            │   ORCHESTRATOR   │
│ (Data Fetch)  │◄──────────────│  (Archetypes)     │◄───────────│ (Portfolio)      │
│               │               │                   │            │                  │
│ sweden_buildings ★            │ archetypes.py ★   │            │ orchestrator.py  │
│ streetview_fetcher ★          │ archetypes_detailed│           │ prioritizer.py   │
│ image_fetcher ★               │ archetype_matcher_v2★         │ qc_agent.py      │
│ overture_fetcher              │ generator_v2.py ★ │            │ surrogate_library│
│ osm_fetcher                   │ generator.py      │            │ portfolio_report │
│ microsoft_buildings           │ llm_archetype_reasoner★       └──────────────────┘
│ lantmateriet_fetcher          │ building_forms.py │
│ satellite_fetcher             │ calibrator.py     │
│ historical_streetview         └───────────────────┘
│ energidek_parser                        │
│ building_extractor                      │
│ brf_parser                              ▼
└───────────────┘               ┌───────────────────┐
        │                       │   CALIBRATION     │
        │                       │                   │
        ▼                       │ bayesian.py ★     │
┌───────────────┐               │ surrogate.py ★    │
│     AI        │               │ sensitivity.py ★  │
│  (Vision)     │               │ pipeline.py       │
│               │               │ calibrator_v2.py  │
│ wwr_detector ★│               │ metrics.py        │
│ material_classifier ★         └───────────────────┘
│ material_classifier_v2                  │
│ facade_analyzer                         │
│ image_quality                           ▼
└───────────────┘               ┌───────────────────┐
        │                       │   SIMULATION      │
        │                       │                   │
        ▼                       │ runner.py ★       │
┌───────────────┐               │ results.py ★      │
│   GEOMETRY    │               │ archetype_cache   │
│               │               │ distributed_worker│
│ building_geometry ★           └───────────────────┘
│ pv_potential ★                          │
│ thermal_mass                            │
└───────────────┘                         ▼
        │                       ┌───────────────────┐
        │                       │      ECM          │
        ▼                       │                   │
┌───────────────┐               │ catalog.py ★      │
│   ANALYSIS    │               │ constraints.py ★  │
│               │               │ idf_modifier.py ★ │
│ roof_analyzer ★               │ combinations.py   │
│ package_generator ★           │ dependencies.py   │
│ package_simulator ★           └───────────────────┘
│ building_analyzer                       │
│ full_pipeline                           │
│ integrated_analyzer                     ▼
│ u_value_calculator            ┌───────────────────┐
│ shading_solar                 │      ROI          │
└───────────────┘               │                   │
        │                       │ costs_sweden.py   │
        │                       │ costs_sweden_v2 ★ │
        ▼                       │ calculator.py     │
┌───────────────┐               └───────────────────┘
│   PLANNING    │                         │
│               │                         │
│ sequencer.py ★│                         ▼
│ cash_flow.py ★│               ┌───────────────────┐
│ effektvakt.py ★               │   REPORTING       │
│ models.py     │               │                   │
└───────────────┘               │ html_report.py ★  │
        │                       └───────────────────┘
        │                                 │
        ▼                                 ▼
┌───────────────┐               ┌───────────────────┐
│    CORE       │               │     EXPORT        │
│               │               │                   │
│ building_context ★            │ energyplus_idf    │
│ address_pipeline ★            │ enriched_json     │
│ idf_parser ★  │               └───────────────────┘
│ models.py     │
│ config.py     │               ┌───────────────────┐
│ coordinates   │               │       DB          │
└───────────────┘               │                   │
        │                       │ client.py         │
        ▼                       │ models.py         │
┌───────────────┐               │ repository.py     │
│    UTILS      │               └───────────────────┘
│               │
│ weather_downloader ★
│ logging_config
│ retry
│ validation
└───────────────┘

★ = Actively used in V2 pipeline
```

---

## DETAILED MODULE STATUS

### 1. INGEST (`src/ingest/`) - Data Fetching

| File | Lines | Status | Used In Pipeline | Notes |
|------|-------|--------|-----------------|-------|
| `sweden_buildings.py` | 474 | ★ ACTIVE | V2 | **PRIMARY DATA SOURCE** - 37,489 Stockholm buildings |
| `streetview_fetcher.py` | 497 | ★ ACTIVE | V2 | Google Street View API for facade images |
| `image_fetcher.py` | 811 | ★ ACTIVE | V1, V2 | Mapillary facade images |
| `overture_fetcher.py` | 281 | AVAILABLE | Fallback | OSM + Microsoft footprints |
| `osm_fetcher.py` | 259 | AVAILABLE | Fallback | Direct OSM queries |
| `microsoft_buildings.py` | 243 | AVAILABLE | Fallback | Microsoft Building Footprints |
| `lantmateriet_fetcher.py` | 237 | AVAILABLE | Not used | Official Swedish data (API incomplete) |
| `satellite_fetcher.py` | 189 | AVAILABLE | Not used | ESRI satellite imagery |
| `historical_streetview.py` | 276 | AVAILABLE | Not used | Historical Street View for renovation detection |
| `energidek_parser.py` | 223 | PARTIAL | Via GeoJSON | Boverket energy declarations |
| `building_extractor.py` | 911 | ORPHANED | Not used | Combined extractor (superseded by V2) |
| `brf_parser.py` | 186 | ORPHANED | Not used | BRF annual report parser (manual data) |

### 2. AI (`src/ai/`) - Computer Vision

| File | Lines | Status | Used In Pipeline | Notes |
|------|-------|--------|-----------------|-------|
| `wwr_detector.py` | 1331 | ★ ACTIVE | V2 | Window-to-wall ratio from images |
| `material_classifier.py` | 419 | ★ ACTIVE | V2 | Facade material classification |
| `material_classifier_v2.py` | 277 | AVAILABLE | Not used | DINOv2-based classifier (experimental) |
| `facade_analyzer.py` | 243 | ORPHANED | Not used | Combined facade analyzer (superseded) |
| `image_quality.py` | 112 | ORPHANED | Not used | Image quality scoring |

### 3. GEOMETRY (`src/geometry/`) - Building Geometry

| File | Lines | Status | Used In Pipeline | Notes |
|------|-------|--------|-----------------|-------|
| `building_geometry.py` | 497 | ★ ACTIVE | V2 | Wall areas per orientation |
| `pv_potential.py` | 257 | ★ ACTIVE | Via roof_analyzer | PV capacity estimation |
| `thermal_mass.py` | 183 | PARTIAL | Via effektvakt | Thermal time constant |

### 4. BASELINE (`src/baseline/`) - Archetype Matching

| File | Lines | Status | Used In Pipeline | Notes |
|------|-------|--------|-----------------|-------|
| `archetypes.py` | 497 | ★ ACTIVE | V1, V2 | 7 Swedish archetypes |
| `archetypes_detailed.py` | 7077 | AVAILABLE | ArchetypeMatcherV2 | 40 detailed archetypes |
| `archetype_matcher_v2.py` | 1540 | ★ ACTIVE | V2 | AI-enhanced matching |
| `generator_v2.py` | 1060 | ★ ACTIVE | V2 | GeomEppy IDF generation |
| `generator.py` | 1057 | LEGACY | V1 | Old IDF generator |
| `llm_archetype_reasoner.py` | 647 | AVAILABLE | Optional | Renovation detection via LLM |
| `building_forms.py` | 458 | ★ ACTIVE | V2 | Lamellhus, skivhus, etc. |
| `calibrator.py` | 237 | LEGACY | V1 | Simple calibration |

### 5. CALIBRATION (`src/calibration/`) - Bayesian Calibration

| File | Lines | Status | Used In Pipeline | Notes |
|------|-------|--------|-----------------|-------|
| `bayesian.py` | 1114 | ★ ACTIVE | V2 | ABC-SMC + priors + ECM effects |
| `surrogate.py` | 665 | ★ ACTIVE | V2 | GP surrogate models |
| `sensitivity.py` | 348 | ★ ACTIVE | Morris screening | Parameter sensitivity analysis |
| `pipeline.py` | 662 | AVAILABLE | V2 optional | Full calibration pipeline |
| `calibrator_v2.py` | 327 | AVAILABLE | Via pipeline | Unified calibrator interface |
| `metrics.py` | 178 | ★ ACTIVE | V2 | ASHRAE calibration metrics |

### 6. ECM (`src/ecm/`) - Energy Conservation Measures

| File | Lines | Status | Used In Pipeline | Notes |
|------|-------|--------|-----------------|-------|
| `catalog.py` | 1305 | ★ ACTIVE | V1, V2 | 27 Swedish ECMs defined |
| `constraints.py` | 367 | ★ ACTIVE | V1, V2 | Constraint engine |
| `idf_modifier.py` | 1291 | ★ ACTIVE | V1, V2 | Apply ECMs to IDF |
| `combinations.py` | 276 | AVAILABLE | Not used | Package combination generator |
| `dependencies.py` | 412 | ★ ACTIVE | V2 | ECM interaction matrix |

### 7. SIMULATION (`src/simulation/`) - EnergyPlus

| File | Lines | Status | Used In Pipeline | Notes |
|------|-------|--------|-----------------|-------|
| `runner.py` | 324 | ★ ACTIVE | V1, V2 | E+ simulation execution |
| `results.py` | 231 | ★ ACTIVE | V1, V2 | Parse E+ output |
| `archetype_cache.py` | 387 | AVAILABLE | Portfolio | Pre-computed results per archetype |
| `distributed_worker.py` | 289 | AVAILABLE | Portfolio | Distributed E+ execution |

### 8. ANALYSIS (`src/analysis/`) - Building Analysis

| File | Lines | Status | Used In Pipeline | Notes |
|------|-------|--------|-----------------|-------|
| `roof_analyzer.py` | 667 | ★ ACTIVE | V2 | Google Solar API integration |
| `package_generator.py` | 557 | ★ ACTIVE | V1, V2 | Generate ECM packages |
| `package_simulator.py` | 669 | ★ ACTIVE | V1, V2 | Simulate packages |
| `full_pipeline.py` | 1749 | PARTIAL | Not default | Library pipeline (needs work) |
| `integrated_analyzer.py` | 829 | ORPHANED | Not used | Old combined analyzer |
| `building_analyzer.py` | 664 | ORPHANED | Not used | Old analyzer |
| `u_value_calculator.py` | 457 | AVAILABLE | Not used | Back-calculate U-values |
| `shading_solar.py` | 377 | ORPHANED | Not used | Solar shading analysis |

### 9. ROI (`src/roi/`) - Cost Calculations

| File | Lines | Status | Used In Pipeline | Notes |
|------|-------|--------|-----------------|-------|
| `costs_sweden_v2.py` | 1541 | ★ ACTIVE | V2 | **COMPREHENSIVE** - BeBo costs, effektvakt |
| `costs_sweden.py` | 633 | LEGACY | V1 | Original cost database |
| `calculator.py` | 267 | ★ ACTIVE | V1, V2 | ROI calculations |

### 10. PLANNING (`src/planning/`) - BRF Planning

| File | Lines | Status | Used In Pipeline | Notes |
|------|-------|--------|-----------------|-------|
| `sequencer.py` | 337 | ★ ACTIVE | V1 | ECM sequencing |
| `cash_flow.py` | 298 | ★ ACTIVE | V1 | 30-year cash flow |
| `effektvakt.py` | 287 | ★ ACTIVE | V1, V2 | Peak shaving optimization |
| `models.py` | 267 | ★ ACTIVE | All | Data models |

### 11. REPORTING (`src/reporting/`)

| File | Lines | Status | Used In Pipeline | Notes |
|------|-------|--------|-----------------|-------|
| `html_report.py` | 1449 | ★ ACTIVE | V1 | Swedish board report generator |

### 12. CORE (`src/core/`) - Core Infrastructure

| File | Lines | Status | Used In Pipeline | Notes |
|------|-------|--------|-----------------|-------|
| `address_pipeline.py` | 1470 | ★ ACTIVE | V2 | Address → data fetcher |
| `building_context.py` | 720 | ★ ACTIVE | V1, V2 | Existing measures detection |
| `idf_parser.py` | 671 | ★ ACTIVE | V1, V2 | Parse IDF files |
| `models.py` | 267 | ★ ACTIVE | All | BRFBuilding, etc. |
| `config.py` | 89 | ★ ACTIVE | All | Settings |
| `coordinates.py` | 54 | ★ ACTIVE | V2 | Coordinate transforms |

### 13. ORCHESTRATOR (`src/orchestrator/`) - Portfolio Scale

| File | Lines | Status | Used In Pipeline | Notes |
|------|-------|--------|-----------------|-------|
| `orchestrator.py` | 1268 | AVAILABLE | Portfolio | Tiered analysis |
| `qc_agent.py` | 907 | AVAILABLE | Portfolio | Agentic QC |
| `prioritizer.py` | 487 | AVAILABLE | Portfolio | Building prioritization |
| `surrogate_library.py` | 378 | AVAILABLE | Portfolio | Pre-trained surrogates |
| `portfolio_report.py` | 289 | AVAILABLE | Portfolio | Aggregate analytics |

### 14. OTHER MODULES

| Module | Status | Notes |
|--------|--------|-------|
| `src/db/` | ORPHANED | Supabase integration (not used) |
| `src/export/` | PARTIAL | IDF export (used), JSON (not used) |
| `src/visualization/` | ORPHANED | 3D visualization (not integrated) |
| `src/api/` | AVAILABLE | FastAPI REST (working but not default) |
| `src/utils/` | ★ ACTIVE | Weather, logging, validation |
| `src/compat/` | ★ ACTIVE | Python 3.10+ compatibility patches |

---

## PIPELINE COMPARISON

### V1 Pipeline (`run_production_pipeline.py`)
```
Address → GeoJSON Lookup → Archetype Match → Generate IDF → Calibrate →
         Simulate Packages → Generate HTML Report
```
**Pros:** Simple, working
**Cons:** No AI analysis, no Google APIs

### V2 Pipeline (`run_production_pipeline_v2.py`)
```
Address → GeoJSON Lookup → Street View (fallback) → AI Analysis →
         Google Solar API → Geometry Calculator → Archetype Match →
         Generate IDF → Bayesian Calibration → Package Simulation →
         Results JSON
```
**Pros:** Full infrastructure, AI facade analysis, thermal mass
**Cons:** No HTML report generation yet

---

## ORPHANED CODE (Not Connected)

### High Value (Should Connect)
1. **`src/analysis/full_pipeline.py`** (1749 lines) - Library version of pipeline
2. **`src/db/`** - Supabase integration (useful for portfolios)
3. **`src/orchestrator/`** - Portfolio-scale analysis (tested, ready)

### Medium Value (Consider Connecting)
1. **`src/ingest/historical_streetview.py`** - Renovation detection via history
2. **`src/ai/material_classifier_v2.py`** - DINOv2 classifier
3. **`src/visualization/building_3d.py`** - 3D model viewer

### Low Value (Deprecated)
1. **`src/ingest/building_extractor.py`** - Superseded by V2 pipeline
2. **`src/analysis/integrated_analyzer.py`** - Superseded
3. **`src/analysis/building_analyzer.py`** - Superseded
4. **`src/ai/facade_analyzer.py`** - Superseded by wwr_detector + material_classifier
5. **`src/ingest/brf_parser.py`** - Manual data entry (not automated)

---

## SCRIPTS STATUS

### Active/Working
| Script | Purpose | Uses |
|--------|---------|------|
| `run_production_pipeline_v2.py` | **MAIN PIPELINE** | Full V2 |
| `run_production_pipeline.py` | V1 pipeline | GeoJSON, basic |
| `run_sjostaden_analysis.py` | Single building | Full E+ simulation |
| `train_all_surrogates.py` | Train GP models | Calibration |
| `build_archetype_cache.py` | Build cache | Simulation |

### Test/Demo
| Script | Purpose | Status |
|--------|---------|--------|
| `test_enhanced_streetview.py` | Test Street View | Working |
| `test_google_streetview.py` | Test Street View | Working |
| `test_mapillary_all.py` | Test Mapillary | Working |
| `test_geometry.py` | Test geometry | Working |
| `test_archetype_matching.py` | Test archetypes | Working |
| `test_material_v2.py` | Test DINOv2 | Working |

### Orphaned/Old
| Script | Status | Notes |
|--------|--------|-------|
| `demo_full_brf_analysis.py` | ORPHANED | Old demo |
| `run_full_pipeline_sjostaden.py` | ORPHANED | Uses old full_pipeline |
| `test_full_pipeline.py` | BROKEN | full_pipeline has issues |
| `process_sjostaden.py` | LEGACY | Old processing |

---

## INTEGRATION GAPS

### 1. V2 Pipeline Missing HTML Report
**Current:** V2 outputs `results_v2.json`
**Needed:** Generate Swedish board report like V1

### 2. Bayesian Calibration Import Error ✅ FIXED
**Was:** `GPSurrogateModel` import fails
**Fixed:** Now uses `BayesianCalibrationPipeline` from `src/calibration/pipeline.py`

### 3. IDF Parameter Application
**Current:** "IDD file needed" warning
**Fix:** Set `ENERGYPLUS_IDD_PATH` environment variable

### 4. Orchestrator Not Connected
**Current:** 32 tests pass but not in main pipeline
**Action:** Add `--portfolio` mode to CLI

### 5. Database Storage
**Current:** Results only saved to JSON files
**Action:** Connect `src/db/` for PostgreSQL storage

---

## RECOMMENDED NEXT STEPS

> **See [`docs/PIPELINE_POLISH_PLAN.md`](PIPELINE_POLISH_PLAN.md) for detailed implementation plan.**

### Immediate (This Session)
1. [x] Fix `GPSurrogateModel` import → Now uses `BayesianCalibrationPipeline` (2025-12-23)
2. [ ] Add package simulation loop to V2 (P0)
3. [ ] Add HTML report generation to V2 (P0)
4. [ ] Add effektvakt analysis to V2 (P1)
5. [ ] Set IDD path in .env (P2)

### Short Term
1. [ ] Connect orchestrator to CLI (`raiden portfolio`)
2. [ ] Connect database for portfolio storage
3. [ ] Clean up orphaned code

### Medium Term
1. [ ] Add Boverket API for non-Stockholm
2. [ ] Train surrogate library for all archetypes
3. [ ] Web dashboard

---

## ENVIRONMENT VARIABLES

```bash
# Required for V2 pipeline
MAPILLARY_TOKEN=MLY|xxx           # Facade images
BRF_GOOGLE_API_KEY=AIzaSy...      # Street View + Solar API

# Optional
KOMILION_API_KEY=xxx              # LLM reasoning
ENERGYPLUS_IDD_PATH=/path/to/Energy+.idd  # IDF parsing
SUPABASE_URL=xxx                  # Database
SUPABASE_KEY=xxx                  # Database
```

---

## FILE STATISTICS

```
Total Python files: 150+
Total lines of code: 56,550
Largest files:
  - archetypes_detailed.py: 7,077 lines (40 archetypes)
  - full_pipeline.py: 1,749 lines
  - costs_sweden_v2.py: 1,541 lines
  - archetype_matcher_v2.py: 1,540 lines
  - html_report.py: 1,449 lines
```

---

*This map should be kept up-to-date when making significant changes.*
