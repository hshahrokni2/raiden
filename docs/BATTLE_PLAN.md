# Raiden Battle Plan - Development Roadmap

## Vision

**Raiden**: The ultimate agentic en-masse Swedish building energy simulator.
Given a portfolio of addresses, automatically analyze 1000+ buildings with AI-powered QC.

## Completed Work

### Session 2025-12-22: Calibration + Orchestrator

#### Calibration Improvements (Phase 1-4) ✅

| Component | File | Status |
|-----------|------|--------|
| Context-aware priors | `src/calibration/bayesian.py:from_building_context()` | ✅ |
| LLM calibration hints | `src/calibration/bayesian.py`, `pipeline.py`, `building_context.py` | ✅ |
| Morris sensitivity | `src/calibration/sensitivity.py` | ✅ |
| FixedParamPredictor | `src/calibration/surrogate.py` | ✅ |
| MC uncertainty propagation | `src/calibration/bayesian.py:ECMUncertaintyPropagator` | ✅ |
| ECM parameter effects | `src/calibration/bayesian.py:ECM_PARAMETER_EFFECTS` | ✅ |
| Tests | `tests/test_calibration.py` (37 tests) | ✅ |

#### RaidenOrchestrator ✅

| Component | File | Status |
|-----------|------|--------|
| Core orchestrator | `src/orchestrator/orchestrator.py` | ✅ |
| Prioritizer (8 strategies) | `src/orchestrator/prioritizer.py` | ✅ |
| ImageQCAgent | `src/orchestrator/qc_agent.py` | ✅ |
| ECMRefinerAgent | `src/orchestrator/qc_agent.py` | ✅ |
| AnomalyAgent | `src/orchestrator/qc_agent.py` | ✅ |
| SurrogateLibrary | `src/orchestrator/surrogate_library.py` | ✅ |
| PortfolioAnalytics | `src/orchestrator/portfolio_report.py` | ✅ |
| Tests | `tests/test_orchestrator.py` (32 tests) | ✅ |

---

## Recent Progress (2025-12-22)

### P0: Production Surrogate Training ✅

**Completed**: Full EnergyPlus-based surrogate training for all 40 archetypes

| Metric | Value |
|--------|-------|
| Archetypes trained | 40/40 |
| E+ simulations run | 6,000 |
| Average train R² | 1.000 |
| Average test R² | 1.000 |
| Training time | ~97 minutes |
| Parameters | 6 (infiltration, wall_u, roof_u, window_u, heat_recovery, setpoint) |

**Energy ranges by archetype type**:
| Archetype Category | Mean Energy (kWh/m²) |
|-------------------|---------------------|
| Historical (pre-1930) | 158-168 |
| Miljonprogrammet (1961-1975) | 137-142 |
| Post-1985 | 69-100 |
| Modern low-energy (2011+) | 26-40 |

**Files created**:
- `surrogates_production/{archetype_id}_gp.pkl` - 40 trained GP models
- `surrogates_production/index.json` - Metadata with parameter bounds

**Bug fixed**: Added missing `set_wall_u_value()`, `set_roof_u_value()`, `set_heating_setpoint()` methods to `IDFParser`

**Usage**:
```bash
# Production training (with E+)
python scripts/train_all_surrogates.py --samples 150 --sim-workers 8 --output ./surrogates_production

# Mock training (no E+, for testing)
python scripts/train_all_surrogates.py --mock --samples 100 --workers 8
```

### P1: CLI Portfolio Command ✅

**Completed**: `raiden portfolio <csv_file> [options]`

```bash
raiden portfolio buildings.csv --output ./reports --format html --workers 50
```

**Features**:
- CSV input (supports address, Address, adress, gatuadress columns)
- Progress bar with Rich
- HTML/Markdown/JSON output
- Parallel processing (configurable workers)
- Skip energy class A/B buildings

### P1: Deep Tier → Full Pipeline ✅

**Completed**: Deep tier now calls `FullPipelineAnalyzer.analyze()`

**Files modified**:
- `src/orchestrator/orchestrator.py:_analyze_single_deep()`

**Features**:
- Async integration with FullPipelineAnalyzer
- Automatic fallback to standard tier on import errors
- ECM results extraction and formatting
- Calibration uncertainty propagation

### P2: PostgreSQL Integration ✅

**Completed**: Full database schema for portfolio management

**Files created**:
- `supabase/migrations/002_portfolio_tables.sql` - Full schema with triggers
- `src/db/models.py` - Pydantic models for all tables

**Tables**:
- `portfolios` - Portfolio metadata and aggregates
- `portfolio_buildings` - Junction table with analysis results
- `qc_logs` - QC intervention logs
- `surrogate_models` - Trained surrogate metadata

**Features**:
- Auto-aggregating triggers for portfolio totals
- JSONB for recommended_ecms and qc_result
- UUID primary keys
- Full indexing for queries

### P2: Parallel E+ Execution ✅

**Completed**: ProcessPoolExecutor for CPU-bound deep tier

**Files modified**:
- `src/orchestrator/orchestrator.py:_analyze_deep()`
- Added `_run_deep_analysis_in_process()` module-level function

**Architecture**:
```python
# Module-level function for pickling
def _run_deep_analysis_in_process(address, building_data, enable_energyplus):
    # Runs in separate process with own event loop
    loop = asyncio.new_event_loop()
    result = loop.run_until_complete(pipeline.analyze(...))
    return result  # Dict for pickling

# In RaidenOrchestrator._analyze_deep()
with ProcessPoolExecutor(max_workers=deep_workers) as executor:
    future = loop.run_in_executor(executor, _run_deep_analysis_in_process, ...)
```

**Features**:
- True CPU parallelism (bypasses GIL)
- Configurable workers (default: 10)
- Automatic fallback to sequential on process pool failure
- Proper event loop handling per process

### P2: Enhanced ImageQCAgent ✅

**Completed**: Multi-source image QC with LLM vision

**Files modified**:
- `src/orchestrator/qc_agent.py:ImageQCAgent`

**Image Sources**:
1. Google Street View (primary)
2. Mapillary (fallback)
3. Existing cached images

**Analysis Pipeline**:
1. Fetch images from multiple directions (N/S/E/W)
2. Try WWR backends: opencv → sam → lang_sam
3. Claude Vision API for facade analysis
4. Aggregate results and confidence

### P2: Hybrid Portfolio Analysis ✅

**Completed**: Surrogate screening + E+ validation for large portfolios

**Files modified**:
- `src/orchestrator/orchestrator.py:analyze_portfolio_hybrid()`
- `src/orchestrator/__init__.py` - Export new function
- `src/cli/main.py` - Added `portfolio-hybrid` command

**Architecture**:
```
1000 buildings (input)
        ↓
Phase 1: Surrogate Screening (~10 sec total)
        ↓
    Rank by savings potential
        ↓
    Select: top 10% OR >20 kWh/m² savings
        ↓
Phase 2: E+ Validation (~10 sec per building)
        ↓
    Merge validated results
        ↓
    Report with validation accuracy metrics
```

**CLI Usage**:
```bash
# Screen 1000 buildings, E+ validate top 100
raiden portfolio-hybrid buildings.csv --validate-top-percent 10

# Validate all with >25 kWh/m² potential
raiden portfolio-hybrid buildings.csv --validate-min-savings 25
```

**Features**:
- Fast surrogate screening for all buildings
- E+ validation only for high-value candidates
- Configurable validation threshold (% or absolute)
- Reports surrogate vs E+ error for validated buildings
- Speedup factor tracking

**Timing Example (1000 buildings)**:
| Phase | Time | Buildings |
|-------|------|-----------|
| Surrogate screening | ~10s | 1000 |
| E+ validation | ~15 min | 100 (top 10%) |
| **Total** | **~15 min** | 1000 screened, 100 validated |

vs Full E+ for all: ~3 hours (10s × 1000)

---

## Next Steps (Priority Order)

### P3: Web Dashboard (PENDING)

**Goal**: React/Vue frontend for portfolio visualization.

**Features**:
- Portfolio upload (CSV)
- Analysis progress tracking
- Interactive building map
- ECM recommendation charts
- Export to PDF

### P3: Boverket API Integration

**Goal**: Auto-fetch energy declarations for non-Stockholm buildings.

**Note**: Stockholm already has 37,489 buildings in GeoJSON with full energy data.

---

## Key Architecture Decisions

1. **Tiered Processing**: Fast (GeoJSON) → Standard (surrogate) → Deep (E+)
2. **Pre-trained Surrogates**: Train once per archetype, reuse for all buildings
3. **Agentic QC**: Only triggered when confidence < threshold
4. **Parallel Execution**: asyncio for I/O, ProcessPoolExecutor for CPU
5. **Uncertainty Propagation**: ECMUncertaintyPropagator for all recommendations

---

## Test Commands

```bash
# Run calibration tests (37 tests)
python -m pytest tests/test_calibration.py -v

# Run orchestrator tests (32 tests)
python -m pytest tests/test_orchestrator.py -v

# Run all tests
python -m pytest tests/ -v

# Run with coverage
python -m pytest tests/ --cov=src --cov-report=html
```

---

## Quick Reference

### Import Patterns

```python
# Orchestrator
from src.orchestrator import (
    RaidenOrchestrator,
    TierConfig,
    BuildingPrioritizer,
    PrioritizationStrategy,
    PortfolioAnalytics,
    generate_portfolio_report,
)

# Calibration
from src.calibration import (
    BayesianCalibrator,
    CalibrationPriors,
    ECMUncertaintyPropagator,
    ECM_PARAMETER_EFFECTS,
    MorrisScreening,
    run_morris_analysis,
)

# QC Agents
from src.orchestrator.qc_agent import (
    ImageQCAgent,
    ECMRefinerAgent,
    AnomalyAgent,
)
```

### Key Data Flows

```
Address → GeoJSON lookup → BuildingData → ArchetypeMatcherV2 → Archetype
                                                    ↓
                                            CalibrationPriors
                                                    ↓
                                            SurrogatePredictor → baseline_kwh_m2
                                                    ↓
                                            ECMUncertaintyPropagator → savings + CI
                                                    ↓
                                            BuildingResult → PortfolioAnalytics
```

---

## Contact

For questions about Raiden architecture, see:
- `CLAUDE.md` - Main project documentation
- `docs/CALIBRATION_SYSTEM.md` - Bayesian calibration details
- `docs/ORCHESTRATOR.md` - Portfolio orchestration details
