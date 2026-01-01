# Raiden Orchestrator - Portfolio-Scale Building Analysis

## Overview

The Raiden Orchestrator enables **portfolio-scale ECM analysis** for 1000+ buildings with:
- Tiered processing (Fast → Standard → Deep)
- Agentic QC when confidence is low
- Pre-trained surrogates for instant predictions
- Parallel execution for throughput

## Architecture

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
│  │ A. ImageQCAgent: Re-analyze facades if WWR conf < 60%    │   │
│  │ B. ECMRefinerAgent: Adjust packages per building context │   │
│  │ C. AnomalyAgent: LLM explains unusual patterns           │   │
│  └──────────────────────────────────────────────────────────┘   │
│         ▼                                                        │
│  OUTPUT: Portfolio report, priority rankings, budget optimizer  │
└─────────────────────────────────────────────────────────────────┘
```

## Module Files

| File | Purpose |
|------|---------|
| `src/orchestrator/__init__.py` | Module exports |
| `src/orchestrator/orchestrator.py` | `RaidenOrchestrator` class |
| `src/orchestrator/prioritizer.py` | `BuildingPrioritizer` with 8 strategies |
| `src/orchestrator/qc_agent.py` | `ImageQCAgent`, `ECMRefinerAgent`, `AnomalyAgent` |
| `src/orchestrator/surrogate_library.py` | `SurrogateLibrary` for pre-trained GPs |
| `src/orchestrator/portfolio_report.py` | `PortfolioAnalytics`, report generation |
| `tests/test_orchestrator.py` | 32 tests (all passing) |

## Usage

### Basic Portfolio Analysis

```python
import asyncio
from src.orchestrator import RaidenOrchestrator, TierConfig

async def analyze_my_portfolio():
    # Configure tiers
    config = TierConfig(
        skip_energy_classes=("A", "B"),  # Skip optimized buildings
        standard_workers=50,              # Parallel surrogate predictions
        deep_workers=10,                  # Parallel E+ simulations
    )

    orchestrator = RaidenOrchestrator(config=config)

    # Analyze portfolio
    addresses = [
        "Bellmansgatan 16, Stockholm",
        "Aktergatan 11, Stockholm",
        # ... 1000+ more addresses
    ]

    result = await orchestrator.analyze_portfolio(addresses)

    # Results
    print(f"Total: {result.total_buildings}")
    print(f"Analyzed: {result.analyzed}")
    print(f"Skipped (A/B): {result.skipped}")
    print(f"Failed: {result.failed}")

    return result

# Run
result = asyncio.run(analyze_my_portfolio())
```

### Building Prioritization

```python
from src.orchestrator import BuildingPrioritizer, PrioritizationStrategy

# Prioritize by ROI potential
prioritizer = BuildingPrioritizer(
    strategy=PrioritizationStrategy.HIGHEST_ROI_POTENTIAL
)

buildings = [
    ("Addr 1", {"energy_class": "G", "atemp_m2": 3000}),
    ("Addr 2", {"energy_class": "E", "atemp_m2": 1500}),
    ("Addr 3", {"energy_class": "A", "atemp_m2": 2000}),  # Will be skipped
]

results = prioritizer.prioritize_portfolio(buildings)
# Returns sorted by priority, skipped last
```

### Available Strategies

| Strategy | Description |
|----------|-------------|
| `HIGHEST_CONSUMPTION_FIRST` | Buildings using most energy |
| `LOWEST_ENERGY_CLASS_FIRST` | G → F → E → ... |
| `HIGHEST_ROI_POTENTIAL` | Best return on investment (default) |
| `QUICKEST_PAYBACK` | Shortest payback period |
| `OLDEST_FIRST` | Oldest buildings first |
| `LARGEST_FIRST` | Largest Atemp first |
| `HIGHEST_CONFIDENCE_FIRST` | Best data quality first |
| `LOWEST_CONFIDENCE_FIRST` | QC focus - worst data first |

### Portfolio Report Generation

```python
from src.orchestrator import PortfolioAnalytics, generate_portfolio_report

# Create analytics from results
analytics = result.analytics

# Generate report
report = generate_portfolio_report(
    analytics,
    output_path="./reports/portfolio.html",
    format="html",  # or "markdown", "json"
)

# Access metrics
print(f"Total savings: {analytics.total_savings_potential_kwh:,.0f} kWh")
print(f"Investment: {analytics.total_investment_sek:,.0f} SEK")
print(f"Portfolio NPV: {analytics.portfolio_npv_sek:,.0f} SEK")
print(f"Payback: {analytics.portfolio_payback_years:.1f} years")

# Top buildings
for b in analytics.top_10_roi:
    print(f"  {b['address']}: {b['payback_years']:.1f} years")
```

## QC Agents

### ImageQCAgent

Triggered when WWR or material confidence < 60%:
1. Retry with different backend (opencv → sam → lang_sam)
2. Fetch additional Mapillary images from different angles
3. Use LLM vision as fallback

### ECMRefinerAgent

Triggered when ECM savings are negative or anomalous:
1. Analyze why savings are negative (LED + heating interaction)
2. Apply ECM interaction matrix
3. Recalculate with building-specific factors

### AnomalyAgent

Triggered for unusual patterns:
- Old building with good energy class → renovation detection
- New building with poor energy class → construction issues
- Uses LLM reasoning for explanation

## QC Triggers

| Trigger | Threshold | Agent |
|---------|-----------|-------|
| Low WWR confidence | < 60% | ImageQCAgent |
| Low material confidence | < 70% | ImageQCAgent |
| Low archetype confidence | < 50 pts | ArchetypeQCAgent |
| Negative ECM savings | < 0 | ECMRefinerAgent |
| Energy class mismatch | > 2 classes | AnomalyAgent |

## Surrogate Library

Pre-trained Gaussian Process surrogates for each archetype:

```python
from src.orchestrator import SurrogateLibrary, get_or_train_surrogate

# Get library
library = SurrogateLibrary(surrogate_dir="./surrogates")

# Check available
available = library.list_available()
print(f"Pre-trained: {len(available)} archetypes")

# Get or train surrogate
surrogate = get_or_train_surrogate("mfh_1961_1975")

# Predict heating demand
params = {
    "infiltration_ach": 0.06,
    "wall_u_value": 0.5,
    "heat_recovery_eff": 0.80,
}
heating_kwh_m2 = surrogate.predict(params)
```

## Configuration

```python
@dataclass
class TierConfig:
    # Skip buildings
    skip_energy_classes: Tuple[str, ...] = ("A", "B")

    # Tier 2 (Standard)
    standard_workers: int = 50
    surrogate_confidence_threshold: float = 0.70

    # Tier 3 (Deep)
    deep_workers: int = 10
    enable_energyplus: bool = True

    # QC thresholds
    wwr_confidence_threshold: float = 0.60
    material_confidence_threshold: float = 0.70
    archetype_score_threshold: float = 50.0

    # Limits
    max_buildings_per_batch: int = 100
    timeout_per_building_sec: float = 300.0
```

## Test Coverage

32 tests in `tests/test_orchestrator.py`:
- Tier configuration and routing
- Prioritization strategies
- QC agent triggers and execution
- Surrogate library management
- Portfolio analytics calculation
- Report generation (HTML, Markdown, JSON)
- Integration tests

Run tests:
```bash
python -m pytest tests/test_orchestrator.py -v
```
