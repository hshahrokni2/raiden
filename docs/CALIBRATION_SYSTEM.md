# Raiden Calibration System Architecture

This document maps the complete Bayesian calibration system implemented in Raiden,
including all components, data flows, and integration points.

## System Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    RAIDEN CALIBRATION SYSTEM                                 │
│                                                                              │
│  ┌─────────────┐    ┌──────────────┐    ┌─────────────┐    ┌─────────────┐  │
│  │   Building  │───▶│   Surrogate  │───▶│  ABC-SMC    │───▶│ Calibrated  │  │
│  │   Context   │    │   Training   │    │ Calibration │    │   Result    │  │
│  └─────────────┘    └──────────────┘    └─────────────┘    └─────────────┘  │
│        │                   │                   │                   │        │
│        ▼                   ▼                   ▼                   ▼        │
│  ┌─────────────┐    ┌──────────────┐    ┌─────────────┐    ┌─────────────┐  │
│  │  Context-   │    │    Morris    │    │   ASHRAE    │    │    ECM      │  │
│  │   Aware     │    │  Screening   │    │   Metrics   │    │  Savings    │  │
│  │   Priors    │    │              │    │             │    │ Uncertainty │  │
│  └─────────────┘    └──────────────┘    └─────────────┘    └─────────────┘  │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Module Map

| Module | File | Purpose |
|--------|------|---------|
| **Pipeline** | `src/calibration/pipeline.py` | Main orchestration |
| **Surrogate** | `src/calibration/surrogate.py` | GP model training |
| **Bayesian** | `src/calibration/bayesian.py` | ABC-SMC + priors |
| **Metrics** | `src/calibration/metrics.py` | ASHRAE compliance |
| **Sensitivity** | `src/calibration/sensitivity.py` | Morris screening |
| **Full Pipeline** | `src/analysis/full_pipeline.py` | End-to-end analysis |

---

## 1. Entry Points

### Primary Entry: `BayesianCalibrationPipeline.calibrate()`

**Location:** `src/calibration/pipeline.py:172`

```python
def calibrate(
    self,
    baseline_idf: Path,
    archetype_id: str,
    measured_kwh_m2: float,
    atemp_m2: float,
    # Context-aware prior constraints
    existing_measures: set = None,
    ventilation_type: str = None,
    heating_system: str = None,
    energy_class: str = None,
) -> CalibrationResult:
```

**Called from:** `FullPipelineAnalyzer._run_baseline()` (line 876)

---

## 2. Data Flow Diagram

```
┌──────────────────────────────────────────────────────────────────────────┐
│                         CALIBRATION DATA FLOW                             │
└──────────────────────────────────────────────────────────────────────────┘

INPUTS                          PROCESSING                         OUTPUTS
───────                         ──────────                         ───────

baseline_idf ──────┐
                   │
archetype_id ──────┼──▶ _get_surrogate() ──▶ TrainedSurrogate
                   │         │                     │
measured_kwh_m2 ───┤         │                     │ ┌──────────────────┐
                   │         ▼                     │ │ training_r2      │
atemp_m2 ──────────┤    LHS Samples               │ │ test_r2          │
                   │    (150 points)              │ │ is_overfit       │
                   │         │                     │ │ param_bounds     │
                   │         ▼                     │ └──────────────────┘
                   │    E+ Simulations                    │
                   │    (parallel)                        │
                   │         │                            ▼
                   │         ▼                    ┌───────────────────┐
                   │    GP Training ◀─────────────│ Morris Screening  │
                   │    (Matern 5/2)              │ (SALib)           │
                   │    80/20 split               │                   │
                   │         │                    │ ┌───────────────┐ │
                   │         │                    │ │ mu_star       │ │
                   │         ▼                    │ │ sigma         │ │
                   │                              │ │ ranking       │ │
existing_measures ─┼──▶ CalibrationPriors ◀──────│ └───────────────┘ │
ventilation_type ──┤    .from_building_context() └───────────────────┘
heating_system ────┤         │
energy_class ──────┘         │
                             ▼
                    ┌─────────────────┐
                    │  ABCSMCCalibrator│
                    │                  │
                    │  n_particles=500 │
                    │  n_generations=8 │
                    │  tolerance=20%   │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ CalibrationPosterior │
                    │                      │
                    │  samples: np.ndarray │
                    │  weights: np.ndarray │
                    │  means: Dict         │
                    │  stds: Dict          │
                    │  ci_90: Dict         │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ UncertaintyPropagator │
                    │                       │
                    │ predict_with_uncertainty()
                    │   → (mean, std, ci_90)│
                    └───────────┬───────────┘
                                │
                                ▼
                    ┌───────────────────────┐
                    │ ASHRAE Metrics        │
                    │                       │
                    │ NMBE: ±X.X%           │
                    │ CVRMSE: X.X%          │
                    │ passes_ashrae: bool   │
                    │ pass_probability: X.X │
                    └───────────┬───────────┘
                                │
                                ▼
                    ┌───────────────────────┐
                    │   CalibrationResult   │
                    │                       │
                    │ calibrated_kwh_m2     │
                    │ kwh_m2_std            │
                    │ calibrated_params     │
                    │ param_stds            │
                    │ ashrae_*              │
                    │ morris_results        │
                    │ surrogate_test_r2     │
                    └───────────────────────┘
```

---

## 3. Component Details

### 3.1 Surrogate Training (`surrogate.py`)

**Class:** `SurrogateTrainer`

**Purpose:** Train Gaussian Process models on E+ simulation results for fast prediction.

**Key Parameters:**
```python
SurrogateConfig(
    n_samples=150,           # Latin Hypercube samples (literature: 10-20 per dim)
    random_state=42,
    param_bounds={           # Swedish building defaults
        'infiltration_ach': (0.02, 0.20),
        'wall_u_value': (0.15, 1.50),
        'roof_u_value': (0.10, 0.60),
        'floor_u_value': (0.15, 0.80),
        'window_u_value': (0.70, 2.50),
        'heat_recovery_eff': (0.0, 0.90),
        'heating_setpoint': (18.0, 23.0),
    }
)
```

**GP Kernel:**
```python
kernel = (
    ConstantKernel(1.0, (1e-3, 1e3)) *
    Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e2), nu=2.5) +
    WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-5, 1e1))
)
```

**Cross-Validation:**
- 80% training / 20% test split
- Overfitting detection: `is_overfit = (train_r2 - test_r2) > 0.10`

**Output:** `TrainedSurrogate` dataclass with:
- `gp_model`: Fitted GP regressor
- `scaler_X`, `scaler_y`: StandardScalers
- `training_r2`, `test_r2`, `is_overfit`

---

### 3.2 Context-Aware Priors (`bayesian.py`)

**Class:** `CalibrationPriors`

**Purpose:** Constrain calibration parameters based on known building characteristics.

**Method:** `from_building_context()`

```python
@classmethod
def from_building_context(
    cls,
    archetype_id: str,
    existing_measures: Optional[set] = None,
    ventilation_type: Optional[str] = None,  # F, FT, FTX, S
    heating_system: Optional[str] = None,
    energy_class: Optional[str] = None,      # A-G
) -> "CalibrationPriors":
```

**Example Constraints:**

| Condition | Parameter | Prior Distribution |
|-----------|-----------|-------------------|
| FTX detected | heat_recovery_eff | Beta(8, 2.5) on [0.65, 0.90] |
| Energy class A-B | infiltration_ach | Truncated to [0.02, 0.08] |
| Pre-1960 building | wall_u_value | Higher mean (0.8-1.5) |

---

### 3.3 ABC-SMC Calibration (`bayesian.py`)

**Class:** `ABCSMCCalibrator`

**Algorithm:** Approximate Bayesian Computation with Sequential Monte Carlo

**Parameters:**
```python
n_particles = 500    # Number of parameter samples
n_generations = 8    # SMC generations (adaptive tolerance)
tolerance_percent = 20.0  # Initial acceptance: |pred - measured| < 20%
```

**Output:** `CalibrationPosterior` with:
- `samples`: (n_accepted, n_params) array
- `weights`: Normalized importance weights
- `means`, `stds`, `ci_90`: Per-parameter statistics

---

### 3.4 Morris Sensitivity Screening (`sensitivity.py`)

**Class:** `MorrisScreening`

**Purpose:** Identify which parameters have largest influence on heating energy.

**Method:**
```python
def analyze(self) -> MorrisResults:
    """
    Returns:
        mu: Mean elementary effects (can cancel out)
        mu_star: Mean |EE| (importance magnitude)
        sigma: Std of EE (nonlinearity/interactions)
        ranking: Parameters ranked by mu_star
    """
```

**Integration Point:** `pipeline.py:221-244`
```python
if self.use_adaptive_calibration:
    morris_results = self._get_morris_results(surrogate, archetype_id)
    important = morris_results.get_important_parameters(
        mu_star_threshold=0.1,
        top_n=self.max_calibration_params,
    )
```

**Parameter Classification:**
- `mu_star_normalized >= 0.10`: Important (calibrate)
- `mu_star_normalized < 0.05`: Negligible (fix at default)
- `sigma/mu_star > 0.5`: Nonlinear effects

---

### 3.5 ASHRAE Guideline 14 Metrics (`metrics.py`)

**Class:** `CalibrationMetrics`

**Metrics:**
```python
NMBE = Σ(measured - simulated) / (n × mean_measured) × 100%
CVRMSE = sqrt(Σ(measured - simulated)² / n) / mean_measured × 100%
```

**Compliance Thresholds:**

| Resolution | NMBE Limit | CVRMSE Limit |
|------------|------------|--------------|
| Monthly | ±10% | 30% |
| Hourly | ±5% | 15% |

**Uncertainty-Adjusted:**
```python
def compute_uncertainty_adjusted_metrics(
    measured_kwh_m2: float,
    simulated_kwh_m2: float,
    simulated_std: float,
) -> Tuple[CalibrationMetrics, float]:
    """Returns (metrics, probability_of_passing_ASHRAE)"""
```

---

### 3.6 ECM Uncertainty Propagation

**Location:** `full_pipeline.py:962-1037`

**Method:** Monte Carlo sampling from calibration posterior

```python
if calibration_result and calibration_result.kwh_m2_std:
    baseline_std = calibration_result.kwh_m2_std
    # Propagate to ECM savings
    savings_std = math.sqrt(2) * baseline_std  # Independence assumption
```

**Note:** Current implementation uses error propagation approximation.
Future improvement: Use `UncertaintyPropagator.compute_savings_distribution()`.

---

## 4. Package Interactions (`dependencies.py`)

**Class:** `ECMDependencyMatrix`

**Relationship Types:**
- `CONFLICT`: Cannot combine (factor = 0)
- `SYNERGY`: Combined > sum (factor > 1.0)
- `ANTI_SYNERGY`: Diminishing returns (factor < 1.0)
- `SUPERSEDES`: One makes other unnecessary

**Key Function:**
```python
def adjust_package_savings(
    baseline_kwh_m2: float,
    ecm_results: Dict[str, float],
) -> Dict:
    """
    Returns:
        naive_total_savings: Simple sum
        synergy_factor: Multiplicative adjustment
        total_savings_adjusted: Realistic combined savings
        warnings: Package compatibility issues
    """
```

**Example Synergies:**
```python
('air_sealing', 'ftx_installation'): 1.15,  # 15% bonus
('wall_external_insulation', 'ftx_installation'): 1.10,
('window_replacement', 'air_sealing'): 1.12,
```

---

## 5. Result Structure

### `CalibrationResult` (Full Schema)

```python
@dataclass
class CalibrationResult:
    # Point estimates
    calibrated_kwh_m2: float
    calibrated_params: Dict[str, float]

    # Uncertainty
    kwh_m2_std: float
    kwh_m2_ci_90: Tuple[float, float]
    param_stds: Dict[str, float]
    param_ci_90: Dict[str, Tuple[float, float]]

    # Posterior (for downstream)
    posterior: Optional[CalibrationPosterior] = None

    # Metadata
    archetype_id: str
    measured_kwh_m2: float
    calibration_error: float
    n_posterior_samples: int

    # Surrogate quality
    surrogate_r2: float           # Training R²
    surrogate_test_r2: float      # Test R² (generalization)
    surrogate_is_overfit: bool    # Warning flag

    # ASHRAE Guideline 14
    ashrae_nmbe: float
    ashrae_cvrmse: float
    ashrae_passes: bool
    ashrae_pass_probability: float

    # Morris sensitivity
    morris_results: Optional[MorrisResults] = None
    calibrated_param_list: Optional[List[str]] = None
    fixed_param_values: Optional[Dict[str, float]] = None
```

---

## 6. Configuration Points

### Pipeline Initialization

```python
BayesianCalibrationPipeline(
    runner=EnergyPlusRunner(),
    weather_path=Path('./weather.epw'),
    cache_dir=Path('./cache'),          # Surrogate caching
    n_surrogate_samples=100,             # LHS samples (default, config overrides to 150)
    n_abc_particles=500,                 # ABC-SMC particles
    n_abc_generations=8,                 # SMC generations
    parallel_sims=4,                     # E+ parallelism
    use_adaptive_calibration=True,       # Enable Morris screening
    min_calibration_params=3,            # Min params to calibrate
    max_calibration_params=5,            # Max params to calibrate
)
```

### Archetype-Specific Bounds

**Location:** `pipeline.py:396-407`

```python
if "post_2010" in archetype_id or "2011" in archetype_id:
    config.param_bounds['infiltration_ach'] = (0.02, 0.08)
    config.param_bounds['wall_u_value'] = (0.10, 0.30)
    config.param_bounds['window_u_value'] = (0.70, 1.30)
elif "1996" in archetype_id or "modern" in archetype_id:
    config.param_bounds['infiltration_ach'] = (0.03, 0.10)
    ...
```

---

## 7. Logging & Diagnostics

### Key Log Messages

```
INFO  Starting Bayesian calibration for {archetype_id}
INFO  Target: {measured_kwh_m2} kWh/m², Area: {atemp_m2} m²
INFO  Running Morris sensitivity screening...
INFO  Morris Sensitivity Ranking:
        1. infiltration_ach: μ*=1.000
        2. heat_recovery_eff: μ*=0.850
        ...
INFO  Calibrating 5 parameters: [infiltration_ach, ...]
INFO  Fixed 2 parameters: [floor_u_value, ...]
INFO  Using context-aware priors (building data constraints)
INFO  Training surrogate on 120/150 valid samples
INFO  Split: 96 train, 24 test samples
INFO  Train R²: 0.9850, RMSE: 1.23 kWh/m²
INFO  Test  R²: 0.9720, RMSE: 1.56 kWh/m²
INFO  ✓ No overfitting (gap=0.013 ≤ 0.10)
INFO  Running ABC-SMC calibration...
INFO  Calibration complete. 487 posterior samples
INFO  ✓ ASHRAE Guideline 14: PASSES (NMBE=+2.3%, CVRMSE=2.3%)
INFO    Probability of passing (with uncertainty): 94.2%
INFO  Calibrated: 51.8 ± 1.9 kWh/m²
INFO  90% CI: [48.7, 54.9] kWh/m²
```

### Warning Conditions

```
WARNING ⚠️ OVERFITTING DETECTED: Train R²=0.995 vs Test R²=0.870
WARNING ⚠️ ASHRAE Guideline 14: FAILS (NMBE=+12.5%, CVRMSE=15.2%)
WARNING ⚠️ Surrogate overfitting detected: Consider more samples.
```

---

## 8. File Dependencies

```
src/calibration/
├── __init__.py          # Exports all calibration classes
├── pipeline.py          # BayesianCalibrationPipeline, CalibrationResult
├── surrogate.py         # SurrogateTrainer, SurrogatePredictor, TrainedSurrogate
├── bayesian.py          # ABCSMCCalibrator, CalibrationPriors, UncertaintyPropagator
├── metrics.py           # CalibrationMetrics, ASHRAE compliance
├── sensitivity.py       # MorrisScreening, AdaptiveCalibration
└── calibrator_v2.py     # (Legacy) BayesianCalibrator, CalibrationResultV2

src/analysis/
├── full_pipeline.py     # FullPipelineAnalyzer (uses calibration)
├── package_generator.py # Uses ECM catalog costs + synergy factors
└── package_simulator.py # E+ simulation of packages

src/ecm/
├── catalog.py           # SWEDISH_ECM_CATALOG (costs, constraints)
├── dependencies.py      # ECMDependencyMatrix, adjust_package_savings()
└── idf_modifier.py      # Apply ECMs to IDF files
```

---

## 9. Testing

### Unit Tests

```bash
# Test calibration imports
python -c "from src.calibration import CalibrationMetrics, MorrisScreening, BayesianCalibrationPipeline"

# Test ASHRAE metrics
python -c "
from src.calibration.metrics import CalibrationMetrics
m = CalibrationMetrics.from_annual_data(53.0, 51.8)
print(m)
"

# Test package synergies
python -c "
from src.ecm import adjust_package_savings
result = adjust_package_savings(50.0, {'wall_external_insulation': 40.0, 'air_sealing': 45.0})
print(f'Synergy factor: {result[\"synergy_factor\"]:.2f}')
"
```

### Integration Test

```bash
python examples/sjostaden_2/run_full_analysis.py
```

---

## 10. Known Limitations

1. **Independence Assumption:** ECM uncertainty propagation assumes independence
   between baseline and ECM simulations. Should use posterior sampling.

2. **Morris Not Filtering:** Morris results identify important parameters but
   `ABCSMCCalibrator` still calibrates all parameters. Future: pass filtered
   param list to calibrator.

3. **Dual Result Classes:** `CalibrationResult` (pipeline.py) and
   `CalibrationResultV2` (calibrator_v2.py) coexist with different field names.
   Use `CalibrationResult` from pipeline.py.

4. **Annual-Only ASHRAE:** ASHRAE metrics computed on annual data only.
   With 1 data point, NMBE = annual error. Monthly data would be more accurate.

---

## References

- ASHRAE Guideline 14-2014: Measurement of Energy, Demand, and Water Savings
- Kennedy, M. C., & O'Hagan, A. (2001). Bayesian calibration of computer models
- Morris, M. D. (1991). Factorial Sampling Plans for Preliminary Computational Experiments
- Macdonald, I. A. (2002). Quantifying the Effects of Uncertainty in Building Simulation
