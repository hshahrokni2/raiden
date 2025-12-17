# Calibration Methodology - Sjostaden 2 EnergyPlus Model

## Document Version
- **Created**: 2025-12-16
- **Target Data**: Measured energy consumption 33 kWh/m²/year (space heating)
- **Achieved**: 42 kWh/m²/year (requires further calibration)

---

## 1. Calibration Overview

### 1.1 Calibration Objective

The goal is to align the EnergyPlus model predictions with measured energy consumption data for BRF Sjostaden 2. The primary calibration metric is **annual space heating energy intensity** (kWh/m²/year).

### 1.2 Calibration Hierarchy

The model uses a hierarchical calibration approach:

```
Level 1: Building Geometry & Envelope
    └── Areas, volumes, U-values, thermal mass

Level 2: Internal Loads
    └── Occupancy, lighting, equipment schedules

Level 3: HVAC System
    └── Heat recovery, fan power, setpoints

Level 4: Fine-tuning
    └── Infiltration, thermal bridges, schedules
```

### 1.3 Data Sources

| Data Type | Source | Confidence |
|-----------|--------|------------|
| Annual heating consumption | Utility bills | High |
| Floor areas | Building drawings | High |
| Envelope construction | Building age/typology | Medium |
| Occupancy patterns | Sveby defaults | Medium |
| Internal loads | Sveby defaults | Medium |
| Infiltration rate | Assumption | Low |

---

## 2. Measured Data

### 2.1 Annual Energy Consumption

| End Use | Measured (kWh/year) | Intensity (kWh/m²) | Source |
|---------|---------------------|-------------------|--------|
| Space Heating | ~74,000 | 33 | District heating bills |
| DHW | ~56,000 | 25 | Sveby default (not metered) |
| Electricity | Unknown | - | Not available |

**Note**: The 33 kWh/m² target is the space heating component only.

### 2.2 Data Quality Assessment

| Factor | Assessment | Impact on Calibration |
|--------|------------|----------------------|
| Weather normalization | Not applied | Could explain ±10-15% |
| Meter accuracy | Unknown | Typically ±2-3% |
| DHW separation | Assumed split | Could be ±5 kWh/m² |
| Common area heating | Included | Adds uncertainty |

---

## 3. Model vs. Measurement Comparison

### 3.1 Current Model Results

| Metric | Model | Measured | Difference |
|--------|-------|----------|------------|
| Space Heating (kWh/m²) | 42 | 33 | +27% |
| Total Heating (MWh/year) | 94 | 74 | +27% |

### 3.2 Gap Analysis

The model predicts **27% higher** heating consumption than measured. Possible explanations:

1. **Weather year mismatch**: IWEC is a typical year; actual billing year may be milder
2. **Actual building is better insulated**: U-values may be lower than assumed
3. **Higher actual heat recovery**: FTX may achieve >75% effectiveness
4. **Lower actual infiltration**: Building may be tighter than 0.06 ACH
5. **Internal gains underestimated**: More appliances/occupancy than modeled
6. **Setpoint lower in practice**: Residents may keep temperatures below 21°C

---

## 4. Calibration Parameters

### 4.1 High-Sensitivity Parameters

These parameters have the largest impact on heating energy:

| Parameter | Current Value | Range for Calibration | Sensitivity |
|-----------|---------------|----------------------|-------------|
| Infiltration (ACH) | 0.06 | 0.03 - 0.10 | High |
| Heat Recovery Effectiveness | 0.75 | 0.70 - 0.85 | High |
| Window U-value (W/m²K) | 1.0 | 0.8 - 1.2 | High |
| Wall Insulation (mm) | 250 | 200 - 300 | Medium |
| Heating Setpoint (°C) | 21 | 20 - 22 | Medium |

### 4.2 Medium-Sensitivity Parameters

| Parameter | Current Value | Range | Sensitivity |
|-----------|---------------|-------|-------------|
| Equipment Load (W/m²) | 10 | 8 - 15 | Medium |
| Lighting Load (W/m²) | 8 | 6 - 10 | Medium |
| Occupancy Density (p/m²) | 0.04 | 0.03 - 0.05 | Low-Medium |
| Roof Insulation (mm) | 350 | 300 - 400 | Low |

### 4.3 Low-Sensitivity Parameters

| Parameter | Current Value | Notes |
|-----------|---------------|-------|
| Thermal mass | Concrete 200mm | Keep fixed |
| Ground temperature | Monthly values | Keep fixed |
| SHGC | 0.45 | Minor impact in Sweden |

---

## 5. Calibration Strategies

### 5.1 Strategy A: Improve Envelope

To reduce heating from 42 to 33 kWh/m² (−21%), adjust envelope:

```
Option A1: Increase wall insulation
    250mm → 350mm mineral wool
    U-value: 0.13 → 0.095 W/m²K
    Expected impact: −3 to −5 kWh/m²

Option A2: Improve windows
    U-value: 1.0 → 0.8 W/m²K
    Expected impact: −2 to −3 kWh/m²

Option A3: Reduce infiltration
    0.06 → 0.04 ACH
    Expected impact: −3 to −4 kWh/m²
```

### 5.2 Strategy B: Improve Heat Recovery

```
Option B1: Increase HR effectiveness
    75% → 82%
    Expected impact: −4 to −6 kWh/m²

Option B2: Add rotary wheel (sensible + latent)
    Add latent recovery 0.65
    Expected impact: −2 to −3 kWh/m²
```

### 5.3 Strategy C: Increase Internal Gains

```
Option C1: Increase equipment loads
    10 → 15 W/m² (more appliances)
    Expected impact: −3 to −4 kWh/m²

Option C2: Increase occupancy
    0.04 → 0.05 person/m²
    Expected impact: −1 to −2 kWh/m²
```

### 5.4 Recommended Calibration Path

Based on Swedish building practices, the most likely adjustments are:

1. **Reduce infiltration to 0.04 ACH** (Swedish buildings are very airtight)
2. **Increase heat recovery to 80%** (modern FTX units achieve this)
3. **Improve window U-value to 0.9** (common in 2000s construction)

Combined expected impact: −8 to −12 kWh/m², bringing model to 30-34 kWh/m²

---

## 6. Calibration Procedure

### 6.1 Step-by-Step Process

```
Step 1: Establish Baseline
    └── Run current model, document all results
    └── Calculate CV(RMSE) and NMBE metrics

Step 2: Sensitivity Analysis
    └── Vary each high-sensitivity parameter ±20%
    └── Identify parameters with largest impact

Step 3: Parameter Adjustment
    └── Adjust most sensitive parameters first
    └── Use engineering judgment for realistic ranges
    └── Document all changes

Step 4: Validation
    └── Run final model
    └── Compare to measured data
    └── Check physical reasonableness

Step 5: Documentation
    └── Document all calibration decisions
    └── Note remaining uncertainties
    └── Define acceptable use cases
```

### 6.2 Calibration Metrics

**ASHRAE Guideline 14 Criteria**:

| Metric | Formula | Threshold |
|--------|---------|-----------|
| NMBE | (Σ(M-S)/ΣM) × 100 | ±5% (monthly) |
| CV(RMSE) | (√(Σ(M-S)²/n) / M̄) × 100 | ≤15% (monthly) |

Where:
- M = Measured value
- S = Simulated value
- n = Number of data points
- M̄ = Mean of measured values

### 6.3 Current Status

| Metric | Current Value | Target |
|--------|---------------|--------|
| NMBE (annual) | +27% | ≤±5% |
| CV(RMSE) | N/A | ≤15% |
| Status | **Uncalibrated** | Calibrated |

---

## 7. IDF Modifications for Calibration

### 7.1 Adjusting Infiltration

```
! In ZoneInfiltration:DesignFlowRate objects:
! Change from:
    0.06,                        !- Air Changes per Hour
! To:
    0.04,                        !- Air Changes per Hour
```

### 7.2 Adjusting Heat Recovery

```
! In ZoneHVAC:IdealLoadsAirSystem objects:
! Change from:
    0.75,                        !- Sensible Heat Recovery Effectiveness
! To:
    0.80,                        !- Sensible Heat Recovery Effectiveness
```

### 7.3 Adjusting Window U-Value

```
! In WindowMaterial:SimpleGlazingSystem:
! Change from:
    1.0,                         !- U-Factor {W/m2-K}
! To:
    0.9,                         !- U-Factor {W/m2-K}
```

### 7.4 Adjusting Wall Insulation

```
! In Material object for mineral wool:
! Change from:
    0.250,                       !- Thickness {m}
! To:
    0.300,                       !- Thickness {m}
```

---

## 8. Uncertainty Analysis

### 8.1 Sources of Uncertainty

| Source | Type | Magnitude |
|--------|------|-----------|
| Weather data | Epistemic | ±10-15% annual |
| Construction details | Epistemic | ±15% U-values |
| Occupant behavior | Aleatory | ±20% internal gains |
| System performance | Epistemic | ±10% HVAC |
| Model simplifications | Epistemic | ±5-10% |

### 8.2 Confidence Bounds

After calibration, expected accuracy:

| Metric | Best Estimate | 90% Confidence Interval |
|--------|---------------|------------------------|
| Annual Heating | 33 kWh/m² | 28 - 38 kWh/m² |
| Peak Heating Load | 35 W/m² | 30 - 42 W/m² |
| ECM Savings | ΔkWh/m² | ±20% relative |

### 8.3 Model Applicability

The calibrated model is most reliable for:
- Relative comparison of ECM options
- Annual energy estimates (±15%)
- Monthly load profiles (±25%)

Less reliable for:
- Hourly predictions
- Peak load sizing (use safety factor)
- Absolute energy guarantees

---

## 9. Future Calibration Improvements

### 9.1 Additional Data Needed

To improve calibration, collect:

1. **Monthly heating data**: Enables CV(RMSE) calculation
2. **Electricity bills**: Validate internal loads model
3. **Blower door test**: Measured infiltration rate
4. **Thermography**: Identify thermal bridges
5. **Submetering**: Separate DHW from space heating

### 9.2 Model Enhancements

Potential model improvements:

1. **Add thermal bridges**: Use linear thermal transmittance (ψ-values)
2. **Multi-zone per floor**: Model individual apartments
3. **Detailed HVAC**: Model actual district heating substation
4. **Demand-controlled ventilation**: If building has CO2 sensors
5. **Solar shading**: Model neighboring buildings

### 9.3 Validation with Other Buildings

To increase confidence:
- Apply same methodology to similar buildings
- Compare model predictions to measured data
- Refine default assumptions based on multiple calibrations

---

## 10. Calibration Log

### Version History

| Date | Version | Changes | Heating (kWh/m²) |
|------|---------|---------|------------------|
| 2025-12-16 | 1.0 | Initial model with Sveby defaults | 42 |
| - | 1.1 | Pending: Reduce infiltration to 0.04 | Est. 38-40 |
| - | 1.2 | Pending: Increase HR to 80% | Est. 34-36 |
| - | 1.3 | Pending: Improve windows to U=0.9 | Est. 32-34 |

### Calibration Notes

**2025-12-16**:
- Initial model created with Swedish archetype defaults
- Model runs successfully in E+ 25.1.0
- Gap of +27% vs measured data identified
- Calibration strategy developed but not yet implemented
- Primary focus was resolving segfault issue (see README.md)

---

## 11. References

1. **ASHRAE Guideline 14-2014** - Measurement of Energy, Demand, and Water Savings
2. **IPMVP** - International Performance Measurement and Verification Protocol
3. **CIBSE TM54** - Evaluating operational energy performance
4. **Sveby** - Brukarindata bostäder (Occupant data for residential)
5. **Reddy, T.A.** - Applied Data Analysis and Modeling for Energy Engineers

---

*Document prepared for BRF Energy Toolkit Project*
*Calibration status: UNCALIBRATED (baseline model)*
