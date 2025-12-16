# Development Log - Sjostaden 2 EnergyPlus Model

## Overview

This document chronicles the development of the Sjostaden 2 EnergyPlus model, including the significant debugging effort required to resolve an EnergyPlus 25.1.0 segmentation fault issue.

---

## Timeline

### Phase 1: Initial Model Development

**Goal**: Create a professional-grade Swedish multi-family residential building model for ECM analysis.

**Approach**:
- Based on Swedish building archetypes (TABULA/EPISCOPE)
- 7-story building with 2,240 m² Atemp
- FTX ventilation with 75% heat recovery
- District heating (modeled as IdealLoadsAirSystem)

**Initial Features Attempted**:
1. Foundation:Kiva for ground coupling - **Removed** (caused errors)
2. Urban shading objects - **Removed** (stability issues)
3. DHW as OtherEquipment - **Removed** (moved to post-processing)
4. FTX fan electricity - **Kept** (working)

---

### Phase 2: The Segfault Problem

**Symptom**: Model crashed with exit code 139 (segmentation fault) during simulation initialization.

```
$ energyplus -w weather.epw -d output sjostaden.idf
EnergyPlus Starting
EnergyPlus, Version 25.1.0
...
Initializing Simulation
[1]    12345 segmentation fault  energyplus ...
```

**Key Observations**:
- No error messages before crash
- `--convert-only` mode passed (syntax was valid)
- Crash occurred during simulation initialization, not input processing
- Exit code 139 = 128 + 11 (SIGSEGV)

---

### Phase 3: Debugging Process

#### Step 1: Test EnergyPlus Installation

```bash
# Test with stock examples
cd /usr/local/EnergyPlus-25-1-0/ExampleFiles
energyplus -w weather.epw -d output 1ZoneUncontrolled.idf
# RESULT: Success - E+ installation is fine
```

#### Step 2: Create Minimal Test Model

Created a 2-zone minimal model with same HVAC configuration as the full model.

```bash
energyplus -w weather.epw -d output minimal_2zone.idf
# RESULT: Same segfault!
```

This proved the issue was in the model configuration, not complexity.

#### Step 3: Compare Working vs. Failing Models

Analyzed differences between working stock examples and failing model:

| Aspect | Stock Example | Our Model |
|--------|---------------|-----------|
| IdealLoadsAirSystem | Basic config | Heat recovery enabled |
| Heat Recovery Type | None | Sensible |
| Dehumidification | None | None |
| Outdoor Air | Simple | DesignSpecification |

#### Step 4: Isolate the Crash

Systematically removed features from minimal model:

| Test | Configuration | Result |
|------|---------------|--------|
| 1 | Full config with HR | CRASH |
| 2 | Removed heat recovery | SUCCESS |
| 3 | HR + different dehumid | SUCCESS |
| 4 | HR + None dehumid | CRASH |

**Root cause identified**: `None` for Dehumidification/Humidification Control Type combined with Heat Recovery causes segfault.

---

### Phase 4: The Solution

**Problem Code** (causes segfault in E+ 25.1.0):
```
ZoneHVAC:IdealLoadsAirSystem,
    Zone1_IdealLoads,
    ...
    None,                        !- Dehumidification Control Type
    0.7,                         !- Cooling Sensible Heat Ratio
    None,                        !- Humidification Control Type
    ...
    Sensible,                    !- Heat Recovery Type
    0.75,                        !- Sensible Heat Recovery Effectiveness
```

**Fixed Code** (works correctly):
```
ZoneHVAC:IdealLoadsAirSystem,
    Zone1_IdealLoads,
    ...
    ConstantSupplyHumidityRatio, !- Dehumidification Control Type
    ,                            !- Cooling Sensible Heat Ratio (BLANK!)
    ConstantSupplyHumidityRatio, !- Humidification Control Type
    ...
    Sensible,                    !- Heat Recovery Type
    0.75,                        !- Sensible Heat Recovery Effectiveness
```

**Key Changes**:
1. Change `None` to `ConstantSupplyHumidityRatio` for both humidity controls
2. Leave Cooling Sensible Heat Ratio **blank** (not a number)
3. Use `autosize` for Maximum Heating/Cooling Air Flow Rate

---

### Phase 5: Model Rebuild

With the root cause identified, rebuilt the complete 7-zone model from scratch using a Python script:

```python
# Model generation approach:
# 1. Define all geometry programmatically
# 2. Generate proper IDF syntax
# 3. Apply correct IdealLoadsAirSystem configuration
# 4. Include all internal loads and schedules
```

**Final Model Characteristics**:
- 7 thermal zones (one per floor)
- Correct vertex ordering for all surfaces
- Working IdealLoadsAirSystem with heat recovery
- FTX fan electricity as ElectricEquipment
- All Sveby-based internal loads

---

### Phase 6: Validation Run

```bash
energyplus -w SWE_Stockholm.Arlanda.024600_IWEC.epw \
           -d output_new \
           sjostaden_7zone.idf
```

**Result**: SUCCESS (23 warnings, 0 severe errors)

**Warnings** (non-critical):
- Ground temperatures outside 15-25°C range (expected for Sweden)
- Weather file location used instead of IDF location (normal)
- Floor/ceiling vertices auto-corrected (cosmetic)
- Return air heat gain applied to zone air (expected with IdealLoads)

---

## Lessons Learned

### 1. EnergyPlus 25.1.0 Has Undocumented Bugs

The `None` + Heat Recovery combination causing a segfault is not documented. The Input Output Reference states `None` is a valid option, but it crashes when combined with heat recovery.

**Recommendation**: When using IdealLoadsAirSystem with heat recovery, always use `ConstantSupplyHumidityRatio` for humidity controls.

### 2. Segfaults Hide Root Causes

Unlike input errors that produce helpful messages, segfaults give no indication of the problem. The only way to debug is systematic elimination.

**Debugging Approach**:
1. Verify E+ installation with stock examples
2. Create minimal model reproducing the crash
3. Binary search: remove half the features, test, repeat
4. Compare field-by-field with working examples

### 3. IDF Field Order Matters

The IdealLoadsAirSystem object has 30+ fields, and the order is critical. Misaligned fields cause cryptic errors or crashes.

**Best Practice**: Use a template from a working example and modify carefully.

### 4. Cooling Sensible Heat Ratio Must Be Blank

When using `ConstantSupplyHumidityRatio`, the Cooling Sensible Heat Ratio field must be blank (`,`), not a number. Putting a number causes schema validation errors.

### 5. autosize Is Your Friend

For Maximum Heating/Cooling Air Flow Rate, use `autosize` instead of calculated values. This avoids potential mismatches and lets E+ handle sizing.

---

## File Evolution

| Version | Filename | Status | Notes |
|---------|----------|--------|-------|
| v0.1 | sjostaden_initial.idf | Deleted | First attempt, many errors |
| v0.2 | sjostaden_ftx.idf | Deleted | Added FTX, still crashing |
| v0.3 | sjostaden_2zone.idf | Deleted | Minimal test model |
| v0.4 | minimal_test.idf | Deleted | Debugging model |
| v1.0 | sjostaden_7zone.idf | **Current** | Working production model |

---

## Model Files Inventory

### Production Files
```
energyplus/
├── sjostaden_7zone.idf      # Main model (use this)
├── README.md                 # User documentation
├── TECHNICAL_NOTES.md        # Detailed technical specs
├── CALIBRATION.md            # Calibration methodology
├── DEVELOPMENT_LOG.md        # This file
└── output_new/               # Latest simulation results
    ├── eplustbl.csv
    ├── eplustbl.htm
    ├── eplusout.err
    ├── eplusout.eso
    └── eplusout.eio
```

### Weather File (Required)
```
# Download from EnergyPlus weather database:
# https://energyplus.net/weather
SWE_Stockholm.Arlanda.024600_IWEC.epw
```

---

## Running the Model

### Basic Run
```bash
energyplus -w /path/to/SWE_Stockholm.Arlanda.024600_IWEC.epw \
           -d output \
           sjostaden_7zone.idf
```

### Convert-Only (Syntax Check)
```bash
energyplus --convert-only sjostaden_7zone.idf
```

### Annual Run with Custom Output
```bash
energyplus -w weather.epw \
           -d output \
           -r \                    # Generate .rdd file
           -x \                    # Generate .expidf file
           sjostaden_7zone.idf
```

---

## Results Summary (Current Model)

| End Use | Annual (MWh) | Intensity (kWh/m²) |
|---------|--------------|-------------------|
| Space Heating (thermal) | 94 | 42 |
| FTX Fan Electricity | 10 | 4.6 |
| Interior Lighting | 41 | 18.4 |
| Interior Equipment | 49 | 21.9 |
| **Total Electricity** | 101 | 45 |

**Note**: Space heating is 27% higher than measured (33 kWh/m²). See CALIBRATION.md for adjustment strategies.

---

## Future Work

1. **Calibration**: Adjust parameters to match measured 33 kWh/m²
2. **ECM Analysis**: Run envelope and HVAC improvement scenarios
3. **Validation**: Compare with other Swedish buildings
4. **Enhancements**: Add thermal bridges, multi-zone per floor

---

## Contact

For questions about this model or the debugging process:
- BRF Energy Toolkit Project
- See README.md for project information

---

*Development log maintained for future reference and knowledge transfer.*
*Last updated: 2025-12-16*
