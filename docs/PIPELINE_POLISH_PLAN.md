# V2 PIPELINE POLISH PLAN

**Date:** 2025-12-23
**Status:** ULTRATHINK ANALYSIS COMPLETE

---

## CURRENT STATE

### V2 Pipeline Flow (What Works)
```
Address
    ↓
┌─────────────────────────────────────────────────────────┐
│ STEP 1: Data Fetch (COMPLETE)                           │
│ • Sweden GeoJSON (37,489 buildings) ✅                  │
│ • Google Street View (4 facades) ✅                     │
│ • AI Analysis (WWR, material) ✅                        │
│ • Google Solar API (roof, PV) ✅                        │
│ • Geometry Calculator (wall areas) ✅                   │
└─────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────┐
│ STEP 2: Archetype Match (COMPLETE)                      │
│ • ArchetypeMatcherV2 ✅                                 │
│ • Envelope properties ✅                                │
│ • Building form detection ✅                            │
└─────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────┐
│ STEP 3: IDF Generation (PARTIAL)                        │
│ • Copy template IDF ✅                                  │
│ • Apply envelope params ⚠️ (IDD warning)               │
│ • GeomEppy generation (available, not default)          │
└─────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────┐
│ STEP 4: Calibration (PARTIAL)                           │
│ • Simple iteration-based ✅                             │
│ • Bayesian pipeline (fixed, needs testing)              │
└─────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────┐
│ OUTPUT: results_v2.json ✅                              │
└─────────────────────────────────────────────────────────┘
```

### MISSING STEPS (From V1)
```
    ↓
┌─────────────────────────────────────────────────────────┐
│ STEP 5: Package Generation ❌ MISSING                   │
│ • Steg 0: Nollkostnad (setpoint, schedules)            │
│ • Effektvakt (peak shaving)                            │
│ • Steg 1: LED + thermostats                            │
│ • Steg 2: Windows                                       │
│ • Steg 3: Premium (FTX upgrade)                        │
│ • Steg 4: Deep renovation                              │
└─────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────┐
│ STEP 6: Package Simulation ❌ MISSING                   │
│ • Run EnergyPlus for each package                      │
│ • Parse heating kWh                                     │
│ • Calculate savings vs baseline                        │
└─────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────┐
│ STEP 7: ROI Calculation ❌ MISSING                      │
│ • Apply Swedish costs (costs_sweden_v2)                │
│ • Calculate payback, NPV                               │
│ • Rank packages                                         │
└─────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────┐
│ STEP 8: Effektvakt Analysis ❌ MISSING                  │
│ • Thermal inertia (τ, coast time)                      │
│ • Peak shaving potential                               │
│ • Ellevio tariff savings                               │
└─────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────┐
│ STEP 9: HTML Report ❌ MISSING                          │
│ • Swedish board-ready format                           │
│ • All packages with costs                              │
│ • Effektvakt section                                   │
│ • Thermal inertia metrics                              │
│ • Recommendation                                        │
└─────────────────────────────────────────────────────────┘
```

---

## V1 vs V2 COMPARISON

| Feature | V1 | V2 |
|---------|----|----|
| Sweden GeoJSON | ✅ | ✅ |
| Google Street View | ❌ | ✅ |
| Mapillary + AI | ❌ | ✅ |
| Google Solar API | ❌ | ✅ |
| Geometry Calculator | ❌ | ✅ |
| Package Generation | ✅ | ❌ |
| Package Simulation | ✅ | ❌ |
| Effektvakt | ✅ | ❌ |
| Thermal Inertia | ✅ | ❌ |
| HTML Report | ✅ | ❌ |
| Bayesian Calibration | ❌ | ✅ (fixed) |

**V2 has better DATA, V1 has better OUTPUT.**

---

## PRIORITY RANKING

### P0: CRITICAL (V2 produces incomplete output)

1. **Add Package Simulation Loop**
   - Generate 6 packages (baseline + 5 ECM levels)
   - Apply ECM params to IDF
   - Run E+ for each
   - Parse results
   - **Effort:** 2 hours
   - **Impact:** HIGH - this is where recommendations come from

2. **Add HTML Report Generation**
   - Call existing `generate_report()` or `HTMLReportGenerator`
   - Pass simulated packages
   - **Effort:** 30 minutes
   - **Impact:** HIGH - deliverable for board

### P1: HIGH VALUE (Unique V1 features)

3. **Add Effektvakt Analysis**
   - Calculate thermal inertia from building data
   - Calculate peak shaving potential
   - Include as Steg 0.5 package
   - **Effort:** 1 hour
   - **Impact:** HIGH - major selling point for BRFs
   - **ROI:** Often 0.7 year payback!

4. **Add Thermal Inertia Metrics**
   - Time constant (τ)
   - Coast duration at -5°C
   - Thermal capacitance
   - **Effort:** 30 minutes
   - **Impact:** MEDIUM - nice-to-have in report

### P2: QUALITY IMPROVEMENTS

5. **Fix IDD Path for Param Application**
   - Set `ENERGYPLUS_IDD_PATH` in .env
   - Update `apply_params_to_idf()` to use it
   - **Effort:** 15 minutes
   - **Impact:** Fixes calibration warnings

6. **Test Bayesian Calibration End-to-End**
   - Verify `BayesianCalibrationPipeline.calibrate()` works
   - Check ASHRAE metrics
   - **Effort:** 1 hour
   - **Impact:** Better calibration quality

### P3: PORTFOLIO SCALE (Future)

7. Connect Orchestrator to CLI
8. Connect Database for storage
9. Train surrogate library

---

## IMPLEMENTATION PLAN

### Phase 1: Complete V2 Output (This Session)

```python
# After calibration in V2 pipeline, add:

# STEP 5: Generate Packages
logger.info("STEP 5: Generating ECM packages")
packages = create_ecm_packages(
    building_data=building_data,
    envelope=envelope,
    calibrated_params=calibrated_params,
    calibrated_kwh_m2=calibrated_kwh_m2,
)

# STEP 6: Simulate Packages
logger.info("STEP 6: Simulating packages")
results = []
for pkg in packages:
    pkg_idf = output_dir / f"pkg_{pkg['id']}.idf"
    shutil.copy(baseline_idf, pkg_idf)
    apply_params_to_idf(pkg_idf, pkg["params"])

    pkg_dir = output_dir / f"sim_{pkg['id']}"
    heating_kwh = run_single_simulation(pkg_idf, weather_path, pkg_dir)
    heating_kwh_m2 = heating_kwh / building_data["atemp_m2"]

    results.append({
        **pkg,
        "heating_kwh": heating_kwh,
        "heating_kwh_m2": heating_kwh_m2,
        "savings_kwh_m2": calibrated_kwh_m2 - heating_kwh_m2,
        "savings_percent": (calibrated_kwh_m2 - heating_kwh_m2) / calibrated_kwh_m2 * 100,
    })

# STEP 7: Calculate ROI
logger.info("STEP 7: Calculating ROI")
for r in results:
    if r["cost_sek"] > 0 and r.get("annual_savings_sek", 0) > 0:
        r["payback_years"] = r["cost_sek"] / r["annual_savings_sek"]
    else:
        r["payback_years"] = 0

# STEP 8: Effektvakt
logger.info("STEP 8: Effektvakt analysis")
thermal_inertia = calculate_thermal_inertia(building_data)
effektvakt = calculate_effektvakt_savings(building_data, thermal_inertia)
# Add effektvakt package to results...

# STEP 9: HTML Report
logger.info("STEP 9: Generating HTML report")
report_path = generate_report(
    output_dir=output_dir,
    address=args.address,
    building_data=building_data,
    packages=results,
    ...
)
```

### What to Copy from V1

From `run_production_pipeline.py`:
- `create_ecm_packages()` function (lines 500-850)
- `calculate_thermal_inertia()` function (lines 350-370)
- `calculate_effektvakt_savings()` function (lines 374-440)
- `generate_report()` function (lines 1493-1820)

---

## EFFORT SUMMARY

| Task | Effort | Impact | Priority |
|------|--------|--------|----------|
| Package simulation loop | 2h | HIGH | P0 |
| HTML report | 30m | HIGH | P0 |
| Effektvakt | 1h | HIGH | P1 |
| Thermal inertia | 30m | MEDIUM | P1 |
| IDD path fix | 15m | MEDIUM | P2 |
| Test Bayesian | 1h | MEDIUM | P2 |

**Total P0+P1:** ~4 hours to complete V2 pipeline

---

## SUCCESS CRITERIA

V2 pipeline is complete when:

1. ✅ Fetches data from all sources (GeoJSON, Street View, Solar, AI)
2. ✅ Matches archetype with confidence score
3. ✅ Generates calibrated baseline
4. ⬜ Simulates 6 ECM packages
5. ⬜ Calculates ROI for each package
6. ⬜ Includes effektvakt analysis
7. ⬜ Generates Swedish HTML report
8. ⬜ Report includes thermal inertia metrics

**End result:** One command → Complete board-ready report
```bash
python scripts/run_production_pipeline_v2.py "Aktergatan 11, Stockholm"
# Outputs: rapport.html with all packages and recommendations
```

---

## DECISION POINT

**Option A:** Merge V1 package/report code into V2
- Pro: One pipeline with all features
- Con: Large code change, risk of breaking

**Option B:** Keep V1 for production, polish V2 incrementally
- Pro: V1 works now
- Con: Two pipelines to maintain

**Option C:** Create V2.1 that calls V1 functions
- Pro: Minimal new code
- Con: Messy dependencies

**Recommendation:** Option A - Merge V1 functionality into V2 since V2 has better data collection.

---

*This plan should be executed in order. Each step builds on the previous.*
