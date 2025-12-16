# Raiden - Swedish Building Energy Modeling Project

## Project Overview

This is a professional-grade EnergyPlus energy modeling project for Swedish multi-family residential buildings. The primary model is BRF Sjostaden 2, a 7-story building in Stockholm.

## Current State

### Working Model
- **File**: `sjostaden_7zone.idf`
- **Status**: Runs successfully in EnergyPlus 25.1.0
- **Results**: 42 kWh/m²/year space heating (target: 33 kWh/m²)

### Documentation (READ THESE FIRST)
- `README.md` - Quick start and overview
- `TECHNICAL_NOTES.md` - Detailed technical specifications
- `CALIBRATION.md` - Calibration methodology and strategies
- `DEVELOPMENT_LOG.md` - Development history including segfault debugging

## Key Technical Details

### EnergyPlus 25.1.0 Bug (IMPORTANT)
When using `ZoneHVAC:IdealLoadsAirSystem` with heat recovery, you MUST use:
```
ConstantSupplyHumidityRatio,  !- Dehumidification Control Type
,                              !- Cooling Sensible Heat Ratio (BLANK!)
ConstantSupplyHumidityRatio,  !- Humidification Control Type
```
Using `None` causes a segmentation fault (exit code 139).

### Swedish Standards Used
- **BBR 6:251**: Ventilation 0.35 l/s/m²
- **Sveby**: Internal loads (8 W/m² lighting, 10 W/m² equipment, 25 m²/person)
- **SFP 1.5**: Fan power kW/(m³/s)

### Building Specs
- 7 floors × 320 m² = 2,240 m² Atemp
- FTX ventilation with 75% heat recovery
- District heating (modeled as IdealLoadsAirSystem)
- Triple glazing U=1.0, walls U=0.13, roof U=0.10

## Pending Work

### 1. Calibration (Priority)
Model shows 42 kWh/m² vs measured 33 kWh/m² (+27% gap). See `CALIBRATION.md` for strategies:
- Reduce infiltration: 0.06 → 0.04 ACH
- Increase heat recovery: 75% → 80%
- Improve windows: U=1.0 → 0.9

### 2. ECM Analysis
After calibration, run energy conservation measures:
- Envelope: wall insulation, window upgrades, air sealing
- HVAC: heat recovery upgrade, demand-controlled ventilation
- Renewables: rooftop solar PV (~320 m² available)

### 3. Additional Buildings
Extend methodology to other Swedish building archetypes.

## Running the Model

```bash
# Download weather file from EnergyPlus website first
energyplus -w SWE_Stockholm.Arlanda.024600_IWEC.epw -d output sjostaden_7zone.idf

# Results in output/eplustbl.csv
```

## File Locations

- Model: `sjostaden_7zone.idf`
- Results: `output_final/`
- Weather: Download `SWE_Stockholm.Arlanda.024600_IWEC.epw` from https://energyplus.net/weather

## Notes for Claude

- Always read the IDF file before making changes
- Use `--convert-only` mode to validate syntax before full simulation
- Check `eplusout.err` for warnings after each run
- The model uses SI units (meters, Watts, Celsius)
