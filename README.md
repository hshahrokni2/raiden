# Sjostaden 2 - Swedish Multi-Family Building Energy Model

## Overview

Professional-grade EnergyPlus model for a 7-story Swedish multi-family residential building (BRF Sjostaden 2) in Stockholm. Developed for energy conservation measure (ECM) analysis using Swedish building archetypes and standards.

**Model Version**: EnergyPlus 25.1.0
**Location**: Stockholm, Sweden (Arlanda weather data)
**Building Type**: Multi-family residential (Flerbostadshus)
**Construction Era**: 1990s-2000s

## Quick Start

```bash
# Run annual simulation
energyplus -w /path/to/SWE_Stockholm.Arlanda.024600_IWEC.epw -d output sjostaden_7zone.idf

# Results in output/eplustbl.csv
```

## Building Specifications

| Parameter | Value | Source |
|-----------|-------|--------|
| Gross Floor Area (Atemp) | 2,240 m² | 7 floors × 320 m² |
| Building Footprint | 20m × 16m | Typical Swedish multi-family |
| Floor Height | 3.0 m | Standard residential |
| Number of Apartments | ~28 (estimated) | ~4 per floor |
| Occupants | ~70 people | Sveby 25 m²/person |

## Envelope Properties

| Component | U-value (W/m²K) | Construction |
|-----------|-----------------|--------------|
| External Walls | 0.13 | 200mm concrete + 250mm mineral wool |
| Roof | 0.10 | 200mm concrete + 350mm mineral wool |
| Ground Floor | 0.13 | 200mm concrete + 250mm mineral wool |
| Windows | 1.00 | Triple glazing, SHGC 0.45 |
| Infiltration | 0.06 ACH | Swedish airtight construction |

## HVAC System

- **Heating**: District heating (modeled as IdealLoadsAirSystem)
- **Ventilation**: FTX (balanced with heat recovery)
  - Airflow: 0.35 l/s/m² (BBR 6:251 requirement)
  - Heat Recovery: 75% sensible effectiveness
  - Fan Power: SFP 1.5 kW/(m³/s) → 1.18 kW total
- **Cooling**: None (Swedish residential standard)

## Simulation Results

| End Use | Annual (kWh) | Intensity (kWh/m²) |
|---------|--------------|-------------------|
| Space Heating (thermal) | 93,765 | 41.9 |
| FTX Fan Electricity | 10,337 | 4.6 |
| Interior Lighting | 41,207 | 18.4 |
| Interior Equipment | 49,056 | 21.9 |
| **Total Electricity** | 100,600 | 44.9 |

## File Structure

```
energyplus/
├── sjostaden_7zone.idf       # Main EnergyPlus model (USE THIS)
├── README.md                 # This file
├── TECHNICAL_NOTES.md        # Detailed technical documentation
├── CALIBRATION.md            # Calibration methodology
├── DEVELOPMENT_LOG.md        # Development history & debugging notes
├── output_final/             # Latest simulation results
│   ├── eplustbl.csv          # Tabular results (end-use breakdown)
│   ├── eplustbl.htm          # HTML formatted summary
│   ├── eplusout.eso          # Time series data (hourly)
│   ├── eplusout.eio          # Initialization outputs
│   └── eplusout.err          # Warnings and errors
└── archive/                  # Debug files (can be deleted)
    ├── debug/                # Test IDF files from troubleshooting
    └── outputs_debug/        # Output directories from test runs
```

## Data Sources

### Swedish Standards & Guidelines
- **BBR** (Boverkets Byggregler) - Building regulations
- **Sveby** - Standardize and verify energy performance
- **SS-EN ISO 14683** - Thermal bridges in building construction

### Building Archetypes
- **TABULA/EPISCOPE** - EU building typology project
- Swedish residential archetypes for 1990s-2000s construction

### Weather Data
- **IWEC** (International Weather for Energy Calculations)
- Station: Stockholm Arlanda (WMO 024600)
- Heating degree days: ~3,800 HDD (base 17°C)

## Known Issues & Solutions

### EnergyPlus 25.1.0 Segfault Issue

**Problem**: Model crashes with exit code 139 (segmentation fault) during initialization.

**Root Cause**: Using `None` for Dehumidification/Humidification Control Type in `ZoneHVAC:IdealLoadsAirSystem` combined with Heat Recovery causes a crash in E+ 25.1.0.

**Solution**: Use `ConstantSupplyHumidityRatio` instead of `None`:
```
ZoneHVAC:IdealLoadsAirSystem,
    ...
    ConstantSupplyHumidityRatio,  !- Dehumidification Control Type
    ,                              !- Cooling Sensible Heat Ratio (leave blank)
    ConstantSupplyHumidityRatio,  !- Humidification Control Type
    ...
```

## ECM Analysis Guidance

This model is designed as a baseline for evaluating energy conservation measures:

### Envelope ECMs
- Additional wall insulation (external/internal)
- Roof insulation upgrade
- Window replacement (U-value, SHGC)
- Air sealing improvements

### HVAC ECMs
- Heat recovery upgrade (75% → 85%)
- Demand-controlled ventilation
- Heat pump integration
- District heating optimization

### Renewable Energy
- Solar PV on roof (~320 m² available)
- Solar thermal for DHW pre-heating

## Post-Processing Notes

**DHW Energy** is not included in the EnergyPlus simulation. Add separately:
- Sveby standard: 25 kWh/m² thermal annually
- With heat pump COP 2.5: ~10 kWh/m² electricity
- Total DHW: 56,000 kWh thermal / 22,400 kWh electricity

**Total Building Energy** (electricity equivalent with heat pump):
- Space heating: 94 MWh thermal ÷ COP 3.2 = ~29 MWh elec
- DHW: 56 MWh thermal ÷ COP 2.5 = ~22 MWh elec
- FTX fans: ~10 MWh elec
- Lighting + equipment: ~90 MWh elec
- **Total: ~151 MWh/year = ~67 kWh/m²/year**

## License

This model is provided for educational and research purposes. Building data is anonymized.

## Contact

BRF Energy Toolkit Project
