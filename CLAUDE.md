# Raiden - Swedish Building ECM Simulator

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
│  │ GEOMETRY (src/geometry/) - NEW                            │   │
│  │ • Wall areas per orientation (N/S/E/W)                   │   │
│  │ • Window areas from WWR                                   │   │
│  │ • PV potential (roof area, slope, shading)               │   │
│  │ • Thermal mass from materials                            │   │
│  └──────────────────────────────────────────────────────────┘   │
│         ▼                                                        │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ BASELINE (src/baseline/) - NEW                            │   │
│  │ • Archetype matching (7 Swedish eras defined)            │   │
│  │ • Auto-generate EnergyPlus IDF                           │   │
│  │ • Calibrate to energy declaration (±10%)                 │   │
│  └──────────────────────────────────────────────────────────┘   │
│         ▼                                                        │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ ECM ENGINE (src/ecm/) - NEW                               │   │
│  │ • 12 Swedish ECMs defined with constraints               │   │
│  │ • Constraint-aware: NO facade insulation on brick        │   │
│  │ • Combination generator (pruned, no dominated options)   │   │
│  │ • IDF modifier (apply ECMs to baseline)                  │   │
│  └──────────────────────────────────────────────────────────┘   │
│         ▼                                                        │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ SIMULATION (src/simulation/) - NEW                        │   │
│  │ • Parallel EnergyPlus execution                          │   │
│  │ • Results parsing                                         │   │
│  └──────────────────────────────────────────────────────────┘   │
│         ▼                                                        │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ ROI (src/roi/) - NEW                                      │   │
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
| `src/ingest/` | EXISTING | brf_parser, overture_fetcher, image_fetcher |
| `src/geometry/` | STUB | building_geometry, pv_potential, thermal_mass |
| `src/baseline/` | STUB | archetypes (COMPLETE), generator, calibrator |
| `src/ecm/` | STUB | catalog (COMPLETE), constraints, combinations, idf_modifier |
| `src/simulation/` | STUB | runner, results |
| `src/roi/` | STUB | costs_sweden (COMPLETE), calculator |

**COMPLETE** = Fully implemented with data
**STUB** = Structure defined, needs implementation

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

### ECM Catalog (COMPLETE)
12 ECMs defined in `src/ecm/catalog.py`:
- **Envelope**: wall_external_insulation, wall_internal_insulation, roof_insulation, window_replacement, air_sealing
- **HVAC**: ftx_upgrade, ftx_installation, demand_controlled_ventilation, heat_pump_integration
- **Renewable**: solar_pv
- **Controls**: smart_thermostats, led_lighting

Each ECM has:
- Parameters (with ranges)
- Constraints (e.g., `facade_material not_in ['brick']`)
- Swedish costs (SEK)
- Typical savings

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

The Sjostaden model in `examples/sjostaden_2/energyplus/` is a working 7-zone model:
- 2,240 m² multi-family
- 42 kWh/m² heating (target: 33 kWh/m² - needs calibration)
- FTX with 75% heat recovery
- See `TECHNICAL_NOTES.md` for full specs

## Data Sources (All Free/Public)

| Source | Data | Access |
|--------|------|--------|
| OSM/Overture | Footprint, height | API |
| Mapillary | Street view, facades | API key |
| Lantmäteriet | LiDAR, terrain | Open data |
| Energideklaration | Energy use, year | Boverket |
| Sveby | Load defaults | Published |

## Files to Read

1. `src/baseline/archetypes.py` - Swedish building archetypes (COMPLETE)
2. `src/ecm/catalog.py` - ECM definitions (COMPLETE)
3. `src/ecm/constraints.py` - Constraint engine
4. `src/roi/costs_sweden.py` - Swedish costs (COMPLETE)
5. `examples/sjostaden_2/energyplus/TECHNICAL_NOTES.md` - Model details
6. `examples/sjostaden_2/energyplus/DEVELOPMENT_LOG.md` - E+ debugging

## Next Steps

1. Implement `BuildingGeometryCalculator` in `src/geometry/building_geometry.py`
2. Implement `BaselineGenerator._generate_*` methods
3. Implement `IDFModifier._apply_*` methods for each ECM
4. Test end-to-end with Sjostaden building
5. Validate against measured data
