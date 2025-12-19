# ECM Module

Energy Conservation Measures for Swedish buildings.

## IDF Modifier

Apply ECMs to EnergyPlus IDF files.

::: src.ecm.idf_modifier.IDFModifier
    options:
      show_root_heading: true

## ECM Catalog

Available ECMs with Swedish-specific parameters.

::: src.ecm.catalog.ECMCatalog
    options:
      show_root_heading: true

::: src.ecm.catalog.ECMDefinition
    options:
      show_root_heading: true

## Supported ECMs

| ECM ID | Description | Typical Savings |
|--------|-------------|-----------------|
| `wall_insulation_external` | External wall insulation | 10-20% |
| `wall_insulation_internal` | Internal wall insulation | 5-10% |
| `roof_insulation` | Attic/roof insulation | 5-15% |
| `window_replacement` | Triple glazing upgrade | 5-10% |
| `air_sealing` | Infiltration reduction | 5-10% |
| `ftx_upgrade` | Heat recovery upgrade | 10-20% |
| `ftx_installation` | Install FTX system | 20-40% |
| `demand_controlled_ventilation` | DCV with CO2 sensors | 5-15% |
| `solar_pv` | Rooftop solar PV | N/A (generation) |
| `led_lighting` | LED retrofit | 3-5% |
| `smart_thermostats` | Night setback control | 5-10% |
| `heat_pump_integration` | Heat pump addition | 20-40% |
