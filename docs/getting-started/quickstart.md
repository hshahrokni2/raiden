# Quick Start

This guide walks through a complete workflow: matching a building to an archetype, generating a baseline model, running a simulation, and analyzing results.

## 1. Match Building to Archetype

```python
from src.baseline.archetypes import ArchetypeMatcher, BuildingType

matcher = ArchetypeMatcher()
archetype = matcher.match(
    construction_year=1968,
    building_type=BuildingType.MULTI_FAMILY,
    facade_material='concrete'
)

print(f"Matched: {archetype.name}")
print(f"Wall U-value: {archetype.envelope.wall_u_value} W/m²K")
print(f"Heat recovery: {archetype.hvac.heat_recovery_efficiency}")
```

## 2. Run Existing Model

```python
from pathlib import Path
from src.simulation.runner import run_simulation

result = run_simulation(
    idf_path=Path('sjostaden_7zone.idf'),
    weather_path=Path('stockholm.epw'),
    output_dir=Path('./output'),
    parse_results=True
)

if result.success:
    print(f"Heating: {result.parsed_results.heating_kwh_m2:.1f} kWh/m²")
    print(f"Floor area: {result.parsed_results.floor_area_m2:.0f} m²")
```

## 3. Calibrate to Energy Declaration

```python
from src.baseline.calibrator import BaselineCalibrator

calibrator = BaselineCalibrator()
result = calibrator.calibrate(
    idf_path=Path('baseline.idf'),
    weather_path=Path('stockholm.epw'),
    measured_heating_kwh_m2=33.0,  # From energy declaration
    output_dir=Path('./calibrated')
)

print(f"Calibrated: {result.calibrated_kwh_m2:.1f} kWh/m²")
print(f"Error: {result.final_error_percent:.1f}%")
```

## 4. Apply ECM

```python
from src.ecm.idf_modifier import IDFModifier

modifier = IDFModifier()
ecm_idf = modifier.apply_single(
    baseline_idf=Path('calibrated.idf'),
    ecm_id='air_sealing',
    params={'reduction_factor': 0.5},
    output_dir=Path('./ecm')
)
```

## 5. Calculate PV Potential

```python
from src.geometry.pv_potential import calculate_pv_potential

potential = calculate_pv_potential(
    roof_area_m2=320,
    latitude=59.3,  # Stockholm
    roof_type='flat'
)

print(f"Capacity: {potential.max_capacity_kwp:.1f} kWp")
print(f"Annual yield: {potential.effective_annual_yield_kwh:.0f} kWh")
```
