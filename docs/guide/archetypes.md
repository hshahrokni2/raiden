# Swedish Building Archetypes

Raiden includes 7 Swedish building archetypes covering construction from 1900 to present, based on TABULA/EPISCOPE building stock studies and Boverket BBR requirements.

## Archetype Eras

| Era | Name | Key Characteristics |
|-----|------|---------------------|
| Pre-1945 | Funkis/Older | Solid brick, natural ventilation |
| 1945-1960 | Folkhemmet | Post-war standardization |
| 1961-1975 | Miljonprogrammet | Concrete panels, F-ventilation |
| 1976-1985 | Post Oil Crisis | Improved insulation |
| 1986-1995 | Well Insulated | FTX becoming common |
| 1996-2010 | Modern | Good BBR compliance |
| 2011+ | Low Energy | Near passive house |

## Million Programme (Miljonprogrammet)

The most common archetype for Swedish renovation projects:

```python
archetype = SWEDISH_ARCHETYPES["1961_1975_concrete"]

# Typical properties
print(f"Wall U: {archetype.envelope.wall_u_value}")  # 0.50 W/m²K
print(f"Window U: {archetype.envelope.window_u_value}")  # 2.0 W/m²K
print(f"Infiltration: {archetype.envelope.infiltration_ach}")  # 0.15 ACH
print(f"Ventilation: {archetype.hvac.ventilation_type}")  # F-system
```

## Matching Algorithm

The `ArchetypeMatcher` prioritizes:

1. **Construction year** - Must fall within archetype era
2. **Building type** - Multi-family, single-family, etc.
3. **Facade material** - Concrete, brick, render, wood

```python
matcher = ArchetypeMatcher()
archetype = matcher.match(
    construction_year=1968,
    building_type=BuildingType.MULTI_FAMILY,
    facade_material='concrete'  # Optional but improves accuracy
)
```

## Customizing Archetypes

Override specific properties while keeping archetype defaults:

```python
from dataclasses import replace

# Start with standard archetype
base = SWEDISH_ARCHETYPES["1961_1975_concrete"]

# Override envelope properties
custom_envelope = replace(
    base.envelope,
    window_u_value=1.2,  # Already upgraded windows
    infiltration_ach=0.08  # After air sealing
)

custom = replace(base, envelope=custom_envelope)
```
