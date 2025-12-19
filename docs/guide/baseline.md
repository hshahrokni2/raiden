# Baseline Generation

Generate EnergyPlus IDF models from building geometry and archetypes.

## Workflow

1. Define building geometry
2. Match to Swedish archetype
3. Generate baseline IDF
4. Run simulation to verify

## Example

```python
from src.baseline.generator import BaselineGenerator
from src.baseline.archetypes import ArchetypeMatcher, BuildingType

# Match archetype
matcher = ArchetypeMatcher()
archetype = matcher.match(construction_year=1968)

# Generate baseline
generator = BaselineGenerator()
idf_path = generator.generate(
    geometry=building_geometry,
    archetype=archetype,
    latitude=59.3,
    longitude=18.0,
    output_dir=Path('./output')
)
```

## Geometry Requirements

The `BuildingGeometry` dataclass requires:

- `footprint_coords_local` - List of (x, y) coordinates
- `footprint_area_m2` - Footprint area in m²
- `gross_floor_area_m2` - Total heated floor area
- `floors` - Number of floors
- `floor_height_m` - Floor-to-floor height
