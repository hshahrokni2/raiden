# Raiden - Swedish Building Energy Modeling

Professional-grade EnergyPlus energy modeling toolkit for Swedish multi-family residential buildings.

## Features

- **Swedish Building Archetypes** - TABULA/EPISCOPE-based archetypes from 1900 to present
- **Baseline Generation** - Automatic IDF generation from building geometry
- **Model Calibration** - Iterative calibration to energy declarations
- **ECM Analysis** - 12 Swedish-specific energy conservation measures
- **PV Potential** - Solar potential calculation with shading analysis

## Quick Example

```python
from src.baseline.archetypes import ArchetypeMatcher, BuildingType
from src.baseline.generator import BaselineGenerator

# Match building to archetype
matcher = ArchetypeMatcher()
archetype = matcher.match(
    construction_year=1968,
    building_type=BuildingType.MULTI_FAMILY,
    facade_material='concrete'
)

# Generate baseline IDF
generator = BaselineGenerator()
idf_path = generator.generate(
    geometry=building_geometry,
    archetype=archetype,
    output_dir=Path('./output')
)
```

## Current Model

The primary model is **BRF Sjostaden 2**, a 7-story building in Stockholm:

- 7 floors x 320 m² = 2,240 m² Atemp
- FTX ventilation with 75% heat recovery
- District heating
- Triple glazing U=1.0, walls U=0.13, roof U=0.10

## Installation

```bash
pip install -e ".[dev]"
```

## Running Simulations

```bash
# Run EnergyPlus simulation
energyplus -w weather.epw -d output model.idf
```

## Documentation

- [Getting Started](getting-started/installation.md) - Installation and setup
- [User Guide](guide/archetypes.md) - Detailed usage instructions
- [API Reference](api/baseline.md) - Python API documentation
