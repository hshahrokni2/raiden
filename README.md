# BRF Energy Toolkit

Building metadata enrichment and energy simulation toolkit for Swedish BRF (Bostadsrättsförening) properties.

## Overview

This toolkit takes BRF building data from Swedish energy declarations and enriches it with:

- **Window-to-Wall Ratio (WWR)** detection from facade images
- **Facade material** classification (brick, concrete, glass, etc.)
- **U-value** estimation based on construction era
- **Solar PV potential** analysis
- **Shading analysis** from neighboring buildings and trees
- **3D visualization** for client presentations
- **EnergyPlus IDF** generation for energy simulation

## Quick Start

```bash
# Install dependencies
pip install -e .

# Process a BRF JSON file
python scripts/process_sjostaden.py

# Or use the CLI
brf enrich data/input/BRF_Sjostaden_2.json --output output/

# Generate 3D visualization
brf visualize data/input/BRF_Sjostaden_2.json --serve
```

## Features

### Data Enrichment Pipeline

```
Input BRF JSON → OSM/Overture Data → AI Analysis → Enriched JSON + IDF + 3D Viewer
```

1. **Parse** - Load BRF JSON from energy declarations (SWEREF99 TM coordinates)
2. **Fetch** - Query OpenStreetMap and Overture Maps for additional building data
3. **Analyze** - Run AI models for WWR detection and material classification
4. **Estimate** - Calculate U-values, infiltration, and solar potential
5. **Export** - Generate enriched JSON, EnergyPlus IDF, and 3D visualization

### 3D Visualization

Interactive Three.js viewer that clients can open directly in their browser:
- Color by energy class, height, or material
- Click buildings to see details
- Pan, zoom, and rotate
- No server required (standalone HTML)

### AI-Powered Analysis

Using Meta's latest vision models:
- **Grounded SAM** / **LangSAM** - Text-prompted window segmentation
- **DINOv2** - Zero-shot material classification
- **Depth Anything V3** - Facade 3D reconstruction (planned)

### Swedish Building Standards

Built-in knowledge of Swedish building regulations (BBR):
- U-value estimates by construction era
- WWR patterns by building period
- Material prevalence by region and era

## Project Structure

```
brf-energy-toolkit/
├── src/
│   ├── core/           # Data models, coordinates, config
│   ├── ingest/         # BRF parser, OSM/Overture fetchers
│   ├── ai/             # WWR detection, material classification
│   ├── export/         # EnergyPlus IDF, JSON exporters
│   └── visualization/  # 3D viewer generation
├── data/
│   ├── input/          # Input BRF JSON files
│   ├── cache/          # Cached API responses
│   └── enriched/       # Output files
├── scripts/            # Processing scripts
└── examples/           # Example outputs
```

## Data Sources (No Account Required)

| Source | Data | Access |
|--------|------|--------|
| [Geofabrik](https://download.geofabrik.de/europe/sweden.html) | OSM buildings, materials | Direct download |
| [Overture Maps](https://docs.overturemaps.org/) | Buildings, height, floors | `pip install overturemaps` |
| [Lantmäteriet](https://www.lantmateriet.se/en/geodata/geodata-products/open-data/) | LiDAR, terrain | CC0 open data |
| Overpass API | Real-time OSM queries | No auth needed |

## Installation

### Basic Installation

```bash
pip install -e .
```

### With AI Models

```bash
pip install -e ".[ai]"
```

### Development

```bash
pip install -e ".[dev]"
```

## CLI Commands

```bash
# Enrich a BRF file
brf enrich INPUT.json [--output DIR] [--osm] [--overture] [--analyze] [--idf]

# Generate 3D visualization
brf visualize INPUT.json [--output FILE] [--serve] [--port PORT] [--color SCHEME]

# Show file info
brf info INPUT.json

# Fetch OSM data for location
brf fetch-osm LAT LON [--radius METERS] [--output FILE]
```

## Input JSON Format

The toolkit expects BRF JSON in this format (from energy declarations):

```json
{
  "brf_name": "BRF SJÖSTADEN 2",
  "coordinate_system": "EPSG:3006 (SWEREF99 TM)",
  "buildings": [
    {
      "building_id": 1,
      "geometry": {
        "type": "MultiPolygon",
        "height_meters": 16.6,
        "coordinates_3d": [[x, y, z], ...],
        "ground_footprint": [[x, y, 0], ...]
      },
      "properties": {
        "building_info": { "construction_year": 2003, ... },
        "dimensions": { "heated_area_sqm": 15350, ... },
        "energy": { "energy_class": "B", ... },
        "location": { "address": "...", ... }
      }
    }
  ],
  "summary": { ... }
}
```

## Output Formats

### Enriched JSON
Full building data with extracted metadata:
- Window-to-wall ratios by facade orientation
- Facade material and confidence
- U-values for envelope components
- Solar potential estimates
- Shading analysis

### EnergyPlus IDF
Ready for simulation:
- Building geometry
- Zone definitions
- Construction materials
- Basic HVAC system

### GeoJSON
For GIS integration:
- Building footprints in WGS84
- Key properties as attributes

### 3D HTML Viewer
Standalone visualization:
- Interactive Three.js scene
- Building info on click
- Color scheme options

## Roadmap

- [ ] Street View image acquisition
- [ ] Mapillary integration
- [ ] Full Grounded SAM pipeline
- [ ] Depth Anything V3 for facade geometry
- [ ] Lantmäteriet LiDAR processing
- [ ] Solar shading from UMEP/SEBE
- [ ] Complete EnergyPlus workflow
- [ ] Batch processing for multiple BRFs

## License

MIT

## Acknowledgments

- Swedish energy declaration data format
- OpenStreetMap contributors
- Overture Maps Foundation
- Meta AI (SAM, DINOv2)
