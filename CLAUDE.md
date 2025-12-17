# Raiden - Swedish Building Energy Modeling Toolkit

## Project Overview

Full-stack toolkit for Swedish multi-family residential building energy analysis. Combines geospatial data ingestion, AI-powered facade analysis, and EnergyPlus simulation.

## Architecture

```
Input BRF JSON → OSM/Overture/Mapillary → AI Analysis → Enriched Data → EnergyPlus IDF → Simulation
```

## Key Components

### 1. Data Ingestion (`src/ingest/`)
- `brf_parser.py` - Parse Swedish BRF energy declaration JSON
- `overture_fetcher.py` - Fetch building data from Overture Maps
- `image_fetcher.py` - Fetch street-level imagery (Mapillary)

### 2. Analysis (`src/analysis/`)
- `u_value_calculator.py` - Estimate U-values by construction era
- `shading_solar.py` - Solar potential and shading analysis

### 3. Visualization (`src/visualization/`)
- `building_3d.py` - Generate interactive 3D HTML viewers
- `server.py` - Local visualization server

### 4. Core (`src/core/`)
- `models.py` - Data models
- `coordinates.py` - SWEREF99 TM ↔ WGS84 conversion
- `config.py` - Configuration

### 5. Scripts (`scripts/`)
- `process_sjostaden.py` - Full pipeline for Sjostaden building
- `export_idf.py` - Generate EnergyPlus IDF files

### 6. EnergyPlus Model
- `sjostaden_7zone.idf` - Working 7-zone model
- `output_final/` - Latest simulation results
- See `TECHNICAL_NOTES.md`, `CALIBRATION.md`, `DEVELOPMENT_LOG.md`

## Data Sources

| Source | Data | Module |
|--------|------|--------|
| OSM/Overpass | Building footprints, materials | `overture_fetcher.py` |
| Overture Maps | Heights, floors, building type | `overture_fetcher.py` |
| Mapillary | Street-level imagery | `image_fetcher.py` |
| Swedish BRF | Energy declarations | `brf_parser.py` |

## Cached Data (`data/cache/`)
- `osm/` - OpenStreetMap query cache
- `images/mapillary/` - Mapillary tile cache

## Example Data (`examples/sjostaden_2/`)
- `BRF_Sjostaden_2.geojson` - Building footprint
- `BRF_Sjostaden_2_enriched.json` - Enriched building data
- `viewer.html` - 3D visualization
- `energyplus/` - EnergyPlus model and results

## Current State

### Working
- BRF JSON parsing
- OSM/Overture data fetching
- U-value estimation
- EnergyPlus model (42 kWh/m², target 33)
- 3D visualization

### Pending
- Full Mapillary integration for WWR detection
- AI facade analysis (Grounded SAM)
- Solar shading calculations
- Model calibration

## EnergyPlus 25.1.0 Bug (CRITICAL)

When using `ZoneHVAC:IdealLoadsAirSystem` with heat recovery:
```
ConstantSupplyHumidityRatio,  !- Dehumidification Control Type
,                              !- Cooling Sensible Heat Ratio (BLANK!)
ConstantSupplyHumidityRatio,  !- Humidification Control Type
```
Using `None` causes segmentation fault (exit code 139).

## Quick Commands

```bash
# Install
pip install -e .

# Process Sjostaden
python scripts/process_sjostaden.py

# Run EnergyPlus simulation
energyplus -w SWE_Stockholm.Arlanda.024600_IWEC.epw -d output sjostaden_7zone.idf

# Start 3D viewer
python -m src.visualization.server
```

## Swedish Standards

- **BBR 6:251**: Ventilation 0.35 l/s/m²
- **Sveby**: Internal loads (8 W/m² lighting, 10 W/m² equipment)
- **TABULA/EPISCOPE**: Building archetypes

## Files to Read First

1. `README.md` - Full toolkit overview
2. `TECHNICAL_NOTES.md` - EnergyPlus model details
3. `CALIBRATION.md` - Calibration strategies
4. `DEVELOPMENT_LOG.md` - Debugging history
