# Visual Analysis Agent Prompt

You have access to a visual analysis toolkit for buildings. Use these tools to extract building information from Street View imagery, satellite photos, and facade images.

## Package Location

```
/Users/hosseins/Dropbox/Dev/Raiden/src/visual/
```

## Environment

The `.env` file is auto-loaded from:
```
/Users/hosseins/Dropbox/Dev/Raiden/.env
```

## Quick Import

```python
from src.visual import (
    # Quick functions
    quick_visual_scan,        # Fast scan → dict with metrics
    get_building_geometry,    # Footprint + height
    classify_facade,          # Analyze image file

    # Full analyzers
    analyze_address,          # Full visual analysis
    analyze_building,         # By coordinates
    extract_footprint,        # Satellite footprint
)
```

## File Locations

### Main Package
| File | Path |
|------|------|
| Package init | `/Users/hosseins/Dropbox/Dev/Raiden/src/visual/__init__.py` |
| Documentation | `/Users/hosseins/Dropbox/Dev/Raiden/src/visual/README.md` |
| Env example | `/Users/hosseins/Dropbox/Dev/Raiden/src/visual/.env.example` |

### Core Components
| Component | Path |
|-----------|------|
| VisualAnalyzer | `/Users/hosseins/Dropbox/Dev/Raiden/src/analysis/visual_analyzer.py` |
| FootprintExtractor | `/Users/hosseins/Dropbox/Dev/Raiden/src/ingest/satellite_fetcher.py` |
| HeightEstimator | `/Users/hosseins/Dropbox/Dev/Raiden/src/ingest/streetview_fetcher.py` |

### AI Models
| Model | Path |
|-------|------|
| WWRDetector | `/Users/hosseins/Dropbox/Dev/Raiden/src/ai/wwr_detector.py` |
| MaterialClassifier | `/Users/hosseins/Dropbox/Dev/Raiden/src/ai/material_classifier.py` |
| MaterialClassifierV2 | `/Users/hosseins/Dropbox/Dev/Raiden/src/ai/material_classifier_v2.py` |
| FacadeAnalyzerLLM | `/Users/hosseins/Dropbox/Dev/Raiden/src/ai/facade_analyzer_llm.py` |
| ImageQualityAssessor | `/Users/hosseins/Dropbox/Dev/Raiden/src/ai/image_quality.py` |
| GroundFloorDetector | `/Users/hosseins/Dropbox/Dev/Raiden/src/ai/ground_floor_detector.py` |

### Data Fetchers
| Fetcher | Path |
|---------|------|
| StreetViewFacadeFetcher | `/Users/hosseins/Dropbox/Dev/Raiden/src/ingest/streetview_fetcher.py` |
| HistoricalStreetViewFetcher | `/Users/hosseins/Dropbox/Dev/Raiden/src/ingest/historical_streetview.py` |
| EsriSatelliteFetcher | `/Users/hosseins/Dropbox/Dev/Raiden/src/ingest/satellite_fetcher.py` |
| MapillaryFetcher | `/Users/hosseins/Dropbox/Dev/Raiden/src/ingest/image_fetcher.py` |

---

## Usage Examples

### 1. Quick Scan (Fastest)

```python
from src.visual import quick_visual_scan

info = quick_visual_scan(address="Kungsgatan 1, Stockholm")

# Returns:
# {
#     "height_m": 18.5,
#     "floors": 6,
#     "material": "brick",
#     "wwr": 0.22,
#     "building_form": "lamellhus",
#     "estimated_era": "1945_1960",
#     "footprint_area_m2": 450,
#     "confidence": 0.85,
#     "lat": 59.33,
#     "lon": 18.07
# }
```

### 2. Full Visual Analysis

```python
from src.visual import analyze_address

result = analyze_address("Bellmansgatan 16, Stockholm")

# Access all fields:
print(f"Height: {result.height_m}m")
print(f"Floors: {result.floor_count}")
print(f"Material: {result.facade_material}")
print(f"WWR average: {result.wwr_average:.0%}")
print(f"WWR by direction: {result.wwr_by_orientation}")  # {'N': 0.15, 'S': 0.25, ...}
print(f"Building form: {result.building_form}")
print(f"Era: {result.estimated_era}")
print(f"Commercial ground floor: {result.has_commercial_ground_floor}")
print(f"Likely renovated: {result.likely_renovated}")
```

### 3. Building Geometry (Footprint + Height)

```python
from src.visual import get_building_geometry

# Single building
geom = get_building_geometry(address="Vasagatan 10, Stockholm")
print(f"Footprint: {geom['footprint_area_m2']:.0f} m²")
print(f"Height: {geom['height_m']:.1f}m")
print(f"GeoJSON: {geom['footprint_geojson']}")

# Multi-building property
geom = get_building_geometry(
    addresses=["Sjöstaden 2A, Stockholm", "Sjöstaden 2B, Stockholm", "Sjöstaden 4A, Stockholm"]
)
print(f"Found {geom['num_buildings']} buildings")
print(f"Total area: {geom['total_area_m2']:.0f} m²")
for fp in geom["all_footprints"]:
    print(f"  Building: {fp['area_m2']:.0f} m²")
```

### 4. Extract Footprint from Satellite

```python
from src.visual import extract_footprint, FootprintExtractor

# Quick
footprint = extract_footprint(lat=59.30, lon=18.10)
print(f"Area: {footprint.area_m2:.0f} m²")
print(f"Method: {footprint.method}")  # sam, llm, edge_detection

# With options
extractor = FootprintExtractor(use_sam=True, use_llm=True)
footprint = extractor.extract_from_address("Kungsgatan 1, Stockholm")
```

### 5. Classify a Facade Image

```python
from src.visual import classify_facade

result = classify_facade("/path/to/facade.jpg")
print(f"Material: {result['material']}")
print(f"WWR: {result['wwr']:.0%}")
print(f"Floors: {result['floors']}")
```

### 6. Low-Level Components

```python
from src.visual import (
    VisualAnalyzer,
    WWRDetector,
    MaterialClassifierV2,
    GeometricHeightEstimator,
    FacadeAnalyzerLLM,
)
from PIL import Image

# WWR detection
detector = WWRDetector(backend="opencv")
image = Image.open("facade.jpg")
wwr, confidence = detector.calculate_wwr(image)

# Material classification
classifier = MaterialClassifierV2()
result = classifier.classify_single_image(image)
print(f"Material: {result.material} ({result.confidence:.0%})")

# LLM analysis (uses Gemini)
llm = FacadeAnalyzerLLM(backend="gemini")
analysis = llm.analyze("facade.jpg")
print(f"Form: {analysis.building_form}, Era: {analysis.estimated_era}")

# Height estimation
estimator = GeometricHeightEstimator()
height = estimator.estimate_from_floor_count(
    floor_count=6,
    building_form="lamellhus",
    has_commercial_ground=True,
)
print(f"Height: {height.height_m:.1f}m")
```

---

## Output Fields Reference

### VisualAnalysisResult

| Field | Type | Description |
|-------|------|-------------|
| `height_m` | float | Building height in meters |
| `height_confidence` | float | Confidence (0-1) |
| `floor_count` | int | Number of floors |
| `facade_material` | str | brick, concrete, plaster, glass, metal, wood |
| `material_confidence` | float | Confidence (0-1) |
| `wwr_average` | float | Average window-to-wall ratio |
| `wwr_by_orientation` | dict | {'N': 0.15, 'S': 0.25, 'E': 0.20, 'W': 0.18} |
| `wwr_confidence` | float | Confidence (0-1) |
| `building_form` | str | lamellhus, skivhus, punkthus, etc. |
| `estimated_era` | str | pre_1930, miljonprogrammet, modern, etc. |
| `has_commercial_ground_floor` | bool | Commercial on ground floor |
| `likely_renovated` | bool | Signs of renovation |

### ExtractedFootprint

| Field | Type | Description |
|-------|------|-------------|
| `geojson` | dict | GeoJSON Polygon |
| `area_m2` | float | Area in square meters |
| `confidence` | float | Extraction confidence |
| `method` | str | sam, sam_point_prompt, llm, edge_detection |
| `center_lat` | float | Center latitude |
| `center_lon` | float | Center longitude |

---

## Building Forms (Swedish)

| Form | Description |
|------|-------------|
| `lamellhus` | Slab block (3-4 stories), 1945-1985 |
| `skivhus` | Large slab (8+ stories), Miljonprogrammet |
| `punkthus` | Point tower (compact footprint) |
| `stjärnhus` | Star-shaped (3+ wings) |
| `loftgångshus` | Gallery access (external corridors) |
| `slutet_kvarter` | Closed perimeter block (pre-1940) |
| `radhus` | Row houses / terraced |

---

## API Keys (from .env)

| Variable | Used For |
|----------|----------|
| `GOOGLE_API_KEY` | Street View, Solar API, Gemini LLM |
| `MAPILLARY_TOKEN` | Alternative street images |
| `KOMILION_API_KEY` | LLM router (optional) |
| `ANTHROPIC_API_KEY` | Claude (optional) |

**No API needed:** ESRI satellite, SAM segmentation, OpenCV analysis
