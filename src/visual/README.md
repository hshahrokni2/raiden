# Raiden Visual Analysis Package

Standalone building visual intelligence toolkit - can be used independently of energy analysis.

## Installation

The visual package is part of Raiden. Ensure you have the dependencies:

```bash
pip install pillow opencv-python geopy requests rich

# Optional: For SAM segmentation
pip install segment-anything torch

# Optional: For LLM analysis
pip install google-generativeai anthropic openai
```

## Quick Start

```python
from src.visual import analyze_address, extract_footprint, quick_visual_scan

# Quick scan - returns essential metrics
info = quick_visual_scan(address="Kungsgatan 1, Stockholm")
print(f"Height: {info['height_m']}m, {info['floors']} floors")
print(f"Material: {info['material']}, WWR: {info['wwr']:.0%}")

# Full analysis
result = analyze_address("Bellmansgatan 16, Stockholm")
print(f"Building form: {result.building_form}")
print(f"Era: {result.estimated_era}")
print(f"WWR by direction: {result.wwr_by_orientation}")

# Extract footprint from satellite
footprint = extract_footprint(lat=59.30, lon=18.10)
print(f"Area: {footprint.area_m2:.0f} m²")
```

## Environment Variables

| Variable | Required | Purpose |
|----------|----------|---------|
| `GOOGLE_API_KEY` | For Street View | Google Street View, Solar API |
| `MAPILLARY_TOKEN` | Alternative | Mapillary street images |
| `KOMILION_API_KEY` | Optional | LLM router (free tier) |
| `ANTHROPIC_API_KEY` | Optional | Claude LLM |
| `OPENAI_API_KEY` | Optional | GPT-4o |

**No API key needed for:**
- ESRI satellite imagery (always free)
- SAM segmentation (local model)
- OpenCV-based analysis

## API Reference

### 1. Quick Functions

#### `quick_visual_scan(address=None, lat=None, lon=None) -> dict`

Fast scan returning essential building metrics:

```python
info = quick_visual_scan(address="Storgatan 5, Malmö")
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
#     "lat": 55.60,
#     "lon": 13.00
# }
```

#### `get_building_geometry(address=None, lat=None, lon=None, addresses=None) -> dict`

Get complete geometry including footprint:

```python
# Single building
geom = get_building_geometry(address="Vasagatan 10, Stockholm")

# Multi-building property (BRF with multiple entrances)
geom = get_building_geometry(
    addresses=["Sjöstaden 2A", "Sjöstaden 2B", "Sjöstaden 4A"],
)
print(f"Found {geom['num_buildings']} buildings")
print(f"Total area: {geom['total_area_m2']:.0f} m²")

for fp in geom["all_footprints"]:
    print(f"  Building: {fp['area_m2']:.0f} m² at ({fp['center_lat']:.4f}, {fp['center_lon']:.4f})")
```

#### `classify_facade(image_path: str) -> dict`

Classify a facade from an image file:

```python
result = classify_facade("./facade_photo.jpg")
# Returns:
# {
#     "material": "concrete",
#     "material_confidence": 0.92,
#     "wwr": 0.25,
#     "wwr_confidence": 0.78,
#     "floors": 8,
#     "building_form": "skivhus",
#     "ground_floor_use": "commercial"
# }
```

---

### 2. VisualAnalyzer (Full Analysis)

Complete facade analysis from Street View imagery:

```python
from src.visual import VisualAnalyzer, VisualAnalysisResult

analyzer = VisualAnalyzer(
    google_api_key="your-key",  # Or set GOOGLE_API_KEY env var
    ai_backend="opencv",         # "opencv", "sam", or "lang_sam"
    ai_device="cpu",             # "cpu", "cuda", "mps"
)

# Analyze by coordinates
result: VisualAnalysisResult = analyzer.analyze_building(
    lat=59.30,
    lon=18.10,
    footprint_geojson=None,      # Optional - auto-generates if not provided
    images_per_facade=3,         # Images per direction (N, S, E, W)
    use_historical=True,         # Include historical Street View
    historical_years=3,          # How far back
)

# Access results
print(f"Height: {result.height_m}m (confidence: {result.height_confidence:.0%})")
print(f"Floors: {result.floor_count}")
print(f"Material: {result.facade_material} ({result.material_confidence:.0%})")
print(f"WWR average: {result.wwr_average:.0%}")
print(f"WWR by direction: {result.wwr_by_orientation}")  # {'N': 0.15, 'S': 0.25, ...}
print(f"Building form: {result.building_form}")  # lamellhus, skivhus, punkthus, etc.
print(f"Estimated era: {result.estimated_era}")
print(f"Likely renovated: {result.likely_renovated}")
print(f"Ground floor commercial: {result.has_commercial_ground_floor}")

# Analyze by address
result = analyzer.analyze_address("Bellmansgatan 16, Stockholm")

# Analyze from existing images
result = analyzer.analyze_from_images({
    "N": ["./north1.jpg", "./north2.jpg"],
    "S": ["./south1.jpg"],
    "E": ["./east1.jpg"],
})
```

#### VisualAnalysisResult Fields

| Field | Type | Description |
|-------|------|-------------|
| `wwr_by_orientation` | `Dict[str, float]` | WWR per direction (N, S, E, W) |
| `wwr_average` | `float` | Average WWR across all facades |
| `wwr_confidence` | `float` | Confidence in WWR detection |
| `facade_material` | `str` | brick, concrete, plaster, glass, metal, wood |
| `material_confidence` | `float` | Confidence in material detection |
| `height_estimate` | `GeometricHeightEstimate` | Full height estimation result |
| `height_m` | `float` | Building height in meters |
| `height_confidence` | `float` | Confidence in height |
| `floor_count` | `int` | Number of floors |
| `ground_floor` | `GroundFloorResult` | Ground floor analysis |
| `has_commercial_ground_floor` | `bool` | True if commercial on ground floor |
| `building_form` | `str` | lamellhus, skivhus, punkthus, etc. |
| `estimated_era` | `str` | pre_1930, miljonprogrammet, modern, etc. |
| `likely_renovated` | `bool` | True if renovation signs detected |
| `llm_facade_result` | `FacadeAnalysis` | Raw LLM analysis (if used) |
| `num_images_analyzed` | `int` | Number of images processed |
| `orientations_covered` | `List[str]` | Which directions were covered |
| `saved_image_paths` | `Dict[str, List[str]]` | Paths to saved images |

---

### 3. FootprintExtractor (Satellite Footprints)

Extract building footprints from satellite imagery:

```python
from src.visual import FootprintExtractor, ExtractedFootprint

extractor = FootprintExtractor(
    use_sam=True,   # Use SAM for segmentation (best accuracy)
    use_llm=True,   # Fall back to LLM if SAM fails
)

# Single building by coordinates
footprint: ExtractedFootprint = extractor.extract_from_coordinates(
    lat=59.30,
    lon=18.10,
    zoom=19,           # Satellite zoom level
    method="auto",     # "auto", "sam", "llm", or "edge"
)

print(f"Area: {footprint.area_m2:.0f} m²")
print(f"Method: {footprint.method}")  # sam, llm, edge_detection
print(f"Confidence: {footprint.confidence:.0%}")
print(f"GeoJSON: {footprint.geojson}")

# By address
footprint = extractor.extract_from_address("Kungsgatan 1, Stockholm")

# Multi-building property (THE SMART WAY!)
# Provide all addresses from energy declaration
result = extractor.extract_all_buildings(
    addresses=[
        "Sjöstaden 2A, Stockholm",
        "Sjöstaden 2B, Stockholm",
        "Sjöstaden 4A, Stockholm",
        "Sjöstaden 4B, Stockholm",
    ],
)

print(f"Found {result.num_buildings} buildings")
print(f"Total area: {result.total_area_m2:.0f} m²")

for fp in result.footprints:
    print(f"  {fp.area_m2:.0f} m² at ({fp.center_lat:.4f}, {fp.center_lon:.4f})")
```

#### ExtractedFootprint Fields

| Field | Type | Description |
|-------|------|-------------|
| `geojson` | `Dict` | GeoJSON Polygon geometry |
| `coordinates` | `List[Tuple[float, float]]` | [(lon, lat), ...] |
| `area_m2` | `float` | Area in square meters |
| `confidence` | `float` | Extraction confidence |
| `method` | `str` | sam, sam_point_prompt, llm, edge_detection |
| `center_lat` | `float` | Building center latitude |
| `center_lon` | `float` | Building center longitude |
| `bbox` | `Tuple[float, float, float, float]` | Bounding box |
| `notes` | `List[str]` | Additional info |

---

### 4. Height Estimation

Multi-position geometric height estimation from Street View:

```python
from src.visual import GeometricHeightEstimator, StreetViewFacadeFetcher

# Fetch Street View images
fetcher = StreetViewFacadeFetcher(api_key="your-key")
images = fetcher.fetch_multi_facade_images(
    footprint_geojson,
    images_per_facade=3,
)

# Estimate height
estimator = GeometricHeightEstimator()

# From multiple Street View positions (MOST ACCURATE)
height_result = estimator.estimate_from_multiple_positions(
    images=images["S"],  # List of StreetViewImage
    facade_lat=59.30,
    facade_lon=18.10,
    roof_position_pct=0.85,  # Where roof appears in image (0-1)
)

print(f"Height: {height_result.height_m:.1f}m")
print(f"Method: {height_result.method}")
print(f"Confidence: {height_result.confidence:.0%}")

# From floor count (when you know floors from other source)
floor_result = estimator.estimate_from_floor_count(
    floor_count=6,
    building_form="lamellhus",
    has_commercial_ground=True,  # Commercial = higher ceiling
    has_attic=False,
)

# Combine estimates (geometric gets priority if high confidence)
combined = estimator.combine_estimates(
    geometric=height_result,
    floor_based=floor_result,
)
```

---

### 5. Individual AI Components

#### WWR Detector

```python
from src.visual import WWRDetector
from PIL import Image

detector = WWRDetector(
    backend="opencv",  # "opencv", "sam", or "lang_sam"
    device="cpu",
)

image = Image.open("facade.jpg")
wwr, confidence = detector.calculate_wwr(
    image,
    crop_facade=True,     # Crop to building only
    use_sam_crop=True,    # Use SAM for building segmentation
)

print(f"WWR: {wwr:.0%} (confidence: {confidence:.0%})")
```

#### Material Classifier

```python
from src.visual import MaterialClassifierV2
from PIL import Image

classifier = MaterialClassifierV2(device="cpu")

# Single image
image = Image.open("facade.jpg")
result = classifier.classify_single_image(image)
print(f"Material: {result.material} ({result.confidence:.0%})")

# Multiple images (higher accuracy)
images = [Image.open(p) for p in ["f1.jpg", "f2.jpg", "f3.jpg"]]
result = classifier.classify_multi_image(
    images,
    use_sam_crop=True,
    building_type="residential",
)
```

#### Facade LLM Analyzer

```python
from src.visual import FacadeAnalyzerLLM

# Uses Gemini by default (fast & accurate)
analyzer = FacadeAnalyzerLLM(backend="gemini")

# Analyze single image
result = analyzer.analyze("facade.jpg")
print(f"Material: {result.facade_material}")
print(f"Floors: {result.visible_floors}")
print(f"WWR: {result.wwr_average:.0%}")
print(f"Building form: {result.building_form}")
print(f"Era: {result.estimated_era}")
print(f"Ground floor: {result.ground_floor_use}")

# Analyze multiple images (per orientation)
result = analyzer.analyze_multiple({
    "N": ["north1.jpg", "north2.jpg"],
    "S": ["south1.jpg"],
}, max_images=3)
```

---

## Use Cases

### 1. Real Estate Analysis

```python
from src.visual import quick_visual_scan, get_building_geometry

def analyze_property(address: str) -> dict:
    """Get property metrics for real estate analysis."""
    visual = quick_visual_scan(address=address)
    geometry = get_building_geometry(address=address)

    return {
        "address": address,
        "floors": visual.get("floors"),
        "height_m": visual.get("height_m"),
        "facade_material": visual.get("material"),
        "building_form": visual.get("building_form"),
        "era": visual.get("estimated_era"),
        "footprint_m2": geometry.get("footprint_area_m2"),
        "has_commercial": visual.get("ground_floor_use") == "commercial",
    }

# Analyze portfolio
portfolio = [
    analyze_property("Kungsgatan 1, Stockholm"),
    analyze_property("Drottninggatan 50, Stockholm"),
    analyze_property("Sveavägen 25, Stockholm"),
]
```

### 2. Urban Planning / GIS

```python
from src.visual import FootprintExtractor, EsriSatelliteFetcher

def map_neighborhood(addresses: list) -> list:
    """Extract all building footprints in a neighborhood."""
    extractor = FootprintExtractor()
    footprints = []

    for addr in addresses:
        fp = extractor.extract_from_address(addr)
        if fp:
            footprints.append({
                "address": addr,
                "geojson": fp.geojson,
                "area_m2": fp.area_m2,
            })

    return footprints

# Export to GeoJSON for QGIS/GIS
import json
features = [
    {"type": "Feature", "geometry": fp["geojson"], "properties": {"address": fp["address"]}}
    for fp in footprints
]
geojson = {"type": "FeatureCollection", "features": features}
with open("neighborhood.geojson", "w") as f:
    json.dump(geojson, f)
```

### 3. Insurance / Risk Assessment

```python
from src.visual import analyze_address

def assess_building_risk(address: str) -> dict:
    """Assess building characteristics for insurance."""
    result = analyze_address(address)

    risk_factors = []

    # Old buildings may have higher risk
    if result.estimated_era in ["pre_1930", "1930_1945"]:
        risk_factors.append("old_construction")

    # Low WWR = less natural light
    if result.wwr_average < 0.15:
        risk_factors.append("low_daylight")

    # Commercial ground floor = different use
    if result.has_commercial_ground_floor:
        risk_factors.append("mixed_use")

    # Material-specific risks
    if result.facade_material == "wood":
        risk_factors.append("fire_risk_facade")

    return {
        "address": address,
        "construction_era": result.estimated_era,
        "building_form": result.building_form,
        "facade_material": result.facade_material,
        "height_m": result.height_m,
        "risk_factors": risk_factors,
    }
```

### 4. Integration with Other Pipelines

```python
from src.visual import VisualAnalyzer, FootprintExtractor

class MyCustomPipeline:
    def __init__(self):
        self.visual = VisualAnalyzer()
        self.footprint = FootprintExtractor()

    def process_building(self, address: str) -> dict:
        # Get visual data
        visual_result = self.visual.analyze_address(address)

        # Get footprint
        fp = self.footprint.extract_from_address(address)

        # Combine with your own logic
        return {
            "visual": {
                "height": visual_result.height_m,
                "floors": visual_result.floor_count,
                "material": visual_result.facade_material,
                "wwr": visual_result.wwr_by_orientation,
            },
            "geometry": {
                "footprint": fp.geojson if fp else None,
                "area_m2": fp.area_m2 if fp else None,
            },
            # Add your custom analysis here...
        }
```

---

## Building Forms (Swedish)

The analyzer detects these Swedish building forms:

| Form | Description | Typical Era |
|------|-------------|-------------|
| `lamellhus` | Slab block (3-4 stories) | 1945-1985 |
| `skivhus` | Large slab (8+ stories) | 1960-1975 (Miljonprogrammet) |
| `punkthus` | Point tower (8+ stories, compact) | 1950-1970 |
| `stjärnhus` | Star-shaped (3+ wings) | 1950-1965 |
| `loftgångshus` | Gallery access (external corridors) | 1965-1980 |
| `slutet_kvarter` | Closed perimeter block | Pre-1940 |
| `vinkelbyggnad` | L-shaped | Various |
| `radhus` | Row houses / terraced | Various |

---

## Facade Materials

Detected materials:

- `brick` - Exposed brick facade
- `concrete` - Exposed concrete / brutalist
- `plaster` - Rendered / plastered (white, colored)
- `glass` - Glass curtain wall
- `metal` - Metal cladding (aluminum, steel)
- `wood` - Wood cladding / timber facade
- `stone` - Natural stone facade
- `mixed` - Multiple materials

---

## Error Handling

All functions handle errors gracefully:

```python
from src.visual import quick_visual_scan

result = quick_visual_scan(address="Invalid Address XYZ123")

if "error" in result:
    print(f"Error: {result['error']}")
elif "visual_error" in result:
    print(f"Visual analysis failed: {result['visual_error']}")
else:
    print(f"Success: {result['floors']} floors")
```

---

## Performance Tips

1. **Batch processing**: Create analyzer once, reuse for multiple buildings
2. **Disable historical**: Set `use_historical=False` for faster analysis
3. **Reduce images**: Set `images_per_facade=1` for quick scans
4. **Local SAM**: Download SAM checkpoint once to `~/.cache/sam/`
5. **Use OpenCV backend**: Fastest for WWR (no GPU needed)
