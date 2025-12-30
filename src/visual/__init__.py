"""
Raiden Visual Analysis Package - Standalone Building Visual Intelligence

ENV FILE: Copy src/visual/.env.example to your project root as .env

This package provides visual analysis capabilities for buildings using:
- Google Street View imagery
- Satellite/aerial imagery (ESRI - FREE, no API key)
- AI models (SAM, LLMs, OpenCV)

Can be used INDEPENDENTLY of the energy analysis pipeline.

## Quick Start

```python
from src.visual import analyze_building, analyze_address, extract_footprint

# Analyze a building by coordinates
result = analyze_building(lat=59.30, lon=18.10)
print(f"Height: {result.height_m}m")
print(f"Material: {result.facade_material}")
print(f"WWR: {result.wwr_average:.0%}")

# Analyze by address
result = analyze_address("Bellmansgatan 16, Stockholm")

# Extract building footprint from satellite
footprint = extract_footprint(lat=59.30, lon=18.10)
print(f"Area: {footprint.area_m2:.0f} m²")
```

## Components

1. **VisualAnalyzer** - Complete facade analysis from Street View
   - Window-to-Wall Ratio (WWR) per orientation
   - Facade material classification
   - Building height (multi-position geometric triangulation)
   - Floor counting
   - Ground floor commercial detection
   - Building form and era estimation

2. **FootprintExtractor** - Building footprint from satellite imagery
   - SAM (Segment Anything) segmentation
   - LLM-based extraction (Gemini/Claude)
   - Edge detection fallback
   - Multi-building property support

3. **HeightEstimator** - Building height from Street View
   - Multi-position geometric triangulation
   - Floor-based estimation
   - Weighted combination with confidence

4. **Individual AI Components**
   - WWRDetector: Window detection
   - MaterialClassifier: Facade material (brick, concrete, etc.)
   - FacadeAnalyzerLLM: LLM-based comprehensive analysis

## Environment Variables

Required for full functionality:
- GOOGLE_API_KEY: Google Street View, Solar API
- MAPILLARY_TOKEN: Mapillary street-level images (alternative to GSV)

Optional (for LLM analysis):
- KOMILION_API_KEY: Komilion router (FREE tier available)
- ANTHROPIC_API_KEY: Claude
- OPENAI_API_KEY: GPT-4o

No API key needed:
- ESRI satellite imagery (always free)
- SAM segmentation (local model)
- OpenCV-based analysis (local)
"""

# Auto-load .env file if python-dotenv is available
import os
from pathlib import Path

def _load_env():
    """Auto-load .env from project root or current directory."""
    try:
        from dotenv import load_dotenv

        # Try project root first (3 levels up from src/visual/__init__.py)
        project_root = Path(__file__).parent.parent.parent
        env_file = project_root / ".env"

        if env_file.exists():
            load_dotenv(env_file)
            return True

        # Try current working directory
        cwd_env = Path.cwd() / ".env"
        if cwd_env.exists():
            load_dotenv(cwd_env)
            return True

    except ImportError:
        pass  # python-dotenv not installed, use existing env vars

    return False

_load_env()

# Main analyzers
from ..analysis.visual_analyzer import (
    VisualAnalyzer,
    VisualAnalysisResult,
    GroundFloorResult,
    analyze_building_visually,
    analyze_address_visually,
)

# Footprint extraction
from ..ingest.satellite_fetcher import (
    FootprintExtractor,
    ExtractedFootprint,
    MultiFootprintResult,
    extract_footprint_from_satellite,
    # Satellite fetchers
    SatelliteImage,
    SatelliteFetcher,
    EsriSatelliteFetcher,
    fetch_esri_satellite,
)

# Height estimation
from ..ingest.streetview_fetcher import (
    GeometricHeightEstimator,
    GeometricHeightEstimate,
    StreetViewFacadeFetcher,
    StreetViewImage,
)
from ..ingest.height_estimator_v3 import (
    OptimalHeightEstimator,
    HeightEstimateV3,
    LLMFacadeData,
    HeightReference,
    ERA_FLOOR_HEIGHTS,
    estimate_height_optimal,
)

# AI components
from ..ai.wwr_detector import WWRDetector
from ..ai.material_classifier import MaterialClassifier, FacadeMaterial
from ..ai.material_classifier_v2 import MaterialClassifierV2
from ..ai.facade_analyzer_llm import FacadeAnalyzerLLM, FacadeAnalysis
from ..ai.image_quality import ImageQualityAssessor
from ..ai.ground_floor_detector import GroundFloorDetector

# Convenience aliases
analyze_building = analyze_building_visually
analyze_address = analyze_address_visually
extract_footprint = extract_footprint_from_satellite


def quick_visual_scan(
    address: str = None,
    lat: float = None,
    lon: float = None,
) -> dict:
    """
    Quick visual scan of a building - returns essential metrics only.

    Returns dict with:
    - height_m: Building height in meters
    - floors: Floor count
    - material: Facade material
    - wwr: Average window-to-wall ratio
    - footprint_area_m2: Building footprint area
    - confidence: Overall confidence score

    Example:
        info = quick_visual_scan(address="Kungsgatan 1, Stockholm")
        print(f"{info['floors']} floors, {info['material']}, WWR={info['wwr']:.0%}")
    """
    result = {}

    # Get coordinates
    if address and not (lat and lon):
        try:
            from geopy.geocoders import Nominatim
            geocoder = Nominatim(user_agent="raiden_visual_quick")
            location = geocoder.geocode(address)
            if location:
                lat, lon = location.latitude, location.longitude
        except Exception:
            pass

    if not (lat and lon):
        return {"error": "Could not geocode address or no coordinates provided"}

    # Visual analysis
    try:
        analyzer = VisualAnalyzer()
        visual = analyzer.analyze_building(lat, lon)
        result["height_m"] = visual.height_m
        result["floors"] = visual.floor_count
        result["material"] = visual.facade_material
        result["wwr"] = visual.wwr_average
        result["building_form"] = visual.building_form
        result["estimated_era"] = visual.estimated_era
        result["confidence"] = visual.height_confidence
    except Exception as e:
        result["visual_error"] = str(e)

    # Footprint
    try:
        extractor = FootprintExtractor()
        footprint = extractor.extract_from_coordinates(lat, lon)
        if footprint:
            result["footprint_area_m2"] = footprint.area_m2
            result["footprint_geojson"] = footprint.geojson
    except Exception as e:
        result["footprint_error"] = str(e)

    result["lat"] = lat
    result["lon"] = lon

    return result


def get_building_geometry(
    address: str = None,
    lat: float = None,
    lon: float = None,
    addresses: list = None,  # For multi-building properties
) -> dict:
    """
    Get complete building geometry from visual sources.

    For multi-building properties, provide list of addresses to
    extract all building footprints using SAM point prompts.

    Returns dict with:
    - footprint_geojson: GeoJSON polygon
    - footprint_area_m2: Area in square meters
    - height_m: Building height
    - floors: Floor count
    - all_footprints: List if multi-building property

    Example:
        # Single building
        geom = get_building_geometry(address="Storgatan 5, Malmö")

        # Multi-building property
        geom = get_building_geometry(
            addresses=["Sjöstaden 2A", "Sjöstaden 2B", "Sjöstaden 4"],
        )
        print(f"Found {len(geom['all_footprints'])} buildings")
    """
    result = {}

    # Geocode if needed
    if address and not (lat and lon):
        try:
            from geopy.geocoders import Nominatim
            geocoder = Nominatim(user_agent="raiden_geometry")
            location = geocoder.geocode(address)
            if location:
                lat, lon = location.latitude, location.longitude
        except Exception:
            pass

    if not (lat and lon) and not addresses:
        return {"error": "Could not geocode or no coordinates/addresses provided"}

    # Footprint extraction
    extractor = FootprintExtractor()

    if addresses and len(addresses) > 1:
        # Multi-building extraction
        multi_result = extractor.extract_all_buildings(
            lat=lat, lon=lon,
            addresses=addresses,
        )
        if multi_result:
            result["all_footprints"] = [
                {
                    "geojson": fp.geojson,
                    "area_m2": fp.area_m2,
                    "center_lat": fp.center_lat,
                    "center_lon": fp.center_lon,
                }
                for fp in multi_result.footprints
            ]
            result["total_area_m2"] = multi_result.total_area_m2
            result["num_buildings"] = multi_result.num_buildings

            # Use largest for main footprint
            if multi_result.footprints:
                largest = max(multi_result.footprints, key=lambda f: f.area_m2)
                result["footprint_geojson"] = largest.geojson
                result["footprint_area_m2"] = largest.area_m2
    else:
        # Single building
        footprint = extractor.extract_from_coordinates(lat, lon) if lat and lon else None
        if not footprint and address:
            footprint = extractor.extract_from_address(address)

        if footprint:
            result["footprint_geojson"] = footprint.geojson
            result["footprint_area_m2"] = footprint.area_m2
            result["extraction_method"] = footprint.method
            result["confidence"] = footprint.confidence

    # Height from Street View
    if lat and lon:
        try:
            analyzer = VisualAnalyzer()
            visual = analyzer.analyze_building(lat, lon)
            result["height_m"] = visual.height_m
            result["floors"] = visual.floor_count
            result["height_confidence"] = visual.height_confidence
        except Exception:
            pass

    result["lat"] = lat
    result["lon"] = lon

    return result


def classify_facade(image_path: str) -> dict:
    """
    Classify facade from a single image file.

    Returns dict with:
    - material: Detected facade material
    - material_confidence: Confidence score
    - wwr: Window-to-wall ratio
    - wwr_confidence: Confidence score
    - floors: Visible floor count
    - building_form: Detected building form

    Example:
        result = classify_facade("./my_building.jpg")
        print(f"Material: {result['material']} ({result['material_confidence']:.0%})")
    """
    from PIL import Image

    result = {}
    img = Image.open(image_path)

    # Material classification
    try:
        classifier = MaterialClassifierV2()
        mat_result = classifier.classify_single_image(img)
        result["material"] = mat_result.material
        result["material_confidence"] = mat_result.confidence
    except Exception as e:
        result["material_error"] = str(e)

    # WWR detection
    try:
        detector = WWRDetector()
        wwr, conf = detector.calculate_wwr(img)
        result["wwr"] = wwr
        result["wwr_confidence"] = conf
    except Exception as e:
        result["wwr_error"] = str(e)

    # LLM analysis for more details
    try:
        llm = FacadeAnalyzerLLM(backend="gemini")
        facade = llm.analyze(image_path)
        if facade:
            result["floors"] = facade.visible_floors
            result["building_form"] = facade.building_form
            result["ground_floor_use"] = facade.ground_floor_use
            result["estimated_era"] = facade.estimated_era
    except Exception:
        pass

    return result


__all__ = [
    # Main analysis
    "VisualAnalyzer",
    "VisualAnalysisResult",
    "GroundFloorResult",
    "analyze_building",
    "analyze_address",
    "analyze_building_visually",
    "analyze_address_visually",
    # Footprint extraction
    "FootprintExtractor",
    "ExtractedFootprint",
    "MultiFootprintResult",
    "extract_footprint",
    "extract_footprint_from_satellite",
    # Satellite
    "SatelliteImage",
    "SatelliteFetcher",
    "EsriSatelliteFetcher",
    "fetch_esri_satellite",
    # Height estimation
    "GeometricHeightEstimator",
    "GeometricHeightEstimate",
    "StreetViewFacadeFetcher",
    "StreetViewImage",
    # V3 optimal height estimation
    "OptimalHeightEstimator",
    "HeightEstimateV3",
    "LLMFacadeData",
    "HeightReference",
    "ERA_FLOOR_HEIGHTS",
    "estimate_height_optimal",
    # AI components
    "WWRDetector",
    "MaterialClassifier",
    "MaterialClassifierV2",
    "FacadeMaterial",
    "FacadeAnalyzerLLM",
    "FacadeAnalysis",
    "ImageQualityAssessor",
    "GroundFloorDetector",
    # Convenience functions
    "quick_visual_scan",
    "get_building_geometry",
    "classify_facade",
]
