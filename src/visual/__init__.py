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
    brf_addresses: list = None,  # NEW: addresses from energy declaration
    city: str = None,            # NEW: city for address lookup
) -> dict:
    """
    Quick visual scan of a building - returns essential metrics only.

    NEW in v2.0: Uses OSM footprints by default (much more accurate).
    Satellite extraction only as fallback for new construction.

    Args:
        address: Single address to geocode
        lat, lon: Coordinates (optional if address provided)
        brf_addresses: List of street addresses from energy declaration
                       Used to correctly slice shared building complexes
        city: City name (improves address resolution)

    Returns dict with:
    - height_m: Building height in meters
    - floors: Floor count
    - material: Facade material
    - wwr: Average window-to-wall ratio
    - footprint_area_m2: Building footprint area
    - footprint_source: 'osm', 'microsoft', or 'satellite'
    - confidence: Overall confidence score

    Example:
        # Simple case
        info = quick_visual_scan(address="Kungsgatan 1, Stockholm")
        
        # Multi-building BRF with addresses from energy declaration
        info = quick_visual_scan(
            lat=59.37, lon=17.98,
            brf_addresses=["Filmgatan 1", "Filmgatan 3", "Filmgatan 5"],
            city="Solna"
        )
    """
    from ..geo.footprint_resolver import FootprintResolver
    
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

    # =========================================================================
    # STEP 1: Resolve footprint from OSM (NEW - much more accurate!)
    # =========================================================================
    try:
        resolver = FootprintResolver()
        
        if brf_addresses:
            # Use address-based slicing for multi-building BRFs
            footprints = resolver.resolve_with_address_slicing(
                lat, lon, 
                brf_addresses=brf_addresses,
                search_radius_m=100
            )
        else:
            footprints = resolver.resolve(lat, lon)
        
        if footprints and footprints[0].geometry:
            primary = footprints[0]
            result["footprint_geojson"] = primary.geometry
            result["footprint_source"] = primary.source
            result["footprint_osm_id"] = primary.osm_id
            result["footprint_address"] = primary.address
            
            # Calculate area from polygon
            if primary.geometry:
                result["footprint_area_m2"] = _calculate_polygon_area(primary.geometry)
            
            # Use OSM height/floors if available
            if primary.height_m:
                result["height_m"] = primary.height_m
            if primary.floors:
                result["floors"] = primary.floors
            
            result["footprint_confidence"] = primary.confidence
            
            # Multi-building info
            if len(footprints) > 1:
                result["is_multi_building"] = True
                result["buildings_count"] = len(footprints)
                result["all_footprints"] = [fp.to_dict() for fp in footprints]
    except Exception as e:
        result["footprint_resolver_error"] = str(e)

    # =========================================================================
    # STEP 2: Visual analysis from Street View (Raiden's core strength)
    # =========================================================================
    try:
        analyzer = VisualAnalyzer()
        visual = analyzer.analyze_building(lat, lon)
        result["material"] = visual.facade_material
        result["wwr"] = visual.wwr_average
        result["building_form"] = visual.building_form
        result["estimated_era"] = visual.estimated_era
        
        # Use visual floors/height if not from OSM
        if not result.get("floors"):
            result["floors"] = visual.floor_count
        if not result.get("height_m"):
            result["height_m"] = visual.height_m
            
        result["height_confidence"] = visual.height_confidence
        result["confidence"] = visual.height_confidence
    except Exception as e:
        result["visual_error"] = str(e)

    # =========================================================================
    # STEP 3: Satellite extraction (ONLY if no OSM footprint found)
    # =========================================================================
    if not result.get("footprint_geojson"):
        try:
            extractor = FootprintExtractor()
            footprint = extractor.extract_from_coordinates(lat, lon)
            if footprint:
                result["footprint_area_m2"] = footprint.area_m2
                result["footprint_geojson"] = footprint.geojson
                result["footprint_source"] = "satellite"
                result["footprint_confidence"] = 0.5  # Lower confidence
        except Exception as e:
            result["satellite_error"] = str(e)

    result["lat"] = lat
    result["lon"] = lon

    return result


def _calculate_polygon_area(geojson: dict) -> float:
    """Calculate area of a GeoJSON polygon in square meters."""
    try:
        coords = geojson.get("coordinates", [[]])[0]
        if len(coords) < 3:
            return 0.0
        
        # Shoelace formula with lat/lon to meters conversion
        # Approximate conversion at Swedish latitudes (~59°N)
        lat_to_m = 111320  # meters per degree latitude
        lon_to_m = 111320 * 0.515  # cos(59°) ≈ 0.515
        
        n = len(coords)
        area = 0.0
        for i in range(n):
            j = (i + 1) % n
            # coords are [lon, lat]
            x1 = coords[i][0] * lon_to_m
            y1 = coords[i][1] * lat_to_m
            x2 = coords[j][0] * lon_to_m
            y2 = coords[j][1] * lat_to_m
            area += x1 * y2
            area -= x2 * y1
        
        return abs(area) / 2.0
    except Exception:
        return 0.0


def get_building_geometry(
    address: str = None,
    lat: float = None,
    lon: float = None,
    addresses: list = None,  # For multi-building properties
    city: str = None,        # City for better address resolution
) -> dict:
    """
    Get complete building geometry from OSM + visual sources.

    NEW in v2.0: Uses OSM footprints by default (accurate, surveyed data).
    For multi-building properties, provide list of addresses from energy
    declaration to correctly slice shared complexes.

    Returns dict with:
    - footprint_geojson: GeoJSON polygon
    - footprint_area_m2: Area in square meters
    - footprint_source: 'osm', 'microsoft', or 'satellite'
    - height_m: Building height
    - floors: Floor count
    - all_footprints: List if multi-building property

    Example:
        # Single building
        geom = get_building_geometry(address="Storgatan 5, Malmö")

        # Multi-building BRF (addresses from energy declaration)
        geom = get_building_geometry(
            lat=59.37, lon=17.98,
            addresses=["Filmgatan 1", "Filmgatan 3", "Filmgatan 5"],
            city="Solna"
        )
        print(f"Found {len(geom['all_footprints'])} buildings")
    """
    from ..geo.footprint_resolver import FootprintResolver
    
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

    # =========================================================================
    # STEP 1: Resolve footprints from OSM (preferred)
    # =========================================================================
    resolver = FootprintResolver()
    osm_footprints = []
    
    try:
        if addresses and len(addresses) > 0:
            # Multi-building: use address-based slicing
            if lat and lon:
                osm_footprints = resolver.resolve_with_address_slicing(
                    lat, lon,
                    brf_addresses=addresses,
                    search_radius_m=100
                )
            else:
                # No coordinates - resolve by addresses directly
                osm_footprints = resolver.resolve_by_addresses(
                    addresses=addresses,
                    city=city or "Stockholm"
                )
        elif lat and lon:
            # Single building
            osm_footprints = resolver.resolve(lat, lon)
    except Exception as e:
        result["osm_error"] = str(e)
    
    # Process OSM results
    if osm_footprints:
        result["all_footprints"] = []
        total_area = 0.0
        
        for fp in osm_footprints:
            if fp.geometry:
                area = _calculate_polygon_area(fp.geometry)
                total_area += area
                result["all_footprints"].append({
                    "geojson": fp.geometry,
                    "area_m2": area,
                    "address": fp.address,
                    "osm_id": fp.osm_id,
                    "height_m": fp.height_m,
                    "floors": fp.floors,
                    "source": fp.source,
                    "confidence": fp.confidence,
                })
        
        result["num_buildings"] = len(result["all_footprints"])
        result["total_area_m2"] = total_area
        result["footprint_source"] = "osm"
        
        # Use primary (first/largest) for main footprint
        if result["all_footprints"]:
            primary = max(result["all_footprints"], key=lambda f: f.get("area_m2", 0))
            result["footprint_geojson"] = primary["geojson"]
            result["footprint_area_m2"] = primary["area_m2"]
            result["confidence"] = primary["confidence"]
            
            # Use OSM height/floors if available
            if primary.get("height_m"):
                result["height_m"] = primary["height_m"]
            if primary.get("floors"):
                result["floors"] = primary["floors"]

    # =========================================================================
    # STEP 2: Satellite fallback (only if no OSM data)
    # =========================================================================
    if not result.get("footprint_geojson") and lat and lon:
        try:
            extractor = FootprintExtractor()
            
            if addresses and len(addresses) > 1:
                multi_result = extractor.extract_all_buildings(
                    lat=lat, lon=lon,
                    addresses=addresses,
                )
                if multi_result:
                    result["all_footprints"] = [
                        {
                            "geojson": fp.geojson,
                            "area_m2": fp.area_m2,
                            "source": "satellite",
                        }
                        for fp in multi_result.footprints
                    ]
                    result["total_area_m2"] = multi_result.total_area_m2
                    result["num_buildings"] = multi_result.num_buildings
                    result["footprint_source"] = "satellite"

                    if multi_result.footprints:
                        largest = max(multi_result.footprints, key=lambda f: f.area_m2)
                        result["footprint_geojson"] = largest.geojson
                        result["footprint_area_m2"] = largest.area_m2
            else:
                footprint = extractor.extract_from_coordinates(lat, lon)
                if footprint:
                    result["footprint_geojson"] = footprint.geojson
                    result["footprint_area_m2"] = footprint.area_m2
                    result["footprint_source"] = "satellite"
                    result["confidence"] = footprint.confidence
        except Exception as e:
            result["satellite_error"] = str(e)

    # =========================================================================
    # STEP 3: Visual analysis (height, floors if not from OSM)
    # =========================================================================
    if lat and lon and not result.get("height_m"):
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
