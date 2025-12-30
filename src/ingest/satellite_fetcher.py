"""
Satellite Image Fetcher & Building Footprint Extractor

Fetches satellite/aerial imagery for:
- Roof analysis and verification
- Building footprint extraction (when not in Microsoft/OSM)
- PV potential assessment

Supports:
- Google Maps Static API (requires API key)
- ESRI World Imagery (FREE, no API key)
- SAM-based footprint extraction
- LLM-based footprint extraction (Gemini/Claude)
"""

import os
import json
import logging
import requests
import numpy as np
from io import BytesIO
from PIL import Image, ImageDraw
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict, Any
import math

from rich.console import Console

console = Console()
logger = logging.getLogger(__name__)


@dataclass
class SatelliteImage:
    """Satellite image with metadata."""
    image: Image.Image
    center_lat: float
    center_lon: float
    zoom: int
    size: Tuple[int, int]
    meters_per_pixel: float


class SatelliteFetcher:
    """
    Fetch satellite imagery from Google Maps Static API.

    Useful for:
    - Roof analysis and verification
    - Building footprint validation
    - PV potential assessment
    - Context around building
    """

    # Earth's circumference at equator
    EARTH_CIRCUMFERENCE_M = 40075016.686

    def __init__(self, api_key: str = None):
        """
        Initialize with Google API key.

        Args:
            api_key: Google Cloud API key with Maps Static API enabled
        """
        self.api_key = api_key or os.getenv("BRF_GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("Google API key required. Set BRF_GOOGLE_API_KEY env var.")

    def fetch_satellite_image(
        self,
        lat: float,
        lon: float,
        zoom: int = 19,
        size: Tuple[int, int] = (640, 640),
        scale: int = 2,
        format: str = "png",
    ) -> Optional[SatelliteImage]:
        """
        Fetch satellite image centered on coordinates.

        Args:
            lat, lon: Center coordinates
            zoom: Zoom level (1-21, 19-20 recommended for buildings)
            size: Image size in pixels (max 640x640)
            scale: Image scale (1 or 2, 2 for retina/higher resolution)
            format: Image format (png, jpg, gif)

        Returns:
            SatelliteImage with image and metadata
        """
        url = "https://maps.googleapis.com/maps/api/staticmap"
        params = {
            "center": f"{lat},{lon}",
            "zoom": zoom,
            "size": f"{size[0]}x{size[1]}",
            "scale": scale,
            "maptype": "satellite",
            "format": format,
            "key": self.api_key,
        }

        try:
            response = requests.get(url, params=params, timeout=30)
            if response.status_code == 200:
                img = Image.open(BytesIO(response.content))

                # Calculate meters per pixel
                mpp = self._meters_per_pixel(lat, zoom, scale)

                return SatelliteImage(
                    image=img,
                    center_lat=lat,
                    center_lon=lon,
                    zoom=zoom,
                    size=(img.width, img.height),
                    meters_per_pixel=mpp,
                )
            else:
                console.print(f"[red]Satellite fetch failed: {response.status_code}[/red]")
                return None
        except Exception as e:
            console.print(f"[red]Satellite fetch error: {e}[/red]")
            return None

    def fetch_building_aerial(
        self,
        footprint: dict,
        padding_m: float = 20,
        min_zoom: int = 18,
        max_zoom: int = 20,
    ) -> Optional[SatelliteImage]:
        """
        Fetch aerial image of a building with optimal zoom.

        Automatically calculates zoom level to fit the building
        with specified padding.

        Args:
            footprint: GeoJSON geometry of building
            padding_m: Padding around building in meters
            min_zoom: Minimum zoom level
            max_zoom: Maximum zoom level

        Returns:
            SatelliteImage covering the building
        """
        # Parse footprint
        coords = self._parse_footprint(footprint)
        if not coords:
            return None

        # Calculate bounding box
        lons = [c[0] for c in coords]
        lats = [c[1] for c in coords]
        min_lon, max_lon = min(lons), max(lons)
        min_lat, max_lat = min(lats), max(lats)

        # Center point
        center_lat = (min_lat + max_lat) / 2
        center_lon = (min_lon + max_lon) / 2

        # Calculate building dimensions in meters
        width_m = self._haversine_distance(center_lat, min_lon, center_lat, max_lon)
        height_m = self._haversine_distance(min_lat, center_lon, max_lat, center_lon)

        # Add padding
        required_width_m = width_m + 2 * padding_m
        required_height_m = height_m + 2 * padding_m
        required_m = max(required_width_m, required_height_m)

        # Find optimal zoom level
        for zoom in range(max_zoom, min_zoom - 1, -1):
            mpp = self._meters_per_pixel(center_lat, zoom, scale=2)
            coverage_m = 640 * mpp  # Image covers 640 pixels
            if coverage_m >= required_m:
                break

        console.print(f"[dim]Satellite: zoom={zoom}, coverage={coverage_m:.0f}m, building={max(width_m, height_m):.0f}m[/dim]")

        return self.fetch_satellite_image(center_lat, center_lon, zoom=zoom)

    def fetch_multi_scale(
        self,
        lat: float,
        lon: float,
        zooms: List[int] = [17, 19, 20],
    ) -> List[SatelliteImage]:
        """
        Fetch satellite images at multiple zoom levels.

        Useful for context at different scales.

        Args:
            lat, lon: Center coordinates
            zooms: List of zoom levels

        Returns:
            List of SatelliteImages
        """
        images = []
        for zoom in zooms:
            img = self.fetch_satellite_image(lat, lon, zoom=zoom)
            if img:
                images.append(img)
        return images

    def _parse_footprint(self, footprint: dict) -> List[Tuple[float, float]]:
        """Parse GeoJSON to coordinate list."""
        import json
        if isinstance(footprint, str):
            footprint = json.loads(footprint)

        if footprint.get('type') == 'Feature':
            footprint = footprint.get('geometry', {})

        geom_type = footprint.get('type')
        coords = footprint.get('coordinates', [])

        if geom_type == 'Polygon':
            if coords and len(coords) > 0:
                return [(c[0], c[1]) for c in coords[0]]
        elif geom_type == 'MultiPolygon':
            if coords and len(coords) > 0 and len(coords[0]) > 0:
                return [(c[0], c[1]) for c in coords[0][0]]

        return []

    def _meters_per_pixel(self, lat: float, zoom: int, scale: int = 1) -> float:
        """Calculate meters per pixel at given latitude and zoom."""
        # At zoom 0, entire world (circumference) fits in 256 pixels
        # Each zoom level doubles the resolution
        return (self.EARTH_CIRCUMFERENCE_M * math.cos(math.radians(lat))) / (256 * scale * (2 ** zoom))

    def _haversine_distance(
        self,
        lat1: float,
        lon1: float,
        lat2: float,
        lon2: float,
    ) -> float:
        """Calculate distance between two points in meters."""
        R = 6371000  # Earth radius in meters

        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        delta_lat = math.radians(lat2 - lat1)
        delta_lon = math.radians(lon2 - lon1)

        a = (math.sin(delta_lat / 2) ** 2 +
             math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(delta_lon / 2) ** 2)
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

        return R * c


def fetch_building_satellite(
    footprint: dict,
    api_key: str = None,
) -> Optional[SatelliteImage]:
    """
    Convenience function to fetch satellite image of a building.

    Args:
        footprint: GeoJSON geometry
        api_key: Google API key (optional, uses env var)

    Returns:
        SatelliteImage or None
    """
    fetcher = SatelliteFetcher(api_key)
    return fetcher.fetch_building_aerial(footprint)


class EsriSatelliteFetcher:
    """
    Fetch satellite imagery from Esri World Imagery.

    Free, no API key required. Good global coverage including Sweden.
    Resolution varies by location (typically 0.3-1m in urban areas).
    """

    TILE_URL = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
    TILE_SIZE = 256
    EARTH_CIRCUMFERENCE_M = 40075016.686

    def fetch_building_aerial(
        self,
        footprint: dict,
        padding_m: float = 20,
        zoom: int = 19,
    ) -> Optional[SatelliteImage]:
        """
        Fetch aerial image covering a building by stitching tiles.

        Args:
            footprint: GeoJSON geometry
            padding_m: Padding around building
            zoom: Zoom level (17-19 recommended)

        Returns:
            SatelliteImage covering the building
        """
        # Parse footprint
        coords = self._parse_footprint(footprint)
        if not coords:
            return None

        # Calculate bounding box
        lons = [c[0] for c in coords]
        lats = [c[1] for c in coords]

        center_lat = (min(lats) + max(lats)) / 2
        center_lon = (min(lons) + max(lons)) / 2

        # Add padding in degrees (rough approximation)
        padding_deg = padding_m / 111000
        min_lat, max_lat = min(lats) - padding_deg, max(lats) + padding_deg
        min_lon, max_lon = min(lons) - padding_deg, max(lons) + padding_deg

        # Get tile coordinates for corners
        min_tile_x, max_tile_y = self._latlon_to_tile(min_lat, min_lon, zoom)
        max_tile_x, min_tile_y = self._latlon_to_tile(max_lat, max_lon, zoom)

        # Ensure we have at least a 2x2 tile grid
        if max_tile_x == min_tile_x:
            max_tile_x += 1
        if max_tile_y == min_tile_y:
            max_tile_y += 1

        # Fetch and stitch tiles
        tiles_x = max_tile_x - min_tile_x + 1
        tiles_y = max_tile_y - min_tile_y + 1

        console.print(f"[dim]Esri: fetching {tiles_x}x{tiles_y} tiles at zoom {zoom}[/dim]")

        # Create composite image
        composite = Image.new('RGB', (tiles_x * self.TILE_SIZE, tiles_y * self.TILE_SIZE))

        for ty in range(min_tile_y, max_tile_y + 1):
            for tx in range(min_tile_x, max_tile_x + 1):
                tile = self._fetch_tile(zoom, tx, ty)
                if tile:
                    px = (tx - min_tile_x) * self.TILE_SIZE
                    py = (ty - min_tile_y) * self.TILE_SIZE
                    composite.paste(tile, (px, py))

        # Calculate resolution
        mpp = self._meters_per_pixel(center_lat, zoom)

        return SatelliteImage(
            image=composite,
            center_lat=center_lat,
            center_lon=center_lon,
            zoom=zoom,
            size=(composite.width, composite.height),
            meters_per_pixel=mpp,
        )

    def fetch_at_location(
        self,
        lat: float,
        lon: float,
        zoom: int = 19,
        tiles: int = 2,
    ) -> Optional[SatelliteImage]:
        """
        Fetch aerial image centered on location.

        Args:
            lat, lon: Center coordinates
            zoom: Zoom level
            tiles: Number of tiles in each direction from center

        Returns:
            SatelliteImage
        """
        center_x, center_y = self._latlon_to_tile(lat, lon, zoom)

        min_x = center_x - tiles
        max_x = center_x + tiles
        min_y = center_y - tiles
        max_y = center_y + tiles

        tiles_x = max_x - min_x + 1
        tiles_y = max_y - min_y + 1

        console.print(f"[dim]Esri: fetching {tiles_x}x{tiles_y} tiles at zoom {zoom}[/dim]")

        composite = Image.new('RGB', (tiles_x * self.TILE_SIZE, tiles_y * self.TILE_SIZE))

        for ty in range(min_y, max_y + 1):
            for tx in range(min_x, max_x + 1):
                tile = self._fetch_tile(zoom, tx, ty)
                if tile:
                    px = (tx - min_x) * self.TILE_SIZE
                    py = (ty - min_y) * self.TILE_SIZE
                    composite.paste(tile, (px, py))

        mpp = self._meters_per_pixel(lat, zoom)

        return SatelliteImage(
            image=composite,
            center_lat=lat,
            center_lon=lon,
            zoom=zoom,
            size=(composite.width, composite.height),
            meters_per_pixel=mpp,
        )

    def _fetch_tile(self, z: int, x: int, y: int) -> Optional[Image.Image]:
        """Fetch a single tile."""
        url = self.TILE_URL.format(z=z, x=x, y=y)
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                return Image.open(BytesIO(response.content))
        except Exception:
            pass
        return None

    def _latlon_to_tile(self, lat: float, lon: float, zoom: int) -> Tuple[int, int]:
        """Convert lat/lon to tile coordinates."""
        lat_rad = math.radians(lat)
        n = 2.0 ** zoom
        x = int((lon + 180.0) / 360.0 * n)
        y = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
        return x, y

    def _meters_per_pixel(self, lat: float, zoom: int) -> float:
        """Calculate meters per pixel."""
        return (self.EARTH_CIRCUMFERENCE_M * math.cos(math.radians(lat))) / (self.TILE_SIZE * (2 ** zoom))

    def _parse_footprint(self, footprint: dict) -> List[Tuple[float, float]]:
        """Parse GeoJSON to coordinate list."""
        import json
        if isinstance(footprint, str):
            footprint = json.loads(footprint)

        if footprint.get('type') == 'Feature':
            footprint = footprint.get('geometry', {})

        geom_type = footprint.get('type')
        coords = footprint.get('coordinates', [])

        if geom_type == 'Polygon':
            if coords and len(coords) > 0:
                return [(c[0], c[1]) for c in coords[0]]
        elif geom_type == 'MultiPolygon':
            if coords and len(coords) > 0 and len(coords[0]) > 0:
                return [(c[0], c[1]) for c in coords[0][0]]

        return []


def fetch_esri_satellite(
    footprint: dict = None,
    lat: float = None,
    lon: float = None,
    zoom: int = 19,
) -> Optional[SatelliteImage]:
    """
    Convenience function for Esri satellite imagery.

    Args:
        footprint: GeoJSON geometry (optional)
        lat, lon: Center coordinates (if no footprint)
        zoom: Zoom level

    Returns:
        SatelliteImage
    """
    fetcher = EsriSatelliteFetcher()
    if footprint:
        return fetcher.fetch_building_aerial(footprint, zoom=zoom)
    elif lat and lon:
        return fetcher.fetch_at_location(lat, lon, zoom=zoom)
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# BUILDING FOOTPRINT EXTRACTOR
# Extract building footprints from satellite imagery when not available in
# Microsoft Building Footprints or OpenStreetMap
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ExtractedFootprint:
    """Result of footprint extraction from satellite imagery."""
    geojson: Dict[str, Any]  # GeoJSON Polygon
    coordinates: List[Tuple[float, float]]  # [(lon, lat), ...]
    area_m2: float
    confidence: float
    method: str  # "sam", "llm", "edge_detection"
    center_lat: float
    center_lon: float
    bbox: Tuple[float, float, float, float]  # (min_lon, min_lat, max_lon, max_lat)
    notes: List[str] = field(default_factory=list)


@dataclass
class MultiFootprintResult:
    """Result when extracting multiple buildings from one property."""
    footprints: List[ExtractedFootprint]
    total_area_m2: float
    num_buildings: int
    center_lat: float
    center_lon: float
    method: str
    notes: List[str] = field(default_factory=list)


class FootprintExtractor:
    """
    Extract building footprints from satellite imagery.

    When Microsoft Building Footprints or OSM don't have coverage,
    we can extract footprints from satellite images using:
    1. SAM (Segment Anything Model) - best accuracy
    2. Vision LLM (Gemini/Claude) - good fallback
    3. Edge detection - fast but lower accuracy

    IMPORTANT: One property/address can have MULTIPLE buildings!
    Use extract_all_buildings() to get all footprints for a property.

    Usage:
        extractor = FootprintExtractor()

        # Single building (centered on coordinates)
        footprint = extractor.extract_from_coordinates(59.30, 18.10)

        # ALL buildings at an address (recommended for BRFs!)
        result = extractor.extract_all_buildings(
            address="Bellmansgatan 16, Stockholm",
            expected_buildings=3,  # Hint from energy declaration
        )
        print(f"Found {result.num_buildings} buildings, total {result.total_area_m2:.0f} m²")
        for fp in result.footprints:
            print(f"  Building: {fp.area_m2:.0f} m²")
    """

    def __init__(
        self,
        use_sam: bool = True,
        use_llm: bool = True,
        sam_checkpoint: str = None,
    ):
        """
        Initialize footprint extractor.

        Args:
            use_sam: Use SAM for segmentation (requires torch)
            use_llm: Use vision LLM as fallback
            sam_checkpoint: Path to SAM checkpoint (optional)
        """
        self.satellite_fetcher = EsriSatelliteFetcher()
        self.use_sam = use_sam
        self.use_llm = use_llm
        self.sam_checkpoint = sam_checkpoint

        # SAM model (loaded lazily)
        self._sam_predictor = None
        self._sam_available = None

    def extract_from_coordinates(
        self,
        lat: float,
        lon: float,
        zoom: int = 19,
        method: str = "auto",
    ) -> Optional[ExtractedFootprint]:
        """
        Extract building footprint at given coordinates.

        Args:
            lat, lon: Building center coordinates (WGS84)
            zoom: Satellite image zoom level (19 recommended)
            method: "auto", "sam", "llm", or "edge"

        Returns:
            ExtractedFootprint with GeoJSON polygon
        """
        console.print(f"[cyan]Extracting footprint at ({lat:.6f}, {lon:.6f})...[/cyan]")

        # Fetch satellite image
        sat_image = self.satellite_fetcher.fetch_at_location(lat, lon, zoom=zoom, tiles=1)
        if not sat_image:
            logger.error("Failed to fetch satellite image")
            return None

        # Choose extraction method
        if method == "auto":
            # Try SAM first, then LLM, then edge detection
            result = self._extract_with_sam(sat_image, lat, lon)
            if not result and self.use_llm:
                result = self._extract_with_llm(sat_image, lat, lon)
            if not result:
                result = self._extract_with_edges(sat_image, lat, lon)
        elif method == "sam":
            result = self._extract_with_sam(sat_image, lat, lon)
        elif method == "llm":
            result = self._extract_with_llm(sat_image, lat, lon)
        elif method == "edge":
            result = self._extract_with_edges(sat_image, lat, lon)
        else:
            result = None

        return result

    def extract_from_address(
        self,
        address: str,
        zoom: int = 19,
        method: str = "auto",
    ) -> Optional[ExtractedFootprint]:
        """
        Extract building footprint by address (geocodes automatically).

        Args:
            address: Street address (e.g., "Bellmansgatan 16, Stockholm")
            zoom: Satellite image zoom level
            method: Extraction method

        Returns:
            ExtractedFootprint or None
        """
        try:
            from geopy.geocoders import Nominatim

            geocoder = Nominatim(user_agent="raiden_footprint_extractor")
            location = geocoder.geocode(address)

            if not location:
                logger.error(f"Could not geocode address: {address}")
                return None

            console.print(f"[cyan]Geocoded: {address} → ({location.latitude:.6f}, {location.longitude:.6f})[/cyan]")

            return self.extract_from_coordinates(
                location.latitude,
                location.longitude,
                zoom=zoom,
                method=method,
            )

        except ImportError:
            logger.error("geopy not installed. Run: pip install geopy")
            return None

    def extract_all_buildings(
        self,
        lat: float = None,
        lon: float = None,
        address: str = None,
        addresses: List[str] = None,  # Multiple addresses = multiple buildings!
        expected_buildings: int = None,
        search_radius_m: float = 100,
        min_building_area_m2: float = 50,
        zoom: int = 18,  # Lower zoom to see more area
    ) -> Optional[MultiFootprintResult]:
        """
        Extract ALL building footprints in a property area.

        One address/property can have multiple buildings (common for BRFs).
        This method finds all buildings within the search radius.

        BEST APPROACH: Provide multiple addresses from energy declaration!
        Each address = one entrance = one building location.
        These are geocoded and used as SAM point prompts.

        Args:
            lat, lon: Property center coordinates (fallback)
            address: Single address (fallback)
            addresses: LIST of addresses from Gripen (RECOMMENDED!)
                       e.g., ["Bellmansgatan 16A", "Bellmansgatan 16B", "Bellmansgatan 18"]
            expected_buildings: Hint from energy declaration
            search_radius_m: Search radius in meters
            min_building_area_m2: Minimum building area
            zoom: Satellite zoom level

        Returns:
            MultiFootprintResult with all buildings found

        Example with Gripen:
            from src.ingest import load_gripen, FootprintExtractor

            gripen = load_gripen()
            building = gripen.find_by_address("Sjöstaden")[0]

            # Get ALL addresses for this property
            all_addresses = gripen.get_all_addresses_for_property(
                building.property_designation
            )
            # → ["Sjöstaden 2A", "Sjöstaden 2B", "Sjöstaden 4A", ...]

            extractor = FootprintExtractor()
            result = extractor.extract_all_buildings(addresses=all_addresses)
        """
        # ═══════════════════════════════════════════════════════════════════════
        # GEOCODE ALL ADDRESSES (the smart approach!)
        # Each address from energy declaration = one building location
        # ═══════════════════════════════════════════════════════════════════════
        building_points: List[Tuple[float, float]] = []  # (lat, lon) for each building

        try:
            from geopy.geocoders import Nominatim
            geocoder = Nominatim(user_agent="raiden_footprint_extractor")

            if addresses and len(addresses) > 0:
                # Multiple addresses provided - geocode all of them!
                console.print(f"[cyan]Geocoding {len(addresses)} addresses...[/cyan]")

                for addr in addresses:
                    try:
                        location = geocoder.geocode(addr)
                        if location:
                            building_points.append((location.latitude, location.longitude))
                            console.print(f"  [dim]✓ {addr} → ({location.latitude:.6f}, {location.longitude:.6f})[/dim]")
                        else:
                            console.print(f"  [yellow]✗ Could not geocode: {addr}[/yellow]")
                    except Exception as e:
                        console.print(f"  [yellow]✗ Geocoding error for {addr}: {e}[/yellow]")

                if building_points:
                    # Calculate center from all points
                    lat = sum(p[0] for p in building_points) / len(building_points)
                    lon = sum(p[1] for p in building_points) / len(building_points)
                    console.print(f"[green]Geocoded {len(building_points)}/{len(addresses)} addresses[/green]")

                    # Update expected buildings if not provided
                    if not expected_buildings:
                        expected_buildings = len(building_points)

            elif address and not (lat and lon):
                # Single address fallback
                location = geocoder.geocode(address)
                if location:
                    lat, lon = location.latitude, location.longitude
                    building_points.append((lat, lon))
                    console.print(f"[cyan]Geocoded: {address} → ({lat:.6f}, {lon:.6f})[/cyan]")
                else:
                    logger.error(f"Could not geocode: {address}")
                    return None

        except ImportError:
            logger.error("geopy not installed. Run: pip install geopy")
            return None

        if not (lat and lon):
            logger.error("Must provide lat/lon, address, or addresses list")
            return None

        console.print(f"[cyan]Extracting buildings (center: {lat:.6f}, {lon:.6f})...[/cyan]")
        if expected_buildings:
            console.print(f"[dim]Expected buildings: {expected_buildings}[/dim]")
        if building_points:
            console.print(f"[dim]Known building locations: {len(building_points)} points[/dim]")

        # Calculate search radius from building points if we have multiple
        if len(building_points) > 1:
            # Expand radius to cover all buildings
            max_dist = 0
            for p in building_points:
                dist = self._haversine_distance(lat, lon, p[0], p[1])
                max_dist = max(max_dist, dist)
            search_radius_m = max(search_radius_m, max_dist + 30)  # Add padding

        # Fetch satellite image
        tiles = max(2, int(search_radius_m / 50))
        sat_image = self.satellite_fetcher.fetch_at_location(lat, lon, zoom=zoom, tiles=tiles)
        if not sat_image:
            logger.error("Failed to fetch satellite image")
            return None

        # ═══════════════════════════════════════════════════════════════════════
        # EXTRACTION PRIORITY:
        # 1. SAM with point prompts (if we have geocoded building locations)
        # 2. SAM automatic (find all buildings)
        # 3. LLM with building count hint
        # ═══════════════════════════════════════════════════════════════════════
        footprints = []

        if building_points and len(building_points) > 0:
            # Use geocoded points as SAM prompts (BEST approach!)
            footprints = self._extract_with_point_prompts(
                sat_image, lat, lon, building_points, min_building_area_m2
            )

        if not footprints:
            # Fall back to automatic SAM
            footprints = self._extract_multiple_with_sam(sat_image, lat, lon, min_building_area_m2)

        if not footprints and self.use_llm:
            footprints = self._extract_multiple_with_llm(
                sat_image, lat, lon, expected_buildings, min_building_area_m2
            )

        if not footprints:
            # Fall back to single building extraction
            single = self._extract_with_edges(sat_image, lat, lon)
            if single:
                footprints = [single]

        if not footprints:
            return None

        total_area = sum(fp.area_m2 for fp in footprints)

        notes = [f"Found {len(footprints)} buildings"]
        if expected_buildings:
            if len(footprints) == expected_buildings:
                notes.append(f"✓ Matches expected count ({expected_buildings})")
            else:
                notes.append(f"⚠ Expected {expected_buildings}, found {len(footprints)}")

        console.print(f"[green]Found {len(footprints)} buildings, total area: {total_area:.0f} m²[/green]")

        return MultiFootprintResult(
            footprints=footprints,
            total_area_m2=total_area,
            num_buildings=len(footprints),
            center_lat=lat,
            center_lon=lon,
            method=footprints[0].method if footprints else "none",
            notes=notes,
        )

    def _haversine_distance(
        self, lat1: float, lon1: float, lat2: float, lon2: float
    ) -> float:
        """Calculate distance between two points in meters."""
        R = 6371000  # Earth radius in meters
        phi1 = math.radians(lat1)
        phi2 = math.radians(lat2)
        delta_phi = math.radians(lat2 - lat1)
        delta_lambda = math.radians(lon2 - lon1)

        a = math.sin(delta_phi / 2) ** 2 + \
            math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

        return R * c

    def _extract_with_point_prompts(
        self,
        sat_image: SatelliteImage,
        center_lat: float,
        center_lon: float,
        building_points: List[Tuple[float, float]],  # (lat, lon) for each building
        min_area_m2: float = 50,
    ) -> List[ExtractedFootprint]:
        """
        Extract building footprints using geocoded addresses as SAM prompts.

        This is the SMARTEST approach:
        - Each address from energy declaration → geocoded point
        - Each point → SAM prompt → precise building segmentation
        - No guessing needed, we know exactly where each building is!

        Args:
            sat_image: Satellite image covering the property
            center_lat, center_lon: Image center
            building_points: List of (lat, lon) for each building
            min_area_m2: Minimum building area

        Returns:
            List of ExtractedFootprint, one per building
        """
        if not self._check_sam_available():
            return []

        try:
            from segment_anything import sam_model_registry, SamPredictor

            console.print(f"[cyan]Running SAM with {len(building_points)} point prompts...[/cyan]")

            # Load SAM
            if self._sam_predictor is None:
                checkpoint = self.sam_checkpoint or os.path.expanduser(
                    "~/.cache/sam/sam_vit_b_01ec64.pth"
                )
                if not os.path.exists(checkpoint):
                    self._download_sam_checkpoint(checkpoint)
                sam = sam_model_registry["vit_b"](checkpoint=checkpoint)
                self._sam_predictor = SamPredictor(sam)

            # Set image
            image_np = np.array(sat_image.image)
            self._sam_predictor.set_image(image_np)

            # Convert lat/lon points to pixel coordinates
            mpp = sat_image.meters_per_pixel
            img_h, img_w = image_np.shape[:2]
            cx, cy = img_w / 2, img_h / 2

            meters_per_deg_lat = 111320
            meters_per_deg_lon = 111320 * math.cos(math.radians(center_lat))

            footprints = []

            for i, (blat, blon) in enumerate(building_points):
                # Convert lat/lon to pixel position
                dx_m = (blon - center_lon) * meters_per_deg_lon
                dy_m = (blat - center_lat) * meters_per_deg_lat

                px = int(cx + dx_m / mpp)
                py = int(cy - dy_m / mpp)  # Y is inverted

                # Check if point is within image
                if not (0 <= px < img_w and 0 <= py < img_h):
                    console.print(f"  [yellow]Point {i+1} outside image bounds[/yellow]")
                    continue

                # Use this point as SAM prompt
                point_coords = np.array([[px, py]])
                point_labels = np.array([1])  # Foreground

                masks, scores, _ = self._sam_predictor.predict(
                    point_coords=point_coords,
                    point_labels=point_labels,
                    multimask_output=True,
                )

                # Use best mask
                best_idx = np.argmax(scores)
                mask = masks[best_idx]
                confidence = float(scores[best_idx])

                # Convert to polygon
                coords = self._mask_to_polygon(mask, sat_image, center_lat, center_lon)
                if len(coords) < 4:
                    console.print(f"  [yellow]Point {i+1}: invalid polygon[/yellow]")
                    continue

                # Calculate area
                area_m2 = self._calculate_polygon_area(coords, center_lat)

                if area_m2 < min_area_m2:
                    console.print(f"  [yellow]Point {i+1}: area too small ({area_m2:.0f} m²)[/yellow]")
                    continue

                geojson = {
                    "type": "Polygon",
                    "coordinates": [coords + [coords[0]]]
                }

                footprints.append(ExtractedFootprint(
                    geojson=geojson,
                    coordinates=coords,
                    area_m2=area_m2,
                    confidence=confidence,
                    method="sam_point_prompt",
                    center_lat=blat,
                    center_lon=blon,
                    bbox=self._coords_to_bbox(coords),
                    notes=[f"Building {i+1}", f"SAM score: {confidence:.2f}"],
                ))

                console.print(f"  [green]Building {i+1}: {area_m2:.0f} m² (conf={confidence:.0%})[/green]")

            console.print(f"[green]Extracted {len(footprints)}/{len(building_points)} buildings using point prompts[/green]")
            return footprints

        except Exception as e:
            logger.warning(f"SAM point prompt extraction failed: {e}")
            return []

    def _extract_multiple_with_sam(
        self,
        sat_image: SatelliteImage,
        center_lat: float,
        center_lon: float,
        min_area_m2: float = 50,
    ) -> List[ExtractedFootprint]:
        """Extract multiple buildings using SAM automatic mask generation."""
        if not self._check_sam_available():
            return []

        try:
            from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

            console.print("[cyan]Running SAM automatic segmentation...[/cyan]")

            # Load SAM model
            if self._sam_predictor is None:
                checkpoint = self.sam_checkpoint or os.path.expanduser(
                    "~/.cache/sam/sam_vit_b_01ec64.pth"
                )
                if not os.path.exists(checkpoint):
                    self._download_sam_checkpoint(checkpoint)
                sam = sam_model_registry["vit_b"](checkpoint=checkpoint)
            else:
                sam = self._sam_predictor.model

            # Use automatic mask generator (finds all objects)
            mask_generator = SamAutomaticMaskGenerator(
                sam,
                points_per_side=16,  # Grid of prompt points
                pred_iou_thresh=0.86,  # Quality threshold
                stability_score_thresh=0.92,
                min_mask_region_area=500,  # Min pixels
            )

            image_np = np.array(sat_image.image)
            masks = mask_generator.generate(image_np)

            console.print(f"[dim]SAM found {len(masks)} segments[/dim]")

            # Filter and convert to footprints
            footprints = []
            mpp = sat_image.meters_per_pixel

            for mask_data in masks:
                mask = mask_data['segmentation']
                area_pixels = mask_data['area']
                area_m2 = area_pixels * (mpp ** 2)

                # Filter by size (buildings typically 50-5000 m²)
                if area_m2 < min_area_m2 or area_m2 > 10000:
                    continue

                # Filter by shape (buildings are more compact than roads/parking)
                bbox = mask_data['bbox']  # x, y, w, h
                aspect_ratio = max(bbox[2], bbox[3]) / max(1, min(bbox[2], bbox[3]))
                if aspect_ratio > 8:  # Too elongated (likely road)
                    continue

                # Convert mask to polygon
                coords = self._mask_to_polygon(mask, sat_image, center_lat, center_lon)
                if len(coords) < 4:
                    continue

                geojson = {
                    "type": "Polygon",
                    "coordinates": [coords + [coords[0]]]
                }

                footprints.append(ExtractedFootprint(
                    geojson=geojson,
                    coordinates=coords,
                    area_m2=area_m2,
                    confidence=mask_data.get('stability_score', 0.8),
                    method="sam_auto",
                    center_lat=center_lat,
                    center_lon=center_lon,
                    bbox=self._coords_to_bbox(coords),
                    notes=[f"SAM IoU: {mask_data.get('predicted_iou', 0):.2f}"],
                ))

            console.print(f"[green]SAM extracted {len(footprints)} building footprints[/green]")
            return footprints

        except Exception as e:
            logger.warning(f"SAM multi-building extraction failed: {e}")
            return []

    def _extract_multiple_with_llm(
        self,
        sat_image: SatelliteImage,
        center_lat: float,
        center_lon: float,
        expected_buildings: int = None,
        min_area_m2: float = 50,
    ) -> List[ExtractedFootprint]:
        """Extract multiple buildings using Komilion/OpenRouter vision LLM."""
        from ..ai.llm_client import LLMClient
        import tempfile

        # Check for Komilion API key
        if not os.environ.get("KOMILION_API_KEY"):
            logger.warning("KOMILION_API_KEY not set for footprint extraction")
            return []

        try:
            console.print("[cyan]Running LLM multi-building extraction...[/cyan]")

            # Save image to temp file for LLM client
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                sat_image.image.save(tmp.name, format="PNG")
                temp_path = tmp.name

            mpp = sat_image.meters_per_pixel
            img_w, img_h = sat_image.image.width, sat_image.image.height

            expected_hint = ""
            if expected_buildings:
                expected_hint = f"\nHint: The property is expected to have approximately {expected_buildings} buildings."

            prompt = f"""Analyze this satellite image and identify ALL building footprints.

Image: {img_w}x{img_h} pixels, each pixel ≈ {mpp:.2f} meters.
Center coordinates: ({center_lat:.6f}, {center_lon:.6f}).{expected_hint}

Return a JSON object with ALL buildings found:
{{
    "buildings": [
        {{
            "polygon_pixels": [[x1, y1], [x2, y2], ...],
            "building_type": "residential/commercial/industrial",
            "estimated_floors": 1-10,
            "confidence": 0.0-1.0
        }},
        ...
    ],
    "total_buildings": N,
    "notes": "observations about the property"
}}

Trace each building's roof outline as a polygon. Include ALL separate buildings.
Return ONLY the JSON."""

            client = LLMClient(mode="balanced")
            response = client.analyze_image(image_path=temp_path, prompt=prompt)

            # Clean up temp file
            os.unlink(temp_path)

            if not response:
                return []

            response_text = response.content.strip()
            if response_text.startswith("```"):
                response_text = response_text.split("```")[1]
                if response_text.startswith("json"):
                    response_text = response_text[4:]

            result = json.loads(response_text)
            buildings_data = result.get("buildings", [])

            console.print(f"[dim]LLM found {len(buildings_data)} buildings[/dim]")

            footprints = []
            for i, bldg in enumerate(buildings_data):
                pixel_coords = bldg.get("polygon_pixels", [])
                if len(pixel_coords) < 3:
                    continue

                coords = self._pixels_to_latlon(pixel_coords, sat_image, center_lat, center_lon)
                area_m2 = self._calculate_polygon_area(coords, center_lat)

                if area_m2 < min_area_m2:
                    continue

                geojson = {
                    "type": "Polygon",
                    "coordinates": [coords + [coords[0]]]
                }

                footprints.append(ExtractedFootprint(
                    geojson=geojson,
                    coordinates=coords,
                    area_m2=area_m2,
                    confidence=bldg.get("confidence", 0.7),
                    method="llm_multi",
                    center_lat=center_lat,
                    center_lon=center_lon,
                    bbox=self._coords_to_bbox(coords),
                    notes=[
                        f"Building {i+1}",
                        f"Type: {bldg.get('building_type', 'unknown')}",
                    ],
                ))

            console.print(f"[green]LLM extracted {len(footprints)} building footprints[/green]")
            return footprints

        except Exception as e:
            logger.warning(f"LLM multi-building extraction failed: {e}")
            return []

    def _extract_with_sam(
        self,
        sat_image: SatelliteImage,
        lat: float,
        lon: float,
    ) -> Optional[ExtractedFootprint]:
        """Extract footprint using Segment Anything Model."""
        if not self._check_sam_available():
            return None

        try:
            from segment_anything import sam_model_registry, SamPredictor

            console.print("[cyan]Running SAM segmentation...[/cyan]")

            # Load SAM model if not already loaded
            if self._sam_predictor is None:
                checkpoint = self.sam_checkpoint or os.path.expanduser(
                    "~/.cache/sam/sam_vit_b_01ec64.pth"
                )
                if not os.path.exists(checkpoint):
                    console.print("[yellow]SAM checkpoint not found, downloading...[/yellow]")
                    self._download_sam_checkpoint(checkpoint)

                sam = sam_model_registry["vit_b"](checkpoint=checkpoint)
                self._sam_predictor = SamPredictor(sam)

            # Convert PIL to numpy
            image_np = np.array(sat_image.image)

            # Set image
            self._sam_predictor.set_image(image_np)

            # Use center point as prompt
            h, w = image_np.shape[:2]
            center_point = np.array([[w // 2, h // 2]])
            center_label = np.array([1])  # 1 = foreground

            # Predict mask
            masks, scores, _ = self._sam_predictor.predict(
                point_coords=center_point,
                point_labels=center_label,
                multimask_output=True,
            )

            # Use highest-scoring mask
            best_idx = np.argmax(scores)
            mask = masks[best_idx]
            confidence = float(scores[best_idx])

            # Convert mask to polygon
            coords = self._mask_to_polygon(mask, sat_image, lat, lon)
            if not coords or len(coords) < 4:
                logger.warning("SAM produced invalid polygon")
                return None

            # Calculate area
            area_m2 = self._calculate_polygon_area(coords, lat)

            # Create GeoJSON
            geojson = {
                "type": "Polygon",
                "coordinates": [coords + [coords[0]]]  # Close polygon
            }

            console.print(f"[green]SAM extracted footprint: {area_m2:.0f} m² (conf={confidence:.0%})[/green]")

            return ExtractedFootprint(
                geojson=geojson,
                coordinates=coords,
                area_m2=area_m2,
                confidence=confidence,
                method="sam",
                center_lat=lat,
                center_lon=lon,
                bbox=self._coords_to_bbox(coords),
                notes=[f"SAM score: {confidence:.2f}"],
            )

        except Exception as e:
            logger.warning(f"SAM extraction failed: {e}")
            return None

    def _extract_with_llm(
        self,
        sat_image: SatelliteImage,
        lat: float,
        lon: float,
    ) -> Optional[ExtractedFootprint]:
        """Extract footprint using Komilion/OpenRouter vision LLM."""
        from ..ai.llm_client import LLMClient
        import tempfile

        # Check for Komilion API key
        if not os.environ.get("KOMILION_API_KEY"):
            logger.warning("KOMILION_API_KEY not set for footprint extraction")
            return None

        try:
            console.print("[cyan]Running LLM footprint extraction...[/cyan]")

            # Save image to temp file for LLM client
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                sat_image.image.save(tmp.name, format="PNG")
                temp_path = tmp.name

            # Calculate image bounds for coordinate conversion
            mpp = sat_image.meters_per_pixel
            img_width = sat_image.image.width
            img_height = sat_image.image.height

            prompt = f"""Analyze this satellite image and identify the building footprint at the center.

The image is {img_width}x{img_height} pixels, centered at coordinates ({lat:.6f}, {lon:.6f}).
Each pixel represents approximately {mpp:.2f} meters.

Return a JSON object with the building footprint as pixel coordinates:
{{
    "building_found": true/false,
    "confidence": 0.0-1.0,
    "polygon_pixels": [[x1, y1], [x2, y2], ...],  // Clockwise from top-left corner
    "building_type": "residential/commercial/industrial/unknown",
    "estimated_floors": 1-10,
    "roof_type": "flat/pitched/complex",
    "notes": "any observations"
}}

If no clear building is visible at the center, set building_found to false.
The polygon should trace the building roof outline.
Return ONLY the JSON, no other text."""

            client = LLMClient(mode="balanced")
            response = client.analyze_image(image_path=temp_path, prompt=prompt)

            # Clean up temp file
            os.unlink(temp_path)

            if not response:
                return None

            # Parse response
            response_text = response.content.strip()
            if response_text.startswith("```"):
                response_text = response_text.split("```")[1]
                if response_text.startswith("json"):
                    response_text = response_text[4:]

            result = json.loads(response_text)

            if not result.get("building_found", False):
                logger.info("LLM did not find a building at center")
                return None

            # Convert pixel coordinates to lat/lon
            pixel_coords = result.get("polygon_pixels", [])
            if len(pixel_coords) < 3:
                logger.warning("LLM returned insufficient polygon points")
                return None

            coords = self._pixels_to_latlon(
                pixel_coords, sat_image, lat, lon
            )

            confidence = result.get("confidence", 0.7)
            area_m2 = self._calculate_polygon_area(coords, lat)

            geojson = {
                "type": "Polygon",
                "coordinates": [coords + [coords[0]]]
            }

            notes = [
                f"LLM confidence: {confidence:.0%}",
                f"Building type: {result.get('building_type', 'unknown')}",
                f"Roof type: {result.get('roof_type', 'unknown')}",
            ]
            if result.get("notes"):
                notes.append(result["notes"])

            console.print(f"[green]LLM extracted footprint: {area_m2:.0f} m² (conf={confidence:.0%})[/green]")

            return ExtractedFootprint(
                geojson=geojson,
                coordinates=coords,
                area_m2=area_m2,
                confidence=confidence,
                method="llm",
                center_lat=lat,
                center_lon=lon,
                bbox=self._coords_to_bbox(coords),
                notes=notes,
            )

        except Exception as e:
            logger.warning(f"LLM extraction failed: {e}")
            return None

    def _extract_with_edges(
        self,
        sat_image: SatelliteImage,
        lat: float,
        lon: float,
    ) -> Optional[ExtractedFootprint]:
        """Extract footprint using edge detection (fast fallback)."""
        try:
            import cv2

            console.print("[cyan]Running edge detection...[/cyan]")

            # Convert to numpy
            image_np = np.array(sat_image.image)
            gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)

            # Edge detection
            edges = cv2.Canny(gray, 50, 150)

            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if not contours:
                logger.warning("No contours found")
                return None

            # Find contour closest to center
            h, w = gray.shape
            center = (w // 2, h // 2)

            best_contour = None
            best_dist = float('inf')
            min_area = (w * h) * 0.01  # At least 1% of image

            for contour in contours:
                area = cv2.contourArea(contour)
                if area < min_area:
                    continue

                # Get contour center
                M = cv2.moments(contour)
                if M["m00"] == 0:
                    continue
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])

                dist = math.sqrt((cx - center[0])**2 + (cy - center[1])**2)
                if dist < best_dist:
                    best_dist = dist
                    best_contour = contour

            if best_contour is None:
                logger.warning("No suitable contour found near center")
                return None

            # Simplify contour
            epsilon = 0.02 * cv2.arcLength(best_contour, True)
            approx = cv2.approxPolyDP(best_contour, epsilon, True)

            # Convert to lat/lon
            pixel_coords = [(int(p[0][0]), int(p[0][1])) for p in approx]
            coords = self._pixels_to_latlon(pixel_coords, sat_image, lat, lon)

            if len(coords) < 3:
                return None

            area_m2 = self._calculate_polygon_area(coords, lat)
            confidence = 0.4  # Edge detection is less reliable

            geojson = {
                "type": "Polygon",
                "coordinates": [coords + [coords[0]]]
            }

            console.print(f"[yellow]Edge detection footprint: {area_m2:.0f} m² (low confidence)[/yellow]")

            return ExtractedFootprint(
                geojson=geojson,
                coordinates=coords,
                area_m2=area_m2,
                confidence=confidence,
                method="edge_detection",
                center_lat=lat,
                center_lon=lon,
                bbox=self._coords_to_bbox(coords),
                notes=["Edge detection - verify manually"],
            )

        except ImportError:
            logger.warning("OpenCV not available for edge detection")
            return None
        except Exception as e:
            logger.warning(f"Edge detection failed: {e}")
            return None

    def _check_sam_available(self) -> bool:
        """Check if SAM is available."""
        if self._sam_available is not None:
            return self._sam_available

        if not self.use_sam:
            self._sam_available = False
            return False

        try:
            from segment_anything import sam_model_registry
            self._sam_available = True
        except ImportError:
            logger.info("SAM not available. Install with: pip install segment-anything")
            self._sam_available = False

        return self._sam_available

    def _download_sam_checkpoint(self, path: str):
        """Download SAM checkpoint."""
        import urllib.request

        os.makedirs(os.path.dirname(path), exist_ok=True)
        url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"

        console.print(f"[cyan]Downloading SAM checkpoint (~375MB)...[/cyan]")
        urllib.request.urlretrieve(url, path)
        console.print(f"[green]SAM checkpoint saved to {path}[/green]")

    def _mask_to_polygon(
        self,
        mask: np.ndarray,
        sat_image: SatelliteImage,
        center_lat: float,
        center_lon: float,
    ) -> List[Tuple[float, float]]:
        """Convert binary mask to lat/lon polygon."""
        try:
            import cv2

            # Find contours
            mask_uint8 = (mask * 255).astype(np.uint8)
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if not contours:
                return []

            # Use largest contour
            largest = max(contours, key=cv2.contourArea)

            # Simplify
            epsilon = 0.01 * cv2.arcLength(largest, True)
            approx = cv2.approxPolyDP(largest, epsilon, True)

            # Convert pixels to lat/lon
            pixel_coords = [(int(p[0][0]), int(p[0][1])) for p in approx]
            return self._pixels_to_latlon(pixel_coords, sat_image, center_lat, center_lon)

        except Exception as e:
            logger.warning(f"Mask to polygon conversion failed: {e}")
            return []

    def _pixels_to_latlon(
        self,
        pixel_coords: List[Tuple[int, int]],
        sat_image: SatelliteImage,
        center_lat: float,
        center_lon: float,
    ) -> List[Tuple[float, float]]:
        """Convert pixel coordinates to lat/lon."""
        mpp = sat_image.meters_per_pixel
        img_width = sat_image.image.width
        img_height = sat_image.image.height

        # Image center in pixels
        cx = img_width / 2
        cy = img_height / 2

        # Meters per degree at this latitude
        meters_per_deg_lat = 111320
        meters_per_deg_lon = 111320 * math.cos(math.radians(center_lat))

        coords = []
        for px, py in pixel_coords:
            # Offset from center in meters
            dx_m = (px - cx) * mpp
            dy_m = (cy - py) * mpp  # Y is inverted

            # Convert to degrees
            d_lon = dx_m / meters_per_deg_lon
            d_lat = dy_m / meters_per_deg_lat

            lon = center_lon + d_lon
            lat = center_lat + d_lat

            coords.append((lon, lat))

        return coords

    def _calculate_polygon_area(
        self,
        coords: List[Tuple[float, float]],
        center_lat: float,
    ) -> float:
        """Calculate polygon area in square meters using Shoelace formula."""
        if len(coords) < 3:
            return 0.0

        # Convert to meters relative to first point
        meters_per_deg_lat = 111320
        meters_per_deg_lon = 111320 * math.cos(math.radians(center_lat))

        ref_lon, ref_lat = coords[0]
        xy_coords = []
        for lon, lat in coords:
            x = (lon - ref_lon) * meters_per_deg_lon
            y = (lat - ref_lat) * meters_per_deg_lat
            xy_coords.append((x, y))

        # Shoelace formula
        n = len(xy_coords)
        area = 0.0
        for i in range(n):
            j = (i + 1) % n
            area += xy_coords[i][0] * xy_coords[j][1]
            area -= xy_coords[j][0] * xy_coords[i][1]

        return abs(area) / 2.0

    def _coords_to_bbox(
        self,
        coords: List[Tuple[float, float]],
    ) -> Tuple[float, float, float, float]:
        """Get bounding box from coordinates."""
        if not coords:
            return (0, 0, 0, 0)

        lons = [c[0] for c in coords]
        lats = [c[1] for c in coords]

        return (min(lons), min(lats), max(lons), max(lats))


def extract_footprint_from_satellite(
    lat: float = None,
    lon: float = None,
    address: str = None,
    method: str = "auto",
) -> Optional[ExtractedFootprint]:
    """
    Convenience function to extract building footprint from satellite imagery.

    Args:
        lat, lon: Building coordinates (if known)
        address: Street address (geocodes if lat/lon not provided)
        method: "auto", "sam", "llm", or "edge"

    Returns:
        ExtractedFootprint with GeoJSON polygon

    Example:
        footprint = extract_footprint_from_satellite(address="Bellmansgatan 16, Stockholm")
        print(f"Area: {footprint.area_m2:.0f} m²")
    """
    extractor = FootprintExtractor()

    if lat and lon:
        return extractor.extract_from_coordinates(lat, lon, method=method)
    elif address:
        return extractor.extract_from_address(address, method=method)
    else:
        logger.error("Must provide lat/lon or address")
        return None
