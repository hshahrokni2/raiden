"""
Lantmäteriet Orthophoto Fetcher

Fetches free Swedish aerial imagery from Lantmäteriet's open data API.
- 0.16m resolution (color)
- 0.4m resolution (infrared)
- CC0 license (free for commercial use)

Requires: Lantmäteriet API key (free registration)
Get one at: https://www.lantmateriet.se/
"""

import os
import requests
from io import BytesIO
from PIL import Image
from dataclasses import dataclass
from typing import Optional, Tuple, List
import math

from rich.console import Console

console = Console()


@dataclass
class OrtophotoImage:
    """Orthophoto image with metadata."""
    image: Image.Image
    center_lat: float
    center_lon: float
    resolution_m: float  # Meters per pixel
    width_m: float
    height_m: float
    year: Optional[int] = None


class LantmaterietFetcher:
    """
    Fetch orthophotos from Lantmäteriet's WMS/WMTS services.

    Free data under CC0 license. Requires API key from Lantmäteriet.

    Coverage:
    - Color: 0.16m resolution
    - IR: 0.4m resolution
    - Updated every 2-4 years depending on region
    """

    # WMS endpoint
    WMS_URL = "https://api.lantmateriet.se/open/topowebb-ccby/v1/wmts"
    ORTOFOTO_WMS = "https://api.lantmateriet.se/open/ortofoto-visning/wms/v1"

    # Swedish coordinate system (SWEREF99 TM)
    SWEREF99_EPSG = "EPSG:3006"

    def __init__(self, api_key: str = None):
        """
        Initialize with Lantmäteriet API key.

        Args:
            api_key: Lantmäteriet API key (free registration required)
        """
        self.api_key = api_key or os.getenv("LANTMATERIET_API_KEY")

    def fetch_ortofoto_wms(
        self,
        lat: float,
        lon: float,
        width_m: float = 100,
        height_m: float = 100,
        resolution_m: float = 0.25,
    ) -> Optional[OrtophotoImage]:
        """
        Fetch orthophoto using WMS GetMap.

        Args:
            lat, lon: Center coordinates (WGS84)
            width_m: Width of area in meters
            height_m: Height of area in meters
            resolution_m: Target resolution in meters/pixel

        Returns:
            OrtophotoImage or None
        """
        if not self.api_key:
            console.print("[yellow]Lantmäteriet API key not set. Get one free at lantmateriet.se[/yellow]")
            return None

        # Convert to SWEREF99 TM
        x, y = self._wgs84_to_sweref99(lat, lon)

        # Calculate bounding box
        half_w = width_m / 2
        half_h = height_m / 2
        bbox = f"{x - half_w},{y - half_h},{x + half_w},{y + half_h}"

        # Calculate image size in pixels
        img_width = int(width_m / resolution_m)
        img_height = int(height_m / resolution_m)

        # WMS GetMap request
        params = {
            "SERVICE": "WMS",
            "VERSION": "1.1.1",
            "REQUEST": "GetMap",
            "LAYERS": "Ortofoto_0.16",  # 0.16m resolution color
            "STYLES": "",
            "SRS": self.SWEREF99_EPSG,
            "BBOX": bbox,
            "WIDTH": img_width,
            "HEIGHT": img_height,
            "FORMAT": "image/png",
        }

        headers = {"Authorization": f"Bearer {self.api_key}"}

        try:
            response = requests.get(
                self.ORTOFOTO_WMS,
                params=params,
                headers=headers,
                timeout=30,
            )

            if response.status_code == 200 and 'image' in response.headers.get('content-type', ''):
                img = Image.open(BytesIO(response.content))
                return OrtophotoImage(
                    image=img,
                    center_lat=lat,
                    center_lon=lon,
                    resolution_m=resolution_m,
                    width_m=width_m,
                    height_m=height_m,
                )
            else:
                console.print(f"[red]Lantmäteriet WMS failed: {response.status_code}[/red]")
                if response.text:
                    console.print(f"[dim]{response.text[:200]}[/dim]")
                return None

        except Exception as e:
            console.print(f"[red]Lantmäteriet fetch error: {e}[/red]")
            return None

    def fetch_building_ortofoto(
        self,
        footprint: dict,
        padding_m: float = 20,
        resolution_m: float = 0.25,
    ) -> Optional[OrtophotoImage]:
        """
        Fetch orthophoto covering a building footprint.

        Args:
            footprint: GeoJSON geometry
            padding_m: Padding around building
            resolution_m: Target resolution

        Returns:
            OrtophotoImage covering the building
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

        # Calculate dimensions
        width_m = self._haversine_distance(center_lat, min(lons), center_lat, max(lons))
        height_m = self._haversine_distance(min(lats), center_lon, max(lats), center_lon)

        # Add padding
        width_m += 2 * padding_m
        height_m += 2 * padding_m

        return self.fetch_ortofoto_wms(
            center_lat, center_lon,
            width_m=width_m,
            height_m=height_m,
            resolution_m=resolution_m,
        )

    def _wgs84_to_sweref99(self, lat: float, lon: float) -> Tuple[float, float]:
        """
        Convert WGS84 to SWEREF99 TM coordinates.

        Simplified conversion - for production use pyproj.
        """
        try:
            from pyproj import Transformer
            transformer = Transformer.from_crs("EPSG:4326", "EPSG:3006", always_xy=True)
            x, y = transformer.transform(lon, lat)
            return x, y
        except ImportError:
            # Fallback: approximate conversion for Stockholm area
            # SWEREF99 TM uses central meridian 15°E
            # This is rough but works for demo purposes
            k0 = 0.9996
            a = 6378137  # WGS84 semi-major axis

            lat_rad = math.radians(lat)
            lon_rad = math.radians(lon)
            lon0_rad = math.radians(15)  # Central meridian

            N = a / math.sqrt(1 - 0.00669438 * math.sin(lat_rad) ** 2)
            T = math.tan(lat_rad) ** 2
            C = 0.006739497 * math.cos(lat_rad) ** 2
            A = (lon_rad - lon0_rad) * math.cos(lat_rad)

            M = a * ((1 - 0.00669438 / 4) * lat_rad
                     - (3 * 0.00669438 / 8) * math.sin(2 * lat_rad))

            x = 500000 + k0 * N * (A + (1 - T + C) * A ** 3 / 6)
            y = k0 * (M + N * math.tan(lat_rad) * (A ** 2 / 2))

            return x, y

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

    def _haversine_distance(
        self,
        lat1: float,
        lon1: float,
        lat2: float,
        lon2: float,
    ) -> float:
        """Calculate distance between two points in meters."""
        R = 6371000
        lat1_rad, lat2_rad = math.radians(lat1), math.radians(lat2)
        delta_lat = math.radians(lat2 - lat1)
        delta_lon = math.radians(lon2 - lon1)

        a = (math.sin(delta_lat / 2) ** 2 +
             math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(delta_lon / 2) ** 2)
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

        return R * c


# Alternative: Use OpenStreetMap tiles for aerial
class OSMAerialFetcher:
    """
    Fetch aerial imagery from OpenStreetMap-compatible tile servers.

    Uses Swedish WMTS services that don't require API keys.
    """

    # Free aerial tile sources for Sweden
    TILE_SOURCES = {
        # Lantmäteriet's free topographic web map (not ortofoto but useful)
        "lantmateriet_topo": "https://api.lantmateriet.se/open/topowebb-ccby/v1/wmts/1.0.0/topowebb/default/3857/{z}/{y}/{x}.png",
        # OpenStreetMap standard
        "osm": "https://tile.openstreetmap.org/{z}/{x}/{y}.png",
    }

    def fetch_tiles(
        self,
        lat: float,
        lon: float,
        zoom: int = 18,
        source: str = "osm",
    ) -> Optional[Image.Image]:
        """Fetch map tiles (note: not aerial, but useful for context)."""
        url_template = self.TILE_SOURCES.get(source)
        if not url_template:
            return None

        # Calculate tile coordinates
        x, y = self._latlon_to_tile(lat, lon, zoom)

        url = url_template.format(z=zoom, x=x, y=y)

        try:
            headers = {"User-Agent": "Raiden-Building-Analyzer/1.0"}
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code == 200:
                return Image.open(BytesIO(response.content))
        except Exception as e:
            console.print(f"[dim]Tile fetch failed: {e}[/dim]")

        return None

    def _latlon_to_tile(self, lat: float, lon: float, zoom: int) -> Tuple[int, int]:
        """Convert lat/lon to tile coordinates."""
        lat_rad = math.radians(lat)
        n = 2.0 ** zoom
        x = int((lon + 180.0) / 360.0 * n)
        y = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
        return x, y
