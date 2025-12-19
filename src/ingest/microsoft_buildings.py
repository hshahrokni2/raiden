"""
Microsoft Global ML Building Footprints fetcher.

Direct access to Microsoft's 1.4B building footprints dataset.
Sweden has 399 files available with some height data.

Use this as fallback when Overture Maps doesn't have the building.

Data source: https://github.com/microsoft/GlobalMLBuildingFootprints
License: ODbL (Open Database License)
"""

from __future__ import annotations

import gzip
import json
import math
from pathlib import Path
from typing import Any, List, Optional, Tuple
import requests

from rich.console import Console

from ..core.config import settings

console = Console()


# Quadkey functions for Microsoft's tile system
def lat_lon_to_quadkey(lat: float, lon: float, level: int = 9) -> str:
    """Convert lat/lon to Microsoft quadkey at given level."""
    x = (lon + 180.0) / 360.0
    sin_lat = math.sin(lat * math.pi / 180.0)
    y = 0.5 - math.log((1 + sin_lat) / (1 - sin_lat)) / (4 * math.pi)

    # Clamp y to valid range
    y = max(0, min(1, y))

    map_size = 1 << level
    tile_x = int(x * map_size)
    tile_y = int(y * map_size)

    # Ensure within bounds
    tile_x = max(0, min(map_size - 1, tile_x))
    tile_y = max(0, min(map_size - 1, tile_y))

    quadkey = ""
    for i in range(level, 0, -1):
        digit = 0
        mask = 1 << (i - 1)
        if (tile_x & mask) != 0:
            digit += 1
        if (tile_y & mask) != 0:
            digit += 2
        quadkey += str(digit)

    return quadkey


def quadkey_to_bbox(quadkey: str) -> Tuple[float, float, float, float]:
    """Convert quadkey to bounding box (min_lon, min_lat, max_lon, max_lat)."""
    level = len(quadkey)
    tile_x = 0
    tile_y = 0

    for i, char in enumerate(quadkey):
        mask = 1 << (level - i - 1)
        digit = int(char)
        if digit & 1:
            tile_x |= mask
        if digit & 2:
            tile_y |= mask

    map_size = 1 << level

    min_lon = tile_x / map_size * 360.0 - 180.0
    max_lon = (tile_x + 1) / map_size * 360.0 - 180.0

    min_lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * (tile_y + 1) / map_size)))
    max_lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * tile_y / map_size)))

    min_lat = min_lat_rad * 180.0 / math.pi
    max_lat = max_lat_rad * 180.0 / math.pi

    return (min_lon, min_lat, max_lon, max_lat)


class MicrosoftBuildingsFetcher:
    """
    Fetch building footprints from Microsoft's Global ML Building Footprints.

    Data is stored in Azure blob storage, partitioned by quadkey.
    Sweden data is under "Sweden" country prefix.

    Features:
    - 1.4B buildings globally
    - Sweden: 399 files available
    - ~20% have height estimates
    - GeoJSONL format (gzipped)
    """

    # Base URL for Microsoft building footprints
    BASE_URL = "https://minedbuildings.z5.web.core.windows.net/global-buildings/dataset-links.csv"
    SWEDEN_PREFIX = "Sweden"

    def __init__(self, cache_dir: Path | None = None):
        self.cache_dir = cache_dir or settings.cache_dir / "microsoft_buildings"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._links_cache: dict[str, str] = {}

    def _load_links(self) -> dict[str, str]:
        """Load dataset links from Microsoft's index."""
        if self._links_cache:
            return self._links_cache

        cache_file = self.cache_dir / "dataset-links.csv"

        # Download if not cached (refresh weekly)
        if not cache_file.exists():
            console.print("[cyan]Downloading Microsoft Buildings index...[/cyan]")
            try:
                response = requests.get(self.BASE_URL, timeout=30)
                response.raise_for_status()
                cache_file.write_text(response.text)
            except Exception as e:
                console.print(f"[red]Failed to fetch index: {e}[/red]")
                return {}

        # Parse CSV (Location,QuadKey,Url,Size)
        for line in cache_file.read_text().strip().split('\n')[1:]:
            parts = line.split(',')
            if len(parts) >= 3:
                location, quadkey, url = parts[0], parts[1], parts[2]
                self._links_cache[f"{location}_{quadkey}"] = url

        return self._links_cache

    def get_buildings_for_location(
        self,
        lat: float,
        lon: float,
        search_radius_m: float = 500,
    ) -> List[dict[str, Any]]:
        """
        Get buildings near a location from Microsoft dataset.

        Args:
            lat: Latitude
            lon: Longitude
            search_radius_m: Search radius in meters

        Returns:
            List of building features (GeoJSON)
        """
        # Get quadkey for location
        quadkey = lat_lon_to_quadkey(lat, lon, level=9)

        console.print(f"[dim]Looking for Microsoft buildings in quadkey {quadkey}[/dim]")

        # Try to find Swedish data file
        links = self._load_links()

        # Look for Sweden files matching this quadkey
        matching_url = None
        for key, url in links.items():
            if key.startswith(self.SWEDEN_PREFIX) and quadkey in key:
                matching_url = url
                break

        if not matching_url:
            # Try shorter quadkey prefix
            for prefix_len in range(len(quadkey), 5, -1):
                prefix = quadkey[:prefix_len]
                for key, url in links.items():
                    if key.startswith(self.SWEDEN_PREFIX) and prefix in key:
                        matching_url = url
                        break
                if matching_url:
                    break

        if not matching_url:
            console.print(f"[yellow]No Microsoft data for quadkey {quadkey}[/yellow]")
            return []

        # Download and parse the file
        buildings = self._download_and_parse(matching_url, quadkey)

        # Filter to search radius
        filtered = self._filter_by_distance(buildings, lat, lon, search_radius_m)

        console.print(f"[green]Found {len(filtered)} buildings within {search_radius_m}m[/green]")

        return filtered

    def _download_and_parse(
        self,
        url: str,
        quadkey: str,
    ) -> List[dict[str, Any]]:
        """Download and parse a Microsoft buildings file."""
        # Cache file
        cache_file = self.cache_dir / f"{quadkey}.geojsonl"

        if not cache_file.exists():
            console.print(f"[cyan]Downloading Microsoft buildings data...[/cyan]")
            try:
                response = requests.get(url, timeout=60)
                response.raise_for_status()

                # Decompress gzip
                content = gzip.decompress(response.content).decode('utf-8')
                cache_file.write_text(content)

            except Exception as e:
                console.print(f"[red]Failed to download: {e}[/red]")
                return []

        # Parse GeoJSONL (one feature per line)
        buildings = []
        for line in cache_file.read_text().strip().split('\n'):
            if line:
                try:
                    feature = json.loads(line)
                    buildings.append(feature)
                except json.JSONDecodeError:
                    continue

        return buildings

    def _filter_by_distance(
        self,
        buildings: List[dict[str, Any]],
        center_lat: float,
        center_lon: float,
        radius_m: float,
    ) -> List[dict[str, Any]]:
        """Filter buildings by distance from center point."""
        filtered = []

        for building in buildings:
            geom = building.get("geometry", {})
            coords = geom.get("coordinates", [])

            if not coords:
                continue

            # Get centroid of polygon
            if geom.get("type") == "Polygon":
                ring = coords[0] if coords else []
                if ring:
                    centroid_lon = sum(p[0] for p in ring) / len(ring)
                    centroid_lat = sum(p[1] for p in ring) / len(ring)

                    # Approximate distance in meters
                    dist = self._haversine_distance(
                        center_lat, center_lon,
                        centroid_lat, centroid_lon
                    )

                    if dist <= radius_m:
                        # Add distance to properties
                        building.setdefault("properties", {})["distance_m"] = dist
                        filtered.append(building)

        # Sort by distance
        filtered.sort(key=lambda b: b.get("properties", {}).get("distance_m", float('inf')))

        return filtered

    def _haversine_distance(
        self,
        lat1: float, lon1: float,
        lat2: float, lon2: float,
    ) -> float:
        """Calculate distance between two points in meters."""
        R = 6371000  # Earth radius in meters

        phi1 = math.radians(lat1)
        phi2 = math.radians(lat2)
        delta_phi = math.radians(lat2 - lat1)
        delta_lambda = math.radians(lon2 - lon1)

        a = (math.sin(delta_phi / 2) ** 2 +
             math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2)
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

        return R * c

    def parse_building(self, feature: dict[str, Any]) -> dict[str, Any]:
        """
        Parse a Microsoft building feature into standardized format.

        Args:
            feature: GeoJSON feature from Microsoft dataset

        Returns:
            Standardized building dict
        """
        props = feature.get("properties", {})
        geom = feature.get("geometry", {})

        return {
            "ms_id": props.get("id"),
            "height": props.get("height"),  # Available for ~20% of buildings
            "confidence": props.get("confidence"),
            "geometry_type": geom.get("type"),
            "coordinates": geom.get("coordinates"),
            "source": "microsoft_buildings",
        }


# Convenience function
def get_microsoft_buildings(
    lat: float,
    lon: float,
    radius_m: float = 500,
) -> List[dict[str, Any]]:
    """
    Get Microsoft building footprints near a location.

    Args:
        lat: Latitude
        lon: Longitude
        radius_m: Search radius in meters

    Returns:
        List of building features
    """
    fetcher = MicrosoftBuildingsFetcher()
    return fetcher.get_buildings_for_location(lat, lon, radius_m)
