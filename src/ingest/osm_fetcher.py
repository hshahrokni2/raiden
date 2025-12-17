"""
OpenStreetMap data fetcher using Overpass API.

No account required - uses public Overpass API endpoints.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import requests
from rich.console import Console

from ..core.config import settings

console = Console()


class OSMFetcher:
    """
    Fetch building and environmental data from OpenStreetMap.

    Uses the Overpass API (no account required).
    Rate-limited to be respectful to public infrastructure.
    """

    OVERPASS_URL = "https://overpass-api.de/api/interpreter"
    OVERPASS_URL_BACKUP = "https://lz4.overpass-api.de/api/interpreter"

    # OSM tags we're interested in for buildings
    BUILDING_TAGS = [
        "building",
        "building:levels",
        "building:material",
        "building:facade:material",
        "building:colour",
        "roof:material",
        "roof:shape",
        "roof:colour",
        "height",
        "addr:street",
        "addr:housenumber",
    ]

    def __init__(self, cache_dir: Path | None = None):
        self.cache_dir = cache_dir or settings.cache_dir / "osm"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._last_request_time = 0
        self._min_request_interval = 1.0  # seconds

    def _rate_limit(self) -> None:
        """Enforce rate limiting for API requests."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self._min_request_interval:
            time.sleep(self._min_request_interval - elapsed)
        self._last_request_time = time.time()

    def _query(self, query: str, use_cache: bool = True) -> dict[str, Any]:
        """
        Execute Overpass query with caching.

        Args:
            query: Overpass QL query string
            use_cache: Whether to use cached results

        Returns:
            Parsed JSON response
        """
        # Create cache key from query hash
        cache_key = str(hash(query))
        cache_file = self.cache_dir / f"{cache_key}.json"

        if use_cache and cache_file.exists():
            console.print(f"[dim]Using cached OSM data[/dim]")
            with open(cache_file, "r") as f:
                return json.load(f)

        self._rate_limit()

        console.print("[cyan]Fetching data from OpenStreetMap...[/cyan]")

        try:
            response = requests.post(
                self.OVERPASS_URL,
                data={"data": query},
                timeout=60,
            )
            response.raise_for_status()
            data = response.json()
        except requests.RequestException as e:
            console.print(f"[yellow]Primary Overpass failed, trying backup: {e}[/yellow]")
            response = requests.post(
                self.OVERPASS_URL_BACKUP,
                data={"data": query},
                timeout=60,
            )
            response.raise_for_status()
            data = response.json()

        # Cache the result
        with open(cache_file, "w") as f:
            json.dump(data, f)

        return data

    def get_buildings_in_bbox(
        self,
        min_lon: float,
        min_lat: float,
        max_lon: float,
        max_lat: float,
        include_tags: bool = True,
    ) -> list[dict[str, Any]]:
        """
        Get all buildings within a bounding box.

        Args:
            min_lon, min_lat, max_lon, max_lat: Bounding box in WGS84
            include_tags: Whether to fetch all tags

        Returns:
            List of building features with geometry and tags
        """
        bbox = f"{min_lat},{min_lon},{max_lat},{max_lon}"

        query = f"""
        [out:json][timeout:60];
        (
          way["building"]({bbox});
          relation["building"]({bbox});
        );
        out body;
        >;
        out skel qt;
        """

        data = self._query(query)
        return self._parse_buildings(data)

    def get_buildings_by_address(
        self, street: str, municipality: str = "Stockholm"
    ) -> list[dict[str, Any]]:
        """
        Search for buildings by address.

        Args:
            street: Street name (e.g., "Aktergatan")
            municipality: Municipality name

        Returns:
            List of matching buildings
        """
        query = f"""
        [out:json][timeout:60];
        area["name"="{municipality}"]->.searchArea;
        (
          way["building"]["addr:street"~"{street}",i](area.searchArea);
        );
        out body;
        >;
        out skel qt;
        """

        data = self._query(query)
        return self._parse_buildings(data)

    def get_trees_in_bbox(
        self,
        min_lon: float,
        min_lat: float,
        max_lon: float,
        max_lat: float,
    ) -> list[dict[str, Any]]:
        """
        Get trees and vegetation within a bounding box.

        Useful for shading analysis.
        """
        bbox = f"{min_lat},{min_lon},{max_lat},{max_lon}"

        query = f"""
        [out:json][timeout:60];
        (
          node["natural"="tree"]({bbox});
          way["natural"="tree_row"]({bbox});
          way["landuse"="forest"]({bbox});
          way["natural"="wood"]({bbox});
          way["leisure"="park"]({bbox});
        );
        out body;
        >;
        out skel qt;
        """

        data = self._query(query)
        return self._parse_vegetation(data)

    def get_nearby_buildings(
        self,
        center_lon: float,
        center_lat: float,
        radius_m: float = 100,
    ) -> list[dict[str, Any]]:
        """
        Get buildings near a point (for shading analysis).

        Args:
            center_lon, center_lat: Center point in WGS84
            radius_m: Search radius in meters

        Returns:
            List of nearby buildings
        """
        query = f"""
        [out:json][timeout:60];
        (
          way["building"](around:{radius_m},{center_lat},{center_lon});
          relation["building"](around:{radius_m},{center_lat},{center_lon});
        );
        out body;
        >;
        out skel qt;
        """

        data = self._query(query)
        return self._parse_buildings(data)

    def _parse_buildings(self, data: dict[str, Any]) -> list[dict[str, Any]]:
        """Parse Overpass response into building features."""
        buildings = []
        nodes = {}

        # First pass: collect all nodes
        for element in data.get("elements", []):
            if element["type"] == "node":
                nodes[element["id"]] = (element["lon"], element["lat"])

        # Second pass: build polygons from ways
        for element in data.get("elements", []):
            if element["type"] == "way" and "tags" in element:
                tags = element["tags"]
                if "building" not in tags:
                    continue

                # Build coordinate list
                coords = []
                for node_id in element.get("nodes", []):
                    if node_id in nodes:
                        coords.append(nodes[node_id])

                if len(coords) < 3:
                    continue

                building = {
                    "osm_id": element["id"],
                    "type": "way",
                    "coordinates": coords,
                    "tags": tags,
                    # Extract specific useful tags
                    "building_type": tags.get("building", "yes"),
                    "levels": self._parse_int(tags.get("building:levels")),
                    "height": self._parse_float(tags.get("height")),
                    "material": tags.get("building:material"),
                    "facade_material": tags.get("building:facade:material"),
                    "roof_material": tags.get("roof:material"),
                    "roof_shape": tags.get("roof:shape"),
                    "address": self._build_address(tags),
                }
                buildings.append(building)

        return buildings

    def _parse_vegetation(self, data: dict[str, Any]) -> list[dict[str, Any]]:
        """Parse Overpass response into vegetation features."""
        vegetation = []
        nodes = {}

        for element in data.get("elements", []):
            if element["type"] == "node":
                nodes[element["id"]] = (element["lon"], element["lat"])

                # Individual trees
                if "tags" in element and element["tags"].get("natural") == "tree":
                    vegetation.append({
                        "type": "tree",
                        "osm_id": element["id"],
                        "coordinates": (element["lon"], element["lat"]),
                        "tags": element["tags"],
                        "height": self._parse_float(element["tags"].get("height")),
                        "diameter_crown": self._parse_float(
                            element["tags"].get("diameter_crown")
                        ),
                    })

        # Tree rows and forests
        for element in data.get("elements", []):
            if element["type"] == "way" and "tags" in element:
                coords = [nodes[nid] for nid in element.get("nodes", []) if nid in nodes]
                if not coords:
                    continue

                veg_type = None
                if element["tags"].get("natural") == "tree_row":
                    veg_type = "tree_row"
                elif element["tags"].get("landuse") == "forest":
                    veg_type = "forest"
                elif element["tags"].get("natural") == "wood":
                    veg_type = "wood"
                elif element["tags"].get("leisure") == "park":
                    veg_type = "park"

                if veg_type:
                    vegetation.append({
                        "type": veg_type,
                        "osm_id": element["id"],
                        "coordinates": coords,
                        "tags": element["tags"],
                    })

        return vegetation

    @staticmethod
    def _parse_int(value: str | None) -> int | None:
        """Parse string to int, handling common formats."""
        if value is None:
            return None
        try:
            # Handle "5" or "5.0" or "5 floors"
            return int(float(value.split()[0]))
        except (ValueError, IndexError):
            return None

    @staticmethod
    def _parse_float(value: str | None) -> float | None:
        """Parse string to float, handling units."""
        if value is None:
            return None
        try:
            # Handle "15.5" or "15.5 m" or "15,5"
            cleaned = value.replace(",", ".").split()[0]
            return float(cleaned)
        except (ValueError, IndexError):
            return None

    @staticmethod
    def _build_address(tags: dict[str, str]) -> str | None:
        """Build address string from OSM tags."""
        street = tags.get("addr:street", "")
        number = tags.get("addr:housenumber", "")
        if street:
            return f"{street} {number}".strip()
        return None

    def find_matching_osm_building(
        self,
        target_coords: list[tuple[float, float]],
        osm_buildings: list[dict],
        tolerance: float = 0.0001,  # ~10m in degrees
    ) -> dict | None:
        """
        Find OSM building that matches target coordinates.

        Uses centroid matching with tolerance.
        """
        from shapely.geometry import Polygon

        # Calculate target centroid
        target_poly = Polygon(target_coords)
        target_centroid = (target_poly.centroid.x, target_poly.centroid.y)

        best_match = None
        best_distance = float("inf")

        for osm_building in osm_buildings:
            osm_coords = osm_building["coordinates"]
            if len(osm_coords) < 3:
                continue

            osm_poly = Polygon(osm_coords)
            osm_centroid = (osm_poly.centroid.x, osm_poly.centroid.y)

            # Calculate distance
            dist = (
                (target_centroid[0] - osm_centroid[0]) ** 2
                + (target_centroid[1] - osm_centroid[1]) ** 2
            ) ** 0.5

            if dist < best_distance and dist < tolerance:
                best_distance = dist
                best_match = osm_building

        return best_match
