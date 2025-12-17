"""
Overture Maps data fetcher.

No account required - direct S3 access via overturemaps CLI.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from rich.console import Console

from ..core.config import settings

console = Console()


class OvertureFetcher:
    """
    Fetch building data from Overture Maps.

    Uses the overturemaps Python CLI (pip install overturemaps).
    No account required - data accessed directly from S3.
    """

    def __init__(self, cache_dir: Path | None = None):
        self.cache_dir = cache_dir or settings.cache_dir / "overture"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._check_cli()

    def _check_cli(self) -> None:
        """Check if overturemaps CLI is available."""
        try:
            subprocess.run(
                ["overturemaps", "--version"],
                capture_output=True,
                check=True,
            )
        except (subprocess.CalledProcessError, FileNotFoundError):
            console.print(
                "[yellow]overturemaps CLI not found. Install with: pip install overturemaps[/yellow]"
            )

    def get_buildings_in_bbox(
        self,
        min_lon: float,
        min_lat: float,
        max_lon: float,
        max_lat: float,
        use_cache: bool = True,
    ) -> list[dict[str, Any]]:
        """
        Get buildings within a bounding box from Overture Maps.

        Args:
            min_lon, min_lat, max_lon, max_lat: Bounding box in WGS84
            use_cache: Whether to use cached results

        Returns:
            List of building features
        """
        bbox_str = f"{min_lon},{min_lat},{max_lon},{max_lat}"
        cache_key = f"buildings_{bbox_str.replace(',', '_').replace('.', 'd')}"
        cache_file = self.cache_dir / f"{cache_key}.geojson"

        if use_cache and cache_file.exists():
            console.print("[dim]Using cached Overture data[/dim]")
            with open(cache_file, "r") as f:
                data = json.load(f)
            return data.get("features", [])

        console.print("[cyan]Fetching data from Overture Maps...[/cyan]")

        try:
            # Use overturemaps CLI
            result = subprocess.run(
                [
                    "overturemaps",
                    "download",
                    f"--bbox={bbox_str}",
                    "-f", "geojson",
                    "--type=building",
                    "-o", str(cache_file),
                ],
                capture_output=True,
                text=True,
                check=True,
            )

            with open(cache_file, "r") as f:
                data = json.load(f)

            return data.get("features", [])

        except subprocess.CalledProcessError as e:
            console.print(f"[red]Overture fetch failed: {e.stderr}[/red]")
            # Fall back to DuckDB method
            return self._fetch_via_duckdb(min_lon, min_lat, max_lon, max_lat, cache_file)

        except FileNotFoundError:
            console.print("[yellow]overturemaps CLI not installed, using DuckDB fallback[/yellow]")
            return self._fetch_via_duckdb(min_lon, min_lat, max_lon, max_lat, cache_file)

    def _fetch_via_duckdb(
        self,
        min_lon: float,
        min_lat: float,
        max_lon: float,
        max_lat: float,
        output_file: Path,
    ) -> list[dict[str, Any]]:
        """
        Fallback: fetch via DuckDB directly.

        Requires: pip install duckdb
        """
        try:
            import duckdb
        except ImportError:
            console.print("[red]Neither overturemaps nor duckdb available[/red]")
            return []

        console.print("[cyan]Fetching via DuckDB...[/cyan]")

        conn = duckdb.connect()
        conn.execute("INSTALL spatial; LOAD spatial;")
        conn.execute("INSTALL httpfs; LOAD httpfs;")

        # Query Overture S3 bucket
        query = f"""
        SELECT
            id,
            names.primary as name,
            height,
            num_floors,
            facade_material,
            roof_material,
            roof_shape,
            ST_AsGeoJSON(geometry) as geometry
        FROM read_parquet('s3://overturemaps-us-west-2/release/2024-11-13.0/theme=buildings/type=building/*')
        WHERE bbox.xmin >= {min_lon}
          AND bbox.xmax <= {max_lon}
          AND bbox.ymin >= {min_lat}
          AND bbox.ymax <= {max_lat}
        """

        try:
            results = conn.execute(query).fetchall()
            columns = ["id", "name", "height", "num_floors", "facade_material",
                      "roof_material", "roof_shape", "geometry"]

            features = []
            for row in results:
                props = dict(zip(columns[:-1], row[:-1]))
                geom = json.loads(row[-1]) if row[-1] else None

                features.append({
                    "type": "Feature",
                    "properties": props,
                    "geometry": geom,
                })

            # Cache result
            geojson = {"type": "FeatureCollection", "features": features}
            with open(output_file, "w") as f:
                json.dump(geojson, f)

            return features

        except Exception as e:
            console.print(f"[red]DuckDB query failed: {e}[/red]")
            return []

    def parse_building(self, feature: dict[str, Any]) -> dict[str, Any]:
        """
        Parse an Overture building feature into standardized format.

        Args:
            feature: GeoJSON feature from Overture

        Returns:
            Standardized building dict
        """
        props = feature.get("properties", {})
        geom = feature.get("geometry", {})

        return {
            "overture_id": props.get("id"),
            "name": props.get("name"),
            "height": props.get("height"),
            "num_floors": props.get("num_floors"),
            "facade_material": props.get("facade_material"),
            "roof_material": props.get("roof_material"),
            "roof_shape": props.get("roof_shape"),
            "geometry_type": geom.get("type"),
            "coordinates": geom.get("coordinates"),
            "source": "overture",
        }

    def get_places_in_bbox(
        self,
        min_lon: float,
        min_lat: float,
        max_lon: float,
        max_lat: float,
    ) -> list[dict[str, Any]]:
        """
        Get places (POIs) within a bounding box.

        Useful for identifying commercial uses in buildings.
        """
        bbox_str = f"{min_lon},{min_lat},{max_lon},{max_lat}"
        cache_key = f"places_{bbox_str.replace(',', '_').replace('.', 'd')}"
        cache_file = self.cache_dir / f"{cache_key}.geojson"

        if cache_file.exists():
            with open(cache_file, "r") as f:
                data = json.load(f)
            return data.get("features", [])

        try:
            subprocess.run(
                [
                    "overturemaps",
                    "download",
                    f"--bbox={bbox_str}",
                    "-f", "geojson",
                    "--type=place",
                    "-o", str(cache_file),
                ],
                capture_output=True,
                text=True,
                check=True,
            )

            with open(cache_file, "r") as f:
                data = json.load(f)

            return data.get("features", [])

        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            console.print(f"[yellow]Places fetch failed: {e}[/yellow]")
            return []

    def enrich_from_overture(
        self,
        target_coords: list[tuple[float, float]],
        overture_buildings: list[dict],
        tolerance: float = 0.0001,
    ) -> dict[str, Any] | None:
        """
        Find and extract Overture data for a building.

        Args:
            target_coords: Building footprint coordinates in WGS84
            overture_buildings: List of Overture building features
            tolerance: Matching tolerance in degrees

        Returns:
            Matched building data or None
        """
        from shapely.geometry import Polygon, shape

        # Calculate target centroid
        target_poly = Polygon(target_coords)
        target_centroid = target_poly.centroid

        best_match = None
        best_distance = float("inf")

        for feature in overture_buildings:
            geom = feature.get("geometry")
            if not geom:
                continue

            try:
                overture_poly = shape(geom)
                overture_centroid = overture_poly.centroid

                dist = target_centroid.distance(overture_centroid)

                if dist < best_distance and dist < tolerance:
                    best_distance = dist
                    best_match = self.parse_building(feature)

            except Exception:
                continue

        return best_match


class OvertureDataMerger:
    """
    Merge Overture data into enriched building models.
    """

    def __init__(self):
        self.fetcher = OvertureFetcher()

    def enrich_building(
        self,
        building_coords_wgs84: list[tuple[float, float]],
        bbox: tuple[float, float, float, float],
    ) -> dict[str, Any]:
        """
        Fetch and merge Overture data for a building.

        Args:
            building_coords_wgs84: Building footprint in WGS84
            bbox: Search bounding box (min_lon, min_lat, max_lon, max_lat)

        Returns:
            Dict with any enriched data found
        """
        enriched = {}

        # Fetch Overture buildings in area
        overture_buildings = self.fetcher.get_buildings_in_bbox(*bbox)

        if overture_buildings:
            match = self.fetcher.enrich_from_overture(
                building_coords_wgs84, overture_buildings
            )

            if match:
                if match.get("height"):
                    enriched["height_m"] = match["height"]
                if match.get("num_floors"):
                    enriched["num_floors"] = match["num_floors"]
                if match.get("facade_material"):
                    enriched["facade_material"] = match["facade_material"]
                if match.get("roof_material"):
                    enriched["roof_material"] = match["roof_material"]
                if match.get("roof_shape"):
                    enriched["roof_shape"] = match["roof_shape"]

        return enriched
