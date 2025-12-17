"""
Enriched JSON exporter.

Exports the enriched BRF data structure with all analysis results.
"""

from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel
from rich.console import Console

from ..core.models import EnrichedBRFProperty, EnrichedBuilding

console = Console()


class EnrichedJSONExporter:
    """
    Export enriched BRF data to JSON format.

    Supports:
    - Full export with all data
    - Summary export for dashboards
    - GeoJSON export for mapping
    """

    def __init__(self, pretty: bool = True):
        """
        Initialize exporter.

        Args:
            pretty: Whether to format JSON with indentation
        """
        self.pretty = pretty

    def export_full(
        self,
        brf: EnrichedBRFProperty,
        output_path: Path | str,
    ) -> Path:
        """
        Export complete enriched data to JSON.

        Args:
            brf: Enriched BRF property
            output_path: Output file path

        Returns:
            Path to exported file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert to dict with custom serialization
        data = self._serialize(brf.model_dump())

        with open(output_path, "w", encoding="utf-8") as f:
            if self.pretty:
                json.dump(data, f, indent=2, ensure_ascii=False, default=str)
            else:
                json.dump(data, f, ensure_ascii=False, default=str)

        console.print(f"[green]Exported enriched JSON: {output_path}[/green]")
        return output_path

    def export_summary(
        self,
        brf: EnrichedBRFProperty,
        output_path: Path | str,
    ) -> Path:
        """
        Export summary data (for dashboards/reports).

        Includes key metrics without full geometry.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        summary = {
            "brf_name": brf.brf_name,
            "export_date": date.today().isoformat(),
            "overview": {
                "total_buildings": len(brf.buildings),
                "total_apartments": brf.original_summary.total_apartments,
                "total_heated_area_sqm": brf.original_summary.total_heated_area_sqm,
                "construction_year": brf.original_summary.construction_year,
                "energy_class": brf.original_summary.energy_class.value,
            },
            "energy": {
                "energy_performance_kwh_per_sqm": brf.original_summary.energy_performance_kwh_per_sqm,
                "has_solar_panels": brf.original_summary.has_solar_panels,
                "total_remaining_pv_potential_kwh": brf.total_remaining_pv_potential_kwh,
            },
            "buildings": [],
        }

        for building in brf.buildings:
            ep = building.energyplus_ready
            env = ep.envelope

            building_summary = {
                "id": building.building_id,
                "address": building.original_properties.location.address,
                "height_m": ep.height_m,
                "floors": ep.num_stories,
                "facade_material": env.facade_material.value if env.facade_material else None,
                "wwr": {
                    "average": env.window_to_wall_ratio.average if env.window_to_wall_ratio else None,
                    "confidence": env.window_to_wall_ratio.confidence if env.window_to_wall_ratio else None,
                },
                "u_values": env.u_values.model_dump() if env.u_values else None,
                "solar_potential_kwh": ep.solar_potential.annual_yield_potential_kwh,
            }
            summary["buildings"].append(building_summary)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        console.print(f"[green]Exported summary JSON: {output_path}[/green]")
        return output_path

    def export_geojson(
        self,
        brf: EnrichedBRFProperty,
        output_path: Path | str,
    ) -> Path:
        """
        Export as GeoJSON FeatureCollection.

        Useful for mapping and GIS integration.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        features = []

        for building in brf.buildings:
            ep = building.energyplus_ready

            # Create GeoJSON polygon from WGS84 coordinates
            coords = ep.footprint_coords_wgs84
            if coords:
                # Ensure closed polygon
                if coords[0] != coords[-1]:
                    coords = coords + [coords[0]]

                feature = {
                    "type": "Feature",
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [coords],
                    },
                    "properties": {
                        "building_id": building.building_id,
                        "brf_name": brf.brf_name,
                        "address": building.original_properties.location.address,
                        "height_m": ep.height_m,
                        "floors": ep.num_stories,
                        "energy_class": building.original_properties.energy.energy_class.value,
                        "energy_kwh_per_sqm": building.original_properties.energy.energy_performance_kwh_per_sqm,
                        "construction_year": building.original_properties.building_info.construction_year,
                        "facade_material": ep.envelope.facade_material.value if ep.envelope.facade_material else None,
                        "wwr_average": ep.envelope.window_to_wall_ratio.average if ep.envelope.window_to_wall_ratio else None,
                        "solar_potential_kwh": ep.solar_potential.annual_yield_potential_kwh,
                    },
                }
                features.append(feature)

        geojson = {
            "type": "FeatureCollection",
            "name": brf.brf_name,
            "crs": {
                "type": "name",
                "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"},
            },
            "features": features,
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(geojson, f, indent=2, ensure_ascii=False)

        console.print(f"[green]Exported GeoJSON: {output_path}[/green]")
        return output_path

    def _serialize(self, obj: Any) -> Any:
        """Recursively serialize object for JSON."""
        if isinstance(obj, dict):
            return {k: self._serialize(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._serialize(item) for item in obj]
        elif isinstance(obj, (date, datetime)):
            return obj.isoformat()
        elif hasattr(obj, "value"):  # Enum
            return obj.value
        else:
            return obj


def load_enriched_json(path: Path | str) -> EnrichedBRFProperty:
    """
    Load enriched BRF data from JSON file.

    Args:
        path: Path to JSON file

    Returns:
        EnrichedBRFProperty model
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return EnrichedBRFProperty.model_validate(data)
