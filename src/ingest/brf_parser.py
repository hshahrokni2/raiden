"""
BRF JSON parser.

Parses the input BRF JSON format (from energy declarations) into Pydantic models.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..core.models import (
    BRFProperty,
    BRFBuilding,
    EnrichedBuilding,
    EnrichedBRFProperty,
    EnergyPlusReady,
    EnvelopeData,
    EnrichmentMetadata,
    estimate_u_values,
    estimate_infiltration,
)
from ..core.coordinates import CoordinateTransformer
from datetime import date


class BRFParser:
    """
    Parser for BRF building JSON files.

    Handles:
    - Loading and validating input JSON
    - Converting to Pydantic models
    - Initializing enrichment structure
    """

    def __init__(self):
        self.transformer = CoordinateTransformer()

    def load_json(self, file_path: str | Path) -> dict[str, Any]:
        """Load raw JSON from file."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"BRF file not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def parse(self, file_path: str | Path) -> BRFProperty:
        """
        Parse BRF JSON file into validated Pydantic model.

        Args:
            file_path: Path to BRF JSON file

        Returns:
            Validated BRFProperty model
        """
        data = self.load_json(file_path)
        return BRFProperty.model_validate(data)

    def parse_from_dict(self, data: dict[str, Any]) -> BRFProperty:
        """Parse BRF data from dictionary."""
        return BRFProperty.model_validate(data)

    def initialize_enriched(
        self, brf: BRFProperty, toolkit_version: str = "0.1.0"
    ) -> EnrichedBRFProperty:
        """
        Initialize an enriched BRF property from parsed input.

        Creates the structure for enrichment with default values
        estimated from the original data.

        Args:
            brf: Parsed BRF property
            toolkit_version: Version of toolkit for metadata

        Returns:
            EnrichedBRFProperty ready for enrichment
        """
        enriched_buildings = []

        for building in brf.buildings:
            enriched = self._initialize_enriched_building(
                building, toolkit_version
            )
            enriched_buildings.append(enriched)

        return EnrichedBRFProperty(
            brf_name=brf.brf_name,
            source_file=brf.source_file,
            coordinate_system=brf.coordinate_system,
            original_summary=brf.summary,
            buildings=enriched_buildings,
        )

    def _initialize_enriched_building(
        self, building: BRFBuilding, toolkit_version: str
    ) -> EnrichedBuilding:
        """Initialize enriched building with estimated defaults."""
        props = building.properties
        geom = building.geometry

        # Get coordinates in WGS84 for visualization
        footprint_wgs84 = self.transformer.coords_3d_to_2d_wgs84(
            geom.ground_footprint
        )

        # Get local coordinates for EnergyPlus
        footprint_local = self.transformer.coords_to_local(geom.ground_footprint)

        # Estimate U-values based on construction era
        u_values = estimate_u_values(
            construction_year=props.building_info.construction_year,
            facade_material=props.building_info.building_type or "unknown",
            renovation_year=props.building_info.last_renovation_year,
        )

        # Estimate infiltration
        infiltration = estimate_infiltration(
            construction_year=props.building_info.construction_year,
            renovation_year=props.building_info.last_renovation_year,
        )

        # Calculate floor-to-floor height
        floor_height = props.dimensions.building_height_m / max(
            props.dimensions.floors_above_ground, 1
        )

        # Create EnergyPlus-ready data
        ep_ready = EnergyPlusReady(
            footprint_coords_wgs84=footprint_wgs84,
            footprint_coords_local=footprint_local,
            height_m=props.dimensions.building_height_m,
            num_stories=props.dimensions.floors_above_ground,
            floor_to_floor_height_m=floor_height,
            envelope=EnvelopeData(u_values=u_values),
            infiltration_ach=infiltration,
            # Adjust occupant density based on space usage
            occupant_density_m2_per_person=self._estimate_occupant_density(
                props.space_usage_percent
            ),
        )

        # Create enrichment metadata
        metadata = EnrichmentMetadata(
            enrichment_date=date.today(),
            toolkit_version=toolkit_version,
            data_sources=["energy_declaration"],
        )

        return EnrichedBuilding(
            building_id=building.building_id,
            original_geometry=geom,
            original_properties=props,
            energyplus_ready=ep_ready,
            enrichment_metadata=metadata,
        )

    def _estimate_occupant_density(self, space_usage) -> float:
        """
        Estimate occupant density based on space usage.

        Returns m2 per person (higher = less dense).
        """
        # Weighted average based on typical densities
        densities = {
            "residential": 30,  # m2/person
            "office": 15,
            "grocery": 10,
            "restaurant": 5,
            "industrial": 50,
        }

        total_weight = 0
        weighted_density = 0

        for usage_type, density in densities.items():
            pct = getattr(space_usage, usage_type, 0)
            if pct > 0:
                total_weight += pct
                weighted_density += pct * density

        if total_weight > 0:
            return weighted_density / total_weight
        return 30  # Default residential

    def get_building_bbox_wgs84(
        self, building: BRFBuilding, buffer_m: float = 100
    ) -> tuple[float, float, float, float]:
        """
        Get bounding box for a building in WGS84.

        Returns (min_lon, min_lat, max_lon, max_lat).
        """
        bbox = self.transformer.get_bounding_box_wgs84(
            building.geometry.ground_footprint, buffer_meters=buffer_m
        )
        return bbox.to_tuple()

    def get_property_bbox_wgs84(
        self, brf: BRFProperty, buffer_m: float = 100
    ) -> tuple[float, float, float, float]:
        """
        Get bounding box encompassing all buildings in WGS84.

        Returns (min_lon, min_lat, max_lon, max_lat).
        """
        all_coords = []
        for building in brf.buildings:
            all_coords.extend(building.geometry.ground_footprint)

        bbox = self.transformer.get_bounding_box_wgs84(
            all_coords, buffer_meters=buffer_m
        )
        return bbox.to_tuple()

    def get_building_centroid_wgs84(
        self, building: BRFBuilding
    ) -> tuple[float, float]:
        """Get building centroid in WGS84 (lon, lat)."""
        return self.transformer.calculate_centroid_wgs84(
            building.geometry.ground_footprint
        )

    def to_geojson(self, brf: BRFProperty) -> dict:
        """
        Convert BRF property to GeoJSON FeatureCollection.

        Useful for visualization and data exchange.
        """
        features = []

        for building in brf.buildings:
            # Convert coordinates to WGS84
            wgs84_coords = self.transformer.coords_3d_to_2d_wgs84(
                building.geometry.ground_footprint
            )

            # Ensure closed polygon
            if wgs84_coords[0] != wgs84_coords[-1]:
                wgs84_coords.append(wgs84_coords[0])

            feature = {
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [wgs84_coords],
                },
                "properties": {
                    "building_id": building.building_id,
                    "height": building.geometry.height_meters,
                    "address": building.properties.location.address,
                    "construction_year": building.properties.building_info.construction_year,
                    "energy_class": building.properties.energy.energy_class.value,
                    "heated_area_sqm": building.properties.dimensions.heated_area_sqm,
                    "floors": building.properties.dimensions.floors_above_ground,
                },
            }
            features.append(feature)

        return {
            "type": "FeatureCollection",
            "properties": {
                "brf_name": brf.brf_name,
                "total_buildings": brf.summary.total_buildings,
            },
            "features": features,
        }

    def save_geojson(self, brf: BRFProperty, output_path: str | Path) -> None:
        """Save BRF property as GeoJSON file."""
        geojson = self.to_geojson(brf)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(geojson, f, indent=2, ensure_ascii=False)
