"""
Shading and Solar Potential Analysis.

Analyzes:
- Neighbor building shading impact
- Tree/vegetation shading
- Rooftop solar PV potential with shading factors
- Remaining available roof space
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from rich.console import Console

console = Console()


@dataclass
class ShadingAnalysis:
    """Results from shading analysis."""

    # Neighbor buildings
    neighbor_count: int = 0
    max_neighbor_height_m: float = 0
    avg_neighbor_height_m: float = 0
    closest_neighbor_distance_m: float = float("inf")

    # Shading factors (0-1, where 1 = no shading)
    north_shading_factor: float = 1.0
    south_shading_factor: float = 1.0
    east_shading_factor: float = 1.0
    west_shading_factor: float = 1.0

    # Trees
    tree_count_nearby: int = 0
    tree_shading_factor: float = 1.0

    # Overall
    overall_shading_factor: float = 1.0
    shading_loss_pct: float = 0.0


@dataclass
class SolarPotential:
    """Solar PV potential analysis results."""

    # Roof characteristics
    total_roof_area_sqm: float = 0
    suitable_area_sqm: float = 0
    available_area_sqm: float = 0  # After existing PV

    # Existing PV
    existing_pv_sqm: float = 0
    existing_pv_kwp: float = 0
    existing_pv_kwh_yr: float = 0

    # Remaining potential
    remaining_capacity_kwp: float = 0
    remaining_annual_kwh: float = 0

    # With shading
    shading_loss_pct: float = 0
    effective_annual_kwh: float = 0

    # Financial (rough estimates)
    estimated_install_cost_sek: float = 0
    payback_years: float = 0

    source: str = "analysis"


def calculate_distance_m(
    lon1: float, lat1: float, lon2: float, lat2: float
) -> float:
    """Calculate distance between two WGS84 points in meters."""
    # Haversine formula
    R = 6371000  # Earth radius in meters

    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    delta_lat = math.radians(lat2 - lat1)
    delta_lon = math.radians(lon2 - lon1)

    a = (
        math.sin(delta_lat / 2) ** 2
        + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(delta_lon / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return R * c


def calculate_bearing(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """Calculate bearing from point 1 to point 2 (0-360, 0=North)."""
    delta_lon = math.radians(lon2 - lon1)
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)

    x = math.sin(delta_lon) * math.cos(lat2_rad)
    y = math.cos(lat1_rad) * math.sin(lat2_rad) - math.sin(lat1_rad) * math.cos(
        lat2_rad
    ) * math.cos(delta_lon)

    bearing = math.degrees(math.atan2(x, y))
    return (bearing + 360) % 360


def analyze_neighbor_shading(
    building_center: tuple[float, float],  # (lon, lat)
    building_height_m: float,
    neighbor_buildings: list[dict[str, Any]],
    max_distance_m: float = 100,
) -> ShadingAnalysis:
    """
    Analyze shading from neighboring buildings.

    Args:
        building_center: Target building centroid (lon, lat)
        building_height_m: Target building height
        neighbor_buildings: List of OSM building dicts with coordinates/height
        max_distance_m: Maximum distance to consider for shading

    Returns:
        ShadingAnalysis with shading factors per direction
    """
    result = ShadingAnalysis()

    if not neighbor_buildings:
        return result

    center_lon, center_lat = building_center

    # Direction buckets for shading analysis
    direction_heights: dict[str, list[tuple[float, float]]] = {
        "N": [],  # (height, distance) pairs
        "S": [],
        "E": [],
        "W": [],
    }

    heights = []
    distances = []

    for neighbor in neighbor_buildings:
        coords = neighbor.get("coordinates", [])
        if len(coords) < 3:
            continue

        # Calculate neighbor centroid
        lons = [c[0] for c in coords]
        lats = [c[1] for c in coords]
        neighbor_lon = sum(lons) / len(lons)
        neighbor_lat = sum(lats) / len(lats)

        # Get neighbor height (estimate if not available)
        neighbor_height = neighbor.get("height")
        if neighbor_height is None:
            levels = neighbor.get("levels")
            if levels:
                neighbor_height = levels * 3  # ~3m per floor
            else:
                neighbor_height = 12  # Default 4-story estimate

        # Calculate distance and bearing
        distance = calculate_distance_m(
            center_lon, center_lat, neighbor_lon, neighbor_lat
        )

        if distance > max_distance_m or distance < 1:
            continue

        bearing = calculate_bearing(center_lon, center_lat, neighbor_lon, neighbor_lat)

        result.neighbor_count += 1
        heights.append(neighbor_height)
        distances.append(distance)

        # Assign to direction bucket
        if 315 <= bearing or bearing < 45:
            direction_heights["N"].append((neighbor_height, distance))
        elif 45 <= bearing < 135:
            direction_heights["E"].append((neighbor_height, distance))
        elif 135 <= bearing < 225:
            direction_heights["S"].append((neighbor_height, distance))
        else:
            direction_heights["W"].append((neighbor_height, distance))

    if heights:
        result.max_neighbor_height_m = max(heights)
        result.avg_neighbor_height_m = sum(heights) / len(heights)
        result.closest_neighbor_distance_m = min(distances)

    # Calculate shading factor per direction
    # Higher buildings closer = more shading
    # Solar altitude in Stockholm: ~10° winter, ~55° summer
    solar_altitude_winter = 10  # degrees
    solar_altitude_summer = 55  # degrees

    for direction, neighbors in direction_heights.items():
        if not neighbors:
            factor = 1.0
        else:
            # Calculate worst-case shading
            max_shading = 0
            for height, dist in neighbors:
                # Shade angle from neighbor
                height_diff = height - building_height_m
                if height_diff <= 0:
                    continue

                shade_angle = math.degrees(math.atan2(height_diff, dist))

                # Shading impact depends on direction and season
                if direction == "S":
                    # South shading matters most (sun is south in northern hemisphere)
                    # Winter sun is low, summer sun is high
                    if shade_angle > solar_altitude_winter:
                        shading = min(1.0, (shade_angle - solar_altitude_winter) / 45)
                    else:
                        shading = 0
                else:
                    # East/West matter for morning/afternoon
                    # North matters less for solar (but affects daylight)
                    shading = min(0.5, shade_angle / 90)

                max_shading = max(max_shading, shading)

            factor = 1.0 - max_shading

        if direction == "N":
            result.north_shading_factor = factor
        elif direction == "S":
            result.south_shading_factor = factor
        elif direction == "E":
            result.east_shading_factor = factor
        else:
            result.west_shading_factor = factor

    # Overall shading factor (weighted by solar importance)
    # South is most important for PV
    result.overall_shading_factor = (
        result.north_shading_factor * 0.1
        + result.south_shading_factor * 0.5
        + result.east_shading_factor * 0.2
        + result.west_shading_factor * 0.2
    )
    result.shading_loss_pct = (1 - result.overall_shading_factor) * 100

    return result


def analyze_tree_shading(
    building_center: tuple[float, float],
    building_height_m: float,
    vegetation: list[dict[str, Any]],
    max_distance_m: float = 30,
) -> float:
    """
    Analyze shading from nearby trees.

    Returns shading factor (0-1, where 1 = no shading).
    """
    if not vegetation:
        return 1.0

    center_lon, center_lat = building_center
    tree_count = 0
    shading_impact = 0

    for veg in vegetation:
        if veg.get("type") == "tree":
            tree_lon, tree_lat = veg.get("coordinates", (0, 0))
            distance = calculate_distance_m(center_lon, center_lat, tree_lon, tree_lat)

            if distance > max_distance_m:
                continue

            tree_count += 1
            tree_height = veg.get("height") or 10  # Default 10m tree

            # Trees within 20m can significantly impact facade windows
            if distance < 20 and tree_height > building_height_m * 0.5:
                shading_impact += 0.05  # Each close tall tree adds ~5% shading

    # Cap total tree shading at 30%
    return max(0.7, 1.0 - shading_impact)


def calculate_solar_potential(
    roof_area_sqm: float,
    existing_pv_sqm: float = 0,
    existing_pv_kwh_yr: float = 0,
    shading_factor: float = 1.0,
    roof_orientation: str = "flat",  # flat, south, east, west
    roof_tilt_deg: float = 0,
) -> SolarPotential:
    """
    Calculate detailed solar PV potential for a roof.

    Args:
        roof_area_sqm: Total roof footprint area
        existing_pv_sqm: Area already covered by PV
        existing_pv_kwh_yr: Existing PV production
        shading_factor: 0-1 factor from shading analysis
        roof_orientation: Main roof orientation
        roof_tilt_deg: Roof tilt angle

    Returns:
        SolarPotential with detailed analysis
    """
    result = SolarPotential()
    result.total_roof_area_sqm = roof_area_sqm
    result.existing_pv_sqm = existing_pv_sqm
    result.existing_pv_kwh_yr = existing_pv_kwh_yr

    # For flat roofs, ~30% is typically suitable after
    # obstacles, access paths, setbacks
    # For tilted roofs, only the south-facing portion is optimal
    if roof_orientation == "flat":
        suitability = 0.35
    elif roof_orientation == "south":
        suitability = 0.50
    else:
        suitability = 0.25

    result.suitable_area_sqm = roof_area_sqm * suitability

    # Available area after existing PV
    result.available_area_sqm = max(0, result.suitable_area_sqm - existing_pv_sqm)

    # Panel density: ~200W/m² for modern panels
    panel_density_wp_sqm = 200
    result.remaining_capacity_kwp = result.available_area_sqm * panel_density_wp_sqm / 1000

    # Existing PV capacity (estimate from area if not given)
    if existing_pv_sqm > 0:
        result.existing_pv_kwp = existing_pv_sqm * panel_density_wp_sqm / 1000

    # Annual yield: Stockholm ~900-950 kWh/kWp
    # Adjust for orientation and tilt
    base_yield_kwh_kwp = 900

    if roof_orientation == "south" and 30 <= roof_tilt_deg <= 45:
        orientation_factor = 1.1  # Optimal orientation
    elif roof_orientation == "flat":
        orientation_factor = 0.95  # Slightly sub-optimal
    elif roof_orientation in ["east", "west"]:
        orientation_factor = 0.85
    else:
        orientation_factor = 0.80

    effective_yield = base_yield_kwh_kwp * orientation_factor

    # Apply shading factor
    result.shading_loss_pct = (1 - shading_factor) * 100
    effective_yield *= shading_factor

    result.remaining_annual_kwh = result.remaining_capacity_kwp * effective_yield
    result.effective_annual_kwh = (
        result.remaining_annual_kwh + result.existing_pv_kwh_yr
    )

    # Financial estimates (rough 2024 Swedish prices)
    # Installation: ~15,000 SEK/kWp for residential, ~10,000 for commercial
    install_cost_per_kwp = 12000  # SEK
    result.estimated_install_cost_sek = result.remaining_capacity_kwp * install_cost_per_kwp

    # Electricity price: ~1.5 SEK/kWh average (varies significantly)
    electricity_price = 1.5  # SEK/kWh
    annual_savings = result.remaining_annual_kwh * electricity_price

    if annual_savings > 0:
        result.payback_years = result.estimated_install_cost_sek / annual_savings
    else:
        result.payback_years = float("inf")

    result.source = "analysis"

    return result


def analyze_building_solar_and_shading(
    building_coords: list[tuple[float, float]],  # WGS84
    building_height_m: float,
    roof_area_sqm: float,
    existing_pv_sqm: float = 0,
    existing_pv_kwh_yr: float = 0,
    osm_fetcher=None,
) -> dict[str, Any]:
    """
    Complete solar and shading analysis for a building.

    Fetches OSM data for neighbors and trees, then analyzes
    shading impact and solar potential.

    Args:
        building_coords: Building footprint in WGS84
        building_height_m: Building height
        roof_area_sqm: Roof area
        existing_pv_sqm: Existing PV area
        existing_pv_kwh_yr: Existing PV production
        osm_fetcher: Optional OSMFetcher instance

    Returns:
        Dict with shading analysis and solar potential
    """
    # Calculate centroid
    lons = [c[0] for c in building_coords]
    lats = [c[1] for c in building_coords]
    center_lon = sum(lons) / len(lons)
    center_lat = sum(lats) / len(lats)

    shading = ShadingAnalysis()
    neighbors = []
    trees = []

    # Fetch OSM data if fetcher provided
    if osm_fetcher is not None:
        try:
            console.print("[cyan]Fetching nearby buildings from OSM...[/cyan]")
            neighbors = osm_fetcher.get_nearby_buildings(
                center_lon, center_lat, radius_m=100
            )
            console.print(f"  Found {len(neighbors)} nearby buildings")

            # Get trees
            bbox = (
                center_lon - 0.001,
                center_lat - 0.001,
                center_lon + 0.001,
                center_lat + 0.001,
            )
            trees = osm_fetcher.get_trees_in_bbox(*bbox)
            console.print(f"  Found {len(trees)} vegetation features")

        except Exception as e:
            console.print(f"[yellow]OSM fetch failed: {e}[/yellow]")

    # Analyze shading from neighbors
    shading = analyze_neighbor_shading(
        building_center=(center_lon, center_lat),
        building_height_m=building_height_m,
        neighbor_buildings=neighbors,
    )

    # Add tree shading
    tree_factor = analyze_tree_shading(
        building_center=(center_lon, center_lat),
        building_height_m=building_height_m,
        vegetation=trees,
    )
    shading.tree_count_nearby = len([v for v in trees if v.get("type") == "tree"])
    shading.tree_shading_factor = tree_factor

    # Combine shading factors
    combined_shading = shading.overall_shading_factor * tree_factor
    shading.overall_shading_factor = combined_shading
    shading.shading_loss_pct = (1 - combined_shading) * 100

    # Calculate solar potential with shading
    solar = calculate_solar_potential(
        roof_area_sqm=roof_area_sqm,
        existing_pv_sqm=existing_pv_sqm,
        existing_pv_kwh_yr=existing_pv_kwh_yr,
        shading_factor=combined_shading,
    )

    return {
        "shading": shading,
        "solar": solar,
        "neighbors_analyzed": len(neighbors),
        "trees_analyzed": shading.tree_count_nearby,
    }
