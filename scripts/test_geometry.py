#!/usr/bin/env python3
"""
Test BuildingGeometryCalculator with Sjostaden building data.

Usage:
    python scripts/test_geometry.py
"""

import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.geometry import BuildingGeometryCalculator, calculate_building_geometry


def test_sjostaden():
    """Test with real Sjostaden building data."""

    # Load Sjostaden GeoJSON
    geojson_path = Path(__file__).parent.parent / "examples/sjostaden_2/BRF_Sjostaden_2.geojson"
    enriched_path = Path(__file__).parent.parent / "examples/sjostaden_2/BRF_Sjostaden_2_enriched.json"

    with open(geojson_path) as f:
        geojson = json.load(f)

    with open(enriched_path) as f:
        enriched = json.load(f)

    # Get building parameters from enriched data
    summary = enriched.get("original_summary", {})
    construction_year = summary.get("construction_year", 2003)
    total_heated_area = summary.get("total_heated_area_sqm", 15350)
    num_buildings = summary.get("total_buildings", 2)

    # Estimated per-building values (for a 7-floor building)
    floors = 7
    floor_height = 3.0
    height_m = floors * floor_height

    print("=" * 70)
    print("BUILDING GEOMETRY CALCULATOR TEST - BRF Sjostaden 2")
    print("=" * 70)
    print(f"\nConstruction year: {construction_year}")
    print(f"Total heated area: {total_heated_area} m² ({num_buildings} buildings)")
    print(f"Assumed floors: {floors}")
    print(f"Assumed height: {height_m} m")

    # WWR from enriched data
    building_1_envelope = enriched["buildings"][0].get("envelope", {})
    wwr_data = building_1_envelope.get("wwr", {})
    wwr_by_orientation = {
        'N': wwr_data.get("north", 0.216),
        'S': wwr_data.get("south", 0.324),
        'E': wwr_data.get("east", 0.27),
        'W': wwr_data.get("west", 0.27),
    }

    print(f"\nWWR by orientation (from enriched data):")
    for direction, wwr in wwr_by_orientation.items():
        print(f"  {direction}: {wwr:.1%}")

    # Create calculator
    calculator = BuildingGeometryCalculator()

    # Process each building
    for i, feature in enumerate(geojson["features"]):
        building_id = feature["properties"].get("building_id", i + 1)
        print(f"\n{'=' * 70}")
        print(f"BUILDING {building_id}")
        print("=" * 70)

        # Calculate geometry from GeoJSON
        geometry = calculator.calculate_from_geojson(
            geojson=feature,
            height_m=height_m,
            floors=floors,
            wwr_by_orientation=wwr_by_orientation,
        )

        # Print results
        print(f"\n📐 DIMENSIONS:")
        print(f"  Footprint area:     {geometry.footprint_area_m2:,.1f} m²")
        print(f"  Gross floor area:   {geometry.gross_floor_area_m2:,.1f} m² (Atemp)")
        print(f"  Building height:    {geometry.height_m:.1f} m")
        print(f"  Floor height:       {geometry.floor_height_m:.1f} m")
        print(f"  Perimeter:          {geometry.perimeter_m:.1f} m")
        print(f"  Volume:             {geometry.volume_m3:,.1f} m³")

        print(f"\n🧱 WALL AREAS BY ORIENTATION:")
        for direction in ['N', 'E', 'S', 'W']:
            facade = geometry.facades[direction]
            print(f"  {direction}: {facade.wall_area_m2:>8.1f} m² wall, "
                  f"{facade.window_area_m2:>6.1f} m² window "
                  f"(WWR: {facade.wwr:.1%}, {facade.segment_count} segments, "
                  f"azimuth: {facade.azimuth_deg:.0f}°)")

        print(f"\n📊 TOTALS:")
        print(f"  Total wall area:    {geometry.total_wall_area_m2:,.1f} m²")
        print(f"  Total window area:  {geometry.total_window_area_m2:,.1f} m²")
        print(f"  Average WWR:        {geometry.average_wwr:.1%}")
        print(f"  Roof area:          {geometry.roof.total_area_m2:,.1f} m²")
        print(f"  Ground floor:       {geometry.ground_floor_area_m2:,.1f} m²")
        print(f"  Total envelope:     {geometry.total_envelope_area_m2:,.1f} m²")

        print(f"\n🔌 PV POTENTIAL:")
        print(f"  Roof type:          flat")
        print(f"  Available PV area:  {geometry.roof.available_pv_area_m2:,.1f} m² (70% of roof)")

        # Wall segment details
        print(f"\n📏 WALL SEGMENTS ({len(geometry.wall_segments)} total):")
        for j, seg in enumerate(geometry.wall_segments[:10]):  # First 10
            print(f"  [{j+1:2d}] {seg.orientation}: {seg.length_m:6.2f} m @ {seg.azimuth_deg:5.1f}°")
        if len(geometry.wall_segments) > 10:
            print(f"  ... and {len(geometry.wall_segments) - 10} more segments")

    print("\n" + "=" * 70)
    print("✅ TEST COMPLETE")
    print("=" * 70)

    return True


def test_simple_rectangle():
    """Test with a simple rectangular building for validation."""

    print("\n" + "=" * 70)
    print("VALIDATION TEST - Simple 20m x 10m Rectangle")
    print("=" * 70)

    # Create a simple 20m x 10m rectangle aligned with cardinal directions
    # Centered at Stockholm (lat 59.3293)
    center_lat = 59.3293
    center_lon = 18.0686

    # At latitude 59.3°:
    # 1° latitude = 111,320 m
    # 1° longitude = 111,320 × cos(59.3°) = 111,320 × 0.511 = 56,885 m
    #
    # For a 20m (E-W) × 10m (N-S) rectangle:
    # Half-width (E-W): 10m / 56885 = 0.0001758°
    # Half-height (N-S): 5m / 111320 = 0.0000449°

    import math
    m_per_deg_lat = 111320
    m_per_deg_lon = 111320 * math.cos(math.radians(center_lat))

    half_width_deg = 10.0 / m_per_deg_lon   # 10m half-width (E-W)
    half_height_deg = 5.0 / m_per_deg_lat   # 5m half-height (N-S)

    # Rectangle vertices (counter-clockwise from SW)
    coords = [
        (center_lon - half_width_deg, center_lat - half_height_deg),  # SW
        (center_lon + half_width_deg, center_lat - half_height_deg),  # SE
        (center_lon + half_width_deg, center_lat + half_height_deg),  # NE
        (center_lon - half_width_deg, center_lat + half_height_deg),  # NW
        (center_lon - half_width_deg, center_lat - half_height_deg),  # Close
    ]

    # Calculate with uniform WWR
    geometry = calculate_building_geometry(
        footprint_coords=coords,
        height_m=10.0,
        floors=3,
        wwr_by_orientation={'N': 0.20, 'S': 0.20, 'E': 0.20, 'W': 0.20},
    )

    print(f"\nExpected:")
    print(f"  Footprint: ~200 m² (20m × 10m)")
    print(f"  Perimeter: ~60 m (2×20 + 2×10)")
    print(f"  North/South walls: ~200 m² each (20m × 10m height)")
    print(f"  East/West walls: ~100 m² each (10m × 10m height)")

    print(f"\nCalculated:")
    print(f"  Footprint: {geometry.footprint_area_m2:.1f} m²")
    print(f"  Perimeter: {geometry.perimeter_m:.1f} m")

    for direction in ['N', 'S', 'E', 'W']:
        facade = geometry.facades[direction]
        print(f"  {direction} wall: {facade.wall_area_m2:.1f} m² ({facade.length_m:.1f}m × 10m)")

    # Validate (within 5% tolerance for small buildings)
    errors = []
    if abs(geometry.footprint_area_m2 - 200) > 20:  # Within 10%
        errors.append(f"Footprint area: expected ~200 m², got {geometry.footprint_area_m2:.1f}")
    if abs(geometry.perimeter_m - 60) > 6:
        errors.append(f"Perimeter: expected ~60 m, got {geometry.perimeter_m:.1f}")

    if errors:
        print(f"\n❌ VALIDATION ERRORS:")
        for e in errors:
            print(f"  - {e}")
        return False
    else:
        print(f"\n✅ VALIDATION PASSED (within 10% tolerance)")
        return True


if __name__ == "__main__":
    success1 = test_simple_rectangle()
    success2 = test_sjostaden()

    sys.exit(0 if (success1 and success2) else 1)
