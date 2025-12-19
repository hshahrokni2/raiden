#!/usr/bin/env python3
"""
Test BuildingDataExtractor with Sjostaden building.

Tests the full pipeline:
1. OSM/Overture footprint fetching
2. Geometry calculation
3. Archetype matching
4. (Optional) AI facade analysis

Usage:
    python scripts/test_extraction.py
    python scripts/test_extraction.py --with-images  # Include Mapillary
"""

import argparse
import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ingest.building_extractor import BuildingDataExtractor, extract_building


def test_sjostaden_from_coordinates():
    """Test extraction using Sjostaden coordinates."""

    print("=" * 70)
    print("BUILDING DATA EXTRACTION TEST - Sjostaden (from coordinates)")
    print("=" * 70)

    # Sjostaden 2 building coordinates (from enriched JSON)
    lat = 59.302
    lon = 18.104
    construction_year = 2003

    print(f"\nInput:")
    print(f"  Coordinates: {lat}, {lon}")
    print(f"  Construction year: {construction_year}")

    # Create extractor (without AI/images for fast test)
    extractor = BuildingDataExtractor(use_ai=False)

    try:
        profile = extractor.extract_from_coordinates(
            lat=lat,
            lon=lon,
            construction_year=construction_year,
            address="Aktergatan 11, Hammarby Sjöstad",
            fetch_images=False,
            run_ai_analysis=False,
        )

        print_profile(profile)
        return profile

    except Exception as e:
        print(f"\n[ERROR] Extraction failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_sjostaden_from_known_footprint():
    """Test extraction using known footprint from GeoJSON."""

    print("\n" + "=" * 70)
    print("BUILDING DATA EXTRACTION TEST - Sjostaden (from known footprint)")
    print("=" * 70)

    # Load known footprint
    geojson_path = Path(__file__).parent.parent / "examples/sjostaden_2/BRF_Sjostaden_2.geojson"

    with open(geojson_path) as f:
        geojson = json.load(f)

    feature = geojson["features"][0]  # First building
    coords = feature["geometry"]["coordinates"][0]
    footprint = [(c[0], c[1]) for c in coords]

    # Calculate centroid
    lons = [c[0] for c in footprint]
    lats = [c[1] for c in footprint]
    center_lon = sum(lons) / len(lons)
    center_lat = sum(lats) / len(lats)

    print(f"\nInput:")
    print(f"  Footprint vertices: {len(footprint)}")
    print(f"  Center: {center_lat:.6f}, {center_lon:.6f}")
    print(f"  Construction year: 2003")

    # Create extractor
    extractor = BuildingDataExtractor(use_ai=False)

    # We need to use extract_from_coordinates but inject the footprint
    # For now, let's test the geometry calculator directly
    from src.geometry.building_geometry import BuildingGeometryCalculator

    calculator = BuildingGeometryCalculator()

    # WWR from enriched data
    wwr_by_orientation = {
        'N': 0.216,
        'S': 0.324,
        'E': 0.27,
        'W': 0.27,
    }

    geometry = calculator.calculate(
        footprint_coords=footprint,
        height_m=21.0,
        floors=7,
        wwr_by_orientation=wwr_by_orientation,
    )

    print(f"\nGeometry Results:")
    print(f"  Footprint area: {geometry.footprint_area_m2:.1f} m2")
    print(f"  Gross floor area: {geometry.gross_floor_area_m2:.1f} m2")
    print(f"  Perimeter: {geometry.perimeter_m:.1f} m")
    print(f"  Total wall area: {geometry.total_wall_area_m2:.1f} m2")
    print(f"  Total window area: {geometry.total_window_area_m2:.1f} m2")
    print(f"  Average WWR: {geometry.average_wwr:.1%}")

    print(f"\nWall Areas by Orientation:")
    for d in ['N', 'E', 'S', 'W']:
        f = geometry.facades[d]
        print(f"  {d}: {f.wall_area_m2:>8.1f} m2 wall, {f.window_area_m2:>6.1f} m2 window")

    # Match archetype
    from src.baseline.archetypes import ArchetypeMatcher, BuildingType

    matcher = ArchetypeMatcher()
    archetype = matcher.match(
        construction_year=2003,
        building_type=BuildingType.MULTI_FAMILY,
        facade_material="brick"
    )

    print(f"\nArchetype Match:")
    print(f"  Name: {archetype.name}")
    print(f"  Wall U-value: {archetype.envelope.wall_u_value} W/m2K")
    print(f"  Window U-value: {archetype.envelope.window_u_value} W/m2K")
    print(f"  Heat recovery: {archetype.hvac.heat_recovery_efficiency:.0%}")

    return geometry


def test_with_mapillary():
    """Test extraction with Mapillary image fetching."""

    print("\n" + "=" * 70)
    print("BUILDING DATA EXTRACTION TEST - With Mapillary Images")
    print("=" * 70)

    lat = 59.302
    lon = 18.104
    construction_year = 2003

    extractor = BuildingDataExtractor(use_ai=True, ai_backend="opencv")

    try:
        profile = extractor.extract_from_coordinates(
            lat=lat,
            lon=lon,
            construction_year=construction_year,
            address="Aktergatan 11, Hammarby Sjöstad",
            fetch_images=True,
            run_ai_analysis=True,
        )

        print_profile(profile)
        return profile

    except Exception as e:
        print(f"\n[ERROR] Extraction failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def print_profile(profile):
    """Pretty-print a BuildingProfile."""

    print(f"\n{'=' * 70}")
    print("EXTRACTED BUILDING PROFILE")
    print("=" * 70)

    print(f"\nIdentification:")
    print(f"  ID: {profile.building_id}")
    print(f"  Address: {profile.address}")
    print(f"  Coordinates: {profile.latitude:.6f}, {profile.longitude:.6f}")
    print(f"  Construction year: {profile.construction_year}")

    print(f"\nGeometry:")
    print(f"  Footprint: {profile.geometry.footprint_area_m2:.1f} m2")
    print(f"  GFA (Atemp): {profile.geometry.gross_floor_area_m2:.1f} m2")
    print(f"  Height: {profile.geometry.height_m:.1f} m")
    print(f"  Floors: {profile.geometry.floors}")
    print(f"  Perimeter: {profile.geometry.perimeter_m:.1f} m")
    print(f"  Volume: {profile.geometry.volume_m3:.1f} m3")

    print(f"\nWall Areas (m2):")
    print(f"  North: {profile.geometry.wall_area_north:.1f}")
    print(f"  South: {profile.geometry.wall_area_south:.1f}")
    print(f"  East: {profile.geometry.wall_area_east:.1f}")
    print(f"  West: {profile.geometry.wall_area_west:.1f}")
    print(f"  Total: {profile.geometry.total_wall_area:.1f}")

    print(f"\nWindow Areas (m2):")
    print(f"  North: {profile.geometry.window_area_north:.1f}")
    print(f"  South: {profile.geometry.window_area_south:.1f}")
    print(f"  East: {profile.geometry.window_area_east:.1f}")
    print(f"  West: {profile.geometry.window_area_west:.1f}")
    print(f"  Total: {profile.geometry.total_window_area:.1f}")

    print(f"\nEnvelope:")
    print(f"  Wall U-value: {profile.envelope.wall_u} W/m2K")
    print(f"  Roof U-value: {profile.envelope.roof_u} W/m2K")
    print(f"  Window U-value: {profile.envelope.window_u} W/m2K")
    print(f"  Infiltration: {profile.envelope.infiltration_ach} ACH")
    print(f"  Facade material: {profile.envelope.facade_material} "
          f"(confidence: {profile.envelope.facade_material_confidence:.1%})")

    print(f"\nWWR:")
    print(f"  North: {profile.envelope.wwr_north:.1%}")
    print(f"  South: {profile.envelope.wwr_south:.1%}")
    print(f"  East: {profile.envelope.wwr_east:.1%}")
    print(f"  West: {profile.envelope.wwr_west:.1%}")
    print(f"  Average: {profile.envelope.wwr_average:.1%}")

    print(f"\nHVAC:")
    print(f"  Heating: {profile.hvac.heating_system}")
    print(f"  Ventilation: {profile.hvac.ventilation_type}")
    print(f"  Heat recovery: {profile.hvac.heat_recovery_efficiency:.0%}")

    if profile.solar_potential:
        sp = profile.solar_potential
        print(f"\nSolar PV Potential:")
        print(f"  Roof type: {sp.roof_type}")
        print(f"  Roof area: {sp.roof_area_total_m2:.1f} m2")
        print(f"  Available area: {sp.net_available_m2:.1f} m2 ({sp.net_available_m2/sp.roof_area_total_m2:.0%} usable)")
        print(f"  Obstruction area: {sp.obstruction_area_m2:.1f} m2")
        print(f"  Existing PV: {sp.existing_pv_kwp:.1f} kWp")
        print(f"  New capacity: {sp.new_capacity_kwp:.1f} kWp")
        print(f"  Annual yield: {sp.annual_yield_kwh_per_kwp:.0f} kWh/kWp")
        print(f"  Annual generation: {sp.annual_generation_kwh:,.0f} kWh/year")
        print(f"  Shading loss: {sp.shading_loss_pct:.1%}")
        print(f"  Install cost: {sp.install_cost_sek:,.0f} SEK")
        print(f"  Payback: {sp.payback_years:.1f} years")

    print(f"\nArchetype: {profile.archetype_name}")
    print(f"Overall confidence: {profile.overall_confidence:.1%}")

    if profile.extraction_notes:
        print(f"\nNotes:")
        for note in profile.extraction_notes:
            print(f"  - {note}")


def main():
    parser = argparse.ArgumentParser(description="Test building data extraction")
    parser.add_argument("--with-images", action="store_true", help="Include Mapillary image fetching")
    args = parser.parse_args()

    # Test 1: From known footprint (always works)
    test_sjostaden_from_known_footprint()

    # Test 2: From coordinates (requires OSM)
    profile = test_sjostaden_from_coordinates()

    # Test 3: With Mapillary (optional)
    if args.with_images:
        test_with_mapillary()

    # Export profile to JSON
    if profile:
        output_path = Path(__file__).parent.parent / "examples/sjostaden_2/extracted_profile.json"
        with open(output_path, "w") as f:
            json.dump(profile.to_dict(), f, indent=2)
        print(f"\nProfile exported to: {output_path}")

    print("\n" + "=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
