#!/usr/bin/env python3
"""
End-to-end test: Extract building data → Generate IDF → (Optionally run simulation)

Tests the complete pipeline from coordinates to EnergyPlus model.

Usage:
    python scripts/test_end_to_end.py
    python scripts/test_end_to_end.py --simulate  # Also run EnergyPlus
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ingest.building_extractor import BuildingDataExtractor
from src.geometry.building_geometry import BuildingGeometryCalculator
from src.baseline.archetypes import ArchetypeMatcher, BuildingType
from src.baseline.generator import BaselineGenerator


def test_full_pipeline(run_simulation: bool = False):
    """Test the full extraction → IDF generation pipeline."""

    print("=" * 70)
    print("END-TO-END TEST: Extract → Generate IDF")
    print("=" * 70)

    # Test building: Sjostaden 2
    lat = 59.302
    lon = 18.104
    construction_year = 2003
    address = "Aktergatan 11, Hammarby Sjöstad"

    print(f"\nBuilding: {address}")
    print(f"Coordinates: {lat}, {lon}")
    print(f"Construction year: {construction_year}")

    # Step 1: Extract building data
    print("\n" + "-" * 70)
    print("STEP 1: Extract building data from maps")
    print("-" * 70)

    extractor = BuildingDataExtractor(use_ai=False)

    profile = extractor.extract_from_coordinates(
        lat=lat,
        lon=lon,
        construction_year=construction_year,
        address=address,
        fetch_images=False,
        run_ai_analysis=False,
    )

    print(f"\nExtracted profile:")
    print(f"  GFA: {profile.geometry.gross_floor_area_m2:,.0f} m²")
    print(f"  Floors: {profile.geometry.floors}")
    print(f"  Height: {profile.geometry.height_m:.1f} m")
    print(f"  Wall area: {profile.geometry.total_wall_area:,.0f} m²")
    print(f"  Window area: {profile.geometry.total_window_area:,.0f} m²")
    print(f"  WWR: {profile.envelope.wwr_average:.1%}")
    print(f"  Archetype: {profile.archetype_name}")

    # Step 2: Get geometry and archetype for IDF generation
    print("\n" + "-" * 70)
    print("STEP 2: Calculate building geometry")
    print("-" * 70)

    # Re-calculate geometry with the geometry calculator
    # (In a real pipeline, this would be stored in the profile)
    calculator = BuildingGeometryCalculator()

    wwr_by_orientation = {
        'N': profile.envelope.wwr_north,
        'S': profile.envelope.wwr_south,
        'E': profile.envelope.wwr_east,
        'W': profile.envelope.wwr_west,
    }

    geometry = calculator.calculate(
        footprint_coords=profile.geometry.footprint_coords_wgs84,
        height_m=profile.geometry.height_m,
        floors=profile.geometry.floors,
        wwr_by_orientation=wwr_by_orientation,
    )

    print(f"\nGeometry calculated:")
    print(f"  Footprint: {geometry.footprint_area_m2:,.0f} m²")
    print(f"  Perimeter: {geometry.perimeter_m:.1f} m")
    print(f"  Volume: {geometry.volume_m3:,.0f} m³")

    # Step 3: Generate IDF
    print("\n" + "-" * 70)
    print("STEP 3: Generate EnergyPlus IDF")
    print("-" * 70)

    output_dir = Path(__file__).parent.parent / "output_generated"
    output_dir.mkdir(parents=True, exist_ok=True)

    generator = BaselineGenerator()

    model = generator.generate(
        geometry=geometry,
        archetype=profile.archetype,
        output_dir=output_dir,
        model_name="sjostaden_auto",
        latitude=lat,
        longitude=lon,
    )

    print(f"\nGenerated IDF:")
    print(f"  Path: {model.idf_path}")
    print(f"  Weather file: {model.weather_file}")
    print(f"  Archetype: {model.archetype_used}")
    print(f"  Floor area: {model.floor_area_m2:,.0f} m²")
    print(f"  Predicted heating: {model.predicted_heating_kwh_m2:.1f} kWh/m²/year")

    # Verify IDF was created
    if not model.idf_path.exists():
        print("\n[ERROR] IDF file was not created!")
        return False

    # Count IDF lines and objects
    with open(model.idf_path) as f:
        content = f.read()
        lines = content.split('\n')
        zones = content.count('Zone,')
        surfaces = content.count('BuildingSurface:Detailed,')
        windows = content.count('FenestrationSurface:Detailed,')

    print(f"\nIDF statistics:")
    print(f"  Total lines: {len(lines)}")
    print(f"  Zones: {zones}")
    print(f"  Surfaces: {surfaces}")
    print(f"  Windows: {windows}")

    # Step 4: Optionally run simulation
    if run_simulation:
        print("\n" + "-" * 70)
        print("STEP 4: Run EnergyPlus simulation")
        print("-" * 70)

        weather_dir = Path(__file__).parent.parent / "data/weather"
        weather_file = weather_dir / model.weather_file

        if not weather_file.exists():
            print(f"\n[WARNING] Weather file not found: {weather_file}")
            print("Download from: https://energyplus.net/weather")
            print("Skipping simulation...")
        else:
            # Run EnergyPlus
            sim_output = output_dir / "simulation"
            sim_output.mkdir(exist_ok=True)

            cmd = [
                "energyplus",
                "-w", str(weather_file),
                "-d", str(sim_output),
                str(model.idf_path)
            ]

            print(f"\nRunning: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode == 0:
                print("\n[SUCCESS] Simulation completed!")

                # Check for results
                results_file = sim_output / "eplustbl.csv"
                if results_file.exists():
                    print(f"Results: {results_file}")
            else:
                print(f"\n[ERROR] Simulation failed (exit code {result.returncode})")
                if result.stderr:
                    print(f"Error: {result.stderr[:500]}")

    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print(f"✓ Building data extracted from coordinates")
    print(f"✓ Geometry calculated ({geometry.floors} floors, {geometry.gross_floor_area_m2:,.0f} m²)")
    print(f"✓ IDF generated ({model.idf_path.name})")
    print(f"✓ Predicted heating: {model.predicted_heating_kwh_m2:.1f} kWh/m²/year")

    if profile.solar_potential:
        print(f"✓ Solar potential: {profile.solar_potential.new_capacity_kwp:.1f} kWp")

    print("\n" + "=" * 70)
    print("END-TO-END TEST COMPLETE")
    print("=" * 70)

    return True


def test_from_geojson():
    """Test IDF generation from known GeoJSON footprint."""

    print("\n" + "=" * 70)
    print("TEST: Generate IDF from known footprint")
    print("=" * 70)

    # Load Sjostaden GeoJSON
    geojson_path = Path(__file__).parent.parent / "examples/sjostaden_2/BRF_Sjostaden_2.geojson"
    enriched_path = Path(__file__).parent.parent / "examples/sjostaden_2/BRF_Sjostaden_2_enriched.json"

    if not geojson_path.exists():
        print(f"[SKIP] GeoJSON not found: {geojson_path}")
        return True

    with open(geojson_path) as f:
        geojson = json.load(f)

    with open(enriched_path) as f:
        enriched = json.load(f)

    # Get first building
    feature = geojson["features"][0]
    coords = feature["geometry"]["coordinates"][0]
    footprint = [(c[0], c[1]) for c in coords]

    # Building parameters
    floors = 7
    height_m = 21.0
    construction_year = 2003

    # WWR from enriched data
    building_envelope = enriched["buildings"][0].get("envelope", {})
    wwr_data = building_envelope.get("wwr", {})
    wwr_by_orientation = {
        'N': wwr_data.get("north", 0.216),
        'S': wwr_data.get("south", 0.324),
        'E': wwr_data.get("east", 0.27),
        'W': wwr_data.get("west", 0.27),
    }

    print(f"\nBuilding from GeoJSON:")
    print(f"  Footprint vertices: {len(footprint)}")
    print(f"  Floors: {floors}")
    print(f"  Height: {height_m} m")
    print(f"  Construction year: {construction_year}")

    # Calculate geometry
    calculator = BuildingGeometryCalculator()
    geometry = calculator.calculate(
        footprint_coords=footprint,
        height_m=height_m,
        floors=floors,
        wwr_by_orientation=wwr_by_orientation,
    )

    print(f"\nGeometry:")
    print(f"  Footprint: {geometry.footprint_area_m2:,.0f} m²")
    print(f"  GFA: {geometry.gross_floor_area_m2:,.0f} m²")
    print(f"  Wall area: {geometry.total_wall_area_m2:,.0f} m²")
    print(f"  Window area: {geometry.total_window_area_m2:,.0f} m²")

    # Match archetype
    matcher = ArchetypeMatcher()
    archetype = matcher.match(
        construction_year=construction_year,
        building_type=BuildingType.MULTI_FAMILY,
        facade_material="brick",
    )

    print(f"\nArchetype: {archetype.name}")
    print(f"  Wall U: {archetype.envelope.wall_u_value} W/m²K")
    print(f"  Window U: {archetype.envelope.window_u_value} W/m²K")
    print(f"  Heat recovery: {archetype.hvac.heat_recovery_efficiency:.0%}")

    # Generate IDF
    output_dir = Path(__file__).parent.parent / "output_generated"
    generator = BaselineGenerator()

    model = generator.generate(
        geometry=geometry,
        archetype=archetype,
        output_dir=output_dir,
        model_name="sjostaden_geojson",
        latitude=59.302,
        longitude=18.104,
    )

    print(f"\nGenerated IDF: {model.idf_path}")
    print(f"Predicted heating: {model.predicted_heating_kwh_m2:.1f} kWh/m²/year")

    print("\n✓ GeoJSON test complete")
    return True


def main():
    parser = argparse.ArgumentParser(description="End-to-end pipeline test")
    parser.add_argument("--simulate", action="store_true", help="Run EnergyPlus simulation")
    args = parser.parse_args()

    # Test 1: Full pipeline from coordinates
    success1 = test_full_pipeline(run_simulation=args.simulate)

    # Test 2: From known GeoJSON
    success2 = test_from_geojson()

    sys.exit(0 if (success1 and success2) else 1)


if __name__ == "__main__":
    main()
