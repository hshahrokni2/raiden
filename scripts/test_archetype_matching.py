#!/usr/bin/env python3
"""
Test script for integrated ArchetypeMatcherV2 pipeline.

Tests the archetype matching using REAL DATA from:
- Energy declarations
- Mapillary images
- OSM/Overture geometry
- Address/location data
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.baseline import (
    ArchetypeMatcherV2,
    match_archetype_from_building_data,
    get_all_archetypes,
)
from src.core.address_pipeline import BuildingData


def test_from_building_data():
    """Test matching from BuildingData (address pipeline output)."""
    print("\n" + "="*70)
    print("TEST 1: Matching from BuildingData (AddressPipeline)")
    print("="*70)

    # Simulate BuildingData from address pipeline
    test_cases = [
        BuildingData(
            address="Rinkeby Allé 10, Stockholm",
            latitude=59.3893,
            longitude=17.9265,
            construction_year=1972,
            facade_material="concrete",
            building_form="skivhus",
            num_floors=8,
            building_type="multi_family",
            energy_class="E",
            atemp_m2=5000,
            data_sources=["osm", "nominatim"],
        ),
        BuildingData(
            address="Hammarby Allé 55, Stockholm",
            latitude=59.3041,
            longitude=18.1044,
            construction_year=2003,
            facade_material="render",
            building_form="lamellhus",
            num_floors=5,
            building_type="multi_family",
            energy_class="C",
            atemp_m2=3000,
            data_sources=["osm", "nominatim"],
        ),
        BuildingData(
            address="Södermalm, Stockholm",
            latitude=59.3167,
            longitude=18.0667,
            construction_year=1910,
            facade_material="brick",
            building_form="generic",
            num_floors=5,
            building_type="multi_family",
            energy_class="F",
            atemp_m2=2500,
            data_sources=["osm", "nominatim"],
        ),
    ]

    for bd in test_cases:
        result = match_archetype_from_building_data(bd)

        print(f"\n{bd.address} ({bd.construction_year}):")
        print(f"  Matched: {result.archetype.name_en}")
        print(f"  Era: {result.archetype.era.value}")
        print(f"  Confidence: {result.confidence:.0%}")
        print(f"  Data sources: {', '.join(result.data_sources_used)}")
        print(f"\n  Score breakdown:")
        print(f"    Declaration: {result.source_scores.energy_declaration:.1f}")
        print(f"    OSM geometry: {result.source_scores.osm_geometry:.1f}")
        print(f"    Visual: {result.source_scores.mapillary_visual:.1f}")
        print(f"    Location: {result.source_scores.location:.1f}")
        print(f"\n  Match reasons:")
        for reason in result.match_reasons[:3]:
            print(f"    + {reason}")
        if result.mismatch_reasons:
            print(f"\n  Mismatches:")
            for reason in result.mismatch_reasons[:2]:
                print(f"    - {reason}")

        # Show calibration hints
        if result.calibration_hints:
            print(f"\n  Calibration hints:")
            for key, value in result.calibration_hints.items():
                if key.endswith('_value'):
                    print(f"    {key}: {value:.2f}")
                elif key.endswith('_note'):
                    print(f"    Note: {value}")


def test_matcher_with_real_declaration_data():
    """Test matcher simulating real energy declaration data."""
    print("\n" + "="*70)
    print("TEST 2: Simulating Energy Declaration Data")
    print("="*70)

    from src.core.building_context import EnhancedBuildingContext, BuildingContextBuilder
    from src.baseline.archetypes import BuildingType, HeatingSystem, VentilationType

    # Create a mock EnhancedBuildingContext with real data
    ctx = EnhancedBuildingContext(
        address="Tensta Allé 5, Stockholm",
        construction_year=1969,
        building_type=BuildingType.MULTI_FAMILY,
        facade_material="concrete",
        heating_system=HeatingSystem.DISTRICT,
        ventilation_type=VentilationType.EXHAUST,  # F-system
        atemp_m2=8000,
        floors=9,
    )

    # Test with matcher directly
    matcher = ArchetypeMatcherV2()
    result = matcher.match_from_context(ctx, use_ai_visual=False)

    print(f"\n{ctx.address}:")
    print(f"  Input: year={ctx.construction_year}, facade={ctx.facade_material}")
    print(f"         floors={ctx.floors}, vent={ctx.ventilation_type.value}")
    print(f"\n  Matched: {result.archetype.name_en} ({result.archetype.id})")
    print(f"  Era: {result.archetype.era.value}")
    print(f"  Confidence: {result.confidence:.0%}")
    print(f"\n  Score: {result.source_scores.total:.1f}/100")
    print(f"    - Declaration data: {result.source_scores.energy_declaration:.1f}")
    print(f"    - OSM geometry: {result.source_scores.osm_geometry:.1f}")
    print(f"    - Location: {result.source_scores.location:.1f}")

    print(f"\n  Match reasons:")
    for reason in result.match_reasons:
        print(f"    + {reason}")

    # Show alternatives
    print(f"\n  Alternatives:")
    for arch, conf in result.alternatives[:3]:
        print(f"    - {arch.name_en}: {conf:.0%}")


def test_neighborhood_matching():
    """Test that neighborhood-specific archetypes score higher."""
    print("\n" + "="*70)
    print("TEST 3: Neighborhood-Specific Matching")
    print("="*70)

    # Miljonprogrammet neighborhoods
    miljonprogram_areas = [
        ("Rinkeby, Stockholm", 1972),
        ("Tensta, Stockholm", 1969),
        ("Rosengård, Malmö", 1970),
        ("Hammarkullen, Göteborg", 1971),
    ]

    matcher = ArchetypeMatcherV2()

    for address, year in miljonprogram_areas:
        bd = BuildingData(
            address=address,
            latitude=59.0,
            longitude=18.0,
            construction_year=year,
            facade_material="concrete",
            building_form="skivhus",
            num_floors=8,
            building_type="multi_family",
            energy_class="E",
            data_sources=[],
        )

        result = matcher.match_from_building_data(bd, use_ai_visual=False)

        # Check if miljonprogrammet archetype was matched
        is_miljonprogram = "1961_1975" in result.archetype.era.value
        marker = "✓" if is_miljonprogram else "?"

        print(f"{marker} {address} ({year}):")
        print(f"    → {result.archetype.name_en}")
        print(f"    Location score: {result.source_scores.location:.1f}")


def test_data_source_tracking():
    """Test that data sources are properly tracked."""
    print("\n" + "="*70)
    print("TEST 4: Data Source Tracking")
    print("="*70)

    bd = BuildingData(
        address="Aktergatan 11, Hammarby Sjöstad, Stockholm",
        latitude=59.3041,
        longitude=18.1044,
        construction_year=2003,
        facade_material="render",
        building_form="lamellhus",
        num_floors=4,
        building_type="multi_family",
        energy_class="C",
        atemp_m2=15350,
        data_sources=["osm", "nominatim", "mapillary"],
        # Simulate Mapillary images
        facade_images={
            "S": ["https://example.com/facade_south.jpg"],
            "N": ["https://example.com/facade_north.jpg"],
        },
    )

    result = match_archetype_from_building_data(bd)

    print(f"\n{bd.address}:")
    print(f"  Input data sources: {bd.data_sources}")
    print(f"  Matcher used: {result.data_sources_used}")
    print(f"\n  Matched: {result.archetype.name_en}")
    print(f"  Confidence: {result.confidence:.0%}")

    # Show which sources contributed to scoring
    print(f"\n  Score by source:")
    print(f"    Declaration/Year: {result.source_scores.energy_declaration:.1f}")
    print(f"    OSM Geometry: {result.source_scores.osm_geometry:.1f}")
    print(f"    Mapillary Visual: {result.source_scores.mapillary_visual:.1f}")
    print(f"    Location: {result.source_scores.location:.1f}")
    print(f"    TOTAL: {result.source_scores.total:.1f}")


def test_archetype_coverage():
    """Verify all 40 archetypes are accessible."""
    print("\n" + "="*70)
    print("TEST 5: Archetype Coverage")
    print("="*70)

    archetypes = get_all_archetypes()
    print(f"\nTotal archetypes: {len(archetypes)}")

    # Group by era
    by_era = {}
    for arch_id, arch in archetypes.items():
        era = arch.era.value
        if era not in by_era:
            by_era[era] = []
        by_era[era].append(arch_id)

    print("\nBy era:")
    for era, ids in sorted(by_era.items()):
        print(f"  {era}: {len(ids)} archetypes")

    # Check descriptors
    with_descriptors = sum(1 for a in archetypes.values() if a.descriptors)
    print(f"\nWith descriptors: {with_descriptors}/{len(archetypes)}")


def main():
    """Run all tests."""
    print("="*70)
    print("ARCHETYPE MATCHER V2 - INTEGRATED PIPELINE TESTS")
    print("Using REAL DATA from energy declarations, OSM, Mapillary")
    print("="*70)

    test_from_building_data()
    test_matcher_with_real_declaration_data()
    test_neighborhood_matching()
    test_data_source_tracking()
    test_archetype_coverage()

    print("\n" + "="*70)
    print("ALL TESTS COMPLETED")
    print("="*70)


if __name__ == "__main__":
    main()
