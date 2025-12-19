#!/usr/bin/env python3
"""
Raiden Demo - Swedish Building ECM Simulator

This demo shows how Raiden analyzes a Swedish building to recommend
energy conservation measures (ECMs) using only public data.

Usage:
    python demo.py
    python demo.py "Storgatan 1, Stockholm"
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))


def demo_building_lookup(address: str = "Bellmansgatan 16, Stockholm"):
    """Demo: Look up building data from address."""
    print("\n" + "=" * 60)
    print("RAIDEN - Swedish Building ECM Simulator")
    print("=" * 60)

    print(f"\n1. FETCHING BUILDING DATA")
    print(f"   Address: {address}")
    print("   " + "-" * 50)

    from src.core.address_pipeline import BuildingDataFetcher

    fetcher = BuildingDataFetcher()
    data = fetcher.fetch(address)

    print(f"   Found: {data.address}")
    print(f"   Construction year: {data.construction_year}")
    print(f"   Building type: {data.building_type}")
    print(f"   Atemp: {data.atemp_m2} m²")
    print(f"   Energy class: {data.energy_class}")
    print(f"   Declared energy: {data.declared_energy_kwh_m2} kWh/m²")
    print(f"   Heating: {data.heating_system}")
    print(f"   Has FTX: {data.has_ftx}")
    print(f"   Has heat pump: {data.has_heat_pump}")
    print(f"   Has solar: {data.has_solar}")
    print(f"   Facade: {data.facade_material}")
    print(f"   Data sources: {data.data_sources}")

    return data


def demo_archetype_matching(data):
    """Demo: Match building to archetype."""
    print(f"\n2. MATCHING ARCHETYPE")
    print("   " + "-" * 50)

    from src.baseline import ArchetypeMatcherV2

    matcher = ArchetypeMatcherV2(use_ai_modules=False)
    result = matcher.match_from_building_data(data)

    arch = result.archetype
    print(f"   Matched: {arch.name_en}")
    print(f"   Era: {arch.era.value} ({arch.year_start}-{arch.year_end})")
    print(f"   Confidence: {result.confidence:.0%}")
    print(f"   Wall U-value: {arch.wall_constructions[0].u_value} W/m²K")
    print(f"   Roof U-value: {arch.roof_construction.u_value} W/m²K")
    print(f"   Window U-value: {arch.window_construction.u_value_installed} W/m²K")

    return result


def demo_ecm_filtering(data):
    """Demo: Filter applicable ECMs."""
    print(f"\n3. FILTERING ECMs")
    print("   " + "-" * 50)

    from src.ecm import ConstraintEngine, get_all_ecms
    from src.ecm.constraints import BuildingContext

    # Build context from data
    ctx = BuildingContext(
        construction_year=data.construction_year,
        building_type=data.building_type,
        facade_material=data.facade_material,
        heating_system=data.heating_system,
        ventilation_type="ftx" if data.has_ftx else "exhaust",
        heritage_listed=False,
        current_window_u=2.0 if data.construction_year < 1980 else 1.2,
        current_heat_recovery=0.75 if data.has_ftx else 0.0,
    )

    engine = ConstraintEngine()
    valid_ecms = engine.get_valid_ecms(ctx)
    excluded_ecms = engine.get_excluded_ecms(ctx)

    print(f"   Total ECMs: {len(get_all_ecms())}")
    print(f"   Applicable: {len(valid_ecms)}")
    print(f"   Excluded: {len(excluded_ecms)}")

    print("\n   Applicable ECMs:")
    for ecm in valid_ecms[:8]:  # Show top 8
        print(f"   - {ecm.name} ({ecm.typical_savings_percent}% savings)")

    if excluded_ecms:
        print("\n   Excluded ECMs (examples):")
        for ecm, reasons in excluded_ecms[:3]:
            print(f"   - {ecm.name}: {reasons[0]}")

    return valid_ecms


def demo_ecm_catalog():
    """Demo: Show ECM catalog."""
    print(f"\n4. ECM CATALOG SUMMARY")
    print("   " + "-" * 50)

    from src.ecm import get_all_ecms, ECMCategory

    ecms = get_all_ecms()

    # Group by category
    by_cat = {}
    for ecm in ecms:
        cat = ecm.category.value
        if cat not in by_cat:
            by_cat[cat] = []
        by_cat[cat].append(ecm)

    for cat, cat_ecms in by_cat.items():
        print(f"\n   {cat.upper()} ({len(cat_ecms)} ECMs):")
        for ecm in cat_ecms[:3]:  # Show top 3 per category
            cost_str = f"{ecm.cost_per_unit:,.0f} SEK/{ecm.cost_unit}"
            print(f"   - {ecm.name_sv}: {cost_str}")


def demo_geojson_stats():
    """Demo: Show GeoJSON statistics."""
    print(f"\n5. SWEDISH BUILDINGS DATABASE")
    print("   " + "-" * 50)

    from src.ingest import load_sweden_buildings

    loader = load_sweden_buildings()
    stats = loader.get_statistics()

    print(f"   Total buildings: {stats['total_buildings']:,}")
    print(f"   With energy class: {stats.get('with_energy_class', 'N/A')}")

    if 'energy_class_distribution' in stats:
        print("\n   Energy class distribution:")
        for cls, count in sorted(stats['energy_class_distribution'].items()):
            pct = count / stats['total_buildings'] * 100
            print(f"   - {cls}: {count:,} ({pct:.1f}%)")


def main():
    """Run the demo."""
    # Get address from command line or use default
    if len(sys.argv) > 1:
        address = " ".join(sys.argv[1:])
    else:
        address = "Bellmansgatan 16, Stockholm"

    try:
        # Run demos
        data = demo_building_lookup(address)
        demo_archetype_matching(data)
        demo_ecm_filtering(data)
        demo_ecm_catalog()
        demo_geojson_stats()

        print("\n" + "=" * 60)
        print("Demo complete! Raiden analyzed the building successfully.")
        print("=" * 60)
        print("\nNext steps:")
        print("  - Run EnergyPlus simulation: python -m src.cli.main analyze ...")
        print("  - Generate report: python -m src.cli.main analyze-address ...")
        print("  - Start API server: uvicorn src.api.main:app --reload")
        print()

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
