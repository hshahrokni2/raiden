#!/usr/bin/env python
"""
Test multi-zone modeling with Grynnan 2 (Hammarby Allé 139).

This building has:
- 88% residential, 6% restaurant, 6% retail
- FTX + F-only ventilation (mixed)
- Declared 147 kWh/m² (high due to restaurant ventilation)

Expected: Multi-zone model with:
- Ground floor: Restaurant zone (F-only, 10 L/s/m², 0% HR)
- Upper floors: Residential zones (FTX, 0.35 L/s/m², 80% HR)
- Effective HR: ~31% (not 80%!)
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ingest.sweden_buildings import SwedenBuildingsLoader
from src.baseline.zone_assignment import assign_zones_to_floors, get_zone_layout_summary
from src.ingest.zone_configs import calculate_effective_ventilation


def test_grynnan_zone_assignment():
    """Test zone assignment for Grynnan 2 mixed-use building."""
    print("=" * 60)
    print("Testing Zone Assignment for Grynnan 2 (Mixed-Use Building)")
    print("=" * 60)

    # Grynnan 2 data from GeoJSON energy declaration
    zone_breakdown = {
        'residential': 0.88,
        'restaurant': 0.06,
        'retail': 0.06,
    }
    total_floors = 8  # Approximate
    footprint_m2 = 850  # Approximate from ~6800 m² Atemp / 8 floors

    print(f"\nInput Zone Breakdown:")
    for ztype, frac in zone_breakdown.items():
        print(f"  {ztype}: {frac:.0%}")

    # Test zone assignment
    layout = assign_zones_to_floors(
        total_floors=total_floors,
        footprint_area_m2=footprint_m2,
        zone_breakdown=zone_breakdown,
        floor_height_m=2.8,
        has_ftx=True,  # Residential has FTX
        has_f_only=True,  # Restaurant/retail has F-only
    )

    print(f"\n{get_zone_layout_summary(layout)}")

    # Calculate effective ventilation
    eff_vent = calculate_effective_ventilation(zone_breakdown, has_ftx=True)

    print(f"\nEffective Ventilation Parameters:")
    print(f"  Airflow: {eff_vent['effective_airflow_l_s_m2']:.2f} L/s/m²")
    print(f"  Heat Recovery: {eff_vent['effective_heat_recovery']:.1%}")
    print(f"  Heat Loss Factor: {eff_vent['heat_loss_factor']:.2f}")

    print(f"\nZone Details:")
    for detail in eff_vent['zone_details']:
        print(f"  {detail['zone']}: {detail['fraction']:.0%} × {detail['airflow']:.2f} L/s/m² "
              f"× (1-{detail['hr']:.0%}) = {detail['heat_loss_contribution']:.3f} loss factor")

    # Verify physics
    print(f"\nPhysics Verification:")
    # Restaurant should dominate heat loss despite only 6% area
    restaurant_detail = next(d for d in eff_vent['zone_details'] if d['zone'] == 'restaurant')
    total_loss = eff_vent['heat_loss_factor']
    restaurant_contribution = restaurant_detail['heat_loss_contribution'] / total_loss * 100

    print(f"  Restaurant area: 6%")
    print(f"  Restaurant heat loss contribution: {restaurant_contribution:.0f}%")

    if restaurant_contribution > 50:
        print("  ✓ Restaurant dominates heat loss (as expected)")
    else:
        print("  ✗ Restaurant should dominate heat loss!")

    # Check effective HR
    if eff_vent['effective_heat_recovery'] < 0.40:
        print(f"  ✓ Effective HR {eff_vent['effective_heat_recovery']:.0%} < 40% (as expected)")
    else:
        print(f"  ✗ Effective HR should be < 40% (got {eff_vent['effective_heat_recovery']:.0%})")

    print("\n" + "=" * 60)
    print("Test complete!")


def test_geojson_lookup():
    """Try to find Grynnan in the GeoJSON database."""
    print("\n" + "=" * 60)
    print("Looking up Grynnan in GeoJSON database...")
    print("=" * 60)

    try:
        loader = SwedenBuildingsLoader()
        stats = loader.get_statistics()
        print(f"Loaded {stats['total_buildings']} buildings")

        # Search for Grynnan
        results = loader.find_by_address("Hammarby Allé")
        print(f"Found {len(results)} results for 'Hammarby Allé'")

        grynnan = None
        for b in results:
            if "139" in str(b.address):
                grynnan = b
                break

        if grynnan:
            print(f"\nFound Grynnan 2:")
            print(f"  Address: {grynnan.address}")
            print(f"  Year: {grynnan.construction_year}")
            print(f"  Atemp: {grynnan.atemp_m2:.0f} m²")
            print(f"  Energy: {grynnan.energy_performance_kwh_m2:.0f} kWh/m²")
            print(f"  Ventilation: FTX={grynnan.has_ftx}, F={grynnan.has_f_only}")

            # Check mixed-use data
            if grynnan.is_mixed_use():
                print(f"  Mixed-use: YES")
                breakdown = grynnan.get_zone_breakdown()
                print(f"  Zone breakdown: {breakdown}")
            else:
                print(f"  Mixed-use: NO (checking raw properties...)")
                # Check raw properties
                props = grynnan.raw_properties
                for key in props:
                    if 'EgenAtemp' in key or 'atemp' in key.lower():
                        val = props[key]
                        if val and val != 0:
                            print(f"    {key}: {val}")
        else:
            print("Grynnan 2 not found by address search")

    except Exception as e:
        print(f"GeoJSON lookup failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_grynnan_zone_assignment()
    test_geojson_lookup()
