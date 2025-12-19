#!/usr/bin/env python3
"""
Example: Full analysis of BRF Sjöstaden 2 using enriched data.

This demonstrates the smart analysis pipeline:
1. Load enriched building data (from energy declaration + public sources)
2. Detect existing measures
3. Filter applicable ECMs (excluding what's already done)
4. Generate baseline and run scenarios
"""

import json
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.core.building_context import (
    EnhancedBuildingContext,
    ExistingMeasure,
    ExistingMeasuresDetector,
    SmartECMFilter,
    BuildingContextBuilder,
)
from src.baseline.archetypes import (
    ArchetypeMatcher, BuildingType, HeatingSystem, VentilationType
)
from src.ecm.catalog import ECMCatalog
from src.ecm.constraints import ConstraintEngine, BuildingContext as ConstraintContext


def load_sjostaden_data() -> dict:
    """Load the enriched building data."""
    data_path = Path(__file__).parent / "BRF_Sjostaden_2_enriched.json"
    with open(data_path) as f:
        return json.load(f)


def build_context_from_enriched(data: dict) -> EnhancedBuildingContext:
    """
    Build EnhancedBuildingContext from enriched JSON data.

    This shows how to integrate data from multiple sources:
    - Energy declaration (PDF extracted)
    - OSM/Overture geometry
    - Mapillary facade analysis
    - Google Solar API roof analysis
    """
    summary = data['original_summary']
    building = data['buildings'][0]  # First building
    pdf_data = data.get('pdf_extracted_data', {})
    facade_data = data.get('facade_analysis', {})

    ctx = EnhancedBuildingContext()

    # Basic info
    ctx.address = f"{building['address']}, {summary['location']}"
    ctx.property_id = "brf_sjostaden_2"
    ctx.construction_year = summary['construction_year']
    ctx.building_type = BuildingType.MULTI_FAMILY

    # Facade from Mapillary analysis
    ctx.facade_material = building['envelope'].get('facade_material', 'brick')

    # Areas
    ctx.atemp_m2 = summary['total_heated_area_sqm']
    ctx.floors = 7  # From extracted_profile.json

    # Current performance from energy declaration
    ctx.current_heating_kwh_m2 = pdf_data.get('specific_energy_kwh_sqm', 33.0)

    # U-values from back-calculation
    u_vals = building['envelope'].get('u_values', {})
    ctx.current_window_u = u_vals.get('windows', 0.8)

    # Envelope
    envelope = building['envelope']
    ctx.window_to_wall_ratio = envelope.get('wwr', {}).get('average', 0.17)

    # Parse heating system - this is the KEY insight
    heating_str = summary.get('heating_system', '').lower()
    if 'ground source' in heating_str or 'bergvärme' in heating_str:
        ctx.heating_system = HeatingSystem.HEAT_PUMP_GROUND
    elif 'fjärrvärme' in heating_str:
        ctx.heating_system = HeatingSystem.DISTRICT
    else:
        ctx.heating_system = HeatingSystem.DISTRICT

    # Ventilation - from PDF extraction
    vent_airflow = pdf_data.get('ventilation_airflow_ls_sqm', 0.35)
    if vent_airflow >= 0.35:
        ctx.ventilation_type = VentilationType.BALANCED  # FTX
        ctx.current_heat_recovery = 0.75  # Typical for modern FTX
    else:
        ctx.ventilation_type = VentilationType.EXHAUST

    # Solar potential from Google Solar API
    solar = building['envelope'].get('solar_potential', {})
    ctx.roof_area_m2 = solar.get('total_roof_area_sqm', 2100)
    ctx.available_pv_area_m2 = solar.get('remaining_suitable_area_sqm', 235)

    # Detect EXISTING MEASURES based on the data
    ctx.existing_measures = detect_existing_measures(summary, building, pdf_data)

    # Match archetype
    matcher = ArchetypeMatcher()
    ctx.archetype = matcher.match(
        construction_year=ctx.construction_year,
        building_type='multi_family',
        facade_material=ctx.facade_material
    )

    return ctx


def detect_existing_measures(summary: dict, building: dict, pdf_data: dict) -> set:
    """
    Detect existing measures from the data.

    This is CRITICAL - determines what NOT to recommend!
    """
    existing = set()

    # Check heating system
    heating = summary.get('heating_system', '').lower()
    if 'ground source' in heating or 'bergvärme' in heating:
        existing.add(ExistingMeasure.HEAT_PUMP_GROUND)
        print("  ✓ Detected: Ground source heat pump")

    if 'exhaust air' in heating or 'frånluft' in heating:
        existing.add(ExistingMeasure.HEAT_PUMP_EXHAUST)
        print("  ✓ Detected: Exhaust air heat pump")

    # Check for existing solar
    solar = building['envelope'].get('solar_potential', {})
    if solar.get('existing_pv_sqm', 0) > 0:
        existing.add(ExistingMeasure.SOLAR_PV)
        print(f"  ✓ Detected: Solar PV ({solar['existing_pv_sqm']} m²)")

    # Check ventilation (FTX implies heat recovery)
    vent_airflow = pdf_data.get('ventilation_airflow_ls_sqm', 0)
    if vent_airflow >= 0.35:
        # Modern FTX with proper airflow
        existing.add(ExistingMeasure.FTX_SYSTEM)
        existing.add(ExistingMeasure.HEAT_RECOVERY)
        print("  ✓ Detected: FTX system with heat recovery")

    # Infer from energy class
    energy_class = summary.get('energy_class', 'D')
    energy_kwh = pdf_data.get('specific_energy_kwh_sqm', 100)

    if energy_class in ['A', 'B'] or energy_kwh < 40:
        # Very efficient building - likely has good windows already
        u_windows = building['envelope'].get('u_values', {}).get('windows', 2.0)
        if u_windows <= 1.0:
            existing.add(ExistingMeasure.WINDOW_REPLACEMENT)
            print(f"  ✓ Detected: Modern windows (U={u_windows})")

    return existing


def analyze_ecm_applicability(ctx: EnhancedBuildingContext):
    """
    Demonstrate the smart ECM filtering.
    """
    print("\n" + "="*60)
    print("ECM APPLICABILITY ANALYSIS")
    print("="*60)

    # Get all ECMs
    catalog = ECMCatalog()
    all_ecms = catalog.all()

    # Create constraint engine
    engine = ConstraintEngine(catalog)

    # Use smart filter
    ecm_filter = SmartECMFilter()
    result = ecm_filter.filter_ecms(all_ecms, ctx, engine)

    # Print results
    print(f"\nBuilding: {ctx.address}")
    print(f"Year: {ctx.construction_year}, Facade: {ctx.facade_material}")
    print(f"Energy: {ctx.current_heating_kwh_m2} kWh/m² (Class {ctx.archetype.name if ctx.archetype else 'Unknown'})")

    print(f"\n📋 EXISTING MEASURES ({len(ctx.existing_measures)}):")
    for m in ctx.existing_measures:
        print(f"   ✅ {m.value}")

    print(f"\n✅ APPLICABLE ECMs ({len(result['applicable'])}):")
    for ecm in result['applicable']:
        print(f"   → {ecm.name}")

    print(f"\n⏭️  ALREADY IMPLEMENTED ({len(result['already_done'])}):")
    for item in result['already_done']:
        print(f"   ✓ {item['ecm'].name} - {item['reason']}")

    print(f"\n❌ NOT APPLICABLE ({len(result['not_applicable'])}):")
    for item in result['not_applicable']:
        reasons = item.get('reasons', [])
        reason_str = reasons[0][1] if reasons else 'Technical constraint'
        print(f"   ✗ {item['ecm'].name} - {reason_str}")

    return result


def estimate_savings_potential(ctx: EnhancedBuildingContext, applicable_ecms: list):
    """
    Estimate savings potential for applicable ECMs.
    """
    print("\n" + "="*60)
    print("SAVINGS POTENTIAL ESTIMATE")
    print("="*60)

    baseline = ctx.current_heating_kwh_m2
    print(f"\nBaseline heating: {baseline} kWh/m²/year")
    print(f"Building area: {ctx.atemp_m2:,.0f} m²")
    print(f"Total heating: {baseline * ctx.atemp_m2:,.0f} kWh/year")

    # Rough savings estimates per ECM
    savings_estimates = {
        'wall_external_insulation': 0.15,  # 15% reduction
        'wall_internal_insulation': 0.10,
        'roof_insulation': 0.05,
        'window_replacement': 0.12,
        'air_sealing': 0.08,
        'ftx_upgrade': 0.10,
        'ftx_installation': 0.25,
        'demand_controlled_ventilation': 0.08,
        'solar_pv': 0.0,  # Doesn't reduce heating
        'led_lighting': 0.02,  # Small indirect effect
        'smart_thermostats': 0.05,
        'heat_pump_integration': 0.20,
    }

    print("\nPotential ECM savings:")
    total_potential = 0
    for ecm in applicable_ecms:
        savings_pct = savings_estimates.get(ecm.id, 0.05)
        savings_kwh = baseline * savings_pct
        total_potential += savings_kwh
        print(f"   {ecm.name}: -{savings_pct*100:.0f}% = -{savings_kwh:.1f} kWh/m²")

    # Account for diminishing returns when combining
    interaction_factor = 0.7  # 70% of additive savings
    combined_savings = total_potential * interaction_factor

    print(f"\nCombined potential (with 70% interaction factor):")
    print(f"   Total savings: -{combined_savings:.1f} kWh/m² ({combined_savings/baseline*100:.0f}%)")
    print(f"   New heating: {baseline - combined_savings:.1f} kWh/m²")
    print(f"   Annual savings: {combined_savings * ctx.atemp_m2:,.0f} kWh")

    # Cost savings (Swedish electricity/district heating ~1.5 SEK/kWh)
    energy_price = 1.5  # SEK/kWh
    annual_savings_sek = combined_savings * ctx.atemp_m2 * energy_price
    print(f"   Cost savings: {annual_savings_sek:,.0f} SEK/year")


def main():
    print("="*60)
    print("BRF SJÖSTADEN 2 - Smart Building Analysis")
    print("="*60)

    # Load enriched data
    print("\n📁 Loading enriched building data...")
    data = load_sjostaden_data()

    summary = data['original_summary']
    print(f"\n📊 Building Summary:")
    print(f"   Name: {data['brf_name']}")
    print(f"   Location: {summary['location']}")
    print(f"   Year: {summary['construction_year']}")
    print(f"   Energy Class: {summary['energy_class']}")
    print(f"   Energy Use: {summary['energy_performance_kwh_per_sqm']} kWh/m²")
    print(f"   Heating: {summary['heating_system']}")
    print(f"   Solar: {'Yes' if summary['has_solar_panels'] else 'No'}")

    # Build context
    print("\n🔧 Building analysis context...")
    print("   Detecting existing measures:")
    ctx = build_context_from_enriched(data)

    # Analyze ECM applicability
    result = analyze_ecm_applicability(ctx)

    # Estimate savings
    estimate_savings_potential(ctx, result['applicable'])

    print("\n" + "="*60)
    print("✅ Analysis complete!")
    print("="*60)

    # Summary recommendation
    print("\n💡 RECOMMENDATION:")
    print(f"   This building is already very efficient (Class {summary['energy_class']}).")
    print(f"   Existing measures: {len(ctx.existing_measures)}")
    print(f"   Additional ECMs available: {len(result['applicable'])}")

    if len(result['applicable']) > 0:
        top_ecm = result['applicable'][0]
        print(f"\n   Top recommendation: {top_ecm.name}")
        print(f"   Focus on: Building envelope improvements or smart controls")

    # Solar note
    solar = data['buildings'][0]['envelope'].get('solar_potential', {})
    if solar.get('remaining_suitable_area_sqm', 0) > 100:
        print(f"\n   ☀️ Additional solar potential:")
        print(f"      Remaining area: {solar['remaining_suitable_area_sqm']:.0f} m²")
        print(f"      Additional capacity: {solar['remaining_capacity_kwp']:.0f} kWp")
        print(f"      Additional yield: {solar['annual_yield_potential_kwh']:,.0f} kWh/year")


if __name__ == "__main__":
    main()
