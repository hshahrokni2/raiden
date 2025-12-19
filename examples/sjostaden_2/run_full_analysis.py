#!/usr/bin/env python3
"""
Full End-to-End Analysis: BRF Sjostaden 2

This demonstrates the complete Raiden workflow:
1. Load enriched building data (from energy declaration + public sources)
2. Detect existing measures
3. Filter applicable ECMs (excluding what's already done)
4. Modify baseline IDF for each applicable ECM
5. Run EnergyPlus simulations (if available)
6. Compare results and generate report

Usage:
    python run_full_analysis.py [--dry-run] [--parallel N]
"""

import json
import argparse
import shutil
import time
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.core.building_context import (
    EnhancedBuildingContext,
    ExistingMeasure,
    SmartECMFilter,
)
from src.baseline.archetypes import (
    ArchetypeMatcher, BuildingType, HeatingSystem, VentilationType
)
from src.ecm.catalog import ECMCatalog
from src.ecm.constraints import ConstraintEngine
from src.ecm.idf_modifier import IDFModifier
from src.analysis.package_simulator import PackageSimulator


def load_building_data():
    """Load enriched building data from JSON."""
    data_path = Path(__file__).parent / "BRF_Sjostaden_2_enriched.json"
    with open(data_path) as f:
        return json.load(f)


def build_context(data: dict) -> EnhancedBuildingContext:
    """Build EnhancedBuildingContext from enriched data."""
    summary = data['original_summary']
    building = data['buildings'][0]
    pdf_data = data.get('pdf_extracted_data', {})

    ctx = EnhancedBuildingContext()
    ctx.address = f"{building['address']}, {summary['location']}"
    ctx.property_id = "brf_sjostaden_2"
    ctx.construction_year = summary['construction_year']
    ctx.building_type = BuildingType.MULTI_FAMILY
    ctx.facade_material = building['envelope'].get('facade_material', 'brick')
    ctx.atemp_m2 = summary['total_heated_area_sqm']
    ctx.floors = 7
    ctx.current_heating_kwh_m2 = pdf_data.get('specific_energy_kwh_sqm', 33.0)

    # U-values
    u_vals = building['envelope'].get('u_values', {})
    ctx.current_window_u = u_vals.get('windows', 0.8)

    # Envelope
    envelope = building['envelope']
    ctx.window_to_wall_ratio = envelope.get('wwr', {}).get('average', 0.17)

    # Heating system
    heating_str = summary.get('heating_system', '').lower()
    if 'ground source' in heating_str or 'bergvarme' in heating_str:
        ctx.heating_system = HeatingSystem.HEAT_PUMP_GROUND
    else:
        ctx.heating_system = HeatingSystem.DISTRICT

    # Ventilation
    vent_airflow = pdf_data.get('ventilation_airflow_ls_sqm', 0.35)
    if vent_airflow >= 0.35:
        ctx.ventilation_type = VentilationType.BALANCED
        ctx.current_heat_recovery = 0.75
    else:
        ctx.ventilation_type = VentilationType.EXHAUST

    # Solar
    solar = building['envelope'].get('solar_potential', {})
    ctx.roof_area_m2 = solar.get('total_roof_area_sqm', 2100)
    ctx.available_pv_area_m2 = solar.get('remaining_suitable_area_sqm', 235)

    # Detect existing measures
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
    """Detect existing measures from data."""
    existing = set()

    heating = summary.get('heating_system', '').lower()
    if 'ground source' in heating or 'bergvarme' in heating:
        existing.add(ExistingMeasure.HEAT_PUMP_GROUND)
    if 'exhaust air' in heating or 'franluft' in heating:
        existing.add(ExistingMeasure.HEAT_PUMP_EXHAUST)

    solar = building['envelope'].get('solar_potential', {})
    if solar.get('existing_pv_sqm', 0) > 0:
        existing.add(ExistingMeasure.SOLAR_PV)

    vent_airflow = pdf_data.get('ventilation_airflow_ls_sqm', 0)
    if vent_airflow >= 0.35:
        existing.add(ExistingMeasure.FTX_SYSTEM)
        existing.add(ExistingMeasure.HEAT_RECOVERY)

    energy_class = summary.get('energy_class', 'D')
    energy_kwh = pdf_data.get('specific_energy_kwh_sqm', 100)
    if energy_class in ['A', 'B'] or energy_kwh < 40:
        u_windows = building['envelope'].get('u_values', {}).get('windows', 2.0)
        if u_windows <= 1.0:
            existing.add(ExistingMeasure.WINDOW_REPLACEMENT)

    return existing


def filter_applicable_ecms(ctx: EnhancedBuildingContext):
    """Filter ECMs based on building context and existing measures."""
    catalog = ECMCatalog()
    all_ecms = catalog.all()
    engine = ConstraintEngine(catalog)
    ecm_filter = SmartECMFilter()
    return ecm_filter.filter_ecms(all_ecms, ctx, engine)


def generate_ecm_scenarios(
    baseline_idf: Path,
    applicable_ecms: list,
    output_dir: Path
) -> dict:
    """Generate modified IDF files for each applicable ECM."""
    modifier = IDFModifier()
    scenarios = {}

    for ecm in applicable_ecms:
        # Get default parameters for this ECM
        params = get_ecm_parameters(ecm)

        try:
            output_path = modifier.apply_single(
                baseline_idf=baseline_idf,
                ecm_id=ecm.id,
                params=params,
                output_dir=output_dir,
                output_name=f"scenario_{ecm.id}"
            )
            scenarios[ecm.id] = {
                'ecm': ecm,
                'idf_path': output_path,
                'params': params,
                'success': True,
            }
            print(f"   [+] Generated: {output_path.name}")
        except Exception as e:
            scenarios[ecm.id] = {
                'ecm': ecm,
                'error': str(e),
                'success': False,
            }
            print(f"   [!] Failed: {ecm.id} - {e}")

    return scenarios


def get_ecm_parameters(ecm) -> dict:
    """Get appropriate parameters for each ECM type."""
    params = {
        # Envelope
        'wall_internal_insulation': {'thickness_mm': 100, 'material': 'mineral_wool'},
        'wall_external_insulation': {'thickness_mm': 100, 'material': 'mineral_wool'},
        'roof_insulation': {'thickness_mm': 150, 'material': 'mineral_wool'},
        'air_sealing': {'reduction_factor': 0.5},
        'window_replacement': {'u_value': 0.9, 'shgc': 0.5},
        # HVAC
        'demand_controlled_ventilation': {'co2_setpoint': 1000},
        'heat_pump_integration': {'cop': 3.5, 'coverage': 0.8},
        'ftx_upgrade': {'effectiveness': 0.85},
        'ftx_installation': {'effectiveness': 0.80},
        # Controls
        'smart_thermostats': {'setback_c': 2},
        'led_lighting': {'power_density': 6},
        # Renewable
        'solar_pv': {'coverage_fraction': 0.7, 'panel_efficiency': 0.20},
        # Zero-cost operational measures
        'duc_calibration': {'curve_reduction_c': 2},
        'heating_curve_adjustment': {'supply_reduction_c': 3},
        'ventilation_schedule_optimization': {'off_hours_reduction': 0.5},
        'radiator_balancing': {'efficiency_improvement': 0.05},
        'effektvakt_optimization': {},  # Peak shaving, no thermal effect
    }
    return params.get(ecm.id, {})


def run_simulations(
    scenarios: dict,
    weather_path: Path,
    output_base: Path,
    parallel: int = 1
) -> dict:
    """Run EnergyPlus simulations for all scenarios."""
    # Try to import simulation runner
    try:
        from src.simulation.runner import SimulationRunner
        runner = SimulationRunner()
    except Exception as e:
        print(f"   [!] EnergyPlus not available: {e}")
        return {}

    results = {}
    for ecm_id, scenario in scenarios.items():
        if not scenario.get('success'):
            continue

        idf_path = scenario['idf_path']
        sim_output = output_base / ecm_id

        print(f"   Running: {ecm_id}...")
        try:
            result = runner.run_and_parse(
                idf_path=idf_path,
                weather_path=weather_path,
                output_dir=sim_output,
                timeout_seconds=300
            )
            results[ecm_id] = result
            if result.success:
                heating = result.parsed_results.heating_kwh_m2 if result.parsed_results else 'N/A'
                print(f"   [+] {ecm_id}: {heating} kWh/m2")
            else:
                print(f"   [!] {ecm_id}: FAILED - {result.error_message}")
        except Exception as e:
            print(f"   [!] {ecm_id}: ERROR - {e}")

    return results


def run_package_simulations(
    ecm_sim_results: dict,
    baseline_idf: Path,
    baseline_kwh_m2: float,
    atemp_m2: float,
    weather_path: Path,
    output_dir: Path,
) -> list:
    """
    Run package simulations (combined ECMs).

    Instead of estimating combined savings with an interaction factor,
    this creates actual combined IDF files and runs EnergyPlus.
    """
    # Convert simulation results to ECM results format
    ecm_results = []
    for ecm_id, result in ecm_sim_results.items():
        if result.success and result.parsed_results:
            heating = result.parsed_results.heating_kwh_m2
            savings = baseline_kwh_m2 - heating
            savings_pct = (savings / baseline_kwh_m2) * 100 if baseline_kwh_m2 > 0 else 0
            ecm_results.append({
                'id': ecm_id,
                'name': ecm_id.replace('_', ' ').title(),
                'savings_percent': savings_pct,
                'heating_kwh_m2': heating,
                'params': get_ecm_parameters(type('ECM', (), {'id': ecm_id})()),
            })

    if not ecm_results:
        print("   [!] No valid ECM results for package simulation")
        return []

    simulator = PackageSimulator()
    packages = simulator.create_and_simulate_packages(
        ecm_results=ecm_results,
        baseline_idf=baseline_idf,
        baseline_kwh_m2=baseline_kwh_m2,
        atemp_m2=atemp_m2,
        weather_path=weather_path,
        output_dir=output_dir / "packages",
        run_simulation=True,
    )

    return packages


def print_summary(ctx, filter_result, scenarios, sim_results, baseline_heating=None, packages=None):
    """Print analysis summary."""
    print("\n" + "="*70)
    print("ANALYSIS SUMMARY")
    print("="*70)

    print(f"\n Building: {ctx.address}")
    print(f"   Year: {ctx.construction_year}, Area: {ctx.atemp_m2:,.0f} m2")
    print(f"   Declared heating: {ctx.current_heating_kwh_m2} kWh/m2/year")

    # Use simulation baseline if provided, otherwise declared value
    if baseline_heating is None:
        baseline_heating = ctx.current_heating_kwh_m2

    print(f"\n Existing Measures ({len(ctx.existing_measures)}):")
    for m in ctx.existing_measures:
        print(f"   - {m.value}")

    print(f"\n Applicable ECMs ({len(filter_result['applicable'])}):")
    for ecm in filter_result['applicable']:
        status = "Generated" if scenarios.get(ecm.id, {}).get('success') else "Skipped"
        sim_status = ""
        if ecm.id in sim_results:
            r = sim_results[ecm.id]
            if r.success and r.parsed_results:
                sim_status = f" -> {r.parsed_results.heating_kwh_m2:.1f} kWh/m2"
        print(f"   - {ecm.name} [{status}]{sim_status}")

    print(f"\n Excluded ECMs:")
    for item in filter_result['already_done']:
        print(f"   - {item['ecm'].name}: {item['reason']}")
    for item in filter_result['not_applicable']:
        reasons = item.get('reasons', [])
        reason = reasons[0][1] if reasons else 'Technical constraint'
        print(f"   - {item['ecm'].name}: {reason}")

    if sim_results:
        print(f"\n Individual ECM Results (baseline: {baseline_heating:.1f} kWh/m2):")
        for ecm_id, result in sorted(sim_results.items(), key=lambda x: x[1].parsed_results.heating_kwh_m2 if x[1].success and x[1].parsed_results else 999):
            if result.success and result.parsed_results:
                new_heating = result.parsed_results.heating_kwh_m2
                savings = baseline_heating - new_heating
                savings_pct = (savings / baseline_heating) * 100 if baseline_heating > 0 else 0
                if savings > 0:
                    print(f"   + {ecm_id}: {new_heating:.1f} kWh/m2 "
                          f"(saves {savings:.1f} kWh/m2, {savings_pct:.0f}%)")
                else:
                    print(f"   - {ecm_id}: {new_heating:.1f} kWh/m2 (no change)")

    # Print package results
    if packages:
        print(f"\n PACKAGE SIMULATIONS (physics-based combined savings):")
        print("-" * 60)
        for pkg in packages:
            status = "SIMULATED" if pkg.simulation_success else "ESTIMATED"
            print(f"\n   {pkg.name_sv} ({pkg.id}):")
            print(f"      ECMs: {', '.join(e.id for e in pkg.ecms)}")
            print(f"      Sum individual: {pkg.sum_individual_savings_percent:.1f}%")
            print(f"      Actual combined: {pkg.simulated_savings_percent:.1f}% [{status}]")
            print(f"      Interaction factor: {pkg.interaction_factor:.2f}")
            print(f"      Heating: {pkg.simulated_heating_kwh_m2:.1f} kWh/m2")
            print(f"      Cost: {pkg.total_cost_sek:,.0f} SEK")
            print(f"      Payback: {pkg.simple_payback_years:.1f} years")


def main():
    parser = argparse.ArgumentParser(description='Run full building analysis')
    parser.add_argument('--dry-run', action='store_true',
                       help='Generate IDFs but skip simulation')
    parser.add_argument('--parallel', type=int, default=1,
                       help='Number of parallel simulations')
    args = parser.parse_args()

    start_time = time.time()

    print("="*70)
    print("RAIDEN - Full Building Energy Analysis Pipeline")
    print("="*70)

    # Step 1: Load data
    print("\n[1/5] Loading enriched building data...")
    data = load_building_data()
    print(f"   Building: {data['brf_name']}")

    # Step 2: Build context
    print("\n[2/5] Building analysis context...")
    ctx = build_context(data)
    print(f"   Address: {ctx.address}")
    print(f"   Existing measures: {len(ctx.existing_measures)}")

    # Step 3: Filter ECMs
    print("\n[3/5] Filtering applicable ECMs...")
    filter_result = filter_applicable_ecms(ctx)
    print(f"   Applicable: {len(filter_result['applicable'])}")
    print(f"   Already done: {len(filter_result['already_done'])}")
    print(f"   Not applicable: {len(filter_result['not_applicable'])}")

    # Step 4: Generate scenario IDFs
    print("\n[4/5] Generating ECM scenario IDFs...")
    project_root = Path(__file__).parent.parent.parent
    baseline_idf = project_root / "sjostaden_7zone.idf"
    output_dir = Path(__file__).parent / "ecm_scenarios"

    if baseline_idf.exists():
        scenarios = generate_ecm_scenarios(
            baseline_idf=baseline_idf,
            applicable_ecms=filter_result['applicable'],
            output_dir=output_dir
        )
        print(f"   Generated {sum(1 for s in scenarios.values() if s.get('success'))} scenario IDFs")
    else:
        print(f"   [!] Baseline IDF not found: {baseline_idf}")
        scenarios = {}

    # Step 5: Run simulations (if not dry-run)
    sim_results = {}
    baseline_heating = None
    if not args.dry_run and scenarios:
        print("\n[5/6] Running EnergyPlus simulations...")
        weather_path = project_root / "tests" / "fixtures" / "stockholm.epw"
        sim_output = Path(__file__).parent / "simulation_results"

        if weather_path.exists():
            # First run baseline
            print("   Running baseline simulation...")
            try:
                from src.simulation.runner import SimulationRunner
                runner = SimulationRunner()
                baseline_result = runner.run_and_parse(
                    idf_path=baseline_idf,
                    weather_path=weather_path,
                    output_dir=sim_output / "baseline",
                    timeout_seconds=300
                )
                if baseline_result.success and baseline_result.parsed_results:
                    baseline_heating = baseline_result.parsed_results.heating_kwh_m2
                    print(f"   [+] Baseline: {baseline_heating:.1f} kWh/m2")
            except Exception as e:
                print(f"   [!] Baseline simulation failed: {e}")

            # Then run ECM scenarios
            print("\n[6/8] Running ECM scenario simulations...")
            sim_results = run_simulations(
                scenarios=scenarios,
                weather_path=weather_path,
                output_base=sim_output,
                parallel=args.parallel
            )

            # Run package simulations (combined ECMs)
            packages = []
            if sim_results and baseline_heating:
                print("\n[7/8] Running PACKAGE simulations (combined ECMs)...")
                packages = run_package_simulations(
                    ecm_sim_results=sim_results,
                    baseline_idf=baseline_idf,
                    baseline_kwh_m2=baseline_heating,
                    atemp_m2=ctx.atemp_m2,
                    weather_path=weather_path,
                    output_dir=sim_output,
                )
                print(f"   Created {len(packages)} packages")
                for pkg in packages:
                    status = "OK" if pkg.simulation_success else "est."
                    print(f"   [{status}] {pkg.name_sv}: {pkg.simulated_savings_percent:.1f}% savings")
        else:
            print(f"   [!] Weather file not found: {weather_path}")
            packages = []
    else:
        print("\n[5/5] Skipping simulations (--dry-run)")
        packages = []

    # Print summary
    print_summary(ctx, filter_result, scenarios, sim_results, baseline_heating, packages)

    # Generate HTML report
    if sim_results:
        print("\n[8/8] Generating HTML report...")
        try:
            from src.reporting.html_report import generate_report
            report_path = Path(__file__).parent / "analysis_report.html"
            generate_report(
                building_data=data,
                simulation_results=sim_results,
                filter_result=filter_result,
                output_path=report_path,
                baseline_heating=baseline_heating,
                packages=packages,  # Pass simulated packages
            )
            print(f"   [+] Report saved: {report_path}")
        except Exception as e:
            print(f"   [!] Report generation failed: {e}")

    elapsed = time.time() - start_time
    print(f"\n Analysis completed in {elapsed:.1f} seconds")
    print("="*70)


if __name__ == "__main__":
    main()
