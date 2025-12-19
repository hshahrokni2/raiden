#!/usr/bin/env python3
"""
Raiden CLI - Swedish Building Energy Analysis

Usage:
    raiden analyze <building_json> [options]
    raiden batch <buildings_dir> [options]
    raiden ecm-list
    raiden --help

Examples:
    raiden analyze examples/sjostaden_2/BRF_Sjostaden_2_enriched.json
    raiden batch data/buildings/ --parallel 4
    raiden analyze building.json --ecms wall_internal_insulation,air_sealing
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional, List

try:
    from rich.console import Console
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
    from rich.table import Table
    from rich.panel import Panel
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.core.building_context import (
    EnhancedBuildingContext,
    ExistingMeasure,
    SmartECMFilter,
)
from src.baseline.archetypes import ArchetypeMatcher, BuildingType, HeatingSystem, VentilationType
from src.ecm.catalog import ECMCatalog
from src.ecm.constraints import ConstraintEngine
from src.ecm.idf_modifier import IDFModifier


def create_console():
    """Create console for rich output."""
    if RICH_AVAILABLE:
        return Console()
    return None


def print_header(console, text):
    """Print header with styling if rich is available."""
    if console:
        console.print(Panel(text, style="bold blue"))
    else:
        print("=" * 70)
        print(text)
        print("=" * 70)


def print_success(console, text):
    """Print success message."""
    if console:
        console.print(f"[green]✓[/green] {text}")
    else:
        print(f"[+] {text}")


def print_warning(console, text):
    """Print warning message."""
    if console:
        console.print(f"[yellow]![/yellow] {text}")
    else:
        print(f"[!] {text}")


def print_error(console, text):
    """Print error message."""
    if console:
        console.print(f"[red]✗[/red] {text}")
    else:
        print(f"[ERROR] {text}")


def load_building_data(json_path: Path) -> dict:
    """Load building data from JSON file."""
    with open(json_path) as f:
        return json.load(f)


def build_context(data: dict) -> EnhancedBuildingContext:
    """Build context from enriched JSON data."""
    summary = data.get('original_summary', {})
    building = data.get('buildings', [{}])[0]
    pdf_data = data.get('pdf_extracted_data', {})

    ctx = EnhancedBuildingContext()
    ctx.address = f"{building.get('address', 'Unknown')}, {summary.get('location', 'Sweden')}"
    ctx.property_id = data.get('brf_name', 'unknown').lower().replace(' ', '_')
    ctx.construction_year = summary.get('construction_year', 1990)
    ctx.building_type = BuildingType.MULTI_FAMILY
    ctx.facade_material = building.get('envelope', {}).get('facade_material', 'brick')
    ctx.atemp_m2 = summary.get('total_heated_area_sqm', 1000)
    ctx.floors = summary.get('floors', 4)
    ctx.current_heating_kwh_m2 = pdf_data.get('specific_energy_kwh_sqm', 100.0)

    # U-values
    u_vals = building.get('envelope', {}).get('u_values', {})
    ctx.current_window_u = u_vals.get('windows', 1.2)

    # Envelope
    envelope = building.get('envelope', {})
    ctx.window_to_wall_ratio = envelope.get('wwr', {}).get('average', 0.20)

    # Heating system
    heating_str = summary.get('heating_system', '').lower()
    if 'ground source' in heating_str or 'bergvarme' in heating_str:
        ctx.heating_system = HeatingSystem.HEAT_PUMP_GROUND
    elif 'fjärrvärme' in heating_str or 'district' in heating_str:
        ctx.heating_system = HeatingSystem.DISTRICT
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
    solar = building.get('envelope', {}).get('solar_potential', {})
    ctx.roof_area_m2 = solar.get('total_roof_area_sqm', 500)
    ctx.available_pv_area_m2 = solar.get('remaining_suitable_area_sqm', 200)

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
    """Detect existing measures from building data."""
    existing = set()

    heating = summary.get('heating_system', '').lower()
    if 'ground source' in heating or 'bergvarme' in heating:
        existing.add(ExistingMeasure.HEAT_PUMP_GROUND)
    if 'exhaust air' in heating or 'franluft' in heating:
        existing.add(ExistingMeasure.HEAT_PUMP_EXHAUST)

    solar = building.get('envelope', {}).get('solar_potential', {})
    if solar.get('existing_pv_sqm', 0) > 0:
        existing.add(ExistingMeasure.SOLAR_PV)

    vent_airflow = pdf_data.get('ventilation_airflow_ls_sqm', 0)
    if vent_airflow >= 0.35:
        existing.add(ExistingMeasure.FTX_SYSTEM)
        existing.add(ExistingMeasure.HEAT_RECOVERY)

    energy_class = summary.get('energy_class', 'D')
    energy_kwh = pdf_data.get('specific_energy_kwh_sqm', 100)
    if energy_class in ['A', 'B'] or energy_kwh < 40:
        u_windows = building.get('envelope', {}).get('u_values', {}).get('windows', 2.0)
        if u_windows <= 1.0:
            existing.add(ExistingMeasure.WINDOW_REPLACEMENT)

    return existing


def filter_ecms(ctx: EnhancedBuildingContext):
    """Filter applicable ECMs."""
    catalog = ECMCatalog()
    all_ecms = catalog.all()
    engine = ConstraintEngine(catalog)
    ecm_filter = SmartECMFilter()
    return ecm_filter.filter_ecms(all_ecms, ctx, engine)


def cmd_analyze(args, console):
    """Run analysis on a single building."""
    json_path = Path(args.building_json)

    if not json_path.exists():
        print_error(console, f"File not found: {json_path}")
        return 1

    print_header(console, "RAIDEN - Building Energy Analysis")

    # Load data
    if console and RICH_AVAILABLE:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Loading building data...", total=None)
            data = load_building_data(json_path)
            progress.update(task, description="Building context...")
            ctx = build_context(data)
            progress.update(task, description="Filtering ECMs...")
            filter_result = filter_ecms(ctx)
    else:
        print("Loading building data...")
        data = load_building_data(json_path)
        ctx = build_context(data)
        filter_result = filter_ecms(ctx)

    # Print results
    print_success(console, f"Building: {ctx.address}")
    print_success(console, f"Year: {ctx.construction_year}, Area: {ctx.atemp_m2:,.0f} m²")
    print_success(console, f"Current heating: {ctx.current_heating_kwh_m2} kWh/m²/year")

    # Print existing measures
    if console and RICH_AVAILABLE:
        console.print("\n[bold]Existing Measures:[/bold]")
        for m in ctx.existing_measures:
            console.print(f"  [green]✓[/green] {m.value}")
    else:
        print("\nExisting Measures:")
        for m in ctx.existing_measures:
            print(f"  + {m.value}")

    # Print applicable ECMs
    if console and RICH_AVAILABLE:
        table = Table(title="ECM Analysis Results")
        table.add_column("Category", style="cyan")
        table.add_column("ECM", style="white")
        table.add_column("Status", style="green")

        for ecm in filter_result['applicable']:
            table.add_row("Applicable", ecm.name, "✓")

        for item in filter_result['already_done']:
            table.add_row("Already Done", item['ecm'].name, f"⏭ {item['reason']}")

        for item in filter_result['not_applicable']:
            reasons = item.get('reasons', [])
            reason = reasons[0][1] if reasons else 'Technical constraint'
            table.add_row("Not Applicable", item['ecm'].name, f"✗ {reason}")

        console.print(table)
    else:
        print(f"\nApplicable ECMs ({len(filter_result['applicable'])}):")
        for ecm in filter_result['applicable']:
            print(f"  → {ecm.name}")

        print(f"\nAlready Done ({len(filter_result['already_done'])}):")
        for item in filter_result['already_done']:
            print(f"  ✓ {item['ecm'].name}: {item['reason']}")

        print(f"\nNot Applicable ({len(filter_result['not_applicable'])}):")
        for item in filter_result['not_applicable']:
            reasons = item.get('reasons', [])
            reason = reasons[0][1] if reasons else 'Technical constraint'
            print(f"  ✗ {item['ecm'].name}: {reason}")

    return 0


def cmd_ecm_list(args, console):
    """List all available ECMs."""
    print_header(console, "Available Energy Conservation Measures")

    catalog = ECMCatalog()
    all_ecms = catalog.all()

    if console and RICH_AVAILABLE:
        table = Table()
        table.add_column("ID", style="cyan")
        table.add_column("Name", style="white")
        table.add_column("Category", style="green")
        table.add_column("Savings", style="yellow")

        for ecm in all_ecms:
            category = str(ecm.category.value) if hasattr(ecm.category, 'value') else str(ecm.category)
            savings = f"{ecm.typical_savings_percent:.0f}%" if hasattr(ecm, 'typical_savings_percent') else "N/A"
            table.add_row(
                ecm.id,
                ecm.name,
                category,
                savings
            )

        console.print(table)
    else:
        print("\nAvailable ECMs:")
        for ecm in all_ecms:
            print(f"  {ecm.id}: {ecm.name} ({ecm.category})")

    return 0


def cmd_batch(args, console):
    """Run batch analysis on multiple buildings."""
    buildings_dir = Path(args.buildings_dir)

    if not buildings_dir.exists():
        print_error(console, f"Directory not found: {buildings_dir}")
        return 1

    json_files = list(buildings_dir.glob("*.json"))
    if not json_files:
        print_error(console, f"No JSON files found in: {buildings_dir}")
        return 1

    print_header(console, f"RAIDEN - Batch Analysis ({len(json_files)} buildings)")

    results = []
    if console and RICH_AVAILABLE:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console
        ) as progress:
            task = progress.add_task("Analyzing buildings...", total=len(json_files))

            for json_file in json_files:
                try:
                    data = load_building_data(json_file)
                    ctx = build_context(data)
                    filter_result = filter_ecms(ctx)
                    results.append({
                        'file': json_file.name,
                        'address': ctx.address,
                        'applicable': len(filter_result['applicable']),
                        'existing': len(ctx.existing_measures),
                        'success': True
                    })
                except Exception as e:
                    results.append({
                        'file': json_file.name,
                        'error': str(e),
                        'success': False
                    })
                progress.advance(task)
    else:
        for i, json_file in enumerate(json_files):
            print(f"[{i+1}/{len(json_files)}] Processing {json_file.name}...")
            try:
                data = load_building_data(json_file)
                ctx = build_context(data)
                filter_result = filter_ecms(ctx)
                results.append({
                    'file': json_file.name,
                    'address': ctx.address,
                    'applicable': len(filter_result['applicable']),
                    'existing': len(ctx.existing_measures),
                    'success': True
                })
            except Exception as e:
                results.append({
                    'file': json_file.name,
                    'error': str(e),
                    'success': False
                })

    # Print summary
    successful = sum(1 for r in results if r['success'])
    print_success(console, f"Analyzed {successful}/{len(json_files)} buildings")

    if console and RICH_AVAILABLE:
        table = Table(title="Batch Results")
        table.add_column("Building", style="white")
        table.add_column("Existing", style="cyan")
        table.add_column("Applicable", style="green")
        table.add_column("Status", style="yellow")

        for r in results:
            if r['success']:
                table.add_row(
                    r['file'][:30],
                    str(r['existing']),
                    str(r['applicable']),
                    "✓"
                )
            else:
                table.add_row(
                    r['file'][:30],
                    "-",
                    "-",
                    f"✗ {r['error'][:20]}"
                )

        console.print(table)

    return 0


def cmd_analyze_address(args, console) -> int:
    """Analyze a building by address - THE VISION COMMAND!"""
    print_header(console, f"Raiden Address Analysis: {args.address}")

    # Build known data from CLI args
    known_data = {}
    if args.year:
        known_data['construction_year'] = args.year
    if args.apartments:
        known_data['num_apartments'] = args.apartments
    if args.area:
        known_data['atemp_m2'] = args.area
    if args.fund:
        known_data['current_fund_sek'] = args.fund
    if getattr(args, 'energy_cost', None):
        known_data['annual_energy_cost_sek'] = args.energy_cost

    try:
        from ..core.address_pipeline import AddressPipeline

        output_dir = Path(args.output)
        pipeline = AddressPipeline(output_dir=output_dir)

        if console and RICH_AVAILABLE:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TimeElapsedColumn(),
                console=console,
            ) as progress:
                task = progress.add_task("Analyzing building...", total=5)

                result = pipeline.analyze(
                    address=args.address,
                    known_data=known_data if known_data else None,
                    skip_simulation=args.skip_simulation,
                    generate_report=True,
                )

                progress.update(task, completed=5)
        else:
            result = pipeline.analyze(
                address=args.address,
                known_data=known_data if known_data else None,
                skip_simulation=args.skip_simulation,
                generate_report=True,
            )

        if result.success:
            print_success(console, f"Analysis complete in {result.processing_time_seconds:.1f}s")

            if result.building_data:
                bd = result.building_data
                print("\nBuilding Data:")
                print(f"  Address: {bd.address}")
                print(f"  Year: {bd.construction_year}")
                print(f"  Area: {bd.atemp_m2:,.0f} m²")
                print(f"  Facade: {bd.facade_material}")
                print(f"  Data sources: {', '.join(bd.data_sources)}")

            if result.effektvakt_result:
                eff = result.effektvakt_result
                print("\nEffektvakt Potential:")
                print(f"  Peak reduction: {eff.el_peak_reduction_kw:.0f} kW el, {eff.fv_peak_reduction_kw:.0f} kW fv")
                print(f"  Annual savings: {eff.total_annual_savings_sek:,.0f} SEK")

            if result.maintenance_plan:
                mp = result.maintenance_plan
                print("\nMaintenance Plan Summary:")
                print(f"  Total investment: {mp.total_investment_sek:,.0f} SEK")
                print(f"  30-year savings: {mp.total_savings_30yr_sek:,.0f} SEK")
                print(f"  NPV: {mp.net_present_value_sek:,.0f} SEK")
                print(f"  Break-even: {mp.break_even_year}")

            if result.report_path:
                print_success(console, f"Report saved to: {result.report_path}")
                print(f"\nOpen with: open {result.report_path}")

            return 0
        else:
            print_error(console, f"Analysis failed: {result.error}")
            return 1

    except ImportError as e:
        print_error(console, f"Import error: {e}")
        print("Make sure all dependencies are installed: pip install -e .")
        return 1
    except Exception as e:
        print_error(console, f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


def main():
    parser = argparse.ArgumentParser(
        description='Raiden - Swedish Building Energy Analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    raiden analyze examples/sjostaden_2/BRF_Sjostaden_2_enriched.json
    raiden analyze-address "Aktergatan 11, Stockholm"
    raiden batch data/buildings/ --parallel 4
    raiden ecm-list
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze a single building')
    analyze_parser.add_argument('building_json', help='Path to building JSON file')
    analyze_parser.add_argument('--ecms', help='Comma-separated list of ECMs to evaluate')
    analyze_parser.add_argument('--simulate', action='store_true', help='Run EnergyPlus simulation')
    analyze_parser.add_argument('--weather', help='Path to weather file')
    analyze_parser.add_argument('--output', help='Output directory')

    # Analyze Address command (THE VISION!)
    addr_parser = subparsers.add_parser('analyze-address', help='Analyze building by address (auto-fetch data)')
    addr_parser.add_argument('address', help='Swedish street address (e.g., "Aktergatan 11, Stockholm")')
    addr_parser.add_argument('--year', type=int, help='Construction year (if known)')
    addr_parser.add_argument('--apartments', type=int, help='Number of apartments (if known)')
    addr_parser.add_argument('--area', type=float, help='Heated area in m² (if known)')
    addr_parser.add_argument('--fund', type=float, help='Current maintenance fund in SEK')
    addr_parser.add_argument('--energy-cost', type=float, help='Annual energy cost in SEK')
    addr_parser.add_argument('--output', help='Output directory', default='./output')
    addr_parser.add_argument('--skip-simulation', action='store_true', help='Skip EnergyPlus simulation')

    # Batch command
    batch_parser = subparsers.add_parser('batch', help='Batch analyze multiple buildings')
    batch_parser.add_argument('buildings_dir', help='Directory containing building JSON files')
    batch_parser.add_argument('--parallel', type=int, default=1, help='Number of parallel workers')
    batch_parser.add_argument('--output', help='Output directory')

    # ECM list command
    subparsers.add_parser('ecm-list', help='List all available ECMs')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    console = create_console()

    if args.command == 'analyze':
        return cmd_analyze(args, console)
    elif args.command == 'analyze-address':
        return cmd_analyze_address(args, console)
    elif args.command == 'batch':
        return cmd_batch(args, console)
    elif args.command == 'ecm-list':
        return cmd_ecm_list(args, console)

    return 0


if __name__ == '__main__':
    sys.exit(main())
