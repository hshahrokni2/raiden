"""
BRF Energy Toolkit CLI.

Command-line interface for building metadata enrichment and energy simulation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .core.config import settings
from .core.models import EnrichedBRFProperty
from .ingest.brf_parser import BRFParser
from .ingest.osm_fetcher import OSMFetcher
from .ingest.overture_fetcher import OvertureFetcher, OvertureDataMerger
from .ai.facade_analyzer import FacadeAnalyzer, create_envelope_data
from .export.energyplus_idf import EnergyPlusExporter
from .export.enriched_json import EnrichedJSONExporter
from .visualization.building_3d import Building3DGenerator
from .visualization.server import start_viewer_server

app = typer.Typer(
    name="brf",
    help="BRF Energy Toolkit - Building metadata enrichment for energy simulation",
    add_completion=False,
)
console = Console()


@app.command()
def enrich(
    input_file: Path = typer.Argument(..., help="Input BRF JSON file"),
    output_dir: Path = typer.Option(
        Path("output"), "--output", "-o", help="Output directory"
    ),
    fetch_osm: bool = typer.Option(True, "--osm/--no-osm", help="Fetch OSM data"),
    fetch_overture: bool = typer.Option(
        True, "--overture/--no-overture", help="Fetch Overture data"
    ),
    analyze_facades: bool = typer.Option(
        False, "--analyze/--no-analyze", help="Run AI facade analysis"
    ),
    generate_idf: bool = typer.Option(
        False, "--idf/--no-idf", help="Generate EnergyPlus IDF"
    ),
):
    """
    Enrich a BRF JSON file with additional building metadata.

    Fetches data from OSM, Overture Maps, and optionally runs AI analysis.
    """
    console.print(Panel.fit(
        "[bold blue]BRF Energy Toolkit[/bold blue]\n"
        "Building Metadata Enrichment Pipeline",
        border_style="blue"
    ))

    # Ensure directories
    settings.ensure_dirs()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Parse input
    console.print(f"\n[cyan]Loading:[/cyan] {input_file}")
    parser = BRFParser()
    brf = parser.parse(input_file)

    console.print(f"[green]Loaded:[/green] {brf.brf_name}")
    console.print(f"  Buildings: {len(brf.buildings)}")
    console.print(f"  Energy class: {brf.summary.energy_class.value}")

    # Initialize enriched structure
    enriched = parser.initialize_enriched(brf)

    # Get bounding box for data fetching
    bbox = parser.get_property_bbox_wgs84(brf, buffer_m=150)
    console.print(f"\n[dim]Search area: {bbox[0]:.4f},{bbox[1]:.4f} to {bbox[2]:.4f},{bbox[3]:.4f}[/dim]")

    # Fetch OSM data
    if fetch_osm:
        console.print("\n[bold]Fetching OpenStreetMap data...[/bold]")
        osm = OSMFetcher()
        osm_buildings = osm.get_buildings_in_bbox(*bbox)
        osm_trees = osm.get_trees_in_bbox(*bbox)

        console.print(f"  Found {len(osm_buildings)} OSM buildings")
        console.print(f"  Found {len(osm_trees)} vegetation features")

        # Try to match and enrich
        for building in enriched.buildings:
            wgs84_coords = building.energyplus_ready.footprint_coords_wgs84
            match = osm.find_matching_osm_building(wgs84_coords, osm_buildings)
            if match:
                if match.get("material"):
                    console.print(f"  [dim]Building {building.building_id}: OSM material = {match['material']}[/dim]")
                if match.get("levels"):
                    console.print(f"  [dim]Building {building.building_id}: OSM levels = {match['levels']}[/dim]")

    # Fetch Overture data
    if fetch_overture:
        console.print("\n[bold]Fetching Overture Maps data...[/bold]")
        try:
            merger = OvertureDataMerger()
            for building in enriched.buildings:
                overture_data = merger.enrich_building(
                    building.energyplus_ready.footprint_coords_wgs84,
                    bbox
                )
                if overture_data:
                    console.print(f"  [dim]Building {building.building_id}: Overture data found[/dim]")
                    for key, value in overture_data.items():
                        if value:
                            console.print(f"    {key}: {value}")
        except Exception as e:
            console.print(f"  [yellow]Overture fetch failed: {e}[/yellow]")

    # AI facade analysis
    if analyze_facades:
        console.print("\n[bold]Running AI facade analysis...[/bold]")
        analyzer = FacadeAnalyzer()

        for building in enriched.buildings:
            # For now, use estimation (images would need to be provided)
            result = analyzer.analyze_building(
                facade_images=None,
                construction_year=building.original_properties.building_info.construction_year,
                renovation_year=building.original_properties.building_info.last_renovation_year,
                location=building.original_properties.location.address,
                use_ai=False,  # No images yet
            )

            # Update envelope data
            building.energyplus_ready.envelope = create_envelope_data(result)
            console.print(f"  Building {building.building_id}:")
            console.print(f"    Material: {result.facade_material.value}")
            if result.wwr:
                console.print(f"    WWR: {result.wwr.average:.1%}")

    # Export enriched JSON
    console.print("\n[bold]Exporting results...[/bold]")
    json_exporter = EnrichedJSONExporter()

    # Full export
    full_path = output_dir / f"{brf.brf_name.replace(' ', '_')}_enriched.json"
    json_exporter.export_full(enriched, full_path)

    # Summary export
    summary_path = output_dir / f"{brf.brf_name.replace(' ', '_')}_summary.json"
    json_exporter.export_summary(enriched, summary_path)

    # GeoJSON export
    geojson_path = output_dir / f"{brf.brf_name.replace(' ', '_')}.geojson"
    json_exporter.export_geojson(enriched, geojson_path)

    # Generate EnergyPlus IDF
    if generate_idf:
        console.print("\n[bold]Generating EnergyPlus IDF...[/bold]")
        idf_exporter = EnergyPlusExporter()
        idf_paths = idf_exporter.export_property(enriched, output_dir / "idf")
        for path in idf_paths:
            console.print(f"  [green]Generated:[/green] {path}")

    console.print("\n[bold green]Enrichment complete![/bold green]")


@app.command()
def visualize(
    input_file: Path = typer.Argument(..., help="Input BRF JSON file"),
    output: Path = typer.Option(
        Path("viewer.html"), "--output", "-o", help="Output HTML file"
    ),
    serve: bool = typer.Option(True, "--serve/--no-serve", help="Start local server"),
    port: int = typer.Option(8080, "--port", "-p", help="Server port"),
    color_by: str = typer.Option(
        "energy_class", "--color", "-c", help="Color scheme: energy_class, height, material"
    ),
):
    """
    Generate 3D visualization of BRF buildings.

    Creates an interactive Three.js viewer the client can open in their browser.
    """
    console.print(Panel.fit(
        "[bold blue]3D Building Visualization[/bold blue]",
        border_style="blue"
    ))

    # Parse input
    parser = BRFParser()
    brf = parser.parse(input_file)

    console.print(f"[cyan]Loaded:[/cyan] {brf.brf_name} ({len(brf.buildings)} buildings)")

    # Generate 3D scene
    generator = Building3DGenerator()
    scene_data = generator.generate_scene(brf, color_by=color_by)

    # Generate HTML viewer
    generator.generate_html_viewer(scene_data, output)
    console.print(f"[green]Generated viewer:[/green] {output}")

    if serve:
        console.print(f"\n[bold]Starting visualization server on port {port}...[/bold]")
        start_viewer_server(output, port=port, open_browser=True)


@app.command()
def info(
    input_file: Path = typer.Argument(..., help="Input BRF JSON file"),
):
    """
    Display information about a BRF JSON file.
    """
    parser = BRFParser()
    brf = parser.parse(input_file)

    # Header
    console.print(Panel.fit(
        f"[bold]{brf.brf_name}[/bold]",
        border_style="green"
    ))

    # Summary table
    table = Table(title="Property Summary")
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="white")

    table.add_row("Buildings", str(brf.summary.total_buildings))
    table.add_row("Apartments", str(brf.summary.total_apartments or "N/A"))
    table.add_row("Heated Area", f"{brf.summary.total_heated_area_sqm:,.0f} m²")
    table.add_row("Construction Year", str(brf.summary.construction_year))
    table.add_row("Energy Class", brf.summary.energy_class.value)
    table.add_row("Energy Use", f"{brf.summary.energy_performance_kwh_per_sqm} kWh/m²/year")
    table.add_row("Has Solar", "Yes" if brf.summary.has_solar_panels else "No")
    table.add_row("Location", brf.summary.location or "N/A")

    console.print(table)

    # Buildings detail
    console.print("\n[bold]Buildings:[/bold]")
    for building in brf.buildings:
        props = building.properties
        console.print(f"\n  [cyan]Building {building.building_id}[/cyan]")
        console.print(f"    Address: {props.location.address}")
        console.print(f"    Height: {props.dimensions.building_height_m}m ({props.dimensions.floors_above_ground} floors)")
        console.print(f"    Heated area: {props.dimensions.heated_area_sqm:,.0f} m²")

        # Heating breakdown
        heating = props.energy.heating
        if heating.ground_source_heat_pump_kwh > 0:
            console.print(f"    Heating: Ground source heat pump ({heating.ground_source_heat_pump_kwh:,.0f} kWh)")
        if heating.district_heating_kwh > 0:
            console.print(f"    Heating: District heating ({heating.district_heating_kwh:,.0f} kWh)")


@app.command()
def fetch_osm(
    lat: float = typer.Argument(..., help="Center latitude"),
    lon: float = typer.Argument(..., help="Center longitude"),
    radius: int = typer.Option(200, "--radius", "-r", help="Search radius in meters"),
    output: Path = typer.Option(None, "--output", "-o", help="Output JSON file"),
):
    """
    Fetch OSM building data for a location.

    Useful for testing data availability in an area.
    """
    osm = OSMFetcher()

    # Calculate bbox from center + radius (approximate)
    lat_offset = radius / 111000  # ~111km per degree
    lon_offset = radius / (111000 * abs(cos(lat * 3.14159 / 180)))

    bbox = (
        lon - lon_offset,
        lat - lat_offset,
        lon + lon_offset,
        lat + lat_offset,
    )

    console.print(f"[cyan]Fetching OSM data around ({lat}, {lon})...[/cyan]")

    buildings = osm.get_buildings_in_bbox(*bbox)
    trees = osm.get_trees_in_bbox(*bbox)

    console.print(f"[green]Found {len(buildings)} buildings[/green]")
    console.print(f"[green]Found {len(trees)} vegetation features[/green]")

    # Show sample data
    if buildings:
        console.print("\n[bold]Sample buildings:[/bold]")
        for b in buildings[:5]:
            console.print(f"  - {b.get('address', 'Unknown')} | {b.get('building_type', 'building')} | levels={b.get('levels', '?')}")

    if output:
        import json
        with open(output, "w") as f:
            json.dump({"buildings": buildings, "trees": trees}, f, indent=2)
        console.print(f"\n[green]Saved to {output}[/green]")


def cos(x: float) -> float:
    """Simple cosine for lat/lon calculations."""
    import math
    return math.cos(x)


@app.command()
def version():
    """Show version information."""
    from . import __version__
    console.print(f"BRF Energy Toolkit v{__version__}")


def main():
    """Entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()
