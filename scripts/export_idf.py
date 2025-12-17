#!/usr/bin/env python3
"""
Export EnergyPlus IDF from BRF Sjöstaden 2 data.

Generates a base model IDF ready for simulation.
"""

import json
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.core.models import (
    BRFProperty,
    EnrichedBuilding,
    EnrichedBRFProperty,
    EnergyPlusReady,
    EnvelopeData,
    UValues,
    WindowToWallRatio,
    FacadeMaterial,
    EnrichmentMetadata,
    Geometry,
    BuildingProperties,
    ShadingAnalysis,
    SolarPotential,
)
from src.export.energyplus_idf import EnergyPlusExporter
from datetime import date
from rich.console import Console

console = Console()

# Paths
BRF_JSON = PROJECT_ROOT / "data" / "input" / "BRF_Sjostaden_2.json"
OUTPUT_DIR = PROJECT_ROOT / "examples" / "sjostaden_2" / "energyplus"
IDD_PATH = Path("/Applications/EnergyPlus-25-1-0/Energy+.idd")


def create_enriched_building(brf_building, specific_energy_kwh_sqm: float = 33.0):
    """Create an EnrichedBuilding from BRF data with U-value back-calculation."""

    props = brf_building.properties
    geom = brf_building.geometry

    # Back-calculate U-values from specific energy
    # Sjöstaden performs 70% better than era typical
    era_u = UValues(walls=0.25, roof=0.15, floor=0.20, windows=1.6)

    # Back-calculated values (from earlier session)
    back_calc_u = UValues(
        walls=0.11,
        roof=0.08,
        floor=0.10,
        windows=0.8,
        doors=1.5,
    )

    # WWR from AI + era blend
    wwr = WindowToWallRatio(
        north=0.14,
        south=0.20,
        east=0.17,
        west=0.17,
        average=0.17,
        source="ai_era_blend",
        confidence=0.7,
    )

    # Envelope
    envelope = EnvelopeData(
        window_to_wall_ratio=wwr,
        facade_material=FacadeMaterial.PLASTER,
        facade_material_confidence=0.8,
        u_values=back_calc_u,
        airtightness_n50=1.0,
    )

    # Convert coordinates to local (centered at origin)
    coords_3d = geom.coordinates_3d
    if coords_3d:
        xs = [c[0] for c in coords_3d]
        ys = [c[1] for c in coords_3d]
        center_x = sum(xs) / len(xs)
        center_y = sum(ys) / len(ys)

        local_coords = [(x - center_x, y - center_y) for x, y in zip(xs, ys)]
    else:
        local_coords = []

    # Shading analysis (minimal for this tall building)
    shading = ShadingAnalysis(
        annual_shadow_hours={},
        solar_access_factor=1.0,  # No significant shading
        primary_shading_sources=[],
        obstructions=[],
    )

    # Solar potential
    solar = SolarPotential(
        suitable_roof_area_sqm=735,  # 35% of total
        remaining_capacity_kwp=47.0,
        annual_yield_potential_kwh=40000,
        optimal_tilt_deg=35,
        optimal_azimuth_deg=180,
        shading_loss_pct=0,
        existing_pv_area_sqm=500,
        source="analysis",
    )

    # EnergyPlus ready data
    ep_ready = EnergyPlusReady(
        footprint_coords_local=local_coords,
        height_m=geom.height_meters,
        num_stories=props.dimensions.floors_above_ground,
        floor_to_floor_height_m=geom.height_meters / props.dimensions.floors_above_ground,
        envelope=envelope,
        shading=shading,
        solar_potential=solar,
        occupant_density_m2_per_person=30,
        lighting_power_density_w_per_m2=8,
        equipment_power_density_w_per_m2=12,
        infiltration_ach=1.0,
    )

    # Enrichment metadata
    metadata = EnrichmentMetadata(
        enrichment_date=date.today(),
        toolkit_version="1.0.0",
        data_sources=["brf_json", "energidek_pdf", "mapillary", "osm"],
        ai_models_used=["opencv_wwr", "hsv_material"],
    )

    return EnrichedBuilding(
        building_id=brf_building.building_id,
        original_geometry=geom,
        original_properties=props,
        energyplus_ready=ep_ready,
        enrichment_metadata=metadata,
    )


def main():
    console.print("\n[bold cyan]═══ EnergyPlus IDF Export ═══[/bold cyan]\n")

    # Load BRF data
    console.print(f"Loading BRF data from: {BRF_JSON}")
    with open(BRF_JSON) as f:
        brf_data = json.load(f)

    brf = BRFProperty(**brf_data)
    console.print(f"  ✓ Loaded: {brf.brf_name}")
    console.print(f"    Buildings: {len(brf.buildings)}")

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Initialize exporter
    console.print(f"\nInitializing EnergyPlus exporter...")
    console.print(f"  IDD path: {IDD_PATH}")

    exporter = EnergyPlusExporter(idd_path=IDD_PATH)

    # Export each building
    generated_files = []

    for brf_building in brf.buildings:
        console.print(f"\n[cyan]Processing Building {brf_building.building_id}[/cyan]")

        # Create enriched building
        enriched = create_enriched_building(brf_building)

        console.print(f"  Height: {enriched.energyplus_ready.height_m}m")
        console.print(f"  Stories: {enriched.energyplus_ready.num_stories}")
        console.print(f"  U-wall: {enriched.energyplus_ready.envelope.u_values.walls} W/m²K")

        # Export IDF
        output_path = OUTPUT_DIR / f"sjostaden_bldg{brf_building.building_id}_base.idf"

        result = exporter.export_building(enriched, output_path)

        if result:
            generated_files.append(result)
            console.print(f"  [green]✓ Exported: {result}[/green]")
        else:
            console.print(f"  [red]✗ Export failed[/red]")

    # Summary
    console.print(f"\n[bold green]═══ Export Complete ═══[/bold green]")
    console.print(f"Generated {len(generated_files)} IDF files in: {OUTPUT_DIR}")

    for f in generated_files:
        console.print(f"  • {f.name}")

    return generated_files


if __name__ == "__main__":
    main()
