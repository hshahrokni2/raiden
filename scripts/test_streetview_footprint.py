#!/usr/bin/env python3
"""
Test Street View fetcher with actual building footprint.
"""

import sys
from pathlib import Path
import json

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from src.ingest.streetview_fetcher import StreetViewFacadeFetcher
from src.ai.wwr_detector import WWRDetector
from src.ai.material_classifier import MaterialClassifier
from rich.console import Console
from rich.table import Table

console = Console()


def main():
    console.print("=" * 60)
    console.print("[bold]STREET VIEW + FOOTPRINT ANALYSIS[/bold]")
    console.print("=" * 60)

    # Load building data with footprint
    data_file = Path("examples/sjostaden_2/BRF_Sjostaden_2_enriched.json")
    if not data_file.exists():
        console.print(f"[red]Data file not found: {data_file}[/red]")
        return

    with open(data_file) as f:
        building_data = json.load(f)

    # Get footprint from buildings array
    buildings = building_data.get("buildings", [])
    if not buildings:
        console.print("[red]No buildings found in data[/red]")
        return

    # Use first building
    building = buildings[0]
    footprint_coords = building.get("wgs84_footprint")

    if not footprint_coords:
        console.print("[red]No footprint found in building data[/red]")
        return

    # Convert to GeoJSON format
    footprint = {
        "type": "Polygon",
        "coordinates": [footprint_coords]
    }

    console.print(f"\n[bold]Building:[/bold] {building_data.get('brf_name', 'Unknown')}")
    console.print(f"[bold]Address:[/bold] {building.get('address', 'Unknown')}")
    console.print(f"[bold]Footprint points:[/bold] {len(footprint_coords)}")

    # Calculate centroid for display
    coords = footprint.get('coordinates', [[]])[0]
    if coords:
        centroid_lat = sum(c[1] for c in coords) / len(coords)
        centroid_lon = sum(c[0] for c in coords) / len(coords)
        console.print(f"[bold]Centroid:[/bold] {centroid_lat:.6f}, {centroid_lon:.6f}")

    # Initialize fetcher
    console.print("\n[bold]Fetching Street View images...[/bold]")
    fetcher = StreetViewFacadeFetcher()

    # Fetch facade images
    images = fetcher.fetch_facade_images(footprint)

    if not images:
        console.print("[red]No Street View images found![/red]")
        return

    console.print(f"\n[green]✓ Got {len(images)} facade images[/green]")

    # Save images
    output_dir = Path("output_streetview_footprint")
    output_dir.mkdir(exist_ok=True)

    for orientation, sv_image in images.items():
        img_path = output_dir / f"facade_{orientation}.jpg"
        sv_image.image.save(img_path)
        console.print(f"  Saved: {img_path}")

    # Analyze with LangSAM
    console.print("\n" + "=" * 60)
    console.print("[bold]LANGSAM ANALYSIS[/bold]")
    console.print("=" * 60)

    detector = WWRDetector(backend="lang_sam", device="mps")
    classifier = MaterialClassifier(device="mps")

    results = {}
    for orientation, sv_image in images.items():
        console.print(f"\n[bold]{orientation} Facade[/bold] (camera: {sv_image.camera_lat:.5f}, {sv_image.camera_lon:.5f}, heading: {sv_image.heading:.0f}°)")

        # Detect WWR
        wwr, conf = detector.calculate_wwr(sv_image.image, crop_facade=True)
        results[orientation] = {"wwr": wwr, "conf": conf}
        console.print(f"  WWR: {wwr:.1%} (confidence: {conf:.1%})")

        # Detect material
        mat_pred = classifier.classify(sv_image.image)
        results[orientation]["material"] = mat_pred.material.value
        results[orientation]["material_conf"] = mat_pred.confidence
        console.print(f"  Material: {mat_pred.material.value} ({mat_pred.confidence:.1%})")

    # Summary table
    console.print("\n" + "=" * 60)
    console.print("[bold]SUMMARY[/bold]")
    console.print("=" * 60)

    table = Table(title="Facade Analysis Results")
    table.add_column("Facade", style="cyan")
    table.add_column("WWR", justify="right")
    table.add_column("Confidence", justify="right")
    table.add_column("Material", justify="center")

    for orientation in ['N', 'E', 'S', 'W']:
        if orientation in results:
            r = results[orientation]
            table.add_row(
                orientation,
                f"{r['wwr']:.1%}",
                f"{r['conf']:.1%}",
                r.get('material', '-'),
            )
        else:
            table.add_row(orientation, "-", "-", "-")

    console.print(table)

    # Average WWR
    wwrs = [r['wwr'] for r in results.values() if r['wwr'] > 0]
    if wwrs:
        avg_wwr = sum(wwrs) / len(wwrs)
        console.print(f"\n[bold]Average WWR:[/bold] {avg_wwr:.1%}")

    console.print()


if __name__ == "__main__":
    main()
