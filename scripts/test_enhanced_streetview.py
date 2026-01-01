#!/usr/bin/env python3
"""
Test enhanced Street View analysis:
1. SAM-based facade cropping
2. Multi-image fetching per facade
3. Consensus-based WWR calculation
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
import numpy as np

console = Console()


def main():
    console.print("=" * 70)
    console.print("[bold]ENHANCED STREET VIEW ANALYSIS[/bold]")
    console.print("SAM facade cropping + Multi-image consensus")
    console.print("=" * 70)

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
        console.print("[red]No buildings found[/red]")
        return

    building = buildings[0]
    footprint_coords = building.get("wgs84_footprint")
    footprint = {"type": "Polygon", "coordinates": [footprint_coords]}

    console.print(f"\n[bold]Building:[/bold] {building_data.get('brf_name', 'Unknown')}")
    console.print(f"[bold]Address:[/bold] {building.get('address', 'Unknown')}")

    # Initialize components
    fetcher = StreetViewFacadeFetcher()
    detector = WWRDetector(backend="lang_sam", device="mps")
    classifier = MaterialClassifier(device="mps")

    # Output directory
    output_dir = Path("output_enhanced_streetview")
    output_dir.mkdir(exist_ok=True)

    # Fetch multiple images per facade
    console.print("\n[bold cyan]Phase 1: Fetching multiple images per facade...[/bold cyan]")
    multi_images = fetcher.fetch_multi_facade_images(footprint, images_per_facade=3)

    total_images = sum(len(imgs) for imgs in multi_images.values())
    console.print(f"\n[green]Total images fetched: {total_images}[/green]")

    # Save all images
    for orientation, imgs in multi_images.items():
        for i, sv_img in enumerate(imgs):
            img_path = output_dir / f"facade_{orientation}_{i+1}.jpg"
            sv_img.image.save(img_path)

    # Analyze each image with both methods
    console.print("\n[bold cyan]Phase 2: Analyzing with SAM facade cropping...[/bold cyan]")

    results = {'N': [], 'E': [], 'S': [], 'W': []}
    materials = []

    for orientation in ['N', 'E', 'S', 'W']:
        imgs = multi_images.get(orientation, [])
        if not imgs:
            console.print(f"  {orientation}: No images")
            continue

        console.print(f"\n[bold]{orientation} Facade ({len(imgs)} images):[/bold]")

        for i, sv_img in enumerate(imgs):
            # Standard analysis (no SAM crop)
            wwr_std, conf_std = detector.calculate_wwr(
                sv_img.image, crop_facade=True, use_sam_crop=False
            )

            # SAM-enhanced analysis
            wwr_sam, conf_sam = detector.calculate_wwr(
                sv_img.image, crop_facade=True, use_sam_crop=True
            )

            results[orientation].append({
                'wwr_std': wwr_std,
                'conf_std': conf_std,
                'wwr_sam': wwr_sam,
                'conf_sam': conf_sam,
            })

            console.print(f"  [{i+1}] Standard: {wwr_std:.1%} ({conf_std:.1%}) | SAM: {wwr_sam:.1%} ({conf_sam:.1%})")

            # Material (just once per facade)
            if i == 0:
                mat_pred = classifier.classify(sv_img.image)
                materials.append((orientation, mat_pred.material.value, mat_pred.confidence))

    # Calculate consensus results
    console.print("\n" + "=" * 70)
    console.print("[bold]CONSENSUS RESULTS[/bold]")
    console.print("=" * 70)

    table = Table(title="WWR Comparison: Standard vs SAM-Enhanced")
    table.add_column("Facade", style="cyan")
    table.add_column("Standard WWR", justify="right")
    table.add_column("Standard Conf", justify="right")
    table.add_column("SAM WWR", justify="right")
    table.add_column("SAM Conf", justify="right")
    table.add_column("Δ Conf", justify="right")

    final_wwr_std = {}
    final_wwr_sam = {}
    final_conf_std = {}
    final_conf_sam = {}

    for orientation in ['N', 'E', 'S', 'W']:
        facade_results = results[orientation]
        if not facade_results:
            table.add_row(orientation, "-", "-", "-", "-", "-")
            continue

        # Weighted average by confidence (standard)
        total_weight_std = sum(r['conf_std'] for r in facade_results)
        if total_weight_std > 0:
            weighted_wwr_std = sum(r['wwr_std'] * r['conf_std'] for r in facade_results) / total_weight_std
            avg_conf_std = total_weight_std / len(facade_results)
        else:
            weighted_wwr_std = np.mean([r['wwr_std'] for r in facade_results])
            avg_conf_std = 0

        # Weighted average by confidence (SAM)
        total_weight_sam = sum(r['conf_sam'] for r in facade_results)
        if total_weight_sam > 0:
            weighted_wwr_sam = sum(r['wwr_sam'] * r['conf_sam'] for r in facade_results) / total_weight_sam
            avg_conf_sam = total_weight_sam / len(facade_results)
        else:
            weighted_wwr_sam = np.mean([r['wwr_sam'] for r in facade_results])
            avg_conf_sam = 0

        final_wwr_std[orientation] = weighted_wwr_std
        final_wwr_sam[orientation] = weighted_wwr_sam
        final_conf_std[orientation] = avg_conf_std
        final_conf_sam[orientation] = avg_conf_sam

        conf_delta = avg_conf_sam - avg_conf_std
        conf_delta_str = f"+{conf_delta:.1%}" if conf_delta > 0 else f"{conf_delta:.1%}"

        table.add_row(
            orientation,
            f"{weighted_wwr_std:.1%}",
            f"{avg_conf_std:.1%}",
            f"{weighted_wwr_sam:.1%}",
            f"{avg_conf_sam:.1%}",
            conf_delta_str,
        )

    console.print(table)

    # Overall averages
    if final_wwr_std:
        avg_std = np.mean(list(final_wwr_std.values()))
        avg_sam = np.mean(list(final_wwr_sam.values()))
        conf_std = np.mean(list(final_conf_std.values()))
        conf_sam = np.mean(list(final_conf_sam.values()))

        console.print(f"\n[bold]Average WWR (Standard):[/bold] {avg_std:.1%} (conf: {conf_std:.1%})")
        console.print(f"[bold]Average WWR (SAM):     [/bold] {avg_sam:.1%} (conf: {conf_sam:.1%})")
        console.print(f"[bold]Confidence improvement:[/bold] +{(conf_sam - conf_std):.1%}")

    # Material consensus
    if materials:
        console.print(f"\n[bold]Material Detections:[/bold]")
        for orientation, material, conf in materials:
            console.print(f"  {orientation}: {material} ({conf:.1%})")

    console.print()


if __name__ == "__main__":
    main()
