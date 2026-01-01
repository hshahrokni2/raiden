#!/usr/bin/env python3
"""
Test improved MaterialClassifierV2 with exclusion-based masking and patch sampling.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from PIL import Image
from rich.console import Console
from rich.table import Table

console = Console()


def test_material_classification():
    """Test material classification on Sjostaden images."""

    # Find facade images - try multiple directories
    image_dirs = [
        Path("output_test_pipeline/streetview_facades"),
        Path("output_full_pipeline/streetview_facades"),
        Path("output_enhanced_streetview"),
        Path("output_streetview_footprint"),
    ]

    facade_images = []
    image_dir = None

    for dir_path in image_dirs:
        if dir_path.exists():
            found = list(dir_path.glob("facade_*.jpg"))
            if found:
                facade_images = found
                image_dir = dir_path
                console.print(f"[green]Found {len(found)} images in {dir_path}[/green]")
                break

    if not facade_images:
        console.print("[red]No facade images found in any directory![/red]")
        return

    console.print(f"\n[bold]Testing material classification on {len(facade_images)} images[/bold]\n")

    # Load classifier
    from src.ai.material_classifier_v2 import MaterialClassifierV2

    classifier = MaterialClassifierV2(device="mps")

    # Load images
    pil_images = []
    for img_path in facade_images[:12]:  # Test on up to 12 images
        try:
            pil_images.append(Image.open(img_path).convert("RGB"))
            console.print(f"  Loaded: {img_path.name}")
        except Exception as e:
            console.print(f"  [red]Failed: {img_path.name} - {e}[/red]")

    if not pil_images:
        console.print("[red]No images loaded![/red]")
        return

    console.print(f"\n[cyan]Running classification with exclusion-based masking + patch sampling...[/cyan]\n")

    # Classify with SAM wall isolation
    result = classifier.classify_multi_image(pil_images, use_sam_crop=True)

    console.print("\n" + "=" * 60)
    console.print("[bold green]MATERIAL CLASSIFICATION RESULTS[/bold green]")
    console.print("=" * 60)

    console.print(f"\n[bold]Material:[/bold] {result.material.upper()}")
    console.print(f"[bold]Confidence:[/bold] {result.confidence:.1%}")
    console.print(f"[bold]Votes:[/bold] {result.vote_count}/{result.total_images}")

    console.print("\n[dim]Vote distribution:[/dim]")

    # Sort by score
    sorted_dist = sorted(result.vote_distribution.items(), key=lambda x: x[1], reverse=True)
    for material, score in sorted_dist:
        bar = "█" * int(score * 40)
        color = "green" if material == result.material else "dim"
        console.print(f"  [{color}]{material:10s} {score:5.1%} {bar}[/{color}]")

    # Expected result for Sjostaden (2003 modern Swedish building)
    expected = "render"  # Should be plastered/rendered

    if result.material == expected:
        console.print(f"\n[bold green]✓ CORRECT! Detected '{result.material}' (expected '{expected}')[/bold green]")
    else:
        console.print(f"\n[bold yellow]! Detected '{result.material}' (expected '{expected}')[/bold yellow]")

        # Show why - what was the render score?
        render_score = result.vote_distribution.get("render", 0)
        console.print(f"  Render scored: {render_score:.1%}")

        if render_score > 0.20:
            console.print("  [dim]Render is a strong candidate - consider tuning thresholds[/dim]")


def test_single_image_with_debug():
    """Test single image with debug visualization."""

    # Find facade images
    image_dirs = [
        Path("output_test_pipeline/streetview_facades"),
        Path("output_full_pipeline/streetview_facades"),
        Path("output_enhanced_streetview"),
    ]

    facade_images = []
    image_dir = None

    for dir_path in image_dirs:
        if dir_path.exists():
            found = list(dir_path.glob("facade_*.jpg"))
            if found:
                facade_images = found
                image_dir = dir_path
                break

    if not facade_images:
        console.print("[red]No images found![/red]")
        return

    # Take first image
    img_path = facade_images[0]
    console.print(f"\n[bold]Debug: Single image analysis of {img_path.name}[/bold]\n")

    from src.ai.material_classifier_v2 import MaterialClassifierV2
    import numpy as np

    classifier = MaterialClassifierV2(device="mps")
    classifier._lazy_init()

    pil_image = Image.open(img_path).convert("RGB")

    # Get wall mask
    console.print("[cyan]Getting exclusion-based wall mask...[/cyan]")
    wall_mask = classifier._get_wall_mask(pil_image)

    if wall_mask is not None:
        wall_ratio = np.mean(wall_mask > 128)
        console.print(f"  Wall mask coverage: {wall_ratio:.1%}")

        # Save mask visualization
        mask_vis = Image.fromarray(wall_mask)
        mask_vis.save(image_dir / "debug_wall_mask.png")
        console.print(f"  Saved: debug_wall_mask.png")

        # Sample patches
        patches = classifier._sample_wall_patches(
            np.array(pil_image), wall_mask, num_patches=5, patch_size=224
        )
        console.print(f"  Sampled {len(patches)} wall patches")

        # Save patches
        for i, patch in enumerate(patches):
            patch.save(image_dir / f"debug_patch_{i}.png")
        console.print(f"  Saved patches to debug_patch_*.png")
    else:
        console.print("  [yellow]Wall mask failed - using full image[/yellow]")

    # Classify
    material, conf, scores = classifier.classify_single(pil_image, wall_mask)

    console.print(f"\n[bold]Single image result: {material} ({conf:.1%})[/bold]")

    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    for mat, score in sorted_scores[:4]:
        console.print(f"  {mat:10s} {score:.1%}")


if __name__ == "__main__":
    console.print("[bold blue]Material Classifier V2 Test[/bold blue]")
    console.print("=" * 60)

    # Run debug first
    test_single_image_with_debug()

    console.print("\n")

    # Then full test
    test_material_classification()
