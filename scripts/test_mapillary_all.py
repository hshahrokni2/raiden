#!/usr/bin/env python3
"""
Fetch ALL Mapillary photos around the building and analyze with LangSAM.
"""

import os
import sys
from pathlib import Path
import requests
from PIL import Image
from io import BytesIO
import math

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

# Building location - BRF Sjöstaden 2
LAT = 59.301855
LON = 18.104948

# Mapillary token
TOKEN = os.getenv("MAPILLARY_TOKEN")


def search_mapillary_images(lat: float, lon: float, radius_m: int = 100) -> list:
    """Search for all Mapillary images around a point."""

    # Convert radius to degrees (rough approximation)
    delta_lat = radius_m / 111000
    delta_lon = radius_m / (111000 * math.cos(math.radians(lat)))

    bbox = f"{lon - delta_lon},{lat - delta_lat},{lon + delta_lon},{lat + delta_lat}"

    url = "https://graph.mapillary.com/images"
    params = {
        "access_token": TOKEN,
        "fields": "id,thumb_1024_url,thumb_2048_url,computed_compass_angle,geometry,captured_at",
        "bbox": bbox,
        "limit": 100,
    }

    response = requests.get(url, params=params)
    if response.status_code == 200:
        data = response.json()
        return data.get("data", [])
    else:
        print(f"Mapillary API error: {response.status_code}")
        print(response.text)
        return []


def download_image(url: str) -> Image.Image:
    """Download an image from URL."""
    response = requests.get(url, timeout=30)
    if response.status_code == 200:
        return Image.open(BytesIO(response.content))
    return None


def compass_to_direction(angle: float) -> str:
    """Convert compass angle to cardinal direction."""
    # Compass angle is where camera is POINTING
    # 0 = North, 90 = East, 180 = South, 270 = West
    if angle is None:
        return "?"
    if 315 <= angle or angle < 45:
        return "N"
    elif 45 <= angle < 135:
        return "E"
    elif 135 <= angle < 225:
        return "S"
    else:
        return "W"


def main():
    print("=" * 60)
    print("MAPILLARY ALL IMAGES + LANGSAM TEST")
    print("=" * 60)
    print(f"Location: {LAT}, {LON}")
    print(f"Token: {'Found' if TOKEN else 'MISSING'}")
    print()

    if not TOKEN:
        print("ERROR: No Mapillary token found!")
        return

    # Search for images - expand radius to 200m for more coverage
    print("Searching Mapillary for images within 200m...")
    images_data = search_mapillary_images(LAT, LON, radius_m=200)
    print(f"Found {len(images_data)} images!")
    print()

    if not images_data:
        return

    # Download and organize by direction
    output_dir = Path("output_mapillary_test")
    output_dir.mkdir(exist_ok=True)

    images_by_direction = {"N": [], "E": [], "S": [], "W": [], "?": []}

    for i, img_data in enumerate(images_data):
        img_id = img_data.get("id")
        angle = img_data.get("computed_compass_angle")
        url = img_data.get("thumb_1024_url") or img_data.get("thumb_2048_url")

        direction = compass_to_direction(angle)

        print(f"  [{i+1}/{len(images_data)}] Image {img_id}: heading {angle:.0f}° → {direction}")

        if url:
            img = download_image(url)
            if img:
                img_path = output_dir / f"{direction}_{img_id}.jpg"
                img.save(img_path)
                images_by_direction[direction].append({
                    "id": img_id,
                    "path": img_path,
                    "image": img,
                    "angle": angle,
                })

    print()
    print("Images by direction:")
    for d, imgs in images_by_direction.items():
        print(f"  {d}: {len(imgs)} images")

    print()
    print("=" * 60)
    print("ANALYZING WITH LANGSAM")
    print("=" * 60)

    # Initialize detectors
    from src.ai.wwr_detector import WWRDetector
    from src.ai.material_classifier import MaterialClassifier

    detector = WWRDetector(backend="lang_sam", device="mps")
    classifier = MaterialClassifier(device="mps")

    wwr_results = {"N": [], "E": [], "S": [], "W": []}
    material_votes = []

    for direction in ["N", "E", "S", "W"]:
        imgs = images_by_direction[direction]
        if not imgs:
            continue

        print(f"\n{direction} Facade ({len(imgs)} images):")

        for img_data in imgs[:5]:  # Analyze up to 5 per direction
            img = img_data["image"]
            img_id = img_data["id"]

            # Detect WWR
            wwr, conf = detector.calculate_wwr(img, crop_facade=True)
            wwr_results[direction].append({"wwr": wwr, "conf": conf})
            print(f"  {img_id}: WWR={wwr:.1%} (conf={conf:.1%})")

            # Detect material
            mat_pred = classifier.classify(img)
            material_votes.append((mat_pred.material.value, mat_pred.confidence))
            print(f"           Material={mat_pred.material.value} ({mat_pred.confidence:.1%})")

    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)

    final_wwr = {}
    for direction in ["N", "E", "S", "W"]:
        results = wwr_results[direction]
        if results:
            # Weight by confidence
            total_weight = sum(r["conf"] for r in results)
            if total_weight > 0:
                weighted_wwr = sum(r["wwr"] * r["conf"] for r in results) / total_weight
            else:
                weighted_wwr = sum(r["wwr"] for r in results) / len(results)
            final_wwr[direction] = weighted_wwr
            print(f"  {direction}: WWR = {weighted_wwr:.1%} (from {len(results)} images)")
        else:
            print(f"  {direction}: No images")

    if final_wwr:
        avg_wwr = sum(final_wwr.values()) / len(final_wwr)
        print(f"\n  Average WWR: {avg_wwr:.1%}")

    # Material consensus
    if material_votes:
        from collections import Counter
        mat_counts = Counter(m for m, c in material_votes)
        most_common = mat_counts.most_common(1)[0]
        print(f"\n  Detected Material: {most_common[0]} ({most_common[1]} votes)")

    print()


if __name__ == "__main__":
    main()
