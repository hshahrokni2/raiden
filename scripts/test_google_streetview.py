#!/usr/bin/env python3
"""
Test WWR detection with Google Street View images.
"""

import os
import sys
from pathlib import Path
import requests
from PIL import Image
from io import BytesIO

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

# Building location
LAT = 59.301855
LON = 18.104948

# Google API key
API_KEY = os.getenv("BRF_GOOGLE_API_KEY")

def fetch_streetview_image(lat: float, lon: float, heading: int, pitch: int = 10, fov: int = 90, radius: int = 100) -> Image.Image:
    """Fetch a Street View image."""
    url = "https://maps.googleapis.com/maps/api/streetview"
    params = {
        "size": "640x480",
        "location": f"{lat},{lon}",
        "heading": heading,
        "pitch": pitch,
        "fov": fov,
        "radius": radius,  # Search radius for nearest panorama
        "source": "outdoor",  # Prefer outdoor imagery
        "key": API_KEY,
    }

    response = requests.get(url, params=params)
    if response.status_code == 200:
        return Image.open(BytesIO(response.content))
    else:
        print(f"Failed to fetch image: {response.status_code}")
        return None


def main():
    print("=" * 60)
    print("GOOGLE STREET VIEW + LANGSAM TEST")
    print("=" * 60)
    print(f"Location: {LAT}, {LON}")
    print(f"API Key: {'Found' if API_KEY else 'MISSING'}")
    print()

    if not API_KEY:
        print("ERROR: No Google API key found!")
        return

    # Fetch images from 4 directions
    output_dir = Path("output_streetview_test")
    output_dir.mkdir(exist_ok=True)

    directions = {
        "N": 0,
        "E": 90,
        "S": 180,
        "W": 270,
    }

    images = {}
    for direction, heading in directions.items():
        print(f"Fetching {direction} facade (heading={heading})...")
        img = fetch_streetview_image(LAT, LON, heading)
        if img:
            img_path = output_dir / f"streetview_{direction}.jpg"
            img.save(img_path)
            images[direction] = img
            print(f"  Saved to {img_path}")
        else:
            print(f"  Failed!")

    print()
    print("=" * 60)
    print("ANALYZING WITH LANGSAM")
    print("=" * 60)

    # Initialize LangSAM WWR detector
    from src.ai.wwr_detector import WWRDetector
    from src.ai.material_classifier import MaterialClassifier

    detector = WWRDetector(backend="lang_sam", device="mps")
    classifier = MaterialClassifier(device="mps")

    results = {}
    for direction, img in images.items():
        print(f"\nAnalyzing {direction} facade...")

        # Detect WWR
        wwr, confidence = detector.calculate_wwr(img, crop_facade=True)
        results[direction] = {"wwr": wwr, "confidence": confidence}
        print(f"  WWR: {wwr:.1%} (confidence: {confidence:.1%})")

        # Detect material
        mat_pred = classifier.classify(img)
        print(f"  Material: {mat_pred.material.value} ({mat_pred.confidence:.1%})")

    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)

    for direction, data in results.items():
        print(f"  {direction}: WWR = {data['wwr']:.1%}")

    avg_wwr = sum(d["wwr"] for d in results.values()) / len(results) if results else 0
    print(f"\n  Average WWR: {avg_wwr:.1%}")
    print()


if __name__ == "__main__":
    main()
