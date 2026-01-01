"""
Historical Street View Fetcher

Fetches Street View images from multiple time periods for the same location.
Different dates provide:
- Different lighting conditions
- Different seasons (trees with/without leaves)
- Multiple angles/camera positions
- Building evolution over time
"""

import os
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
from PIL import Image
from io import BytesIO
import requests

from rich.console import Console

console = Console()

# Try to import streetview package
try:
    import streetview as sv
    STREETVIEW_AVAILABLE = True
except ImportError:
    STREETVIEW_AVAILABLE = False
    console.print("[yellow]streetview package not installed. Install with: pip install streetview[/yellow]")


@dataclass
class HistoricalPanorama:
    """A historical Street View panorama."""
    pano_id: str
    date: Optional[str]  # YYYY-MM format
    lat: float
    lon: float
    year: Optional[int] = None

    def __post_init__(self):
        if self.date:
            try:
                self.year = int(self.date.split('-')[0])
            except:
                self.year = None


@dataclass
class HistoricalImage:
    """Downloaded historical Street View image."""
    image: Image.Image
    pano_id: str
    date: Optional[str]
    lat: float
    lon: float
    heading: float
    pitch: float


class HistoricalStreetViewFetcher:
    """
    Fetch Street View images from multiple time periods.

    Uses the streetview package to access historical panoramas
    that aren't available through the official Static API.
    """

    def __init__(self, api_key: str = None):
        """
        Initialize fetcher.

        Args:
            api_key: Google API key (optional, used for some requests)
        """
        self.api_key = api_key or os.getenv("BRF_GOOGLE_API_KEY")

        if not STREETVIEW_AVAILABLE:
            raise ImportError("streetview package required. Install with: pip install streetview")

    def search_panoramas(
        self,
        lat: float,
        lon: float,
        max_results: int = 20,
    ) -> List[HistoricalPanorama]:
        """
        Search for all available panoramas near a location.

        Returns panoramas from different time periods.
        """
        try:
            panos = sv.search_panoramas(lat, lon)

            results = []
            for p in panos[:max_results]:
                results.append(HistoricalPanorama(
                    pano_id=p.pano_id,
                    date=p.date,
                    lat=p.lat,
                    lon=p.lon,
                ))

            # Sort by date (newest first), None dates at end
            results.sort(key=lambda x: x.date or "0000-00", reverse=True)

            return results

        except Exception as e:
            console.print(f"[red]Panorama search failed: {e}[/red]")
            return []

    def fetch_historical_images(
        self,
        lat: float,
        lon: float,
        heading: float,
        pitches: List[int] = [5, 25, 45],
        years_back: int = 5,
        max_per_year: int = 1,
    ) -> List[HistoricalImage]:
        """
        Fetch images from multiple time periods.

        Args:
            lat, lon: Location coordinates
            heading: Camera heading (compass direction)
            pitches: List of pitch angles to capture
            years_back: How many years of history to fetch
            max_per_year: Max images per year

        Returns:
            List of HistoricalImage objects
        """
        # Search for panoramas
        panos = self.search_panoramas(lat, lon, max_results=50)

        if not panos:
            console.print("[yellow]No historical panoramas found[/yellow]")
            return []

        # Filter panoramas with dates
        dated_panos = [p for p in panos if p.date]

        # Group by year
        by_year: Dict[int, List[HistoricalPanorama]] = {}
        for p in dated_panos:
            if p.year:
                if p.year not in by_year:
                    by_year[p.year] = []
                by_year[p.year].append(p)

        # Select panoramas from available years (sorted newest first)
        # Instead of only looking at recent years, select from whatever years are available
        available_years = sorted(by_year.keys(), reverse=True)
        selected_panos = []

        # Take up to years_back different years from available panoramas
        for year in available_years[:years_back + 1]:
            selected_panos.extend(by_year[year][:max_per_year])

        console.print(f"[dim]Historical: {len(selected_panos)} panoramas from {len(by_year)} years[/dim]")

        # Fetch images from selected panoramas
        images = []
        for pano in selected_panos:
            for pitch in pitches:
                img = self._fetch_image(pano, heading, pitch)
                if img:
                    images.append(img)

        return images

    def _fetch_image(
        self,
        pano: HistoricalPanorama,
        heading: float,
        pitch: float,
        fov: int = 90,
        size: str = "640x480",
    ) -> Optional[HistoricalImage]:
        """Fetch a single image from a panorama."""
        # Use the official API with pano_id
        url = "https://maps.googleapis.com/maps/api/streetview"
        params = {
            "pano": pano.pano_id,
            "heading": heading,
            "pitch": pitch,
            "fov": fov,
            "size": size,
            "key": self.api_key,
        }

        try:
            response = requests.get(url, params=params, timeout=30)
            if response.status_code == 200 and 'image' in response.headers.get('content-type', ''):
                img = Image.open(BytesIO(response.content))
                return HistoricalImage(
                    image=img,
                    pano_id=pano.pano_id,
                    date=pano.date,
                    lat=pano.lat,
                    lon=pano.lon,
                    heading=heading,
                    pitch=pitch,
                )
        except Exception as e:
            console.print(f"[dim]Failed to fetch {pano.date}: {e}[/dim]")

        return None

    def fetch_multi_year_facades(
        self,
        lat: float,
        lon: float,
        headings: Dict[str, float] = None,
        pitches: List[int] = [5, 25, 45],
        years_back: int = 3,
    ) -> Dict[str, List[HistoricalImage]]:
        """
        Fetch facade images from multiple years for all directions.

        Args:
            lat, lon: Building center coordinates
            headings: Dict of orientation -> heading (e.g., {'N': 0, 'E': 90})
            pitches: Pitch angles to capture
            years_back: Years of history to include

        Returns:
            Dict mapping orientation to list of images
        """
        if headings is None:
            headings = {'N': 0, 'E': 90, 'S': 180, 'W': 270}

        results = {}

        for orientation, heading in headings.items():
            console.print(f"  Fetching {orientation} historical images...")
            images = self.fetch_historical_images(
                lat, lon, heading,
                pitches=pitches,
                years_back=years_back,
            )
            results[orientation] = images
            console.print(f"    ✓ Got {len(images)} images")

        return results


def fetch_historical_facades(
    lat: float,
    lon: float,
    years_back: int = 3,
) -> Dict[str, List[HistoricalImage]]:
    """
    Convenience function to fetch historical facade images.
    """
    fetcher = HistoricalStreetViewFetcher()
    return fetcher.fetch_multi_year_facades(lat, lon, years_back=years_back)
