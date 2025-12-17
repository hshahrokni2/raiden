"""
Facade Image Fetcher - Multiple sources for street-level imagery.

Supports:
- Mapillary (CC-BY-SA, best coverage in Sweden)
- OpenStreetCam/KartaView (open source alternative)
- Manual image upload/path specification

All sources work without API keys or accounts for basic usage.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Literal
from datetime import datetime

import requests


@dataclass
class FacadeImage:
    """Metadata and path for a facade image."""

    image_id: str
    source: Literal["mapillary", "kartaview", "manual", "generated"]

    # Location
    latitude: float
    longitude: float

    # Camera orientation
    compass_angle: float  # 0-360, where 0 is North

    # Image data
    local_path: Optional[Path] = None
    url: Optional[str] = None
    thumbnail_url: Optional[str] = None

    # Metadata
    captured_at: Optional[datetime] = None
    camera_make: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None

    # Computed
    facade_direction: Optional[str] = None  # N, S, E, W based on compass_angle
    distance_to_building_m: Optional[float] = None

    def __post_init__(self):
        """Compute facade direction from compass angle."""
        if self.compass_angle is not None:
            # Compass angle is where camera is pointing
            # 0° = North, 90° = East, 180° = South, 270° = West
            angle = self.compass_angle % 360
            if 315 <= angle or angle < 45:
                self.facade_direction = "N"
            elif 45 <= angle < 135:
                self.facade_direction = "E"
            elif 135 <= angle < 225:
                self.facade_direction = "S"
            else:
                self.facade_direction = "W"


@dataclass
class ImageSearchResult:
    """Results from searching for facade images."""

    images: list[FacadeImage] = field(default_factory=list)
    total_found: int = 0
    search_bbox: tuple[float, float, float, float] = (0, 0, 0, 0)
    source: str = ""
    error: Optional[str] = None


class MapillaryFetcher:
    """
    Fetch street-level images from Mapillary.

    Uses Mapillary Graph API v4 for:
    - Searching images by location
    - Getting image metadata (compass angle, captured_at, etc.)
    - Downloading images at various resolutions

    Requires an access token from https://www.mapillary.com/dashboard/developers
    """

    # Mapillary Graph API v4
    API_URL = "https://graph.mapillary.com"

    # Image fields to request
    IMAGE_FIELDS = "id,captured_at,compass_angle,geometry,thumb_1024_url,thumb_2048_url"

    def __init__(self, access_token: Optional[str] = None, cache_dir: Optional[Path] = None):
        """
        Initialize Mapillary fetcher.

        Args:
            access_token: Mapillary access token (required for API access)
            cache_dir: Directory for caching responses
        """
        self.access_token = access_token
        self.cache_dir = cache_dir or Path("data/cache/mapillary")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "BRF-Energy-Toolkit/0.1 (building-analysis)",
            "Authorization": f"OAuth {access_token}" if access_token else "",
        })

    def search_images(
        self,
        bbox: tuple[float, float, float, float],
        max_results: int = 50,
    ) -> ImageSearchResult:
        """
        Search for images within a bounding box using Mapillary Graph API.

        Args:
            bbox: (min_lon, min_lat, max_lon, max_lat) in WGS84
            max_results: Maximum number of images to return

        Returns:
            ImageSearchResult with found images
        """
        if not self.access_token:
            return ImageSearchResult(
                error="Mapillary access token required",
                source="mapillary"
            )

        min_lon, min_lat, max_lon, max_lat = bbox

        try:
            # Use the images endpoint with bbox filter
            url = f"{self.API_URL}/images"
            params = {
                "access_token": self.access_token,
                "fields": self.IMAGE_FIELDS,
                "bbox": f"{min_lon},{min_lat},{max_lon},{max_lat}",
                "limit": min(max_results, 100),  # API limit
            }

            response = self.session.get(url, params=params, timeout=30)

            if response.status_code == 401:
                return ImageSearchResult(
                    error="Invalid Mapillary access token",
                    source="mapillary"
                )

            if response.status_code != 200:
                return ImageSearchResult(
                    error=f"Mapillary API error: {response.status_code}",
                    source="mapillary"
                )

            data = response.json()
            images = []

            for item in data.get("data", []):
                try:
                    # Parse geometry
                    geom = item.get("geometry", {})
                    coords = geom.get("coordinates", [0, 0])

                    # Get image URLs
                    thumb_url = item.get("thumb_2048_url") or item.get("thumb_1024_url")

                    # Parse timestamp
                    captured_at = None
                    if item.get("captured_at"):
                        try:
                            captured_at = datetime.fromtimestamp(item["captured_at"] / 1000)
                        except:
                            pass

                    image = FacadeImage(
                        image_id=str(item.get("id", "")),
                        source="mapillary",
                        latitude=coords[1] if len(coords) > 1 else 0,
                        longitude=coords[0] if len(coords) > 0 else 0,
                        compass_angle=float(item.get("compass_angle", 0)),
                        url=thumb_url,
                        thumbnail_url=item.get("thumb_1024_url"),
                        captured_at=captured_at,
                    )
                    images.append(image)

                except (KeyError, ValueError, TypeError) as e:
                    continue

            return ImageSearchResult(
                images=images,
                total_found=len(images),
                search_bbox=bbox,
                source="mapillary"
            )

        except requests.RequestException as e:
            return ImageSearchResult(
                error=str(e),
                source="mapillary"
            )

    def get_images_near_building(
        self,
        building_coords: list[tuple[float, float]],
        search_radius_m: float = 50,
        orientations: list[str] = ["N", "S", "E", "W"],
    ) -> dict[str, list[FacadeImage]]:
        """
        Find images looking at each facade of a building.

        Args:
            building_coords: List of (lon, lat) coordinates of building footprint
            search_radius_m: How far to search from building
            orientations: Which facades to find images for

        Returns:
            Dict mapping orientation to list of images
        """
        # Calculate building centroid
        lons = [c[0] for c in building_coords]
        lats = [c[1] for c in building_coords]
        center_lon = sum(lons) / len(lons)
        center_lat = sum(lats) / len(lats)

        # Create search bbox
        # Approximate: 1 degree lat ≈ 111km, 1 degree lon ≈ 111km * cos(lat)
        lat_offset = search_radius_m / 111000
        lon_offset = search_radius_m / (111000 * math.cos(math.radians(center_lat)))

        bbox = (
            center_lon - lon_offset,
            center_lat - lat_offset,
            center_lon + lon_offset,
            center_lat + lat_offset,
        )

        # Search for images
        result = self.search_images(bbox, max_results=100)

        # Group by facade direction
        by_direction: dict[str, list[FacadeImage]] = {d: [] for d in orientations}

        for image in result.images:
            if image.facade_direction in by_direction:
                by_direction[image.facade_direction].append(image)

        return by_direction


class WikimediaCommonsFetcher:
    """
    Fetch geotagged images from Wikimedia Commons.

    Fully open, no authentication required.
    Images are CC-licensed.
    """

    API_URL = "https://commons.wikimedia.org/w/api.php"

    def __init__(self, cache_dir: Optional[Path] = None):
        self.cache_dir = cache_dir or Path("data/cache/wikimedia")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "BRF-Energy-Toolkit/0.1 (building-analysis; github.com/komilion)"
        })

    def search_images(
        self,
        center: tuple[float, float],
        radius_m: int = 500,
        max_results: int = 50,
    ) -> ImageSearchResult:
        """
        Search for geotagged images near a location.

        Args:
            center: (latitude, longitude) of search center
            radius_m: Search radius in meters
            max_results: Maximum results

        Returns:
            ImageSearchResult
        """
        lat, lon = center

        try:
            # Geosearch for nearby files
            params = {
                "action": "query",
                "format": "json",
                "list": "geosearch",
                "gscoord": f"{lat}|{lon}",
                "gsradius": min(radius_m, 10000),  # Max 10km
                "gslimit": max_results,
                "gsnamespace": 6,  # File namespace
            }

            response = self.session.get(self.API_URL, params=params, timeout=15)

            if response.status_code != 200:
                return ImageSearchResult(error=f"Wikimedia API error: {response.status_code}")

            data = response.json()
            results = data.get("query", {}).get("geosearch", [])

            # Get image info for each result
            images = []
            if results:
                titles = [r["title"] for r in results]
                image_info = self._get_image_info(titles)

                for result in results:
                    title = result["title"]
                    info = image_info.get(title, {})

                    # Skip non-image files
                    if not info.get("url", "").lower().endswith((".jpg", ".jpeg", ".png")):
                        continue

                    images.append(FacadeImage(
                        image_id=str(result.get("pageid", title)),
                        source="wikimedia",
                        latitude=result.get("lat", lat),
                        longitude=result.get("lon", lon),
                        compass_angle=0,  # Unknown - would need EXIF
                        url=info.get("url"),
                        thumbnail_url=info.get("thumburl"),
                        width=info.get("width"),
                        height=info.get("height"),
                    ))

            return ImageSearchResult(
                images=images,
                total_found=len(images),
                search_bbox=(lon - 0.01, lat - 0.01, lon + 0.01, lat + 0.01),
                source="wikimedia_commons"
            )

        except Exception as e:
            return ImageSearchResult(error=str(e), source="wikimedia_commons")

    def _get_image_info(self, titles: list[str]) -> dict:
        """Get URLs and metadata for image files."""
        if not titles:
            return {}

        params = {
            "action": "query",
            "format": "json",
            "titles": "|".join(titles[:50]),  # API limit
            "prop": "imageinfo",
            "iiprop": "url|size|extmetadata",
            "iiurlwidth": 1024,  # Get thumbnail
        }

        try:
            response = self.session.get(self.API_URL, params=params, timeout=15)
            if response.status_code != 200:
                return {}

            data = response.json()
            pages = data.get("query", {}).get("pages", {})

            result = {}
            for page in pages.values():
                title = page.get("title", "")
                imageinfo = page.get("imageinfo", [{}])[0]
                result[title] = {
                    "url": imageinfo.get("url"),
                    "thumburl": imageinfo.get("thumburl"),
                    "width": imageinfo.get("width"),
                    "height": imageinfo.get("height"),
                }

            return result

        except Exception:
            return {}


class KartaViewFetcher:
    """
    Fetch images from KartaView (formerly OpenStreetCam).

    Fully open source alternative to Mapillary.
    API: https://kartaview.org/api-doc
    """

    API_URL = "https://api.openstreetcam.org/2.0"

    def __init__(self, cache_dir: Optional[Path] = None):
        self.cache_dir = cache_dir or Path("data/cache/kartaview")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "BRF-Energy-Toolkit/0.1"
        })

    def search_images(
        self,
        bbox: tuple[float, float, float, float],
        max_results: int = 50,
    ) -> ImageSearchResult:
        """
        Search for images within a bounding box.

        Args:
            bbox: (min_lon, min_lat, max_lon, max_lat)
            max_results: Maximum results

        Returns:
            ImageSearchResult
        """
        min_lon, min_lat, max_lon, max_lat = bbox

        try:
            # KartaView nearby photos endpoint
            url = f"{self.API_URL}/photo/"
            params = {
                "bbTopLeft": f"{max_lat},{min_lon}",
                "bbBottomRight": f"{min_lat},{max_lon}",
                "limit": max_results,
            }

            response = self.session.get(url, params=params, timeout=15)

            if response.status_code != 200:
                return ImageSearchResult(
                    error=f"KartaView API error: {response.status_code}"
                )

            data = response.json()

            images = []
            for item in data.get("result", {}).get("data", []):
                try:
                    image = FacadeImage(
                        image_id=str(item.get("id", "")),
                        source="kartaview",
                        latitude=float(item.get("lat", 0)),
                        longitude=float(item.get("lng", 0)),
                        compass_angle=float(item.get("heading", 0)),
                        url=item.get("lth_name"),  # Large thumbnail
                        thumbnail_url=item.get("th_name"),
                        captured_at=datetime.fromisoformat(item["date_added"]) if item.get("date_added") else None,
                    )
                    images.append(image)
                except (KeyError, ValueError):
                    continue

            return ImageSearchResult(
                images=images,
                total_found=len(images),
                search_bbox=bbox,
                source="kartaview"
            )

        except Exception as e:
            return ImageSearchResult(
                error=str(e),
                source="kartaview"
            )


class ManualImageLoader:
    """
    Load manually provided facade images.

    Useful when user has their own photos of the building.
    """

    def __init__(self, image_dir: Optional[Path] = None):
        self.image_dir = image_dir or Path("data/images/manual")
        self.image_dir.mkdir(parents=True, exist_ok=True)

    def load_image(
        self,
        image_path: Path,
        latitude: float,
        longitude: float,
        compass_angle: float,
        metadata: Optional[dict] = None,
    ) -> FacadeImage:
        """
        Load a manually provided image.

        Args:
            image_path: Path to the image file
            latitude: Camera latitude
            longitude: Camera longitude
            compass_angle: Direction camera was pointing (0=N, 90=E, etc.)
            metadata: Optional additional metadata

        Returns:
            FacadeImage object
        """
        # Generate ID from path
        image_id = hashlib.md5(str(image_path).encode()).hexdigest()[:12]

        # Copy to managed directory if not already there
        if image_path.parent != self.image_dir:
            dest = self.image_dir / f"{image_id}_{image_path.name}"
            if not dest.exists():
                import shutil
                shutil.copy2(image_path, dest)
            local_path = dest
        else:
            local_path = image_path

        return FacadeImage(
            image_id=image_id,
            source="manual",
            latitude=latitude,
            longitude=longitude,
            compass_angle=compass_angle,
            local_path=local_path,
        )

    def load_from_directory(
        self,
        directory: Path,
        default_location: tuple[float, float],
    ) -> list[FacadeImage]:
        """
        Load all images from a directory.

        Attempts to extract GPS from EXIF if available.
        """
        images = []

        for ext in ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]:
            for image_path in directory.glob(ext):
                # Try to get GPS from EXIF
                lat, lon, heading = self._extract_gps(image_path)

                if lat is None:
                    lat, lon = default_location
                    heading = 0  # Unknown

                images.append(FacadeImage(
                    image_id=hashlib.md5(str(image_path).encode()).hexdigest()[:12],
                    source="manual",
                    latitude=lat,
                    longitude=lon,
                    compass_angle=heading or 0,
                    local_path=image_path,
                ))

        return images

    def _extract_gps(self, image_path: Path) -> tuple[Optional[float], Optional[float], Optional[float]]:
        """Extract GPS coordinates from image EXIF."""
        try:
            from PIL import Image
            from PIL.ExifTags import TAGS, GPSTAGS

            img = Image.open(image_path)
            exif = img._getexif()

            if not exif:
                return None, None, None

            gps_info = {}
            for tag, value in exif.items():
                if TAGS.get(tag) == "GPSInfo":
                    for gps_tag, gps_value in value.items():
                        gps_info[GPSTAGS.get(gps_tag, gps_tag)] = gps_value

            if not gps_info:
                return None, None, None

            # Parse coordinates
            def dms_to_decimal(dms, ref):
                degrees = float(dms[0])
                minutes = float(dms[1])
                seconds = float(dms[2])
                decimal = degrees + minutes/60 + seconds/3600
                if ref in ['S', 'W']:
                    decimal = -decimal
                return decimal

            lat = dms_to_decimal(
                gps_info.get("GPSLatitude", (0, 0, 0)),
                gps_info.get("GPSLatitudeRef", "N")
            )
            lon = dms_to_decimal(
                gps_info.get("GPSLongitude", (0, 0, 0)),
                gps_info.get("GPSLongitudeRef", "E")
            )
            heading = float(gps_info.get("GPSImgDirection", 0))

            return lat, lon, heading

        except Exception:
            return None, None, None


class FacadeImageFetcher:
    """
    Unified facade image fetcher.

    Tries multiple sources in order of preference:
    1. Wikimedia Commons (fully open, no auth)
    2. Manual images (user-provided)
    3. Mapillary (needs token)
    4. KartaView (may need auth now)
    """

    def __init__(
        self,
        mapillary_token: Optional[str] = None,
        cache_dir: Optional[Path] = None,
    ):
        self.cache_dir = cache_dir or Path("data/cache/images")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.wikimedia = WikimediaCommonsFetcher(
            cache_dir=self.cache_dir / "wikimedia"
        )
        self.mapillary = MapillaryFetcher(
            access_token=mapillary_token,
            cache_dir=self.cache_dir / "mapillary"
        )
        self.kartaview = KartaViewFetcher(
            cache_dir=self.cache_dir / "kartaview"
        )
        self.manual = ManualImageLoader(
            image_dir=self.cache_dir / "manual"
        )

    def fetch_for_building(
        self,
        building_coords: list[tuple[float, float]],
        building_id: str,
        search_radius_m: float = 100,
        orientations: list[str] = ["N", "S", "E", "W"],
        prefer_source: str = "any",
    ) -> dict[str, list[FacadeImage]]:
        """
        Fetch facade images for a building from all available sources.

        Args:
            building_coords: Building footprint in (lon, lat) pairs
            building_id: Identifier for caching
            search_radius_m: Search radius around building
            orientations: Which facades to fetch images for
            prefer_source: "wikimedia", "mapillary", "kartaview", or "any"

        Returns:
            Dict mapping orientation to list of images
        """
        results: dict[str, list[FacadeImage]] = {d: [] for d in orientations}
        unclassified: list[FacadeImage] = []

        # Calculate centroid
        lons = [c[0] for c in building_coords]
        lats = [c[1] for c in building_coords]
        center_lon = sum(lons) / len(lons)
        center_lat = sum(lats) / len(lats)

        # Try Wikimedia Commons first (fully open, good quality)
        if prefer_source in ["wikimedia", "any"]:
            wiki_results = self.wikimedia.search_images(
                center=(center_lat, center_lon),
                radius_m=int(search_radius_m),
                max_results=50
            )
            for image in wiki_results.images:
                if image.facade_direction and image.facade_direction in results:
                    results[image.facade_direction].append(image)
                else:
                    unclassified.append(image)

        # Try Mapillary (needs token for downloads)
        if prefer_source in ["mapillary", "any"]:
            try:
                mapillary_results = self.mapillary.get_images_near_building(
                    building_coords, search_radius_m, orientations
                )
                for direction, images in mapillary_results.items():
                    results[direction].extend(images)
            except Exception:
                pass  # Mapillary may fail without token

        # Try KartaView
        if prefer_source in ["kartaview", "any"]:
            lat_offset = search_radius_m / 111000
            lon_offset = search_radius_m / (111000 * math.cos(math.radians(center_lat)))
            bbox = (
                center_lon - lon_offset,
                center_lat - lat_offset,
                center_lon + lon_offset,
                center_lat + lat_offset,
            )
            try:
                kv_results = self.kartaview.search_images(bbox)
                for image in kv_results.images:
                    if image.facade_direction in results:
                        results[image.facade_direction].append(image)
            except Exception:
                pass

        # Add unclassified images to a special key
        if unclassified:
            results["unclassified"] = unclassified

        return results

    def search_wikimedia(
        self,
        center: tuple[float, float],
        radius_m: int = 500,
    ) -> ImageSearchResult:
        """
        Direct search on Wikimedia Commons.

        Args:
            center: (latitude, longitude)
            radius_m: Search radius

        Returns:
            ImageSearchResult with found images
        """
        return self.wikimedia.search_images(center, radius_m)

    def download_image(
        self,
        image: FacadeImage,
        output_dir: Optional[Path] = None,
    ) -> Optional[Path]:
        """
        Download an image to local storage.

        Args:
            image: FacadeImage to download
            output_dir: Where to save (defaults to cache)

        Returns:
            Local path to downloaded image, or None if failed
        """
        if image.local_path and image.local_path.exists():
            return image.local_path

        if not image.url:
            return None

        output_dir = output_dir or self.cache_dir / "downloaded"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Determine file extension from URL
        ext = ".jpg"
        if ".png" in image.url.lower():
            ext = ".png"

        output_path = output_dir / f"{image.source}_{image.image_id}{ext}"

        if output_path.exists():
            image.local_path = output_path
            return output_path

        try:
            headers = {"User-Agent": "BRF-Energy-Toolkit/0.1 (building-analysis)"}
            response = requests.get(image.url, headers=headers, timeout=60)
            if response.status_code == 200 and len(response.content) > 1000:
                with open(output_path, 'wb') as f:
                    f.write(response.content)
                image.local_path = output_path
                return output_path
        except Exception as e:
            print(f"Download failed: {e}")

        return None


# Convenience function
def fetch_facade_images(
    building_coords: list[tuple[float, float]],
    building_id: str = "building",
    search_radius_m: float = 50,
) -> dict[str, list[FacadeImage]]:
    """
    Quick function to fetch facade images for a building.

    Args:
        building_coords: Building footprint as (lon, lat) pairs
        building_id: Identifier for the building
        search_radius_m: How far to search

    Returns:
        Dict mapping facade direction (N/S/E/W) to list of images
    """
    fetcher = FacadeImageFetcher()
    return fetcher.fetch_for_building(
        building_coords=building_coords,
        building_id=building_id,
        search_radius_m=search_radius_m,
    )
