"""
Swedish Weather File Downloader.

Downloads EnergyPlus weather files (.epw) from EnergyPlus.net
for Swedish locations.
"""

import urllib.request
import zipfile
import io
from pathlib import Path
from typing import Optional, Dict, List
import logging

logger = logging.getLogger(__name__)

# Swedish weather stations available from EnergyPlus.net
# Format: {city_name: (filename_base, WMO_ID, lat, lon)}
SWEDISH_WEATHER_STATIONS = {
    'stockholm': ('SWE_Stockholm.Arlanda.024600_IWEC', '024600', 59.65, 17.95),
    'arlanda': ('SWE_Stockholm.Arlanda.024600_IWEC', '024600', 59.65, 17.95),
    'goteborg': ('SWE_Goteborg.Landvetter.025130_IWEC', '025130', 57.67, 12.30),
    'gothenburg': ('SWE_Goteborg.Landvetter.025130_IWEC', '025130', 57.67, 12.30),
    'malmo': ('SWE_Malmo.023660_IWEC', '023660', 55.53, 13.37),
    'kiruna': ('SWE_Kiruna.020440_IWEC', '020440', 67.82, 20.33),
    'lulea': ('SWE_Lulea.021920_IWEC', '021920', 65.55, 22.13),
    'ostersund': ('SWE_Ostersund.022360_IWEC', '022360', 63.18, 14.50),
    'sundsvall': ('SWE_Sundsvall-Harnos.023660_IWEC', '023660', 62.53, 17.45),
    'karlstad': ('SWE_Karlstad.024180_IWEC', '024180', 59.37, 13.47),
    'visby': ('SWE_Visby.025900_IWEC', '025900', 57.67, 18.35),
    'linkoping': ('SWE_Linkoping.024620_IWEC', '024620', 58.40, 15.53),
    'norrkoping': ('SWE_Norrkoping.Bravalla.024640_IWEC', '024640', 58.62, 16.10),
}

# Base URLs for EnergyPlus weather files
# Primary: climate.onebuilding.org (most up-to-date)
# Fallback: EnergyPlus GitHub
ONEBUILDING_URL = "https://climate.onebuilding.org/WMO_Region_6_Europe/SWE_Sweden/"
GITHUB_WEATHER_URL = "https://raw.githubusercontent.com/NREL/EnergyPlus/develop/weather/"


class WeatherDownloader:
    """Download and manage EnergyPlus weather files for Swedish locations."""

    def __init__(self, cache_dir: Optional[Path] = None):
        """
        Initialize weather downloader.

        Args:
            cache_dir: Directory to cache downloaded files. Defaults to ~/.raiden/weather/
        """
        self.cache_dir = cache_dir or Path.home() / '.raiden' / 'weather'
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def list_available(self) -> List[str]:
        """List available Swedish weather stations."""
        return sorted(set(SWEDISH_WEATHER_STATIONS.keys()))

    def get_station_info(self, city: str) -> Optional[Dict]:
        """Get information about a weather station."""
        city_lower = city.lower()
        if city_lower in SWEDISH_WEATHER_STATIONS:
            filename, wmo, lat, lon = SWEDISH_WEATHER_STATIONS[city_lower]
            return {
                'city': city,
                'filename': filename,
                'wmo_id': wmo,
                'latitude': lat,
                'longitude': lon,
            }
        return None

    def find_nearest(self, lat: float, lon: float) -> str:
        """
        Find nearest weather station to given coordinates.

        Args:
            lat: Latitude
            lon: Longitude

        Returns:
            City name of nearest station
        """
        min_dist = float('inf')
        nearest = 'stockholm'

        for city, (_, _, slat, slon) in SWEDISH_WEATHER_STATIONS.items():
            # Simple Euclidean distance (good enough for Sweden)
            dist = ((lat - slat) ** 2 + (lon - slon) ** 2) ** 0.5
            if dist < min_dist:
                min_dist = dist
                nearest = city

        return nearest

    def download(self, city: str, force: bool = False) -> Path:
        """
        Download weather file for a Swedish city.

        Args:
            city: City name (e.g., 'stockholm', 'goteborg')
            force: Force re-download even if cached

        Returns:
            Path to downloaded .epw file

        Raises:
            ValueError: If city not recognized
            RuntimeError: If download fails
        """
        city_lower = city.lower()

        if city_lower not in SWEDISH_WEATHER_STATIONS:
            available = ', '.join(sorted(set(
                k for k in SWEDISH_WEATHER_STATIONS.keys()
                if not k.endswith('arlanda') and k != 'gothenburg'
            )))
            raise ValueError(f"Unknown city: {city}. Available: {available}")

        filename_base = SWEDISH_WEATHER_STATIONS[city_lower][0]
        epw_filename = f"{filename_base}.epw"
        cache_path = self.cache_dir / epw_filename

        # Return cached if exists and not forcing
        if cache_path.exists() and not force:
            logger.info(f"Using cached weather file: {cache_path}")
            return cache_path

        # Check for local test fixture (Stockholm)
        local_fixture = Path(__file__).parent.parent.parent / 'tests' / 'fixtures' / 'stockholm.epw'
        if city_lower in ['stockholm', 'arlanda'] and local_fixture.exists():
            logger.info(f"Using local fixture: {local_fixture}")
            # Copy to cache
            import shutil
            shutil.copy(local_fixture, cache_path)
            print(f"Using local weather file: {local_fixture}")
            return cache_path

        # Try multiple download sources
        urls_to_try = self._get_download_urls(filename_base, city_lower)

        logger.info(f"Downloading weather file for {city}...")
        print(f"Downloading weather file for {city}...")

        last_error = None
        for url in urls_to_try:
            try:
                logger.debug(f"Trying: {url}")
                return self._download_from_url(url, cache_path)
            except Exception as e:
                logger.debug(f"Failed: {url} - {e}")
                last_error = e
                continue

        # All sources failed - provide manual download instructions
        manual_url = "https://climate.onebuilding.org/WMO_Region_6_Europe/SWE_Sweden/"
        print(f"\nAutomatic download failed. Please download manually from:")
        print(f"  {manual_url}")
        print(f"\nSave the EPW file to: {cache_path}")
        raise RuntimeError(f"Failed to download weather file for {city}. Download manually from {manual_url}")

    def _get_download_urls(self, filename_base: str, city: str) -> List[str]:
        """Get list of URLs to try for downloading weather file."""
        urls = []

        # Climate.onebuilding.org format - try zip first, then direct epw
        city_path = self._get_onebuilding_path(city)
        if city_path:
            urls.append(f"{ONEBUILDING_URL}{city_path}/{filename_base}.zip")
            urls.append(f"{ONEBUILDING_URL}{city_path}/{filename_base}.epw")

        return urls

    def _get_onebuilding_path(self, city: str) -> Optional[str]:
        """Get the subdirectory path for a city on climate.onebuilding.org."""
        # Mapping of cities to their subdirectory paths
        paths = {
            'stockholm': 'STO_Stockholm',
            'arlanda': 'STO_Stockholm',
            'goteborg': 'VAS_Vastra.Gotaland',
            'gothenburg': 'VAS_Vastra.Gotaland',
            'malmo': 'SKA_Skane',
            'kiruna': 'NBD_Norrbotten',
            'lulea': 'NBD_Norrbotten',
            'ostersund': 'JAM_Jamtland',
            'sundsvall': 'VNL_Vasternorrland',
            'karlstad': 'VML_Varmland',
            'visby': 'GOT_Gotland',
            'linkoping': 'OGE_Ostergotland',
            'norrkoping': 'OGE_Ostergotland',
        }
        return paths.get(city)

    def _download_from_url(self, url: str, cache_path: Path) -> Path:
        """Download file from URL and save to cache."""
        try:
            with urllib.request.urlopen(url, timeout=60) as response:
                data = response.read()

            # Check if it's a zip file
            if url.endswith('.zip'):
                with zipfile.ZipFile(io.BytesIO(data)) as zf:
                    epw_files = [f for f in zf.namelist() if f.endswith('.epw')]
                    if not epw_files:
                        raise RuntimeError("No .epw file found in archive")
                    with zf.open(epw_files[0]) as epw_file:
                        content = epw_file.read()
                        with open(cache_path, 'wb') as f:
                            f.write(content)
            else:
                # Direct EPW file
                with open(cache_path, 'wb') as f:
                    f.write(data)

            logger.info(f"Weather file downloaded: {cache_path}")
            print(f"Weather file saved: {cache_path}")
            return cache_path

        except urllib.error.URLError as e:
            raise RuntimeError(f"Download failed: {e}")
        except zipfile.BadZipFile as e:
            raise RuntimeError(f"Invalid archive: {e}")

    def get_or_download(self, city: str) -> Path:
        """
        Get weather file from cache or download if not present.

        Args:
            city: City name

        Returns:
            Path to .epw file
        """
        return self.download(city, force=False)

    def get_for_location(self, lat: float, lon: float) -> Path:
        """
        Get weather file for nearest station to given coordinates.

        Args:
            lat: Latitude
            lon: Longitude

        Returns:
            Path to .epw file for nearest station
        """
        nearest = self.find_nearest(lat, lon)
        logger.info(f"Nearest weather station for ({lat}, {lon}): {nearest}")
        return self.get_or_download(nearest)


def download_weather(city: str = 'stockholm', cache_dir: Optional[Path] = None) -> Path:
    """
    Convenience function to download Swedish weather file.

    Args:
        city: Swedish city name (stockholm, goteborg, malmo, etc.)
        cache_dir: Optional cache directory

    Returns:
        Path to downloaded .epw file
    """
    downloader = WeatherDownloader(cache_dir)
    return downloader.download(city)


def list_swedish_stations() -> None:
    """Print list of available Swedish weather stations."""
    print("\nAvailable Swedish Weather Stations:")
    print("-" * 50)

    seen = set()
    for city, (filename, wmo, lat, lon) in sorted(SWEDISH_WEATHER_STATIONS.items()):
        if filename in seen:
            continue
        seen.add(filename)
        print(f"  {city.title():15} WMO {wmo}  ({lat:.2f}°N, {lon:.2f}°E)")

    print("\nUsage:")
    print("  from src.utils.weather_downloader import download_weather")
    print("  epw_path = download_weather('stockholm')")


if __name__ == '__main__':
    list_swedish_stations()
