"""
Configuration management for BRF Energy Toolkit.
"""

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings.

    Can be configured via environment variables or .env file.
    """

    model_config = SettingsConfigDict(
        env_prefix="BRF_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Paths
    data_dir: Path = Field(default=Path("data"))
    cache_dir: Path = Field(default=Path("data/cache"))
    models_dir: Path = Field(default=Path("data/models"))

    # API Keys (optional - for enhanced features)
    mapillary_token: str | None = Field(default=None, description="Mapillary API token")
    google_api_key: str | None = Field(default=None, description="Google Maps API key")

    # Processing settings
    default_buffer_meters: float = Field(default=100.0, description="Buffer around buildings for data fetching")
    image_resolution: int = Field(default=2048, description="Target image resolution for analysis")

    # AI Model settings
    ai_device: Literal["cpu", "cuda", "mps"] = Field(default="cpu", description="Device for AI inference")
    sam_model_type: Literal["vit_h", "vit_l", "vit_b"] = Field(default="vit_b", description="SAM model size")

    # EnergyPlus settings
    energyplus_idd_path: Path | None = Field(default=None, description="Path to Energy+.idd")
    weather_file_dir: Path = Field(default=Path("data/weather"), description="EPW weather files directory")
    default_weather_file: str = Field(default="SWE_Stockholm.Arlanda.024600_IWEC.epw")

    # Swedish defaults
    default_climate_zone: str = Field(default="III", description="Swedish climate zone (I-IV)")
    default_indoor_temp_heating: float = Field(default=21.0, description="Indoor heating setpoint (°C)")
    default_indoor_temp_cooling: float = Field(default=26.0, description="Indoor cooling setpoint (°C)")

    # 3D Visualization
    viz_port: int = Field(default=8080, description="Port for visualization server")

    @property
    def input_dir(self) -> Path:
        return self.data_dir / "input"

    @property
    def output_dir(self) -> Path:
        return self.data_dir / "enriched"

    @property
    def imagery_dir(self) -> Path:
        return self.data_dir / "imagery"

    def ensure_dirs(self) -> None:
        """Create all required directories."""
        for dir_path in [
            self.data_dir,
            self.cache_dir,
            self.models_dir,
            self.input_dir,
            self.output_dir,
            self.imagery_dir,
            self.weather_file_dir,
        ]:
            dir_path.mkdir(parents=True, exist_ok=True)


# Global settings instance
settings = Settings()
