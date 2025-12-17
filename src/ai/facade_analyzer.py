"""
Combined facade analysis pipeline.

Orchestrates WWR detection, material classification, and other facade analysis.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from PIL import Image
from rich.console import Console
from rich.progress import Progress, TaskID

from ..core.models import (
    EnvelopeData,
    WindowToWallRatio,
    FacadeMaterial,
    UValues,
    estimate_u_values,
)
from .wwr_detector import WWRDetector, estimate_wwr_from_era
from .material_classifier import MaterialClassifier, estimate_material_from_era

console = Console()


@dataclass
class FacadeImage:
    """Reference to a facade image."""

    path: Path
    direction: Literal["north", "south", "east", "west"]
    source: str = "manual"  # or "mapillary", "google_streetview"
    heading: float | None = None
    latitude: float | None = None
    longitude: float | None = None


@dataclass
class FacadeAnalysisResult:
    """Complete analysis result for a building's facades."""

    wwr: WindowToWallRatio | None = None
    facade_material: FacadeMaterial = FacadeMaterial.UNKNOWN
    material_confidence: float = 0.0
    u_values: UValues | None = None
    analysis_details: dict[str, Any] = field(default_factory=dict)


class FacadeAnalyzer:
    """
    Unified facade analysis pipeline.

    Combines:
    - Window-to-Wall Ratio detection
    - Material classification
    - U-value estimation
    """

    def __init__(
        self,
        device: str = "cpu",
        wwr_backend: str = "lang_sam",
    ):
        """
        Initialize facade analyzer.

        Args:
            device: Device for AI inference
            wwr_backend: Backend for WWR detection
        """
        self.device = device
        self.wwr_detector = WWRDetector(backend=wwr_backend, device=device)
        self.material_classifier = MaterialClassifier(device=device)

    def analyze_building(
        self,
        facade_images: dict[str, FacadeImage | Path | str] | None = None,
        construction_year: int | None = None,
        renovation_year: int | None = None,
        location: str = "Stockholm",
        use_ai: bool = True,
    ) -> FacadeAnalysisResult:
        """
        Analyze all facades of a building.

        Args:
            facade_images: Dict mapping direction to image path/FacadeImage
            construction_year: Building construction year (for estimation fallback)
            renovation_year: Last renovation year
            location: Location name (for era-based estimation)
            use_ai: Whether to use AI models (if False, uses estimation only)

        Returns:
            FacadeAnalysisResult with all analysis data
        """
        result = FacadeAnalysisResult()

        # Analyze WWR
        if facade_images and use_ai:
            result.wwr = self._analyze_wwr(facade_images)
            result.analysis_details["wwr_source"] = "ai"
        elif construction_year:
            result.wwr = estimate_wwr_from_era(construction_year)
            result.analysis_details["wwr_source"] = "era_estimation"

        # Analyze material
        if facade_images and use_ai:
            material_result = self._analyze_material(facade_images)
            result.facade_material = material_result["material"]
            result.material_confidence = material_result["confidence"]
            result.analysis_details["material_source"] = "ai"
        elif construction_year:
            result.facade_material = estimate_material_from_era(construction_year, location)
            result.material_confidence = 0.5
            result.analysis_details["material_source"] = "era_estimation"

        # Estimate U-values
        if construction_year:
            result.u_values = estimate_u_values(
                construction_year=construction_year,
                facade_material=result.facade_material,
                renovation_year=renovation_year,
            )
            result.analysis_details["u_values_source"] = "bbr_estimation"

        return result

    def _analyze_wwr(
        self,
        facade_images: dict[str, FacadeImage | Path | str],
    ) -> WindowToWallRatio:
        """Analyze WWR from facade images."""
        # Convert to standard format
        images_dict = {}
        for direction, img_ref in facade_images.items():
            if isinstance(img_ref, FacadeImage):
                images_dict[direction] = img_ref.path
            else:
                images_dict[direction] = Path(img_ref)

        return self.wwr_detector.analyze_all_facades(images_dict)

    def _analyze_material(
        self,
        facade_images: dict[str, FacadeImage | Path | str],
    ) -> dict[str, Any]:
        """Analyze material from facade images."""
        predictions = []

        for direction, img_ref in facade_images.items():
            if isinstance(img_ref, FacadeImage):
                img_path = img_ref.path
            else:
                img_path = Path(img_ref)

            if not img_path.exists():
                continue

            pred = self.material_classifier.classify(img_path)
            predictions.append(pred)

        if not predictions:
            return {
                "material": FacadeMaterial.UNKNOWN,
                "confidence": 0.0,
            }

        # Vote on material
        material_votes = {}
        for pred in predictions:
            mat = pred.material
            if mat not in material_votes:
                material_votes[mat] = []
            material_votes[mat].append(pred.confidence)

        # Find material with highest weighted votes
        best_material = None
        best_score = 0
        for mat, confs in material_votes.items():
            score = sum(confs) / len(confs) * len(confs)  # avg * count
            if score > best_score:
                best_score = score
                best_material = mat

        avg_confidence = sum(p.confidence for p in predictions) / len(predictions)

        return {
            "material": best_material or FacadeMaterial.UNKNOWN,
            "confidence": avg_confidence,
        }

    def analyze_from_directory(
        self,
        image_dir: Path,
        construction_year: int | None = None,
    ) -> FacadeAnalysisResult:
        """
        Analyze facades from a directory of images.

        Expected naming convention:
        - north.jpg, south.jpg, east.jpg, west.jpg
        - Or: facade_north.jpg, facade_south.jpg, etc.
        """
        image_dir = Path(image_dir)
        facade_images = {}

        for direction in ["north", "south", "east", "west"]:
            # Try different naming patterns
            patterns = [
                f"{direction}.jpg",
                f"{direction}.png",
                f"facade_{direction}.jpg",
                f"facade_{direction}.png",
                f"{direction}_facade.jpg",
                f"{direction}_facade.png",
            ]

            for pattern in patterns:
                img_path = image_dir / pattern
                if img_path.exists():
                    facade_images[direction] = img_path
                    break

        return self.analyze_building(
            facade_images=facade_images if facade_images else None,
            construction_year=construction_year,
        )


def create_envelope_data(
    analysis_result: FacadeAnalysisResult,
) -> EnvelopeData:
    """
    Create EnvelopeData model from analysis result.

    Ready for integration into EnrichedBuilding.
    """
    return EnvelopeData(
        window_to_wall_ratio=analysis_result.wwr,
        facade_material=analysis_result.facade_material,
        facade_material_confidence=analysis_result.material_confidence,
        u_values=analysis_result.u_values,
    )
