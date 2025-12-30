"""
Visual Analyzer - Standalone facade analysis from Street View imagery.

This module provides a complete visual analysis pipeline that can be:
1. Called standalone for any building (by address or coordinates)
2. Integrated into the full Raiden pipeline
3. Used by other pipelines (e.g., portfolio analysis, quick screening)

Analysis includes:
- Google Street View image fetching (multi-position, historical)
- LLM-based facade analysis (Gemini/Claude/Komilion)
- Window-to-Wall Ratio (WWR) detection
- Facade material classification
- Building height estimation (multi-position geometric + floor counting)
- Ground floor commercial detection
- Building form and era estimation

Usage:
    # Standalone usage
    from src.analysis.visual_analyzer import VisualAnalyzer, VisualAnalysisResult

    analyzer = VisualAnalyzer()
    result = analyzer.analyze_building(
        lat=59.30, lon=18.10,
        footprint_geojson={"type": "Polygon", "coordinates": [...]},
    )

    print(f"Height: {result.height_estimate.height_m}m")
    print(f"Material: {result.facade_material}")
    print(f"WWR: {result.wwr_by_orientation}")

    # Or with just an address (geocodes automatically)
    result = analyzer.analyze_address("Bellmansgatan 16, Stockholm")
"""

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from PIL import Image
from rich.console import Console

# Raiden imports - Street View & Image Fetching
from ..ingest.streetview_fetcher import (
    StreetViewFacadeFetcher,
    StreetViewImage,
    GeometricHeightEstimator,
    GeometricHeightEstimate,
)
from ..ingest.historical_streetview import HistoricalStreetViewFetcher, STREETVIEW_AVAILABLE
from ..ingest.height_estimator_v3 import (
    OptimalHeightEstimator,
    HeightEstimateV3,
    LLMFacadeData,
)

# Raiden imports - AI Analysis
from ..ai.wwr_detector import WWRDetector
from ..ai.material_classifier import MaterialClassifier
from ..ai.material_classifier_v2 import MaterialClassifierV2
from ..ai.facade_analyzer_llm import FacadeAnalyzerLLM, FacadeAnalysis
from ..ai.image_quality import ImageQualityAssessor
from ..ai.ground_floor_detector import GroundFloorDetector

logger = logging.getLogger(__name__)
console = Console()


@dataclass
class GroundFloorResult:
    """Result of ground floor commercial detection."""
    is_commercial: bool = False
    detected_use: str = "residential"  # residential, commercial, mixed
    confidence: float = 0.0
    commercial_pct_estimate: float = 0.0


@dataclass
class VisualAnalysisResult:
    """
    Complete result of visual facade analysis.

    This contains everything extracted from Street View imagery:
    - WWR per orientation
    - Facade material with confidence
    - Height estimate (geometric + floor-based)
    - Ground floor use
    - Building form and era hints
    - Raw LLM analysis result
    """
    # WWR Detection
    wwr_by_orientation: Dict[str, float] = field(default_factory=dict)
    wwr_average: float = 0.0
    wwr_confidence: float = 0.0

    # Material Classification
    facade_material: str = "unknown"
    material_confidence: float = 0.0

    # Height Estimation (key feature!)
    height_estimate: Optional[Union[GeometricHeightEstimate, HeightEstimateV3]] = None
    height_m: float = 0.0
    height_confidence: float = 0.0
    floor_count: int = 0
    height_uncertainty_m: float = 0.0  # ±uncertainty from v3 estimator

    # Ground Floor Detection
    ground_floor: Optional[GroundFloorResult] = None
    has_commercial_ground_floor: bool = False

    # Building Form & Era (from LLM)
    building_form: str = "unknown"  # lamellhus, skivhus, punkthus, etc.
    estimated_era: str = "unknown"  # pre_1930, miljonprogrammet, modern, etc.

    # Renovation Hints
    likely_renovated: bool = False
    renovation_hints: List[str] = field(default_factory=list)

    # Raw LLM Result (for advanced users)
    llm_facade_result: Optional[FacadeAnalysis] = None

    # Metadata
    num_images_analyzed: int = 0
    orientations_covered: List[str] = field(default_factory=list)
    data_sources: List[str] = field(default_factory=list)
    analysis_timestamp: str = ""

    # Image Paths (for debugging/reports)
    saved_image_paths: Dict[str, List[str]] = field(default_factory=dict)


class VisualAnalyzer:
    """
    Standalone visual analyzer for building facade analysis.

    Uses Google Street View imagery + AI to extract:
    - Window-to-Wall Ratio (WWR) per orientation
    - Facade material (brick, concrete, plaster, etc.)
    - Building height (multi-position geometric triangulation)
    - Ground floor commercial detection
    - Building form (lamellhus, skivhus, etc.)
    - Era estimation and renovation detection

    This module can be used:
    1. Standalone: `analyzer.analyze_building(lat, lon, footprint)`
    2. With address: `analyzer.analyze_address("Bellmansgatan 16")`
    3. From full pipeline: Integrated into FullPipelineAnalyzer
    """

    def __init__(
        self,
        google_api_key: str = None,
        ai_backend: str = "opencv",
        ai_device: str = "cpu",
        output_dir: Path = None,
    ):
        """
        Initialize the visual analyzer.

        Args:
            google_api_key: Google Cloud API key (for Street View)
            ai_backend: Backend for WWR detection ("opencv", "sam", "lang_sam")
            ai_device: Device for AI models ("cpu", "cuda", "mps")
            output_dir: Directory to save images (default: ./visual_analysis_output)
        """
        self.google_api_key = google_api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("BRF_GOOGLE_API_KEY")
        self.output_dir = output_dir or Path("./visual_analysis_output")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize fetchers
        if self.google_api_key:
            self.streetview_fetcher = StreetViewFacadeFetcher(api_key=self.google_api_key)
            self.historical_fetcher = (
                HistoricalStreetViewFetcher(api_key=self.google_api_key)
                if STREETVIEW_AVAILABLE else None
            )
        else:
            self.streetview_fetcher = None
            self.historical_fetcher = None
            logger.warning("No Google API key - Street View fetching disabled")

        # Initialize AI modules
        self.wwr_detector = WWRDetector(backend=ai_backend, device=ai_device)
        self.material_classifier = MaterialClassifier(device=ai_device)
        self.material_classifier_v2 = MaterialClassifierV2(device=ai_device)
        self.image_quality_assessor = ImageQualityAssessor()
        self.ground_floor_detector = GroundFloorDetector()
        self.height_estimator = GeometricHeightEstimator()

        # LLM analyzer (initialized on-demand based on available API keys)
        self._llm_analyzer = None

    def _get_llm_analyzer(self) -> Optional[FacadeAnalyzerLLM]:
        """Get or create LLM analyzer based on available API keys."""
        if self._llm_analyzer:
            return self._llm_analyzer

        # Priority: Komilion (FREE) > Gemini > Claude > OpenAI
        if os.environ.get("KOMILION_API_KEY"):
            self._llm_analyzer = FacadeAnalyzerLLM(backend="komilion", komilion_mode="balanced")
        elif os.environ.get("GOOGLE_API_KEY"):
            self._llm_analyzer = FacadeAnalyzerLLM(backend="gemini")
        elif os.environ.get("ANTHROPIC_API_KEY"):
            self._llm_analyzer = FacadeAnalyzerLLM(backend="claude")
        elif os.environ.get("OPENAI_API_KEY"):
            self._llm_analyzer = FacadeAnalyzerLLM(backend="openai")

        return self._llm_analyzer

    def analyze_building(
        self,
        lat: float,
        lon: float,
        footprint_geojson: Optional[Dict] = None,
        save_dir: Path = None,
        images_per_facade: int = 3,
        use_historical: bool = True,
        historical_years: int = 3,
        use_sam_crop: bool = True,
    ) -> VisualAnalysisResult:
        """
        Analyze a building using Street View imagery.

        This is the main entry point for visual analysis. It:
        1. Fetches Street View images from multiple positions
        2. Runs LLM-based facade analysis (material, WWR, floors, era)
        3. Performs multi-position geometric height estimation
        4. Detects ground floor commercial use
        5. Combines all results with confidence scores

        Args:
            lat: Building latitude (WGS84)
            lon: Building longitude (WGS84)
            footprint_geojson: Optional GeoJSON polygon for the building footprint
            save_dir: Directory to save images
            images_per_facade: Number of images per facade direction
            use_historical: Include historical Street View imagery
            historical_years: How many years back to fetch
            use_sam_crop: Use SAM for building segmentation

        Returns:
            VisualAnalysisResult with all extracted data
        """
        result = VisualAnalysisResult(
            analysis_timestamp=datetime.now().isoformat(),
        )

        if not self.streetview_fetcher:
            logger.warning("Street View fetcher not available (no Google API key)")
            result.data_sources.append("none_no_api_key")
            return result

        save_dir = save_dir or self.output_dir / "facades"
        save_dir.mkdir(parents=True, exist_ok=True)

        # Create footprint from coordinates if not provided
        if not footprint_geojson:
            # Create a simple square footprint (30m x 30m) centered on coordinates
            offset = 0.00015  # ~15m in degrees at Swedish latitudes
            footprint_geojson = {
                "type": "Polygon",
                "coordinates": [[
                    [lon - offset, lat - offset],
                    [lon + offset, lat - offset],
                    [lon + offset, lat + offset],
                    [lon - offset, lat + offset],
                    [lon - offset, lat - offset],
                ]]
            }

        try:
            # ═══════════════════════════════════════════════════════════════════
            # STEP 1: Fetch Street View Images (Multi-Position)
            # ═══════════════════════════════════════════════════════════════════
            console.print("[bold]📷 Fetching Street View facades...[/bold]")

            multi_images = self.streetview_fetcher.fetch_multi_facade_images(
                footprint_geojson, images_per_facade=images_per_facade
            )
            total_count = sum(len(imgs) for imgs in multi_images.values())
            console.print(f"[green]Got {total_count} current facade images[/green]")
            result.data_sources.append("google_streetview")

            # Add historical images
            if use_historical and self.historical_fetcher:
                try:
                    console.print(f"[cyan]Fetching historical imagery ({historical_years} years)...[/cyan]")
                    headings = {'N': 0, 'E': 90, 'S': 180, 'W': 270}
                    historical_images = self.historical_fetcher.fetch_multi_year_facades(
                        lat, lon,
                        headings=headings,
                        pitches=[15, 35],
                        years_back=historical_years,
                    )

                    historical_count = 0
                    for orientation, hist_imgs in historical_images.items():
                        for hist_img in hist_imgs:
                            sv_img = StreetViewImage(
                                orientation=orientation,
                                image=hist_img.image,
                                camera_lat=hist_img.lat,
                                camera_lon=hist_img.lon,
                                heading=hist_img.heading,
                                pitch=getattr(hist_img, 'pitch', 0),
                                fov=getattr(hist_img, 'fov', 90),
                            )
                            multi_images[orientation].append(sv_img)
                            historical_count += 1

                    if historical_count > 0:
                        console.print(f"[green]Added {historical_count} historical images[/green]")
                        result.data_sources.append("google_historical")
                except Exception as e:
                    logger.warning(f"Historical imagery fetch failed: {e}")

            # ═══════════════════════════════════════════════════════════════════
            # STEP 2: Save Images & Run Quality Filter
            # ═══════════════════════════════════════════════════════════════════
            saved_paths: Dict[str, List[str]] = {}
            all_pil_images: List[Image.Image] = []
            quality_images: List[Tuple[StreetViewImage, float]] = []  # (image, quality_score)

            for orientation in ['N', 'E', 'S', 'W']:
                images = multi_images.get(orientation, [])
                if not images:
                    continue

                result.orientations_covered.append(orientation)
                saved_paths[orientation] = []

                for i, sv_image in enumerate(images):
                    img_path = save_dir / f"facade_{orientation}_{i+1}.jpg"
                    sv_image.image.save(img_path)
                    saved_paths[orientation].append(str(img_path))

                    # Quality assessment
                    try:
                        quality_result = self.image_quality_assessor.assess(sv_image.image)
                        if quality_result.is_usable:
                            quality_score = quality_result.overall_score
                            quality_images.append((sv_image, quality_score))
                            if len(all_pil_images) < 16 and quality_score > 0.4:
                                all_pil_images.append(sv_image.image)
                    except Exception:
                        quality_images.append((sv_image, 0.5))
                        all_pil_images.append(sv_image.image)

            result.saved_image_paths = saved_paths
            result.num_images_analyzed = len(quality_images)

            # ═══════════════════════════════════════════════════════════════════
            # STEP 3: LLM-Based Facade Analysis
            # ═══════════════════════════════════════════════════════════════════
            llm_analyzer = self._get_llm_analyzer()
            llm_result: Optional[FacadeAnalysis] = None

            if llm_analyzer and saved_paths:
                try:
                    backend_name = {
                        "komilion": "Komilion (balanced)",
                        "gemini": "Gemini 2.0 Flash",
                        "claude": "Claude Sonnet",
                        "openai": "GPT-4o",
                    }.get(llm_analyzer.backend, llm_analyzer.backend)

                    console.print(f"[cyan]Running {backend_name} facade analysis...[/cyan]")
                    llm_result = llm_analyzer.analyze_multiple(saved_paths, max_images=3)

                    if llm_result:
                        result.llm_facade_result = llm_result
                        result.facade_material = llm_result.facade_material
                        result.material_confidence = llm_result.material_confidence
                        result.building_form = llm_result.building_form or "unknown"
                        result.estimated_era = llm_result.estimated_era or "unknown"
                        result.likely_renovated = llm_result.likely_renovated

                        console.print(f"[green]LLM Material: {llm_result.facade_material} ({llm_result.material_confidence:.0%})[/green]")
                        console.print(f"[green]LLM Form: {llm_result.building_form}, Era: {llm_result.estimated_era}[/green]")
                        console.print(f"[green]LLM Floors: {llm_result.visible_floors}, WWR: {llm_result.wwr_average:.0%}[/green]")
                except Exception as e:
                    logger.warning(f"LLM facade analysis failed: {e}")

            # ═══════════════════════════════════════════════════════════════════
            # STEP 4: WWR Detection (CV + LLM fusion)
            # ═══════════════════════════════════════════════════════════════════
            wwr_results: Dict[str, float] = {}
            wwr_confidences: Dict[str, float] = {}

            for orientation in ['N', 'E', 'S', 'W']:
                images = multi_images.get(orientation, [])
                if not images:
                    continue

                facade_wwrs = []
                facade_weights = []

                for sv_image in images:
                    try:
                        wwr, conf = self.wwr_detector.calculate_wwr(
                            sv_image.image,
                            crop_facade=True,
                            use_sam_crop=use_sam_crop,
                        )
                        if wwr > 0 and conf > 0.15:
                            facade_wwrs.append(wwr)
                            facade_weights.append(conf)
                    except Exception:
                        continue

                if facade_wwrs:
                    total_weight = sum(facade_weights)
                    if total_weight > 0:
                        weighted_wwr = sum(w * wt for w, wt in zip(facade_wwrs, facade_weights)) / total_weight
                        avg_conf = total_weight / len(facade_weights)
                    else:
                        weighted_wwr = sum(facade_wwrs) / len(facade_wwrs)
                        avg_conf = 0.3

                    wwr_results[orientation] = weighted_wwr
                    wwr_confidences[orientation] = avg_conf

            # LLM override for unreliable CV
            if llm_result and llm_result.wwr_average > 0.05:
                llm_wwr = llm_result.wwr_average
                for orient in ['N', 'E', 'S', 'W']:
                    cv_wwr = wwr_results.get(orient, 0)
                    cv_conf = wwr_confidences.get(orient, 0)

                    needs_override = (
                        cv_wwr < 0.10 or
                        cv_conf < 0.50 or
                        (llm_wwr > 0.15 and cv_wwr < llm_wwr * 0.33)
                    )

                    if needs_override:
                        wwr_results[orient] = llm_wwr
                        wwr_confidences[orient] = 0.7  # LLM confidence

            result.wwr_by_orientation = wwr_results
            if wwr_results:
                result.wwr_average = sum(wwr_results.values()) / len(wwr_results)
                result.wwr_confidence = sum(wwr_confidences.values()) / len(wwr_confidences)

            # ═══════════════════════════════════════════════════════════════════
            # STEP 5: Height Estimation (V3 Optimal: Reference > Era > Geometric)
            # ═══════════════════════════════════════════════════════════════════
            if llm_result and llm_result.visible_floors > 0:
                # Handle both FacadeAnalysis and MultiFacadeAnalysis attribute names
                has_attic = getattr(llm_result, 'has_visible_attic', None)
                if has_attic is None:
                    has_attic = getattr(llm_result, 'has_attic', False)
                has_basement = getattr(llm_result, 'has_visible_basement', None)
                if has_basement is None:
                    has_basement = getattr(llm_result, 'has_basement', False)

                # Build LLM data for v3 estimator
                llm_data = LLMFacadeData(
                    visible_floors=llm_result.visible_floors,
                    floor_count_confidence=getattr(llm_result, 'floor_count_confidence', 0.7),
                    estimated_floor_height_m=getattr(llm_result, 'estimated_floor_height_m', 2.8),
                    height_reference_used=getattr(llm_result, 'height_reference_used', 'none'),
                    roof_position_pct=getattr(llm_result, 'roof_position_pct', 0.0),
                    ground_position_pct=getattr(llm_result, 'ground_position_pct', 0.0),
                    has_visible_attic=has_attic,
                    has_visible_basement=has_basement,
                    estimated_era=llm_result.estimated_era or "unknown",
                    building_form=llm_result.building_form or "lamellhus",
                )

                # Collect all images for geometric validation
                all_images_with_meta = []
                for orient, images in multi_images.items():
                    all_images_with_meta.extend(images)

                # Run v3 optimal height estimation
                v3_estimator = OptimalHeightEstimator()
                height_v3 = v3_estimator.estimate(
                    llm_data=llm_data,
                    camera_images=all_images_with_meta if len(all_images_with_meta) >= 2 else None,
                    facade_lat=lat,
                    facade_lon=lon,
                )

                # Log estimation details
                ref_used = llm_data.height_reference_used
                if ref_used and ref_used != "none":
                    console.print(
                        f"[green]Reference-calibrated: {llm_result.visible_floors} floors × "
                        f"{llm_data.estimated_floor_height_m:.2f}m (ref={ref_used})[/green]"
                    )
                else:
                    console.print(
                        f"[green]Era-calibrated: {llm_result.visible_floors} floors × "
                        f"{height_v3.floor_height_m:.2f}m ({llm_data.estimated_era})[/green]"
                    )

                # Log geometric validation if available
                if height_v3.geometric_validation:
                    console.print(
                        f"[dim]Geometric validation: {height_v3.geometric_validation:.1f}m[/dim]"
                    )

                # Store result
                result.height_estimate = height_v3
                result.height_m = height_v3.height_m
                result.height_confidence = height_v3.confidence
                result.height_uncertainty_m = height_v3.uncertainty_m
                result.floor_count = height_v3.floor_count

                console.print(
                    f"[cyan]Final height: {height_v3.height_m:.1f}m ±{height_v3.uncertainty_m:.1f}m "
                    f"({height_v3.method}, conf={height_v3.confidence:.0%})[/cyan]"
                )

                # Log any notes from the estimator
                for note in height_v3.notes:
                    logger.debug(f"Height estimation: {note}")

            # ═══════════════════════════════════════════════════════════════════
            # STEP 6: Ground Floor Detection
            # ═══════════════════════════════════════════════════════════════════
            if llm_result and llm_result.ground_floor_use != "unknown":
                result.ground_floor = GroundFloorResult(
                    is_commercial=llm_result.ground_floor_use in ("commercial", "mixed"),
                    detected_use=llm_result.ground_floor_use,
                    confidence=0.8,
                    commercial_pct_estimate=0.5 if llm_result.ground_floor_use == "mixed" else (
                        1.0 if llm_result.ground_floor_use == "commercial" else 0.0
                    ),
                )
                result.has_commercial_ground_floor = result.ground_floor.is_commercial
            elif all_pil_images and self.ground_floor_detector:
                try:
                    gf_result = self.ground_floor_detector.detect(all_pil_images)
                    result.ground_floor = GroundFloorResult(
                        is_commercial=gf_result.is_commercial,
                        detected_use=gf_result.detected_use,
                        confidence=gf_result.confidence,
                        commercial_pct_estimate=gf_result.commercial_pct_estimate,
                    )
                    result.has_commercial_ground_floor = gf_result.is_commercial
                except Exception as e:
                    logger.warning(f"Ground floor detection failed: {e}")

            # ═══════════════════════════════════════════════════════════════════
            # STEP 7: Material Fallback (if LLM didn't work)
            # ═══════════════════════════════════════════════════════════════════
            if result.facade_material == "unknown" and all_pil_images:
                try:
                    material_result = self.material_classifier_v2.classify_multi_image(
                        all_pil_images,
                        use_sam_crop=True,
                        building_type="residential",
                    )
                    result.facade_material = material_result.material
                    result.material_confidence = material_result.confidence
                    console.print(f"[green]Material V2: {result.facade_material} ({result.material_confidence:.0%})[/green]")
                except Exception:
                    try:
                        mat_pred = self.material_classifier.classify(all_pil_images[0])
                        result.facade_material = mat_pred.material.value
                        result.material_confidence = mat_pred.confidence
                    except Exception:
                        pass

            return result

        except Exception as e:
            logger.error(f"Visual analysis failed: {e}")
            result.data_sources.append("error")
            return result

    def analyze_address(
        self,
        address: str,
        footprint_geojson: Optional[Dict] = None,
        **kwargs,
    ) -> VisualAnalysisResult:
        """
        Analyze a building by address (geocodes automatically).

        Args:
            address: Street address (e.g., "Bellmansgatan 16, Stockholm")
            footprint_geojson: Optional building footprint
            **kwargs: Additional arguments passed to analyze_building()

        Returns:
            VisualAnalysisResult
        """
        try:
            from geopy.geocoders import Nominatim

            geocoder = Nominatim(user_agent="raiden_visual_analyzer")
            location = geocoder.geocode(address)

            if not location:
                logger.error(f"Could not geocode address: {address}")
                return VisualAnalysisResult(
                    data_sources=["geocoding_failed"],
                    analysis_timestamp=datetime.now().isoformat(),
                )

            console.print(f"[cyan]Geocoded: {address} → ({location.latitude:.6f}, {location.longitude:.6f})[/cyan]")

            return self.analyze_building(
                lat=location.latitude,
                lon=location.longitude,
                footprint_geojson=footprint_geojson,
                **kwargs,
            )

        except ImportError:
            logger.error("geopy not installed. Run: pip install geopy")
            return VisualAnalysisResult(
                data_sources=["geopy_not_installed"],
                analysis_timestamp=datetime.now().isoformat(),
            )

    def analyze_from_images(
        self,
        image_paths: Dict[str, List[str]],
        lat: float = 59.3,
        lon: float = 18.1,
    ) -> VisualAnalysisResult:
        """
        Analyze pre-fetched images (useful for testing or when images are already available).

        Args:
            image_paths: Dict mapping orientation ('N', 'S', 'E', 'W') to list of image paths
            lat, lon: Building coordinates (for height estimation)

        Returns:
            VisualAnalysisResult
        """
        result = VisualAnalysisResult(
            analysis_timestamp=datetime.now().isoformat(),
            saved_image_paths=image_paths,
        )

        # Load images
        all_pil_images = []
        for orientation, paths in image_paths.items():
            result.orientations_covered.append(orientation)
            for path in paths:
                try:
                    img = Image.open(path)
                    all_pil_images.append(img)
                except Exception as e:
                    logger.warning(f"Failed to load image {path}: {e}")

        result.num_images_analyzed = len(all_pil_images)
        result.data_sources.append("pre_fetched_images")

        # Run LLM analysis
        llm_analyzer = self._get_llm_analyzer()
        if llm_analyzer:
            try:
                llm_result = llm_analyzer.analyze_multiple(image_paths, max_images=3)
                if llm_result:
                    result.llm_facade_result = llm_result
                    result.facade_material = llm_result.facade_material
                    result.material_confidence = llm_result.material_confidence
                    result.building_form = llm_result.building_form or "unknown"
                    result.estimated_era = llm_result.estimated_era or "unknown"
                    result.wwr_average = llm_result.wwr_average
                    result.floor_count = llm_result.visible_floors

                    # Height from floor count using v3 estimator
                    if llm_result.visible_floors > 0:
                        has_attic = getattr(llm_result, 'has_visible_attic', None)
                        if has_attic is None:
                            has_attic = getattr(llm_result, 'has_attic', False)

                        llm_data = LLMFacadeData(
                            visible_floors=llm_result.visible_floors,
                            floor_count_confidence=getattr(llm_result, 'floor_count_confidence', 0.7),
                            estimated_floor_height_m=getattr(llm_result, 'estimated_floor_height_m', 2.8),
                            height_reference_used=getattr(llm_result, 'height_reference_used', 'none'),
                            roof_position_pct=getattr(llm_result, 'roof_position_pct', 0.0),
                            ground_position_pct=getattr(llm_result, 'ground_position_pct', 0.0),
                            has_visible_attic=has_attic,
                            has_visible_basement=getattr(llm_result, 'has_visible_basement', False),
                            estimated_era=llm_result.estimated_era or "unknown",
                            building_form=llm_result.building_form or "lamellhus",
                        )

                        v3_estimator = OptimalHeightEstimator()
                        height_v3 = v3_estimator.estimate(llm_data)
                        result.height_estimate = height_v3
                        result.height_m = height_v3.height_m
                        result.height_confidence = height_v3.confidence
                        result.height_uncertainty_m = height_v3.uncertainty_m
            except Exception as e:
                logger.warning(f"LLM analysis failed: {e}")

        # Material fallback
        if result.facade_material == "unknown" and all_pil_images:
            try:
                material_result = self.material_classifier_v2.classify_multi_image(all_pil_images)
                result.facade_material = material_result.material
                result.material_confidence = material_result.confidence
            except Exception:
                pass

        return result


# ═══════════════════════════════════════════════════════════════════════════════
# Convenience Functions for Quick Analysis
# ═══════════════════════════════════════════════════════════════════════════════

def analyze_building_visually(
    lat: float,
    lon: float,
    footprint_geojson: Optional[Dict] = None,
    **kwargs,
) -> VisualAnalysisResult:
    """
    Quick function to analyze a building visually.

    Example:
        result = analyze_building_visually(59.30, 18.10)
        print(f"Height: {result.height_m}m, Material: {result.facade_material}")
    """
    analyzer = VisualAnalyzer()
    return analyzer.analyze_building(lat, lon, footprint_geojson, **kwargs)


def analyze_address_visually(address: str, **kwargs) -> VisualAnalysisResult:
    """
    Quick function to analyze a building by address.

    Example:
        result = analyze_address_visually("Bellmansgatan 16, Stockholm")
        print(f"Height: {result.height_m}m, Material: {result.facade_material}")
    """
    analyzer = VisualAnalyzer()
    return analyzer.analyze_address(address, **kwargs)
