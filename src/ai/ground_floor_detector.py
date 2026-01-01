"""
Ground Floor Commercial Detector.

Uses street-level imagery (Mapillary/GSV) to detect commercial use
on ground floors when energy declaration data is incomplete.

Detection targets:
- Shopfronts (large glass, display windows)
- Restaurants (signage, outdoor seating, menus)
- Retail (awnings, commercial entrances)
- Offices (business signage)

This is critical for accurate multi-zone modeling because commercial
ground floors typically have:
- F-only ventilation (no heat recovery)
- Much higher airflow (restaurants: 10+ L/s·m²)
- Different operating schedules
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Optional, Any, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Try to import image processing libraries
try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


class GroundFloorUse(Enum):
    """Detected ground floor use type."""
    RESIDENTIAL = "residential"
    RETAIL = "retail"
    RESTAURANT = "restaurant"
    GROCERY = "grocery"
    OFFICE = "office"
    MIXED_COMMERCIAL = "mixed_commercial"
    UNKNOWN = "unknown"


@dataclass
class GroundFloorDetection:
    """Result of ground floor analysis."""
    detected_use: GroundFloorUse
    confidence: float  # 0.0-1.0

    # Evidence from image analysis
    has_large_windows: bool = False
    has_signage: bool = False
    has_outdoor_seating: bool = False
    has_awning: bool = False
    has_commercial_entrance: bool = False
    has_display_lighting: bool = False

    # Estimated commercial fraction (0-100%)
    commercial_pct_estimate: float = 0.0

    # Raw scores for each use type
    use_scores: Dict[str, float] = field(default_factory=dict)

    # Source images analyzed
    images_analyzed: int = 0

    notes: str = ""


class GroundFloorDetector:
    """
    Detect commercial ground floor use from street-level images.

    Uses multiple signals:
    1. Window-to-wall ratio at ground level (commercial: high WWR)
    2. Color analysis (commercial: brighter, more varied)
    3. Edge detection (signage, awnings)
    4. Geometric patterns (regular grid = commercial display)
    """

    def __init__(self, device: str = "cpu"):
        """
        Initialize detector.

        Args:
            device: Device for ML inference ("cpu" or "cuda")
        """
        self.device = device
        self._model_loaded = False

        # Detection thresholds
        self.ground_floor_height_ratio = 0.4  # Bottom 40% of image is ground floor
        self.commercial_wwr_threshold = 0.5  # >50% glass = likely commercial
        self.signage_edge_threshold = 100  # Canny edge density for signage

    def detect(
        self,
        images: List[Image.Image],
        image_directions: Optional[List[str]] = None,
    ) -> GroundFloorDetection:
        """
        Analyze street-level images to detect ground floor use.

        Args:
            images: List of PIL Images from street view
            image_directions: Optional list of directions (N, S, E, W)

        Returns:
            GroundFloorDetection with analysis results
        """
        if not images:
            return GroundFloorDetection(
                detected_use=GroundFloorUse.UNKNOWN,
                confidence=0.0,
                notes="No images provided"
            )

        # Analyze each image
        all_features = []
        for i, img in enumerate(images):
            direction = image_directions[i] if image_directions else None
            features = self._analyze_ground_floor(img, direction)
            all_features.append(features)

        # Aggregate features across all images
        aggregated = self._aggregate_features(all_features)

        # Classify based on aggregated features
        detection = self._classify(aggregated, len(images))

        return detection

    def _analyze_ground_floor(
        self,
        image: Image.Image,
        direction: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Analyze a single image for ground floor commercial indicators.

        Returns dict with detected features.
        """
        if not PIL_AVAILABLE:
            return {"error": "PIL not available"}

        # Convert to numpy array
        img_array = np.array(image)
        height, width = img_array.shape[:2]

        # Extract ground floor region (bottom portion of image)
        ground_floor_top = int(height * (1 - self.ground_floor_height_ratio))
        ground_floor = img_array[ground_floor_top:, :, :]

        features = {
            "direction": direction,
            "image_size": (width, height),
        }

        # 1. Window/glass detection at ground level
        features.update(self._detect_glass_areas(ground_floor))

        # 2. Signage detection (high contrast, text-like patterns)
        features.update(self._detect_signage(ground_floor))

        # 3. Awning detection (horizontal lines, colored bands)
        features.update(self._detect_awnings(img_array, ground_floor_top))

        # 4. Color analysis (commercial = brighter, more saturated)
        features.update(self._analyze_colors(ground_floor))

        # 5. Entrance type detection
        features.update(self._detect_entrance_type(ground_floor))

        return features

    def _detect_glass_areas(self, ground_floor: np.ndarray) -> Dict[str, Any]:
        """Detect glass/window areas in ground floor region."""
        if not CV2_AVAILABLE:
            return {"glass_ratio": 0.0, "has_large_windows": False}

        # Convert to grayscale
        gray = cv2.cvtColor(ground_floor, cv2.COLOR_RGB2GRAY)

        # Glass appears as relatively uniform, darker regions with reflections
        # Use adaptive thresholding to find potential glass areas
        blur = cv2.GaussianBlur(gray, (5, 5), 0)

        # Detect high-contrast edges (glass edges are sharp)
        edges = cv2.Canny(blur, 50, 150)

        # Find contours (glass panes are rectangular)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Calculate glass-like area ratio
        total_area = ground_floor.shape[0] * ground_floor.shape[1]
        glass_area = 0
        large_rectangles = 0

        for contour in contours:
            area = cv2.contourArea(contour)
            if area > total_area * 0.01:  # At least 1% of ground floor
                # Check if roughly rectangular (shopfront windows)
                perimeter = cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
                if 4 <= len(approx) <= 6:  # Rectangular-ish
                    glass_area += area
                    if area > total_area * 0.05:
                        large_rectangles += 1

        glass_ratio = glass_area / total_area if total_area > 0 else 0

        return {
            "glass_ratio": min(1.0, glass_ratio),
            "has_large_windows": large_rectangles > 0,
            "large_window_count": large_rectangles,
        }

    def _detect_signage(self, ground_floor: np.ndarray) -> Dict[str, Any]:
        """Detect commercial signage in ground floor region."""
        if not CV2_AVAILABLE:
            return {"has_signage": False, "signage_score": 0.0}

        gray = cv2.cvtColor(ground_floor, cv2.COLOR_RGB2GRAY)

        # Signage typically has high contrast and sharp edges
        edges = cv2.Canny(gray, 100, 200)
        edge_density = np.sum(edges > 0) / edges.size

        # Look for text-like patterns (high local variance)
        local_std = cv2.blur((gray.astype(float) - cv2.blur(gray, (15, 15)).astype(float))**2, (15, 15))
        high_variance_ratio = np.sum(local_std > 500) / local_std.size

        # Commercial signage score
        signage_score = (edge_density * 2 + high_variance_ratio * 3) / 5

        return {
            "has_signage": signage_score > 0.1,
            "signage_score": min(1.0, signage_score),
            "edge_density": edge_density,
        }

    def _detect_awnings(
        self,
        full_image: np.ndarray,
        ground_floor_top: int,
    ) -> Dict[str, Any]:
        """Detect awnings above ground floor (strong commercial indicator)."""
        if not CV2_AVAILABLE:
            return {"has_awning": False, "awning_score": 0.0}

        # Awning region is just above ground floor
        awning_bottom = ground_floor_top
        awning_top = max(0, ground_floor_top - int(full_image.shape[0] * 0.1))
        awning_region = full_image[awning_top:awning_bottom, :, :]

        if awning_region.size == 0:
            return {"has_awning": False, "awning_score": 0.0}

        # Awnings often have distinct colors (red, green, striped)
        hsv = cv2.cvtColor(awning_region, cv2.COLOR_RGB2HSV)

        # Check for saturated colors (awnings are often colorful)
        saturation = hsv[:, :, 1]
        high_saturation_ratio = np.sum(saturation > 100) / saturation.size

        # Check for horizontal line patterns
        gray = cv2.cvtColor(awning_region, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)

        # Horizontal edges (awning bottom edge)
        horizontal_kernel = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
        horizontal_edges = cv2.filter2D(edges.astype(float), -1, horizontal_kernel)
        horizontal_ratio = np.sum(np.abs(horizontal_edges) > 50) / horizontal_edges.size

        awning_score = (high_saturation_ratio + horizontal_ratio) / 2

        return {
            "has_awning": awning_score > 0.15,
            "awning_score": min(1.0, awning_score),
        }

    def _analyze_colors(self, ground_floor: np.ndarray) -> Dict[str, Any]:
        """
        Analyze color patterns for commercial indicators.

        Commercial: Brighter, more varied colors (displays, lighting)
        Residential: More uniform, muted colors
        """
        if not CV2_AVAILABLE:
            return {"brightness": 0.5, "color_variance": 0.0}

        # Convert to HSV for better color analysis
        hsv = cv2.cvtColor(ground_floor, cv2.COLOR_RGB2HSV)

        # Brightness (Value channel)
        brightness = np.mean(hsv[:, :, 2]) / 255

        # Color variance (commercial has more varied colors)
        h_std = np.std(hsv[:, :, 0]) / 180
        s_std = np.std(hsv[:, :, 1]) / 255
        color_variance = (h_std + s_std) / 2

        # Warm colors often indicate restaurants (warm lighting)
        # Red/orange/yellow hues: 0-30 or 150-180
        warm_mask = ((hsv[:, :, 0] < 30) | (hsv[:, :, 0] > 150)) & (hsv[:, :, 1] > 50)
        warm_ratio = np.sum(warm_mask) / warm_mask.size

        return {
            "brightness": brightness,
            "color_variance": color_variance,
            "warm_color_ratio": warm_ratio,
            "is_brightly_lit": brightness > 0.5,
        }

    def _detect_entrance_type(self, ground_floor: np.ndarray) -> Dict[str, Any]:
        """
        Detect entrance type (commercial vs residential).

        Commercial: Wide glass doors, automatic doors, accessible
        Residential: Narrower, often recessed, intercom systems
        """
        if not CV2_AVAILABLE:
            return {"entrance_type": "unknown"}

        # This is a simplified heuristic
        # A more sophisticated approach would use object detection

        height, width = ground_floor.shape[:2]
        center_region = ground_floor[:, width//4:3*width//4, :]

        gray = cv2.cvtColor(center_region, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)

        # Vertical edges in center = potential door
        vertical_kernel = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
        vertical_edges = cv2.filter2D(edges.astype(float), -1, vertical_kernel)
        vertical_ratio = np.sum(np.abs(vertical_edges) > 50) / vertical_edges.size

        # Commercial doors are typically wider
        # Use horizontal edge analysis to estimate door width

        return {
            "entrance_type": "commercial" if vertical_ratio > 0.1 else "unknown",
            "door_vertical_score": vertical_ratio,
        }

    def _aggregate_features(
        self,
        all_features: List[Dict[str, Any]],
    ) -> Dict[str, float]:
        """Aggregate features from multiple images."""
        if not all_features:
            return {}

        # Numeric features to aggregate
        numeric_keys = [
            "glass_ratio", "signage_score", "awning_score",
            "brightness", "color_variance", "warm_color_ratio",
        ]

        aggregated = {}
        for key in numeric_keys:
            values = [f.get(key, 0) for f in all_features if key in f]
            if values:
                aggregated[key] = np.mean(values)
                aggregated[f"{key}_max"] = max(values)

        # Boolean features: True if any image shows it
        bool_keys = ["has_large_windows", "has_signage", "has_awning", "is_brightly_lit"]
        for key in bool_keys:
            aggregated[key] = any(f.get(key, False) for f in all_features)

        # Count features
        aggregated["large_window_count"] = sum(
            f.get("large_window_count", 0) for f in all_features
        )

        return aggregated

    def _classify(
        self,
        features: Dict[str, float],
        num_images: int,
    ) -> GroundFloorDetection:
        """Classify ground floor use based on aggregated features."""

        # Initialize scores for each use type
        scores = {
            GroundFloorUse.RESIDENTIAL: 0.5,  # Default assumption
            GroundFloorUse.RETAIL: 0.0,
            GroundFloorUse.RESTAURANT: 0.0,
            GroundFloorUse.GROCERY: 0.0,
            GroundFloorUse.OFFICE: 0.0,
        }

        # Evidence scoring
        glass_ratio = features.get("glass_ratio", 0)
        signage_score = features.get("signage_score", 0)
        awning_score = features.get("awning_score", 0)
        brightness = features.get("brightness", 0.5)
        warm_ratio = features.get("warm_color_ratio", 0)

        has_large_windows = features.get("has_large_windows", False)
        has_signage = features.get("has_signage", False)
        has_awning = features.get("has_awning", False)

        # Retail indicators
        if glass_ratio > 0.3:
            scores[GroundFloorUse.RETAIL] += 0.3
        if has_large_windows:
            scores[GroundFloorUse.RETAIL] += 0.2
        if has_signage:
            scores[GroundFloorUse.RETAIL] += 0.2
        if has_awning:
            scores[GroundFloorUse.RETAIL] += 0.15

        # Restaurant indicators
        if warm_ratio > 0.15:
            scores[GroundFloorUse.RESTAURANT] += 0.25
        if brightness > 0.55:
            scores[GroundFloorUse.RESTAURANT] += 0.15
        if has_awning:
            scores[GroundFloorUse.RESTAURANT] += 0.1

        # Grocery indicators (large windows + bright)
        if glass_ratio > 0.5 and brightness > 0.5:
            scores[GroundFloorUse.GROCERY] += 0.2

        # Office indicators (glass + signage but no awning)
        if glass_ratio > 0.2 and has_signage and not has_awning:
            scores[GroundFloorUse.OFFICE] += 0.15

        # Reduce residential score if commercial indicators present
        commercial_evidence = glass_ratio + signage_score + awning_score
        if commercial_evidence > 0.5:
            scores[GroundFloorUse.RESIDENTIAL] -= commercial_evidence * 0.5

        # Determine best match
        best_use = max(scores, key=scores.get)
        best_score = scores[best_use]

        # Calculate confidence
        total_score = sum(max(0, s) for s in scores.values())
        confidence = best_score / total_score if total_score > 0 else 0.0

        # If commercial use detected, estimate percentage
        commercial_pct = 0.0
        if best_use != GroundFloorUse.RESIDENTIAL:
            # Assume commercial is 1-2 floors, estimate percentage
            # This is a rough estimate, refined by building height
            commercial_pct = min(25.0, glass_ratio * 50 + signage_score * 25)

        return GroundFloorDetection(
            detected_use=best_use,
            confidence=confidence,
            has_large_windows=has_large_windows,
            has_signage=has_signage,
            has_outdoor_seating=False,  # Would need object detection
            has_awning=has_awning,
            has_commercial_entrance=features.get("entrance_type") == "commercial",
            has_display_lighting=brightness > 0.55,
            commercial_pct_estimate=commercial_pct,
            use_scores={k.value: v for k, v in scores.items()},
            images_analyzed=num_images,
            notes=f"Analyzed {num_images} images. Glass ratio: {glass_ratio:.1%}",
        )


def detect_ground_floor_use(
    images: List[Image.Image],
    directions: Optional[List[str]] = None,
) -> GroundFloorDetection:
    """
    Convenience function to detect ground floor use from images.

    Args:
        images: Street-level images
        directions: Optional image directions (N, S, E, W)

    Returns:
        GroundFloorDetection result
    """
    detector = GroundFloorDetector()
    return detector.detect(images, directions)
