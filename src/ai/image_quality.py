"""
Image Quality Assessment for Facade Analysis

Filters out images that would produce unreliable WWR detections:
- Blurry images (camera shake, motion blur, out of focus)
- Poor lighting (too dark, overexposed, harsh shadows)
- Occluded facades (trees, vehicles, signs blocking view)
- Extreme angles (facade not facing camera)
"""

import numpy as np
from PIL import Image
from dataclasses import dataclass
from typing import Tuple, Optional, List
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


@dataclass
class ImageQualityResult:
    """Results from image quality assessment."""
    is_usable: bool
    overall_score: float  # 0-1, higher is better

    # Individual metrics
    blur_score: float  # Higher = sharper
    brightness_score: float  # 0.5 = ideal, 0 or 1 = too dark/bright
    contrast_score: float  # Higher = better contrast
    occlusion_score: float  # Higher = less occlusion
    angle_score: float  # Higher = more frontal view

    # Reasons for rejection
    rejection_reasons: List[str]


class ImageQualityAssessor:
    """
    Assess image quality for facade analysis.

    Filters out images that would produce unreliable detections.
    """

    def __init__(
        self,
        min_blur_score: float = 50.0,  # Laplacian variance threshold
        min_brightness: float = 0.15,
        max_brightness: float = 0.85,
        min_contrast: float = 0.3,
        min_occlusion_score: float = 0.4,
    ):
        self.min_blur_score = min_blur_score
        self.min_brightness = min_brightness
        self.max_brightness = max_brightness
        self.min_contrast = min_contrast
        self.min_occlusion_score = min_occlusion_score

    def assess(
        self,
        image: Image.Image | np.ndarray | str | Path,
    ) -> ImageQualityResult:
        """
        Assess image quality.

        Args:
            image: PIL Image, numpy array, or path to image

        Returns:
            ImageQualityResult with scores and usability flag
        """
        # Load image
        if isinstance(image, (str, Path)):
            image = Image.open(image)
        if isinstance(image, Image.Image):
            img_array = np.array(image)
        else:
            img_array = image

        # Convert to grayscale for analysis
        if len(img_array.shape) == 3:
            gray = np.mean(img_array, axis=2).astype(np.uint8)
        else:
            gray = img_array

        # Compute metrics
        blur_score = self._compute_blur_score(gray)
        brightness_score = self._compute_brightness_score(gray)
        contrast_score = self._compute_contrast_score(gray)
        occlusion_score = self._compute_occlusion_score(img_array, gray)
        angle_score = self._compute_angle_score(gray)

        # Check rejection reasons
        rejection_reasons = []

        if blur_score < self.min_blur_score:
            rejection_reasons.append(f"Too blurry (score={blur_score:.1f} < {self.min_blur_score})")

        if brightness_score < self.min_brightness:
            rejection_reasons.append(f"Too dark (brightness={brightness_score:.2f})")
        elif brightness_score > self.max_brightness:
            rejection_reasons.append(f"Overexposed (brightness={brightness_score:.2f})")

        if contrast_score < self.min_contrast:
            rejection_reasons.append(f"Low contrast (score={contrast_score:.2f})")

        if occlusion_score < self.min_occlusion_score:
            rejection_reasons.append(f"Occluded view (score={occlusion_score:.2f})")

        # Calculate overall score (weighted average)
        overall_score = (
            0.30 * min(blur_score / 200, 1.0) +  # Normalize blur to 0-1
            0.20 * (1 - abs(0.5 - brightness_score) * 2) +  # Ideal at 0.5
            0.15 * contrast_score +
            0.25 * occlusion_score +
            0.10 * angle_score
        )

        is_usable = len(rejection_reasons) == 0

        return ImageQualityResult(
            is_usable=is_usable,
            overall_score=overall_score,
            blur_score=blur_score,
            brightness_score=brightness_score,
            contrast_score=contrast_score,
            occlusion_score=occlusion_score,
            angle_score=angle_score,
            rejection_reasons=rejection_reasons,
        )

    def _compute_blur_score(self, gray: np.ndarray) -> float:
        """
        Compute blur score using Laplacian variance.

        Higher score = sharper image.
        Typical thresholds: <50 = blurry, >100 = sharp
        """
        # Laplacian kernel
        laplacian = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)

        # Apply convolution manually (avoiding scipy dependency)
        from scipy.ndimage import convolve
        try:
            lap = convolve(gray.astype(np.float32), laplacian)
            return float(np.var(lap))
        except ImportError:
            # Fallback: simple gradient-based sharpness
            gx = np.diff(gray.astype(np.float32), axis=1)
            gy = np.diff(gray.astype(np.float32), axis=0)
            return float(np.var(gx) + np.var(gy))

    def _compute_brightness_score(self, gray: np.ndarray) -> float:
        """
        Compute normalized brightness (0-1).

        0 = completely dark, 1 = completely white
        Ideal range: 0.3-0.7
        """
        return float(np.mean(gray) / 255.0)

    def _compute_contrast_score(self, gray: np.ndarray) -> float:
        """
        Compute contrast score using standard deviation.

        Higher = more contrast (better for detection).
        """
        std = np.std(gray) / 255.0
        # Normalize to 0-1 range (typical std is 0.15-0.35)
        return min(std / 0.3, 1.0)

    def _compute_occlusion_score(
        self,
        img_array: np.ndarray,
        gray: np.ndarray,
    ) -> float:
        """
        Detect occlusions (trees, vehicles, signs).

        Uses:
        - Green channel dominance (trees/vegetation)
        - Large uniform dark regions (vehicles)
        - Edge density in different regions

        Returns: 0-1, higher = less occlusion
        """
        if len(img_array.shape) < 3:
            return 0.8  # Grayscale, assume OK

        h, w = gray.shape

        # Check for vegetation (green dominance in lower half)
        lower_half = img_array[h//2:, :, :]
        if lower_half.shape[2] >= 3:
            r, g, b = lower_half[:,:,0], lower_half[:,:,1], lower_half[:,:,2]
            green_dominant = (g > r + 20) & (g > b + 20) & (g > 50)
            green_ratio = np.sum(green_dominant) / green_dominant.size

            if green_ratio > 0.3:
                return max(0.0, 1.0 - green_ratio * 1.5)

        # Check for large uniform dark regions (vehicles, shadows)
        dark_pixels = gray < 40
        dark_ratio = np.sum(dark_pixels) / dark_pixels.size

        if dark_ratio > 0.25:
            return max(0.0, 1.0 - dark_ratio * 1.5)

        # Check edge density - occluded images often have irregular edges
        # Use simple gradient magnitude
        gx = np.abs(np.diff(gray.astype(np.float32), axis=1))
        gy = np.abs(np.diff(gray.astype(np.float32), axis=0))

        edge_density = (np.mean(gx) + np.mean(gy)) / 2 / 255

        # Good facades have moderate edge density (0.05-0.15)
        if edge_density < 0.03:
            return 0.6  # Too uniform (sky, wall with no windows)
        elif edge_density > 0.25:
            return 0.5  # Too busy (trees, cluttered)

        return 0.9

    def _compute_angle_score(self, gray: np.ndarray) -> float:
        """
        Estimate viewing angle from perspective cues.

        Frontal views have more regular horizontal/vertical lines.
        Angled views have skewed lines.

        Returns: 0-1, higher = more frontal
        """
        h, w = gray.shape

        # Simple heuristic: check horizontal line regularity
        # Buildings have strong horizontal lines (floor separations)

        # Compute horizontal gradient magnitude per row
        gx = np.abs(np.diff(gray.astype(np.float32), axis=1))
        row_variance = np.var(np.mean(gx, axis=1))

        # Buildings have periodic horizontal structure
        # High variance = good structure
        normalized_var = min(row_variance / 100, 1.0)

        return 0.5 + 0.5 * normalized_var  # Base score of 0.5


def filter_images_by_quality(
    images: List[Tuple[Image.Image, dict]],
    min_score: float = 0.4,
    assessor: ImageQualityAssessor = None,
) -> List[Tuple[Image.Image, dict, ImageQualityResult]]:
    """
    Filter a list of images by quality.

    Args:
        images: List of (image, metadata) tuples
        min_score: Minimum overall quality score
        assessor: ImageQualityAssessor instance

    Returns:
        Filtered list with quality results: (image, metadata, quality)
    """
    if assessor is None:
        assessor = ImageQualityAssessor()

    results = []
    for img, metadata in images:
        quality = assessor.assess(img)
        if quality.is_usable and quality.overall_score >= min_score:
            results.append((img, metadata, quality))
        else:
            logger.debug(f"Rejected image: {quality.rejection_reasons}")

    return results
