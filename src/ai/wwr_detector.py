"""
Window-to-Wall Ratio (WWR) detection using AI.

Supports multiple backends:
- Grounded SAM (text-prompted segmentation)
- LangSAM (simpler alternative)
- Manual annotation fallback
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from PIL import Image
from rich.console import Console

from ..core.models import WindowToWallRatio
from .image_quality import ImageQualityAssessor, ImageQualityResult

console = Console()


@dataclass
class WindowGeometry:
    """Geometric properties of a detected window."""
    x: int
    y: int
    width: int
    height: int
    aspect_ratio: float
    rectangularity: float
    area: int
    center_y_normalized: float  # 0-1, position in image


@dataclass
class DetectionResult:
    """Result from window detection."""

    mask: np.ndarray  # Binary mask
    confidence: float
    bbox: tuple[int, int, int, int] | None = None  # x1, y1, x2, y2


class WWRDetector:
    """
    Detect windows in facade images and calculate Window-to-Wall Ratio.

    Supports multiple AI backends for detection.
    """

    SUPPORTED_BACKENDS = ["grounded_sam", "lang_sam", "sam", "opencv", "manual"]

    def __init__(
        self,
        backend: Literal["grounded_sam", "lang_sam", "sam", "opencv", "manual"] = "opencv",
        device: str = "cpu",
        model_dir: Path | None = None,
        sam_model_type: str = "vit_b",  # vit_b, vit_l, vit_h
    ):
        """
        Initialize WWR detector.

        Args:
            backend: Detection backend to use
            device: Device for inference ("cpu", "cuda", "mps")
            model_dir: Directory containing model weights
            sam_model_type: SAM model variant (vit_b, vit_l, vit_h)
        """
        self.backend = backend
        self.device = device
        self.model_dir = model_dir
        self.sam_model_type = sam_model_type
        self._model = None
        self._sam_predictor = None
        self._initialized = False
        self._quality_assessor = ImageQualityAssessor()

        # Tuned prompts for different detection scenarios
        self.window_prompts = [
            "window",  # Primary prompt
            "glass window pane",  # More specific
            "rectangular window",  # Shape-focused
        ]
        self.facade_prompts = [
            "building facade",  # Primary
            "apartment building wall",  # More specific for Swedish buildings
        ]

    def _lazy_init(self) -> None:
        """Lazily initialize the model on first use."""
        if self._initialized:
            return

        console.print(f"[cyan]Initializing WWR detector ({self.backend})...[/cyan]")

        if self.backend == "lang_sam":
            self._init_lang_sam()
        elif self.backend == "grounded_sam":
            self._init_grounded_sam()
        elif self.backend == "sam":
            self._init_sam()
        elif self.backend == "opencv":
            self._init_opencv()
        elif self.backend == "manual":
            pass  # No model needed

        self._initialized = True

    def assess_image_quality(
        self,
        image: Image.Image | np.ndarray | str | Path,
    ) -> ImageQualityResult:
        """
        Assess image quality before processing.

        Returns:
            ImageQualityResult with usability flag and scores
        """
        return self._quality_assessor.assess(image)

    def apply_geometric_filter(
        self,
        detections: list[DetectionResult],
        image_height: int,
        image_width: int,
        min_aspect: float = 0.3,
        max_aspect: float = 3.0,
        min_rectangularity: float = 0.5,
        min_area_ratio: float = 0.001,  # 0.1% of image
        max_area_ratio: float = 0.15,   # 15% of image
        remove_outliers: bool = True,
    ) -> list[DetectionResult]:
        """
        Apply geometric constraints to filter false positive window detections.

        Windows have consistent geometric properties:
        - Rectangular shape (rectangularity > 0.5)
        - Aspect ratio 0.3-3.0 (typically 0.5-2.0)
        - Similar sizes within a facade (remove outliers)
        - Located in building region (not at very top/bottom)

        Args:
            detections: Raw detection results
            image_height: Image height in pixels
            image_width: Image width in pixels
            min_aspect: Minimum aspect ratio (width/height)
            max_aspect: Maximum aspect ratio
            min_rectangularity: Minimum rectangularity score (0-1)
            min_area_ratio: Minimum window area as fraction of image
            max_area_ratio: Maximum window area as fraction of image
            remove_outliers: Remove windows with sizes that are outliers

        Returns:
            Filtered detection results
        """
        if not detections:
            return []

        image_area = image_height * image_width
        min_area = image_area * min_area_ratio
        max_area = image_area * max_area_ratio

        # Extract geometry from each detection
        geometries = []
        valid_indices = []

        for i, det in enumerate(detections):
            if det.bbox is None:
                continue

            x1, y1, x2, y2 = det.bbox
            width = x2 - x1
            height = y2 - y1

            if width <= 0 or height <= 0:
                continue

            aspect = width / height
            area = width * height
            center_y = (y1 + y2) / 2 / image_height

            # Calculate rectangularity from mask
            mask_area = det.mask.sum()
            rect_area = width * height
            rectangularity = mask_area / rect_area if rect_area > 0 else 0

            geom = WindowGeometry(
                x=x1, y=y1,
                width=width, height=height,
                aspect_ratio=aspect,
                rectangularity=rectangularity,
                area=area,
                center_y_normalized=center_y,
            )
            geometries.append(geom)
            valid_indices.append(i)

        if not geometries:
            return []

        # Apply geometric filters
        filtered_indices = []
        for idx, geom in zip(valid_indices, geometries):
            # Aspect ratio check
            if geom.aspect_ratio < min_aspect or geom.aspect_ratio > max_aspect:
                continue

            # Rectangularity check
            if geom.rectangularity < min_rectangularity:
                continue

            # Area check
            if geom.area < min_area or geom.area > max_area:
                continue

            # Position check - windows usually in middle of facade
            if geom.center_y_normalized < 0.05 or geom.center_y_normalized > 0.95:
                continue

            filtered_indices.append(idx)

        if not filtered_indices:
            return []

        # Remove size outliers if requested
        if remove_outliers and len(filtered_indices) > 3:
            areas = [geometries[valid_indices.index(i)].area for i in filtered_indices]
            median_area = np.median(areas)
            std_area = np.std(areas)

            # Keep windows within 2 standard deviations of median
            final_indices = []
            for idx in filtered_indices:
                geom = geometries[valid_indices.index(idx)]
                if abs(geom.area - median_area) <= 2 * std_area:
                    final_indices.append(idx)

            filtered_indices = final_indices if final_indices else filtered_indices

        # Return filtered detections with boosted confidence
        filtered_results = []
        for idx in filtered_indices:
            det = detections[idx]
            # Boost confidence for geometrically valid windows
            boosted_conf = min(1.0, det.confidence * 1.15)
            filtered_results.append(DetectionResult(
                mask=det.mask,
                confidence=boosted_conf,
                bbox=det.bbox,
            ))

        return filtered_results

    def detect_window_grid_pattern(
        self,
        detections: list[DetectionResult],
        image_width: int,
        tolerance: float = 0.15,
    ) -> float:
        """
        Detect if windows form a regular grid pattern (common in buildings).

        Grid detection increases confidence as real windows are often evenly spaced.

        Args:
            detections: Window detections
            image_width: Image width
            tolerance: Spacing tolerance (fraction of average spacing)

        Returns:
            Grid score (0-1), higher = more regular pattern
        """
        if len(detections) < 4:
            return 0.5  # Not enough windows to detect pattern

        # Extract window center X positions
        centers_x = []
        for det in detections:
            if det.bbox:
                x1, y1, x2, y2 = det.bbox
                centers_x.append((x1 + x2) / 2)

        if len(centers_x) < 4:
            return 0.5

        # Sort by X position
        centers_x = sorted(centers_x)

        # Calculate spacings
        spacings = [centers_x[i+1] - centers_x[i] for i in range(len(centers_x)-1)]

        if not spacings:
            return 0.5

        # Check if spacings are regular
        mean_spacing = np.mean(spacings)
        std_spacing = np.std(spacings)

        if mean_spacing <= 0:
            return 0.5

        # Coefficient of variation (lower = more regular)
        cv = std_spacing / mean_spacing

        # Convert to score (0-1)
        grid_score = max(0, 1 - cv * 2)

        return float(grid_score)

    def _init_opencv(self) -> None:
        """Initialize OpenCV-based detection (lightweight, no GPU needed)."""
        try:
            import cv2
            self._model = "opencv"
            console.print("[green]OpenCV backend initialized (lightweight)[/green]")
        except ImportError:
            console.print("[yellow]OpenCV not installed. Install with: pip install opencv-python[/yellow]")
            self._model = None

    def detect_facade_region(
        self,
        image: Image.Image | np.ndarray | str | Path,
    ) -> tuple[np.ndarray, tuple[int, int, int, int]]:
        """
        Detect the facade/building region in a street-level image.

        Removes sky (top) and ground/street (bottom) to isolate the building.

        Args:
            image: Input street-level image

        Returns:
            (cropped_image, bbox) where bbox is (x1, y1, x2, y2)
        """
        import cv2

        # Load image
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray) and len(image.shape) == 3:
            image = Image.fromarray(image)

        img_array = np.array(image)
        h, w = img_array.shape[:2]

        # Convert to different color spaces for analysis
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)

        # === SKY DETECTION (top region) ===
        # Sky is usually bright and has high saturation blue or uniform gray
        # Check the upper portion of the image
        upper_region = hsv[:h//3, :, :]

        # Sky detection: high value (brightness) and either blue hue or low saturation (gray sky)
        v_channel = upper_region[:, :, 2]
        s_channel = upper_region[:, :, 1]
        h_channel = upper_region[:, :, 0]

        # Sky mask: bright + (blue or gray)
        bright_mask = v_channel > 150
        blue_mask = (h_channel > 90) & (h_channel < 130)  # Blue hue range
        gray_sky_mask = s_channel < 50  # Low saturation = gray
        sky_mask = bright_mask & (blue_mask | gray_sky_mask)

        # Find where sky ends (row-wise)
        sky_ratio_per_row = sky_mask.mean(axis=1)
        sky_end = 0
        for i, ratio in enumerate(sky_ratio_per_row):
            if ratio > 0.3:  # More than 30% sky pixels
                sky_end = i
            else:
                break

        # Add small buffer and ensure we don't cut too much
        sky_end = min(sky_end + 10, h // 4)

        # === GROUND DETECTION (bottom region) ===
        # Ground/street is usually darker, more uniform, in lower portion
        lower_region = gray[2*h//3:, :]

        # Ground detection: look for uniform texture (low variance) or dark regions
        ground_start = h
        for i in range(lower_region.shape[0] - 1, -1, -1):
            row = lower_region[i, :]
            row_variance = np.var(row)
            row_mean = np.mean(row)

            # Ground: low variance (uniform) or dark
            if row_variance < 500 or row_mean < 80:
                ground_start = 2*h//3 + i
            else:
                break

        # Add buffer and ensure we don't cut too much
        ground_start = max(ground_start - 10, 3*h//4)

        # === SIDE DETECTION ===
        # For now, use full width but could add lateral cropping later
        left = 0
        right = w

        # Apply minimum bounds
        top = max(0, sky_end)
        bottom = min(h, ground_start)

        # Ensure we have a valid region (at least 40% of image height)
        if bottom - top < h * 0.4:
            # Fall back to middle 60% of image
            top = int(h * 0.15)
            bottom = int(h * 0.85)

        bbox = (left, top, right, bottom)
        cropped = img_array[top:bottom, left:right]

        return cropped, bbox

    def detect_facade_with_sam(
        self,
        image: Image.Image | np.ndarray | str | Path,
        prompt: str = "building facade",
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """
        Detect building facade using SAM with text prompt.

        Uses LangSAM to segment the building facade from street-level images,
        providing more accurate isolation than simple CV-based cropping.

        Args:
            image: Input street-level image
            prompt: Text prompt for building detection

        Returns:
            (cropped_image, mask, confidence)
        """
        self._lazy_init()

        # Load image
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        img_array = np.array(image)
        h, w = img_array.shape[:2]

        if self.backend != "lang_sam" or self._model is None:
            # Fallback to CV-based detection
            cropped, bbox = self.detect_facade_region(image)
            mask = np.ones((cropped.shape[0], cropped.shape[1]), dtype=np.uint8) * 255
            return cropped, mask, 0.5

        try:
            # Use LangSAM to detect building facade
            prediction_results = self._model.predict(
                images_pil=[image],
                texts_prompt=[prompt],
                box_threshold=0.2,
                text_threshold=0.2,
            )

            if not prediction_results or len(prediction_results) == 0:
                # Fallback
                cropped, bbox = self.detect_facade_region(image)
                mask = np.ones((cropped.shape[0], cropped.shape[1]), dtype=np.uint8) * 255
                return cropped, mask, 0.3

            result = prediction_results[0]
            masks = result.get("masks", np.array([]))
            scores = result.get("scores", np.array([]))
            boxes = result.get("boxes", np.array([]))

            if len(masks) == 0:
                cropped, bbox = self.detect_facade_region(image)
                mask = np.ones((cropped.shape[0], cropped.shape[1]), dtype=np.uint8) * 255
                return cropped, mask, 0.3

            # Find largest building mask (by area)
            best_idx = 0
            best_area = 0
            for i, m in enumerate(masks):
                mask_array = m.cpu().numpy() if hasattr(m, 'cpu') else m
                if mask_array.ndim == 3:
                    mask_array = mask_array[0]
                area = mask_array.sum()
                if area > best_area:
                    best_area = area
                    best_idx = i

            # Get the best mask
            best_mask = masks[best_idx]
            if hasattr(best_mask, 'cpu'):
                best_mask = best_mask.cpu().numpy()
            if best_mask.ndim == 3:
                best_mask = best_mask[0]

            confidence = float(scores[best_idx]) if len(scores) > best_idx else 0.5

            # Get bounding box of mask
            rows = np.any(best_mask, axis=1)
            cols = np.any(best_mask, axis=0)
            if not np.any(rows) or not np.any(cols):
                cropped, bbox = self.detect_facade_region(image)
                mask = np.ones((cropped.shape[0], cropped.shape[1]), dtype=np.uint8) * 255
                return cropped, mask, 0.3

            y_min, y_max = np.where(rows)[0][[0, -1]]
            x_min, x_max = np.where(cols)[0][[0, -1]]

            # Add small padding
            pad = 10
            y_min = max(0, y_min - pad)
            y_max = min(h, y_max + pad)
            x_min = max(0, x_min - pad)
            x_max = min(w, x_max + pad)

            # Crop image and mask
            cropped = img_array[y_min:y_max, x_min:x_max]
            cropped_mask = best_mask[y_min:y_max, x_min:x_max].astype(np.uint8) * 255

            console.print(f"[dim]SAM facade: {cropped.shape[1]}x{cropped.shape[0]}, conf={confidence:.1%}[/dim]")
            return cropped, cropped_mask, confidence

        except Exception as e:
            console.print(f"[yellow]SAM facade detection failed: {e}[/yellow]")
            cropped, bbox = self.detect_facade_region(image)
            mask = np.ones((cropped.shape[0], cropped.shape[1]), dtype=np.uint8) * 255
            return cropped, mask, 0.3

    def _init_lang_sam(self) -> None:
        """Initialize LangSAM backend."""
        try:
            from lang_sam import LangSAM

            self._model = LangSAM()
            console.print("[green]LangSAM initialized[/green]")
        except ImportError:
            console.print(
                "[yellow]LangSAM not installed. Install with: pip install lang-sam[/yellow]"
            )
            self._model = None

    def _init_grounded_sam(self) -> None:
        """Initialize Grounded SAM backend."""
        try:
            # This requires the full Grounded-SAM installation
            from groundingdino.util.inference import load_model, predict
            from segment_anything import sam_model_registry, SamPredictor

            console.print("[yellow]Grounded SAM requires manual setup[/yellow]")
            console.print("See: https://github.com/IDEA-Research/Grounded-Segment-Anything")
            self._model = None
        except ImportError:
            console.print(
                "[yellow]Grounded SAM not installed[/yellow]"
            )
            self._model = None

    def _init_sam(self) -> None:
        """
        Initialize Meta's Segment Anything Model (SAM).

        Downloads checkpoint automatically if not present.
        Uses HuggingFace transformers for easier setup.
        """
        try:
            from transformers import SamModel, SamProcessor
            import torch

            # Map model types to HuggingFace model names
            model_map = {
                "vit_b": "facebook/sam-vit-base",
                "vit_l": "facebook/sam-vit-large",
                "vit_h": "facebook/sam-vit-huge",
            }

            model_name = model_map.get(self.sam_model_type, "facebook/sam-vit-base")

            console.print(f"[cyan]Loading SAM model: {model_name}...[/cyan]")

            self._sam_processor = SamProcessor.from_pretrained(model_name)
            self._model = SamModel.from_pretrained(model_name)

            # Move to device
            if self.device == "cuda" and torch.cuda.is_available():
                self._model = self._model.to("cuda")
            elif self.device == "mps" and torch.backends.mps.is_available():
                self._model = self._model.to("mps")
            else:
                self._model = self._model.to("cpu")

            self._model.eval()
            console.print(f"[green]SAM ({self.sam_model_type}) initialized on {self.device}[/green]")

        except ImportError:
            console.print(
                "[yellow]transformers not installed for SAM. Install with: pip install transformers torch[/yellow]"
            )
            self._model = None
        except Exception as e:
            console.print(f"[red]SAM initialization failed: {e}[/red]")
            self._model = None

    def detect_windows(
        self,
        image: Image.Image | np.ndarray | str | Path,
        text_prompt: str = "window",
        confidence_threshold: float = 0.3,
    ) -> list[DetectionResult]:
        """
        Detect windows in a facade image.

        Args:
            image: Input image (PIL, numpy, or path)
            text_prompt: Text prompt for detection
            confidence_threshold: Minimum confidence for detections

        Returns:
            List of DetectionResult objects
        """
        self._lazy_init()

        # Load image if path
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        if self._model is None:
            console.print("[yellow]No AI model available, returning empty results[/yellow]")
            return []

        if self.backend == "lang_sam":
            return self._detect_lang_sam(image, text_prompt, confidence_threshold)
        elif self.backend == "grounded_sam":
            return self._detect_grounded_sam(image, text_prompt, confidence_threshold)
        elif self.backend == "sam":
            return self._detect_sam(image, confidence_threshold)
        elif self.backend == "opencv":
            return self._detect_opencv(image, confidence_threshold)
        else:
            return []

    def _detect_opencv(
        self,
        image: Image.Image,
        confidence_threshold: float,
    ) -> list[DetectionResult]:
        """
        Detect windows using OpenCV with multiple strategies.

        Combines:
        1. Canny edge detection + contour analysis
        2. Color-based segmentation (windows often darker/different color)
        3. Grid pattern detection (windows often in regular patterns)
        """
        import cv2

        # Convert PIL to OpenCV format
        img_array = np.array(image)
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
        else:
            gray = img_array
            hsv = None

        h, w = gray.shape

        # Resize if image is very large (faster processing)
        max_dim = 1500
        scale = 1.0
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            new_h, new_w = int(h * scale), int(w * scale)
            gray = cv2.resize(gray, (new_w, new_h))
            if hsv is not None:
                hsv = cv2.resize(hsv, (new_w, new_h))
            h, w = new_h, new_w

        results = []

        # Strategy 1: Edge detection + morphology
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)

        # Dilate to connect edges
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        dilated = cv2.dilate(edges, kernel, iterations=2)

        # Close gaps
        closed = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel, iterations=3)

        # Find contours
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        min_area = (h * w) * 0.0005  # 0.05% of image
        max_area = (h * w) * 0.1     # 10% of image

        for contour in contours:
            area = cv2.contourArea(contour)

            if area < min_area or area > max_area:
                continue

            # Approximate to polygon
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)

            # Windows are typically 4-sided (rectangular)
            if len(approx) < 4 or len(approx) > 8:
                continue

            x, y, rect_w, rect_h = cv2.boundingRect(contour)

            # Aspect ratio filter (windows 0.4 to 2.5)
            aspect = rect_w / rect_h if rect_h > 0 else 0
            if aspect < 0.4 or aspect > 2.5:
                continue

            # Rectangularity
            rect_area = rect_w * rect_h
            rectangularity = area / rect_area if rect_area > 0 else 0

            if rectangularity < 0.6:
                continue

            confidence = min(1.0, rectangularity * 0.7 + len(approx) * 0.05)

            if confidence >= confidence_threshold:
                # Scale bbox back if we resized
                if scale != 1.0:
                    x = int(x / scale)
                    y = int(y / scale)
                    rect_w = int(rect_w / scale)
                    rect_h = int(rect_h / scale)

                # Create mask at original resolution
                orig_h, orig_w = img_array.shape[:2]
                mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
                cv2.rectangle(mask, (x, y), (x + rect_w, y + rect_h), 255, -1)

                results.append(DetectionResult(
                    mask=mask > 0,
                    confidence=confidence,
                    bbox=(x, y, x + rect_w, y + rect_h),
                ))

        # Strategy 2: Dark region detection (windows often appear darker)
        if hsv is not None:
            v_channel = hsv[:, :, 2]
            mean_v = np.mean(v_channel)

            # Find regions darker than average
            dark_mask = v_channel < (mean_v * 0.7)
            dark_mask = dark_mask.astype(np.uint8) * 255

            # Clean up
            dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_OPEN, kernel, iterations=2)
            dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

            dark_contours, _ = cv2.findContours(dark_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in dark_contours:
                area = cv2.contourArea(contour)

                if area < min_area or area > max_area:
                    continue

                x, y, rect_w, rect_h = cv2.boundingRect(contour)
                aspect = rect_w / rect_h if rect_h > 0 else 0

                if aspect < 0.4 or aspect > 2.5:
                    continue

                rect_area = rect_w * rect_h
                rectangularity = area / rect_area if rect_area > 0 else 0

                if rectangularity < 0.5:
                    continue

                confidence = rectangularity * 0.6  # Lower confidence for color-based

                if confidence >= confidence_threshold:
                    if scale != 1.0:
                        x = int(x / scale)
                        y = int(y / scale)
                        rect_w = int(rect_w / scale)
                        rect_h = int(rect_h / scale)

                    orig_h, orig_w = img_array.shape[:2]
                    mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
                    cv2.rectangle(mask, (x, y), (x + rect_w, y + rect_h), 255, -1)

                    results.append(DetectionResult(
                        mask=mask > 0,
                        confidence=confidence,
                        bbox=(x, y, x + rect_w, y + rect_h),
                    ))

        return results

    def _detect_lang_sam(
        self,
        image: Image.Image,
        text_prompt: str,
        confidence_threshold: float,
    ) -> list[DetectionResult]:
        """Detection using LangSAM 0.2.1+ API."""
        try:
            # LangSAM 0.2.1 API: takes lists, returns list of dicts
            # predict(images_pil: list[Image], texts_prompt: list[str], ...)
            # Returns: [{"boxes": np.ndarray, "scores": np.ndarray, "masks": np.ndarray, "mask_scores": np.ndarray}]

            prediction_results = self._model.predict(
                images_pil=[image],
                texts_prompt=[text_prompt],
                box_threshold=confidence_threshold,
                text_threshold=0.25,
            )

            if not prediction_results or len(prediction_results) == 0:
                console.print("[dim]LangSAM: No detections[/dim]")
                return []

            # Get first image result
            result = prediction_results[0]

            boxes = result.get("boxes", np.array([]))
            scores = result.get("scores", np.array([]))
            masks = result.get("masks", np.array([]))

            if len(boxes) == 0:
                console.print("[dim]LangSAM: No objects found[/dim]")
                return []

            detection_results = []
            for i in range(len(boxes)):
                conf = float(scores[i]) if i < len(scores) else 0.5

                if conf >= confidence_threshold:
                    # Get mask for this detection
                    if i < len(masks):
                        mask = masks[i]
                        # Handle different mask formats
                        if hasattr(mask, 'cpu'):
                            mask_np = mask.cpu().numpy()
                        else:
                            mask_np = np.array(mask)

                        # Squeeze extra dimensions
                        while mask_np.ndim > 2:
                            mask_np = mask_np.squeeze()
                        if mask_np.ndim > 2:
                            mask_np = mask_np[0]
                    else:
                        # No mask, create from box
                        box = boxes[i]
                        h, w = image.size[1], image.size[0]
                        mask_np = np.zeros((h, w), dtype=bool)
                        x1, y1, x2, y2 = [int(v) for v in box[:4]]
                        mask_np[y1:y2, x1:x2] = True

                    # Get box
                    box = boxes[i]
                    if hasattr(box, 'tolist'):
                        box_tuple = tuple(int(x) for x in box.tolist()[:4])
                    else:
                        box_tuple = tuple(int(x) for x in box[:4])

                    detection_results.append(DetectionResult(
                        mask=mask_np,
                        confidence=conf,
                        bbox=box_tuple,
                    ))

            console.print(f"[green]LangSAM: Found {len(detection_results)} windows[/green]")
            return detection_results

        except Exception as e:
            console.print(f"[yellow]LangSAM failed ({type(e).__name__}: {e}), using OpenCV fallback[/yellow]")
            return self._detect_opencv(image, confidence_threshold)

    def _detect_grounded_sam(
        self,
        image: Image.Image,
        text_prompt: str,
        confidence_threshold: float,
    ) -> list[DetectionResult]:
        """Detection using Grounded SAM."""
        # Placeholder for Grounded SAM implementation
        console.print("[yellow]Grounded SAM detection not yet implemented[/yellow]")
        return []

    def _detect_sam(
        self,
        image: Image.Image,
        confidence_threshold: float,
    ) -> list[DetectionResult]:
        """
        Detection using Meta's SAM with automatic mask generation.

        SAM generates all possible masks, then we filter for window-like regions
        based on aspect ratio, size, and position.
        """
        import torch
        import cv2

        try:
            img_array = np.array(image)
            h, w = img_array.shape[:2]

            # Generate grid of point prompts to find all objects
            # Use a 16x16 grid of points
            grid_size = 16
            point_coords = []
            for i in range(grid_size):
                for j in range(grid_size):
                    px = int((j + 0.5) * w / grid_size)
                    py = int((i + 0.5) * h / grid_size)
                    point_coords.append([px, py])

            point_coords = np.array(point_coords)

            # Process in batches for efficiency
            batch_size = 64
            all_masks = []

            for i in range(0, len(point_coords), batch_size):
                batch_points = point_coords[i:i + batch_size]

                # Prepare inputs with point prompts
                inputs = self._sam_processor(
                    image,
                    input_points=[batch_points.tolist()],
                    return_tensors="pt",
                )

                # Move to device
                device = next(self._model.parameters()).device
                inputs = {k: v.to(device) if hasattr(v, 'to') else v for k, v in inputs.items()}

                with torch.no_grad():
                    outputs = self._model(**inputs)

                # Get masks from outputs
                masks = self._sam_processor.image_processor.post_process_masks(
                    outputs.pred_masks,
                    inputs["original_sizes"],
                    inputs["reshaped_input_sizes"],
                )

                # Process each mask
                if masks and len(masks) > 0:
                    for mask_batch in masks:
                        for mask in mask_batch:
                            mask_np = mask.cpu().numpy().squeeze()
                            if mask_np.ndim == 2:
                                all_masks.append(mask_np)
                            elif mask_np.ndim == 3:
                                # Take first channel if multi-channel
                                all_masks.append(mask_np[0])

            # Filter masks for window-like characteristics
            results = []
            min_area = (h * w) * 0.001  # 0.1% of image
            max_area = (h * w) * 0.15    # 15% of image

            # Remove duplicates by computing IoU
            unique_masks = []
            for mask in all_masks:
                mask_bool = mask > 0.5
                area = mask_bool.sum()

                if area < min_area or area > max_area:
                    continue

                # Check for duplicates
                is_duplicate = False
                for existing in unique_masks:
                    intersection = (mask_bool & existing).sum()
                    union = (mask_bool | existing).sum()
                    iou = intersection / union if union > 0 else 0
                    if iou > 0.7:
                        is_duplicate = True
                        break

                if not is_duplicate:
                    unique_masks.append(mask_bool)

            # Analyze each unique mask for window characteristics
            for mask_bool in unique_masks:
                # Find bounding box
                coords = np.where(mask_bool)
                if len(coords[0]) == 0:
                    continue

                y1, y2 = coords[0].min(), coords[0].max()
                x1, x2 = coords[1].min(), coords[1].max()

                rect_w = x2 - x1
                rect_h = y2 - y1

                if rect_w < 10 or rect_h < 10:
                    continue

                # Aspect ratio check (windows are typically 0.4 to 2.5)
                aspect = rect_w / rect_h
                if aspect < 0.3 or aspect > 3.0:
                    continue

                # Rectangularity check
                mask_area = mask_bool.sum()
                rect_area = rect_w * rect_h
                rectangularity = mask_area / rect_area if rect_area > 0 else 0

                if rectangularity < 0.5:
                    continue

                # Position check - windows usually in middle portion of facade
                center_y = (y1 + y2) / 2 / h
                if center_y < 0.1 or center_y > 0.95:  # Not at very top (sky) or very bottom (ground)
                    continue

                # Calculate confidence based on characteristics
                aspect_score = 1.0 - abs(aspect - 1.2) / 2  # Optimal ~1.2
                rect_score = rectangularity
                position_score = 1.0 if 0.2 < center_y < 0.85 else 0.7

                confidence = (aspect_score + rect_score + position_score) / 3

                if confidence >= confidence_threshold:
                    # Create mask at full resolution
                    full_mask = np.zeros((h, w), dtype=np.uint8)
                    full_mask[mask_bool] = 255

                    results.append(DetectionResult(
                        mask=full_mask > 0,
                        confidence=float(confidence),
                        bbox=(int(x1), int(y1), int(x2), int(y2)),
                    ))

            console.print(f"[dim]SAM detected {len(results)} window candidates[/dim]")
            return results

        except Exception as e:
            console.print(f"[red]SAM detection failed: {e}[/red]")
            # Fall back to OpenCV
            console.print("[yellow]Falling back to OpenCV detection[/yellow]")
            return self._detect_opencv(image, confidence_threshold)

    def calculate_wwr(
        self,
        image: Image.Image | np.ndarray | str | Path,
        wall_mask: np.ndarray | None = None,
        crop_facade: bool = True,
        use_sam_crop: bool = False,
        apply_quality_filter: bool = True,
        apply_geometry_filter: bool = True,
    ) -> tuple[float, float]:
        """
        Calculate Window-to-Wall Ratio for a facade image.

        Enhanced version with:
        - Image quality assessment
        - Geometric post-processing
        - Grid pattern detection for confidence boost

        Args:
            image: Facade image
            wall_mask: Optional mask of wall area (if None, uses full image)
            crop_facade: If True, crop out sky/ground for street-level images
            use_sam_crop: If True, use SAM to segment building facade (higher quality but slower)
            apply_quality_filter: If True, assess image quality first
            apply_geometry_filter: If True, filter detections by geometric properties

        Returns:
            (wwr, confidence) tuple
        """
        # Load image
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        img_array = np.array(image)
        facade_confidence_boost = 0.0
        quality_penalty = 0.0

        # Quality assessment (optional)
        if apply_quality_filter:
            quality = self.assess_image_quality(image)
            if not quality.is_usable:
                # Return low confidence for unusable images
                return 0.0, 0.1
            # Apply quality-based confidence adjustment
            quality_penalty = max(0, (0.6 - quality.overall_score) * 0.3)

        # Optionally crop to facade region (removes sky/ground)
        if crop_facade and wall_mask is None:
            try:
                if use_sam_crop and self.backend == "lang_sam":
                    # Use SAM for more accurate building segmentation
                    cropped, facade_mask, facade_conf = self.detect_facade_with_sam(image)
                    detection_image = Image.fromarray(cropped)
                    # Use mask to calculate actual facade area (not just bounding box)
                    facade_area = facade_mask.sum() / 255 if facade_mask.max() > 0 else cropped.shape[0] * cropped.shape[1]
                    # Boost confidence if SAM segmentation was good
                    facade_confidence_boost = facade_conf * 0.2
                else:
                    # Use simple CV-based cropping
                    cropped, bbox = self.detect_facade_region(image)
                    detection_image = Image.fromarray(cropped)
                    facade_area = cropped.shape[0] * cropped.shape[1]
            except Exception as e:
                console.print(f"[dim]Facade crop failed: {e}[/dim]")
                detection_image = image
                facade_area = img_array.shape[0] * img_array.shape[1]
        else:
            detection_image = image
            facade_area = img_array.shape[0] * img_array.shape[1]

        # Detect windows on the (possibly cropped) image
        window_results = self.detect_windows(detection_image, "window")

        if not window_results:
            return 0.0, 0.0

        # Apply geometric filtering (optional)
        det_array = np.array(detection_image)
        det_h, det_w = det_array.shape[:2]

        if apply_geometry_filter:
            window_results = self.apply_geometric_filter(
                window_results,
                image_height=det_h,
                image_width=det_w,
            )

            if not window_results:
                return 0.0, 0.15  # Low confidence if all windows filtered

            # Check for grid pattern (boosts confidence)
            grid_score = self.detect_window_grid_pattern(window_results, det_w)
            if grid_score > 0.7:
                facade_confidence_boost += 0.1

        # Calculate total window area
        combined_mask = np.zeros(det_array.shape[:2], dtype=bool)

        confidences = []
        for result in window_results:
            # Resize mask if needed
            if result.mask.shape != combined_mask.shape:
                from PIL import Image as PILImage
                mask_img = PILImage.fromarray(result.mask.astype(np.uint8) * 255)
                mask_img = mask_img.resize(combined_mask.shape[::-1])
                result_mask = np.array(mask_img) > 127
            else:
                result_mask = result.mask > 0.5

            combined_mask |= result_mask
            confidences.append(result.confidence)

        window_area = combined_mask.sum()

        # Calculate wall area
        if wall_mask is not None:
            wall_area = wall_mask.sum()
        else:
            # Use facade area (cropped or full)
            wall_area = facade_area

        wwr = window_area / wall_area if wall_area > 0 else 0.0
        avg_confidence = np.mean(confidences) if confidences else 0.0

        # Combine confidence adjustments
        final_confidence = min(1.0, max(0.0,
            avg_confidence + facade_confidence_boost - quality_penalty
        ))

        return float(wwr), float(final_confidence)

    def analyze_facade(
        self,
        image: Image.Image | np.ndarray | str | Path,
        direction: Literal["north", "south", "east", "west"],
    ) -> dict[str, Any]:
        """
        Full facade analysis for a single direction.

        Returns detailed analysis including:
        - WWR value
        - Window count
        - Window positions
        - Confidence score
        """
        # Load image
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        # Detect windows
        window_results = self.detect_windows(image, "window")

        # Detect wall/facade area
        wall_results = self.detect_windows(image, "wall OR facade OR building facade")

        # Calculate WWR
        img_array = np.array(image)
        h, w = img_array.shape[:2]

        # Combine window masks
        window_mask = np.zeros((h, w), dtype=bool)
        for result in window_results:
            if result.mask.shape == (h, w):
                window_mask |= result.mask > 0.5
            else:
                # Resize if needed
                from scipy.ndimage import zoom
                scale_h = h / result.mask.shape[0]
                scale_w = w / result.mask.shape[1]
                resized = zoom(result.mask.astype(float), (scale_h, scale_w)) > 0.5
                window_mask |= resized

        # Combine wall masks (or use full image)
        if wall_results:
            wall_mask = np.zeros((h, w), dtype=bool)
            for result in wall_results:
                if result.mask.shape == (h, w):
                    wall_mask |= result.mask > 0.5
            wall_area = wall_mask.sum()
        else:
            wall_area = h * w

        window_area = window_mask.sum()
        wwr = window_area / wall_area if wall_area > 0 else 0.0

        # Average confidence
        confidences = [r.confidence for r in window_results]
        avg_conf = np.mean(confidences) if confidences else 0.0

        return {
            "direction": direction,
            "wwr": float(wwr),
            "window_count": len(window_results),
            "window_area_pixels": int(window_area),
            "wall_area_pixels": int(wall_area),
            "confidence": float(avg_conf),
            "image_size": (w, h),
        }

    def analyze_all_facades(
        self,
        facade_images: dict[str, Image.Image | str | Path],
    ) -> WindowToWallRatio:
        """
        Analyze all facades and return combined WWR model.

        Args:
            facade_images: Dict mapping direction to image
                          {"north": img_n, "south": img_s, ...}

        Returns:
            WindowToWallRatio model with all directions
        """
        results = {}
        confidences = []

        for direction, image in facade_images.items():
            if direction in ["north", "south", "east", "west"]:
                analysis = self.analyze_facade(image, direction)
                results[direction] = analysis["wwr"]
                confidences.append(analysis["confidence"])

        # Calculate average
        values = [v for v in results.values() if v > 0]
        average = np.mean(values) if values else 0.0
        avg_confidence = np.mean(confidences) if confidences else 0.0

        return WindowToWallRatio(
            north=results.get("north", 0.0),
            south=results.get("south", 0.0),
            east=results.get("east", 0.0),
            west=results.get("west", 0.0),
            average=float(average),
            source=f"ai_{self.backend}",
            confidence=float(avg_confidence),
        )


def estimate_wwr_from_era(construction_year: int) -> WindowToWallRatio:
    """
    Estimate WWR based on construction era (fallback when no images).

    Swedish building practices by era:
    - Pre-1960: Smaller windows, ~15-20%
    - 1960-1975 (Million program): Standardized, ~20-25%
    - 1975-1990: Energy conscious, ~20%
    - 1990-2010: Increasing glass, ~25-30%
    - Post-2010: Larger windows, ~30-40%
    """
    if construction_year < 1960:
        base_wwr = 0.18
    elif construction_year < 1975:
        base_wwr = 0.22
    elif construction_year < 1990:
        base_wwr = 0.20
    elif construction_year < 2010:
        base_wwr = 0.27
    else:
        base_wwr = 0.35

    # South typically has more glass
    south_factor = 1.2
    north_factor = 0.8

    return WindowToWallRatio(
        north=base_wwr * north_factor,
        south=base_wwr * south_factor,
        east=base_wwr,
        west=base_wwr,
        average=base_wwr,
        source="era_estimation",
        confidence=0.5,  # Lower confidence for estimates
    )
