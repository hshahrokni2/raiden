"""
SOLOv2/YOLOv8-based Window-to-Wall Ratio Detection (2026 Roadmap).

Upgrades from OpenCV edge detection (~70% accuracy) to instance
segmentation (93% mAP according to literature).

Supports multiple backends in priority order:
1. SOLOv2 (detectron2) - Best accuracy, complex installation
2. YOLOv8-seg (ultralytics) - Good accuracy, easy installation
3. SAM + CLIP (existing) - Fallback

Reference: https://www.sciencedirect.com/science/article/abs/pii/S0378778823005054

Hardware Requirements:
- NVIDIA GPU with 4GB+ VRAM (recommended)
- Or CPU inference (slower)

Installation:
    # Option 1: YOLOv8 (RECOMMENDED - easiest)
    pip install ultralytics

    # Option 2: SOLOv2 (best accuracy)
    pip install detectron2 torch torchvision
    # Download weights from: https://github.com/WongKinYiu/SOLO

Usage:
    from src.ai.wwr_detector_v2 import SOLOv2WWRDetector

    detector = SOLOv2WWRDetector(device="cuda")
    wwr = detector.calculate_wwr(facade_image)
    print(f"WWR: {wwr.average:.1%}")
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
from pathlib import Path
from enum import Enum
import logging
import os

import numpy as np

logger = logging.getLogger(__name__)

# Check for available backends
PYTORCH_AVAILABLE = False
DETECTRON2_AVAILABLE = False
ULTRALYTICS_AVAILABLE = False

try:
    import torch
    import torchvision
    PYTORCH_AVAILABLE = True
except ImportError:
    logger.info("PyTorch not available.")

try:
    from detectron2.config import get_cfg
    from detectron2.engine import DefaultPredictor
    from detectron2 import model_zoo
    DETECTRON2_AVAILABLE = True
except ImportError:
    logger.info("detectron2 not available. SOLOv2 disabled.")

try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    logger.info("ultralytics not available. YOLOv8 disabled.")

# Determine best available backend
if DETECTRON2_AVAILABLE:
    SOLOV2_AVAILABLE = True
    DEFAULT_BACKEND = "solov2"
elif ULTRALYTICS_AVAILABLE:
    SOLOV2_AVAILABLE = True  # YOLOv8 is good enough
    DEFAULT_BACKEND = "yolov8"
elif PYTORCH_AVAILABLE:
    SOLOV2_AVAILABLE = True  # SAM fallback
    DEFAULT_BACKEND = "sam"
else:
    SOLOV2_AVAILABLE = False
    DEFAULT_BACKEND = "opencv"


class SegmentationBackend(str, Enum):
    """Available segmentation backends."""
    SOLOV2 = "solov2"  # detectron2 SOLOv2
    YOLOV8 = "yolov8"  # ultralytics YOLOv8-seg
    SAM = "sam"        # Segment Anything + prompts
    OPENCV = "opencv"  # Legacy edge detection


@dataclass
class WindowDetection:
    """Single window detection result."""
    mask: np.ndarray  # Binary mask of window region
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    confidence: float
    area_pixels: int
    aspect_ratio: float = 1.0  # width/height
    is_rectangular: bool = True


@dataclass
class WindowToWallRatio:
    """WWR results per orientation and overall."""
    north: float = 0.0
    south: float = 0.0
    east: float = 0.0
    west: float = 0.0
    average: float = 0.0
    confidence: float = 0.0
    n_windows_detected: int = 0
    backend_used: str = ""
    processing_time_ms: float = 0.0


class SOLOv2WWRDetector:
    """
    Instance segmentation-based window detection for WWR calculation.

    Uses SOLOv2/YOLOv8 for high-accuracy window segmentation.

    Literature Performance:
    - Window mAP: 93%
    - WWR error: ~5% (within ±5% of ground truth for 94% of facades)

    Attributes:
        model: Loaded segmentation model
        device: CUDA device or CPU
        confidence_threshold: Minimum detection confidence
        backend: Active backend (solov2, yolov8, sam, or opencv)
    """

    # COCO class IDs that might be windows (for pretrained models)
    WINDOW_CLASSES = {
        "window": [0],  # Would need custom trained model
        # For COCO-pretrained, we can use proxy classes
        "tv": [62],  # Similar rectangular shape
        "laptop": [63],  # Screen-like
        "clock": [74],  # Sometimes square
    }

    # Custom window class if using fine-tuned model
    WINDOW_CLASS_ID = 0

    def __init__(
        self,
        device: str = "cuda",
        confidence_threshold: float = 0.5,
        model_weights: Optional[Path] = None,
        backend: Optional[str] = None,
        use_pretrained_windows: bool = True,
    ):
        """
        Initialize SOLOv2/YOLOv8 detector.

        Args:
            device: "cuda" or "cpu"
            confidence_threshold: Minimum confidence for window detection
            model_weights: Path to custom weights (None = auto-download)
            backend: Force specific backend ("solov2", "yolov8", "sam")
            use_pretrained_windows: If True, use pretrained window model
        """
        self.device = device if PYTORCH_AVAILABLE and torch.cuda.is_available() else "cpu"
        self.confidence_threshold = confidence_threshold
        self.model = None
        self.backend = backend or DEFAULT_BACKEND
        self.use_pretrained_windows = use_pretrained_windows
        self._model_weights = model_weights
        self._initialized = False

        if not SOLOV2_AVAILABLE and self.backend != "opencv":
            logger.warning(
                f"Requested backend '{self.backend}' not available. "
                f"Install ultralytics: pip install ultralytics"
            )
            self.backend = "opencv"

        logger.info(f"SOLOv2WWRDetector initialized (backend={self.backend}, device={self.device})")

    def _lazy_init(self):
        """Initialize model on first use."""
        if self._initialized:
            return

        import time
        start = time.time()

        if self.backend == "yolov8" and ULTRALYTICS_AVAILABLE:
            self._init_yolov8()
        elif self.backend == "solov2" and DETECTRON2_AVAILABLE:
            self._init_solov2()
        elif self.backend == "sam" and PYTORCH_AVAILABLE:
            self._init_sam()
        else:
            self.backend = "opencv"
            logger.info("Using OpenCV fallback for window detection")

        self._initialized = True
        logger.info(f"Model initialization took {(time.time() - start)*1000:.0f}ms")

    def _init_yolov8(self):
        """Initialize YOLOv8 segmentation model."""
        if self._model_weights and self._model_weights.exists():
            # Use custom window-trained model
            self.model = YOLO(str(self._model_weights))
            logger.info(f"Loaded custom YOLOv8 model from {self._model_weights}")
        else:
            # Use pretrained YOLOv8-seg (COCO)
            # yolov8n-seg is fastest, yolov8x-seg is most accurate
            model_name = "yolov8m-seg.pt"  # Medium - good balance
            self.model = YOLO(model_name)
            logger.info(f"Loaded pretrained YOLOv8 model: {model_name}")

    def _init_solov2(self):
        """Initialize SOLOv2 with detectron2."""
        cfg = get_cfg()

        if self._model_weights and self._model_weights.exists():
            # Custom weights
            cfg.merge_from_file(model_zoo.get_config_file(
                "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
            ))
            cfg.MODEL.WEIGHTS = str(self._model_weights)
        else:
            # Use Mask R-CNN as SOLOv2 alternative (similar performance)
            cfg.merge_from_file(model_zoo.get_config_file(
                "COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"
            ))
            cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
                "COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"
            )

        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = self.confidence_threshold
        cfg.MODEL.DEVICE = self.device

        self.model = DefaultPredictor(cfg)
        logger.info("Loaded detectron2 Mask R-CNN model")

    def _init_sam(self):
        """Initialize SAM as fallback."""
        try:
            from segment_anything import sam_model_registry, SamPredictor

            # Try to find SAM weights
            sam_checkpoint = Path("models/sam_vit_b_01ec64.pth")
            if not sam_checkpoint.exists():
                sam_checkpoint = Path.home() / ".cache" / "sam" / "sam_vit_b_01ec64.pth"

            if sam_checkpoint.exists():
                sam = sam_model_registry["vit_b"](checkpoint=str(sam_checkpoint))
                sam.to(device=self.device)
                self.model = SamPredictor(sam)
                logger.info("Loaded SAM model")
            else:
                logger.warning("SAM weights not found, falling back to OpenCV")
                self.backend = "opencv"
        except ImportError:
            logger.warning("segment_anything not available, using OpenCV")
            self.backend = "opencv"

    def detect_windows(self, image: np.ndarray) -> List[WindowDetection]:
        """
        Detect windows in facade image.

        Args:
            image: BGR image (OpenCV format)

        Returns:
            List of WindowDetection results
        """
        self._lazy_init()

        if self.backend == "yolov8":
            return self._detect_yolov8(image)
        elif self.backend == "solov2":
            return self._detect_solov2(image)
        elif self.backend == "sam":
            return self._detect_sam(image)
        else:
            return self._detect_opencv(image)

    def _detect_yolov8(self, image: np.ndarray) -> List[WindowDetection]:
        """Detect windows using YOLOv8-seg."""
        # Run inference
        results = self.model(image, verbose=False)

        detections = []
        for result in results:
            if result.masks is None:
                continue

            masks = result.masks.data.cpu().numpy()
            boxes = result.boxes.data.cpu().numpy()

            for i, (mask, box) in enumerate(zip(masks, boxes)):
                # box format: x1, y1, x2, y2, conf, class_id
                x1, y1, x2, y2, conf, class_id = box[:6]

                # For custom window model, filter by class
                # For COCO model, use geometric filtering instead
                if not self.use_pretrained_windows:
                    # Custom model - filter by window class
                    if int(class_id) != self.WINDOW_CLASS_ID:
                        continue

                # Resize mask to image size if needed
                h, w = image.shape[:2]
                if mask.shape != (h, w):
                    import cv2
                    mask = cv2.resize(mask.astype(np.uint8), (w, h))

                # Calculate area
                area = int(np.sum(mask > 0.5))

                # Aspect ratio (width/height)
                width = x2 - x1
                height = y2 - y1
                aspect = width / height if height > 0 else 1.0

                # Filter by window-like characteristics
                if self.use_pretrained_windows:
                    # Geometric filtering for COCO models
                    if not self._is_window_like(aspect, area, h * w):
                        continue

                detections.append(WindowDetection(
                    mask=(mask > 0.5).astype(np.uint8),
                    bbox=(int(x1), int(y1), int(x2), int(y2)),
                    confidence=float(conf),
                    area_pixels=area,
                    aspect_ratio=aspect,
                    is_rectangular=True,
                ))

        return detections

    def _detect_solov2(self, image: np.ndarray) -> List[WindowDetection]:
        """Detect windows using SOLOv2/Mask R-CNN."""
        outputs = self.model(image)

        detections = []
        instances = outputs["instances"]

        if len(instances) == 0:
            return detections

        masks = instances.pred_masks.cpu().numpy()
        boxes = instances.pred_boxes.tensor.cpu().numpy()
        scores = instances.scores.cpu().numpy()
        classes = instances.pred_classes.cpu().numpy()

        h, w = image.shape[:2]

        for mask, box, score, cls in zip(masks, boxes, scores, classes):
            if score < self.confidence_threshold:
                continue

            x1, y1, x2, y2 = box
            width = x2 - x1
            height = y2 - y1
            aspect = width / height if height > 0 else 1.0
            area = int(np.sum(mask))

            # Geometric filtering
            if not self._is_window_like(aspect, area, h * w):
                continue

            detections.append(WindowDetection(
                mask=mask.astype(np.uint8),
                bbox=(int(x1), int(y1), int(x2), int(y2)),
                confidence=float(score),
                area_pixels=area,
                aspect_ratio=aspect,
                is_rectangular=True,
            ))

        return detections

    def _detect_sam(self, image: np.ndarray) -> List[WindowDetection]:
        """Detect windows using SAM with grid prompts."""
        # This uses automatic mask generation with geometric filtering
        try:
            from segment_anything import SamAutomaticMaskGenerator

            mask_generator = SamAutomaticMaskGenerator(
                self.model.model,
                points_per_side=32,
                pred_iou_thresh=0.88,
                stability_score_thresh=0.95,
                min_mask_region_area=100,
            )

            # Generate all masks
            masks = mask_generator.generate(image)

            detections = []
            h, w = image.shape[:2]

            for mask_data in masks:
                mask = mask_data["segmentation"]
                bbox = mask_data["bbox"]  # x, y, w, h format
                x, y, mw, mh = bbox
                area = mask_data["area"]
                conf = mask_data["predicted_iou"]

                # Convert bbox to x1, y1, x2, y2
                x1, y1 = x, y
                x2, y2 = x + mw, y + mh

                aspect = mw / mh if mh > 0 else 1.0

                # Geometric filtering for window-like shapes
                if not self._is_window_like(aspect, area, h * w):
                    continue

                detections.append(WindowDetection(
                    mask=mask.astype(np.uint8),
                    bbox=(int(x1), int(y1), int(x2), int(y2)),
                    confidence=float(conf),
                    area_pixels=int(area),
                    aspect_ratio=aspect,
                    is_rectangular=True,
                ))

            return detections

        except Exception as e:
            logger.warning(f"SAM detection failed: {e}, falling back to OpenCV")
            return self._detect_opencv(image)

    def _detect_opencv(self, image: np.ndarray) -> List[WindowDetection]:
        """
        Fallback OpenCV-based window detection.

        Uses two complementary strategies:
        1. Edge detection with morphological operations
        2. Dark region detection (windows often appear darker than walls)
        """
        import cv2

        h, w = image.shape[:2]

        # Resize for faster processing if image is large
        max_dim = 1500
        scale = 1.0
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            new_h, new_w = int(h * scale), int(w * scale)
            proc_image = cv2.resize(image, (new_w, new_h))
        else:
            proc_image = image
            new_h, new_w = h, w

        # Convert to grayscale and HSV
        gray = cv2.cvtColor(proc_image, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(proc_image, cv2.COLOR_BGR2HSV)

        detections = []

        # Area thresholds (more permissive like V1)
        min_area = (new_h * new_w) * 0.0005  # 0.05% of image (V1 threshold)
        max_area = (new_h * new_w) * 0.15    # 15% of image

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

        # === Strategy 1: Edge detection + morphology ===
        # Apply Gaussian blur before edge detection (reduces noise)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Multi-scale edge detection
        edges1 = cv2.Canny(blurred, 30, 100)
        edges2 = cv2.Canny(blurred, 50, 150)
        edges = cv2.bitwise_or(edges1, edges2)

        # Dilate to connect nearby edges
        dilated = cv2.dilate(edges, kernel, iterations=2)

        # Close gaps to form complete window outlines
        closed = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel, iterations=3)

        # Find contours
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            area = cv2.contourArea(contour)

            if area < min_area or area > max_area:
                continue

            # Approximate to polygon (windows are typically 4-8 sided)
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)

            if len(approx) < 4 or len(approx) > 8:
                continue

            x, y, cw, ch = cv2.boundingRect(contour)

            if cw <= 0 or ch <= 0:
                continue

            aspect = cw / ch

            # Aspect ratio filter (windows 0.4 to 2.5)
            if aspect < 0.4 or aspect > 2.5:
                continue

            # Rectangularity check
            rect_area = cw * ch
            rectangularity = area / rect_area if rect_area > 0 else 0

            if rectangularity < 0.6:
                continue

            # Calculate confidence based on shape
            confidence = min(1.0, rectangularity * 0.7 + len(approx) * 0.05)

            # Scale bbox back to original size if needed
            if scale != 1.0:
                x = int(x / scale)
                y = int(y / scale)
                cw = int(cw / scale)
                ch = int(ch / scale)

            # Create mask at original resolution
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.rectangle(mask, (x, y), (x + cw, y + ch), 1, -1)

            detections.append(WindowDetection(
                mask=mask,
                bbox=(x, y, x + cw, y + ch),
                confidence=confidence,
                area_pixels=cw * ch,
                aspect_ratio=aspect,
                is_rectangular=True,
            ))

        # === Strategy 2: Dark region detection ===
        # Windows often appear darker than the surrounding wall
        v_channel = hsv[:, :, 2]
        mean_v = np.mean(v_channel)

        # Find regions darker than average
        dark_mask = (v_channel < (mean_v * 0.7)).astype(np.uint8) * 255

        # Clean up with morphological operations
        dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_OPEN, kernel, iterations=2)
        dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        dark_contours, _ = cv2.findContours(dark_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in dark_contours:
            area = cv2.contourArea(contour)

            if area < min_area or area > max_area:
                continue

            x, y, cw, ch = cv2.boundingRect(contour)

            if cw <= 0 or ch <= 0:
                continue

            aspect = cw / ch

            if aspect < 0.4 or aspect > 2.5:
                continue

            rect_area = cw * ch
            rectangularity = area / rect_area if rect_area > 0 else 0

            if rectangularity < 0.5:
                continue

            # Lower confidence for color-based detection
            confidence = rectangularity * 0.6

            if confidence < self.confidence_threshold:
                continue

            # Scale back if needed
            if scale != 1.0:
                x = int(x / scale)
                y = int(y / scale)
                cw = int(cw / scale)
                ch = int(ch / scale)

            # Create mask at original resolution
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.rectangle(mask, (x, y), (x + cw, y + ch), 1, -1)

            detections.append(WindowDetection(
                mask=mask,
                bbox=(x, y, x + cw, y + ch),
                confidence=confidence,
                area_pixels=cw * ch,
                aspect_ratio=aspect,
                is_rectangular=True,
            ))

        # Non-maximum suppression to remove overlapping detections
        detections = self._nms_detections(detections, iou_threshold=0.5)

        return detections

    def _nms_detections(
        self,
        detections: List[WindowDetection],
        iou_threshold: float = 0.5,
    ) -> List[WindowDetection]:
        """Apply non-maximum suppression to remove overlapping detections."""
        if len(detections) <= 1:
            return detections

        # Sort by area (larger first)
        detections = sorted(detections, key=lambda d: d.area_pixels, reverse=True)

        keep = []
        for det in detections:
            # Check overlap with kept detections
            should_keep = True
            x1, y1, x2, y2 = det.bbox

            for kept in keep:
                kx1, ky1, kx2, ky2 = kept.bbox

                # Calculate IoU
                ix1 = max(x1, kx1)
                iy1 = max(y1, ky1)
                ix2 = min(x2, kx2)
                iy2 = min(y2, ky2)

                if ix1 < ix2 and iy1 < iy2:
                    intersection = (ix2 - ix1) * (iy2 - iy1)
                    area1 = (x2 - x1) * (y2 - y1)
                    area2 = (kx2 - kx1) * (ky2 - ky1)
                    union = area1 + area2 - intersection
                    iou = intersection / union if union > 0 else 0

                    if iou > iou_threshold:
                        should_keep = False
                        break

            if should_keep:
                keep.append(det)

        return keep

    def _is_window_like(
        self,
        aspect_ratio: float,
        area: int,
        image_area: int,
    ) -> bool:
        """
        Check if detection has window-like geometric properties.

        Windows typically:
        - Aspect ratio 0.4-2.5 (width/height)
        - Cover 0.5%-15% of facade
        - Are rectangular
        """
        # Aspect ratio check (windows are roughly square to tall rectangles)
        if aspect_ratio < 0.3 or aspect_ratio > 3.0:
            return False

        # Size check (relative to image)
        relative_size = area / image_area if image_area > 0 else 0
        if relative_size < 0.005 or relative_size > 0.20:  # 0.5% - 20%
            return False

        return True

    def calculate_wwr(
        self,
        image: np.ndarray,
        facade_mask: Optional[np.ndarray] = None,
    ) -> WindowToWallRatio:
        """
        Calculate window-to-wall ratio from facade image.

        Args:
            image: Facade image (BGR)
            facade_mask: Optional mask defining facade region (excludes sky, ground)

        Returns:
            WindowToWallRatio with average and per-direction values
        """
        import time
        start = time.time()

        # Get window detections
        windows = self.detect_windows(image)

        if not windows:
            return WindowToWallRatio(
                confidence=0.0,
                backend_used=self.backend,
                processing_time_ms=(time.time() - start) * 1000,
            )

        # Calculate facade area (total or masked)
        if facade_mask is not None:
            facade_area = np.sum(facade_mask > 0)
        else:
            facade_area = image.shape[0] * image.shape[1]

        # Calculate window area (union of all masks to avoid double counting)
        combined_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        for w in windows:
            combined_mask = np.maximum(combined_mask, w.mask)
        window_area = np.sum(combined_mask > 0)

        # Calculate WWR
        wwr = window_area / facade_area if facade_area > 0 else 0.0

        # Clamp to reasonable range
        wwr = min(0.90, max(0.0, wwr))

        # Average confidence
        avg_confidence = np.mean([w.confidence for w in windows]) if windows else 0.0

        return WindowToWallRatio(
            average=wwr,
            confidence=avg_confidence,
            n_windows_detected=len(windows),
            backend_used=self.backend,
            processing_time_ms=(time.time() - start) * 1000,
        )

    def calculate_wwr_from_images(
        self,
        images: Dict[str, np.ndarray],  # {"N": img, "S": img, "E": img, "W": img}
    ) -> WindowToWallRatio:
        """
        Calculate WWR from multiple facade images (one per direction).

        Args:
            images: Dict mapping direction to facade image

        Returns:
            WindowToWallRatio with per-direction values
        """
        results = {}
        total_window_area = 0
        total_facade_area = 0
        confidences = []
        n_windows = 0
        total_time = 0.0

        for direction, image in images.items():
            wwr = self.calculate_wwr(image)
            results[direction.lower()] = wwr.average
            confidences.append(wwr.confidence)
            n_windows += wwr.n_windows_detected
            total_time += wwr.processing_time_ms

            # Accumulate for overall average
            facade_area = image.shape[0] * image.shape[1]
            total_facade_area += facade_area
            total_window_area += wwr.average * facade_area

        overall_wwr = total_window_area / total_facade_area if total_facade_area > 0 else 0.0
        avg_confidence = np.mean(confidences) if confidences else 0.0

        return WindowToWallRatio(
            north=results.get("n", 0.0),
            south=results.get("s", 0.0),
            east=results.get("e", 0.0),
            west=results.get("w", 0.0),
            average=overall_wwr,
            confidence=avg_confidence,
            n_windows_detected=n_windows,
            backend_used=self.backend,
            processing_time_ms=total_time,
        )


def get_wwr_detector(
    backend: str = "auto",
    device: str = "cuda",
    model_weights: Optional[Path] = None,
) -> SOLOv2WWRDetector:
    """
    Factory function to get appropriate WWR detector.

    Args:
        backend: "auto", "solov2", "yolov8", "sam", or "opencv"
        device: "cuda" or "cpu"
        model_weights: Path to custom model weights

    Returns:
        WWR detector instance
    """
    if backend == "auto":
        backend = DEFAULT_BACKEND

    # Validate backend availability
    if backend == "solov2" and not DETECTRON2_AVAILABLE:
        logger.warning("SOLOv2 requires detectron2. Falling back to YOLOv8 or SAM.")
        backend = "yolov8" if ULTRALYTICS_AVAILABLE else "sam"

    if backend == "yolov8" and not ULTRALYTICS_AVAILABLE:
        logger.warning("YOLOv8 requires ultralytics. Falling back to SAM or OpenCV.")
        backend = "sam" if PYTORCH_AVAILABLE else "opencv"

    if backend == "sam" and not PYTORCH_AVAILABLE:
        logger.warning("SAM requires PyTorch. Falling back to OpenCV.")
        backend = "opencv"

    return SOLOv2WWRDetector(
        device=device,
        backend=backend,
        model_weights=model_weights,
    )


# Convenience function for quick WWR calculation
def calculate_wwr_fast(
    image: np.ndarray,
    device: str = "cpu",
) -> float:
    """
    Quick WWR calculation with auto-selected backend.

    Args:
        image: Facade image (BGR)
        device: "cuda" or "cpu"

    Returns:
        WWR as float (0-1)
    """
    detector = get_wwr_detector(backend="auto", device=device)
    result = detector.calculate_wwr(image)
    return result.average
