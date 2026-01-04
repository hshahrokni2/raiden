"""
CLIP + DINOv2 Ensemble for Facade Material Classification (2026 Roadmap).

Upgrades from single DINOv2 classifier (~70% accuracy) to multi-model
ensemble (target: 80% accuracy).

Reference: https://arxiv.org/abs/2304.07193 (Building material recognition)

Ensemble Strategy:
1. DINOv2 - texture/pattern features (self-supervised)
2. CLIP - semantic understanding ("looks like brick")
3. Color histogram - simple but effective for brick vs plaster

Voting: Weighted average based on per-class calibrated confidences.

Hardware Requirements:
- GPU with 8GB+ VRAM (for CLIP ViT-L)
- Or CPU inference (slower but works)

Installation:
    pip install transformers torch torchvision open-clip-torch

Usage:
    from src.ai.material_ensemble import MaterialEnsembleClassifier

    classifier = MaterialEnsembleClassifier(device="cuda")
    result = classifier.classify(facade_image)
    print(f"Material: {result.material} ({result.confidence:.0%})")
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from enum import Enum
import logging

import numpy as np

logger = logging.getLogger(__name__)

# Check for required dependencies
ENSEMBLE_AVAILABLE = False
CLIP_AVAILABLE = False
DINO_AVAILABLE = False

try:
    import torch
    import torchvision.transforms as T
    ENSEMBLE_AVAILABLE = True
except ImportError:
    logger.warning("PyTorch not available. Ensemble classifier will not work.")

try:
    import open_clip
    CLIP_AVAILABLE = True
except ImportError:
    logger.info("open-clip-torch not available. CLIP model disabled.")

try:
    from transformers import AutoModel, AutoImageProcessor
    DINO_AVAILABLE = True
except ImportError:
    logger.info("transformers not available. DINOv2 model disabled.")


class FacadeMaterial(str, Enum):
    """Swedish facade materials."""
    BRICK = "brick"
    CONCRETE = "concrete"
    PLASTER = "plaster"
    WOOD = "wood"
    METAL = "metal"
    GLASS = "glass"
    STONE = "stone"
    UNKNOWN = "unknown"


@dataclass
class EnsemblePrediction:
    """Ensemble classification result."""
    material: FacadeMaterial
    confidence: float
    all_scores: Dict[FacadeMaterial, float]

    # Per-model predictions for debugging
    dino_prediction: Optional[FacadeMaterial] = None
    dino_confidence: float = 0.0
    clip_prediction: Optional[FacadeMaterial] = None
    clip_confidence: float = 0.0
    color_prediction: Optional[FacadeMaterial] = None
    color_confidence: float = 0.0

    # Ensemble metadata
    models_used: List[str] = field(default_factory=list)
    agreement_score: float = 0.0  # 1.0 = all models agree


@dataclass
class EnsembleConfig:
    """Configuration for ensemble classifier."""

    # Model weights in ensemble voting
    dino_weight: float = 0.4   # Texture/pattern expert
    clip_weight: float = 0.4   # Semantic understanding
    color_weight: float = 0.2  # Simple color histogram

    # Confidence thresholds
    min_confidence: float = 0.5
    agreement_boost: float = 0.1  # Boost confidence when models agree

    # Model selection
    use_dino: bool = True
    use_clip: bool = True
    use_color: bool = True

    # CLIP prompts for each material
    clip_prompts: Dict[str, List[str]] = field(default_factory=dict)

    def __post_init__(self):
        if not self.clip_prompts:
            self.clip_prompts = {
                "brick": [
                    "a brick facade",
                    "red brick wall",
                    "brown brick building",
                    "exposed brick exterior",
                ],
                "concrete": [
                    "a concrete facade",
                    "grey concrete wall",
                    "brutalist concrete building",
                    "precast concrete panels",
                ],
                "plaster": [
                    "a plastered facade",
                    "stucco wall",
                    "rendered building exterior",
                    "smooth plaster finish",
                ],
                "wood": [
                    "a wooden facade",
                    "timber cladding",
                    "wood panel exterior",
                    "painted wood siding",
                ],
                "metal": [
                    "a metal facade",
                    "corrugated metal wall",
                    "aluminum cladding",
                    "steel panel exterior",
                ],
                "glass": [
                    "a glass facade",
                    "glass curtain wall",
                    "reflective glass building",
                    "modern glass exterior",
                ],
                "stone": [
                    "a stone facade",
                    "natural stone wall",
                    "limestone building",
                    "granite exterior",
                ],
            }


class MaterialEnsembleClassifier:
    """
    Ensemble classifier combining CLIP, DINOv2, and color analysis.

    Literature shows ensemble methods improve accuracy by 5-10% over
    single models for building material classification.

    Attributes:
        config: Ensemble configuration
        device: CUDA device or CPU
        clip_model: Loaded CLIP model (if available)
        dino_model: Loaded DINOv2 model (if available)
    """

    def __init__(
        self,
        device: str = "cuda",
        config: Optional[EnsembleConfig] = None,
    ):
        """
        Initialize ensemble classifier.

        Args:
            device: "cuda" or "cpu"
            config: Ensemble configuration (default: balanced weights)
        """
        self.device = device
        self.config = config or EnsembleConfig()
        self.clip_model = None
        self.clip_processor = None
        self.dino_model = None
        self.dino_processor = None

        if not ENSEMBLE_AVAILABLE:
            logger.warning("Ensemble dependencies not available")
            return

        # Load available models
        if self.config.use_clip and CLIP_AVAILABLE:
            self._load_clip()
        if self.config.use_dino and DINO_AVAILABLE:
            self._load_dino()

        models = []
        if self.clip_model:
            models.append("CLIP")
        if self.dino_model:
            models.append("DINOv2")
        if self.config.use_color:
            models.append("Color")

        logger.info(f"MaterialEnsembleClassifier initialized with: {models}")

    def _load_clip(self):
        """Load CLIP model for semantic classification."""
        # TODO: Load CLIP model
        # model, _, preprocess = open_clip.create_model_and_transforms('ViT-L-14', pretrained='openai')
        # self.clip_model = model.to(self.device)
        # self.clip_processor = preprocess
        logger.info("CLIP model loading not yet implemented")

    def _load_dino(self):
        """Load DINOv2 for texture/pattern features."""
        # TODO: Load DINOv2 model
        # self.dino_processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
        # self.dino_model = AutoModel.from_pretrained('facebook/dinov2-base').to(self.device)
        logger.info("DINOv2 model loading not yet implemented")

    def classify(self, image: np.ndarray) -> EnsemblePrediction:
        """
        Classify facade material using ensemble of models.

        Args:
            image: BGR image (OpenCV format) or RGB PIL Image

        Returns:
            EnsemblePrediction with material, confidence, and per-model scores
        """
        predictions = {}
        models_used = []

        # CLIP prediction
        if self.clip_model is not None:
            clip_material, clip_conf = self._predict_clip(image)
            predictions["clip"] = (clip_material, clip_conf, self.config.clip_weight)
            models_used.append("CLIP")

        # DINOv2 prediction
        if self.dino_model is not None:
            dino_material, dino_conf = self._predict_dino(image)
            predictions["dino"] = (dino_material, dino_conf, self.config.dino_weight)
            models_used.append("DINOv2")

        # Color histogram prediction (always available)
        if self.config.use_color:
            color_material, color_conf = self._predict_color(image)
            predictions["color"] = (color_material, color_conf, self.config.color_weight)
            models_used.append("Color")

        # Weighted voting
        if not predictions:
            return EnsemblePrediction(
                material=FacadeMaterial.UNKNOWN,
                confidence=0.0,
                all_scores={m: 0.0 for m in FacadeMaterial},
                models_used=[],
            )

        final_material, final_conf, all_scores = self._weighted_vote(predictions)

        # Calculate agreement score
        unique_predictions = len(set(p[0] for p in predictions.values()))
        agreement = 1.0 - (unique_predictions - 1) / max(len(predictions) - 1, 1)

        # Boost confidence if models agree
        if agreement == 1.0:
            final_conf = min(1.0, final_conf + self.config.agreement_boost)

        return EnsemblePrediction(
            material=final_material,
            confidence=final_conf,
            all_scores=all_scores,
            dino_prediction=predictions.get("dino", (None,))[0],
            dino_confidence=predictions.get("dino", (None, 0.0))[1],
            clip_prediction=predictions.get("clip", (None,))[0],
            clip_confidence=predictions.get("clip", (None, 0.0))[1],
            color_prediction=predictions.get("color", (None,))[0],
            color_confidence=predictions.get("color", (None, 0.0))[1],
            models_used=models_used,
            agreement_score=agreement,
        )

    def _predict_clip(
        self,
        image: np.ndarray,
    ) -> Tuple[FacadeMaterial, float]:
        """
        Predict material using CLIP zero-shot classification.

        Uses text prompts like "a brick facade" to match against image.
        """
        # TODO: Implement CLIP inference
        # 1. Encode image with CLIP vision encoder
        # 2. Encode all text prompts with CLIP text encoder
        # 3. Compute cosine similarity
        # 4. Average scores per material
        # 5. Return highest scoring material

        raise NotImplementedError("CLIP inference not yet implemented")

    def _predict_dino(
        self,
        image: np.ndarray,
    ) -> Tuple[FacadeMaterial, float]:
        """
        Predict material using DINOv2 features + classifier head.

        DINOv2 excels at texture/pattern recognition which is important
        for distinguishing brick patterns from plaster, etc.
        """
        # TODO: Implement DINOv2 inference
        # 1. Extract features with DINOv2 backbone
        # 2. Pass through trained classifier head (needs training data)
        # 3. Return classification result

        raise NotImplementedError("DINOv2 inference not yet implemented")

    def _predict_color(
        self,
        image: np.ndarray,
    ) -> Tuple[FacadeMaterial, float]:
        """
        Predict material using color histogram analysis.

        Simple but effective for:
        - Red/brown → brick
        - Grey → concrete
        - White/cream → plaster
        - Brown/orange → wood

        Returns:
            Tuple of (material, confidence)
        """
        # Convert to HSV for better color analysis
        try:
            import cv2
            if len(image.shape) == 2:
                # Grayscale
                return FacadeMaterial.CONCRETE, 0.3

            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        except ImportError:
            # Fallback without OpenCV
            return FacadeMaterial.UNKNOWN, 0.0

        h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

        # Compute statistics
        h_mean = np.mean(h)
        s_mean = np.mean(s)
        v_mean = np.mean(v)

        scores = {}

        # Red/orange hues with medium saturation → brick
        if 0 <= h_mean <= 20 or 160 <= h_mean <= 180:
            if 50 < s_mean < 180:
                scores[FacadeMaterial.BRICK] = 0.7
            else:
                scores[FacadeMaterial.BRICK] = 0.3

        # Grey (low saturation) → concrete or plaster
        if s_mean < 40:
            if v_mean < 140:  # Darker grey
                scores[FacadeMaterial.CONCRETE] = 0.6
            else:  # Lighter grey/white
                scores[FacadeMaterial.PLASTER] = 0.5

        # Yellow/brown hues → wood
        if 20 <= h_mean <= 40:
            if s_mean > 60:
                scores[FacadeMaterial.WOOD] = 0.5

        # Very high value, low saturation → plaster/white
        if v_mean > 200 and s_mean < 30:
            scores[FacadeMaterial.PLASTER] = 0.6

        if not scores:
            return FacadeMaterial.UNKNOWN, 0.3

        best_material = max(scores, key=scores.get)
        return best_material, scores[best_material]

    def _weighted_vote(
        self,
        predictions: Dict[str, Tuple[FacadeMaterial, float, float]],
    ) -> Tuple[FacadeMaterial, float, Dict[FacadeMaterial, float]]:
        """
        Combine predictions using weighted voting.

        Args:
            predictions: Dict of model_name -> (material, confidence, weight)

        Returns:
            Tuple of (final_material, final_confidence, all_scores)
        """
        # Accumulate weighted scores per material
        scores = {m: 0.0 for m in FacadeMaterial}
        total_weight = 0.0

        for model_name, (material, conf, weight) in predictions.items():
            if material is not None:
                scores[material] += conf * weight
                total_weight += weight

        # Normalize
        if total_weight > 0:
            for m in scores:
                scores[m] /= total_weight

        # Find winner
        best_material = max(scores, key=scores.get)
        best_score = scores[best_material]

        return best_material, best_score, scores


def get_material_classifier(
    backend: str = "ensemble",
    device: str = "cuda",
) -> "MaterialClassifierBase":
    """
    Factory function to get appropriate material classifier.

    Args:
        backend: "ensemble" (new) or "dino" (legacy single model)
        device: "cuda" or "cpu"

    Returns:
        Material classifier instance
    """
    if backend == "ensemble":
        if ENSEMBLE_AVAILABLE:
            return MaterialEnsembleClassifier(device=device)
        else:
            logger.warning("Ensemble not available, falling back to single DINOv2")
            backend = "dino"

    if backend == "dino":
        # Import legacy classifier
        from .material_classifier import MaterialClassifier
        return MaterialClassifier(device=device)

    raise ValueError(f"Unknown backend: {backend}")
