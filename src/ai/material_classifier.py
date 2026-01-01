"""
Facade material classification using AI.

Uses DINOv2 features with reference embeddings for zero-shot classification.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from rich.console import Console

from ..core.models import FacadeMaterial

console = Console()


@dataclass
class MaterialPrediction:
    """Material classification prediction."""

    material: FacadeMaterial
    confidence: float
    all_scores: dict[str, float]


class MaterialClassifier:
    """
    Classify facade materials from images.

    Uses DINOv2 for feature extraction with reference embeddings.
    """

    # Reference descriptions for each material (for CLIP-based approach)
    MATERIAL_DESCRIPTIONS = {
        FacadeMaterial.BRICK: [
            "red brick wall",
            "brown brick facade",
            "brick building exterior",
            "masonry brick pattern",
        ],
        FacadeMaterial.CONCRETE: [
            "concrete wall",
            "gray concrete facade",
            "exposed concrete building",
            "brutalist concrete",
        ],
        FacadeMaterial.PLASTER: [
            "white plaster wall",
            "stucco facade",
            "rendered wall",
            "plastered building exterior",
        ],
        FacadeMaterial.GLASS: [
            "glass curtain wall",
            "glass facade",
            "reflective glass building",
            "modern glass exterior",
        ],
        FacadeMaterial.METAL: [
            "metal panel facade",
            "aluminum cladding",
            "steel building exterior",
            "corrugated metal wall",
        ],
        FacadeMaterial.WOOD: [
            "wooden facade",
            "timber cladding",
            "wood panel exterior",
            "wooden building wall",
        ],
        FacadeMaterial.STONE: [
            "stone facade",
            "granite building exterior",
            "limestone wall",
            "natural stone cladding",
        ],
    }

    def __init__(
        self,
        device: str = "cpu",
        model_name: str = "facebook/dinov2-base",
    ):
        """
        Initialize material classifier.

        Args:
            device: Device for inference
            model_name: HuggingFace model name for DINOv2
        """
        self.device = device
        self.model_name = model_name
        self._model = None
        self._processor = None
        self._reference_features = None
        self._initialized = False

    def _lazy_init(self) -> None:
        """Lazily initialize the model."""
        if self._initialized:
            return

        console.print("[cyan]Initializing material classifier...[/cyan]")

        try:
            from transformers import AutoImageProcessor, AutoModel
            import torch

            self._processor = AutoImageProcessor.from_pretrained(self.model_name)
            self._model = AutoModel.from_pretrained(self.model_name)
            self._model.to(self.device)
            self._model.eval()

            console.print("[green]DINOv2 model loaded[/green]")

            # Initialize reference features (would need reference images)
            self._reference_features = {}

        except ImportError:
            console.print(
                "[yellow]transformers not installed. Install with: pip install transformers torch[/yellow]"
            )
            self._model = None

        self._initialized = True

    def extract_features(
        self,
        image: Image.Image | np.ndarray | str | Path,
        return_spatial: bool = False,
    ) -> np.ndarray | dict | None:
        """
        Extract DINOv2 features from an image.

        Args:
            image: Input image
            return_spatial: If True, return dict with CLS + patch features

        Returns:
            Feature vector, dict with spatial features, or None if unavailable
        """
        self._lazy_init()

        if self._model is None:
            return None

        # Load image
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        try:
            import torch

            inputs = self._processor(images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self._model(**inputs)

            # Get all hidden states
            all_features = outputs.last_hidden_state.cpu().numpy()[0]

            if return_spatial:
                # CLS token is first, patch tokens are rest
                cls_features = all_features[0]
                patch_features = all_features[1:]

                # DINOv2 uses 14x14 patches for 224x224 input (16x16 patch size)
                # Calculate grid size
                num_patches = patch_features.shape[0]
                grid_size = int(np.sqrt(num_patches))

                return {
                    "cls": cls_features,
                    "patches": patch_features,
                    "grid_size": grid_size,
                    "feature_dim": cls_features.shape[0],
                }

            # Just return CLS token
            return all_features[0]

        except Exception as e:
            console.print(f"[red]Feature extraction failed: {e}[/red]")
            return None

    def segment_materials(
        self,
        image: Image.Image | np.ndarray | str | Path,
        n_clusters: int = 3,
    ) -> dict[str, Any]:
        """
        Segment image into material regions using spatial clustering.

        Uses DINOv2 patch features + K-means to find distinct material regions.

        Args:
            image: Facade image
            n_clusters: Number of material clusters to find

        Returns:
            Dict with material masks, cluster labels, and classifications
        """
        from sklearn.cluster import KMeans

        # Load image
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        # Extract spatial features
        features = self.extract_features(image, return_spatial=True)
        if features is None:
            return {"error": "Feature extraction failed"}

        patch_features = features["patches"]
        grid_size = features["grid_size"]

        # Cluster patch features
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(patch_features)

        # Reshape to spatial grid
        cluster_grid = cluster_labels.reshape(grid_size, grid_size)

        # Upscale cluster grid to image size
        img_array = np.array(image)
        h, w = img_array.shape[:2]

        # Create masks for each cluster
        masks = {}
        cluster_classifications = {}

        for cluster_id in range(n_clusters):
            # Create mask at grid resolution
            grid_mask = (cluster_grid == cluster_id).astype(np.uint8)

            # Upscale to image size
            from PIL import Image as PILImage
            mask_pil = PILImage.fromarray(grid_mask * 255)
            mask_upscaled = mask_pil.resize((w, h), PILImage.Resampling.NEAREST)
            masks[f"cluster_{cluster_id}"] = np.array(mask_upscaled) > 127

            # Classify this cluster region
            # Get average patch features for this cluster
            cluster_patch_indices = np.where(cluster_labels == cluster_id)[0]
            if len(cluster_patch_indices) > 0:
                cluster_avg_features = patch_features[cluster_patch_indices].mean(axis=0)

                # Compare with reference features if available
                if self._reference_features:
                    best_material = None
                    best_score = -1
                    for material, ref_feat in self._reference_features.items():
                        sim = np.dot(cluster_avg_features, ref_feat) / (
                            np.linalg.norm(cluster_avg_features) * np.linalg.norm(ref_feat) + 1e-8
                        )
                        if sim > best_score:
                            best_score = sim
                            best_material = material

                    if best_material:
                        cluster_classifications[f"cluster_{cluster_id}"] = {
                            "material": best_material.value,
                            "confidence": float(best_score),
                        }

                # Extract region colors for heuristic backup
                region_mask = masks[f"cluster_{cluster_id}"]
                region_pixels = img_array[region_mask]
                if len(region_pixels) > 0:
                    avg_color = region_pixels.mean(axis=0)
                    cluster_classifications.setdefault(f"cluster_{cluster_id}", {})
                    cluster_classifications[f"cluster_{cluster_id}"]["avg_color_rgb"] = avg_color.tolist()
                    cluster_classifications[f"cluster_{cluster_id}"]["area_pct"] = region_mask.sum() / (h * w) * 100

        return {
            "masks": masks,
            "cluster_grid": cluster_grid,
            "classifications": cluster_classifications,
            "n_clusters": n_clusters,
            "grid_size": grid_size,
        }

    def classify(
        self,
        image: Image.Image | np.ndarray | str | Path,
    ) -> MaterialPrediction:
        """
        Classify facade material from image.

        Args:
            image: Facade image

        Returns:
            MaterialPrediction with material and confidence
        """
        self._lazy_init()

        # Extract features
        features = self.extract_features(image)

        if features is None:
            # Fallback to heuristic
            return self._heuristic_classify(image)

        # Compare with reference features
        if self._reference_features:
            scores = {}
            for material, ref_feat in self._reference_features.items():
                # Cosine similarity
                similarity = np.dot(features, ref_feat) / (
                    np.linalg.norm(features) * np.linalg.norm(ref_feat)
                )
                scores[material.value] = float(similarity)

            best_material = max(scores, key=scores.get)
            best_score = scores[best_material]

            return MaterialPrediction(
                material=FacadeMaterial(best_material),
                confidence=best_score,
                all_scores=scores,
            )

        # Without reference features, use color-based heuristic
        return self._heuristic_classify(image)

    def _heuristic_classify(
        self,
        image: Image.Image | np.ndarray | str | Path,
    ) -> MaterialPrediction:
        """
        Improved color and texture-based heuristic classification.

        Uses HSV color space and texture analysis for better accuracy.
        """
        import cv2

        # Load image
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        img_array = np.array(image)
        h, w = img_array.shape[:2]

        # Crop to middle region (more likely to be facade)
        crop_h = int(h * 0.6)
        crop_w = int(w * 0.8)
        start_h = int(h * 0.2)
        start_w = int(w * 0.1)
        facade_region = img_array[start_h:start_h+crop_h, start_w:start_w+crop_w]

        # Convert to HSV for better color analysis
        hsv = cv2.cvtColor(facade_region, cv2.COLOR_RGB2HSV)
        h_channel = hsv[:, :, 0]
        s_channel = hsv[:, :, 1]
        v_channel = hsv[:, :, 2]

        # Get color statistics
        avg_hue = np.mean(h_channel)
        avg_sat = np.mean(s_channel)
        avg_val = np.mean(v_channel)
        std_hue = np.std(h_channel)

        # RGB stats too
        r, g, b = facade_region.mean(axis=(0, 1))
        gray_diff = max(abs(r - g), abs(g - b), abs(r - b))

        # Texture analysis - edge density (high for brick, low for smooth plaster)
        gray = cv2.cvtColor(facade_region, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = edges.sum() / (edges.shape[0] * edges.shape[1] * 255)

        # Initialize scores
        scores = {m.value: 0.0 for m in FacadeMaterial if m != FacadeMaterial.UNKNOWN}

        # === BRICK DETECTION ===
        # Brick: red/orange hue (0-20 or 160-180), medium-high saturation, textured
        is_reddish = (avg_hue < 20 or avg_hue > 160) and avg_sat > 50
        brick_score = 0.0
        if is_reddish:
            brick_score += 0.4
        if edge_density > 0.1:  # Textured surface
            brick_score += 0.2
        if 100 < r < 200 and r > g and r > b:  # Red tones in RGB
            brick_score += 0.2
        scores[FacadeMaterial.BRICK.value] = min(0.8, brick_score)

        # === CONCRETE DETECTION ===
        # Concrete: gray (low saturation), medium-low value
        concrete_score = 0.0
        if avg_sat < 40 and gray_diff < 25:  # Gray (neutral)
            concrete_score += 0.4
        if 80 < avg_val < 180:  # Medium brightness
            concrete_score += 0.2
        if edge_density < 0.15:  # Relatively smooth
            concrete_score += 0.1
        scores[FacadeMaterial.CONCRETE.value] = min(0.7, concrete_score)

        # === PLASTER/RENDER DETECTION ===
        # Swedish plaster/render: can be white, cream, OR colored (orange, yellow, pink)
        # Key features: relatively smooth surface, uniform color
        plaster_score = 0.0

        # Swedish buildings often have colored render (orange, yellow, pink, green)
        # These have hue 10-60 (warm) or 70-160 (cool pastels) with medium brightness
        is_colored_render = (
            (15 < avg_hue < 60 or 70 < avg_hue < 160)  # Warm or cool colors
            and 20 < avg_sat < 80  # Medium saturation (not gray, not vivid)
            and 120 < avg_val < 230  # Medium to bright
        )

        # White/cream render: low saturation, high brightness
        is_white_render = avg_sat < 30 and avg_val > 180

        if is_colored_render:
            plaster_score += 0.5  # Strong signal for colored Swedish render
        elif is_white_render:
            plaster_score += 0.4

        # Medium brightness (not too dark = concrete, not white only)
        if 130 < avg_val < 230:
            plaster_score += 0.15

        # Relatively smooth surface (but allow some texture from windows/balconies)
        if edge_density < 0.15:  # Smooth to medium texture
            plaster_score += 0.15

        # Uniform color (low hue variation = consistent render color)
        if std_hue < 25:  # Uniform hue across facade
            plaster_score += 0.1

        scores[FacadeMaterial.PLASTER.value] = min(0.85, plaster_score)

        # === GLASS DETECTION ===
        # Glass: often reflects sky (blue), or dark, shiny
        glass_score = 0.0
        if 90 < avg_hue < 130 and avg_sat > 30:  # Blue hue
            glass_score += 0.3
        if avg_val < 100:  # Dark (tinted glass)
            glass_score += 0.2
        if std_hue > 30:  # High hue variance (reflections)
            glass_score += 0.2
        scores[FacadeMaterial.GLASS.value] = min(0.7, glass_score)

        # === METAL DETECTION ===
        # Metal: gray/silver, may have shine (high local contrast)
        metal_score = 0.0
        if avg_sat < 30 and 140 < avg_val < 220:  # Silvery
            metal_score += 0.3
        if gray_diff < 20 and avg_val > 150:  # Bright gray
            metal_score += 0.2
        scores[FacadeMaterial.METAL.value] = min(0.6, metal_score)

        # === WOOD DETECTION ===
        # Wood: warm brown tones, some texture
        wood_score = 0.0
        if 10 < avg_hue < 40 and avg_sat > 40:  # Brown/tan hue
            wood_score += 0.4
        if r > b and g > b * 0.8 and 100 < r < 200:  # Brown in RGB
            wood_score += 0.3
        if edge_density > 0.05:  # Some texture
            wood_score += 0.1
        scores[FacadeMaterial.WOOD.value] = min(0.7, wood_score)

        # === STONE DETECTION ===
        # Stone: gray or cool beige, textured, NOT warm orange/yellow
        # Swedish natural stone (granite, limestone): gray or cool beige
        # Exclude: warm render colors (orange, yellow) which are modern plaster
        stone_score = 0.0

        # Stone is typically gray (very low sat) or cool beige (hue 0-15, low sat)
        # EXCLUDE warm tones (hue 20-60) which are colored Swedish render
        is_gray_stone = avg_sat < 25 and 100 < avg_val < 200  # Gray stone
        is_cool_beige = avg_hue < 15 and avg_sat < 40 and avg_val < 180  # Limestone/sandstone

        if is_gray_stone or is_cool_beige:
            stone_score += 0.35
        # Penalize warm colors that indicate render, not stone
        if 20 < avg_hue < 60 and avg_sat > 20:  # Warm colors = render
            stone_score -= 0.3

        if edge_density > 0.12:  # Stone has clear texture from joints
            stone_score += 0.15
        if std_hue > 20 and avg_sat < 30:  # Gray variation (stone joints)
            stone_score += 0.15

        scores[FacadeMaterial.STONE.value] = max(0.0, min(0.7, stone_score))

        # Find best match
        best_material = max(scores, key=scores.get)
        best_score = scores[best_material]

        if best_score < 0.25:
            best_material = FacadeMaterial.UNKNOWN.value
            best_score = 0.25

        return MaterialPrediction(
            material=FacadeMaterial(best_material),
            confidence=best_score,
            all_scores=scores,
        )

    def load_reference_features(
        self,
        reference_dir: str | Path,
    ) -> None:
        """
        Load reference images for each material class.

        Expected structure:
        reference_dir/
          brick/
            img1.jpg
            img2.jpg
          concrete/
            img1.jpg
          ...
        """
        self._lazy_init()

        if self._model is None:
            console.print("[yellow]Cannot load references without model[/yellow]")
            return

        reference_dir = Path(reference_dir)
        if not reference_dir.exists():
            console.print(f"[yellow]Reference directory not found: {reference_dir}[/yellow]")
            return

        self._reference_features = {}

        for material in FacadeMaterial:
            if material == FacadeMaterial.UNKNOWN:
                continue

            material_dir = reference_dir / material.value
            if not material_dir.exists():
                continue

            features_list = []
            for img_path in material_dir.glob("*.jpg"):
                feat = self.extract_features(img_path)
                if feat is not None:
                    features_list.append(feat)

            for img_path in material_dir.glob("*.png"):
                feat = self.extract_features(img_path)
                if feat is not None:
                    features_list.append(feat)

            if features_list:
                # Average features for this material
                avg_features = np.mean(features_list, axis=0)
                self._reference_features[material] = avg_features
                console.print(f"[green]Loaded {len(features_list)} references for {material.value}[/green]")


def estimate_material_from_era(construction_year: int, location: str = "Stockholm") -> FacadeMaterial:
    """
    Estimate facade material based on construction era.

    Swedish building practices by era:
    - Pre-1930: Brick, stone, plaster
    - 1930-1960: Brick, plaster
    - 1960-1975 (Million program): Concrete, brick
    - 1975-1990: Concrete, brick, some metal
    - 1990-2010: Mixed - plaster/render common in modern developments
    - Post-2010: Glass, metal panels, mixed

    Hammarby Sjöstad specifics:
    - Built 1999-2017 as eco-district
    - Primarily rendered/plastered facades in various colors
    - Some brick accents, metal/glass balconies
    """
    if construction_year < 1930:
        return FacadeMaterial.BRICK
    elif construction_year < 1960:
        return FacadeMaterial.BRICK
    elif construction_year < 1975:
        return FacadeMaterial.CONCRETE
    elif construction_year < 1990:
        return FacadeMaterial.CONCRETE
    elif construction_year < 2010:
        # Modern Swedish residential - typically rendered/plastered
        # Hammarby Sjöstad (1999-2017) - eco-district with plaster facades
        if "sjöstad" in location.lower() or "hammarby" in location.lower():
            return FacadeMaterial.PLASTER  # Rendered facades, not brick
        return FacadeMaterial.PLASTER
    else:
        return FacadeMaterial.GLASS
