"""
Material Classifier V2 - Strong Vision Approach

Uses CLIP for zero-shot classification with:
- Multi-image consensus voting
- SAM-based wall segmentation (exclude windows/balconies)
- Swedish-specific material prompts
- Confidence-weighted voting across all facade images
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from PIL import Image
from rich.console import Console
from collections import defaultdict

console = Console()


@dataclass
class MaterialVote:
    """Single material classification vote."""
    material: str
    confidence: float
    image_index: int


@dataclass
class MaterialConsensus:
    """Consensus result from multiple images."""
    material: str
    confidence: float
    vote_count: int
    total_images: int
    vote_distribution: Dict[str, float]


class MaterialClassifierV2:
    """
    Strong vision-based material classifier.

    Uses CLIP + SAM for accurate Swedish building material detection.
    """

    # Swedish-specific material prompts (more specific = better CLIP matching)
    # Focus on PRIMARY WALL SURFACE, not accessories like railings
    MATERIAL_PROMPTS = {
        "render": [
            "smooth plastered building wall",
            "white rendered exterior wall surface",
            "stucco wall texture",
            "painted plaster wall close up",
            "smooth cement wall finish",
            "light colored plaster facade wall",
            "off-white rendered wall",
        ],
        "brick": [
            "red brick wall pattern",
            "exposed brick masonry wall",
            "brown brick wall texture",
            "yellow brick wall surface",
            "brick mortar pattern wall",
            "clay brick exterior wall",
        ],
        "concrete": [
            "exposed concrete wall surface",
            "gray concrete panel texture",
            "brutalist concrete wall",
            "precast concrete wall panel",
            "raw concrete wall finish",
            "cast concrete wall texture",
        ],
        "wood": [
            "wooden cladding wall surface",
            "timber wall panel texture",
            "wooden board wall siding",
            "natural wood wall grain",
            "painted wood wall panels",
            "wood plank wall exterior",
        ],
        "metal": [
            # NOTE: These must describe WALL CLADDING, not railings/accessories
            "metal sheet wall cladding",
            "corrugated metal wall surface",
            "aluminum wall panel facade",
            "steel sheet wall covering",
            "zinc wall cladding panels",
            # Explicitly not matching: railings, balconies, window frames
        ],
        "glass": [
            "glass curtain wall building",
            "fully glazed building facade",
            "reflective glass wall panels",
            "floor to ceiling glass wall",
        ],
    }

    # Negative prompts to help CLIP distinguish wall vs accessories
    NEGATIVE_PROMPTS = [
        "metal balcony railing",
        "window frame",
        "door handle",
        "air conditioning unit",
        "satellite dish",
    ]

    def __init__(self, device: str = "cpu"):
        self.device = device
        self._clip_model = None
        self._clip_processor = None
        self._sam_model = None
        self._initialized = False

    def _lazy_init(self):
        """Initialize CLIP model on first use."""
        if self._initialized:
            return

        console.print("[cyan]Initializing CLIP for material classification...[/cyan]")

        try:
            from transformers import CLIPProcessor, CLIPModel
            import torch

            # Use CLIP ViT-L/14 for best quality
            model_name = "openai/clip-vit-large-patch14"

            self._clip_processor = CLIPProcessor.from_pretrained(model_name)
            self._clip_model = CLIPModel.from_pretrained(model_name)

            # Move to device
            if self.device == "cuda" and torch.cuda.is_available():
                self._clip_model = self._clip_model.to("cuda")
            elif self.device == "mps" and torch.backends.mps.is_available():
                self._clip_model = self._clip_model.to("mps")
            else:
                self._clip_model = self._clip_model.to("cpu")

            self._clip_model.eval()
            console.print("[green]CLIP ViT-L/14 loaded[/green]")
            self._initialized = True

        except ImportError:
            console.print("[yellow]transformers not installed for CLIP[/yellow]")
            self._clip_model = None
        except Exception as e:
            console.print(f"[red]CLIP init failed: {e}[/red]")
            self._clip_model = None

    def classify_single(
        self,
        image: Image.Image | np.ndarray | str | Path,
        wall_mask: np.ndarray = None,
        use_patch_sampling: bool = True,
        num_patches: int = 5,
        patch_size: int = 224,
    ) -> Tuple[str, float, Dict[str, float]]:
        """
        Classify material from a single image.

        Args:
            image: Facade image
            wall_mask: Optional mask to isolate wall area
            use_patch_sampling: If True, sample patches from wall regions
            num_patches: Number of patches to sample
            patch_size: Size of each patch (CLIP uses 224x224)

        Returns:
            (material, confidence, all_scores)
        """
        self._lazy_init()

        if self._clip_model is None:
            return "unknown", 0.0, {}

        # Load image
        if isinstance(image, (str, Path)):
            pil_image = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            pil_image = Image.fromarray(image)
        else:
            pil_image = image

        img_array = np.array(pil_image)

        import torch

        # Build all text prompts
        all_prompts = []
        prompt_to_material = {}
        for material, prompts in self.MATERIAL_PROMPTS.items():
            for prompt in prompts:
                all_prompts.append(prompt)
                prompt_to_material[prompt] = material

        # Collect images to classify (patches or full masked image)
        images_to_classify = []

        if use_patch_sampling and wall_mask is not None:
            # Sample patches from wall regions
            patches = self._sample_wall_patches(
                img_array, wall_mask, num_patches, patch_size
            )
            if patches:
                images_to_classify = patches

        if not images_to_classify:
            # Fallback: use masked region or full image
            if wall_mask is not None:
                masked_image = self._apply_mask(pil_image, wall_mask)
                images_to_classify = [masked_image]
            else:
                images_to_classify = [pil_image]

        # Classify each image/patch and aggregate
        all_material_scores = defaultdict(float)
        patch_count = 0

        for img in images_to_classify:
            # Process with CLIP
            inputs = self._clip_processor(
                text=all_prompts,
                images=img,
                return_tensors="pt",
                padding=True,
            )

            # Move to device
            device = next(self._clip_model.parameters()).device
            inputs = {k: v.to(device) if hasattr(v, 'to') else v for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self._clip_model(**inputs)

            # Get similarity scores
            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=1).cpu().numpy()[0]

            # Aggregate scores by material
            for i, prompt in enumerate(all_prompts):
                material = prompt_to_material[prompt]
                all_material_scores[material] += probs[i]

            patch_count += 1

        # Normalize
        total = sum(all_material_scores.values())
        if total > 0:
            material_scores = {k: v/total for k, v in all_material_scores.items()}
        else:
            material_scores = dict(all_material_scores)

        # Get best material
        best_material = max(material_scores, key=material_scores.get)
        best_score = material_scores[best_material]

        return best_material, float(best_score), dict(material_scores)

    def _sample_wall_patches(
        self,
        img_array: np.ndarray,
        wall_mask: np.ndarray,
        num_patches: int,
        patch_size: int,
    ) -> List[Image.Image]:
        """
        Sample patches from pure wall regions.

        Finds regions with high wall mask density and extracts patches.
        """
        h, w = img_array.shape[:2]
        patches = []

        # Need enough space for patches
        if h < patch_size or w < patch_size:
            return []

        # Find valid patch locations (where mask is mostly wall)
        valid_locations = []

        # Stride through image
        stride = patch_size // 2
        for y in range(0, h - patch_size, stride):
            for x in range(0, w - patch_size, stride):
                # Check mask density in this patch
                patch_mask = wall_mask[y:y+patch_size, x:x+patch_size]
                density = np.mean(patch_mask > 128)

                if density > 0.85:  # At least 85% wall (more strict)
                    valid_locations.append((x, y, density))

        if not valid_locations:
            return []

        # Sort by density (most wall-like first)
        valid_locations.sort(key=lambda x: x[2], reverse=True)

        # Take top patches, ensuring some spatial diversity
        selected = []
        min_distance = patch_size * 0.5  # Don't sample overlapping patches

        for x, y, density in valid_locations:
            # Check distance to already selected patches
            too_close = False
            for sx, sy, _ in selected:
                dist = ((x - sx) ** 2 + (y - sy) ** 2) ** 0.5
                if dist < min_distance:
                    too_close = True
                    break

            if not too_close:
                selected.append((x, y, density))
                if len(selected) >= num_patches:
                    break

        # Extract patches
        for x, y, _ in selected:
            patch = img_array[y:y+patch_size, x:x+patch_size]
            patches.append(Image.fromarray(patch))

        return patches

    def classify_multi_image(
        self,
        images: List[Image.Image | np.ndarray | str | Path],
        use_sam_crop: bool = True,
        building_type: str = "residential",
    ) -> MaterialConsensus:
        """
        Classify material using consensus from multiple images.

        This is the recommended method - uses all available facade images
        and votes for the most likely material.

        Args:
            images: List of facade images
            use_sam_crop: If True, use SAM to isolate wall regions
            building_type: "residential" or "commercial" - affects valid materials

        Returns:
            MaterialConsensus with final result
        """
        if not images:
            return MaterialConsensus(
                material="unknown",
                confidence=0.0,
                vote_count=0,
                total_images=0,
                vote_distribution={},
            )

        console.print(f"[dim]Classifying material from {len(images)} images...[/dim]")

        votes: List[MaterialVote] = []

        # OPTIMIZATION: Only run SAM wall isolation on first N images
        # (SAM is slow - ~2-3s per image with all prompts)
        max_sam_images = 6  # Limit expensive SAM processing

        for i, img in enumerate(images):
            try:
                # Optionally crop to wall region using SAM (only first N images)
                wall_mask = None
                if use_sam_crop and i < max_sam_images:
                    wall_mask = self._get_wall_mask(img)

                # Skip patch sampling if no wall mask (use full image, lower weight)
                use_patches = wall_mask is not None
                material, conf, scores = self.classify_single(
                    img, wall_mask, use_patch_sampling=use_patches
                )

                # Lower confidence for images without wall isolation
                if wall_mask is None:
                    conf = conf * 0.7

                if conf > 0.1:  # Only count confident votes
                    votes.append(MaterialVote(
                        material=material,
                        confidence=conf,
                        image_index=i,
                    ))

            except Exception as e:
                console.print(f"[dim]Image {i} failed: {e}[/dim]")
                continue

        if not votes:
            return MaterialConsensus(
                material="unknown",
                confidence=0.0,
                vote_count=0,
                total_images=len(images),
                vote_distribution={},
            )

        # Filter invalid materials based on building type
        # For residential buildings, glass curtain walls are not valid wall materials
        # (glass is for windows, not walls in multi-family buildings)
        invalid_materials = set()
        if building_type == "residential":
            invalid_materials = {"glass"}
            console.print(f"[dim]Filtering out glass votes (residential building)[/dim]")

        # Weighted voting
        material_weights = defaultdict(float)
        material_counts = defaultdict(int)

        for vote in votes:
            if vote.material in invalid_materials:
                # Redistribute vote to second-best material
                # For now, just skip these votes
                continue
            material_weights[vote.material] += vote.confidence
            material_counts[vote.material] += 1

        if not material_weights:
            # All votes were filtered out - use fallback
            console.print("[yellow]All votes filtered, using fallback[/yellow]")
            return MaterialConsensus(
                material="render",  # Most common for modern Swedish buildings
                confidence=0.5,
                vote_count=0,
                total_images=len(images),
                vote_distribution={"render": 1.0},
            )

        # Normalize weights
        total_weight = sum(material_weights.values())
        vote_distribution = {k: v/total_weight for k, v in material_weights.items()}

        # Winner
        winner = max(material_weights, key=material_weights.get)
        winner_weight = material_weights[winner]
        winner_count = material_counts[winner]

        # Confidence based on vote concentration
        confidence = vote_distribution[winner]

        # Boost confidence if majority agrees
        if winner_count > len(votes) * 0.5:
            confidence = min(1.0, confidence * 1.2)

        console.print(f"[green]Material consensus: {winner} ({confidence:.0%}, {winner_count}/{len(votes)} votes)[/green]")

        return MaterialConsensus(
            material=winner,
            confidence=float(confidence),
            vote_count=winner_count,
            total_images=len(images),
            vote_distribution=vote_distribution,
        )

    def _get_wall_mask(self, image: Image.Image | np.ndarray | str | Path) -> Optional[np.ndarray]:
        """
        Use SAM to segment wall regions by EXCLUDING non-wall elements.

        Strategy: Detect windows, balconies, railings, then subtract from building.
        This is more robust than trying to detect "wall" directly.

        Returns mask where wall = 255, other = 0
        """
        try:
            from lang_sam import LangSAM

            if not hasattr(self, '_lang_sam') or self._lang_sam is None:
                self._lang_sam = LangSAM()

            # Load image
            if isinstance(image, (str, Path)):
                pil_image = Image.open(image).convert("RGB")
            elif isinstance(image, np.ndarray):
                pil_image = Image.fromarray(image)
            else:
                pil_image = image

            h, w = np.array(pil_image).shape[:2]

            # STEP 1: First detect the building facade as the base mask
            # This constrains us to the building, excluding sky/trees/ground
            wall_mask = np.zeros((h, w), dtype=np.float32)

            try:
                building_prompts = ["building facade", "apartment building wall", "building exterior"]
                for bp in building_prompts:
                    results = self._lang_sam.predict(
                        images_pil=[pil_image],
                        texts_prompt=[bp],
                        box_threshold=0.15,
                        text_threshold=0.15,
                    )

                    if results and len(results) > 0:
                        result = results[0]
                        masks = result.get("masks", [])

                        for mask in masks:
                            if hasattr(mask, 'cpu'):
                                mask_np = mask.cpu().numpy()
                            else:
                                mask_np = np.array(mask)
                            if mask_np.ndim == 3:
                                mask_np = mask_np[0]

                            # Add building areas to wall mask
                            wall_mask = np.maximum(wall_mask, (mask_np > 0.5).astype(np.float32))

            except Exception as e:
                console.print(f"[dim]Building detection failed: {e}[/dim]")
                # Fallback: use center region of image
                wall_mask[h//4:3*h//4, w//4:3*w//4] = 1.0

            building_coverage = np.mean(wall_mask > 0.5)
            if building_coverage < 0.1:
                console.print(f"[dim]Building mask too small ({building_coverage:.1%}), using center fallback[/dim]")
                wall_mask = np.zeros((h, w), dtype=np.float32)
                wall_mask[h//4:3*h//4, w//4:3*w//4] = 1.0

            console.print(f"[dim]Building detected: {np.mean(wall_mask > 0.5):.1%} coverage[/dim]")

            # STEP 2: Now exclude windows/glass/balconies FROM the building mask
            # OPTIMIZED: Use fewer, more comprehensive prompts to reduce SAM calls
            # (Each SAM call takes ~0.5-1s, so 14 prompts = 7-14s per image)
            exclusion_prompts = [
                "window",           # Catches most window variants
                "balcony",          # Catches balconies including glass ones
                "door",             # Catches all door types
            ]

            for prompt in exclusion_prompts:
                try:
                    results = self._lang_sam.predict(
                        images_pil=[pil_image],
                        texts_prompt=[prompt],
                        box_threshold=0.20,  # Lower threshold to catch more
                        text_threshold=0.20,
                    )

                    if results and len(results) > 0:
                        result = results[0]
                        masks = result.get("masks", [])

                        for mask in masks:
                            if hasattr(mask, 'cpu'):
                                mask_np = mask.cpu().numpy()
                            else:
                                mask_np = np.array(mask)
                            if mask_np.ndim == 3:
                                mask_np = mask_np[0]

                            # Subtract this element from wall mask
                            # Use LARGER dilation to exclude edges and reflections
                            from scipy import ndimage
                            dilated = ndimage.binary_dilation(mask_np > 0.5, iterations=15)
                            wall_mask = wall_mask * (1 - dilated.astype(np.float32))

                except Exception:
                    continue

            # Convert to uint8
            final_mask = (wall_mask * 255).astype(np.uint8)

            # Check if we have enough wall area (at least 5% of image after aggressive exclusion)
            wall_ratio = np.sum(final_mask > 128) / (h * w)
            if wall_ratio < 0.05:
                console.print(f"[dim]Wall mask too small ({wall_ratio:.1%}), using fallback[/dim]")
                return None

            console.print(f"[dim]Wall mask: {wall_ratio:.1%} coverage after exclusions[/dim]")

            return final_mask

        except ImportError:
            return None
        except Exception as e:
            console.print(f"[dim]Wall segmentation failed: {e}[/dim]")
            return None

    def _apply_mask(self, image: Image.Image, mask: np.ndarray) -> Image.Image:
        """Apply mask to image, keeping only masked region."""
        img_array = np.array(image)

        # Expand mask to 3 channels
        if mask.ndim == 2:
            mask_3d = np.stack([mask, mask, mask], axis=-1)
        else:
            mask_3d = mask

        # Apply mask
        masked = img_array * (mask_3d > 0).astype(np.uint8)

        # Find bounding box of mask
        rows = np.any(mask > 0, axis=1)
        cols = np.any(mask > 0, axis=0)

        if not np.any(rows) or not np.any(cols):
            return image

        y_min, y_max = np.where(rows)[0][[0, -1]]
        x_min, x_max = np.where(cols)[0][[0, -1]]

        # Crop to mask region
        cropped = masked[y_min:y_max, x_min:x_max]

        return Image.fromarray(cropped)


def classify_building_material(
    images: List[Image.Image],
    device: str = "cpu",
) -> Tuple[str, float]:
    """
    Convenience function for material classification.

    Args:
        images: List of facade images
        device: Device for inference

    Returns:
        (material, confidence)
    """
    classifier = MaterialClassifierV2(device=device)
    result = classifier.classify_multi_image(images, use_sam_crop=False)
    return result.material, result.confidence
