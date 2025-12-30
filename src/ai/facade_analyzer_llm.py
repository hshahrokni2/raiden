"""
LLM-based Facade Analyzer using Komilion/OpenRouter.

All LLM calls route through Komilion for:
- Single API key management (KOMILION_API_KEY)
- Unified billing through OpenRouter
- Easy model switching via modes (frugal/balanced/premium)

Can detect:
- Facade material (brick, concrete, plaster/render, wood, etc.)
- Ground floor use (commercial, residential, garage, etc.)
- Window-to-wall ratio estimate
- Balcony type
- Renovation indicators
- Building form hints
- Floor count and height estimation
"""

from __future__ import annotations

import base64
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Optional, Any, Literal

from rich.console import Console

console = Console()


@dataclass
class FacadeAnalysis:
    """Complete facade analysis from LLM."""

    # Primary classifications
    facade_material: str  # brick, concrete, plaster, wood, glass, metal, stone
    material_confidence: float  # 0-1
    material_color: str  # e.g., "orange", "white", "gray", "red"

    # Ground floor
    ground_floor_use: str  # residential, commercial, garage, mixed, unknown
    ground_floor_confidence: float

    # Windows
    wwr_estimate: float  # 0-1 window-to-wall ratio
    window_type: str  # single, double, triple, unknown

    # Balconies
    balcony_type: str  # projecting, recessed, french, loggia, none

    # Building characteristics
    estimated_era: str  # e.g., "1960-1975", "2000-2010"
    building_form: str  # lamellhus, skivhus, punkthus, etc.

    # Floor count and height estimation (NEW - from street view)
    visible_floors: int = 0  # Floors visible above ground
    floor_count_confidence: float = 0.0  # 0-1 confidence
    has_visible_attic: bool = False  # Visible attic/mansard
    has_visible_basement: bool = False  # Visible basement windows/entrance
    estimated_floor_height_m: float = 2.8  # Estimated floor-to-floor height
    height_reference_used: str = "none"  # door, window, person, car, none
    # Geometric height estimation (for use with camera position)
    roof_position_pct: float = 0.0  # 0-1, where roof edge appears in image (0=bottom, 1=top)
    ground_position_pct: float = 0.0  # 0-1, where ground level is in image

    # Renovation indicators
    likely_renovated: bool = False
    renovation_signs: List[str] = field(default_factory=list)

    # Raw LLM response for debugging
    raw_response: Dict[str, Any] = field(default_factory=dict)

    # Direction this analysis is for
    direction: str = "unknown"  # N, S, E, W

    @property
    def estimated_height_m(self) -> float:
        """Estimate building height from floor count."""
        if self.visible_floors <= 0:
            return 0.0
        # Base height from visible floors
        height = self.visible_floors * self.estimated_floor_height_m
        # Add for attic if visible (typically 2-3m)
        if self.has_visible_attic:
            height += 2.5
        return height


@dataclass
class MultiFacadeAnalysis:
    """Aggregated analysis from multiple facade images."""

    # Consensus results
    facade_material: str
    material_confidence: float
    ground_floor_use: str
    wwr_average: float
    balcony_type: str
    estimated_era: str
    building_form: str
    likely_renovated: bool

    # Floor count and height estimation (aggregated from multiple views)
    visible_floors: int = 0  # Max visible floors across all views
    floor_count_confidence: float = 0.0
    has_attic: bool = False
    has_visible_attic: bool = False  # Alias for compatibility with FacadeAnalysis
    has_basement: bool = False
    has_visible_basement: bool = False  # Alias for compatibility
    estimated_height_m: float = 0.0  # Estimated building height
    roof_position_pct: float = 0.0  # 0-1, where roof appears in image (for geometric height)

    # Per-direction results
    per_direction: Dict[str, FacadeAnalysis] = field(default_factory=dict)

    # Vote distribution
    material_votes: Dict[str, int] = field(default_factory=dict)


SYSTEM_PROMPT = """You are an expert in Swedish building architecture and construction.
Analyze the facade image and provide structured information about the building.

Swedish building context:
- Pre-1930: Often brick (tegel) or stone with plaster/render (puts)
- 1930-1960: Brick apartments, functionalist style
- 1961-1975 (Miljonprogrammet): Concrete panels, prefab elements
- 1976-1990: Better insulated, varied materials
- 1990-2010: Modern render, often colored (orange, yellow, pink common)
- 2010+: Glass, metal cladding, mixed materials

Building forms:
- Lamellhus: Slab block, 3-4 stories, rectangular
- Skivhus: Large slab, 8+ stories, miljonprogrammet
- Punkthus: Point tower, compact footprint
- Stjärnhus: Star-shaped, 3+ wings
- Loftgångshus: Gallery access with external corridors

Ground floor types:
- Residential: Apartments at ground level
- Commercial: Shops, restaurants, offices
- Garage: Parking garage entrance
- Mixed: Commercial + residential entrance
- Utility: Storage, bicycle parking, laundry

FLOOR COUNTING (CRITICAL for energy modeling):
Count the number of floors (stories) visible above ground level.
- Count rows of windows as floors (each row = 1 floor typically)
- Ground floor counts as floor 1
- Note if attic/mansard roof visible (pitched roof with dormer windows)
- Note if basement visible (windows below ground level)

Use reference objects to estimate floor height:
- Standard door: ~2.1m tall
- Standard window: ~1.2m tall
- Standard floor-to-floor height: 2.7-3.0m (apartments), 3.5-4.5m (commercial ground floor)
- Person: ~1.7m average
- Car: ~1.5m tall

ROOF/GROUND POSITION (for geometric height calculation):
Estimate where in the image the building extends:
- roof_position_pct: Where the roof/top of building is (0.0=bottom, 1.0=top of image)
- ground_position_pct: Where ground level is (0.0=bottom, 1.0=top of image)
Example: If roof is at top quarter of image, roof_position_pct=0.75
Example: If ground is at bottom third of image, ground_position_pct=0.33

Respond ONLY with valid JSON matching this schema:
{
  "facade_material": "brick|concrete|plaster|wood|glass|metal|stone",
  "material_confidence": 0.0-1.0,
  "material_color": "color description",
  "ground_floor_use": "residential|commercial|garage|mixed|utility|unknown",
  "ground_floor_confidence": 0.0-1.0,
  "wwr_estimate": 0.0-1.0,
  "window_type": "single|double|triple|unknown",
  "balcony_type": "projecting|recessed|french|loggia|none",
  "estimated_era": "year range or period name",
  "building_form": "lamellhus|skivhus|punkthus|stjarnhus|loftgangshus|radhus|other",
  "visible_floors": 1-50,
  "floor_count_confidence": 0.0-1.0,
  "has_visible_attic": true|false,
  "has_visible_basement": true|false,
  "estimated_floor_height_m": 2.5-4.5,
  "height_reference_used": "door|window|person|car|balcony|none",
  "roof_position_pct": 0.0-1.0,
  "ground_position_pct": 0.0-1.0,
  "likely_renovated": true|false,
  "renovation_signs": ["list", "of", "indicators"],
  "reasoning": "brief explanation including floor count logic"
}"""


class FacadeAnalyzerLLM:
    """
    LLM-based facade analyzer using Komilion/OpenRouter.

    All LLM calls route through Komilion for:
    - Single API key management
    - Unified billing
    - Easy model switching via modes

    Modes:
    - frugal: Cheapest (auto-upgrades to balanced for vision)
    - balanced: Free vision-capable models (recommended)
    - premium: Best quality (expensive)

    Cost: ~$0.01-0.02 per building (multiple images).
    """

    def __init__(
        self,
        backend: Literal["komilion", "claude", "gemini", "openai", "auto"] = "auto",
        model: Optional[str] = None,
        komilion_mode: Literal["frugal", "balanced", "premium"] = "balanced",
    ):
        """
        Initialize the LLM facade analyzer.

        Args:
            backend: Ignored - all calls route through Komilion (kept for compatibility)
            model: Ignored - Komilion selects model based on mode
            komilion_mode: Komilion mode - frugal, balanced (default), premium
        """
        # Import unified client
        from .llm_client import LLMClient

        self.backend = "komilion"  # Always use Komilion
        self.model = model
        self.komilion_mode = komilion_mode
        self._client = LLMClient(mode=komilion_mode)
        self._initialized = False

    def _lazy_init(self) -> bool:
        """Check if client is ready."""
        if self._initialized:
            return True

        if not self._client.api_key:
            console.print("[yellow]KOMILION_API_KEY not set[/yellow]")
            return False

        self._initialized = True
        console.print(f"[green]Initialized Komilion ({self.komilion_mode} mode) for facade analysis[/green]")
        return True

    def _encode_image(self, image_path: Path | str) -> tuple[str, str]:
        """Encode image to base64 for API."""
        image_path = Path(image_path)

        # Determine MIME type
        suffix = image_path.suffix.lower()
        mime_types = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".webp": "image/webp",
        }
        mime_type = mime_types.get(suffix, "image/jpeg")

        # Read and encode
        with open(image_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode("utf-8")

        return image_data, mime_type

    def _analyze_single(self, image_path: Path, direction: str) -> Optional[FacadeAnalysis]:
        """Analyze single image using unified Komilion client."""
        prompt = f"{SYSTEM_PROMPT}\n\nAnalyze this {direction} facade image:"

        response = self._client.analyze_image(
            image_path=image_path,
            prompt=prompt,
        )

        if response:
            return self._parse_response(response.content, direction)
        return None

    def _parse_response(self, response_text: str, direction: str) -> Optional[FacadeAnalysis]:
        """Parse JSON response from any backend."""
        try:
            response_text = response_text.strip()

            # Handle markdown code blocks
            if response_text.startswith("```"):
                lines = response_text.split("\n")
                # Remove first line (```json) and last line (```)
                response_text = "\n".join(lines[1:-1] if lines[-1] == "```" else lines[1:])

            # Extract JSON from response (handle labels like "[N facade]: {...")
            first_brace = response_text.find('{')
            last_brace = response_text.rfind('}')
            if first_brace != -1 and last_brace != -1:
                response_text = response_text[first_brace:last_brace + 1]

            result = json.loads(response_text)

            return FacadeAnalysis(
                facade_material=result.get("facade_material", "unknown"),
                material_confidence=float(result.get("material_confidence", 0.5)),
                material_color=result.get("material_color", "unknown"),
                ground_floor_use=result.get("ground_floor_use", "unknown"),
                ground_floor_confidence=float(result.get("ground_floor_confidence", 0.5)),
                wwr_estimate=float(result.get("wwr_estimate", 0.2)),
                window_type=result.get("window_type", "unknown"),
                balcony_type=result.get("balcony_type", "none"),
                estimated_era=result.get("estimated_era", "unknown"),
                building_form=result.get("building_form", "other"),
                # Floor count and height estimation
                visible_floors=int(result.get("visible_floors", 0)),
                floor_count_confidence=float(result.get("floor_count_confidence", 0.0)),
                has_visible_attic=result.get("has_visible_attic", False),
                has_visible_basement=result.get("has_visible_basement", False),
                estimated_floor_height_m=float(result.get("estimated_floor_height_m", 2.8)),
                height_reference_used=result.get("height_reference_used", "none"),
                # Geometric height estimation (roof/ground position in image)
                roof_position_pct=float(result.get("roof_position_pct", 0.0)),
                ground_position_pct=float(result.get("ground_position_pct", 0.0)),
                # Renovation
                likely_renovated=result.get("likely_renovated", False),
                renovation_signs=result.get("renovation_signs", []),
                raw_response=result,
                direction=direction,
            )
        except json.JSONDecodeError as e:
            console.print(f"[yellow]Failed to parse response: {e}[/yellow]")
            console.print(f"[dim]Response: {response_text[:200]}...[/dim]")
            return None

    def analyze_single(
        self,
        image_path: Path | str,
        direction: str = "unknown",
    ) -> Optional[FacadeAnalysis]:
        """
        Analyze a single facade image.

        Args:
            image_path: Path to facade image
            direction: Facade direction (N, S, E, W)

        Returns:
            FacadeAnalysis or None if failed
        """
        if not self._lazy_init():
            return None

        image_path = Path(image_path)
        if not image_path.exists():
            console.print(f"[yellow]Image not found: {image_path}[/yellow]")
            return None

        try:
            return self._analyze_single(image_path, direction)
        except Exception as e:
            console.print(f"[red]Facade analysis failed: {e}[/red]")
            return None

    # Alias for backwards compatibility
    analyze = analyze_single

    def analyze_multiple(
        self,
        image_paths: Dict[str, List[Path | str]],
        max_images: int = 3,
    ) -> Optional[MultiFacadeAnalysis]:
        """
        Analyze multiple facade images in a SINGLE API call.

        Args:
            image_paths: Dict mapping direction (N/S/E/W) to list of image paths
            max_images: Max total images to send (default 3)

        Returns:
            MultiFacadeAnalysis with results from single LLM call
        """
        if not self._lazy_init():
            return None

        # Collect up to max_images, spreading across directions
        selected_images: List[tuple[str, Path]] = []
        directions = list(image_paths.keys())

        # Round-robin selection across directions
        idx = 0
        while len(selected_images) < max_images:
            added_any = False
            for direction in directions:
                if len(selected_images) >= max_images:
                    break
                paths = image_paths[direction]
                if idx < len(paths):
                    path = Path(paths[idx])
                    if path.exists():
                        selected_images.append((direction, path))
                        added_any = True
            if not added_any:
                break
            idx += 1

        if not selected_images:
            return None

        # Build single request with all images
        return self._analyze_batch(selected_images)

    def _analyze_batch(self, images: List[tuple[str, Path]]) -> Optional[MultiFacadeAnalysis]:
        """Send all images in a single API call using unified Komilion client."""
        directions_str = ", ".join([d for d, _ in images])
        prompt = f"""{SYSTEM_PROMPT}

These {len(images)} images show the same building from different angles ({directions_str}).
Analyze them together and provide a SINGLE combined JSON response for the building.
Do NOT provide separate responses per image - give ONE JSON object representing your overall assessment."""

        image_paths = [path for _, path in images]

        try:
            response = self._client.analyze_images(
                image_paths=image_paths,
                prompt=prompt,
                max_images=len(images),
            )

            if response:
                return self._parse_multi_response(response.content, images)
            return None
        except Exception as e:
            console.print(f"[red]Batch analysis failed: {e}[/red]")
            return None

    def _parse_multi_response(self, text: str, images: List) -> Optional[MultiFacadeAnalysis]:
        """Parse response from batch analysis."""
        analysis = self._parse_response(text, "batch")
        if not analysis:
            return None

        # Calculate estimated height from floor count
        estimated_height = analysis.estimated_height_m

        return MultiFacadeAnalysis(
            facade_material=analysis.facade_material,
            material_confidence=analysis.material_confidence,
            ground_floor_use=analysis.ground_floor_use,
            wwr_average=analysis.wwr_estimate,
            balcony_type=analysis.balcony_type,
            estimated_era=analysis.estimated_era,
            building_form=analysis.building_form,
            likely_renovated=analysis.likely_renovated,
            # Floor count and height estimation
            visible_floors=analysis.visible_floors,
            floor_count_confidence=analysis.floor_count_confidence,
            has_attic=analysis.has_visible_attic,
            has_visible_attic=analysis.has_visible_attic,  # Alias for compatibility
            has_basement=analysis.has_visible_basement,
            has_visible_basement=analysis.has_visible_basement,  # Alias for compatibility
            estimated_height_m=estimated_height,
            roof_position_pct=analysis.roof_position_pct,  # For geometric height estimation
            per_direction={},
            material_votes={analysis.facade_material: len(images)},
        )


# Convenience function
def analyze_facade_with_llm(
    image_paths: Dict[str, List[Path | str]],
    backend: Literal["komilion", "claude", "gemini", "openai", "auto"] = "auto",
    komilion_mode: Literal["frugal", "balanced", "premium"] = "balanced",
    max_images: int = 3,
) -> Optional[MultiFacadeAnalysis]:
    """
    Analyze building facades using Komilion/OpenRouter (single API call).

    All LLM calls route through Komilion for unified billing.

    Args:
        image_paths: Dict mapping direction (N/S/E/W) to list of image paths
        backend: Ignored - all calls route through Komilion (kept for compatibility)
        komilion_mode: Komilion mode - frugal, balanced (default), premium
        max_images: Max images to analyze (default 3, sent in 1 API call)

    Returns:
        MultiFacadeAnalysis with material, ground floor use, WWR, etc.

    Example:
        result = analyze_facade_with_llm({
            "N": ["facade_N_1.jpg"],
            "S": ["facade_S_1.jpg"],
            "E": ["facade_E_1.jpg"],
        })
        print(f"Material: {result.facade_material}")
    """
    analyzer = FacadeAnalyzerLLM(komilion_mode=komilion_mode)
    return analyzer.analyze_multiple(image_paths, max_images=max_images)
