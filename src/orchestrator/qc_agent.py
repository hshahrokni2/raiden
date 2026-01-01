"""
Agentic QC agents for building analysis.

Triggered when analysis confidence is low:
- ImageQCAgent: Re-analyze facade images with different parameters
- ECMRefinerAgent: Adjust ECM packages for building-specific context
- AnomalyAgent: Investigate unusual patterns
"""

import base64
import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image

logger = logging.getLogger(__name__)


class QCTriggerType(Enum):
    """Types of QC triggers."""

    LOW_WWR_CONFIDENCE = "low_wwr_confidence"
    LOW_MATERIAL_CONFIDENCE = "low_material_confidence"
    LOW_ARCHETYPE_CONFIDENCE = "low_archetype_confidence"
    NEGATIVE_SAVINGS = "negative_savings"
    ENERGY_CLASS_MISMATCH = "energy_class_mismatch"
    ANOMALOUS_PATTERN = "anomalous_pattern"
    MISSING_DATA = "missing_data"


@dataclass
class QCTrigger:
    """QC trigger with details."""

    trigger_type: str  # One of QCTriggerType values
    confidence: float = 0.0
    message: str = ""

    def get_trigger_enum(self) -> Optional[QCTriggerType]:
        """Convert trigger_type string to enum."""
        try:
            return QCTriggerType(self.trigger_type)
        except ValueError:
            return None


@dataclass
class QCResult:
    """Result of QC intervention."""

    trigger: QCTrigger
    success: bool
    action_taken: str

    # Updated values
    updated_values: Dict[str, Any] = field(default_factory=dict)

    # Confidence after QC
    new_confidence: float = 0.0

    # Explanation
    explanation: str = ""

    # Recommendations
    recommendations: List[str] = field(default_factory=list)

    # Flags
    needs_human_review: bool = False
    escalated: bool = False


class QCAgent(ABC):
    """Base class for QC agents."""

    @abstractmethod
    async def run(self, building_result: Any) -> QCResult:
        """Run QC on a building result."""
        pass

    @abstractmethod
    def can_handle(self, trigger: "QCTriggerType | QCTrigger | str") -> bool:
        """Check if agent can handle this trigger.

        Args:
            trigger: Either a QCTriggerType enum, QCTrigger dataclass, or trigger string
        """
        pass

    def _normalize_trigger(self, trigger: "QCTriggerType | QCTrigger | str") -> Optional[QCTriggerType]:
        """Convert trigger to QCTriggerType enum."""
        if isinstance(trigger, QCTriggerType):
            return trigger
        elif isinstance(trigger, QCTrigger):
            return trigger.get_trigger_enum()
        elif isinstance(trigger, str):
            try:
                return QCTriggerType(trigger)
            except ValueError:
                return None
        return None


class ImageQCAgent(QCAgent):
    """
    QC agent for facade image analysis.

    Re-analyzes images when WWR or material confidence is low.
    Strategies:
    1. Try different WWR detection backend (opencv → sam → lang_sam)
    2. Fetch additional images from multiple sources (Mapillary, Google Street View)
    3. Use Claude Vision for material classification and WWR estimation
    """

    # WWR backends to try in order
    WWR_BACKENDS = ["opencv", "sam", "lang_sam"]

    # Material classification threshold
    MATERIAL_CONFIDENCE_THRESHOLD = 0.7

    def __init__(
        self,
        max_retries: int = 3,
        google_api_key: str = None,
        mapillary_token: str = None,
        komilion_api_key: str = None,  # Renamed from anthropic_api_key
    ):
        self.max_retries = max_retries
        self.google_api_key = google_api_key or os.environ.get("GOOGLE_API_KEY")
        self.mapillary_token = mapillary_token or os.environ.get("MAPILLARY_TOKEN")
        self.komilion_api_key = komilion_api_key or os.environ.get("KOMILION_API_KEY")

        self._wwr_detector = None
        self._material_classifier = None
        self._cached_images: Dict[str, List[Image.Image]] = {}

    def can_handle(self, trigger: "QCTriggerType | QCTrigger | str") -> bool:
        normalized = self._normalize_trigger(trigger)
        return normalized in (
            QCTriggerType.LOW_WWR_CONFIDENCE,
            QCTriggerType.LOW_MATERIAL_CONFIDENCE,
        )

    async def run(self, building_result: Any) -> QCResult:
        """
        Re-analyze facade images with different strategies.

        Args:
            building_result: BuildingResult with low confidence

        Returns:
            QCResult with updated values
        """
        result = QCResult(
            trigger=QCTrigger(trigger_type="low_wwr_confidence"),
            success=False,
            action_taken="image_reanalysis",
        )

        address = building_result.address
        lat = getattr(building_result, "lat", None)
        lon = getattr(building_result, "lon", None)

        try:
            # Get building coordinates if not provided
            if lat is None or lon is None:
                lat, lon = await self._geocode_address(address)

            if lat is None:
                result.explanation = "Could not geocode address"
                result.needs_human_review = True
                return result

            # Strategy 1: Fetch images from multiple sources
            images = await self._fetch_multi_source_images(address, lat, lon)

            if not images:
                result.explanation = "No images available from any source"
                result.needs_human_review = True
                return result

            logger.info(f"ImageQCAgent: Fetched {len(images)} images for {address}")

            # Strategy 2: Try different WWR backends on the images
            wwr_result = await self._retry_wwr_detection_with_images(images)
            if wwr_result and wwr_result.get("confidence", 0) > 0.7:
                result.updated_values["wwr"] = wwr_result
                result.new_confidence = wwr_result["confidence"]
                result.success = True
                result.action_taken = f"wwr_backend_{wwr_result.get('backend', 'unknown')}"
                result.explanation = f"WWR detected using {wwr_result.get('backend')} backend"

            # Strategy 3: Material classification
            material_result = await self._classify_material(images)
            if material_result:
                result.updated_values["material"] = material_result.get("material")
                result.updated_values["material_confidence"] = material_result.get("confidence")
                if not result.success:
                    result.new_confidence = material_result.get("confidence", 0.6)
                    result.success = True
                    result.action_taken = "material_classification"

            # Strategy 4: Claude Vision fallback for low-confidence results
            if not result.success or result.new_confidence < 0.7:
                llm_result = await self._claude_vision_analysis(images, address)
                if llm_result:
                    result.updated_values.update(llm_result.get("values", {}))
                    result.new_confidence = max(result.new_confidence, llm_result.get("confidence", 0.6))
                    result.success = True
                    result.action_taken = "claude_vision_analysis"
                    result.explanation = llm_result.get("explanation", "")

            # If still unsuccessful
            if not result.success:
                result.needs_human_review = True
                result.explanation = "All automatic QC strategies failed"
                result.recommendations = [
                    "Manual image review recommended",
                    "Consider site visit for accurate assessment",
                ]

        except Exception as e:
            logger.error(f"ImageQCAgent failed: {e}")
            result.success = False
            result.explanation = str(e)

        return result

    async def _geocode_address(self, address: str) -> Tuple[Optional[float], Optional[float]]:
        """Geocode address to coordinates."""
        try:
            import requests

            # Use Nominatim for geocoding
            url = "https://nominatim.openstreetmap.org/search"
            params = {
                "q": address,
                "format": "json",
                "limit": 1,
                "countrycodes": "se",
            }
            headers = {"User-Agent": "Raiden/1.0 (building-energy-analysis)"}

            response = requests.get(url, params=params, headers=headers, timeout=10)
            if response.status_code == 200:
                results = response.json()
                if results:
                    return float(results[0]["lat"]), float(results[0]["lon"])

        except Exception as e:
            logger.debug(f"Geocoding failed: {e}")

        return None, None

    async def _fetch_multi_source_images(
        self,
        address: str,
        lat: float,
        lon: float,
    ) -> List[Dict[str, Any]]:
        """
        Fetch facade images from multiple sources.

        Sources (in order of preference):
        1. Google Street View (best quality, requires API key)
        2. Mapillary (crowdsourced, requires token)
        3. Cached images from previous fetches
        """
        images = []

        # Source 1: Google Street View
        if self.google_api_key:
            try:
                gsv_images = await self._fetch_google_streetview(lat, lon)
                images.extend(gsv_images)
                logger.debug(f"Fetched {len(gsv_images)} images from Google Street View")
            except Exception as e:
                logger.debug(f"Google Street View fetch failed: {e}")

        # Source 2: Mapillary
        if self.mapillary_token:
            try:
                mapillary_images = await self._fetch_mapillary(lat, lon)
                images.extend(mapillary_images)
                logger.debug(f"Fetched {len(mapillary_images)} images from Mapillary")
            except Exception as e:
                logger.debug(f"Mapillary fetch failed: {e}")

        # Source 3: Check cache
        cache_key = f"{lat:.4f},{lon:.4f}"
        if cache_key in self._cached_images:
            cached = self._cached_images[cache_key]
            logger.debug(f"Using {len(cached)} cached images")
            images.extend([{"image": img, "source": "cache"} for img in cached])

        return images

    async def _fetch_google_streetview(self, lat: float, lon: float) -> List[Dict[str, Any]]:
        """Fetch Street View images for cardinal directions."""
        import requests

        images = []
        headings = {"N": 0, "E": 90, "S": 180, "W": 270}

        for direction, heading in headings.items():
            try:
                url = "https://maps.googleapis.com/maps/api/streetview"
                params = {
                    "size": "640x480",
                    "location": f"{lat},{lon}",
                    "heading": heading,
                    "pitch": 10,
                    "fov": 90,
                    "key": self.google_api_key,
                }

                response = requests.get(url, params=params, timeout=15)
                if response.status_code == 200:
                    img = Image.open(BytesIO(response.content))
                    images.append({
                        "image": img,
                        "source": "google_streetview",
                        "direction": direction,
                        "heading": heading,
                    })

            except Exception as e:
                logger.debug(f"GSV {direction} fetch failed: {e}")

        return images

    async def _fetch_mapillary(self, lat: float, lon: float, radius_m: int = 50) -> List[Dict[str, Any]]:
        """Fetch Mapillary images near the location."""
        import requests

        images = []

        try:
            # Search for images near location
            url = "https://graph.mapillary.com/images"
            params = {
                "access_token": self.mapillary_token,
                "fields": "id,thumb_1024_url,compass_angle",
                "bbox": f"{lon-0.0005},{lat-0.0005},{lon+0.0005},{lat+0.0005}",
                "limit": 8,
            }

            response = requests.get(url, params=params, timeout=15)
            if response.status_code == 200:
                data = response.json()
                for item in data.get("data", [])[:4]:  # Limit to 4 images
                    img_url = item.get("thumb_1024_url")
                    if img_url:
                        img_response = requests.get(img_url, timeout=15)
                        if img_response.status_code == 200:
                            img = Image.open(BytesIO(img_response.content))
                            heading = item.get("compass_angle", 0)
                            direction = self._heading_to_direction(heading)
                            images.append({
                                "image": img,
                                "source": "mapillary",
                                "direction": direction,
                                "heading": heading,
                                "image_id": item.get("id"),
                            })

        except Exception as e:
            logger.debug(f"Mapillary fetch failed: {e}")

        return images

    def _heading_to_direction(self, heading: float) -> str:
        """Convert compass heading to cardinal direction."""
        if 315 <= heading or heading < 45:
            return "N"
        elif 45 <= heading < 135:
            return "E"
        elif 135 <= heading < 225:
            return "S"
        else:
            return "W"

    async def _retry_wwr_detection_with_images(
        self,
        images: List[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        """Try WWR detection with different backends on fetched images."""
        best_result = None
        best_confidence = 0.0

        for backend in self.WWR_BACKENDS:
            try:
                from src.ai.wwr_detector import WWRDetector
                detector = WWRDetector(backend=backend)

                for img_data in images:
                    img = img_data.get("image")
                    if img is None:
                        continue

                    try:
                        wwr_result = detector.calculate_wwr(img)
                        confidence = getattr(wwr_result, "confidence", 0.7)

                        if confidence > best_confidence:
                            best_confidence = confidence
                            best_result = {
                                "average": wwr_result.average,
                                "north": wwr_result.north,
                                "south": wwr_result.south,
                                "east": wwr_result.east,
                                "west": wwr_result.west,
                                "confidence": confidence,
                                "backend": backend,
                                "source": img_data.get("source"),
                            }

                            if confidence >= 0.85:
                                # Good enough, stop searching
                                return best_result

                    except Exception as e:
                        logger.debug(f"WWR analysis failed for image: {e}")
                        continue

            except ImportError:
                logger.debug(f"WWR backend {backend} not available")
                continue
            except Exception as e:
                logger.debug(f"WWR detection with {backend} failed: {e}")
                continue

        return best_result

    async def _classify_material(self, images: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Classify facade material from images."""
        try:
            from src.ai.material_classifier import MaterialClassifier
            classifier = MaterialClassifier()

            best_result = None
            best_confidence = 0.0

            for img_data in images:
                img = img_data.get("image")
                if img is None:
                    continue

                try:
                    prediction = classifier.classify(img)
                    confidence = prediction.confidence

                    if confidence > best_confidence:
                        best_confidence = confidence
                        best_result = {
                            "material": prediction.material.value,
                            "confidence": confidence,
                            "all_scores": prediction.all_scores,
                            "source": img_data.get("source"),
                        }

                except Exception as e:
                    logger.debug(f"Material classification failed: {e}")
                    continue

            return best_result

        except ImportError:
            logger.debug("Material classifier not available")
            return None

    async def _claude_vision_analysis(
        self,
        images: List[Dict[str, Any]],
        address: str,
    ) -> Optional[Dict[str, Any]]:
        """Use Komilion/OpenRouter for comprehensive facade analysis."""
        if not self.komilion_api_key:
            return None

        try:
            from ..ai.llm_client import LLMClient
            import tempfile
            import json
            import re

            # Select best image (prefer south-facing for sunlight)
            selected_img = None
            for img_data in images:
                if img_data.get("direction") == "S":
                    selected_img = img_data
                    break
            if selected_img is None and images:
                selected_img = images[0]

            if selected_img is None:
                return None

            # Save image to temp file for LLM client
            img = selected_img["image"]
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                img.save(tmp.name, format="JPEG", quality=85)
                temp_path = tmp.name

            prompt = f"""Analyze this building facade image for {address}.

Please provide:
1. **Facade Material**: What is the primary facade material? (brick, concrete, plaster/render, wood, glass, metal, stone)
2. **Window-to-Wall Ratio**: Estimate the percentage of the facade covered by windows (0-100%)
3. **Building Era**: Based on architectural style, estimate construction era (pre-1930, 1930-1960, 1961-1975, 1976-1995, 1996-2010, 2010+)
4. **Condition**: Overall facade condition (good, fair, poor)
5. **Notable Features**: Any visible features relevant to energy efficiency (e.g., thermal bridges, balconies, shading devices)

Respond in JSON format:
{{
  "material": "...",
  "wwr_percent": 0-100,
  "estimated_era": "...",
  "condition": "...",
  "notable_features": ["..."],
  "confidence": 0.0-1.0,
  "explanation": "Brief explanation of your assessment"
}}"""

            client = LLMClient(mode="balanced")
            response = client.analyze_image(image_path=temp_path, prompt=prompt)

            # Clean up temp file
            import os
            os.unlink(temp_path)

            if not response:
                return None

            # Parse response
            response_text = response.content

            json_match = re.search(r"\{.*\}", response_text, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                return {
                    "values": {
                        "material": data.get("material"),
                        "wwr": data.get("wwr_percent", 0) / 100.0,
                        "estimated_era": data.get("estimated_era"),
                        "condition": data.get("condition"),
                        "notable_features": data.get("notable_features", []),
                    },
                    "confidence": data.get("confidence", 0.7),
                    "explanation": data.get("explanation", ""),
                }

        except Exception as e:
            logger.debug(f"Komilion Vision analysis failed: {e}")

        return None


class ECMRefinerAgent(QCAgent):
    """
    QC agent for ECM package refinement.

    Adjusts ECM recommendations when:
    - Negative savings detected
    - Unusual ECM interactions
    - Building-specific constraints not met
    """

    def can_handle(self, trigger: "QCTriggerType | QCTrigger | str") -> bool:
        normalized = self._normalize_trigger(trigger)
        return normalized in (
            QCTriggerType.NEGATIVE_SAVINGS,
            QCTriggerType.ANOMALOUS_PATTERN,
        )

    async def run(self, building_result: Any) -> QCResult:
        """
        Refine ECM recommendations for building-specific context.

        Args:
            building_result: BuildingResult with problematic ECMs

        Returns:
            QCResult with refined ECM list
        """
        result = QCResult(
            trigger=QCTrigger(trigger_type="negative_savings"),
            success=False,
            action_taken="ecm_refinement",
        )

        try:
            # Check for negative savings
            problematic_ecms = []
            for ecm in building_result.recommended_ecms:
                if ecm.get("savings_kwh_m2", 0) < 0:
                    problematic_ecms.append(ecm)

            if problematic_ecms:
                # Analyze why savings are negative
                for ecm in problematic_ecms:
                    explanation = self._analyze_negative_savings(ecm, building_result)
                    result.explanation += f"{ecm['ecm_id']}: {explanation}\n"

                # Recalculate with interaction matrix
                refined_ecms = self._recalculate_with_interactions(
                    building_result.recommended_ecms,
                    building_result,
                )

                result.updated_values["recommended_ecms"] = refined_ecms
                result.success = True
                result.new_confidence = 0.8

            # Check for unusual patterns
            if self._has_unusual_patterns(building_result):
                result.recommendations.extend([
                    "Manual review recommended for unusual building characteristics",
                    "Consider site-specific factors not in model",
                ])
                result.needs_human_review = True

        except Exception as e:
            logger.error(f"ECMRefinerAgent failed: {e}")
            result.success = False
            result.explanation = str(e)

        return result

    def _analyze_negative_savings(
        self,
        ecm: Dict[str, Any],
        building_result: Any,
    ) -> str:
        """Analyze why an ECM shows negative savings."""
        ecm_id = ecm.get("ecm_id", "")

        # LED lighting + heating interaction
        if ecm_id == "led_lighting":
            return (
                "LED lighting reduces internal heat gains, "
                "increasing heating demand in cold climate. "
                "Net effect depends on cooling vs heating balance."
            )

        # FTX + DCV interaction
        if ecm_id == "demand_controlled_ventilation":
            if building_result.archetype_id and "ftx" in building_result.archetype_id.lower():
                return (
                    "DCV may conflict with existing FTX system. "
                    "Heat recovery efficiency may decrease with variable flow."
                )

        # External insulation on thermal mass
        if ecm_id == "wall_external_insulation":
            return (
                "External insulation may reduce thermal mass benefits. "
                "For buildings with good thermal mass, savings may be lower than expected."
            )

        return "Unknown interaction causing negative savings"

    def _recalculate_with_interactions(
        self,
        ecms: List[Dict[str, Any]],
        building_result: Any,
    ) -> List[Dict[str, Any]]:
        """Recalculate ECM savings with interaction matrix."""
        # ECM interaction matrix (multiplicative factors)
        INTERACTION_MATRIX = {
            ("wall_external_insulation", "window_replacement"): 0.85,
            ("ftx_installation", "demand_controlled_ventilation"): 0.75,
            ("led_lighting", "smart_thermostats"): 0.90,
            ("roof_insulation", "wall_external_insulation"): 0.90,
            ("air_sealing", "ftx_installation"): 0.95,
        }

        refined = []
        applied_ecms = set()

        for ecm in ecms:
            ecm_id = ecm.get("ecm_id", "")
            original_savings = ecm.get("savings_kwh_m2", 0)

            # Apply interaction factors
            interaction_factor = 1.0
            for (ecm1, ecm2), factor in INTERACTION_MATRIX.items():
                if ecm_id == ecm2 and ecm1 in applied_ecms:
                    interaction_factor *= factor
                elif ecm_id == ecm1 and ecm2 in applied_ecms:
                    interaction_factor *= factor

            adjusted_savings = original_savings * interaction_factor

            # Skip if adjusted savings are negative or negligible
            if adjusted_savings <= 0.5:
                continue

            refined_ecm = ecm.copy()
            refined_ecm["savings_kwh_m2"] = adjusted_savings
            refined_ecm["interaction_factor"] = interaction_factor

            refined.append(refined_ecm)
            applied_ecms.add(ecm_id)

        return refined

    def _has_unusual_patterns(self, building_result: Any) -> bool:
        """Check for unusual patterns in building result."""
        # Energy class mismatch with archetype
        if building_result.energy_class and building_result.archetype_id:
            # Pre-1945 with class C or better is unusual
            if building_result.construction_year and building_result.construction_year < 1945:
                if building_result.energy_class in ("A", "B", "C"):
                    return True

        # Very high savings estimate
        if building_result.total_savings_kwh_m2 > 100:
            return True

        # Very low payback
        if building_result.simple_payback_years and building_result.simple_payback_years < 1:
            return True

        return False


class AnomalyAgent(QCAgent):
    """
    QC agent for investigating anomalous patterns.

    Uses LLM reasoning to explain unusual building characteristics
    and suggest investigation priorities.
    """

    def can_handle(self, trigger: "QCTriggerType | QCTrigger | str") -> bool:
        normalized = self._normalize_trigger(trigger)
        return normalized in (
            QCTriggerType.ENERGY_CLASS_MISMATCH,
            QCTriggerType.ANOMALOUS_PATTERN,
        )

    async def run(self, building_result: Any) -> QCResult:
        """
        Investigate anomalous patterns using LLM reasoning.

        Args:
            building_result: BuildingResult with anomalies

        Returns:
            QCResult with explanation and recommendations
        """
        result = QCResult(
            trigger=QCTrigger(trigger_type="anomalous_pattern"),
            success=False,
            action_taken="anomaly_investigation",
        )

        try:
            # Identify anomalies
            anomalies = self._identify_anomalies(building_result)

            if not anomalies:
                result.success = True
                result.explanation = "No significant anomalies detected"
                return result

            # Use LLM to reason about anomalies
            llm_analysis = await self._llm_reasoning(building_result, anomalies)

            if llm_analysis:
                result.explanation = llm_analysis.get("explanation", "")
                result.recommendations = llm_analysis.get("recommendations", [])
                result.updated_values["renovation_detected"] = llm_analysis.get("renovation_detected", False)
                result.updated_values["likely_upgrades"] = llm_analysis.get("likely_upgrades", [])
                result.new_confidence = llm_analysis.get("confidence", 0.7)
                result.success = True
            else:
                # Fallback to rule-based analysis
                result.explanation = self._rule_based_analysis(building_result, anomalies)
                result.recommendations = self._generate_recommendations(anomalies)
                result.success = True

        except Exception as e:
            logger.error(f"AnomalyAgent failed: {e}")
            result.success = False
            result.explanation = str(e)

        return result

    def _identify_anomalies(self, building_result: Any) -> List[str]:
        """Identify anomalies in building result."""
        anomalies = []

        year = building_result.construction_year
        energy_class = building_result.energy_class

        # Old building with good energy class
        if year and year < 1945 and energy_class in ("A", "B", "C"):
            anomalies.append(f"pre_1945_with_{energy_class}")

        # New building with poor energy class
        if year and year > 2010 and energy_class in ("E", "F", "G"):
            anomalies.append(f"post_2010_with_{energy_class}")

        # Miljonprogrammet with class A/B (rare)
        if year and 1961 <= year <= 1975 and energy_class in ("A", "B"):
            anomalies.append(f"miljonprogrammet_with_{energy_class}")

        # Very high baseline consumption
        if building_result.current_kwh_m2 and building_result.current_kwh_m2 > 200:
            anomalies.append("very_high_consumption")

        # Very low consumption for old building
        if year and year < 1960:
            if building_result.current_kwh_m2 and building_result.current_kwh_m2 < 50:
                anomalies.append("unexpectedly_low_consumption")

        return anomalies

    async def _llm_reasoning(
        self,
        building_result: Any,
        anomalies: List[str],
    ) -> Optional[Dict[str, Any]]:
        """Use LLM to reason about anomalies."""
        try:
            # Check for LLM availability
            from src.baseline.llm_archetype_reasoner import LLMArchetypeReasoner

            reasoner = LLMArchetypeReasoner()

            # Build prompt
            prompt = f"""
Analyze this building with unusual characteristics:

Address: {building_result.address}
Construction Year: {building_result.construction_year}
Energy Class: {building_result.energy_class}
Current Consumption: {building_result.current_kwh_m2} kWh/m²
Archetype: {building_result.archetype_id}

Detected Anomalies: {', '.join(anomalies)}

Questions:
1. What could explain these anomalies?
2. Has this building likely been renovated?
3. What upgrades might have been done?
4. What should we investigate further?

Provide a structured analysis.
"""

            # This is a placeholder - real implementation would call LLM
            logger.debug(f"LLM reasoning for anomalies: {anomalies}")
            return None

        except Exception as e:
            logger.debug(f"LLM reasoning failed: {e}")
            return None

    def _rule_based_analysis(
        self,
        building_result: Any,
        anomalies: List[str],
    ) -> str:
        """Generate rule-based explanation for anomalies."""
        explanations = []

        for anomaly in anomalies:
            if "pre_1945" in anomaly:
                explanations.append(
                    f"A {building_result.construction_year} building with energy class "
                    f"{building_result.energy_class} suggests significant renovation. "
                    "Pre-1945 buildings typically have class F-G without upgrades."
                )

            elif "post_2010" in anomaly:
                explanations.append(
                    f"A post-2010 building with energy class {building_result.energy_class} "
                    "is unusual. Possible causes: construction quality issues, "
                    "incorrect declaration, or heavy use patterns."
                )

            elif "miljonprogrammet" in anomaly:
                explanations.append(
                    "Miljonprogrammet building (1961-1975) with excellent energy class "
                    "indicates comprehensive renovation including envelope and ventilation upgrades."
                )

            elif "very_high_consumption" in anomaly:
                explanations.append(
                    f"Very high consumption ({building_result.current_kwh_m2} kWh/m²) "
                    "suggests poor building envelope, inefficient heating, "
                    "or unusual use patterns."
                )

            elif "unexpectedly_low" in anomaly:
                explanations.append(
                    "Unexpectedly low consumption for old building age. "
                    "Likely renovated or declaration may be incomplete."
                )

        return " ".join(explanations)

    def _generate_recommendations(self, anomalies: List[str]) -> List[str]:
        """Generate recommendations based on anomalies."""
        recommendations = []

        for anomaly in anomalies:
            if "pre_1945" in anomaly or "miljonprogrammet" in anomaly:
                recommendations.append("Check renovation history from building records")
                recommendations.append("Verify current ventilation system type")

            if "post_2010" in anomaly:
                recommendations.append("Review original building permit and energy calculation")
                recommendations.append("Check for construction defects or thermal bridges")

            if "very_high_consumption" in anomaly:
                recommendations.append("Prioritize energy audit")
                recommendations.append("Check heating system efficiency and controls")

        return list(set(recommendations))  # Remove duplicates
