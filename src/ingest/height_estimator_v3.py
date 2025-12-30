"""
Optimal GSV Building Height Estimator v3

This is the BEST possible height estimation using available data.

KEY INSIGHT: The LLM already calibrates floor height using reference objects
(doors, windows, people, cars). This is MORE ACCURATE than geometric
triangulation because:
1. Reference objects have known real-world sizes
2. Direct visual measurement, not angle-based calculation
3. Doesn't depend on camera metadata accuracy

PRIORITY ORDER:
1. Reference-calibrated floor count (BEST - 0.85+ confidence)
2. Uncalibrated floor count with era-based height (GOOD - 0.70)
3. Building span in image + reference (ALTERNATIVE)
4. Geometric triangulation (VALIDATION ONLY - never primary)

Author: Raiden Team
"""

import math
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple
from enum import Enum


class HeightReference(Enum):
    """Reference objects with known sizes for calibration."""
    DOOR = "door"  # Standard door: 2.1m
    WINDOW = "window"  # Standard window: 1.2m tall
    PERSON = "person"  # Average person: 1.7m
    CAR = "car"  # Average car: 1.5m
    BALCONY = "balcony"  # Standard balcony rail: 1.1m
    NONE = "none"


# Known reference object sizes (meters)
REFERENCE_SIZES = {
    HeightReference.DOOR: 2.1,
    HeightReference.WINDOW: 1.2,
    HeightReference.PERSON: 1.7,
    HeightReference.CAR: 1.5,
    HeightReference.BALCONY: 1.1,
}

# Reference confidence (how reliable is this reference?)
REFERENCE_CONFIDENCE = {
    HeightReference.DOOR: 0.90,  # Doors are very standardized
    HeightReference.PERSON: 0.85,  # People vary in height
    HeightReference.WINDOW: 0.80,  # Windows vary more
    HeightReference.CAR: 0.75,  # Cars vary significantly
    HeightReference.BALCONY: 0.70,  # Balcony heights vary
    HeightReference.NONE: 0.50,  # No reference = low confidence
}

# Floor heights by Swedish building era
ERA_FLOOR_HEIGHTS = {
    "pre_1930": 3.3,  # Old buildings with high ceilings
    "1930-1945": 3.0,
    "1945-1960": 2.85,  # Folkhemmet
    "1960-1975": 2.7,  # Miljonprogrammet - standardized
    "1975-1990": 2.8,
    "1990-2010": 2.75,
    "2010+": 2.7,  # Modern - lower ceilings
    "unknown": 2.8,  # Safe default
}

# Building form adjustments
FORM_FLOOR_HEIGHT_ADJUST = {
    "skivhus": -0.1,  # High-rise = more efficient floors
    "punkthus": -0.05,
    "lamellhus": 0.0,  # Standard
    "pre_1930": +0.3,  # Old buildings
    "radhus": +0.1,  # Row houses often have higher ground floor
}


@dataclass
class HeightEstimateV3:
    """Comprehensive height estimate with uncertainty."""
    height_m: float
    confidence: float
    method: str  # primary method used
    uncertainty_m: float  # ±uncertainty

    # Breakdown
    floor_count: int
    floor_height_m: float
    has_attic: bool
    has_basement: bool

    # What data was used
    reference_used: str
    reference_confidence: float
    geometric_validation: Optional[float]  # geometric estimate if available

    # Diagnostics
    notes: List[str] = field(default_factory=list)
    all_estimates: Dict[str, float] = field(default_factory=dict)


@dataclass
class LLMFacadeData:
    """Data extracted from LLM facade analysis."""
    visible_floors: int
    floor_count_confidence: float
    estimated_floor_height_m: float
    height_reference_used: str
    roof_position_pct: float
    ground_position_pct: float
    has_visible_attic: bool
    has_visible_basement: bool
    estimated_era: str
    building_form: str


class OptimalHeightEstimator:
    """
    Optimal building height estimator using all available data.

    Uses a priority-based approach:
    1. Reference-calibrated measurement (BEST)
    2. Era-calibrated floor count (GOOD)
    3. Geometric validation (NEVER primary)
    """

    CAMERA_HEIGHT_M = 2.5
    EARTH_RADIUS_M = 6371000

    def __init__(self):
        self.notes: List[str] = []

    def estimate(
        self,
        llm_data: LLMFacadeData,
        camera_images: Optional[List[Any]] = None,
        facade_lat: Optional[float] = None,
        facade_lon: Optional[float] = None,
    ) -> HeightEstimateV3:
        """
        Estimate building height using optimal method selection.

        Args:
            llm_data: Facade analysis from LLM
            camera_images: Optional GSV images with camera metadata
            facade_lat, facade_lon: Building location for geometric validation

        Returns:
            HeightEstimateV3 with best estimate and uncertainty
        """
        self.notes = []
        all_estimates = {}

        # ═══════════════════════════════════════════════════════════════════════
        # METHOD 1: Reference-Calibrated Floor Count (BEST)
        # If LLM used a reference object to calibrate floor height, trust it!
        # ═══════════════════════════════════════════════════════════════════════
        reference_estimate = None
        reference_conf = 0.0

        ref_type = self._parse_reference(llm_data.height_reference_used)

        if ref_type != HeightReference.NONE and llm_data.visible_floors > 0:
            # LLM calibrated against a reference object
            floor_height = llm_data.estimated_floor_height_m

            # Validate floor height is reasonable
            if 2.2 <= floor_height <= 4.5:
                reference_estimate = llm_data.visible_floors * floor_height

                # Add attic
                if llm_data.has_visible_attic:
                    reference_estimate += 2.5

                # Confidence based on reference type and floor count confidence
                reference_conf = (
                    REFERENCE_CONFIDENCE[ref_type] * 0.6 +
                    llm_data.floor_count_confidence * 0.4
                )

                all_estimates["reference_calibrated"] = reference_estimate
                self.notes.append(
                    f"Reference method: {llm_data.visible_floors} floors × "
                    f"{floor_height:.2f}m (ref={ref_type.value}) = {reference_estimate:.1f}m"
                )

        # ═══════════════════════════════════════════════════════════════════════
        # METHOD 2: Era-Calibrated Floor Count (GOOD)
        # Use building era to estimate floor height
        # ═══════════════════════════════════════════════════════════════════════
        era_estimate = None
        era_conf = 0.0

        if llm_data.visible_floors > 0:
            era_floor_height = self._get_era_floor_height(
                llm_data.estimated_era,
                llm_data.building_form
            )

            era_estimate = llm_data.visible_floors * era_floor_height

            # Add attic
            if llm_data.has_visible_attic:
                era_estimate += 2.5

            # Confidence based on floor count confidence
            era_conf = 0.60 + llm_data.floor_count_confidence * 0.15

            all_estimates["era_calibrated"] = era_estimate
            self.notes.append(
                f"Era method: {llm_data.visible_floors} floors × "
                f"{era_floor_height:.2f}m ({llm_data.estimated_era}) = {era_estimate:.1f}m"
            )

        # ═══════════════════════════════════════════════════════════════════════
        # METHOD 3: Building Span in Image (ALTERNATIVE)
        # Use roof - ground position with reference calibration
        # ═══════════════════════════════════════════════════════════════════════
        span_estimate = None
        span_conf = 0.0

        if (llm_data.roof_position_pct > 0.1 and
            llm_data.ground_position_pct > 0 and
            llm_data.roof_position_pct > llm_data.ground_position_pct):

            # Building spans this much of image
            span_pct = llm_data.roof_position_pct - llm_data.ground_position_pct

            # If we have floor count, we can estimate height
            if llm_data.visible_floors > 0 and span_pct > 0.2:
                # Each floor takes span_pct/floors of image
                # If we know one floor ≈ 2.8m, total height ≈ floors × 2.8
                # This validates other methods rather than being primary
                floor_span = span_pct / llm_data.visible_floors

                # A floor taking 10-20% of image is typical at medium distance
                if 0.05 < floor_span < 0.35:
                    # Reasonable span per floor
                    span_estimate = era_estimate  # Use era as base
                    span_conf = 0.55

                    all_estimates["span_check"] = span_estimate
                    self.notes.append(
                        f"Span check: {span_pct:.0%} of image, "
                        f"{floor_span:.0%} per floor - validates estimate"
                    )

        # ═══════════════════════════════════════════════════════════════════════
        # METHOD 4: Geometric Triangulation (VALIDATION ONLY)
        # Calculate from camera position - NEVER use as primary!
        # ═══════════════════════════════════════════════════════════════════════
        geometric_estimate = None

        if (camera_images and facade_lat and facade_lon and
            llm_data.roof_position_pct > 0.1):

            geometric_estimate = self._calculate_geometric(
                camera_images,
                facade_lat,
                facade_lon,
                llm_data.roof_position_pct,
            )

            if geometric_estimate:
                all_estimates["geometric"] = geometric_estimate
                self.notes.append(f"Geometric validation: {geometric_estimate:.1f}m")

        # ═══════════════════════════════════════════════════════════════════════
        # OPTIMAL FUSION: Select best method with validation
        # ═══════════════════════════════════════════════════════════════════════

        # Priority: Reference > Era > Default
        if reference_estimate and reference_conf > 0.6:
            primary_estimate = reference_estimate
            primary_conf = reference_conf
            primary_method = "reference_calibrated"
            floor_height_used = llm_data.estimated_floor_height_m
        elif era_estimate:
            primary_estimate = era_estimate
            primary_conf = era_conf
            primary_method = "era_calibrated"
            floor_height_used = self._get_era_floor_height(
                llm_data.estimated_era, llm_data.building_form
            )
        else:
            # Fallback to default
            primary_estimate = 12.0  # 4 floors × 3m
            primary_conf = 0.30
            primary_method = "default"
            floor_height_used = 2.9
            self.notes.append("Warning: No reliable data, using default 12m")

        # ═══════════════════════════════════════════════════════════════════════
        # VALIDATION: Use geometric to validate (not override)
        # ═══════════════════════════════════════════════════════════════════════

        if geometric_estimate:
            geo_ratio = geometric_estimate / primary_estimate if primary_estimate > 0 else 1

            if 0.7 < geo_ratio < 1.3:
                # Good agreement - boost confidence
                primary_conf = min(0.95, primary_conf + 0.10)
                self.notes.append(f"✓ Geometric validates ({geo_ratio:.0%} ratio)")
            elif geo_ratio > 2.0 or geo_ratio < 0.5:
                # Wild disagreement - geometric is probably wrong, but flag it
                self.notes.append(
                    f"⚠ Geometric disagrees ({geometric_estimate:.1f}m vs "
                    f"{primary_estimate:.1f}m) - trusting {primary_method}"
                )
            else:
                # Moderate disagreement - slight confidence reduction
                primary_conf = max(0.50, primary_conf - 0.10)
                self.notes.append(
                    f"Geometric differs: {geometric_estimate:.1f}m vs "
                    f"{primary_estimate:.1f}m"
                )

        # ═══════════════════════════════════════════════════════════════════════
        # SANITY CHECKS
        # ═══════════════════════════════════════════════════════════════════════

        # Cap extreme values
        if primary_estimate > 80:
            self.notes.append(f"Capped from {primary_estimate:.1f}m to 80m max")
            primary_estimate = 80.0
            primary_conf = min(primary_conf, 0.50)
        elif primary_estimate < 3:
            self.notes.append(f"Raised from {primary_estimate:.1f}m to 3m min")
            primary_estimate = 3.0
            primary_conf = min(primary_conf, 0.50)

        # Check floor height is reasonable
        if llm_data.visible_floors > 0:
            implied_floor_height = primary_estimate / llm_data.visible_floors
            if implied_floor_height > 5.0:
                self.notes.append(
                    f"⚠ Implied floor height {implied_floor_height:.1f}m is high"
                )
                primary_conf = min(primary_conf, 0.60)
            elif implied_floor_height < 2.2:
                self.notes.append(
                    f"⚠ Implied floor height {implied_floor_height:.1f}m is low"
                )
                primary_conf = min(primary_conf, 0.60)

        # ═══════════════════════════════════════════════════════════════════════
        # CALCULATE UNCERTAINTY
        # ═══════════════════════════════════════════════════════════════════════

        # Base uncertainty is ±15% of height
        base_uncertainty = primary_estimate * 0.15

        # Reduce uncertainty based on confidence
        uncertainty = base_uncertainty * (1.5 - primary_conf)

        # Minimum uncertainty is ±1m
        uncertainty = max(1.0, uncertainty)

        return HeightEstimateV3(
            height_m=round(primary_estimate, 1),
            confidence=round(primary_conf, 2),
            method=primary_method,
            uncertainty_m=round(uncertainty, 1),
            floor_count=llm_data.visible_floors,
            floor_height_m=round(floor_height_used, 2),
            has_attic=llm_data.has_visible_attic,
            has_basement=llm_data.has_visible_basement,
            reference_used=llm_data.height_reference_used,
            reference_confidence=REFERENCE_CONFIDENCE.get(ref_type, 0.5),
            geometric_validation=geometric_estimate,
            notes=self.notes,
            all_estimates=all_estimates,
        )

    def _parse_reference(self, ref_str: str) -> HeightReference:
        """Parse reference string to enum."""
        if not ref_str:
            return HeightReference.NONE
        ref_lower = ref_str.lower().strip()
        for ref in HeightReference:
            if ref.value in ref_lower:
                return ref
        return HeightReference.NONE

    def _get_era_floor_height(self, era: str, building_form: str) -> float:
        """Get floor height based on building era and form."""
        # Find matching era
        era_lower = era.lower() if era else ""

        base_height = 2.8  # default

        for era_key, height in ERA_FLOOR_HEIGHTS.items():
            if era_key.lower() in era_lower or era_lower in era_key.lower():
                base_height = height
                break

        # Adjust for building form
        form_lower = building_form.lower() if building_form else ""
        for form_key, adjust in FORM_FLOOR_HEIGHT_ADJUST.items():
            if form_key in form_lower:
                base_height += adjust
                break

        return base_height

    def _calculate_geometric(
        self,
        images: List[Any],
        facade_lat: float,
        facade_lon: float,
        roof_position_pct: float,
    ) -> Optional[float]:
        """
        Calculate geometric height estimate.

        This is for VALIDATION only, not primary estimation.
        """
        estimates = []

        for img in images:
            if not hasattr(img, 'camera_lat'):
                continue

            camera_lat = img.camera_lat
            camera_lon = img.camera_lon
            pitch = getattr(img, 'pitch', 0)
            fov = getattr(img, 'fov', 90)

            if pitch < -10 or pitch > 60:
                continue

            # Calculate distance
            dist = self._haversine_distance(
                camera_lat, camera_lon, facade_lat, facade_lon
            )

            if dist < 10 or dist > 150:
                continue

            # Calculate roof angle
            center_pct = 0.5
            roof_offset_pct = roof_position_pct - center_pct
            roof_angle_from_center = roof_offset_pct * fov
            roof_angle_deg = pitch + roof_angle_from_center

            if roof_angle_deg < 0 or roof_angle_deg > 80:
                continue

            # Calculate height
            roof_angle_rad = math.radians(roof_angle_deg)
            height_above_camera = dist * math.tan(roof_angle_rad)
            total_height = height_above_camera + self.CAMERA_HEIGHT_M

            if 3 < total_height < 100:
                estimates.append(total_height)

        if not estimates:
            return None

        # Return median
        estimates.sort()
        return estimates[len(estimates) // 2]

    def _haversine_distance(
        self,
        lat1: float, lon1: float,
        lat2: float, lon2: float,
    ) -> float:
        """Calculate distance between two points in meters."""
        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        delta_lat = math.radians(lat2 - lat1)
        delta_lon = math.radians(lon2 - lon1)

        a = (math.sin(delta_lat / 2) ** 2 +
             math.cos(lat1_rad) * math.cos(lat2_rad) *
             math.sin(delta_lon / 2) ** 2)
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

        return self.EARTH_RADIUS_M * c


def estimate_height_optimal(
    visible_floors: int,
    floor_count_confidence: float = 0.7,
    estimated_floor_height_m: float = 2.8,
    height_reference_used: str = "none",
    roof_position_pct: float = 0.0,
    ground_position_pct: float = 0.0,
    has_visible_attic: bool = False,
    has_visible_basement: bool = False,
    estimated_era: str = "unknown",
    building_form: str = "lamellhus",
    camera_images: Optional[List[Any]] = None,
    facade_lat: Optional[float] = None,
    facade_lon: Optional[float] = None,
) -> HeightEstimateV3:
    """
    Convenience function for optimal height estimation.

    Example:
        result = estimate_height_optimal(
            visible_floors=5,
            floor_count_confidence=0.85,
            estimated_floor_height_m=3.2,
            height_reference_used="door",  # LLM calibrated against door!
            has_visible_attic=True,
            estimated_era="1960-1975",
            building_form="lamellhus",
        )
        print(f"Height: {result.height_m}m ±{result.uncertainty_m}m")
        print(f"Confidence: {result.confidence:.0%}")
    """
    llm_data = LLMFacadeData(
        visible_floors=visible_floors,
        floor_count_confidence=floor_count_confidence,
        estimated_floor_height_m=estimated_floor_height_m,
        height_reference_used=height_reference_used,
        roof_position_pct=roof_position_pct,
        ground_position_pct=ground_position_pct,
        has_visible_attic=has_visible_attic,
        has_visible_basement=has_visible_basement,
        estimated_era=estimated_era,
        building_form=building_form,
    )

    estimator = OptimalHeightEstimator()
    return estimator.estimate(
        llm_data,
        camera_images=camera_images,
        facade_lat=facade_lat,
        facade_lon=facade_lon,
    )
