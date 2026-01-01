#!/usr/bin/env python3
"""
PRODUCTION PIPELINE V2: Full Infrastructure Utilization

This version uses ALL available Raiden infrastructure:
1. Mapillary facade images → WWR detection + material classification
2. Google Solar API → roof analysis + PV potential
3. BuildingGeometryCalculator → actual wall areas per orientation
4. Bayesian ABC-SMC calibration → proper parameter estimation
5. ArchetypeMatcherV2 → AI-enhanced archetype scoring
6. LLM Archetype Reasoner → renovation detection
7. ECM Catalog + Constraint Engine → proper ECM filtering
8. Energy Signature Analysis → building thermal characterization

Usage:
    python scripts/run_production_pipeline_v2.py "Aktergatan 11, Stockholm" --year 2003
"""

import argparse
import asyncio
import json
import logging
import math
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv
load_dotenv()  # Load .env before accessing environment variables

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class PipelineConfig:
    """Configuration for pipeline V2."""
    # Feature flags
    use_mapillary_images: bool = True
    use_google_streetview: bool = True  # Google Street View for facades
    use_google_solar_api: bool = True
    use_bayesian_calibration: bool = True
    use_llm_reasoner: bool = False  # Requires API key
    use_ai_facade_analysis: bool = True

    # Calibration settings
    calibration_samples: int = 100  # LHS samples for surrogate
    calibration_iterations: int = 5  # ABC-SMC iterations

    # API keys (from environment)
    mapillary_token: str = field(default_factory=lambda: os.environ.get("MAPILLARY_TOKEN", ""))
    google_api_key: str = field(default_factory=lambda: os.environ.get("BRF_GOOGLE_API_KEY") or os.environ.get("GOOGLE_API_KEY", ""))
    komilion_api_key: str = field(default_factory=lambda: os.environ.get("KOMILION_API_KEY", ""))


# ============================================================================
# PRICING AND COSTS
# ============================================================================

@dataclass
class ElectricityPricing:
    """Ellevio electricity cost model with peak demand charges."""
    spot_price: float = 0.80
    grid_energy_fee: float = 0.32
    energy_tax: float = 0.45
    vat_rate: float = 0.25
    grid_peak_fee_sek_kw_month: float = 70.0  # Effektavgift

    @property
    def total_energy_price(self) -> float:
        base = self.spot_price + self.grid_energy_fee + self.energy_tax
        return base * (1 + self.vat_rate)


# Heat pump COPs (Seasonal Performance Factor)
HEAT_PUMP_COP = {
    "ground_source": 3.8,
    "exhaust_air": 2.8,
    "air_water": 3.0,
    "air_air": 2.5,
    "ground_source_hp": 3.8,
    "direct_electric": 1.0,
}

# Legacy ECM costs (fallback only - prefer SwedishCostCalculatorV2)
ECM_COSTS = {
    "effektvakt_simple": 50_000,
    "effektvakt_advanced": 150_000,
    "led_common_areas": 300,
    "smart_thermostats": 500,
    "ftx_heat_exchanger_upgrade": 200,
    "ftx_new_installation": 2000,
    "window_replacement_standard": 4000,
    "window_replacement_energy": 5500,
    "window_replacement_passive": 7000,
    "air_sealing_basic": 50,
    "air_sealing_comprehensive": 150,
    "facade_insulation_100mm": 2000,
    "roof_insulation_200mm": 800,
}


def get_cost_calculator(building_data: Dict) -> "SwedishCostCalculatorV2":
    """Get cost calculator with proper region and building context."""
    try:
        from src.roi.costs_sweden_v2 import SwedishCostCalculatorV2, Region, OwnerType

        # Determine region from address
        address = building_data.get("address", "").lower()
        if "stockholm" in address:
            region = Region.STOCKHOLM
        elif "göteborg" in address or "gothenburg" in address:
            region = Region.GOTHENBURG
        elif "malmö" in address or "malmo" in address:
            region = Region.MALMO
        else:
            region = Region.MEDIUM_CITY

        return SwedishCostCalculatorV2(
            region=region,
            year=2025,
            owner_type=OwnerType.BRF,  # Multi-family default
        )
    except ImportError:
        return None


def calculate_ecm_cost_v2(
    ecm_id: str,
    quantity: float,
    building_data: Dict,
    fallback_cost: float = 0,
) -> Tuple[float, Dict]:
    """
    Calculate ECM cost using SwedishCostCalculatorV2.

    Returns (total_cost, cost_breakdown_dict).
    Falls back to simple calculation if V2 not available.
    """
    calc = get_cost_calculator(building_data)

    if calc:
        try:
            floor_area = building_data.get("atemp_m2", 2000)
            cost = calc.calculate_ecm_cost(
                ecm_id=ecm_id,
                quantity=quantity,
                floor_area_m2=floor_area,
                include_maintenance=False,
            )
            return cost.total_after_deductions, cost.to_dict()
        except (ValueError, KeyError) as e:
            logger.debug(f"Cost V2 not available for {ecm_id}: {e}")

    # Fallback to simple cost
    return fallback_cost * quantity, {"source": "fallback", "ecm_id": ecm_id}


# ============================================================================
# THERMAL INERTIA AND EFFEKTVAKT
# ============================================================================

def estimate_peak_demand_kw(building_data: Dict) -> float:
    """Estimate building peak electrical demand."""
    atemp_m2 = building_data.get("atemp_m2", 2000)
    has_heat_pump = building_data.get("has_heat_pump", False)
    heat_pump_types = building_data.get("heat_pump_types", [])

    base_load_kw = atemp_m2 * 7 / 1000

    if has_heat_pump:
        hp_thermal_capacity_kw = atemp_m2 * 40 / 1000
        design_cop = 2.8 if "ground_source" in heat_pump_types else 2.2
        hp_electrical_kw = hp_thermal_capacity_kw / design_cop
        return base_load_kw + hp_electrical_kw
    return base_load_kw


def calculate_thermal_inertia(building_data: Dict) -> Dict:
    """Calculate building thermal inertia and coast time."""
    atemp_m2 = building_data.get("atemp_m2", 2000)
    construction_year = building_data.get("construction_year", 1970)
    facade_material = building_data.get("facade_material", "concrete")

    capacitance_per_m2 = {
        "concrete": 70, "brick": 50, "plaster": 45, "wood": 25, "metal": 20
    }.get(facade_material, 50)

    total_capacitance = atemp_m2 * capacitance_per_m2

    if construction_year >= 2010:
        ua_per_m2 = 0.5
    elif construction_year >= 1990:
        ua_per_m2 = 0.7
    elif construction_year >= 1975:
        ua_per_m2 = 1.0
    else:
        ua_per_m2 = 1.5

    total_ua = atemp_m2 * ua_per_m2
    time_constant_hours = total_capacitance / total_ua

    t_in, t_out, delta_t = 21, -5, 1
    coast_time_1deg = time_constant_hours * math.log((t_in - t_out) / (t_in - delta_t - t_out))

    return {
        "thermal_capacitance_kwh_k": total_capacitance / 1000,
        "heat_loss_coefficient_kw_k": total_ua / 1000,
        "time_constant_hours": time_constant_hours,
        "coast_time_1deg_design_hours": coast_time_1deg,
    }


def calculate_effektvakt_savings(building_data: Dict, thermal_inertia: Dict) -> Dict:
    """Calculate potential savings from Effektvakt (peak shaving)."""
    current_peak_kw = estimate_peak_demand_kw(building_data)
    coast_time = thermal_inertia.get("coast_time_1deg_design_hours", 3)

    if coast_time >= 4:
        peak_reduction_percent = 0.35
        strategy = "Avancerad effektvakt med full termisk styrning"
        investment = ECM_COSTS["effektvakt_advanced"]
    elif coast_time >= 2:
        peak_reduction_percent = 0.20
        strategy = "Standard effektvakt med förvärmning"
        investment = ECM_COSTS["effektvakt_simple"]
    else:
        peak_reduction_percent = 0.10
        strategy = "Enkel effektövervakning"
        investment = int(ECM_COSTS["effektvakt_simple"] * 0.5)

    new_peak_kw = current_peak_kw * (1 - peak_reduction_percent)
    peak_reduction_kw = current_peak_kw - new_peak_kw

    pricing = ElectricityPricing()
    annual_peak_savings = peak_reduction_kw * pricing.grid_peak_fee_sek_kw_month * 12 * 1.25

    payback_years = investment / annual_peak_savings if annual_peak_savings > 0 else float('inf')

    return {
        "current_peak_kw": current_peak_kw,
        "new_peak_kw": new_peak_kw,
        "peak_reduction_kw": peak_reduction_kw,
        "peak_reduction_percent": peak_reduction_percent * 100,
        "annual_savings_sek": annual_peak_savings,
        "investment_sek": investment,
        "payback_years": payback_years,
        "strategy": strategy,
        "coast_time_hours": coast_time,
    }


def get_energy_price_and_cop(building_data: Dict) -> Tuple[float, float]:
    """Get energy price and COP based on heating system."""
    heating_system = building_data.get("heating_system", "unknown")
    heat_pump_types = building_data.get("heat_pump_types", [])

    if building_data.get("has_heat_pump", False):
        if "ground_source" in heat_pump_types:
            cop = HEAT_PUMP_COP["ground_source"]
        elif "exhaust_air" in heat_pump_types:
            cop = HEAT_PUMP_COP["exhaust_air"]
        else:
            cop = 3.0
        pricing = ElectricityPricing()
        return pricing.total_energy_price, cop
    elif building_data.get("has_district_heating", False):
        return 0.75, 1.0  # District heating price
    else:
        pricing = ElectricityPricing()
        return pricing.total_energy_price, 1.0


# ============================================================================
# STEP 1: COMPREHENSIVE DATA FETCHING
# ============================================================================

def fetch_building_data_v2(address: str, config: PipelineConfig, known_data: Dict = None) -> Dict:
    """
    Fetch building data from ALL available sources in parallel.

    Sources:
    1. Sweden Buildings GeoJSON (primary - 37,489 buildings with 167 properties)
    2. Overture/OSM (footprint, height, tags)
    3. Mapillary images → AI analysis (WWR, material)
    4. Google Solar API → roof analysis
    5. Nominatim geocoding (fallback)
    """
    logger.info(f"STEP 1: Comprehensive data fetch for: {address}")

    result = {
        "address": address,
        "data_sources": [],
        "confidence_scores": {},
        "ai_analysis": {},
    }

    # Merge known data
    if known_data:
        result.update(known_data)
        result["data_sources"].append("user_input")

    # === 1. Sweden Buildings GeoJSON (PRIMARY SOURCE) ===
    try:
        from src.ingest import load_sweden_buildings

        loader = load_sweden_buildings()
        search_address = address.split(",")[0].strip()
        matches = loader.find_by_address(search_address)

        if matches:
            building = matches[0]

            # Extract comprehensive heating data
            hp_data = {}
            if building.ground_source_hp_kwh > 0:
                hp_data["ground_source"] = building.ground_source_hp_kwh
            if building.exhaust_air_hp_kwh > 0:
                hp_data["exhaust_air"] = building.exhaust_air_hp_kwh
            if building.air_water_hp_kwh > 0:
                hp_data["air_water"] = building.air_water_hp_kwh
            if building.air_air_hp_kwh > 0:
                hp_data["air_air"] = building.air_air_hp_kwh

            # Get WGS84 coordinates from centroid
            lat, lon = building.get_centroid_wgs84()

            result.update({
                "construction_year": building.construction_year or result.get("construction_year"),
                "atemp_m2": building.atemp_m2 or result.get("atemp_m2"),
                "energy_class": building.energy_class,
                "declared_kwh_m2": building.energy_performance_kwh_m2,
                "ventilation_type": building.ventilation_type,
                "heating_system": building.get_primary_heating(),
                "has_ftx": building.ventilation_type == "FTX",
                "has_ft": building.ventilation_type == "FT",
                "has_solar_pv": building.has_solar_pv,
                "has_solar_thermal": building.has_solar_thermal,
                "solar_production_kwh": building.solar_production_kwh,
                "has_heat_pump": len(hp_data) > 0,
                "heat_pump_types": list(hp_data.keys()),
                "heat_pump_kwh": hp_data,
                "has_district_heating": building.district_heating_kwh > 0,
                "has_district_cooling": building.district_cooling_kwh > 0,
                "district_heating_kwh": building.district_heating_kwh,
                "heating_sources": {
                    "hp_ground_source": building.ground_source_hp_kwh,
                    "hp_exhaust_air": building.exhaust_air_hp_kwh,
                },
                "property_electricity_kwh": building.property_electricity_kwh,
                "hot_water_electricity_kwh": building.hot_water_electricity_kwh,
                "total_energy_kwh": building.total_energy_kwh,
                "primary_energy_kwh": building.primary_energy_kwh,
                "num_apartments": building.num_apartments,
                "num_floors": building.num_floors,
                "footprint_area_m2": building.footprint_area_m2,
                "footprint_coords": building.footprint_coords,  # ACTUAL FOOTPRINT!
                "lat": lat,
                "lon": lon,
            })
            result["data_sources"].append("sweden_geojson")
            result["confidence_scores"]["energy_data"] = 0.90

            logger.info(f"  ✓ GeoJSON: {building.energy_class}, {building.atemp_m2} m²")

    except Exception as e:
        logger.warning(f"  GeoJSON lookup failed: {e}")

    # === 2. Mapillary Facade Images + AI Analysis ===
    if config.use_mapillary_images and config.mapillary_token and result.get("footprint_coords"):
        try:
            logger.info("  Fetching Mapillary facade images...")
            from src.ingest.image_fetcher import FacadeImageFetcher

            fetcher = FacadeImageFetcher(mapillary_token=config.mapillary_token)
            # Create building_id from address
            building_id = address.replace(" ", "_").replace(",", "")[:30]
            images = fetcher.fetch_for_building(
                building_coords=result["footprint_coords"],
                building_id=building_id,
                search_radius_m=100
            )

            result["facade_images"] = {
                direction: [img.url for img in imgs[:3]]  # Top 3 per direction
                for direction, imgs in images.items()
                if imgs
            }

            total_images = sum(len(imgs) for imgs in result["facade_images"].values())
            logger.info(f"  ✓ Mapillary: {total_images} images ({', '.join(result['facade_images'].keys())})")
            result["data_sources"].append("mapillary")

            # === AI Analysis of Facade Images ===
            if config.use_ai_facade_analysis and total_images > 0:
                result["ai_analysis"] = analyze_facade_images(images, config)

        except Exception as e:
            logger.warning(f"  Mapillary fetch failed: {e}")

    # === 2b. Google Street View (fallback/complement to Mapillary) ===
    streetview_images = {}
    if config.use_google_streetview and config.google_api_key and result.get("lat"):
        # Only fetch Street View if we don't have good Mapillary coverage
        existing_images = sum(len(imgs) for imgs in result.get("facade_images", {}).values())
        if existing_images < 4:  # Less than 1 image per facade on average
            try:
                logger.info("  Fetching Google Street View images...")
                from src.ingest.streetview_fetcher import StreetViewFacadeFetcher

                # Build GeoJSON footprint from WGS84 coords
                wgs84_coords = []
                if result.get("footprint_coords"):
                    from src.ingest.sweden_buildings import sweref99_to_wgs84
                    coords = result["footprint_coords"]
                    if coords and isinstance(coords[0], list) and len(coords[0]) > 0:
                        ring = coords[0]
                        for p in ring:
                            if len(p) >= 2:
                                lat, lon = sweref99_to_wgs84(p[0], p[1])
                                wgs84_coords.append([lon, lat])
                    else:
                        for p in coords:
                            if len(p) >= 2:
                                lat, lon = sweref99_to_wgs84(p[0], p[1])
                                wgs84_coords.append([lon, lat])

                if wgs84_coords:
                    footprint_geojson = {
                        "type": "Polygon",
                        "coordinates": [wgs84_coords]
                    }

                    fetcher = StreetViewFacadeFetcher(api_key=config.google_api_key)
                    streetview_images = fetcher.fetch_facade_images(
                        footprint=footprint_geojson,
                        image_size="640x480",
                        fov=90,
                        pitch=25
                    )

                    sv_count = len(streetview_images)
                    logger.info(f"  ✓ Street View: {sv_count} facade images")
                    if sv_count > 0:
                        result["data_sources"].append("google_streetview")
                        result["confidence_scores"]["streetview"] = 0.85

                        # Run AI analysis on Street View images if we don't have Mapillary results
                        if config.use_ai_facade_analysis and not result.get("ai_analysis"):
                            result["ai_analysis"] = analyze_streetview_images(streetview_images, config)

            except Exception as e:
                logger.warning(f"  Google Street View failed: {e}")

    # === 3. Google Solar API for Roof Analysis ===
    if config.use_google_solar_api and config.google_api_key and result.get("lat"):
        try:
            logger.info("  Fetching Google Solar API data...")
            from src.analysis.roof_analyzer import RoofAnalyzer

            analyzer = RoofAnalyzer(google_api_key=config.google_api_key)
            roof_analysis = analyzer.analyze(
                lat=result["lat"],
                lon=result["lon"],
                footprint_area_m2=result.get("footprint_area_m2", 500)
            )

            result["roof_analysis"] = {
                "roof_type": roof_analysis.roof_type.value if hasattr(roof_analysis.roof_type, 'value') else str(roof_analysis.roof_type),
                "total_area_m2": roof_analysis.total_area_m2,
                "usable_area_m2": roof_analysis.net_available_m2,
                "azimuth_deg": roof_analysis.primary_azimuth_deg,
                "slope_deg": roof_analysis.primary_pitch_deg,
                "existing_solar": roof_analysis.existing_solar is not None,
                "pv_potential_kwp": roof_analysis.optimal_capacity_kwp,
                "annual_yield_kwh": roof_analysis.optimal_capacity_kwp * roof_analysis.annual_yield_kwh_per_kwp,
                "obstructions": [o.type.value if hasattr(o.type, 'value') else str(o.type) for o in roof_analysis.obstructions],
            }

            logger.info(f"  ✓ Google Solar: {roof_analysis.roof_type}, "
                       f"{roof_analysis.optimal_capacity_kwp:.0f} kWp potential")
            result["data_sources"].append("google_solar")
            result["confidence_scores"]["roof_data"] = 0.85

        except Exception as e:
            logger.warning(f"  Google Solar API failed: {e}")

    # === 4. Calculate Actual Building Geometry ===
    if result.get("footprint_coords"):
        try:
            from src.geometry.building_geometry import BuildingGeometryCalculator
            from src.ingest.sweden_buildings import sweref99_to_wgs84

            # Get WWR from AI analysis or default
            wwr_by_orientation = result.get("ai_analysis", {}).get("wwr_by_orientation", {
                "N": 0.15, "S": 0.25, "E": 0.20, "W": 0.20
            })

            # Convert SWEREF99 TM coords to WGS84
            # sweden_buildings has 3D coords (x, y, z) in nested rings
            coords = result["footprint_coords"]
            wgs84_coords = []

            if coords and isinstance(coords[0], list) and len(coords[0]) > 0:
                # It's a ring of rings - take first ring
                ring = coords[0]
                for p in ring:
                    if len(p) >= 2:
                        # p is (x, y) or (x, y, z) in SWEREF99 TM
                        lat, lon = sweref99_to_wgs84(p[0], p[1])
                        wgs84_coords.append((lon, lat))  # BuildingGeom expects (lon, lat)
            else:
                for p in coords:
                    if len(p) >= 2:
                        lat, lon = sweref99_to_wgs84(p[0], p[1])
                        wgs84_coords.append((lon, lat))

            if not wgs84_coords:
                raise ValueError("No valid coordinates found")

            calculator = BuildingGeometryCalculator()
            geom = calculator.calculate(
                footprint_coords=wgs84_coords,
                height_m=result.get("num_floors", 4) * 2.8,
                floors=result.get("num_floors", 4),
                wwr_by_orientation=wwr_by_orientation
            )

            result["geometry"] = {
                "facade_areas": {d: f.wall_area_m2 for d, f in geom.facades.items()},
                "window_areas": {d: f.window_area_m2 for d, f in geom.facades.items()},
                "total_facade_area_m2": geom.total_wall_area_m2,
                "total_window_area_m2": geom.total_window_area_m2,
                "perimeter_m": geom.perimeter_m,
                "average_wwr": geom.average_wwr,
            }

            logger.info(f"  ✓ Geometry: {geom.total_wall_area_m2:.0f} m² facade, "
                       f"{geom.total_window_area_m2:.0f} m² windows, WWR={geom.average_wwr:.0%}")
            result["confidence_scores"]["geometry"] = 0.80

        except Exception as e:
            logger.warning(f"  Geometry calculation failed: {e}")
            # Fallback to simple calculation
            result["geometry"] = calculate_simple_geometry(result)
    else:
        result["geometry"] = calculate_simple_geometry(result)

    # Set defaults
    result["construction_year"] = result.get("construction_year") or 1970
    result["atemp_m2"] = result.get("atemp_m2") or 2000
    result["declared_kwh_m2"] = result.get("declared_kwh_m2") or 100
    result["num_floors"] = result.get("num_floors") or 4
    result["num_apartments"] = result.get("num_apartments") or 50
    result["facade_material"] = result.get("ai_analysis", {}).get("facade_material") or result.get("facade_material") or "concrete"

    # Calculate overall confidence
    result["confidence"] = sum(result["confidence_scores"].values()) / max(len(result["confidence_scores"]), 1)

    logger.info(f"  Data sources: {', '.join(result['data_sources'])}")
    logger.info(f"  Overall confidence: {result['confidence']:.0%}")

    return result


def analyze_facade_images(images: Dict, config: PipelineConfig) -> Dict:
    """
    Run AI analysis on facade images.

    Returns:
    - wwr_by_orientation: Dict[str, float] - Window-to-wall ratio per direction
    - facade_material: str - Detected material (brick, concrete, plaster, etc.)
    - material_confidence: float - Detection confidence
    """
    logger.info("  Running AI facade analysis...")

    result = {
        "wwr_by_orientation": {},
        "facade_material": None,
        "material_confidence": 0.0,
        "material_votes": {},
    }

    try:
        from src.ai.wwr_detector import WWRDetector
        from src.ai.material_classifier import MaterialClassifier
        from PIL import Image
        import requests
        from io import BytesIO

        wwr_detector = WWRDetector(backend="opencv")
        material_classifier = MaterialClassifier()

        material_votes = {}

        for direction, facade_images in images.items():
            if not facade_images:
                continue

            direction_wwrs = []

            for img_data in facade_images[:3]:  # Top 3 images per direction
                try:
                    # Fetch image
                    if hasattr(img_data, 'url'):
                        response = requests.get(img_data.url, timeout=10)
                        pil_image = Image.open(BytesIO(response.content))
                    else:
                        continue

                    # WWR detection - returns (wwr, confidence) tuple
                    wwr, wwr_confidence = wwr_detector.calculate_wwr(pil_image)
                    if wwr > 0 and wwr_confidence > 0.3:
                        direction_wwrs.append(wwr)
                        logger.debug(f"    WWR: {wwr:.2f} (conf: {wwr_confidence:.2f})")

                    # Material classification
                    material_result = material_classifier.classify(pil_image)
                    if material_result.confidence > 0.5:
                        material = material_result.material.value
                        material_votes[material] = material_votes.get(material, 0) + material_result.confidence

                except Exception as e:
                    logger.debug(f"  Image analysis failed: {e}")
                    continue

            # Average WWR for this direction
            if direction_wwrs:
                result["wwr_by_orientation"][direction] = sum(direction_wwrs) / len(direction_wwrs)

        # Determine facade material by weighted voting
        if material_votes:
            result["facade_material"] = max(material_votes, key=material_votes.get)
            total_votes = sum(material_votes.values())
            result["material_confidence"] = material_votes[result["facade_material"]] / total_votes
            result["material_votes"] = material_votes

        # Fill in missing orientations with average
        if result["wwr_by_orientation"]:
            avg_wwr = sum(result["wwr_by_orientation"].values()) / len(result["wwr_by_orientation"])
            for d in ["N", "S", "E", "W"]:
                if d not in result["wwr_by_orientation"]:
                    result["wwr_by_orientation"][d] = avg_wwr

        logger.info(f"  ✓ AI Analysis: material={result['facade_material']} ({result['material_confidence']:.0%}), "
                   f"WWR={result['wwr_by_orientation']}")

    except ImportError as e:
        logger.warning(f"  AI modules not available: {e}")
    except Exception as e:
        logger.warning(f"  AI analysis failed: {e}")

    return result


def analyze_streetview_images(images: Dict, config: PipelineConfig) -> Dict:
    """
    Run AI analysis on Google Street View images.

    Street View images are already PIL Images (not URLs), so simpler processing.

    Returns:
    - wwr_by_orientation: Dict[str, float] - Window-to-wall ratio per direction
    - facade_material: str - Detected material
    - material_confidence: float
    """
    logger.info("  Running AI analysis on Street View images...")

    result = {
        "wwr_by_orientation": {},
        "facade_material": None,
        "material_confidence": 0.0,
        "material_votes": {},
        "source": "google_streetview",
    }

    try:
        from src.ai.wwr_detector import WWRDetector
        from src.ai.material_classifier import MaterialClassifier

        wwr_detector = WWRDetector(backend="opencv")
        material_classifier = MaterialClassifier()

        material_votes = {}

        for direction, sv_image in images.items():
            if not sv_image or not hasattr(sv_image, 'image'):
                continue

            pil_image = sv_image.image  # StreetViewImage has .image attribute

            try:
                # WWR detection - returns (wwr, confidence) tuple
                wwr, wwr_confidence = wwr_detector.calculate_wwr(pil_image)
                if wwr > 0 and wwr_confidence > 0.3:
                    result["wwr_by_orientation"][direction] = wwr
                    logger.debug(f"    {direction} WWR: {wwr:.2f} (conf: {wwr_confidence:.2f})")

                # Material classification
                material_result = material_classifier.classify(pil_image)
                if material_result.confidence > 0.4:
                    material = material_result.material.value
                    material_votes[material] = material_votes.get(material, 0) + material_result.confidence

            except Exception as e:
                logger.debug(f"    {direction} analysis failed: {e}")
                continue

        # Determine facade material by weighted voting
        if material_votes:
            result["facade_material"] = max(material_votes, key=material_votes.get)
            total_votes = sum(material_votes.values())
            result["material_confidence"] = material_votes[result["facade_material"]] / total_votes
            result["material_votes"] = material_votes

        # Fill in missing orientations with average
        if result["wwr_by_orientation"]:
            avg_wwr = sum(result["wwr_by_orientation"].values()) / len(result["wwr_by_orientation"])
            for d in ["N", "S", "E", "W"]:
                if d not in result["wwr_by_orientation"]:
                    result["wwr_by_orientation"][d] = avg_wwr

        logger.info(f"  ✓ Street View AI: material={result['facade_material']} ({result['material_confidence']:.0%}), "
                   f"WWR={result['wwr_by_orientation']}")

    except ImportError as e:
        logger.warning(f"  AI modules not available: {e}")
    except Exception as e:
        logger.warning(f"  Street View analysis failed: {e}")

    return result


def calculate_simple_geometry(building_data: Dict) -> Dict:
    """Fallback geometry calculation when footprint not available."""
    atemp_m2 = building_data.get("atemp_m2", 2000)
    num_floors = building_data.get("num_floors", 4)
    footprint_m2 = building_data.get("footprint_area_m2", atemp_m2 / num_floors)

    # Assume rectangular building
    perimeter_m = 4 * math.sqrt(footprint_m2)
    floor_height_m = 2.8
    facade_area_m2 = perimeter_m * floor_height_m * num_floors

    # Default WWR
    wwr = building_data.get("ai_analysis", {}).get("wwr_by_orientation", {})
    avg_wwr = sum(wwr.values()) / len(wwr) if wwr else 0.20

    window_area_m2 = facade_area_m2 * avg_wwr

    return {
        "facade_areas": {"N": facade_area_m2/4, "S": facade_area_m2/4, "E": facade_area_m2/4, "W": facade_area_m2/4},
        "window_areas": {"N": window_area_m2/4, "S": window_area_m2/4, "E": window_area_m2/4, "W": window_area_m2/4},
        "total_facade_area_m2": facade_area_m2,
        "total_window_area_m2": window_area_m2,
        "perimeter_m": perimeter_m,
        "compactness_ratio": 1.0,
        "building_form": "unknown",
    }


# ============================================================================
# STEP 2: ADVANCED ARCHETYPE MATCHING
# ============================================================================

def match_archetype_v2(building_data: Dict, config: PipelineConfig) -> Tuple[str, float, Dict]:
    """
    Archetype matching using year + energy data + AI enhancements.

    Scoring from:
    1. Energy declaration (year, energy class, ventilation)
    2. AI facade analysis (material, WWR) - enhances confidence
    3. Geometry (building form) - adjusts thermal bridges
    """
    logger.info("STEP 2: Archetype matching")

    year = building_data.get("construction_year", 1970)
    energy_class = building_data.get("energy_class")
    has_ftx = building_data.get("has_ftx", False)
    has_ft = building_data.get("has_ft", False)
    facade_material = building_data.get("ai_analysis", {}).get("facade_material") or building_data.get("facade_material")

    # Era-based archetype selection
    if year >= 2011:
        archetype_id = "mfh_2011_plus"
        envelope = {"wall_u_value": 0.18, "roof_u_value": 0.13, "window_u_value": 0.9, "floor_u_value": 0.15}
    elif year >= 1996:
        archetype_id = "mfh_1996_2010"
        envelope = {"wall_u_value": 0.25, "roof_u_value": 0.15, "window_u_value": 1.2, "floor_u_value": 0.20}
    elif year >= 1986:
        archetype_id = "mfh_1986_1995"
        envelope = {"wall_u_value": 0.30, "roof_u_value": 0.18, "window_u_value": 1.6, "floor_u_value": 0.25}
    elif year >= 1976:
        archetype_id = "mfh_1976_1985"
        envelope = {"wall_u_value": 0.35, "roof_u_value": 0.22, "window_u_value": 2.0, "floor_u_value": 0.30}
    elif year >= 1961:
        archetype_id = "mfh_1961_1975"
        envelope = {"wall_u_value": 0.50, "roof_u_value": 0.30, "window_u_value": 2.5, "floor_u_value": 0.40}
    elif year >= 1945:
        archetype_id = "mfh_1945_1960"
        envelope = {"wall_u_value": 0.65, "roof_u_value": 0.40, "window_u_value": 2.8, "floor_u_value": 0.50}
    else:
        archetype_id = "mfh_pre_1945"
        envelope = {"wall_u_value": 0.80, "roof_u_value": 0.50, "window_u_value": 3.0, "floor_u_value": 0.60}

    # Set ventilation heat recovery
    if has_ftx:
        envelope["heat_recovery_eff"] = 0.75
    elif has_ft:
        envelope["heat_recovery_eff"] = 0.0  # FT has no heat recovery
    else:
        envelope["heat_recovery_eff"] = 0.0

    # Adjust infiltration based on energy class
    if energy_class in ["A", "B"]:
        envelope["infiltration_ach"] = 0.03
    elif energy_class in ["C", "D"]:
        envelope["infiltration_ach"] = 0.05
    else:
        envelope["infiltration_ach"] = 0.07

    # Confidence scoring
    confidence = 0.50  # Base

    # Bonus for having energy declaration data
    if "sweden_geojson" in building_data.get("data_sources", []):
        confidence += 0.30

    # Bonus for AI analysis
    ai_analysis = building_data.get("ai_analysis", {})
    if ai_analysis.get("facade_material"):
        confidence += 0.10
    if ai_analysis.get("wwr_by_orientation"):
        confidence += 0.10

    # Material-specific adjustments
    if facade_material == "brick" and year < 1960:
        envelope["wall_u_value"] *= 0.9  # Brick has better thermal mass

    logger.info(f"  ✓ Matched: {archetype_id} (year={year}, class={energy_class})")
    logger.info(f"  Envelope: wall_U={envelope['wall_u_value']:.2f}, HR={envelope['heat_recovery_eff']:.0%}")
    logger.info(f"  Confidence: {confidence:.0%}")

    return archetype_id, min(confidence, 1.0), envelope


# ============================================================================
# STEP 3-4: IDF GENERATION AND BAYESIAN CALIBRATION
# ============================================================================

def run_bayesian_calibration(
    idf_path: Path,
    building_data: Dict,
    envelope: Dict,
    weather_path: Path,
    output_dir: Path,
    config: PipelineConfig,
) -> Tuple[Dict, float, Dict]:
    """
    Run full Bayesian ABC-SMC calibration if enabled, otherwise simple calibration.

    Returns:
    - calibrated_params: Dict of calibrated envelope parameters
    - calibrated_kwh_m2: Simulated heating after calibration
    - calibration_report: Dict with calibration quality metrics
    """
    logger.info("STEP 4: Calibration")

    atemp_m2 = building_data.get("atemp_m2", 2000)

    # Get actual heating target
    actual_heating_kwh, heating_sources = get_actual_heating_kwh(building_data)

    if actual_heating_kwh > 0:
        target_heating_kwh_m2 = actual_heating_kwh / atemp_m2
        logger.info(f"  Target: {target_heating_kwh_m2:.1f} kWh/m² (from {heating_sources})")
    else:
        target_heating_kwh_m2 = building_data.get("declared_kwh_m2", 100) * 0.5
        logger.info(f"  Target: {target_heating_kwh_m2:.1f} kWh/m² (estimated from primary energy)")

    calibration_report = {
        "method": "simple",
        "target_kwh_m2": target_heating_kwh_m2,
        "iterations": 1,
    }

    # === Full Bayesian Calibration ===
    if config.use_bayesian_calibration:
        try:
            calibrated_params, calibrated_kwh_m2, calibration_report = run_abc_smc_calibration(
                idf_path=idf_path,
                building_data=building_data,
                envelope=envelope,
                target_kwh_m2=target_heating_kwh_m2,
                weather_path=weather_path,
                output_dir=output_dir,
                config=config,
            )
            return calibrated_params, calibrated_kwh_m2, calibration_report

        except Exception as e:
            logger.warning(f"  Bayesian calibration failed, using simple: {e}")

    # === Simple Calibration Fallback ===
    baseline_kwh = run_single_simulation(idf_path, weather_path, output_dir / "calibration_baseline")
    baseline_kwh_m2 = baseline_kwh / atemp_m2 if atemp_m2 > 0 else 0

    gap_percent = ((baseline_kwh_m2 - target_heating_kwh_m2) / target_heating_kwh_m2 * 100) if target_heating_kwh_m2 > 0 else 0

    logger.info(f"  Simulated: {baseline_kwh_m2:.1f} kWh/m² (gap: {gap_percent:+.1f}%)")

    calibrated_params = envelope.copy()

    if abs(gap_percent) > 10:
        logger.info("  Adjusting parameters...")

        if gap_percent > 0:
            calibrated_params["infiltration_ach"] *= 0.7
            calibrated_params["window_u_value"] *= 0.9
            if calibrated_params.get("heat_recovery_eff", 0) > 0:
                calibrated_params["heat_recovery_eff"] = min(0.85, calibrated_params["heat_recovery_eff"] * 1.1)
        else:
            calibrated_params["infiltration_ach"] *= 1.3
            calibrated_params["window_u_value"] *= 1.1

        # Apply and re-run
        apply_params_to_idf(idf_path, calibrated_params)
        calibrated_kwh = run_single_simulation(idf_path, weather_path, output_dir / "calibration_final")
        calibrated_kwh_m2 = calibrated_kwh / atemp_m2

        new_gap = ((calibrated_kwh_m2 - target_heating_kwh_m2) / target_heating_kwh_m2 * 100) if target_heating_kwh_m2 > 0 else 0
        logger.info(f"  After calibration: {calibrated_kwh_m2:.1f} kWh/m² (gap: {new_gap:+.1f}%)")

        calibration_report["final_gap_percent"] = new_gap
    else:
        calibrated_kwh_m2 = baseline_kwh_m2
        calibration_report["final_gap_percent"] = gap_percent

    return calibrated_params, calibrated_kwh_m2, calibration_report


def run_abc_smc_calibration(
    idf_path: Path,
    building_data: Dict,
    envelope: Dict,
    target_kwh_m2: float,
    weather_path: Path,
    output_dir: Path,
    config: PipelineConfig,
) -> Tuple[Dict, float, Dict]:
    """
    Full Bayesian calibration using BayesianCalibrationPipeline.

    Uses Morris screening for parameter selection and ABC-SMC for inference.
    """
    logger.info("  Running Bayesian calibration pipeline...")

    from src.calibration.pipeline import BayesianCalibrationPipeline
    from src.simulation.runner import SimulationRunner

    # Initialize runner and pipeline
    runner = SimulationRunner(weather_path=weather_path)

    pipeline = BayesianCalibrationPipeline(
        runner=runner,
        weather_path=weather_path,
        cache_dir=output_dir / "surrogate_cache",
        n_surrogate_samples=config.calibration_samples,
        n_abc_particles=500,
        n_abc_generations=config.calibration_iterations,
        use_adaptive_calibration=True,  # Uses Morris screening
    )

    # Get archetype ID for caching
    archetype_id = building_data.get("archetype_id", "mfh_1996_2010")

    # Run calibration
    result = pipeline.calibrate(
        baseline_idf=idf_path,
        archetype_id=archetype_id,
        measured_kwh_m2=target_kwh_m2,
        atemp_m2=building_data.get("atemp_m2", 2000),
        has_ftx=building_data.get("has_ftx", False),
        energy_class=building_data.get("energy_class"),
        construction_year=building_data.get("construction_year", 1970),
    )

    calibrated_params = result.calibrated_params
    calibrated_kwh_m2 = result.calibrated_kwh_m2
    final_gap = result.calibration_error / target_kwh_m2 * 100 if target_kwh_m2 > 0 else 0

    calibration_report = {
        "method": "ABC-SMC + Morris",
        "target_kwh_m2": target_kwh_m2,
        "calibrated_kwh_m2": calibrated_kwh_m2,
        "final_gap_percent": final_gap,
        "surrogate_train_r2": result.surrogate_r2,
        "surrogate_test_r2": result.surrogate_test_r2,
        "surrogate_is_overfit": result.surrogate_is_overfit,
        "ashrae_nmbe": result.ashrae_nmbe,
        "ashrae_cvrmse": result.ashrae_cvrmse,
        "ashrae_passes": result.ashrae_passes,
        "calibrated_params": list(result.calibrated_param_list or []),
        "fixed_params": result.fixed_param_values,
        "parameter_uncertainties": {
            name: {
                "mean": result.calibrated_params[name],
                "std": result.param_stds.get(name, 0),
            }
            for name in result.calibrated_params
        },
    }

    logger.info(f"  ✓ Bayesian calibration: {calibrated_kwh_m2:.1f} kWh/m² (gap: {final_gap:+.1f}%)")
    if result.ashrae_passes is not None:
        logger.info(f"    ASHRAE: NMBE={result.ashrae_nmbe:.1f}%, CVRMSE={result.ashrae_cvrmse:.1f}%, passes={result.ashrae_passes}")

    return calibrated_params, calibrated_kwh_m2, calibration_report


def apply_params_to_idf(idf_path: Path, params: Dict):
    """Apply calibrated parameters to IDF file."""
    try:
        from eppy.modeleditor import IDF
        from src.core.idf_parser import IDFParser

        # Set IDD file path (required by eppy)
        idd_paths = [
            os.environ.get("ENERGYPLUS_IDD_PATH"),
            "/Applications/EnergyPlus-25-1-0/Energy+.idd",
            "/usr/local/bin/Energy+.idd",
            "/usr/local/EnergyPlus-25-1-0/Energy+.idd",
            Path.home() / "EnergyPlus-25-1-0" / "Energy+.idd",
        ]

        idd_path = None
        for p in idd_paths:
            if p and Path(p).exists():
                idd_path = str(p)
                break

        if idd_path:
            try:
                IDF.setiddname(idd_path)
            except Exception:
                pass  # Already set
        else:
            logger.warning("  IDD file not found - parameter application may fail")

        idf = IDF(str(idf_path))
        parser = IDFParser()

        if "infiltration_ach" in params:
            parser.set_infiltration_ach(idf, params["infiltration_ach"])
        if "window_u_value" in params:
            parser.set_window_u_value(idf, params["window_u_value"])
        if "wall_u_value" in params:
            parser.set_wall_u_value(idf, params["wall_u_value"])
        if "roof_u_value" in params:
            parser.set_roof_u_value(idf, params["roof_u_value"])
        if "heat_recovery_eff" in params and params["heat_recovery_eff"] > 0:
            parser.set_heat_recovery_effectiveness(idf, params["heat_recovery_eff"])
        if "heating_setpoint" in params:
            parser.set_heating_setpoint(idf, params["heating_setpoint"])

        idf.save()

    except Exception as e:
        logger.warning(f"  Failed to apply params: {e}")


# ============================================================================
# UTILITY FUNCTIONS (reused from V1)
# ============================================================================

# ============================================================================
# STEP 5-7: PACKAGE GENERATION AND SIMULATION
# ============================================================================

def detect_existing_measures(building_data: Dict) -> Dict[str, bool]:
    """
    Detect what ECMs are already implemented in the building.
    Used to skip redundant recommendations.
    """
    return {
        "has_ftx": building_data.get("has_ftx", False),
        "has_solar_pv": building_data.get("has_solar_pv", False),
        "has_heat_pump": building_data.get("has_heat_pump", False),
        "has_led": False,  # Assume not unless stated
        "has_smart_thermostats": False,
        "heat_recovery_eff": building_data.get("calibrated_params", {}).get("heat_recovery_eff", 0) or 0,
        "window_u_value": building_data.get("calibrated_params", {}).get("window_u_value", 2.0) or 2.0,
        "energy_class": building_data.get("energy_class", "D"),
        "solar_production_kwh": building_data.get("solar_production_kwh", 0),
    }


def create_ecm_packages(
    building_data: Dict,
    envelope: Dict,
    calibrated_kwh_m2: float,
) -> List[Dict]:
    """
    Create ECM packages with smart filtering for existing measures.

    Packages:
    - Steg 0: Nollkostnad (setpoint, schedules)
    - Steg 0.5: Effektvakt (peak shaving)
    - Steg 1: LED + smart thermostats
    - Steg 2: Window replacement (if old windows)
    - Steg 3: FTX upgrade/installation (if no FTX or old FTX)
    - Steg 4: Deep renovation (facade + roof)
    - Steg 5: Solar PV expansion (if roof potential > existing)
    """
    packages = []
    atemp_m2 = building_data.get("atemp_m2", 2000)
    has_ftx = building_data.get("has_ftx", False)
    energy_price, cop = get_energy_price_and_cop(building_data)

    # Detect existing measures
    existing = detect_existing_measures(building_data)
    logger.info(f"  Existing measures: FTX={existing['has_ftx']}, Solar={existing['has_solar_pv']}, "
               f"HP={existing['has_heat_pump']}, Energy class={existing['energy_class']}")

    # Calculate thermal inertia for effektvakt
    thermal_inertia = calculate_thermal_inertia(building_data)
    effektvakt = calculate_effektvakt_savings(building_data, thermal_inertia)

    # Steg 0: Nollkostnad (Zero cost measures)
    packages.append({
        "id": "steg_0",
        "name_sv": "Steg 0: Nollkostnad",
        "name_en": "Step 0: Zero Cost",
        "description_sv": "Sänk innetemperatur till 20°C, optimera drifttider",
        "ecms": ["heating_setpoint_reduction", "schedule_optimization"],
        "params": {
            **envelope,
            "heating_setpoint": 20.0,  # Lower from 21°C
        },
        "cost_sek": 0,
        "estimated_savings_percent": 5,
    })

    # Steg 0.5: Effektvakt
    packages.append({
        "id": "steg_0_5",
        "name_sv": "Steg 0.5: Effektvakt",
        "name_en": "Step 0.5: Peak Shaving",
        "description_sv": effektvakt["strategy"],
        "ecms": ["effektvakt"],
        "params": envelope.copy(),
        "cost_sek": effektvakt["investment_sek"],
        "effektvakt": effektvakt,
        "estimated_savings_percent": 0,  # No energy savings, only peak
        "peak_savings_sek": effektvakt["annual_savings_sek"],
    })

    # Steg 1: LED + Smart Thermostats (using V2 cost calculator)
    # LED only covers common areas (stairwells, corridors, garage) - typically 5-8% of atemp
    num_apartments = building_data.get("num_apartments", 20)
    common_area_m2 = max(
        atemp_m2 * 0.07,  # 7% of total area for common spaces
        num_apartments * 5,  # Or at least 5 m² per apartment (stairwells)
    )
    led_cost, led_breakdown = calculate_ecm_cost_v2(
        "led_lighting", common_area_m2, building_data,
        fallback_cost=ECM_COSTS["led_common_areas"] * num_apartments
    )
    # Thermostats: V2 model uses SEK/m² floor, but only for apartment areas (not common)
    # Use per-apartment fallback which is more accurate (~1,500 SEK/apt installed)
    apartment_area_m2 = atemp_m2 * 0.93  # 93% is apartment area (excl. common)
    thermostat_cost, thermo_breakdown = calculate_ecm_cost_v2(
        "smart_thermostats", apartment_area_m2, building_data,
        fallback_cost=ECM_COSTS["smart_thermostats"] * num_apartments
    )
    packages.append({
        "id": "steg_1",
        "name_sv": "Steg 1: LED + Termostater",
        "name_en": "Step 1: LED + Thermostats",
        "description_sv": "LED i trapphus, smarta radiatortermostater",
        "ecms": ["led_lighting", "smart_thermostats"],
        "params": {
            **envelope,
            "heating_setpoint": 20.0,
        },
        "cost_sek": led_cost + thermostat_cost,
        "cost_breakdown": {"led": led_breakdown, "thermostats": thermo_breakdown},
        "estimated_savings_percent": 8,
    })

    # Steg 2: Window Replacement (using V2 cost calculator)
    window_area = building_data.get("geometry", {}).get("total_window_area_m2", atemp_m2 * 0.15)
    window_cost, window_breakdown = calculate_ecm_cost_v2(
        "window_replacement", window_area, building_data,
        fallback_cost=ECM_COSTS["window_replacement_energy"]
    )
    new_window_u = 0.9  # Triple glazed
    packages.append({
        "id": "steg_2",
        "name_sv": "Steg 2: Fönsterbyte",
        "name_en": "Step 2: Window Replacement",
        "description_sv": f"Energifönster (U={new_window_u} W/m²K), {window_area:.0f} m²",
        "ecms": ["window_replacement"],
        "params": {
            **envelope,
            "window_u_value": new_window_u,
            "heating_setpoint": 20.0,
        },
        "cost_sek": window_cost,
        "cost_breakdown": window_breakdown,
        "estimated_savings_percent": 12,
    })

    # Steg 3: FTX upgrade or installation (skip if already high-efficiency FTX)
    current_hr_eff = envelope.get("heat_recovery_eff", 0)
    if has_ftx and current_hr_eff >= 0.80:
        # Already has high-efficiency FTX - skip this package
        logger.info(f"  Skipping FTX upgrade: already has {current_hr_eff:.0%} heat recovery")
        ftx_cost = 0
        packages.append({
            "id": "steg_3",
            "name_sv": "Steg 3: Ventilation (EJ AKTUELLT)",
            "name_en": "Step 3: Ventilation (N/A)",
            "description_sv": f"Redan FTX med {current_hr_eff:.0%} värmeåtervinning - ingen åtgärd",
            "ecms": [],
            "params": envelope.copy(),
            "cost_sek": 0,
            "estimated_savings_percent": 0,
            "skipped": True,
            "skip_reason": "already_has_high_eff_ftx",
        })
    elif has_ftx:
        ftx_cost, ftx_breakdown = calculate_ecm_cost_v2(
            "ftx_upgrade", atemp_m2, building_data,
            fallback_cost=ECM_COSTS["ftx_heat_exchanger_upgrade"]
        )
        new_hr_eff = min(0.88, current_hr_eff + 0.10)
        ftx_desc = f"Uppgradering till {new_hr_eff:.0%} värmeåtervinning"
        packages.append({
            "id": "steg_3",
            "name_sv": "Steg 3: Ventilation",
            "name_en": "Step 3: Ventilation Upgrade",
            "description_sv": ftx_desc,
            "ecms": ["ftx_upgrade"],
            "params": {
                **envelope,
                "heat_recovery_eff": new_hr_eff,
                "heating_setpoint": 20.0,
            },
            "cost_sek": ftx_cost,
            "cost_breakdown": ftx_breakdown,
            "estimated_savings_percent": 8,
        })
    else:
        ftx_cost, ftx_breakdown = calculate_ecm_cost_v2(
            "ftx_installation", atemp_m2, building_data,
            fallback_cost=ECM_COSTS["ftx_new_installation"]
        )
        new_hr_eff = 0.82
        ftx_desc = "Installation av FTX-system med 82% värmeåtervinning"
        packages.append({
            "id": "steg_3",
            "name_sv": "Steg 3: Ventilation",
            "name_en": "Step 3: Ventilation Upgrade",
            "description_sv": ftx_desc,
            "ecms": ["ftx_installation"],
            "params": {
                **envelope,
                "heat_recovery_eff": new_hr_eff,
                "heating_setpoint": 20.0,
            },
            "cost_sek": ftx_cost,
            "cost_breakdown": ftx_breakdown,
            "estimated_savings_percent": 20,
        })

    # Steg 4: Deep renovation (facade + roof + windows + FTX)
    facade_area = building_data.get("geometry", {}).get("total_facade_area_m2", atemp_m2 * 0.8)
    roof_area = building_data.get("roof_analysis", {}).get("total_area_m2", atemp_m2 / building_data.get("num_floors", 4))

    # Use V2 cost calculator for accurate BeBo-sourced costs
    facade_cost, facade_breakdown = calculate_ecm_cost_v2(
        "wall_external_insulation", facade_area, building_data,
        fallback_cost=ECM_COSTS["facade_insulation_100mm"]
    )
    roof_cost, roof_breakdown = calculate_ecm_cost_v2(
        "roof_insulation", roof_area, building_data,
        fallback_cost=ECM_COSTS["roof_insulation_200mm"]
    )

    deep_cost = facade_cost + roof_cost + window_cost + ftx_cost

    packages.append({
        "id": "steg_4",
        "name_sv": "Steg 4: Totalrenovering",
        "name_en": "Step 4: Deep Renovation",
        "description_sv": "Tilläggsisolering fasad, tak, fönsterbyte, FTX",
        "ecms": ["wall_external_insulation", "roof_insulation", "window_replacement", "ftx_upgrade"],
        "params": {
            "wall_u_value": 0.18,
            "roof_u_value": 0.10,
            "window_u_value": 0.9,
            "floor_u_value": envelope.get("floor_u_value", 0.20),
            "heat_recovery_eff": 0.88,
            "infiltration_ach": 0.03,
            "heating_setpoint": 20.0,
        },
        "cost_sek": deep_cost,
        "cost_breakdown": {
            "facade": facade_breakdown,
            "roof": roof_breakdown,
            "windows": window_breakdown,
            "ftx": ftx_breakdown if 'ftx_breakdown' in dir() else None,
        },
        "estimated_savings_percent": 45,
    })

    # Steg 5: Solar PV expansion (if roof has more potential than existing)
    existing_solar_kwh = building_data.get("solar_production_kwh", 0)
    roof_pv_potential_kwp = building_data.get("roof_analysis", {}).get("pv_potential_kwp", 0)
    roof_annual_yield = building_data.get("roof_analysis", {}).get("annual_yield_kwh", 0)

    # Estimate existing capacity from production (assuming 900 kWh/kWp in Stockholm)
    existing_capacity_kwp = existing_solar_kwh / 900 if existing_solar_kwh > 0 else 0
    expansion_potential_kwp = max(0, roof_pv_potential_kwp - existing_capacity_kwp)

    if expansion_potential_kwp > 10:  # At least 10 kWp expansion worthwhile
        expansion_kwh = expansion_potential_kwp * 900  # Stockholm yield

        # Use V2 cost calculator for accurate solar PV costs
        expansion_cost, pv_breakdown = calculate_ecm_cost_v2(
            "solar_pv", expansion_potential_kwp, building_data,
            fallback_cost=12000  # SEK/kWp fallback
        )

        # Electricity savings (self-consumption + export)
        property_el_kwh = building_data.get("property_electricity_kwh", 0)
        self_consumption_rate = min(0.7, property_el_kwh / expansion_kwh) if expansion_kwh > 0 else 0.5
        el_price_self = ElectricityPricing().total_energy_price  # ~2 SEK/kWh
        el_price_export = 0.50  # SEK/kWh for sold electricity

        annual_pv_savings = (
            expansion_kwh * self_consumption_rate * el_price_self +
            expansion_kwh * (1 - self_consumption_rate) * el_price_export
        )

        packages.append({
            "id": "steg_5",
            "name_sv": "Steg 5: Solcellsutbyggnad",
            "name_en": "Step 5: Solar PV Expansion",
            "description_sv": f"Utöka solceller med {expansion_potential_kwp:.0f} kWp ({expansion_kwh/1000:.0f} MWh/år)",
            "ecms": ["solar_pv"],
            "params": envelope.copy(),  # No heating impact
            "cost_sek": expansion_cost,
            "cost_breakdown": pv_breakdown,
            "estimated_savings_percent": 0,  # Not heating savings
            "annual_savings_sek": annual_pv_savings,
            "pv_expansion_kwp": expansion_potential_kwp,
            "pv_expansion_kwh": expansion_kwh,
            "existing_pv_kwp": existing_capacity_kwp,
            "is_electricity_savings": True,
        })
        logger.info(f"  Added solar expansion: +{expansion_potential_kwp:.0f} kWp, {annual_pv_savings:,.0f} SEK/year")
    elif existing_solar_kwh > 0:
        packages.append({
            "id": "steg_5",
            "name_sv": "Steg 5: Solceller (EJ AKTUELLT)",
            "name_en": "Step 5: Solar PV (N/A)",
            "description_sv": f"Redan {existing_capacity_kwp:.0f} kWp installerat, taket fullt utnyttjat",
            "ecms": [],
            "params": envelope.copy(),
            "cost_sek": 0,
            "estimated_savings_percent": 0,
            "skipped": True,
            "skip_reason": "roof_fully_utilized",
        })

    return packages


def simulate_packages(
    packages: List[Dict],
    baseline_idf: Path,
    weather_path: Path,
    output_dir: Path,
    building_data: Dict,
    calibrated_kwh_m2: float,
) -> List[Dict]:
    """
    Simulate each package and calculate actual savings.
    """
    logger.info("STEP 6: Simulating ECM packages")

    atemp_m2 = building_data.get("atemp_m2", 2000)
    energy_price, cop = get_energy_price_and_cop(building_data)

    results = []

    for pkg in packages:
        pkg_id = pkg["id"]
        logger.info(f"  Simulating {pkg_id}: {pkg['name_sv']}")

        # Skip packages marked as skipped (already implemented)
        if pkg.get("skipped"):
            pkg["heating_kwh_m2"] = calibrated_kwh_m2
            pkg["savings_kwh_m2"] = 0
            pkg["savings_percent"] = 0
            pkg["annual_savings_sek"] = 0
            results.append(pkg)
            logger.info(f"    → Skipped: {pkg.get('skip_reason', 'already implemented')}")
            continue

        # Skip effektvakt simulation (no energy change, only peak)
        if pkg_id == "steg_0_5":
            pkg["heating_kwh_m2"] = calibrated_kwh_m2
            pkg["savings_kwh_m2"] = 0
            pkg["savings_percent"] = 0
            pkg["annual_savings_sek"] = pkg.get("peak_savings_sek", 0)
            results.append(pkg)
            continue

        # Skip solar PV (electricity savings, not heating)
        if pkg.get("is_electricity_savings"):
            pkg["heating_kwh_m2"] = calibrated_kwh_m2
            pkg["savings_kwh_m2"] = 0
            pkg["savings_percent"] = 0
            # annual_savings_sek already set
            results.append(pkg)
            logger.info(f"    → Electricity savings: {pkg.get('annual_savings_sek', 0):,.0f} SEK/year")
            continue

        # Copy and modify IDF
        pkg_idf = output_dir / f"pkg_{pkg_id}.idf"
        shutil.copy(baseline_idf, pkg_idf)
        apply_params_to_idf(pkg_idf, pkg["params"])

        # Run simulation
        pkg_dir = output_dir / f"sim_{pkg_id}"
        heating_kwh = run_single_simulation(pkg_idf, weather_path, pkg_dir)
        heating_kwh_m2 = heating_kwh / atemp_m2 if atemp_m2 > 0 else 0

        # Calculate savings
        savings_kwh_m2 = max(0, calibrated_kwh_m2 - heating_kwh_m2)
        savings_percent = (savings_kwh_m2 / calibrated_kwh_m2 * 100) if calibrated_kwh_m2 > 0 else 0

        # Annual energy cost savings
        annual_savings_kwh = savings_kwh_m2 * atemp_m2
        annual_savings_sek = annual_savings_kwh * energy_price / cop

        # Add peak savings for packages with effektvakt component
        if pkg.get("peak_savings_sek"):
            annual_savings_sek += pkg["peak_savings_sek"]

        pkg["heating_kwh_m2"] = heating_kwh_m2
        pkg["savings_kwh_m2"] = savings_kwh_m2
        pkg["savings_percent"] = savings_percent
        pkg["annual_savings_sek"] = annual_savings_sek

        logger.info(f"    → {heating_kwh_m2:.1f} kWh/m² (−{savings_percent:.0f}%)")

        results.append(pkg)

    return results


def calculate_package_roi(packages: List[Dict], building_data: Dict) -> List[Dict]:
    """
    Calculate ROI metrics for each package.
    """
    logger.info("STEP 7: Calculating ROI")

    for pkg in packages:
        cost = pkg.get("cost_sek", 0)
        annual_savings = pkg.get("annual_savings_sek", 0)

        if cost > 0 and annual_savings > 0:
            pkg["payback_years"] = cost / annual_savings

            # NPV calculation (20 year horizon, 4% discount rate)
            discount_rate = 0.04
            npv = -cost
            for year in range(1, 21):
                npv += annual_savings / ((1 + discount_rate) ** year)
            pkg["npv_sek"] = npv

            # Simple IRR approximation
            pkg["irr_percent"] = (annual_savings / cost) * 100 if cost > 0 else 0
        else:
            pkg["payback_years"] = 0 if cost == 0 else float('inf')
            pkg["npv_sek"] = 0
            pkg["irr_percent"] = 0

        logger.info(f"  {pkg['id']}: payback={pkg['payback_years']:.1f}y, NPV={pkg.get('npv_sek', 0)/1e6:.2f}M SEK")

    # Sort by payback (excluding zero-cost)
    packages.sort(key=lambda p: (p["payback_years"] if p["payback_years"] > 0 else 0.1, -p.get("npv_sek", 0)))

    return packages


# ============================================================================
# STEP 8-9: EFFEKTVAKT ANALYSIS AND HTML REPORT
# ============================================================================

def generate_html_report(
    output_dir: Path,
    address: str,
    building_data: Dict,
    packages: List[Dict],
    calibrated_kwh_m2: float,
    calibration_report: Dict,
    archetype_id: str,
) -> Path:
    """
    Generate Swedish board-ready HTML report.
    """
    logger.info("STEP 9: Generating HTML report")

    thermal_inertia = calculate_thermal_inertia(building_data)
    effektvakt = calculate_effektvakt_savings(building_data, thermal_inertia)

    # Try to use existing HTMLReportGenerator
    try:
        from src.reporting.html_report import HTMLReportGenerator

        generator = HTMLReportGenerator()

        # Build data structure expected by generator
        report_data = {
            "building": {
                "address": address,
                "construction_year": building_data.get("construction_year"),
                "atemp_m2": building_data.get("atemp_m2"),
                "num_apartments": building_data.get("num_apartments"),
                "num_floors": building_data.get("num_floors"),
                "energy_class": building_data.get("energy_class"),
                "declared_kwh_m2": building_data.get("declared_kwh_m2"),
                "heating_system": building_data.get("heating_system"),
                "ventilation_type": building_data.get("ventilation_type"),
                "has_ftx": building_data.get("has_ftx"),
                "facade_material": building_data.get("facade_material"),
            },
            "baseline": {
                "heating_kwh_m2": calibrated_kwh_m2,
                "archetype_id": archetype_id,
            },
            "calibration": calibration_report,
            "packages": packages,
            "thermal_inertia": thermal_inertia,
            "effektvakt": effektvakt,
            "data_sources": building_data.get("data_sources", []),
            "confidence": building_data.get("confidence", 0),
        }

        report_path = output_dir / "rapport.html"
        generator.generate(report_data, report_path)

        logger.info(f"  ✓ Report generated: {report_path}")
        return report_path

    except Exception as e:
        logger.warning(f"  HTMLReportGenerator failed: {e}, using simple report")
        return generate_simple_html_report(output_dir, address, building_data, packages, calibrated_kwh_m2, thermal_inertia, effektvakt)


def generate_simple_html_report(
    output_dir: Path,
    address: str,
    building_data: Dict,
    packages: List[Dict],
    calibrated_kwh_m2: float,
    thermal_inertia: Dict,
    effektvakt: Dict,
) -> Path:
    """
    Generate simple HTML report as fallback.
    """
    report_path = output_dir / "rapport.html"

    html = f"""<!DOCTYPE html>
<html lang="sv">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Energianalys - {address}</title>
    <style>
        body {{ font-family: 'Segoe UI', Arial, sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; background: #f5f5f5; }}
        .card {{ background: white; border-radius: 8px; padding: 24px; margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        h1 {{ color: #1a365d; margin-top: 0; }}
        h2 {{ color: #2c5282; border-bottom: 2px solid #3182ce; padding-bottom: 8px; }}
        table {{ width: 100%; border-collapse: collapse; margin: 16px 0; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #e2e8f0; }}
        th {{ background: #edf2f7; font-weight: 600; }}
        .highlight {{ background: #ebf8ff; font-weight: 600; }}
        .good {{ color: #276749; }}
        .warning {{ color: #c05621; }}
        .metric {{ display: inline-block; padding: 8px 16px; background: #e2e8f0; border-radius: 4px; margin: 4px; }}
        .metric-value {{ font-size: 1.5em; font-weight: 600; color: #2c5282; }}
        .footer {{ text-align: center; color: #718096; padding: 20px; font-size: 0.9em; }}
    </style>
</head>
<body>
    <div class="card">
        <h1>🏢 Energianalys</h1>
        <h2>{address}</h2>
        <p>Genererad: {time.strftime('%Y-%m-%d %H:%M')}</p>
    </div>

    <div class="card">
        <h2>📊 Byggnadsdata</h2>
        <table>
            <tr><td>Byggår</td><td><strong>{building_data.get('construction_year', 'Okänt')}</strong></td></tr>
            <tr><td>Atemp</td><td><strong>{building_data.get('atemp_m2', 0):,.0f} m²</strong></td></tr>
            <tr><td>Antal lägenheter</td><td><strong>{building_data.get('num_apartments', 0):.0f}</strong></td></tr>
            <tr><td>Energiklass</td><td><strong>{building_data.get('energy_class', 'Okänt')}</strong></td></tr>
            <tr><td>Deklarerad energi</td><td><strong>{building_data.get('declared_kwh_m2', 0):.0f} kWh/m²</strong></td></tr>
            <tr><td>Uppvärmning</td><td><strong>{building_data.get('heating_system', 'Okänt')}</strong></td></tr>
            <tr><td>Ventilation</td><td><strong>{building_data.get('ventilation_type', 'Okänt')}</strong></td></tr>
        </table>
    </div>

    <div class="card">
        <h2>🔋 Termisk tröghet & Effektvakt</h2>
        <div>
            <span class="metric"><span class="metric-value">{thermal_inertia.get('time_constant_hours', 0):.1f}h</span><br>Tidskonstant (τ)</span>
            <span class="metric"><span class="metric-value">{thermal_inertia.get('coast_time_1deg_design_hours', 0):.1f}h</span><br>Glidsträcka vid −5°C</span>
            <span class="metric"><span class="metric-value">{effektvakt.get('peak_reduction_kw', 0):.0f} kW</span><br>Effektreduktion</span>
            <span class="metric"><span class="metric-value">{effektvakt.get('annual_savings_sek', 0):,.0f} kr/år</span><br>Besparing</span>
        </div>
        <p><strong>Strategi:</strong> {effektvakt.get('strategy', '')}</p>
        <p><strong>Återbetalningstid:</strong> {effektvakt.get('payback_years', 0):.1f} år</p>
    </div>

    <div class="card">
        <h2>💰 Åtgärdspaket (rankade efter lönsamhet)</h2>
        <table>
            <tr>
                <th>Paket</th>
                <th>Beskrivning</th>
                <th>Investering</th>
                <th>Besparing/år</th>
                <th>Återbetalning</th>
                <th>NPV (20 år)</th>
            </tr>"""

    for pkg in packages:
        payback_class = "good" if pkg.get("payback_years", 99) < 10 else "warning"
        html += f"""
            <tr class="{'highlight' if pkg.get('payback_years', 99) < 5 else ''}">
                <td><strong>{pkg.get('name_sv', pkg['id'])}</strong></td>
                <td>{pkg.get('description_sv', '')}</td>
                <td>{pkg.get('cost_sek', 0):,.0f} kr</td>
                <td>{pkg.get('annual_savings_sek', 0):,.0f} kr</td>
                <td class="{payback_class}">{pkg.get('payback_years', 0):.1f} år</td>
                <td>{pkg.get('npv_sek', 0):,.0f} kr</td>
            </tr>"""

    html += f"""
        </table>
    </div>

    <div class="card">
        <h2>📈 Rekommendation</h2>
        <p>Baserat på analysen rekommenderas följande åtgärder i prioritetsordning:</p>
        <ol>"""

    for pkg in packages[:3]:
        if pkg.get("payback_years", 99) < 15:
            html += f"<li><strong>{pkg.get('name_sv', pkg['id'])}</strong> - {pkg.get('payback_years', 0):.1f} års återbetalningstid</li>"

    html += f"""
        </ol>
        <p>Total potentiell besparing med alla åtgärder: <strong>{sum(p.get('annual_savings_sek', 0) for p in packages):,.0f} kr/år</strong></p>
    </div>

    <div class="footer">
        <p>Rapport genererad av Raiden v2.0 | Data från: {', '.join(building_data.get('data_sources', []))}</p>
        <p>Konfidensgrad: {building_data.get('confidence', 0):.0%}</p>
    </div>
</body>
</html>"""

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html)

    logger.info(f"  ✓ Simple report generated: {report_path}")
    return report_path


def get_actual_heating_kwh(building_data: Dict) -> Tuple[float, str]:
    """Extract actual heating from energy declaration."""
    heat_pump_kwh = building_data.get("heat_pump_kwh", {})

    total_heating = 0.0
    sources = []

    for hp_type, kwh in heat_pump_kwh.items():
        if kwh > 0:
            total_heating += kwh
            sources.append(f"{hp_type}: {kwh/1000:.0f} MWh")

    district_kwh = building_data.get("district_heating_kwh", 0)
    if district_kwh > 0:
        total_heating += district_kwh
        sources.append(f"district: {district_kwh/1000:.0f} MWh")

    if total_heating == 0:
        total_energy = building_data.get("total_energy_kwh", 0)
        property_el = building_data.get("property_electricity_kwh", 0)
        hot_water_el = building_data.get("hot_water_electricity_kwh", 0)
        heating_estimate = max(0, total_energy - property_el - hot_water_el)
        if heating_estimate > 0:
            total_heating = heating_estimate
            sources.append(f"estimated: {heating_estimate/1000:.0f} MWh")

    return total_heating, ", ".join(sources) if sources else "unknown"


def run_single_simulation(idf_path: Path, weather_path: Path, output_dir: Path) -> float:
    """Run single E+ simulation, return heating kWh."""
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = ["energyplus", "-w", str(weather_path), "-d", str(output_dir), "-r", str(idf_path)]

    try:
        result = subprocess.run(cmd, capture_output=True, timeout=300)
        if result.returncode == 0:
            return parse_heating_kwh(output_dir)
    except Exception as e:
        logger.warning(f"  Simulation failed: {e}")

    return 0.0


def parse_heating_kwh(output_dir: Path) -> float:
    """Parse heating kWh from E+ output."""
    import re

    table_csv = output_dir / "eplustbl.csv"
    if table_csv.exists():
        with open(table_csv) as f:
            content = f.read()

            match = re.search(r'District Heating Water Intensity \[kWh/m2\],(\d+\.?\d*)', content)
            if match:
                intensity = float(match.group(1))
                area_match = re.search(r'Total Building Area,(\d+\.?\d*)', content)
                if area_match:
                    return intensity * float(area_match.group(1))

            lines = content.split('\n')
            for line in lines:
                if ',Heating,General,' in line or ',Heating,Unassigned,' in line:
                    parts = line.split(',')
                    for part in parts:
                        try:
                            val = float(part)
                            if val > 100:
                                return val
                        except:
                            pass

    return 0.0


def find_weather_file() -> Path:
    """Find Stockholm weather file."""
    paths = [
        PROJECT_ROOT / "tests" / "fixtures" / "stockholm.epw",
        PROJECT_ROOT / "weather" / "SWE_Stockholm.AP.024600_TMYx.epw",
    ]
    for p in paths:
        if p.exists():
            return p
    return paths[0]


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Raiden Production Pipeline V2")
    parser.add_argument("address", help="Building address")
    parser.add_argument("--year", type=int, help="Construction year")
    parser.add_argument("--atemp", type=float, help="Heated floor area (m²)")
    parser.add_argument("--output", type=Path, help="Output directory")
    parser.add_argument("--no-mapillary", action="store_true", help="Skip Mapillary images")
    parser.add_argument("--no-streetview", action="store_true", help="Skip Google Street View")
    parser.add_argument("--no-solar", action="store_true", help="Skip Google Solar API")
    parser.add_argument("--no-bayesian", action="store_true", help="Skip Bayesian calibration")
    parser.add_argument("--use-llm", action="store_true", help="Enable LLM reasoning")
    args = parser.parse_args()

    config = PipelineConfig(
        use_mapillary_images=not args.no_mapillary,
        use_google_streetview=not args.no_streetview,
        use_google_solar_api=not args.no_solar,
        use_bayesian_calibration=not args.no_bayesian,
        use_llm_reasoner=args.use_llm,
    )

    known_data = {}
    if args.year:
        known_data["construction_year"] = args.year
    if args.atemp:
        known_data["atemp_m2"] = args.atemp

    print("=" * 70)
    print("RAIDEN PRODUCTION PIPELINE V2")
    print("=" * 70)
    print(f"Address: {args.address}")
    print(f"Config: Mapillary={config.use_mapillary_images}, StreetView={config.use_google_streetview}, "
          f"Solar={config.use_google_solar_api}, Bayesian={config.use_bayesian_calibration}, LLM={config.use_llm_reasoner}")
    print()

    # Output directory
    output_dir = args.output
    if output_dir is None:
        safe_addr = args.address.replace(" ", "_").replace(",", "")[:30]
        output_dir = PROJECT_ROOT / "output_v2" / safe_addr
    output_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.time()

    # STEP 1: Comprehensive data fetch
    building_data = fetch_building_data_v2(args.address, config, known_data)

    # STEP 2: Advanced archetype matching
    archetype_id, arch_confidence, envelope = match_archetype_v2(building_data, config)

    # STEP 3: Generate baseline IDF
    logger.info("STEP 3: Generating baseline IDF")
    template_path = PROJECT_ROOT / "examples" / "sjostaden_2" / "energyplus" / "sjostaden_7zone.idf"
    baseline_idf = output_dir / "baseline.idf"

    if template_path.exists():
        shutil.copy(template_path, baseline_idf)
        apply_params_to_idf(baseline_idf, envelope)
        logger.info(f"  ✓ Generated: {baseline_idf.name}")
    else:
        logger.error(f"  Template IDF not found: {template_path}")
        return 1

    # STEP 4: Calibration
    weather_path = find_weather_file()
    calibrated_params, calibrated_kwh_m2, calibration_report = run_bayesian_calibration(
        baseline_idf, building_data, envelope, weather_path, output_dir, config
    )

    # STEP 5: Generate ECM packages
    logger.info("STEP 5: Generating ECM packages")
    packages = create_ecm_packages(building_data, calibrated_params, calibrated_kwh_m2)
    logger.info(f"  ✓ Generated {len(packages)} packages")

    # STEP 6: Simulate packages
    packages = simulate_packages(
        packages=packages,
        baseline_idf=baseline_idf,
        weather_path=weather_path,
        output_dir=output_dir,
        building_data=building_data,
        calibrated_kwh_m2=calibrated_kwh_m2,
    )

    # STEP 7: Calculate ROI
    packages = calculate_package_roi(packages, building_data)

    # STEP 8: Effektvakt analysis (already included in packages)
    logger.info("STEP 8: Effektvakt analysis")
    thermal_inertia = calculate_thermal_inertia(building_data)
    effektvakt = calculate_effektvakt_savings(building_data, thermal_inertia)
    logger.info(f"  ✓ Coast time: {effektvakt['coast_time_hours']:.1f}h, "
               f"Peak reduction: {effektvakt['peak_reduction_kw']:.0f} kW, "
               f"Savings: {effektvakt['annual_savings_sek']:,.0f} SEK/year")

    # STEP 9: Generate HTML report
    report_path = generate_html_report(
        output_dir=output_dir,
        address=args.address,
        building_data=building_data,
        packages=packages,
        calibrated_kwh_m2=calibrated_kwh_m2,
        calibration_report=calibration_report,
        archetype_id=archetype_id,
    )

    total_time = time.time() - start_time

    # Save results
    results = {
        "address": args.address,
        "building_data": building_data,
        "archetype_id": archetype_id,
        "archetype_confidence": arch_confidence,
        "calibrated_params": calibrated_params,
        "calibrated_kwh_m2": calibrated_kwh_m2,
        "calibration_report": calibration_report,
        "packages": packages,
        "thermal_inertia": thermal_inertia,
        "effektvakt": effektvakt,
        "report_path": str(report_path),
        "total_time_sec": total_time,
        "config": {
            "use_mapillary": config.use_mapillary_images,
            "use_solar": config.use_google_solar_api,
            "use_bayesian": config.use_bayesian_calibration,
            "use_llm": config.use_llm_reasoner,
        }
    }

    json_path = output_dir / "results_v2.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    # Print summary
    print()
    print("=" * 70)
    print("PIPELINE V2 COMPLETE")
    print("=" * 70)
    print(f"Total time: {total_time:.1f}s")
    print(f"Output: {output_dir}")
    print(f"Confidence: {building_data['confidence']:.0%}")
    print(f"Calibrated: {calibrated_kwh_m2:.1f} kWh/m² (gap: {calibration_report.get('final_gap_percent', 0):+.1f}%)")
    print()
    print("ECM PACKAGES (ranked by payback):")
    print("-" * 70)
    for pkg in packages:
        print(f"  {pkg['name_sv']:<30} | {pkg.get('savings_percent', 0):>5.1f}% savings | "
              f"{pkg.get('payback_years', 0):>5.1f}y payback | {pkg.get('annual_savings_sek', 0):>10,.0f} SEK/year")
    print()
    print(f"HTML Report: {report_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
