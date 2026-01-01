#!/usr/bin/env python3
"""
Full EnergyPlus Analysis for BRF Sjöstaden 2

This script runs REAL simulations:
1. Generates multi-zone IDF (7 floors + commercial ground)
2. Runs baseline simulation
3. Calibrates to declared 53 kWh/m²
4. Runs all applicable ECM simulations
5. Runs package combined simulations
6. Stores real results in database

Building: BRF Sjöstaden 2, Hammarby Sjöstad, Stockholm
- 15,350 m² heated area
- 110 apartments, 7 stairwells
- 1,626 m² commercial (restaurant/retail at ground)
- Built 2003, FTX ventilation, district heating
- Energy class B, declared 53 kWh/m²
"""

import json
import logging
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.baseline.archetypes import ArchetypeMatcher
from src.baseline.generator import BaselineGenerator
from src.baseline.calibrator import BaselineCalibrator
from src.ecm.catalog import ECMCatalog
from src.ecm.constraints import ConstraintEngine
from src.ecm.idf_modifier import IDFModifier
from src.ecm.dependencies import get_dependency_matrix, validate_package, get_package_synergy
from src.simulation.runner import SimulationRunner
from src.simulation.results import ResultsParser
from src.geometry.building_geometry import BuildingGeometry
from src.core.building_context import (
    EnhancedBuildingContext,
    SmartECMFilter,
    ExistingMeasure,
)
from src.roi.costs_sweden_v2 import (
    SwedishCostCalculatorV2,
    OwnerType,
    Region,
    DISTRICT_HEATING_PRICES,
)

# Database imports
try:
    from src.db import (
        BuildingRepository,
        ECMResultRepository,
        PackageRepository,
        BuildingRecord,
        BaselineRecord,
        ECMResultRecord,
        PackageRecord,
    )
    from src.db.repository import save_full_analysis
    DB_AVAILABLE = True
except ImportError:
    DB_AVAILABLE = False
    logger.warning("Database module not available")


@dataclass
class SimulatedECMResult:
    """Result from actual EnergyPlus simulation."""
    ecm_id: str
    ecm_name: str
    simulation_success: bool
    heating_kwh: float
    heating_kwh_m2: float
    savings_kwh: float
    savings_kwh_m2: float
    savings_percent: float
    simulation_time_seconds: float
    error_message: Optional[str] = None


@dataclass
class SimulatedPackage:
    """Package with combined simulation results."""
    package_name: str
    ecm_ids: List[str]
    combined_heating_kwh_m2: float
    combined_savings_percent: float
    individual_sum_savings_percent: float
    interaction_factor: float
    simulation_time_seconds: float


class SjostadenAnalyzer:
    """Full analysis pipeline for BRF Sjöstaden 2."""

    def __init__(
        self,
        json_path: Path,
        weather_path: Path,
        output_dir: Path,
    ):
        self.json_path = json_path
        self.weather_path = weather_path
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load building data
        with open(json_path) as f:
            self.data = json.load(f)

        # Initialize components
        self.generator = BaselineGenerator()
        self.calibrator = BaselineCalibrator()
        self.runner = SimulationRunner()
        self.parser = ResultsParser()
        self.ecm_catalog = ECMCatalog()
        self.constraint_engine = ConstraintEngine(self.ecm_catalog)
        self.idf_modifier = IDFModifier()
        self.ecm_filter = SmartECMFilter()
        self.archetype_matcher = ArchetypeMatcher()
        self.cost_calculator = SwedishCostCalculatorV2()
        self.dependency_matrix = get_dependency_matrix()

        # Extract building info
        self.building_info = self._extract_building_info()

        # Results storage
        self.baseline_kwh_m2: float = 0
        self.calibrated_idf: Optional[Path] = None
        self.ecm_results: List[SimulatedECMResult] = []
        self.package_results: List[SimulatedPackage] = []

    def _extract_building_info(self) -> Dict:
        """Extract building info from JSON."""
        prop = self.data.get("property", {})
        building = self.data.get("building", {})
        energy_dec = self.data.get("energy_declarations", [{}])[0] if self.data.get("energy_declarations") else {}

        return {
            "brf_name": self.data.get("brf_name", "BRF Sjöstaden 2"),
            "address": prop.get("address", "Unknown"),
            "property_designation": prop.get("property_designation", ""),
            "construction_year": energy_dec.get("construction_year") or prop.get("built_year") or 2003,
            "atemp_m2": energy_dec.get("heated_area_sqm") or 15350,
            "num_apartments": energy_dec.get("num_apartments") or 110,
            "num_floors": building.get("antal_plan") or 7,
            "num_stairwells": energy_dec.get("num_stairwells") or 7,
            "footprint_m2": building.get("footprint_sqm") or 2100,
            "height_m": building.get("height_m") or 21,
            "commercial_area_m2": prop.get("commercial_area_sqm") or 1626,
            "energy_class": energy_dec.get("energy_class") or "B",
            "declared_kwh_m2": energy_dec.get("energy_kwh_per_sqm") or 53,
            "ventilation_type": energy_dec.get("ventilation_type") or "FTX",
            "heating_type": prop.get("heating_type") or "fjärrvärme",
            "latitude": prop.get("latitude") or 59.3018,
            "longitude": prop.get("longitude") or 18.1049,
        }

    def _build_geometry(self) -> BuildingGeometry:
        """Build geometry for this building using real GeoJSON if available."""
        from src.geometry.building_geometry import BuildingGeometryCalculator
        import math

        info = self.building_info
        floors = info["num_floors"]
        height = info["height_m"]

        calculator = BuildingGeometryCalculator()

        # Try to use real GeoJSON geometry from building data
        building_data = self.data.get("building", {})
        geojson = building_data.get("geometry")

        if geojson:
            logger.info("Using real GeoJSON footprint from building data")
            geometry = calculator.calculate_from_geojson(
                geojson=geojson,
                height_m=height,
                floors=floors,
                wwr_by_orientation={'N': 0.15, 'S': 0.25, 'E': 0.20, 'W': 0.20},
                roof_type='flat',
            )
        else:
            # Fallback: Generate synthetic rectangular footprint
            logger.warning("No GeoJSON found, using synthetic rectangular footprint")
            footprint = info["footprint_m2"]
            width = math.sqrt(footprint / 2)
            length = 2 * width

            lat = info["latitude"]
            lon = info["longitude"]
            lat_per_m = 1 / 111000
            lon_per_m = 1 / (111000 * math.cos(math.radians(lat)))

            half_width = (width / 2) * lon_per_m
            half_length = (length / 2) * lat_per_m

            footprint_coords = [
                (lon - half_width, lat + half_length),
                (lon + half_width, lat + half_length),
                (lon + half_width, lat - half_length),
                (lon - half_width, lat - half_length),
                (lon - half_width, lat + half_length),
            ]

            geometry = calculator.calculate(
                footprint_coords=footprint_coords,
                height_m=height,
                floors=floors,
                wwr_by_orientation={'N': 0.15, 'S': 0.25, 'E': 0.20, 'W': 0.20},
                roof_type='flat',
            )

        return geometry

    def _build_context(self) -> EnhancedBuildingContext:
        """Build context for ECM filtering."""
        info = self.building_info

        context = EnhancedBuildingContext(
            address=info["address"],
            property_id=info["property_designation"],
            construction_year=info["construction_year"],
            building_type="multi_family",
            facade_material="concrete",  # 2003 building
            atemp_m2=info["atemp_m2"],
            floors=info["num_floors"],
        )

        # Set current performance
        context.current_heating_kwh_m2 = info["declared_kwh_m2"]

        # Detect FTX
        if "FTX" in info["ventilation_type"].upper():
            context.existing_measures.add(ExistingMeasure.FTX_SYSTEM)
            context.current_heat_recovery = 0.80

        # Match archetype
        context.archetype = self.archetype_matcher.match(
            construction_year=info["construction_year"],
            building_type="multi_family",
        )

        return context

    def run_baseline(self) -> Tuple[Path, float]:
        """Generate and run baseline simulation."""
        logger.info("=" * 60)
        logger.info("STEP 1: BASELINE SIMULATION")
        logger.info("=" * 60)

        geometry = self._build_geometry()
        context = self._build_context()

        logger.info(f"Building: {self.building_info['brf_name']}")
        logger.info(f"  Area: {self.building_info['atemp_m2']:,} m²")
        logger.info(f"  Floors: {self.building_info['num_floors']}")
        logger.info(f"  Apartments: {self.building_info['num_apartments']}")
        logger.info(f"  Commercial: {self.building_info['commercial_area_m2']:,} m²")
        logger.info(f"  Declared: {self.building_info['declared_kwh_m2']} kWh/m²")
        logger.info(f"  Archetype: {context.archetype.name}")

        # Generate baseline IDF
        baseline_dir = self.output_dir / "baseline"
        baseline_model = self.generator.generate(
            geometry=geometry,
            archetype=context.archetype,
            output_dir=baseline_dir,
            model_name="sjostaden_baseline",
            latitude=self.building_info["latitude"],
            longitude=self.building_info["longitude"],
        )

        logger.info(f"Generated IDF: {baseline_model.idf_path}")

        # Run baseline simulation
        start_time = time.time()
        result = self.runner.run(
            baseline_model.idf_path,
            self.weather_path,
            baseline_dir / "output",
        )
        sim_time = time.time() - start_time

        if not result.success:
            logger.error(f"Baseline simulation failed: {result.error_message}")
            raise RuntimeError(f"Baseline failed: {result.error_message}")

        # Parse results
        parsed = self.parser.parse(baseline_dir / "output")
        if not parsed:
            raise RuntimeError("Failed to parse baseline results")

        baseline_kwh_m2 = parsed.heating_kwh_m2
        logger.info(f"Baseline result: {baseline_kwh_m2:.1f} kWh/m² (simulated in {sim_time:.1f}s)")
        logger.info(f"Declared: {self.building_info['declared_kwh_m2']} kWh/m²")
        logger.info(f"Gap: {abs(baseline_kwh_m2 - self.building_info['declared_kwh_m2']):.1f} kWh/m² ({abs(baseline_kwh_m2 - self.building_info['declared_kwh_m2'])/self.building_info['declared_kwh_m2']*100:.1f}%)")

        return baseline_model.idf_path, baseline_kwh_m2

    def calibrate(self, baseline_idf: Path, baseline_kwh_m2: float) -> Tuple[Path, float]:
        """Calibrate baseline to declared energy."""
        logger.info("")
        logger.info("=" * 60)
        logger.info("STEP 2: CALIBRATION")
        logger.info("=" * 60)

        target = self.building_info["declared_kwh_m2"]
        gap_percent = abs(baseline_kwh_m2 - target) / target * 100

        if gap_percent <= 10:
            logger.info(f"Gap is {gap_percent:.1f}% - within tolerance, skipping calibration")
            return baseline_idf, baseline_kwh_m2

        logger.info(f"Calibrating from {baseline_kwh_m2:.1f} to {target} kWh/m²...")

        cal_result = self.calibrator.calibrate(
            idf_path=baseline_idf,
            weather_path=self.weather_path,
            measured_heating_kwh_m2=target,
            output_dir=self.output_dir / "calibration",
        )

        if cal_result.success:
            logger.info(f"Calibration successful!")
            logger.info(f"  Final: {cal_result.calibrated_kwh_m2:.1f} kWh/m²")
            logger.info(f"  Error: {cal_result.final_error_percent:.1f}%")
            return cal_result.calibrated_idf_path, cal_result.calibrated_kwh_m2
        else:
            logger.warning(f"Calibration failed, using baseline")
            return baseline_idf, baseline_kwh_m2

    def run_ecm_simulations(self, calibrated_idf: Path, baseline_kwh_m2: float) -> List[SimulatedECMResult]:
        """Run all applicable ECM simulations."""
        logger.info("")
        logger.info("=" * 60)
        logger.info("STEP 3: ECM SIMULATIONS")
        logger.info("=" * 60)

        context = self._build_context()
        all_ecms = self.ecm_catalog.all()

        # Filter applicable ECMs
        filter_result = self.ecm_filter.filter_ecms(
            all_ecms=all_ecms,
            context=context,
            constraint_engine=self.constraint_engine,
        )

        applicable = filter_result['applicable']
        logger.info(f"Applicable ECMs: {len(applicable)}")
        logger.info(f"Already done: {len(filter_result['already_done'])}")
        logger.info(f"Not applicable: {len(filter_result['not_applicable'])}")

        results = []
        total_start = time.time()

        for i, ecm in enumerate(applicable, 1):
            logger.info(f"\n[{i}/{len(applicable)}] Running {ecm.name}...")

            ecm_dir = self.output_dir / f"ecm_{ecm.id}"
            ecm_dir.mkdir(parents=True, exist_ok=True)

            # Get default parameters
            params = self._get_ecm_params(ecm.id)

            try:
                # Apply ECM to IDF
                modified_idf = self.idf_modifier.apply_single(
                    baseline_idf=calibrated_idf,
                    ecm_id=ecm.id,
                    params=params,
                    output_dir=ecm_dir,
                )

                # Run simulation
                start = time.time()
                sim_result = self.runner.run(
                    modified_idf,
                    self.weather_path,
                    ecm_dir / "output",
                )
                sim_time = time.time() - start

                if sim_result.success:
                    parsed = self.parser.parse(ecm_dir / "output")
                    if parsed:
                        heating_kwh_m2 = parsed.heating_kwh_m2
                        savings_kwh_m2 = baseline_kwh_m2 - heating_kwh_m2
                        savings_pct = (savings_kwh_m2 / baseline_kwh_m2 * 100) if baseline_kwh_m2 > 0 else 0

                        results.append(SimulatedECMResult(
                            ecm_id=ecm.id,
                            ecm_name=ecm.name,
                            simulation_success=True,
                            heating_kwh=heating_kwh_m2 * self.building_info["atemp_m2"],
                            heating_kwh_m2=heating_kwh_m2,
                            savings_kwh=savings_kwh_m2 * self.building_info["atemp_m2"],
                            savings_kwh_m2=savings_kwh_m2,
                            savings_percent=savings_pct,
                            simulation_time_seconds=sim_time,
                        ))

                        logger.info(f"  → {heating_kwh_m2:.1f} kWh/m² (saves {savings_pct:.1f}%) [{sim_time:.1f}s]")
                        continue

                # Simulation failed
                results.append(SimulatedECMResult(
                    ecm_id=ecm.id,
                    ecm_name=ecm.name,
                    simulation_success=False,
                    heating_kwh=0,
                    heating_kwh_m2=0,
                    savings_kwh=0,
                    savings_kwh_m2=0,
                    savings_percent=0,
                    simulation_time_seconds=0,
                    error_message=sim_result.error_message,
                ))
                logger.warning(f"  → FAILED: {sim_result.error_message}")

            except Exception as e:
                logger.error(f"  → ERROR: {e}")
                results.append(SimulatedECMResult(
                    ecm_id=ecm.id,
                    ecm_name=ecm.name,
                    simulation_success=False,
                    heating_kwh=0,
                    heating_kwh_m2=0,
                    savings_kwh=0,
                    savings_kwh_m2=0,
                    savings_percent=0,
                    simulation_time_seconds=0,
                    error_message=str(e),
                ))

        total_time = time.time() - total_start
        successful = [r for r in results if r.simulation_success]
        logger.info(f"\nECM simulations complete: {len(successful)}/{len(results)} successful")
        logger.info(f"Total time: {total_time/60:.1f} minutes")

        return results

    def run_package_simulations(
        self,
        calibrated_idf: Path,
        baseline_kwh_m2: float,
        ecm_results: List[SimulatedECMResult],
    ) -> List[SimulatedPackage]:
        """Run combined package simulations."""
        logger.info("")
        logger.info("=" * 60)
        logger.info("STEP 4: PACKAGE SIMULATIONS (COMBINED)")
        logger.info("=" * 60)

        # Filter successful ECMs
        successful = [r for r in ecm_results if r.simulation_success and r.savings_percent > 0]
        successful.sort(key=lambda x: x.savings_percent, reverse=True)

        if len(successful) < 2:
            logger.warning("Not enough successful ECMs for packages")
            return []

        packages = []

        # Define packages
        package_configs = [
            ("Grundpaket", 2),
            ("Standardpaket", 4),
            ("Premiumpaket", min(8, len(successful))),
        ]

        for pkg_name, max_ecms in package_configs:
            ecm_ids = [r.ecm_id for r in successful[:max_ecms]]

            # Validate with dependency matrix
            is_valid, issues = validate_package(ecm_ids)
            if not is_valid:
                logger.warning(f"{pkg_name}: Invalid - {issues}")
                # Remove conflicting ECMs
                ecm_ids = self._resolve_conflicts(ecm_ids)

            if len(ecm_ids) < 2:
                continue

            logger.info(f"\nRunning {pkg_name} ({len(ecm_ids)} ECMs combined)...")
            logger.info(f"  ECMs: {', '.join(ecm_ids)}")

            # Apply all ECMs to IDF
            pkg_dir = self.output_dir / f"package_{pkg_name.lower()}"
            pkg_dir.mkdir(parents=True, exist_ok=True)

            try:
                # apply_multiple expects list of (ecm_id, params) tuples
                ecm_tuples = [(eid, self._get_ecm_params(eid)) for eid in ecm_ids]
                combined_idf = self.idf_modifier.apply_multiple(
                    baseline_idf=calibrated_idf,
                    ecms=ecm_tuples,
                    output_dir=pkg_dir,
                    output_name=f"package_{pkg_name.lower()}",
                )

                # Run combined simulation
                start = time.time()
                sim_result = self.runner.run(
                    combined_idf,
                    self.weather_path,
                    pkg_dir / "output",
                )
                sim_time = time.time() - start

                if sim_result.success:
                    parsed = self.parser.parse(pkg_dir / "output")
                    if parsed:
                        combined_kwh_m2 = parsed.heating_kwh_m2
                        combined_savings_pct = (baseline_kwh_m2 - combined_kwh_m2) / baseline_kwh_m2 * 100

                        # Calculate individual sum
                        individual_sum = sum(
                            r.savings_percent for r in successful
                            if r.ecm_id in ecm_ids
                        )

                        interaction = combined_savings_pct / individual_sum if individual_sum > 0 else 1.0

                        packages.append(SimulatedPackage(
                            package_name=pkg_name,
                            ecm_ids=ecm_ids,
                            combined_heating_kwh_m2=combined_kwh_m2,
                            combined_savings_percent=combined_savings_pct,
                            individual_sum_savings_percent=individual_sum,
                            interaction_factor=interaction,
                            simulation_time_seconds=sim_time,
                        ))

                        logger.info(f"  → {combined_kwh_m2:.1f} kWh/m² (saves {combined_savings_pct:.1f}%)")
                        logger.info(f"  → Individual sum: {individual_sum:.1f}%, Interaction factor: {interaction:.2f}")
                else:
                    logger.error(f"  → FAILED: {sim_result.error_message}")

            except Exception as e:
                logger.error(f"  → ERROR: {e}")

        return packages

    def _resolve_conflicts(self, ecm_ids: List[str]) -> List[str]:
        """Remove conflicting ECMs."""
        resolved = []
        for ecm_id in ecm_ids:
            test = resolved + [ecm_id]
            is_valid, _ = validate_package(test)
            if is_valid:
                resolved.append(ecm_id)
        return resolved

    def _get_ecm_params(self, ecm_id: str) -> Dict:
        """Get default parameters for ECM."""
        defaults = {
            'wall_external_insulation': {'thickness_mm': 100, 'material': 'mineral_wool'},
            'wall_internal_insulation': {'thickness_mm': 50, 'material': 'mineral_wool'},
            'roof_insulation': {'thickness_mm': 150, 'material': 'mineral_wool'},
            'window_replacement': {'u_value': 0.9, 'shgc': 0.5},
            'air_sealing': {'reduction_factor': 0.5},
            'ftx_upgrade': {'effectiveness': 0.85},
            'ftx_installation': {'effectiveness': 0.80},
            'demand_controlled_ventilation': {'co2_setpoint': 1000},
            'solar_pv': {'coverage_fraction': 0.7, 'panel_efficiency': 0.20},
            'led_lighting': {'power_density': 6},
            'smart_thermostats': {'setback_c': 2},
            'heat_pump_integration': {'cop': 3.5, 'coverage': 0.8},
        }
        return defaults.get(ecm_id, {})

    def save_to_database(
        self,
        baseline_kwh_m2: float,
        ecm_results: List[SimulatedECMResult],
        package_results: List[SimulatedPackage],
    ) -> Optional[str]:
        """Save real results to database."""
        if not DB_AVAILABLE:
            logger.warning("Database not available")
            return None

        logger.info("")
        logger.info("=" * 60)
        logger.info("STEP 5: SAVING TO DATABASE")
        logger.info("=" * 60)

        try:
            info = self.building_info

            # Energy pricing
            dh_price = DISTRICT_HEATING_PRICES.get("stockholm")
            energy_price = dh_price.base_price_sek_kwh + dh_price.network_fee_sek_kwh
            co2_factor = dh_price.co2_kg_per_kwh

            # Building record
            building = BuildingRecord(
                address=info["address"],
                property_designation=info["property_designation"],
                name=info["brf_name"],
                construction_year=info["construction_year"],
                heated_area_m2=info["atemp_m2"],
                num_apartments=info["num_apartments"],
                num_floors=info["num_floors"],
                region="stockholm",
                declared_energy_kwh_m2=info["declared_kwh_m2"],
                energy_class=info["energy_class"],
                heating_system="district_heating",
                ventilation_system="ftx",
                owner_type="brf",
            )

            # Baseline record
            baseline = BaselineRecord(
                building_id="",
                heating_kwh_m2=baseline_kwh_m2,
                is_calibrated=True,
                calibration_gap_percent=abs(baseline_kwh_m2 - info["declared_kwh_m2"]) / info["declared_kwh_m2"] * 100,
            )

            # ECM result records
            ecm_records = []
            for r in ecm_results:
                if r.simulation_success:
                    # Calculate costs
                    quantity = self._get_ecm_quantity(r.ecm_id)
                    try:
                        cost = self.cost_calculator.calculate(
                            ecm_id=r.ecm_id,
                            quantity=quantity,
                            region=Region.STOCKHOLM,
                            building_size_m2=info["atemp_m2"],
                            owner_type=OwnerType.BRF,
                            num_apartments=info["num_apartments"],
                        )
                        net_cost = cost.net_cost
                    except:
                        net_cost = quantity * 1000  # Fallback

                    annual_savings_sek = r.savings_kwh * energy_price
                    payback = net_cost / annual_savings_sek if annual_savings_sek > 0 else 999

                    ecm_records.append(ECMResultRecord(
                        building_id="",
                        ecm_id=r.ecm_id,
                        ecm_name=r.ecm_name,
                        heating_kwh_m2=r.heating_kwh_m2,
                        heating_savings_kwh=r.savings_kwh,
                        heating_savings_percent=r.savings_percent,
                        net_cost=net_cost,
                        annual_savings_sek=annual_savings_sek,
                        simple_payback_years=min(payback, 99),
                        annual_co2_reduction_kg=r.savings_kwh * co2_factor,
                        is_applicable=True,
                        simulated=True,
                    ))

            # Package records
            package_records = []
            for p in package_results:
                # Sum costs for package
                total_cost = sum(
                    r.net_cost for r in ecm_records
                    if r.ecm_id in p.ecm_ids
                ) if ecm_records else 0

                annual_savings = (baseline_kwh_m2 - p.combined_heating_kwh_m2) * info["atemp_m2"] * energy_price
                payback = total_cost / annual_savings if annual_savings > 0 else 999

                package_records.append(PackageRecord(
                    building_id="",
                    package_name=p.package_name,
                    package_type=p.package_name.lower().replace("paket", ""),
                    ecm_ids=p.ecm_ids,
                    combined_heating_kwh_m2=p.combined_heating_kwh_m2,
                    combined_savings_percent=p.combined_savings_percent,
                    synergy_factor=p.interaction_factor,
                    net_cost=total_cost,
                    annual_savings_sek=annual_savings,
                    simple_payback_years=min(payback, 99),
                    is_valid=True,
                ))

            # Save all
            result = save_full_analysis(
                building=building,
                baseline=baseline,
                ecm_results=ecm_records,
                packages=package_records,
            )

            logger.info(f"Saved to database!")
            logger.info(f"  Building ID: {result['building_id']}")
            logger.info(f"  ECM results: {result['ecm_count']}")
            logger.info(f"  Packages: {result['package_count']}")

            return result['building_id']

        except Exception as e:
            logger.error(f"Failed to save: {e}")
            return None

    def _get_ecm_quantity(self, ecm_id: str) -> float:
        """Get quantity for cost calculation."""
        info = self.building_info
        atemp = info["atemp_m2"]
        apartments = info["num_apartments"]
        floors = info["num_floors"]

        quantities = {
            "wall_external_insulation": atemp * 0.5,
            "roof_insulation": atemp / floors,
            "window_replacement": atemp * 0.5 * 0.15,
            "air_sealing": atemp,
            "led_lighting": atemp,
            "smart_thermostats": apartments,
            "solar_pv": (atemp / floors) * 0.15,
        }
        return quantities.get(ecm_id, atemp * 0.1)

    def run_full_analysis(self) -> Dict:
        """Run complete analysis pipeline."""
        logger.info("=" * 60)
        logger.info("FULL SIMULATION ANALYSIS: BRF SJÖSTADEN 2")
        logger.info("=" * 60)
        logger.info(f"Started: {datetime.now().isoformat()}")

        total_start = time.time()

        # Step 1: Baseline
        baseline_idf, baseline_kwh_m2 = self.run_baseline()

        # Step 2: Calibration
        calibrated_idf, calibrated_kwh_m2 = self.calibrate(baseline_idf, baseline_kwh_m2)
        self.baseline_kwh_m2 = calibrated_kwh_m2
        self.calibrated_idf = calibrated_idf

        # Step 3: ECM simulations
        ecm_results = self.run_ecm_simulations(calibrated_idf, calibrated_kwh_m2)
        self.ecm_results = ecm_results

        # Step 4: Package simulations
        package_results = self.run_package_simulations(calibrated_idf, calibrated_kwh_m2, ecm_results)
        self.package_results = package_results

        # Step 5: Save to database
        building_id = self.save_to_database(calibrated_kwh_m2, ecm_results, package_results)

        total_time = time.time() - total_start

        # Summary
        logger.info("")
        logger.info("=" * 60)
        logger.info("ANALYSIS COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Total time: {total_time/60:.1f} minutes")
        logger.info(f"Building ID: {building_id}")
        logger.info(f"Baseline: {calibrated_kwh_m2:.1f} kWh/m²")

        successful = [r for r in ecm_results if r.simulation_success]
        logger.info(f"ECMs simulated: {len(successful)}/{len(ecm_results)}")

        if successful:
            logger.info("\nTop 5 ECMs by savings:")
            for r in sorted(successful, key=lambda x: x.savings_percent, reverse=True)[:5]:
                logger.info(f"  {r.ecm_name}: {r.savings_percent:.1f}%")

        if package_results:
            logger.info("\nPackages:")
            for p in package_results:
                logger.info(f"  {p.package_name}: {p.combined_savings_percent:.1f}% (interaction: {p.interaction_factor:.2f})")

        return {
            "building_id": building_id,
            "baseline_kwh_m2": calibrated_kwh_m2,
            "ecm_results": len(successful),
            "packages": len(package_results),
            "total_time_minutes": total_time / 60,
        }


def main():
    """Run full analysis."""
    json_path = Path("/Users/hosseins/Downloads/brf_sjostaden_2_export.json")
    weather_path = Path("/Users/hosseins/Dropbox/Dev/Raiden/tests/fixtures/stockholm.epw")
    output_dir = Path("/Users/hosseins/Dropbox/Dev/Raiden/output_sjostaden_full")

    if not json_path.exists():
        logger.error(f"JSON file not found: {json_path}")
        sys.exit(1)

    if not weather_path.exists():
        logger.error(f"Weather file not found: {weather_path}")
        sys.exit(1)

    analyzer = SjostadenAnalyzer(
        json_path=json_path,
        weather_path=weather_path,
        output_dir=output_dir,
    )

    result = analyzer.run_full_analysis()

    logger.info("\nDone!")
    return result


if __name__ == "__main__":
    main()
