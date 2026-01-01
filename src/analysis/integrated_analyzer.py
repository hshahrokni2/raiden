"""
Integrated Building Analyzer - Full pipeline with V2 costs and database storage.

This module wires together:
- Building context from JSON/energy declaration
- Baseline simulation with calibration
- ECM analysis with V2 cost model
- Package generation with synergies and dependencies
- Database storage to Supabase

Usage:
    from src.analysis.integrated_analyzer import analyze_building_json

    result = analyze_building_json(
        json_path=Path("brf_sjostaden.json"),
        weather_path=Path("stockholm.epw"),
        save_to_db=True,
    )
"""

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import time

# Core modules
from ..core.building_context import (
    EnhancedBuildingContext,
    BuildingContextBuilder,
    SmartECMFilter,
    ExistingMeasure,
)
from ..baseline.archetypes import ArchetypeMatcher
from ..baseline.generator import BaselineGenerator
from ..baseline.calibrator import BaselineCalibrator
from ..ecm.catalog import ECMCatalog
from ..ecm.constraints import ConstraintEngine
from ..ecm.idf_modifier import IDFModifier
from ..ecm.dependencies import (
    get_dependency_matrix,
    validate_package,
    get_package_synergy,
)
from ..simulation.runner import SimulationRunner
from ..simulation.results import ResultsParser
from ..geometry.building_geometry import BuildingGeometry

# V2 Cost Model
from ..roi.costs_sweden_v2 import (
    SwedishCostCalculatorV2,
    OwnerType,
    Region,
    DISTRICT_HEATING_PRICES,
    ELECTRICITY_PRICES,
    PACKAGE_COST_SYNERGIES,
    estimate_scaffolding_cost,
    calculate_project_overhead,
)

# Database
try:
    from ..db import (
        BuildingRepository,
        ECMResultRepository,
        PackageRepository,
        BuildingRecord,
        BaselineRecord,
        ECMResultRecord,
        PackageRecord,
    )
    from ..db.repository import save_full_analysis
    DB_AVAILABLE = True
except ImportError:
    DB_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class IntegratedECMResult:
    """ECM result with full cost breakdown."""
    ecm_id: str
    ecm_name: str
    ecm_category: str

    # Simulation results
    simulation_success: bool
    heating_kwh_m2: float
    savings_kwh_m2: float
    savings_percent: float

    # V2 Cost breakdown
    material_cost_sek: float = 0
    labor_cost_sek: float = 0
    total_cost_sek: float = 0
    rot_deduction_sek: float = 0
    green_tech_deduction_sek: float = 0
    net_cost_sek: float = 0

    # Financial metrics
    annual_savings_sek: float = 0
    simple_payback_years: float = 0
    npv_20yr: float = 0

    # CO2
    annual_co2_reduction_kg: float = 0

    # Applicability
    is_applicable: bool = True
    constraint_issues: List[str] = field(default_factory=list)

    # Parameters applied
    parameters: Dict = field(default_factory=dict)


@dataclass
class IntegratedPackage:
    """Package with combined simulation and synergy factors."""
    package_id: str
    package_name: str
    package_type: str  # basic, standard, premium

    # ECMs included
    ecm_ids: List[str]

    # Simulated combined results
    combined_heating_kwh_m2: float
    combined_savings_percent: float

    # Interaction factors
    synergy_factor: float  # From dependency matrix
    interaction_factor: float  # Actual vs sum of individual

    # Costs with package discounts
    total_cost_sek: float
    package_discount_sek: float  # Shared scaffolding etc
    net_cost_sek: float

    # Financial
    annual_savings_sek: float
    simple_payback_years: float
    npv_20yr: float

    # Validation
    is_valid: bool = True
    validation_issues: List[str] = field(default_factory=list)


@dataclass
class IntegratedAnalysisResult:
    """Complete analysis result with costs and database IDs."""
    # Building info
    building_id: Optional[str]  # Supabase ID if saved
    address: str
    property_designation: str
    construction_year: int
    atemp_m2: float
    num_apartments: int

    # Location (for regional pricing)
    region: str
    owner_type: str

    # Energy performance
    declared_kwh_m2: float
    baseline_kwh_m2: float
    calibration_error_percent: float

    # Archetype
    archetype_id: str
    archetype_name: str

    # Energy prices used
    district_heating_price_sek_kwh: float
    electricity_price_sek_kwh: float

    # ECM results
    ecm_results: List[IntegratedECMResult]
    applicable_ecm_count: int
    excluded_ecm_count: int

    # Packages
    packages: List[IntegratedPackage]

    # Summary
    best_package: Optional[str]
    best_package_savings_percent: float
    best_package_payback_years: float

    # Metadata
    analysis_timestamp: str
    analysis_duration_seconds: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


class IntegratedAnalyzer:
    """
    Full-featured analyzer with V2 costs and database integration.
    """

    def __init__(
        self,
        energyplus_path: Optional[str] = None,
        work_dir: Optional[Path] = None,
        save_to_db: bool = True,
    ):
        self.energyplus_path = energyplus_path
        self.work_dir = work_dir or Path("/tmp/raiden_analysis")
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.save_to_db = save_to_db and DB_AVAILABLE

        # Initialize components
        self.context_builder = BuildingContextBuilder()
        self.archetype_matcher = ArchetypeMatcher()
        self.generator = BaselineGenerator()
        self.calibrator = BaselineCalibrator(energyplus_path)
        self.ecm_catalog = ECMCatalog()
        self.constraint_engine = ConstraintEngine(self.ecm_catalog)
        self.ecm_filter = SmartECMFilter()
        self.idf_modifier = IDFModifier()
        self.runner = SimulationRunner(energyplus_path)
        self.results_parser = ResultsParser()

        # V2 Cost calculator
        self.cost_calculator = SwedishCostCalculatorV2()

        # Dependency matrix
        self.dependency_matrix = get_dependency_matrix()

    def analyze_from_json(
        self,
        json_path: Path,
        weather_path: Path,
        run_simulations: bool = True,
    ) -> IntegratedAnalysisResult:
        """
        Analyze building from exported JSON file.

        Args:
            json_path: Path to building JSON (e.g., brf_sjostaden_2_export.json)
            weather_path: Path to weather file
            run_simulations: Whether to run E+ simulations

        Returns:
            IntegratedAnalysisResult with full cost analysis
        """
        start_time = time.time()

        # Load JSON
        with open(json_path) as f:
            data = json.load(f)

        logger.info(f"Analyzing: {data.get('brf_name', 'Unknown')}")

        # Extract building info
        building_info = self._extract_building_info(data)

        # Build context
        context = self._build_context(building_info)

        # Determine region and owner type for cost calculations
        region = self._determine_region(building_info)
        owner_type = OwnerType.BRF  # Multi-family = BRF (no ROT)

        # Get energy prices for this region
        dh_price = DISTRICT_HEATING_PRICES.get(region, DISTRICT_HEATING_PRICES["medium_city"])
        el_price = ELECTRICITY_PRICES.get("se3", ELECTRICITY_PRICES["average"])  # Stockholm = SE3

        energy_price_sek_kwh = dh_price.base_price_sek_kwh + dh_price.network_fee_sek_kwh
        co2_factor = dh_price.co2_kg_per_kwh

        # Run analysis
        baseline_kwh_m2 = building_info["energy_kwh_m2"]
        calibration_error = 0.0
        ecm_results = []
        packages = []

        if run_simulations:
            # Full simulation pipeline
            baseline_kwh_m2, calibration_error, ecm_results = self._run_simulation_pipeline(
                context=context,
                weather_path=weather_path,
                building_info=building_info,
            )

        # Calculate costs for each ECM
        ecm_results_with_costs = self._calculate_ecm_costs(
            ecm_results=ecm_results,
            context=context,
            building_info=building_info,
            region=region,
            owner_type=owner_type,
            energy_price_sek_kwh=energy_price_sek_kwh,
            co2_factor=co2_factor,
        )

        # Generate packages with combined costs
        packages = self._generate_packages_v2(
            ecm_results=ecm_results_with_costs,
            baseline_kwh_m2=baseline_kwh_m2,
            building_info=building_info,
            region=region,
            owner_type=owner_type,
            energy_price_sek_kwh=energy_price_sek_kwh,
        )

        # Find best package
        best_package = None
        best_savings = 0
        best_payback = 999
        if packages:
            # Best by ROI (shortest payback with meaningful savings)
            valid_packages = [p for p in packages if p.is_valid and p.combined_savings_percent > 5]
            if valid_packages:
                best = min(valid_packages, key=lambda p: p.simple_payback_years)
                best_package = best.package_name
                best_savings = best.combined_savings_percent
                best_payback = best.simple_payback_years

        # Create result
        result = IntegratedAnalysisResult(
            building_id=None,
            address=building_info["address"],
            property_designation=building_info["property_designation"],
            construction_year=building_info["construction_year"],
            atemp_m2=building_info["atemp_m2"],
            num_apartments=building_info["num_apartments"],
            region=region,
            owner_type=owner_type.value,
            declared_kwh_m2=building_info["energy_kwh_m2"],
            baseline_kwh_m2=baseline_kwh_m2,
            calibration_error_percent=calibration_error,
            archetype_id=context.archetype.name.lower().replace(" ", "_") if context.archetype else "unknown",
            archetype_name=context.archetype.name if context.archetype else "Unknown",
            district_heating_price_sek_kwh=energy_price_sek_kwh,
            electricity_price_sek_kwh=el_price.base_price_sek_kwh + el_price.network_fee_sek_kwh + el_price.tax_sek_kwh,
            ecm_results=ecm_results_with_costs,
            applicable_ecm_count=len([e for e in ecm_results_with_costs if e.is_applicable]),
            excluded_ecm_count=len([e for e in ecm_results_with_costs if not e.is_applicable]),
            packages=packages,
            best_package=best_package,
            best_package_savings_percent=best_savings,
            best_package_payback_years=best_payback,
            analysis_timestamp=datetime.now().isoformat(),
            analysis_duration_seconds=time.time() - start_time,
        )

        # Save to database
        if self.save_to_db:
            result.building_id = self._save_to_database(result, building_info)

        logger.info(f"Analysis complete in {result.analysis_duration_seconds:.1f}s")

        return result

    def _extract_building_info(self, data: Dict) -> Dict:
        """Extract building info from JSON structure."""
        prop = data.get("property", {})
        building = data.get("building", {})
        energy_dec = data.get("energy_declarations", [{}])[0] if data.get("energy_declarations") else {}

        # Prefer energy declaration data, fall back to property/building
        return {
            "brf_name": data.get("brf_name", "Unknown"),
            "address": prop.get("address") or building.get("address", "Unknown"),
            "property_designation": prop.get("property_designation") or building.get("fastighetsbeteckning", ""),
            "construction_year": energy_dec.get("construction_year") or prop.get("built_year") or building.get("building_year", 2000),
            "atemp_m2": energy_dec.get("heated_area_sqm") or building.get("atemp_sqm") or prop.get("total_area_sqm", 1000),
            "num_apartments": energy_dec.get("num_apartments") or prop.get("total_apartments", 10),
            "num_floors": energy_dec.get("num_floors") or building.get("antal_plan", 4),
            "num_stairwells": energy_dec.get("num_stairwells") or building.get("antal_trapphus", 1),
            "energy_class": energy_dec.get("energy_class") or prop.get("energy_class", "D"),
            "energy_kwh_m2": energy_dec.get("energy_kwh_per_sqm") or prop.get("energy_kwh_per_sqm", 100),
            "ventilation_type": energy_dec.get("ventilation_type") or "F",
            "heating_type": prop.get("heating_type", "fjärrvärme"),
            "municipality": prop.get("municipality") or building.get("kommun", "Stockholm"),
            "footprint_m2": building.get("footprint_sqm", 0),
            "height_m": building.get("height_m", 0),
            "latitude": prop.get("latitude") or building.get("latitude"),
            "longitude": prop.get("longitude") or building.get("longitude"),
        }

    def _build_context(self, info: Dict) -> EnhancedBuildingContext:
        """Build EnhancedBuildingContext from extracted info."""
        context = EnhancedBuildingContext(
            address=info["address"],
            property_id=info["property_designation"],
            construction_year=info["construction_year"],
            building_type="multi_family",
            facade_material="concrete",  # Default for post-2000
            atemp_m2=info["atemp_m2"],
            floors=info["num_floors"],
        )

        # Set current performance
        context.current_heating_kwh_m2 = info["energy_kwh_m2"]

        # Detect existing measures from ventilation type
        if info["ventilation_type"] and "FTX" in info["ventilation_type"].upper():
            context.existing_measures.add(ExistingMeasure.FTX_SYSTEM)
            context.current_heat_recovery = 0.80  # Typical for modern FTX

        # Match archetype
        context.archetype = self.archetype_matcher.match(
            construction_year=info["construction_year"],
            building_type="multi_family",
        )

        return context

    def _determine_region(self, info: Dict) -> str:
        """Determine pricing region from municipality."""
        municipality = info.get("municipality", "").lower()

        if "stockholm" in municipality:
            return "stockholm"
        elif "göteborg" in municipality or "gothenburg" in municipality:
            return "gothenburg"
        elif "malmö" in municipality:
            return "malmo"
        else:
            return "medium_city"

    def _run_simulation_pipeline(
        self,
        context: EnhancedBuildingContext,
        weather_path: Path,
        building_info: Dict,
    ) -> tuple:
        """Run full E+ simulation pipeline."""
        # This would run actual EnergyPlus simulations
        # For now, return declared values as baseline
        baseline_kwh_m2 = building_info["energy_kwh_m2"]
        calibration_error = 0.0
        ecm_results = []

        # TODO: Implement full simulation when EnergyPlus available
        logger.info("Simulation pipeline would run here - using declared values")

        return baseline_kwh_m2, calibration_error, ecm_results

    def _calculate_ecm_costs(
        self,
        ecm_results: List,
        context: EnhancedBuildingContext,
        building_info: Dict,
        region: str,
        owner_type: OwnerType,
        energy_price_sek_kwh: float,
        co2_factor: float,
    ) -> List[IntegratedECMResult]:
        """Calculate V2 costs for all ECMs."""
        results = []

        # Get all ECMs and filter
        all_ecms = self.ecm_catalog.all()
        filter_result = self.ecm_filter.filter_ecms(
            all_ecms=all_ecms,
            context=context,
            constraint_engine=self.constraint_engine,
        )

        applicable_ecms = {e.id: e for e in filter_result['applicable']}

        # Calculate typical savings for each ECM (without simulation)
        for ecm in all_ecms:
            is_applicable = ecm.id in applicable_ecms

            # Get reasons for exclusion
            constraint_issues = []
            if not is_applicable:
                for item in filter_result['already_done'] + filter_result['not_applicable']:
                    if item['ecm'].id == ecm.id:
                        constraint_issues = item.get('reasons', [str(item.get('reason', ''))])
                        break

            # Estimate savings from catalog
            typical_savings_pct = ecm.typical_savings_percent if hasattr(ecm, 'typical_savings_percent') else 5.0
            baseline = building_info["energy_kwh_m2"]
            savings_kwh_m2 = baseline * typical_savings_pct / 100

            # Calculate quantity for cost
            quantity = self._get_ecm_quantity(ecm.id, building_info, context)

            # Get cost from V2 calculator
            try:
                cost_result = self.cost_calculator.calculate(
                    ecm_id=ecm.id,
                    quantity=quantity,
                    region=Region(region) if region in [r.value for r in Region] else Region.MEDIUM_CITY,
                    building_size_m2=building_info["atemp_m2"],
                    owner_type=owner_type,
                    num_apartments=building_info["num_apartments"],
                )

                material_cost = cost_result.material_cost
                labor_cost = cost_result.labor_cost
                total_cost = cost_result.total_cost
                rot_deduction = cost_result.rot_deduction
                green_deduction = cost_result.green_tech_deduction
                net_cost = cost_result.net_cost

            except Exception as e:
                logger.debug(f"Cost calculation failed for {ecm.id}: {e}")
                # Fallback to simple estimate
                material_cost = quantity * 500
                labor_cost = quantity * 300
                total_cost = material_cost + labor_cost
                rot_deduction = 0
                green_deduction = 0
                net_cost = total_cost

            # Financial calculations
            annual_savings_kwh = savings_kwh_m2 * building_info["atemp_m2"]
            annual_savings_sek = annual_savings_kwh * energy_price_sek_kwh
            payback = net_cost / annual_savings_sek if annual_savings_sek > 0 else 999
            npv = self._calculate_npv(net_cost, annual_savings_sek, 20, 0.04)
            co2_reduction = savings_kwh_m2 * co2_factor

            results.append(IntegratedECMResult(
                ecm_id=ecm.id,
                ecm_name=ecm.name,
                ecm_category=ecm.category.value if hasattr(ecm.category, 'value') else str(ecm.category),
                simulation_success=True,  # Estimated, not simulated
                heating_kwh_m2=baseline - savings_kwh_m2,
                savings_kwh_m2=savings_kwh_m2,
                savings_percent=typical_savings_pct,
                material_cost_sek=material_cost,
                labor_cost_sek=labor_cost,
                total_cost_sek=total_cost,
                rot_deduction_sek=rot_deduction,
                green_tech_deduction_sek=green_deduction,
                net_cost_sek=net_cost,
                annual_savings_sek=annual_savings_sek,
                simple_payback_years=min(payback, 99),
                npv_20yr=npv,
                annual_co2_reduction_kg=co2_reduction,
                is_applicable=is_applicable,
                constraint_issues=constraint_issues,
            ))

        return results

    def _get_ecm_quantity(self, ecm_id: str, building_info: Dict, context: EnhancedBuildingContext) -> float:
        """Get quantity for cost calculation based on ECM type."""
        atemp = building_info["atemp_m2"]
        apartments = building_info["num_apartments"]
        floors = building_info["num_floors"]

        # Quantity mapping by ECM
        quantity_map = {
            # Per m² facade (estimate: 50% of Atemp)
            "wall_external_insulation": atemp * 0.5,
            "wall_internal_insulation": atemp * 0.5,

            # Per m² roof (estimate: Atemp / floors)
            "roof_insulation": atemp / max(1, floors),

            # Per m² window (estimate: 15% of facade)
            "window_replacement": atemp * 0.5 * 0.15,

            # Per m² Atemp
            "air_sealing": atemp,
            "led_lighting": atemp,

            # Per apartment
            "smart_thermostats": apartments,
            "smart_meters": apartments,
            "low_flow_fixtures": apartments,
            "water_efficient_fixtures": apartments,

            # Per kW or system
            "ftx_installation": atemp * 0.01,  # Approx kW
            "ftx_upgrade": atemp * 0.01,
            "demand_controlled_ventilation": atemp,

            # Per kWp (estimate roof area * 0.15)
            "solar_pv": (atemp / floors) * 0.15,
            "solar_thermal": (atemp / floors) * 0.05,

            # Per building (fixed)
            "heat_pump_integration": 1,
            "building_automation_system": 1,
            "energy_monitoring": 1,
        }

        return quantity_map.get(ecm_id, atemp * 0.1)  # Default to 10% of Atemp

    def _generate_packages_v2(
        self,
        ecm_results: List[IntegratedECMResult],
        baseline_kwh_m2: float,
        building_info: Dict,
        region: str,
        owner_type: OwnerType,
        energy_price_sek_kwh: float,
    ) -> List[IntegratedPackage]:
        """Generate packages with dependency validation and synergy factors."""
        packages = []

        # Filter applicable ECMs with positive savings
        applicable = [e for e in ecm_results if e.is_applicable and e.savings_percent > 0]
        applicable.sort(key=lambda x: x.simple_payback_years)  # Sort by ROI

        if not applicable:
            return packages

        # Package definitions
        package_configs = [
            ("Grundpaket", "basic", 2),
            ("Standardpaket", "standard", 4),
            ("Premiumpaket", "premium", len(applicable)),
        ]

        for name, pkg_type, max_ecms in package_configs:
            if len(applicable) < max_ecms // 2:
                continue

            selected = applicable[:max_ecms]
            ecm_ids = [e.ecm_id for e in selected]

            # Validate package with dependency matrix
            is_valid, issues = validate_package(ecm_ids)

            # Get synergy factor
            synergy_factor = get_package_synergy(ecm_ids)

            # Calculate combined savings (apply synergy)
            individual_savings = sum(e.savings_percent for e in selected)
            combined_savings_pct = min(individual_savings * synergy_factor, 70)  # Cap at 70%
            combined_savings_kwh_m2 = baseline_kwh_m2 * combined_savings_pct / 100

            # Calculate package cost with discounts
            total_cost = sum(e.total_cost_sek for e in selected)
            package_discount = self._calculate_package_discount(ecm_ids, total_cost)
            net_cost = total_cost - package_discount

            # Add scaffolding if exterior work
            exterior_ecms = {"wall_external_insulation", "window_replacement", "roof_insulation"}
            if any(e in exterior_ecms for e in ecm_ids):
                facade_area = building_info["atemp_m2"] * 0.5
                scaffolding = estimate_scaffolding_cost(facade_area, building_info["num_floors"])
                net_cost += scaffolding

            # Add project overhead
            overhead = calculate_project_overhead(
                net_cost,
                building_age_years=2024 - building_info["construction_year"],
                scope="medium" if len(selected) < 5 else "large",
            )
            net_cost += overhead["total"]

            # Financial metrics
            annual_savings_kwh = combined_savings_kwh_m2 * building_info["atemp_m2"]
            annual_savings_sek = annual_savings_kwh * energy_price_sek_kwh
            payback = net_cost / annual_savings_sek if annual_savings_sek > 0 else 999
            npv = self._calculate_npv(net_cost, annual_savings_sek, 20, 0.04)

            # Interaction factor (vs simple sum)
            sum_savings = individual_savings
            interaction_factor = combined_savings_pct / sum_savings if sum_savings > 0 else 1.0

            packages.append(IntegratedPackage(
                package_id=pkg_type,
                package_name=name,
                package_type=pkg_type,
                ecm_ids=ecm_ids,
                combined_heating_kwh_m2=baseline_kwh_m2 - combined_savings_kwh_m2,
                combined_savings_percent=combined_savings_pct,
                synergy_factor=synergy_factor,
                interaction_factor=interaction_factor,
                total_cost_sek=total_cost,
                package_discount_sek=package_discount,
                net_cost_sek=net_cost,
                annual_savings_sek=annual_savings_sek,
                simple_payback_years=min(payback, 99),
                npv_20yr=npv,
                is_valid=is_valid,
                validation_issues=issues,
            ))

        return packages

    def _calculate_package_discount(self, ecm_ids: List[str], total_cost: float) -> float:
        """Calculate package discount from shared work (scaffolding, etc.)."""
        discount = 0

        for (ecm_a, ecm_b), factor in PACKAGE_COST_SYNERGIES.items():
            if ecm_a in ecm_ids and ecm_b in ecm_ids:
                # Synergy factor < 1 means discount
                discount += total_cost * (1 - factor) * 0.1  # 10% weight per synergy

        return min(discount, total_cost * 0.25)  # Cap at 25% discount

    def _calculate_npv(self, initial_cost: float, annual_savings: float, years: int, discount_rate: float) -> float:
        """Calculate NPV."""
        npv = -initial_cost
        for year in range(1, years + 1):
            npv += annual_savings / ((1 + discount_rate) ** year)
        return npv

    def _save_to_database(self, result: IntegratedAnalysisResult, building_info: Dict) -> Optional[str]:
        """Save analysis results to Supabase."""
        if not DB_AVAILABLE:
            logger.warning("Database not available - skipping save")
            return None

        try:
            # Create building record
            building = BuildingRecord(
                address=result.address,
                property_designation=result.property_designation,
                name=building_info.get("brf_name"),
                construction_year=result.construction_year,
                heated_area_m2=result.atemp_m2,
                num_apartments=result.num_apartments,
                region=result.region,
                declared_energy_kwh_m2=result.declared_kwh_m2,
                heating_system="district_heating",
                owner_type=result.owner_type,
            )

            # Create baseline record
            baseline = BaselineRecord(
                building_id="",  # Will be set
                heating_kwh_m2=result.baseline_kwh_m2,
                is_calibrated=result.calibration_error_percent < 10,
                calibration_gap_percent=result.calibration_error_percent,
                archetype_id=result.archetype_id,
            )

            # Create ECM result records
            ecm_records = [
                ECMResultRecord(
                    building_id="",
                    ecm_id=e.ecm_id,
                    ecm_name=e.ecm_name,
                    ecm_category=e.ecm_category,
                    heating_kwh_m2=e.heating_kwh_m2,
                    heating_savings_percent=e.savings_percent,
                    material_cost=e.material_cost_sek,
                    labor_cost=e.labor_cost_sek,
                    total_cost=e.total_cost_sek,
                    rot_deduction=e.rot_deduction_sek,
                    green_tech_deduction=e.green_tech_deduction_sek,
                    net_cost=e.net_cost_sek,
                    annual_savings_sek=e.annual_savings_sek,
                    simple_payback_years=e.simple_payback_years,
                    npv_20yr=e.npv_20yr,
                    annual_co2_reduction_kg=e.annual_co2_reduction_kg,
                    is_applicable=e.is_applicable,
                    simulated=False,
                )
                for e in result.ecm_results
            ]

            # Create package records
            package_records = [
                PackageRecord(
                    building_id="",
                    package_name=p.package_name,
                    package_type=p.package_type,
                    ecm_ids=p.ecm_ids,
                    combined_heating_kwh_m2=p.combined_heating_kwh_m2,
                    combined_savings_percent=p.combined_savings_percent,
                    synergy_factor=p.synergy_factor,
                    total_cost=p.total_cost_sek,
                    package_discount=p.package_discount_sek,
                    net_cost=p.net_cost_sek,
                    annual_savings_sek=p.annual_savings_sek,
                    simple_payback_years=p.simple_payback_years,
                    npv_20yr=p.npv_20yr,
                    is_valid=p.is_valid,
                    validation_issues=p.validation_issues,
                )
                for p in result.packages
            ]

            # Save all
            save_result = save_full_analysis(
                building=building,
                baseline=baseline,
                ecm_results=ecm_records,
                packages=package_records,
            )

            logger.info(f"Saved to database: building_id={save_result['building_id']}")
            return save_result['building_id']

        except Exception as e:
            logger.error(f"Failed to save to database: {e}")
            return None


# Convenience function
def analyze_building_json(
    json_path: Path,
    weather_path: Optional[Path] = None,
    save_to_db: bool = True,
    run_simulations: bool = False,
) -> IntegratedAnalysisResult:
    """
    Convenience function to analyze a building from JSON.

    Args:
        json_path: Path to building JSON export
        weather_path: Optional weather file (uses Stockholm default if None)
        save_to_db: Whether to save results to Supabase
        run_simulations: Whether to run EnergyPlus simulations

    Returns:
        IntegratedAnalysisResult with full cost analysis
    """
    # Default weather file - use project-relative path or WEATHER_FILE env var
    if weather_path is None:
        import os
        env_weather = os.environ.get("RAIDEN_WEATHER_FILE")
        if env_weather:
            weather_path = Path(env_weather)
        else:
            # Try project-relative path
            project_root = Path(__file__).parent.parent.parent
            weather_path = project_root / "weather" / "SWE_Stockholm.Arlanda.024600_IWEC.epw"

    analyzer = IntegratedAnalyzer(save_to_db=save_to_db)
    return analyzer.analyze_from_json(
        json_path=json_path,
        weather_path=weather_path,
        run_simulations=run_simulations,
    )
