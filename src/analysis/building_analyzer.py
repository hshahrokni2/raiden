"""
BuildingAnalyzer - End-to-end building energy analysis orchestrator.

This is the main entry point for the vision:
"Generate any Swedish building from annual report + energy declaration + public data,
 run baseline + all ECM packages"

Workflow:
1. Gather data from all sources
2. Build enhanced building context
3. Match archetype and adjust for existing measures
4. Generate baseline IDF
5. Run baseline simulation
6. Filter applicable ECMs (smart filtering)
7. Generate and run ECM scenarios
8. Calculate ROI and rank packages
9. Return comprehensive results
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import logging
import shutil
import tempfile

from ..core.building_context import (
    EnhancedBuildingContext,
    BuildingContextBuilder,
    SmartECMFilter,
    ExistingMeasure,
)
from ..baseline.archetypes import ArchetypeMatcher, SwedishArchetype
from ..baseline.generator import BaselineGenerator, BaselineModel
from ..baseline.calibrator import BaselineCalibrator, CalibrationResult
from ..ecm.catalog import ECMCatalog, ECM
from ..ecm.constraints import ConstraintEngine, BuildingContext as ConstraintContext
from ..ecm.idf_modifier import IDFModifier
from ..ecm.combinations import CombinationGenerator
from ..simulation.runner import SimulationRunner, SimulationResult
from ..simulation.results import ResultsParser, AnnualResults
from ..ingest.energidek_parser import EnergyDeclarationParser, EnergyDeclarationData
from ..geometry.building_geometry import BuildingGeometry, BuildingGeometryCalculator

logger = logging.getLogger(__name__)


@dataclass
class ECMScenarioResult:
    """Result of running a single ECM scenario."""
    ecm_id: str
    ecm_name: str
    parameters: Dict
    simulation_success: bool
    heating_kwh_m2: float
    savings_kwh_m2: float
    savings_percent: float
    error_message: Optional[str] = None


@dataclass
class AnalysisPackage:
    """A package of ECMs with combined results."""
    package_id: str
    name: str  # e.g., "Basic", "Standard", "Premium"
    ecm_ids: List[str]
    total_savings_kwh_m2: float
    total_savings_percent: float
    estimated_cost_sek: float
    simple_payback_years: float
    npv_25yr: float
    co2_savings_kg_m2: float


@dataclass
class BuildingAnalysisResult:
    """Complete analysis result for a building."""
    # Building identification
    address: str
    property_id: str

    # Building characteristics
    construction_year: int
    building_type: str
    facade_material: str
    atemp_m2: float
    archetype_name: str

    # Current performance (from energy declaration)
    declared_heating_kwh_m2: float

    # Baseline simulation
    baseline_heating_kwh_m2: float
    calibration_success: bool
    calibration_error_percent: float

    # Existing measures detected
    existing_measures: List[str]

    # ECM analysis
    applicable_ecms: List[str]
    excluded_ecms: List[Dict]  # [{'ecm': name, 'reason': ...}]
    ecm_results: List[ECMScenarioResult]

    # Recommended packages
    packages: List[AnalysisPackage]

    # Metadata
    analysis_duration_seconds: float
    data_sources: List[str]


class BuildingAnalyzer:
    """
    End-to-end building energy analysis.

    Usage:
        analyzer = BuildingAnalyzer()

        # From energy declaration PDF
        result = analyzer.analyze_from_declaration(
            pdf_path=Path("energideklaration.pdf"),
            weather_path=Path("stockholm.epw"),
        )

        # Or with all data sources
        result = analyzer.analyze(
            address="Sjöstadsparterren 2, Stockholm",
            energy_declaration_path=Path("energidek.pdf"),
            weather_path=Path("stockholm.epw"),
        )

        # Results include baseline, all ECMs, and ranked packages
        print(f"Baseline: {result.baseline_heating_kwh_m2} kWh/m²")
        for pkg in result.packages:
            print(f"{pkg.name}: saves {pkg.total_savings_percent:.0f}%")
    """

    def __init__(
        self,
        energyplus_path: Optional[str] = None,
        work_dir: Optional[Path] = None,
    ):
        """
        Initialize analyzer.

        Args:
            energyplus_path: Path to EnergyPlus executable (auto-detect if None)
            work_dir: Working directory for temporary files (temp dir if None)
        """
        self.energyplus_path = energyplus_path
        self.work_dir = work_dir or Path(tempfile.mkdtemp(prefix="raiden_"))

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
        self.declaration_parser = EnergyDeclarationParser()

    def analyze_from_declaration(
        self,
        pdf_path: Path,
        weather_path: Path,
        facade_material: Optional[str] = None,
        geometry_data: Optional[Dict] = None,
        run_all_ecms: bool = True,
        calibrate: bool = True,
    ) -> BuildingAnalysisResult:
        """
        Analyze building from energy declaration PDF.

        Args:
            pdf_path: Path to energy declaration PDF
            weather_path: Path to weather file (.epw)
            facade_material: Optional facade material override
            geometry_data: Optional geometry from OSM/Overture
            run_all_ecms: Whether to run all applicable ECM scenarios
            calibrate: Whether to calibrate baseline to declared energy

        Returns:
            Complete BuildingAnalysisResult
        """
        import time
        start_time = time.time()

        logger.info(f"Starting analysis from: {pdf_path}")

        # Step 1: Parse energy declaration
        declaration = self.declaration_parser.parse(pdf_path)
        logger.info(f"Parsed declaration: {declaration.energy_class} class, "
                    f"{declaration.specific_energy_kwh_sqm} kWh/m²")

        # Step 2: Build enhanced context
        context = self.context_builder.build_from_declaration(
            declaration=declaration,
            geometry_data=geometry_data,
            facade_material=facade_material,
        )

        # Step 3: Run analysis
        result = self._run_analysis(
            context=context,
            weather_path=weather_path,
            run_all_ecms=run_all_ecms,
            calibrate=calibrate,
        )

        result.analysis_duration_seconds = time.time() - start_time
        logger.info(f"Analysis complete in {result.analysis_duration_seconds:.1f}s")

        return result

    def analyze_from_context(
        self,
        context: EnhancedBuildingContext,
        weather_path: Path,
        run_all_ecms: bool = True,
        calibrate: bool = True,
    ) -> BuildingAnalysisResult:
        """
        Analyze building from pre-built context.

        Useful when you've already gathered data from multiple sources.
        """
        return self._run_analysis(
            context=context,
            weather_path=weather_path,
            run_all_ecms=run_all_ecms,
            calibrate=calibrate,
        )

    def _run_analysis(
        self,
        context: EnhancedBuildingContext,
        weather_path: Path,
        run_all_ecms: bool,
        calibrate: bool,
    ) -> BuildingAnalysisResult:
        """Core analysis workflow."""
        import time
        start_time = time.time()

        # Create working directory for this analysis
        analysis_dir = self.work_dir / f"analysis_{context.property_id or 'unknown'}"
        analysis_dir.mkdir(parents=True, exist_ok=True)

        # Ensure we have an archetype
        if context.archetype is None:
            context.archetype = self.archetype_matcher.match(
                construction_year=context.construction_year,
                building_type='multi_family',
                facade_material=context.facade_material,
            )
            logger.info(f"Matched archetype: {context.archetype.name}")

        # Step 1: Generate baseline IDF
        logger.info("Step 1: Generating baseline model...")
        geometry = self._build_geometry(context)
        baseline_model = self.generator.generate(
            geometry=geometry,
            archetype=context.archetype,
            output_dir=analysis_dir / "baseline",
            model_name="baseline",
        )

        # Step 2: Run baseline simulation
        logger.info("Step 2: Running baseline simulation...")
        baseline_result = self.runner.run(
            baseline_model.idf_path,
            weather_path,
            analysis_dir / "baseline_output",
        )

        baseline_kwh_m2 = 0.0
        if baseline_result.success:
            parsed = self.results_parser.parse(analysis_dir / "baseline_output")
            if parsed:
                baseline_kwh_m2 = parsed.heating_kwh_m2
                logger.info(f"Baseline heating: {baseline_kwh_m2:.1f} kWh/m²")

        # Step 3: Calibrate (if requested and we have declaration data)
        calibration_success = False
        calibration_error = 0.0
        declared_kwh_m2 = context.current_heating_kwh_m2

        if calibrate and declared_kwh_m2 > 0:
            logger.info("Step 3: Calibrating to declared energy...")
            cal_result = self.calibrator.calibrate(
                idf_path=baseline_model.idf_path,
                weather_path=weather_path,
                measured_heating_kwh_m2=declared_kwh_m2,
                output_dir=analysis_dir / "calibration",
            )
            calibration_success = cal_result.success
            calibration_error = cal_result.final_error_percent
            if cal_result.success:
                baseline_kwh_m2 = cal_result.calibrated_kwh_m2
                # Update baseline IDF to calibrated version
                if cal_result.calibrated_idf_path:
                    baseline_model.idf_path = cal_result.calibrated_idf_path

        # Step 4: Filter ECMs
        logger.info("Step 4: Filtering applicable ECMs...")
        all_ecms = self.ecm_catalog.all()
        filter_result = self.ecm_filter.filter_ecms(
            all_ecms=all_ecms,
            context=context,
            constraint_engine=self.constraint_engine,
        )

        applicable_ecms = filter_result['applicable']
        logger.info(f"Applicable ECMs: {len(applicable_ecms)}")
        logger.info(f"Already done: {len(filter_result['already_done'])}")
        logger.info(f"Not applicable: {len(filter_result['not_applicable'])}")

        # Step 5: Run ECM scenarios
        ecm_results = []
        if run_all_ecms:
            logger.info("Step 5: Running ECM scenarios...")
            ecm_results = self._run_ecm_scenarios(
                baseline_idf=baseline_model.idf_path,
                weather_path=weather_path,
                applicable_ecms=applicable_ecms,
                baseline_kwh_m2=baseline_kwh_m2,
                analysis_dir=analysis_dir,
            )

        # Step 6: Generate packages
        logger.info("Step 6: Generating packages...")
        packages = self._generate_packages(
            ecm_results=ecm_results,
            baseline_kwh_m2=baseline_kwh_m2,
            context=context,
        )

        # Build result
        return BuildingAnalysisResult(
            address=context.address,
            property_id=context.property_id,
            construction_year=context.construction_year,
            building_type=context.building_type.value if hasattr(context.building_type, 'value') else str(context.building_type),
            facade_material=context.facade_material,
            atemp_m2=context.atemp_m2,
            archetype_name=context.archetype.name if context.archetype else "Unknown",
            declared_heating_kwh_m2=declared_kwh_m2,
            baseline_heating_kwh_m2=baseline_kwh_m2,
            calibration_success=calibration_success,
            calibration_error_percent=calibration_error,
            existing_measures=[m.value for m in context.existing_measures],
            applicable_ecms=[e.id for e in applicable_ecms],
            excluded_ecms=[
                {'ecm': item['ecm'].id, 'reason': item.get('reason', str(item.get('reasons', [])))}
                for item in filter_result['already_done'] + filter_result['not_applicable']
            ],
            ecm_results=ecm_results,
            packages=packages,
            analysis_duration_seconds=time.time() - start_time,
            data_sources=['energy_declaration'],
        )

    def _build_geometry(self, context: EnhancedBuildingContext) -> BuildingGeometry:
        """Build geometry from context."""
        # If we have detailed geometry, use it
        if context.wall_area_m2 > 0 and context.roof_area_m2 > 0:
            return BuildingGeometry(
                footprint_area_m2=context.atemp_m2 / max(1, context.floors),
                gross_floor_area_m2=context.atemp_m2,
                floors=context.floors,
                height_m=context.floors * context.floor_height_m,
                floor_height_m=context.floor_height_m,
                perimeter_m=context.wall_area_m2 / (context.floors * context.floor_height_m),
                wall_area_m2=context.wall_area_m2,
                window_area_m2=context.window_area_m2,
                roof_area_m2=context.roof_area_m2,
            )

        # Otherwise, estimate from Atemp
        floors = context.floors or 4  # Default 4 floors
        floor_height = context.floor_height_m or 2.7
        footprint = context.atemp_m2 / floors

        # Assume rectangular building with 2:1 aspect ratio
        import math
        width = math.sqrt(footprint / 2)
        length = 2 * width
        perimeter = 2 * (width + length)
        wall_area = perimeter * floors * floor_height
        wwr = context.window_to_wall_ratio or 0.15
        window_area = wall_area * wwr

        return BuildingGeometry(
            footprint_area_m2=footprint,
            gross_floor_area_m2=context.atemp_m2,
            floors=floors,
            height_m=floors * floor_height,
            floor_height_m=floor_height,
            perimeter_m=perimeter,
            wall_area_m2=wall_area,
            window_area_m2=window_area,
            roof_area_m2=footprint,
        )

    def _run_ecm_scenarios(
        self,
        baseline_idf: Path,
        weather_path: Path,
        applicable_ecms: List[ECM],
        baseline_kwh_m2: float,
        analysis_dir: Path,
    ) -> List[ECMScenarioResult]:
        """Run individual ECM scenarios."""
        results = []

        for ecm in applicable_ecms:
            logger.info(f"  Running ECM: {ecm.name}")

            # Get default parameters for this ECM
            params = self._get_default_ecm_params(ecm)

            try:
                # Apply ECM to create modified IDF
                ecm_dir = analysis_dir / f"ecm_{ecm.id}"
                ecm_dir.mkdir(parents=True, exist_ok=True)

                modified_idf = self.idf_modifier.apply_single(
                    baseline_idf=baseline_idf,
                    ecm_id=ecm.id,
                    params=params,
                    output_dir=ecm_dir,
                )

                # Run simulation
                sim_result = self.runner.run(
                    modified_idf,
                    weather_path,
                    ecm_dir / "output",
                )

                if sim_result.success:
                    parsed = self.results_parser.parse(ecm_dir / "output")
                    if parsed:
                        heating_kwh_m2 = parsed.heating_kwh_m2
                        savings = baseline_kwh_m2 - heating_kwh_m2
                        savings_pct = (savings / baseline_kwh_m2 * 100) if baseline_kwh_m2 > 0 else 0

                        results.append(ECMScenarioResult(
                            ecm_id=ecm.id,
                            ecm_name=ecm.name,
                            parameters=params,
                            simulation_success=True,
                            heating_kwh_m2=heating_kwh_m2,
                            savings_kwh_m2=savings,
                            savings_percent=savings_pct,
                        ))
                        logger.info(f"    → {heating_kwh_m2:.1f} kWh/m² (saves {savings_pct:.1f}%)")
                        continue

                # Simulation failed
                results.append(ECMScenarioResult(
                    ecm_id=ecm.id,
                    ecm_name=ecm.name,
                    parameters=params,
                    simulation_success=False,
                    heating_kwh_m2=0,
                    savings_kwh_m2=0,
                    savings_percent=0,
                    error_message=sim_result.error_message,
                ))

            except Exception as e:
                logger.warning(f"    ECM {ecm.id} failed: {e}")
                results.append(ECMScenarioResult(
                    ecm_id=ecm.id,
                    ecm_name=ecm.name,
                    parameters=params,
                    simulation_success=False,
                    heating_kwh_m2=0,
                    savings_kwh_m2=0,
                    savings_percent=0,
                    error_message=str(e),
                ))

        return results

    def _get_default_ecm_params(self, ecm: ECM) -> Dict:
        """Get default parameters for an ECM."""
        # Default parameter sets for each ECM
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
        return defaults.get(ecm.id, {})

    def _generate_packages(
        self,
        ecm_results: List[ECMScenarioResult],
        baseline_kwh_m2: float,
        context: EnhancedBuildingContext,
    ) -> List[AnalysisPackage]:
        """Generate ECM packages (Basic, Standard, Premium)."""
        packages = []

        # Filter successful ECMs and sort by savings
        successful = [r for r in ecm_results if r.simulation_success and r.savings_kwh_m2 > 0]
        successful.sort(key=lambda x: x.savings_kwh_m2, reverse=True)

        if not successful:
            return packages

        # Swedish energy prices (2024)
        energy_price_sek_kwh = 1.5  # Approx district heating
        co2_factor_kg_kwh = 0.05  # District heating Stockholm

        # Basic package: Top 1-2 ECMs
        basic_ecms = successful[:min(2, len(successful))]
        if basic_ecms:
            packages.append(self._create_package(
                'basic', 'Basic', basic_ecms,
                baseline_kwh_m2, context.atemp_m2,
                energy_price_sek_kwh, co2_factor_kg_kwh,
            ))

        # Standard package: Top 3-4 ECMs
        if len(successful) >= 3:
            standard_ecms = successful[:min(4, len(successful))]
            packages.append(self._create_package(
                'standard', 'Standard', standard_ecms,
                baseline_kwh_m2, context.atemp_m2,
                energy_price_sek_kwh, co2_factor_kg_kwh,
            ))

        # Premium package: All applicable ECMs
        if len(successful) >= 5:
            packages.append(self._create_package(
                'premium', 'Premium', successful,
                baseline_kwh_m2, context.atemp_m2,
                energy_price_sek_kwh, co2_factor_kg_kwh,
            ))

        return packages

    def _create_package(
        self,
        pkg_id: str,
        name: str,
        ecms: List[ECMScenarioResult],
        baseline_kwh_m2: float,
        atemp_m2: float,
        energy_price: float,
        co2_factor: float,
    ) -> AnalysisPackage:
        """Create an analysis package from ECM results."""
        # Simple additive savings (not quite right for combined ECMs, but reasonable estimate)
        # Real implementation would run combined scenario
        total_savings_kwh_m2 = sum(e.savings_kwh_m2 for e in ecms) * 0.8  # 80% interaction factor
        total_savings_pct = (total_savings_kwh_m2 / baseline_kwh_m2 * 100) if baseline_kwh_m2 > 0 else 0

        # Estimate costs (simplified)
        estimated_cost = self._estimate_package_cost(ecms, atemp_m2)

        # Annual savings
        annual_savings_kwh = total_savings_kwh_m2 * atemp_m2
        annual_savings_sek = annual_savings_kwh * energy_price

        # Payback
        payback = estimated_cost / annual_savings_sek if annual_savings_sek > 0 else 999

        # Simple NPV (25 years, 4% discount)
        npv = self._calculate_npv(estimated_cost, annual_savings_sek, 25, 0.04)

        # CO2 savings
        co2_savings = total_savings_kwh_m2 * co2_factor

        return AnalysisPackage(
            package_id=pkg_id,
            name=name,
            ecm_ids=[e.ecm_id for e in ecms],
            total_savings_kwh_m2=total_savings_kwh_m2,
            total_savings_percent=total_savings_pct,
            estimated_cost_sek=estimated_cost,
            simple_payback_years=payback,
            npv_25yr=npv,
            co2_savings_kg_m2=co2_savings,
        )

    def _estimate_package_cost(self, ecms: List[ECMScenarioResult], atemp_m2: float) -> float:
        """Estimate total cost for a package of ECMs."""
        # Simplified cost estimates (SEK per m² Atemp)
        cost_per_m2 = {
            'wall_external_insulation': 2000,
            'wall_internal_insulation': 800,
            'roof_insulation': 300,
            'window_replacement': 500,
            'air_sealing': 100,
            'ftx_upgrade': 200,
            'ftx_installation': 1500,
            'demand_controlled_ventilation': 150,
            'solar_pv': 400,
            'led_lighting': 50,
            'smart_thermostats': 30,
            'heat_pump_integration': 600,
        }

        total = 0
        for ecm in ecms:
            cost = cost_per_m2.get(ecm.ecm_id, 200)
            total += cost * atemp_m2

        return total

    def _calculate_npv(
        self,
        initial_cost: float,
        annual_savings: float,
        years: int,
        discount_rate: float,
    ) -> float:
        """Calculate Net Present Value."""
        npv = -initial_cost
        for year in range(1, years + 1):
            npv += annual_savings / ((1 + discount_rate) ** year)
        return npv


# Convenience function
def analyze_building(
    energy_declaration_path: Path,
    weather_path: Path,
    facade_material: Optional[str] = None,
) -> BuildingAnalysisResult:
    """
    Convenience function to analyze a building.

    Args:
        energy_declaration_path: Path to energy declaration PDF
        weather_path: Path to weather file (.epw)
        facade_material: Optional facade material override

    Returns:
        Complete BuildingAnalysisResult
    """
    analyzer = BuildingAnalyzer()
    return analyzer.analyze_from_declaration(
        pdf_path=energy_declaration_path,
        weather_path=weather_path,
        facade_material=facade_material,
    )
