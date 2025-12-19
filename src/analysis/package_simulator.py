"""
Package Simulator - Simulate combined ECM packages.

Instead of estimating combined savings with an interaction factor,
this module creates actual combined IDF files and runs EnergyPlus
to get physics-based package savings.

Architecture:
1. Individual ECMs are simulated first
2. ECMs sorted by ROI (using Swedish cost database)
3. Packages created by investment tier (Steg 0-3):
   - Steg 0: Zero-cost operational measures (ALWAYS FIRST)
   - Steg 1: Quick wins (< 500k SEK)
   - Steg 2: Standard (500k-2M SEK)
   - Steg 3: Premium (> 2M SEK)
4. Combined IDFs generated using apply_multiple()
5. Package simulations run to get actual combined savings
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
from pathlib import Path
import logging

from ..ecm.idf_modifier import IDFModifier
from ..roi.costs_sweden import SwedishCosts, ECM_COSTS, ENERGY_PRICES, CostCategory

logger = logging.getLogger(__name__)


@dataclass
class PackageECM:
    """ECM within a package."""
    id: str
    name: str
    name_sv: str
    individual_savings_percent: float
    individual_savings_kwh_m2: float
    cost_sek: float
    payback_years: float
    params: Dict[str, Any]


@dataclass
class SimulatedPackage:
    """Package with simulation results."""
    id: str
    name: str
    name_sv: str
    description: str
    description_sv: str
    ecms: List[PackageECM]
    # Individual sum (for comparison)
    sum_individual_savings_percent: float
    sum_individual_savings_kwh_m2: float
    # Actual simulated (the real physics)
    simulated_savings_percent: float
    simulated_savings_kwh_m2: float
    simulated_heating_kwh_m2: float
    # Interaction factor (actual vs sum)
    interaction_factor: float
    # Economics
    total_cost_sek: float
    annual_savings_sek: float
    simple_payback_years: float
    co2_reduction_kg_m2: float
    # Simulation metadata
    idf_path: Optional[Path] = None
    simulation_success: bool = False


class PackageSimulator:
    """
    Create and simulate ECM packages.

    Usage:
        simulator = PackageSimulator()

        # From individual ECM simulation results
        packages = simulator.create_and_simulate_packages(
            ecm_results=ecm_results,      # List of dicts with id, savings_percent, etc.
            baseline_idf=Path('./baseline.idf'),
            baseline_kwh_m2=36.3,
            atemp_m2=2240,
            weather_path=Path('./stockholm.epw'),
            output_dir=Path('./packages'),
        )
    """

    # Package definitions - Investment-tier based (Steg 0-3)
    # Steg 0 is always first (zero-cost), then capital ECMs by investment level
    PACKAGE_DEFS = {
        'steg0_zero_cost': {
            'name': 'Step 0: Zero-Cost',
            'name_sv': 'Steg 0: Nollkostnad',
            'description': 'Operational optimizations - DO THIS FIRST',
            'description_sv': 'Driftoptimering - GÖR DETTA FÖRST',
            'order': 0,
            'category_filter': CostCategory.ZERO_COST,
            'include_low_savings': True,  # Include even 0% thermal savings
        },
        'steg1_basic': {
            'name': 'Step 1: Quick Wins',
            'name_sv': 'Steg 1: Snabba vinster',
            'description': 'Low investment, fast payback (< 500k SEK)',
            'description_sv': 'Låg investering, snabb återbetalning (< 500k SEK)',
            'order': 1,
            'max_investment_sek': 500_000,
            'exclude_categories': [CostCategory.ZERO_COST],
        },
        'steg2_standard': {
            'name': 'Step 2: Standard Package',
            'name_sv': 'Steg 2: Standardpaket',
            'description': 'Balanced investment (500k - 2M SEK)',
            'description_sv': 'Balanserad investering (500k - 2M SEK)',
            'order': 2,
            'min_investment_sek': 500_000,
            'max_investment_sek': 2_000_000,
            'exclude_categories': [CostCategory.ZERO_COST],
        },
        'steg3_premium': {
            'name': 'Step 3: Premium Package',
            'name_sv': 'Steg 3: Premiumpaket',
            'description': 'Major renovation (> 2M SEK)',
            'description_sv': 'Större renovering (> 2M SEK)',
            'order': 3,
            'min_investment_sek': 2_000_000,
            'exclude_categories': [CostCategory.ZERO_COST],
        },
    }

    def __init__(
        self,
        costs: Optional[SwedishCosts] = None,
        energy_type: str = 'district_heating',
        co2_intensity_kg_kwh: float = 0.05,
    ):
        """
        Initialize package simulator.

        Args:
            costs: Swedish cost database (defaults to built-in)
            energy_type: Energy type for price lookup
            co2_intensity_kg_kwh: CO2 intensity for Swedish grid
        """
        self.costs = costs or SwedishCosts()
        self.energy_type = energy_type
        self.co2_intensity = co2_intensity_kg_kwh
        self.modifier = IDFModifier()

    def create_and_simulate_packages(
        self,
        ecm_results: List[Dict],
        baseline_idf: Path,
        baseline_kwh_m2: float,
        atemp_m2: float,
        weather_path: Path,
        output_dir: Path,
        run_simulation: bool = True,
        ecm_params: Optional[Dict[str, Dict]] = None,
    ) -> List[SimulatedPackage]:
        """
        Create packages and simulate them.

        Args:
            ecm_results: List of ECM results with 'id', 'savings_percent', 'name', etc.
            baseline_idf: Path to baseline IDF file
            baseline_kwh_m2: Baseline heating demand
            atemp_m2: Building area
            weather_path: Path to weather file
            output_dir: Directory for package IDFs and results
            run_simulation: Whether to run EnergyPlus (False = just create IDFs)
            ecm_params: Optional override for ECM parameters

        Returns:
            List of SimulatedPackage with results
        """
        # Step 1: Enrich ECM results with costs and sort by ROI
        enriched_ecms = self._enrich_and_sort_ecms(
            ecm_results, baseline_kwh_m2, atemp_m2
        )

        logger.info(f"Enriched {len(enriched_ecms)} ECMs for package creation")

        # Step 2: Create package definitions (sorted by order: Steg 0 first)
        packages = []
        sorted_pkg_defs = sorted(
            self.PACKAGE_DEFS.items(),
            key=lambda x: x[1].get('order', 99)
        )

        for pkg_id, pkg_def in sorted_pkg_defs:
            pkg_ecms = self._select_package_ecms(
                enriched_ecms,
                pkg_def,
                ecm_params or {},
            )
            # Zero-cost package: include even if no thermal savings
            min_ecms = 1 if pkg_def.get('include_low_savings') else 1
            if len(pkg_ecms) >= min_ecms:
                package = self._create_package(
                    pkg_id, pkg_def, pkg_ecms, baseline_kwh_m2, atemp_m2
                )
                packages.append(package)
                logger.info(f"Created {pkg_id} package with {len(pkg_ecms)} ECMs")

        # Step 3: Generate combined IDFs
        output_dir.mkdir(parents=True, exist_ok=True)
        for package in packages:
            ecm_list = [
                (ecm.id, ecm.params) for ecm in package.ecms
            ]
            try:
                idf_path = self.modifier.apply_multiple(
                    baseline_idf=baseline_idf,
                    ecms=ecm_list,
                    output_dir=output_dir,
                    output_name=f"package_{package.id}"
                )
                package.idf_path = idf_path
                logger.info(f"Generated IDF for {package.id}: {idf_path}")
            except Exception as e:
                logger.error(f"Failed to create IDF for {package.id}: {e}")

        # Step 4: Run simulations
        if run_simulation:
            self._run_package_simulations(
                packages, weather_path, output_dir, baseline_kwh_m2, atemp_m2
            )

        return packages

    def _enrich_and_sort_ecms(
        self,
        ecm_results: List[Dict],
        baseline_kwh_m2: float,
        atemp_m2: float,
    ) -> List[Dict]:
        """Enrich ECM results with costs and sort by ROI."""
        energy_price = self.costs.energy_price(self.energy_type)
        enriched = []

        for ecm in ecm_results:
            ecm_id = ecm.get('id', ecm.get('ecm_id', 'unknown'))
            savings_pct = ecm.get('savings_percent', 0)

            # Get cost data to check category
            cost_data = ECM_COSTS.get(ecm_id)
            is_zero_cost = cost_data and cost_data.category == CostCategory.ZERO_COST

            # Skip ECMs with negative savings (like LED heating increase)
            # BUT keep zero-cost ECMs even with 0% thermal savings
            if savings_pct < 0 and not is_zero_cost:
                logger.debug(f"Skipping {ecm_id}: negative savings {savings_pct:.1f}%")
                continue
            if savings_pct < 1 and not is_zero_cost:
                logger.debug(f"Skipping {ecm_id}: savings {savings_pct:.1f}% too low")
                continue

            # Calculate savings
            savings_kwh_m2 = baseline_kwh_m2 * (savings_pct / 100)
            annual_savings_sek = savings_kwh_m2 * atemp_m2 * energy_price.price_sek_per_kwh

            # Get cost from database
            cost_data = ECM_COSTS.get(ecm_id)
            if cost_data:
                # Calculate quantity based on unit type
                if cost_data.unit == 'm² floor':
                    quantity = atemp_m2
                elif cost_data.unit == 'm² wall':
                    quantity = atemp_m2 * 0.8  # Estimate wall area
                elif cost_data.unit == 'm² roof':
                    quantity = atemp_m2 / 7  # Single floor footprint
                elif cost_data.unit == 'm² window':
                    quantity = atemp_m2 * 0.17  # WWR estimate
                elif cost_data.unit == 'kWp':
                    quantity = atemp_m2 / 7 * 0.15  # Rough PV sizing
                elif cost_data.unit == 'kW':
                    quantity = 50  # Estimate heat pump size
                elif cost_data.unit == 'building':
                    quantity = 1
                elif cost_data.unit == 'radiator':
                    quantity = atemp_m2 / 15  # ~1 radiator per 15 m²
                else:
                    quantity = 1

                total_cost = self.costs.ecm_cost(ecm_id, quantity)
                category = cost_data.category
            else:
                # Fallback to flat cost estimate
                total_cost = 300 * atemp_m2
                category = CostCategory.MEDIUM_COST

            # Calculate payback
            payback = total_cost / annual_savings_sek if annual_savings_sek > 0 else 999

            enriched.append({
                **ecm,
                'id': ecm_id,
                'savings_kwh_m2': savings_kwh_m2,
                'annual_savings_sek': annual_savings_sek,
                'cost_sek': total_cost,
                'payback_years': payback,
                'category': category,
            })

        # Sort by payback (best ROI first)
        return sorted(enriched, key=lambda x: x['payback_years'])

    def _select_package_ecms(
        self,
        enriched_ecms: List[Dict],
        pkg_def: Dict,
        ecm_params: Dict[str, Dict],
    ) -> List[PackageECM]:
        """Select ECMs for a package based on investment-tier definition."""
        selected = []
        cumulative_cost = 0

        for ecm in enriched_ecms:
            ecm_category = ecm.get('category')
            ecm_cost = ecm.get('cost_sek', 0)

            # Category filter (for zero-cost package)
            if 'category_filter' in pkg_def:
                if ecm_category != pkg_def['category_filter']:
                    continue

            # Exclude categories (for capital packages)
            if 'exclude_categories' in pkg_def:
                if ecm_category in pkg_def['exclude_categories']:
                    continue

            # Investment range filter
            # For cumulative packages, check if adding this ECM stays within range
            min_inv = pkg_def.get('min_investment_sek', 0)
            max_inv = pkg_def.get('max_investment_sek', float('inf'))

            # Check if this ECM fits in the investment tier
            if 'min_investment_sek' in pkg_def or 'max_investment_sek' in pkg_def:
                # For cumulative approach: include ECMs that bring total within range
                new_total = cumulative_cost + ecm_cost

                # Skip if we'd exceed max investment
                if new_total > max_inv:
                    continue

                # Add ECM
                cumulative_cost = new_total

            # Payback filter (optional, for legacy support)
            if pkg_def.get('max_payback_years'):
                if ecm['payback_years'] > pkg_def['max_payback_years']:
                    continue

            # Get parameters
            params = ecm_params.get(ecm['id'], ecm.get('params', {}))

            selected.append(PackageECM(
                id=ecm['id'],
                name=ecm.get('name', ecm['id']),
                name_sv=ecm.get('name_sv', ecm.get('name', ecm['id'])),
                individual_savings_percent=ecm['savings_percent'],
                individual_savings_kwh_m2=ecm['savings_kwh_m2'],
                cost_sek=ecm_cost,
                payback_years=ecm['payback_years'],
                params=params,
            ))

            # Count limit (optional)
            if pkg_def.get('count') and len(selected) >= pkg_def['count']:
                break

        # For packages with min_investment, only return if we meet minimum
        if 'min_investment_sek' in pkg_def:
            total = sum(e.cost_sek for e in selected)
            if total < pkg_def['min_investment_sek']:
                return []  # Not enough investment for this tier

        return selected

    def _create_package(
        self,
        pkg_id: str,
        pkg_def: Dict,
        ecms: List[PackageECM],
        baseline_kwh_m2: float,
        atemp_m2: float,
    ) -> SimulatedPackage:
        """Create a package from selected ECMs."""
        energy_price = self.costs.energy_price(self.energy_type)

        # Sum individual savings (for comparison)
        sum_savings_pct = sum(e.individual_savings_percent for e in ecms)
        sum_savings_kwh = sum(e.individual_savings_kwh_m2 for e in ecms)

        # Cap at realistic maximum
        sum_savings_pct = min(sum_savings_pct, 60)

        # Total cost
        total_cost = sum(e.cost_sek for e in ecms)

        # Estimated annual savings (will be replaced by simulation)
        annual_savings = sum_savings_kwh * atemp_m2 * energy_price.price_sek_per_kwh

        # Payback
        payback = total_cost / annual_savings if annual_savings > 0 else 999

        # CO2
        co2_reduction = sum_savings_kwh * self.co2_intensity

        return SimulatedPackage(
            id=pkg_id,
            name=pkg_def['name'],
            name_sv=pkg_def['name_sv'],
            description=pkg_def['description'],
            description_sv=pkg_def['description_sv'],
            ecms=ecms,
            sum_individual_savings_percent=sum_savings_pct,
            sum_individual_savings_kwh_m2=sum_savings_kwh,
            # Placeholder until simulation
            simulated_savings_percent=sum_savings_pct * 0.7,  # Conservative estimate
            simulated_savings_kwh_m2=sum_savings_kwh * 0.7,
            simulated_heating_kwh_m2=baseline_kwh_m2 * (1 - sum_savings_pct * 0.7 / 100),
            interaction_factor=0.7,
            total_cost_sek=total_cost,
            annual_savings_sek=annual_savings * 0.7,
            simple_payback_years=payback / 0.7,
            co2_reduction_kg_m2=co2_reduction * 0.7,
        )

    def _run_package_simulations(
        self,
        packages: List[SimulatedPackage],
        weather_path: Path,
        output_dir: Path,
        baseline_kwh_m2: float,
        atemp_m2: float,
    ):
        """Run EnergyPlus simulations for packages."""
        try:
            from ..simulation.runner import SimulationRunner
            runner = SimulationRunner()
        except ImportError as e:
            logger.warning(f"Simulation runner not available: {e}")
            return

        energy_price = self.costs.energy_price(self.energy_type)

        for package in packages:
            if not package.idf_path or not package.idf_path.exists():
                logger.warning(f"No IDF for package {package.id}")
                continue

            sim_output = output_dir / f"sim_{package.id}"
            logger.info(f"Running simulation for {package.id}...")

            try:
                result = runner.run_and_parse(
                    idf_path=package.idf_path,
                    weather_path=weather_path,
                    output_dir=sim_output,
                    timeout_seconds=300
                )

                if result.success and result.parsed_results:
                    # Update with actual simulation results
                    simulated_heating = result.parsed_results.heating_kwh_m2
                    actual_savings_kwh = baseline_kwh_m2 - simulated_heating
                    actual_savings_pct = (actual_savings_kwh / baseline_kwh_m2) * 100

                    # Calculate actual interaction factor
                    if package.sum_individual_savings_percent > 0:
                        interaction = actual_savings_pct / package.sum_individual_savings_percent
                    else:
                        interaction = 1.0

                    # Update package
                    package.simulated_heating_kwh_m2 = simulated_heating
                    package.simulated_savings_kwh_m2 = actual_savings_kwh
                    package.simulated_savings_percent = actual_savings_pct
                    package.interaction_factor = interaction
                    package.simulation_success = True

                    # Update economics with actual savings
                    package.annual_savings_sek = (
                        actual_savings_kwh * atemp_m2 * energy_price.price_sek_per_kwh
                    )
                    package.simple_payback_years = (
                        package.total_cost_sek / package.annual_savings_sek
                        if package.annual_savings_sek > 0 else 999
                    )
                    package.co2_reduction_kg_m2 = actual_savings_kwh * self.co2_intensity

                    logger.info(
                        f"  {package.id}: {simulated_heating:.1f} kWh/m² "
                        f"(saves {actual_savings_pct:.1f}%, interaction={interaction:.2f})"
                    )
                else:
                    logger.error(f"  {package.id}: Simulation failed - {result.error_message}")

            except Exception as e:
                logger.error(f"  {package.id}: Simulation error - {e}")


def create_packages_from_ecm_results(
    ecm_results: List[Dict],
    baseline_idf: Path,
    baseline_kwh_m2: float,
    atemp_m2: float,
    weather_path: Path,
    output_dir: Path,
    run_simulation: bool = True,
) -> List[SimulatedPackage]:
    """
    Convenience function to create and simulate packages.

    Args:
        ecm_results: Results from individual ECM simulations
        baseline_idf: Path to baseline IDF
        baseline_kwh_m2: Baseline heating demand
        atemp_m2: Building area
        weather_path: Weather file path
        output_dir: Output directory
        run_simulation: Whether to run EnergyPlus

    Returns:
        List of SimulatedPackage objects
    """
    simulator = PackageSimulator()
    return simulator.create_and_simulate_packages(
        ecm_results=ecm_results,
        baseline_idf=baseline_idf,
        baseline_kwh_m2=baseline_kwh_m2,
        atemp_m2=atemp_m2,
        weather_path=weather_path,
        output_dir=output_dir,
        run_simulation=run_simulation,
    )
