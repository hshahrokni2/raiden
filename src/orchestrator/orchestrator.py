"""
RaidenOrchestrator - Portfolio-scale building analysis engine.

Tiered processing architecture:
- Tier 1 (Fast): GeoJSON lookup, 10 buildings/sec
- Tier 2 (Standard): Surrogate prediction, 50 concurrent
- Tier 3 (Deep): EnergyPlus simulation, 10 concurrent

Agentic QC triggers on low confidence results.
"""

import asyncio
import logging
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ============================================================================
# MODULE-LEVEL FUNCTION FOR ProcessPoolExecutor
# Must be at module level (not inside class) to be picklable
# ============================================================================


def _run_deep_analysis_in_process(
    address: str,
    building_data: Optional[Dict[str, Any]],
    enable_energyplus: bool = True,
) -> Dict[str, Any]:
    """
    Run deep analysis in a separate process.

    This function MUST be at module level for ProcessPoolExecutor to pickle it.
    It runs synchronously in its own process.

    Args:
        address: Building address
        building_data: Pre-fetched building data dict (or None)
        enable_energyplus: Whether to run E+ simulations

    Returns:
        Dict with analysis results (picklable)
    """
    import os
    import time
    start = time.time()

    result = {
        "address": address,
        "tier": "deep",
        "success": False,
        "error": None,
        "processing_time_sec": 0.0,
    }

    try:
        # Import inside function to avoid pickling issues
        from src.core.address_pipeline import BuildingDataFetcher

        # Fetch building data if not provided
        if not building_data:
            fetcher = BuildingDataFetcher()
            fetched = fetcher.fetch(address)
            if fetched:
                building_data = {
                    "address": fetched.address,
                    "construction_year": fetched.construction_year,
                    "atemp_m2": fetched.atemp_m2,
                    "energy_class": getattr(fetched, "energy_class", None),
                    "facade_material": getattr(fetched, "facade_material", None),
                    "heating_system": getattr(fetched, "heating_system", None),
                    "ventilation_type": getattr(fetched, "ventilation_type", None),
                    "lat": getattr(fetched, "lat", None),
                    "lon": getattr(fetched, "lon", None),
                }

        if building_data:
            result["construction_year"] = building_data.get("construction_year")
            result["atemp_m2"] = building_data.get("atemp_m2")
            result["energy_class"] = building_data.get("energy_class")

        # Try full pipeline if available
        try:
            from src.analysis.full_pipeline import FullPipelineAnalyzer

            google_api_key = os.environ.get("GOOGLE_API_KEY")
            mapillary_token = os.environ.get("MAPILLARY_TOKEN")

            pipeline = FullPipelineAnalyzer(
                google_api_key=google_api_key,
                mapillary_token=mapillary_token,
                use_bayesian_calibration=True,
            )

            # Run pipeline synchronously (we're already in a separate process)
            # FullPipelineAnalyzer.analyze() is async, so we need to run it in event loop
            import asyncio

            async def _run_pipeline():
                return await pipeline.analyze(
                    address=address,
                    building_data=building_data,
                    run_simulations=enable_energyplus,
                )

            # Create new event loop for this process
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                pipeline_result = loop.run_until_complete(_run_pipeline())
            finally:
                loop.close()

            if pipeline_result:
                result["archetype_id"] = pipeline_result.get("archetype_id")
                result["archetype_confidence"] = pipeline_result.get("archetype_confidence", 0.0)
                result["current_kwh_m2"] = pipeline_result.get("calibrated_kwh_m2")

                # ECM results
                ecm_results = pipeline_result.get("ecm_results", [])
                result["recommended_ecms"] = [
                    {
                        "id": ecm.get("id"),
                        "name": ecm.get("name"),
                        "savings_kwh_m2": ecm.get("savings_kwh_m2", 0),
                        "cost_sek": ecm.get("cost_sek", 0),
                    }
                    for ecm in ecm_results
                ]

                result["total_savings_kwh_m2"] = pipeline_result.get("total_savings_kwh_m2", 0.0)
                result["total_investment_sek"] = pipeline_result.get("total_investment_sek", 0.0)
                result["npv_sek"] = pipeline_result.get("npv_sek")
                result["simple_payback_years"] = pipeline_result.get("payback_years")

            result["success"] = True

        except ImportError as e:
            # Full pipeline not available, use surrogate-based fallback
            logger.warning(f"Full pipeline not available in process: {e}")
            result["error"] = f"Pipeline unavailable: {e}"
            result["success"] = False

    except Exception as e:
        logger.error(f"Deep analysis failed in process for {address}: {e}")
        result["success"] = False
        result["error"] = str(e)

    result["processing_time_sec"] = time.time() - start
    return result


class AnalysisTier(Enum):
    """Analysis tier for a building."""

    SKIP = "skip"  # Already optimized (energy class A/B)
    FAST = "fast"  # Tier 1: GeoJSON lookup only
    STANDARD = "standard"  # Tier 2: Surrogate-based prediction
    DEEP = "deep"  # Tier 3: Full EnergyPlus simulation


@dataclass
class TierConfig:
    """Configuration for analysis tiers."""

    # Tier 1 (Fast) settings
    skip_energy_classes: Tuple[str, ...] = ("A", "B")  # Already optimized

    # Tier 2 (Standard) settings
    standard_workers: int = 50
    surrogate_confidence_threshold: float = 0.70

    # Tier 3 (Deep) settings
    deep_workers: int = 10
    enable_energyplus: bool = True

    # QC thresholds
    wwr_confidence_threshold: float = 0.60
    material_confidence_threshold: float = 0.70
    archetype_score_threshold: float = 50.0

    # Processing limits
    max_buildings_per_batch: int = 100
    timeout_per_building_sec: float = 300.0


@dataclass
class BuildingResult:
    """Result for a single building analysis."""

    address: str
    tier: AnalysisTier
    success: bool = True

    # Building data (from GeoJSON or fallback)
    construction_year: Optional[int] = None
    atemp_m2: Optional[float] = None
    energy_class: Optional[str] = None
    current_kwh_m2: Optional[float] = None

    # Archetype matching
    archetype_id: Optional[str] = None
    archetype_confidence: float = 0.0

    # ECM recommendations
    recommended_ecms: List[Dict[str, Any]] = field(default_factory=list)
    total_savings_kwh_m2: float = 0.0
    total_investment_sek: float = 0.0
    simple_payback_years: Optional[float] = None
    npv_sek: Optional[float] = None

    # Uncertainty (from ECMUncertaintyPropagator)
    savings_uncertainty_kwh_m2: float = 0.0
    savings_ci_90: Tuple[float, float] = (0.0, 0.0)

    # QC flags
    needs_qc: bool = False
    qc_triggers: List[str] = field(default_factory=list)
    qc_completed: bool = False
    qc_result: Optional[Dict[str, Any]] = None

    # Error tracking
    error: Optional[str] = None
    warnings: List[str] = field(default_factory=list)

    # Processing metadata
    processing_time_sec: float = 0.0
    data_source: str = "unknown"  # geojson, osm, mapillary, inferred


@dataclass
class PortfolioResult:
    """Result for a portfolio analysis."""

    # Summary
    total_buildings: int
    analyzed: int
    skipped: int
    failed: int

    # Building results
    results: List[BuildingResult] = field(default_factory=list)

    # Aggregate analytics (populated by portfolio_report)
    analytics: Optional[Any] = None  # PortfolioAnalytics

    # QC summary
    flagged_for_qc: int = 0
    qc_completed: int = 0


class RaidenOrchestrator:
    """
    Portfolio-scale building analysis orchestrator.

    Implements tiered processing:
    1. Fast triage via Sweden Buildings GeoJSON
    2. Standard analysis with pre-trained surrogates
    3. Deep analysis with EnergyPlus simulation

    Triggers agentic QC when confidence is low.
    """

    def __init__(
        self,
        config: Optional[TierConfig] = None,
        surrogate_dir: Optional[Path] = None,
        enable_qc: bool = True,
    ):
        """
        Initialize the orchestrator.

        Args:
            config: Tier configuration (default: TierConfig())
            surrogate_dir: Directory for pre-trained surrogates
            enable_qc: Whether to enable agentic QC
        """
        self.config = config or TierConfig()
        self.surrogate_dir = surrogate_dir or Path("./surrogates")
        self.enable_qc = enable_qc

        # Lazy-loaded components
        self._geojson_loader = None
        self._archetype_matcher = None
        self._surrogate_library = None
        self._qc_agents = None

        logger.info(
            f"RaidenOrchestrator initialized: "
            f"standard_workers={self.config.standard_workers}, "
            f"deep_workers={self.config.deep_workers}, "
            f"enable_qc={enable_qc}"
        )

    @property
    def geojson_loader(self):
        """Lazy-load Sweden Buildings GeoJSON."""
        if self._geojson_loader is None:
            try:
                from src.ingest import load_sweden_buildings
                self._geojson_loader = load_sweden_buildings()
                logger.info(f"Loaded {self._geojson_loader.total_buildings} buildings from GeoJSON")
            except Exception as e:
                logger.warning(f"Could not load GeoJSON: {e}")
                self._geojson_loader = None
        return self._geojson_loader

    @property
    def archetype_matcher(self):
        """Lazy-load archetype matcher."""
        if self._archetype_matcher is None:
            try:
                from src.baseline import ArchetypeMatcherV2
                self._archetype_matcher = ArchetypeMatcherV2(use_ai_modules=True)
            except Exception as e:
                logger.warning(f"Could not load ArchetypeMatcherV2: {e}")
                # Fallback to v1
                from src.baseline import ArchetypeMatcher
                self._archetype_matcher = ArchetypeMatcher()
        return self._archetype_matcher

    async def analyze_portfolio(
        self,
        addresses: List[str],
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> PortfolioResult:
        """
        Analyze a portfolio of buildings.

        Args:
            addresses: List of building addresses
            progress_callback: Optional callback(completed, total)

        Returns:
            PortfolioResult with all building results and analytics
        """
        total = len(addresses)
        results: List[BuildingResult] = []
        skipped = 0
        failed = 0

        logger.info(f"Starting portfolio analysis: {total} buildings")

        # Phase 1: Fast triage
        logger.info("Phase 1: Fast triage via GeoJSON")
        triage_results = await self._triage_buildings(addresses)

        # Separate by tier
        skip_list = []
        fast_list = []
        standard_list = []
        deep_list = []

        for addr, tier, building_data in triage_results:
            if tier == AnalysisTier.SKIP:
                skip_list.append((addr, building_data))
            elif tier == AnalysisTier.FAST:
                fast_list.append((addr, building_data))
            elif tier == AnalysisTier.STANDARD:
                standard_list.append((addr, building_data))
            else:
                deep_list.append((addr, building_data))

        logger.info(
            f"Triage complete: skip={len(skip_list)}, fast={len(fast_list)}, "
            f"standard={len(standard_list)}, deep={len(deep_list)}"
        )

        # Process skipped buildings (already optimized)
        for addr, building_data in skip_list:
            result = BuildingResult(
                address=addr,
                tier=AnalysisTier.SKIP,
                energy_class=building_data.get("energy_class") if building_data else None,
                data_source="geojson",
            )
            results.append(result)
            skipped += 1

        # Phase 2: Fast analysis (GeoJSON data only)
        if fast_list:
            logger.info(f"Phase 2: Fast analysis for {len(fast_list)} buildings")
            fast_results = await self._analyze_fast(fast_list)
            results.extend(fast_results)

        # Phase 3: Standard analysis (surrogate-based)
        if standard_list:
            logger.info(f"Phase 3: Standard analysis for {len(standard_list)} buildings")
            standard_results = await self._analyze_standard(standard_list)
            results.extend(standard_results)

        # Phase 4: Deep analysis (EnergyPlus)
        if deep_list and self.config.enable_energyplus:
            logger.info(f"Phase 4: Deep analysis for {len(deep_list)} buildings")
            deep_results = await self._analyze_deep(deep_list)
            results.extend(deep_results)

        # Phase 5: Agentic QC
        if self.enable_qc:
            qc_needed = [r for r in results if r.needs_qc and not r.qc_completed]
            if qc_needed:
                logger.info(f"Phase 5: Agentic QC for {len(qc_needed)} buildings")
                await self._run_qc(qc_needed)

        # Count failures
        failed = sum(1 for r in results if not r.success)

        # Generate portfolio analytics
        from .portfolio_report import PortfolioAnalytics
        analytics = PortfolioAnalytics.from_results(results)

        return PortfolioResult(
            total_buildings=total,
            analyzed=len(results) - skipped,
            skipped=skipped,
            failed=failed,
            results=results,
            analytics=analytics,
            flagged_for_qc=sum(1 for r in results if r.needs_qc),
            qc_completed=sum(1 for r in results if r.qc_completed),
        )

    async def _triage_buildings(
        self,
        addresses: List[str],
    ) -> List[Tuple[str, AnalysisTier, Optional[Dict]]]:
        """
        Triage buildings to determine analysis tier.

        Uses Sweden Buildings GeoJSON for fast lookup.
        Buildings with energy class A/B are skipped.
        Buildings found in GeoJSON use fast/standard tier.
        Buildings not found use standard/deep tier.
        """
        results = []

        for addr in addresses:
            building_data = None
            tier = AnalysisTier.STANDARD  # Default

            # Try GeoJSON lookup
            if self.geojson_loader:
                try:
                    matches = self.geojson_loader.find_by_address(addr)
                    if matches:
                        building = matches[0]
                        building_data = {
                            "construction_year": building.construction_year,
                            "atemp_m2": building.atemp_m2,
                            "energy_class": building.energy_class,
                            "energy_performance_kwh_m2": building.energy_performance_kwh_m2,
                            "ventilation_type": building.ventilation_type,
                            "has_ftx": building.ventilation_type == "FTX",
                            "heating_system": building.get_primary_heating(),
                            "has_solar_pv": building.has_solar_pv,
                            "footprint_area_m2": building.footprint_area_m2,
                        }

                        # Check if already optimized
                        if building.energy_class in self.config.skip_energy_classes:
                            tier = AnalysisTier.SKIP
                        else:
                            # Rich data = fast analysis sufficient
                            tier = AnalysisTier.FAST
                except Exception as e:
                    logger.debug(f"GeoJSON lookup failed for {addr}: {e}")

            results.append((addr, tier, building_data))

        return results

    async def _analyze_fast(
        self,
        buildings: List[Tuple[str, Optional[Dict]]],
    ) -> List[BuildingResult]:
        """
        Fast analysis using GeoJSON data only.

        No EnergyPlus simulation, uses archetype defaults.
        """
        results = []

        for addr, building_data in buildings:
            try:
                result = BuildingResult(
                    address=addr,
                    tier=AnalysisTier.FAST,
                    data_source="geojson",
                )

                if building_data:
                    result.construction_year = building_data.get("construction_year")
                    result.atemp_m2 = building_data.get("atemp_m2")
                    result.energy_class = building_data.get("energy_class")
                    result.current_kwh_m2 = building_data.get("energy_performance_kwh_m2")

                    # Match archetype
                    try:
                        from src.baseline import get_archetype_for_year
                        archetype = get_archetype_for_year(result.construction_year or 1970)
                        if archetype:
                            result.archetype_id = archetype.id
                            result.archetype_confidence = 0.6  # Lower for fast tier
                    except Exception:
                        pass

                    # Estimate ECM potential from energy class
                    result = self._estimate_ecm_potential(result, building_data)

                results.append(result)

            except Exception as e:
                logger.error(f"Fast analysis failed for {addr}: {e}")
                results.append(BuildingResult(
                    address=addr,
                    tier=AnalysisTier.FAST,
                    success=False,
                    error=str(e),
                ))

        return results

    async def _analyze_standard(
        self,
        buildings: List[Tuple[str, Optional[Dict]]],
    ) -> List[BuildingResult]:
        """
        Standard analysis using pre-trained surrogates.

        Parallel execution with surrogate predictions.
        """
        results = []

        # Process in batches
        batch_size = self.config.max_buildings_per_batch

        for i in range(0, len(buildings), batch_size):
            batch = buildings[i:i + batch_size]

            # Process batch concurrently
            tasks = [
                self._analyze_single_standard(addr, data)
                for addr, data in batch
            ]
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)

            for (addr, _), result in zip(batch, batch_results):
                if isinstance(result, Exception):
                    results.append(BuildingResult(
                        address=addr,
                        tier=AnalysisTier.STANDARD,
                        success=False,
                        error=str(result),
                    ))
                else:
                    results.append(result)

        return results

    async def _analyze_single_standard(
        self,
        address: str,
        building_data: Optional[Dict],
    ) -> BuildingResult:
        """Analyze a single building with standard tier."""
        import time
        start = time.time()

        result = BuildingResult(
            address=address,
            tier=AnalysisTier.STANDARD,
        )

        try:
            # Get building data if not from GeoJSON
            if not building_data:
                from src.core.address_pipeline import BuildingDataFetcher
                fetcher = BuildingDataFetcher()
                fetched = fetcher.fetch(address)
                if fetched:
                    building_data = {
                        "construction_year": fetched.construction_year,
                        "atemp_m2": fetched.atemp_m2,
                        "energy_class": getattr(fetched, "energy_class", None),
                        "energy_performance_kwh_m2": getattr(fetched, "energy_kwh_m2", None),
                        "facade_material": getattr(fetched, "facade_material", None),
                        "num_floors": getattr(fetched, "num_floors", None),
                    }
                    result.data_source = "address_pipeline"
            else:
                result.data_source = "geojson"

            if building_data:
                result.construction_year = building_data.get("construction_year")
                result.atemp_m2 = building_data.get("atemp_m2")
                result.energy_class = building_data.get("energy_class")
                result.current_kwh_m2 = building_data.get("energy_performance_kwh_m2")

            # Match archetype
            match_result = await self._match_archetype(building_data or {})
            if match_result:
                result.archetype_id = match_result.get("archetype_id")
                result.archetype_confidence = match_result.get("confidence", 0.0)

                # Check if QC needed
                if result.archetype_confidence < self.config.surrogate_confidence_threshold:
                    result.needs_qc = True
                    result.qc_triggers.append("low_archetype_confidence")

            # Get ECM recommendations using surrogate
            if result.archetype_id:
                ecm_result = await self._predict_ecms_surrogate(
                    result.archetype_id,
                    building_data or {},
                )
                if ecm_result:
                    result.recommended_ecms = ecm_result.get("ecms", [])
                    result.total_savings_kwh_m2 = ecm_result.get("total_savings", 0.0)
                    result.total_investment_sek = ecm_result.get("total_investment", 0.0)
                    result.savings_uncertainty_kwh_m2 = ecm_result.get("uncertainty", 0.0)

                    if result.total_savings_kwh_m2 > 0 and result.total_investment_sek > 0:
                        energy_price = 1.20  # SEK/kWh
                        annual_savings = result.total_savings_kwh_m2 * (result.atemp_m2 or 1000) * energy_price
                        if annual_savings > 0:
                            result.simple_payback_years = result.total_investment_sek / annual_savings

            result.success = True

        except Exception as e:
            logger.error(f"Standard analysis failed for {address}: {e}")
            result.success = False
            result.error = str(e)

        result.processing_time_sec = time.time() - start
        return result

    async def _analyze_deep(
        self,
        buildings: List[Tuple[str, Optional[Dict]]],
    ) -> List[BuildingResult]:
        """
        Deep analysis with EnergyPlus simulation.

        Uses ProcessPoolExecutor for CPU-bound simulations.
        This is critical for performance as E+ simulations are CPU-bound
        and the GIL would prevent true parallelism with ThreadPoolExecutor.
        """
        from concurrent.futures import ProcessPoolExecutor
        import multiprocessing

        results = []

        # Determine number of workers (respect config, but limit to CPU count)
        max_workers = min(
            self.config.deep_workers,
            multiprocessing.cpu_count(),
            len(buildings)
        )

        if max_workers == 0:
            return results

        logger.info(f"Starting deep analysis of {len(buildings)} buildings with {max_workers} workers")

        # For E+ simulations, we use ProcessPoolExecutor
        # Each worker gets its own process with separate memory space
        loop = asyncio.get_event_loop()

        try:
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                # Submit all tasks to the process pool
                futures = []
                for addr, data in buildings:
                    # Convert building data to dict for pickling
                    bd_dict = self._building_data_to_dict(data) if data else None
                    future = loop.run_in_executor(
                        executor,
                        _run_deep_analysis_in_process,  # Module-level function for pickling
                        addr,
                        bd_dict,
                        self.config.enable_energyplus,
                    )
                    futures.append((addr, future))

                # Wait for all to complete
                for addr, future in futures:
                    try:
                        result_dict = await future
                        result = self._dict_to_building_result(result_dict)
                        results.append(result)
                    except Exception as e:
                        logger.error(f"Deep analysis failed for {addr}: {e}")
                        results.append(BuildingResult(
                            address=addr,
                            tier=AnalysisTier.DEEP,
                            success=False,
                            error=str(e),
                        ))

        except Exception as e:
            logger.error(f"ProcessPoolExecutor failed: {e}")
            # Fallback to sequential processing
            for addr, data in buildings:
                try:
                    result = await self._analyze_single_deep(addr, data)
                    results.append(result)
                except Exception as ex:
                    results.append(BuildingResult(
                        address=addr,
                        tier=AnalysisTier.DEEP,
                        success=False,
                        error=str(ex),
                    ))

        logger.info(f"Completed deep analysis: {sum(1 for r in results if r.success)}/{len(results)} successful")
        return results

    def _building_data_to_dict(self, data: Any) -> Optional[Dict[str, Any]]:
        """Convert building data to picklable dict."""
        if data is None:
            return None
        if isinstance(data, dict):
            return data
        # Handle BuildingData or similar objects
        return {
            "address": getattr(data, "address", None),
            "construction_year": getattr(data, "construction_year", None),
            "atemp_m2": getattr(data, "atemp_m2", None),
            "energy_class": getattr(data, "energy_class", None),
            "facade_material": getattr(data, "facade_material", None),
            "heating_system": getattr(data, "heating_system", None),
            "ventilation_type": getattr(data, "ventilation_type", None),
            "lat": getattr(data, "lat", None),
            "lon": getattr(data, "lon", None),
        }

    def _dict_to_building_result(self, data: Dict[str, Any]) -> BuildingResult:
        """Convert dict from process pool back to BuildingResult."""
        return BuildingResult(
            address=data.get("address", ""),
            tier=AnalysisTier(data.get("tier", "deep")),
            success=data.get("success", False),
            construction_year=data.get("construction_year"),
            atemp_m2=data.get("atemp_m2"),
            energy_class=data.get("energy_class"),
            current_kwh_m2=data.get("current_kwh_m2"),
            archetype_id=data.get("archetype_id"),
            archetype_confidence=data.get("archetype_confidence", 0.0),
            recommended_ecms=data.get("recommended_ecms", []),
            total_savings_kwh_m2=data.get("total_savings_kwh_m2", 0.0),
            total_investment_sek=data.get("total_investment_sek", 0.0),
            simple_payback_years=data.get("simple_payback_years"),
            npv_sek=data.get("npv_sek"),
            error=data.get("error"),
            processing_time_sec=data.get("processing_time_sec", 0.0),
        )

    async def _analyze_single_deep(
        self,
        address: str,
        building_data: Optional[Dict],
    ) -> BuildingResult:
        """Analyze a single building with full EnergyPlus simulation."""
        import os
        import time
        start = time.time()

        result = BuildingResult(
            address=address,
            tier=AnalysisTier.DEEP,
        )

        try:
            # Use the full analysis pipeline
            from src.analysis.full_pipeline import FullPipelineAnalyzer

            # Get API keys from environment
            google_api_key = os.environ.get("GOOGLE_API_KEY")
            mapillary_token = os.environ.get("MAPILLARY_TOKEN")

            pipeline = FullPipelineAnalyzer(
                google_api_key=google_api_key,
                mapillary_token=mapillary_token,
                use_bayesian_calibration=True,
            )

            # Build context from address
            from src.core.address_pipeline import BuildingDataFetcher
            fetcher = BuildingDataFetcher()
            fetched = fetcher.fetch(address)

            if fetched:
                result.construction_year = fetched.construction_year
                result.atemp_m2 = fetched.atemp_m2
                result.energy_class = fetched.energy_class

                # Convert BuildingData to dict for pipeline
                bd_dict = {
                    "address": fetched.address,
                    "construction_year": fetched.construction_year,
                    "atemp_m2": fetched.atemp_m2,
                    "energy_class": fetched.energy_class,
                    "facade_material": fetched.facade_material,
                    "heating_system": fetched.heating_system,
                    "ventilation_type": fetched.ventilation_type,
                    "lat": getattr(fetched, "lat", None),
                    "lon": getattr(fetched, "lon", None),
                }

                # Run full pipeline (this includes E+ simulation)
                # FullPipelineAnalyzer.analyze() is async, so we can call it directly
                pipeline_result = await pipeline.analyze(
                    address=address,
                    building_data=bd_dict,
                    run_simulations=self.config.enable_energyplus,
                )

                if pipeline_result:
                    # Extract results from pipeline dict
                    result.archetype_id = pipeline_result.get("archetype_id")
                    result.archetype_confidence = pipeline_result.get("archetype_confidence", 0.0)
                    result.current_kwh_m2 = pipeline_result.get("calibrated_kwh_m2")

                    # ECM results
                    ecm_results = pipeline_result.get("ecm_results", [])
                    result.recommended_ecms = [
                        {
                            "id": ecm.get("id"),
                            "name": ecm.get("name"),
                            "savings_kwh_m2": ecm.get("savings_kwh_m2", 0),
                            "cost_sek": ecm.get("cost_sek", 0),
                        }
                        for ecm in ecm_results
                    ]

                    result.total_savings_kwh_m2 = pipeline_result.get("total_savings_kwh_m2", 0.0)
                    result.total_investment_sek = pipeline_result.get("total_investment_sek", 0.0)
                    result.npv_sek = pipeline_result.get("npv_sek")
                    result.simple_payback_years = pipeline_result.get("payback_years")

                    # Calibration uncertainty if available
                    if "calibration_result" in pipeline_result:
                        cal_result = pipeline_result["calibration_result"]
                        if hasattr(cal_result, "posterior_mean"):
                            result.warnings.append(
                                f"Calibrated with posterior mean: {cal_result.posterior_mean}"
                            )

            result.success = True

        except ImportError as e:
            logger.warning(f"Full pipeline not available: {e}. Falling back to standard analysis.")
            # Fall back to standard analysis
            return await self._analyze_single_standard(address, building_data)

        except Exception as e:
            logger.error(f"Deep analysis failed for {address}: {e}")
            result.success = False
            result.error = str(e)

        result.processing_time_sec = time.time() - start
        return result

    async def _match_archetype(self, building_data: Dict) -> Optional[Dict]:
        """Match building to archetype."""
        try:
            # Use V2 matcher if available
            matcher = self.archetype_matcher

            construction_year = building_data.get("construction_year", 1970)

            if hasattr(matcher, "match_from_building_data"):
                # V2 matcher
                from src.core.address_pipeline import BuildingData
                bd = BuildingData(
                    construction_year=construction_year,
                    atemp_m2=building_data.get("atemp_m2"),
                    energy_class=building_data.get("energy_class"),
                    facade_material=building_data.get("facade_material"),
                    num_floors=building_data.get("num_floors"),
                )
                result = matcher.match_from_building_data(bd)
                return {
                    "archetype_id": result.archetype.id,
                    "confidence": result.confidence,
                }
            else:
                # V1 matcher
                from src.baseline import BuildingType
                result = matcher.match(
                    construction_year=construction_year,
                    building_type=BuildingType.MULTI_FAMILY,
                )
                return {
                    "archetype_id": result.archetype.id,
                    "confidence": 0.7,
                }
        except Exception as e:
            logger.debug(f"Archetype matching failed: {e}")
            return None

    async def _predict_ecms_surrogate(
        self,
        archetype_id: str,
        building_data: Dict,
    ) -> Optional[Dict]:
        """Predict ECM savings using pre-trained surrogate."""
        try:
            from .surrogate_library import get_or_train_surrogate

            surrogate = get_or_train_surrogate(archetype_id, self.surrogate_dir)
            if not surrogate:
                return None

            # Get current building parameters
            params = self._extract_params_for_surrogate(building_data, archetype_id)

            # Predict baseline
            baseline_kwh_m2 = surrogate.predict(params)

            # Predict each ECM
            ecm_results = []
            total_savings = 0.0
            total_investment = 0.0

            from src.calibration.bayesian import ECM_PARAMETER_EFFECTS, get_ecm_effect
            from src.ecm import get_all_ecms

            for ecm in get_all_ecms():
                if ecm.id in ECM_PARAMETER_EFFECTS:
                    # Apply ECM effect to parameters
                    effect_fn = get_ecm_effect(ecm.id)
                    modified_params = effect_fn(params.copy())

                    # Predict with ECM
                    ecm_kwh_m2 = surrogate.predict(modified_params)
                    savings = baseline_kwh_m2 - ecm_kwh_m2

                    if savings > 0:
                        cost = ecm.typical_cost_sek_m2 * building_data.get("atemp_m2", 1000)
                        ecm_results.append({
                            "ecm_id": ecm.id,
                            "ecm_name": ecm.name_sv,
                            "savings_kwh_m2": savings,
                            "cost_sek": cost,
                        })
                        total_savings += savings
                        total_investment += cost

            return {
                "ecms": ecm_results,
                "total_savings": total_savings,
                "total_investment": total_investment,
                "uncertainty": total_savings * 0.15,  # Rough estimate
            }

        except Exception as e:
            logger.debug(f"Surrogate prediction failed: {e}")
            return None

    def _extract_params_for_surrogate(
        self,
        building_data: Dict,
        archetype_id: str,
    ) -> Dict[str, float]:
        """Extract surrogate parameters from building data."""
        from src.baseline import get_archetype

        archetype = get_archetype(archetype_id)

        # Start with archetype defaults
        params = {
            "infiltration_ach": 0.06,
            "wall_u_value": 0.5,
            "roof_u_value": 0.3,
            "window_u_value": 2.0,
            "heat_recovery_eff": 0.0,
            "heating_setpoint": 21.0,
        }

        if archetype:
            params["wall_u_value"] = archetype.envelope.wall_u_value
            params["roof_u_value"] = archetype.envelope.roof_u_value
            params["window_u_value"] = archetype.envelope.window_u_value
            if archetype.ventilation.has_heat_recovery:
                params["heat_recovery_eff"] = archetype.ventilation.heat_recovery_efficiency

        # Override with building-specific data
        if building_data.get("has_ftx"):
            params["heat_recovery_eff"] = 0.80

        return params

    def _estimate_ecm_potential(
        self,
        result: BuildingResult,
        building_data: Dict,
    ) -> BuildingResult:
        """Estimate ECM potential from energy class."""
        energy_class = result.energy_class

        # Rough savings estimates by energy class
        savings_by_class = {
            "G": 40.0,  # kWh/m²
            "F": 30.0,
            "E": 20.0,
            "D": 15.0,
            "C": 10.0,
            "B": 5.0,
            "A": 2.0,
        }

        if energy_class in savings_by_class:
            result.total_savings_kwh_m2 = savings_by_class[energy_class]

            # Rough cost estimate: 500 SEK/m² for moderate renovation
            result.total_investment_sek = 500 * (result.atemp_m2 or 1000)

            # Calculate payback
            energy_price = 1.20  # SEK/kWh
            annual_savings = result.total_savings_kwh_m2 * (result.atemp_m2 or 1000) * energy_price
            if annual_savings > 0:
                result.simple_payback_years = result.total_investment_sek / annual_savings

        return result

    async def _run_qc(self, results: List[BuildingResult]) -> None:
        """Run agentic QC on flagged buildings."""
        from .qc_agent import ImageQCAgent, ECMRefinerAgent

        image_agent = ImageQCAgent()
        ecm_agent = ECMRefinerAgent()

        for result in results:
            try:
                for trigger in result.qc_triggers:
                    if trigger in ("low_wwr_confidence", "low_material_confidence"):
                        qc_result = await image_agent.run(result)
                        result.qc_result = qc_result
                        result.qc_completed = True

                    elif trigger == "negative_savings":
                        qc_result = await ecm_agent.run(result)
                        result.qc_result = qc_result
                        result.qc_completed = True

            except Exception as e:
                logger.error(f"QC failed for {result.address}: {e}")
                result.warnings.append(f"QC failed: {e}")


async def analyze_portfolio(
    addresses: List[str],
    config: Optional[TierConfig] = None,
) -> PortfolioResult:
    """
    Convenience function to analyze a portfolio.

    Args:
        addresses: List of building addresses
        config: Optional tier configuration

    Returns:
        PortfolioResult with analysis results
    """
    orchestrator = RaidenOrchestrator(config=config)
    return await orchestrator.analyze_portfolio(addresses)


async def analyze_portfolio_hybrid(
    addresses: List[str],
    validate_top_percent: float = 10.0,
    validate_min_savings_kwh_m2: float = 20.0,
    config: Optional[TierConfig] = None,
    progress_callback: Optional[Callable[[str, int, int], None]] = None,
) -> PortfolioResult:
    """
    Hybrid analysis: Surrogate screening → E+ validation for top candidates.

    This is the recommended approach for large portfolios (1000+ buildings):
    1. Screen ALL buildings with surrogates (~10 sec total for 1000 buildings)
    2. Rank by savings potential
    3. Run full EnergyPlus on top N% or those above savings threshold
    4. Return validated results with uncertainty bounds

    This gives you:
    - Fast screening for the entire portfolio
    - Accurate E+ validation where it matters (high-value buildings)
    - Confidence intervals from surrogate vs E+ comparison

    Args:
        addresses: List of building addresses
        validate_top_percent: % of top buildings to validate with E+ (default: 10%)
        validate_min_savings_kwh_m2: Validate all above this threshold (default: 20)
        config: Optional tier configuration
        progress_callback: Optional callback(phase, completed, total)

    Returns:
        PortfolioResult with validated top candidates

    Example:
        >>> # Screen 1000 buildings, E+ validate top 100
        >>> result = await analyze_portfolio_hybrid(
        ...     addresses=all_1000_addresses,
        ...     validate_top_percent=10.0,  # Top 10%
        ...     validate_min_savings_kwh_m2=25.0,  # Plus all with >25 kWh/m² savings
        ... )
        >>> print(f"Screened: {result.total_buildings}")
        >>> print(f"E+ validated: {result.deep_validated}")
    """
    import time
    start_time = time.time()

    orchestrator = RaidenOrchestrator(config=config)

    # Phase 1: Surrogate screening (fast)
    logger.info(f"Phase 1: Surrogate screening for {len(addresses)} buildings")
    if progress_callback:
        progress_callback("screening", 0, len(addresses))

    # Use standard tier (surrogate-based) for all buildings
    screening_config = TierConfig(
        enable_energyplus=False,  # No E+ in screening phase
        standard_workers=100,
    )
    screening_orchestrator = RaidenOrchestrator(config=screening_config, enable_qc=False)

    # Triage and run standard analysis
    triage_results = await screening_orchestrator._triage_buildings(addresses)
    all_buildings = [(addr, data) for addr, tier, data in triage_results]

    screened_results = []
    batch_size = 100
    for i in range(0, len(all_buildings), batch_size):
        batch = all_buildings[i:i + batch_size]
        batch_results = await screening_orchestrator._analyze_standard(batch)
        screened_results.extend(batch_results)
        if progress_callback:
            progress_callback("screening", min(i + batch_size, len(all_buildings)), len(all_buildings))

    screening_time = time.time() - start_time
    logger.info(f"Screening completed in {screening_time:.1f}s: {len(screened_results)} buildings")

    # Phase 2: Rank and select for E+ validation
    logger.info("Phase 2: Ranking buildings for E+ validation")

    # Sort by savings potential (descending)
    ranked = sorted(
        [r for r in screened_results if r.total_savings_kwh_m2 > 0],
        key=lambda r: r.total_savings_kwh_m2,
        reverse=True,
    )

    # Select buildings for validation
    num_for_validation = max(1, int(len(ranked) * validate_top_percent / 100))
    to_validate = []

    for r in ranked:
        # Include if in top N% OR above savings threshold
        if len(to_validate) < num_for_validation or r.total_savings_kwh_m2 >= validate_min_savings_kwh_m2:
            to_validate.append(r)

    logger.info(
        f"Selected {len(to_validate)} buildings for E+ validation "
        f"(top {validate_top_percent}% or >{validate_min_savings_kwh_m2} kWh/m²)"
    )

    # Phase 3: E+ validation (slower but accurate)
    validated_results = []
    if to_validate and orchestrator.config.enable_energyplus:
        logger.info(f"Phase 3: E+ validation for {len(to_validate)} buildings")
        if progress_callback:
            progress_callback("validation", 0, len(to_validate))

        validation_start = time.time()

        # Prepare for deep analysis
        deep_inputs = [
            (r.address, {"construction_year": r.construction_year, "atemp_m2": r.atemp_m2})
            for r in to_validate
        ]

        # Run E+ validation
        validated_results = await orchestrator._analyze_deep(deep_inputs)

        validation_time = time.time() - validation_start
        logger.info(f"Validation completed in {validation_time:.1f}s")

        if progress_callback:
            progress_callback("validation", len(to_validate), len(to_validate))

        # Merge validated results back
        validated_addresses = {r.address for r in validated_results}
        for i, result in enumerate(screened_results):
            if result.address in validated_addresses:
                # Find the validated result
                for vr in validated_results:
                    if vr.address == result.address:
                        # Update with validated data
                        screened_results[i] = vr
                        screened_results[i].warnings.append(
                            f"E+ validated (surrogate: {result.total_savings_kwh_m2:.1f}, "
                            f"E+: {vr.total_savings_kwh_m2:.1f} kWh/m²)"
                        )
                        break

    # Calculate validation accuracy (for buildings that were validated)
    validation_errors = []
    for sr in screened_results:
        if "E+ validated" in " ".join(sr.warnings):
            # Parse surrogate vs E+ values from warning
            for warning in sr.warnings:
                if "surrogate:" in warning and "E+:" in warning:
                    try:
                        import re
                        match = re.search(r"surrogate: ([\d.]+).*E\+: ([\d.]+)", warning)
                        if match:
                            surrogate_val = float(match.group(1))
                            ep_val = float(match.group(2))
                            if surrogate_val > 0:
                                error_pct = abs(ep_val - surrogate_val) / surrogate_val * 100
                                validation_errors.append(error_pct)
                    except (ValueError, AttributeError):
                        pass

    mean_error = sum(validation_errors) / len(validation_errors) if validation_errors else 0

    total_time = time.time() - start_time

    # Generate portfolio analytics
    from .portfolio_report import PortfolioAnalytics
    analytics = PortfolioAnalytics.from_results(screened_results)

    # Add hybrid-specific stats
    analytics.hybrid_stats = {
        "screening_time_sec": screening_time,
        "validation_time_sec": total_time - screening_time,
        "total_time_sec": total_time,
        "buildings_validated": len(to_validate),
        "validation_percent": len(to_validate) / len(screened_results) * 100 if screened_results else 0,
        "mean_surrogate_error_pct": mean_error,
        "speedup_factor": len(screened_results) / max(1, len(to_validate)),
    }

    result = PortfolioResult(
        total_buildings=len(addresses),
        analyzed=len(screened_results),
        skipped=len(addresses) - len(screened_results),
        failed=sum(1 for r in screened_results if not r.success),
        results=screened_results,
        analytics=analytics,
    )

    # Add custom attribute for validation count
    result.deep_validated = len(to_validate)

    logger.info(
        f"Hybrid analysis complete: {len(screened_results)} screened, "
        f"{len(to_validate)} validated, {total_time:.1f}s total "
        f"(speedup: {analytics.hybrid_stats['speedup_factor']:.1f}x)"
    )

    return result
