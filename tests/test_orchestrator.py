"""
Tests for the Raiden Orchestrator module.

Tests portfolio-scale analysis, tiered processing, and agentic QC.
"""

import asyncio
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.orchestrator.orchestrator import (
    RaidenOrchestrator,
    BuildingResult,
    PortfolioResult,
    AnalysisTier,
    TierConfig,
    analyze_portfolio,
)
from src.orchestrator.prioritizer import (
    BuildingPrioritizer,
    PrioritizationStrategy,
    TriageResult,
    quick_triage,
)
from src.orchestrator.qc_agent import (
    QCAgent,
    ImageQCAgent,
    ECMRefinerAgent,
    AnomalyAgent,
    QCResult,
    QCTrigger,
    QCTriggerType,
)
from src.orchestrator.surrogate_library import (
    SurrogateLibrary,
    ArchetypeSurrogate,
    get_or_train_surrogate,
)
from src.orchestrator.portfolio_report import (
    PortfolioAnalytics,
    generate_portfolio_report,
)


class TestAnalysisTier:
    """Tests for AnalysisTier enum."""

    def test_tier_values(self):
        """Test tier enum values."""
        assert AnalysisTier.SKIP.value == "skip"
        assert AnalysisTier.FAST.value == "fast"
        assert AnalysisTier.STANDARD.value == "standard"
        assert AnalysisTier.DEEP.value == "deep"


class TestTierConfig:
    """Tests for TierConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = TierConfig()

        assert config.skip_energy_classes == ("A", "B")
        assert config.standard_workers == 50
        assert config.deep_workers == 10
        assert config.surrogate_confidence_threshold == 0.70
        assert config.wwr_confidence_threshold == 0.60

    def test_custom_config(self):
        """Test custom configuration."""
        config = TierConfig(
            skip_energy_classes=("A",),
            standard_workers=100,
            deep_workers=20,
        )

        assert config.skip_energy_classes == ("A",)
        assert config.standard_workers == 100
        assert config.deep_workers == 20


class TestBuildingResult:
    """Tests for BuildingResult dataclass."""

    def test_default_result(self):
        """Test default building result."""
        result = BuildingResult(
            address="Test Address",
            tier=AnalysisTier.STANDARD,
        )

        assert result.address == "Test Address"
        assert result.tier == AnalysisTier.STANDARD
        assert result.success is True
        assert result.recommended_ecms == []
        assert result.needs_qc is False

    def test_result_with_data(self):
        """Test result with building data."""
        result = BuildingResult(
            address="Bellmansgatan 16",
            tier=AnalysisTier.FAST,
            construction_year=1965,
            atemp_m2=2500.0,
            energy_class="E",
            current_kwh_m2=120.0,
            archetype_id="mfh_1961_1975",
            archetype_confidence=0.85,
            total_savings_kwh_m2=25.0,
            data_source="geojson",
        )

        assert result.construction_year == 1965
        assert result.energy_class == "E"
        assert result.archetype_confidence == 0.85

    def test_result_with_qc_flags(self):
        """Test result with QC flags."""
        result = BuildingResult(
            address="Test",
            tier=AnalysisTier.STANDARD,
            needs_qc=True,
            qc_triggers=["low_wwr_confidence", "low_material_confidence"],
        )

        assert result.needs_qc is True
        assert len(result.qc_triggers) == 2


class TestBuildingPrioritizer:
    """Tests for BuildingPrioritizer."""

    def test_default_prioritizer(self):
        """Test default prioritizer configuration."""
        prioritizer = BuildingPrioritizer()

        assert prioritizer.strategy == PrioritizationStrategy.HIGHEST_ROI_POTENTIAL
        assert prioritizer.skip_optimized is True

    def test_triage_skip_optimized(self):
        """Test that optimized buildings are skipped."""
        prioritizer = BuildingPrioritizer()

        result = prioritizer.triage_building(
            "Test Address",
            {"energy_class": "A"},
        )

        assert result.tier == AnalysisTier.SKIP
        assert "Already optimized" in result.skip_reason

    def test_triage_with_energy_class(self):
        """Test triage with energy class data."""
        prioritizer = BuildingPrioritizer(
            strategy=PrioritizationStrategy.LOWEST_ENERGY_CLASS_FIRST
        )

        # G class = highest priority
        result_g = prioritizer.triage_building(
            "Test G",
            {"energy_class": "G"},
        )

        # C class = lower priority
        result_c = prioritizer.triage_building(
            "Test C",
            {"energy_class": "C"},
        )

        assert result_g.priority_score > result_c.priority_score

    def test_triage_no_data(self):
        """Test triage with no building data."""
        prioritizer = BuildingPrioritizer()

        result = prioritizer.triage_building("Unknown Address", None)

        assert result.tier == AnalysisTier.STANDARD
        assert result.data_source == "none"
        assert result.needs_qc is True

    def test_prioritize_portfolio(self):
        """Test portfolio prioritization."""
        prioritizer = BuildingPrioritizer()

        buildings = [
            ("Address A", {"energy_class": "A"}),  # Should be skipped
            ("Address E", {"energy_class": "E", "atemp_m2": 3000}),
            ("Address G", {"energy_class": "G", "atemp_m2": 2000}),
            ("Address C", {"energy_class": "C", "atemp_m2": 1000}),
        ]

        results = prioritizer.prioritize_portfolio(buildings)

        # Skipped should be first in sort (tier=0 in sort key)
        skipped = [r for r in results if r.tier == AnalysisTier.SKIP]
        assert len(skipped) == 1
        assert skipped[0].address == "Address A"

        # Among non-skipped, G should be before C (higher priority for ROI)
        non_skipped = [r for r in results if r.tier != AnalysisTier.SKIP]
        g_idx = next(i for i, r in enumerate(non_skipped) if r.address == "Address G")
        c_idx = next(i for i, r in enumerate(non_skipped) if r.address == "Address C")
        assert g_idx < c_idx

    def test_prioritization_strategies(self):
        """Test different prioritization strategies."""
        building_data = {
            "energy_class": "E",
            "atemp_m2": 2000,
            "construction_year": 1965,
            "energy_performance_kwh_m2": 150,
        }

        # Test each strategy
        for strategy in PrioritizationStrategy:
            prioritizer = BuildingPrioritizer(strategy=strategy)
            result = prioritizer.triage_building("Test", building_data)
            assert 0 <= result.priority_score <= 1


class TestQCAgents:
    """Tests for QC agents."""

    def test_image_qc_agent_triggers(self):
        """Test ImageQCAgent trigger handling."""
        agent = ImageQCAgent()

        assert agent.can_handle(QCTriggerType.LOW_WWR_CONFIDENCE)
        assert agent.can_handle(QCTriggerType.LOW_MATERIAL_CONFIDENCE)
        assert not agent.can_handle(QCTriggerType.NEGATIVE_SAVINGS)

        # Also test with string trigger
        assert agent.can_handle("low_wwr_confidence")

    def test_ecm_refiner_agent_triggers(self):
        """Test ECMRefinerAgent trigger handling."""
        agent = ECMRefinerAgent()

        assert agent.can_handle(QCTriggerType.NEGATIVE_SAVINGS)
        assert agent.can_handle(QCTriggerType.ANOMALOUS_PATTERN)
        assert not agent.can_handle(QCTriggerType.LOW_WWR_CONFIDENCE)

    def test_anomaly_agent_triggers(self):
        """Test AnomalyAgent trigger handling."""
        agent = AnomalyAgent()

        assert agent.can_handle(QCTriggerType.ENERGY_CLASS_MISMATCH)
        assert agent.can_handle(QCTriggerType.ANOMALOUS_PATTERN)
        assert not agent.can_handle(QCTriggerType.NEGATIVE_SAVINGS)

    @pytest.mark.asyncio
    async def test_image_qc_agent_run(self):
        """Test ImageQCAgent run method."""
        agent = ImageQCAgent()

        building_result = BuildingResult(
            address="Test Address",
            tier=AnalysisTier.STANDARD,
            needs_qc=True,
            qc_triggers=["low_wwr_confidence"],
        )

        result = await agent.run(building_result)

        assert isinstance(result, QCResult)
        assert result.action_taken == "image_reanalysis"

    @pytest.mark.asyncio
    async def test_ecm_refiner_with_negative_savings(self):
        """Test ECMRefinerAgent with negative savings."""
        agent = ECMRefinerAgent()

        building_result = BuildingResult(
            address="Test Address",
            tier=AnalysisTier.STANDARD,
            archetype_id="mfh_1961_1975",
            recommended_ecms=[
                {"ecm_id": "led_lighting", "savings_kwh_m2": -5.0},
                {"ecm_id": "air_sealing", "savings_kwh_m2": 10.0},
            ],
        )

        result = await agent.run(building_result)

        assert isinstance(result, QCResult)
        assert result.action_taken == "ecm_refinement"


class TestArchetypeSurrogate:
    """Tests for ArchetypeSurrogate."""

    def test_surrogate_creation(self, tmp_path):
        """Test surrogate creation."""
        surrogate = ArchetypeSurrogate(
            archetype_id="mfh_1961_1975",
            surrogate_path=tmp_path / "test_gp.pkl",
            train_r2=0.94,
            test_r2=0.89,
            n_samples=150,
        )

        assert surrogate.archetype_id == "mfh_1961_1975"
        assert surrogate.train_r2 == 0.94
        assert surrogate.test_r2 == 0.89


class TestSurrogateLibrary:
    """Tests for SurrogateLibrary."""

    def test_library_creation(self, tmp_path):
        """Test library creation."""
        library = SurrogateLibrary(surrogate_dir=tmp_path)

        assert library.surrogate_dir == tmp_path
        assert len(library.list_available()) == 0

    def test_library_has_default_bounds(self, tmp_path):
        """Test library has default parameter bounds."""
        library = SurrogateLibrary(surrogate_dir=tmp_path)

        assert "infiltration_ach" in library.DEFAULT_PARAM_BOUNDS
        assert "wall_u_value" in library.DEFAULT_PARAM_BOUNDS
        assert "heat_recovery_eff" in library.DEFAULT_PARAM_BOUNDS


class TestPortfolioAnalytics:
    """Tests for PortfolioAnalytics."""

    def test_empty_analytics(self):
        """Test analytics with no results."""
        analytics = PortfolioAnalytics.from_results([])

        assert analytics.total_buildings == 0
        assert analytics.analyzed == 0
        assert analytics.total_savings_potential_kwh == 0

    def test_analytics_from_results(self):
        """Test analytics from building results."""
        results = [
            BuildingResult(
                address="Building 1",
                tier=AnalysisTier.FAST,
                success=True,
                atemp_m2=1000,
                energy_class="E",
                current_kwh_m2=100,
                total_savings_kwh_m2=20,
                total_investment_sek=50000,
                archetype_id="mfh_1961_1975",
                recommended_ecms=[
                    {"ecm_id": "air_sealing", "savings_kwh_m2": 10},
                    {"ecm_id": "roof_insulation", "savings_kwh_m2": 10},
                ],
            ),
            BuildingResult(
                address="Building 2",
                tier=AnalysisTier.STANDARD,
                success=True,
                atemp_m2=2000,
                energy_class="F",
                current_kwh_m2=150,
                total_savings_kwh_m2=40,
                total_investment_sek=100000,
                archetype_id="mfh_1961_1975",
                recommended_ecms=[
                    {"ecm_id": "air_sealing", "savings_kwh_m2": 20},
                ],
            ),
            BuildingResult(
                address="Building 3",
                tier=AnalysisTier.SKIP,
                success=True,
                energy_class="A",
            ),
        ]

        analytics = PortfolioAnalytics.from_results(results)

        assert analytics.total_buildings == 3
        assert analytics.analyzed == 2
        assert analytics.skipped_already_optimized == 1
        assert analytics.total_savings_potential_kwh == 20 * 1000 + 40 * 2000
        assert analytics.total_investment_sek == 150000

        # ECM frequency
        assert analytics.ecm_frequency["air_sealing"] == 2
        assert analytics.ecm_frequency["roof_insulation"] == 1

        # Energy class distribution
        assert analytics.energy_class_distribution["E"] == 1
        assert analytics.energy_class_distribution["F"] == 1
        assert analytics.energy_class_distribution["A"] == 1

        # Archetype distribution
        assert analytics.archetype_distribution["mfh_1961_1975"] == 2

    def test_analytics_top_buildings(self):
        """Test top building lists."""
        results = [
            BuildingResult(
                address=f"Building {i}",
                tier=AnalysisTier.STANDARD,
                success=True,
                atemp_m2=1000 * (i + 1),
                current_kwh_m2=100 + i * 20,
                total_savings_kwh_m2=10 + i * 5,
                total_investment_sek=10000 + i * 5000,
                simple_payback_years=3 + i * 0.5,
            )
            for i in range(15)
        ]

        analytics = PortfolioAnalytics.from_results(results)

        assert len(analytics.top_10_roi) == 10
        assert len(analytics.top_10_savings) == 10
        assert len(analytics.worst_10_consumption) == 10

        # Top ROI should have lowest payback
        assert analytics.top_10_roi[0]["payback_years"] < analytics.top_10_roi[-1]["payback_years"]


class TestPortfolioReport:
    """Tests for portfolio report generation."""

    def test_generate_markdown_report(self):
        """Test markdown report generation."""
        analytics = PortfolioAnalytics(
            total_buildings=100,
            analyzed=80,
            skipped_already_optimized=15,
            failed=5,
            total_savings_potential_kwh=1000000,
            total_investment_sek=5000000,
            portfolio_payback_years=5.5,
        )

        report = generate_portfolio_report(analytics, format="markdown")

        assert "# Portfolio Energy Analysis Report" in report
        assert "100" in report
        assert "80" in report

    def test_generate_html_report(self):
        """Test HTML report generation."""
        analytics = PortfolioAnalytics(
            total_buildings=50,
            analyzed=45,
        )

        report = generate_portfolio_report(analytics, format="html")

        assert "<html>" in report
        assert "Portfolio Energy Analysis Report" in report

    def test_generate_json_report(self):
        """Test JSON report generation."""
        import json

        analytics = PortfolioAnalytics(
            total_buildings=25,
            analyzed=20,
            total_savings_potential_kwh=500000,
        )

        report = generate_portfolio_report(analytics, format="json")
        data = json.loads(report)

        assert data["summary"]["total_buildings"] == 25
        assert data["summary"]["analyzed"] == 20

    def test_save_report_to_file(self, tmp_path):
        """Test saving report to file."""
        analytics = PortfolioAnalytics(
            total_buildings=10,
            analyzed=8,
        )

        output_path = tmp_path / "report.md"
        generate_portfolio_report(analytics, output_path=output_path, format="markdown")

        assert output_path.exists()
        content = output_path.read_text()
        assert "Portfolio Energy Analysis Report" in content


class TestRaidenOrchestrator:
    """Tests for RaidenOrchestrator."""

    def test_orchestrator_initialization(self):
        """Test orchestrator initialization."""
        orchestrator = RaidenOrchestrator()

        assert orchestrator.config is not None
        assert orchestrator.enable_qc is True

    def test_orchestrator_with_custom_config(self, tmp_path):
        """Test orchestrator with custom config."""
        config = TierConfig(
            standard_workers=100,
            deep_workers=20,
        )

        orchestrator = RaidenOrchestrator(
            config=config,
            surrogate_dir=tmp_path,
            enable_qc=False,
        )

        assert orchestrator.config.standard_workers == 100
        assert orchestrator.enable_qc is False

    @pytest.mark.asyncio
    async def test_analyze_portfolio_empty(self):
        """Test analyze_portfolio with empty list."""
        result = await analyze_portfolio([])

        assert result.total_buildings == 0
        assert result.analyzed == 0

    @pytest.mark.asyncio
    async def test_orchestrator_triage(self):
        """Test building triage logic."""
        orchestrator = RaidenOrchestrator()

        # Mock the geojson loader
        mock_building = MagicMock()
        mock_building.construction_year = 1970
        mock_building.atemp_m2 = 2000
        mock_building.energy_class = "E"
        mock_building.energy_performance_kwh_m2 = 120
        mock_building.ventilation_type = "F"
        mock_building.get_primary_heating.return_value = "district_heating"
        mock_building.has_solar_pv = False
        mock_building.footprint_area_m2 = 500

        mock_loader = MagicMock()
        mock_loader.find_by_address.return_value = [mock_building]

        orchestrator._geojson_loader = mock_loader

        results = await orchestrator._triage_buildings(["Test Address"])

        assert len(results) == 1
        addr, tier, data = results[0]
        assert addr == "Test Address"
        assert tier == AnalysisTier.FAST  # Has GeoJSON data
        assert data["energy_class"] == "E"


class TestIntegration:
    """Integration tests for the orchestrator module."""

    @pytest.mark.asyncio
    async def test_full_workflow(self, tmp_path):
        """Test full workflow from addresses to report."""
        # Create mock buildings
        addresses = [
            "Building A",
            "Building B",
            "Building C",
        ]

        config = TierConfig(enable_energyplus=False)  # No E+ for testing

        orchestrator = RaidenOrchestrator(
            config=config,
            surrogate_dir=tmp_path,
        )

        # Run analysis
        result = await orchestrator.analyze_portfolio(addresses)

        assert result.total_buildings == 3
        assert result.analytics is not None

        # Generate report
        report = generate_portfolio_report(
            result.analytics,
            output_path=tmp_path / "report.md",
            format="markdown",
        )

        assert (tmp_path / "report.md").exists()
