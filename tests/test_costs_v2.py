"""
Tests for the sophisticated Swedish cost model (V2).

Tests:
- Cost inflation adjustment
- Regional price variations
- Building size scaling
- Swedish tax deductions (ROT, green tech)
- Cost breakdowns
"""

import pytest
import math

from src.roi.costs_sweden_v2 import (
    CostSource,
    CostCategory,
    Region,
    OwnerType,
    CostEntry,
    ECMCostModel,
    CostBreakdown,
    SwedishCostCalculatorV2,
    ECM_COSTS_V2,
    REGIONAL_MULTIPLIERS,
    size_scaling_factor,
    quick_estimate,
    compare_costs_by_region,
)


class TestCostEntry:
    """Test CostEntry dataclass."""

    def test_inflation_forward(self):
        """Costs inflate forward correctly."""
        entry = CostEntry(
            value_sek=1000,
            unit="SEK/m²",
            source=CostSource.BEBO_LONSAMHET_2023,
            year=2023,
            confidence=0.8,
        )

        # 4% annual inflation for 2 years
        inflated = entry.inflate_to(2025, annual_rate=0.04)
        expected = 1000 * (1.04 ** 2)  # 1081.6

        assert inflated == pytest.approx(expected, rel=0.001)

    def test_inflation_same_year(self):
        """Same year returns original value."""
        entry = CostEntry(
            value_sek=1000,
            unit="SEK/m²",
            source=CostSource.WIKELLS_2024,
            year=2024,
            confidence=0.8,
        )

        assert entry.inflate_to(2024) == 1000

    def test_confidence_adjustment(self):
        """Low confidence adds safety buffer."""
        high_conf = CostEntry(
            value_sek=1000,
            unit="SEK/m²",
            source=CostSource.WIKELLS_2024,
            year=2024,
            confidence=0.9,
        )
        low_conf = CostEntry(
            value_sek=1000,
            unit="SEK/m²",
            source=CostSource.ESTIMATED,
            year=2024,
            confidence=0.5,
        )

        assert high_conf.with_confidence_adjustment() == 1000
        assert low_conf.with_confidence_adjustment() == 1200  # 20% buffer


class TestRegionalMultipliers:
    """Test regional cost variations."""

    def test_stockholm_premium(self):
        """Stockholm has highest multiplier."""
        assert REGIONAL_MULTIPLIERS[Region.STOCKHOLM] > 1.1

    def test_rural_discount(self):
        """Rural areas have discount."""
        assert REGIONAL_MULTIPLIERS[Region.RURAL] < 1.0

    def test_medium_city_baseline(self):
        """Medium city is baseline (1.0)."""
        assert REGIONAL_MULTIPLIERS[Region.MEDIUM_CITY] == 1.0


class TestSizeScaling:
    """Test building size economies of scale."""

    def test_small_building_premium(self):
        """Small buildings have cost premium."""
        small = size_scaling_factor(200)
        medium = size_scaling_factor(1000)

        assert small > medium

    def test_large_building_discount(self):
        """Large buildings have discount."""
        medium = size_scaling_factor(1000)
        large = size_scaling_factor(5000)

        assert large < medium

    def test_baseline_1000(self):
        """1000 m² is baseline (1.0)."""
        assert size_scaling_factor(1000) == pytest.approx(1.0, rel=0.01)

    def test_scaling_clamped(self):
        """Scaling factor is clamped to [0.7, 1.4]."""
        tiny = size_scaling_factor(10)
        huge = size_scaling_factor(100000)

        assert 0.7 <= tiny <= 1.4
        assert 0.7 <= huge <= 1.4


class TestECMCostModel:
    """Test ECM cost model calculations."""

    @pytest.fixture
    def sample_model(self):
        """Create a sample ECM cost model."""
        return ECMCostModel(
            ecm_id="test_ecm",
            name_sv="Test åtgärd",
            material_cost=CostEntry(
                value_sek=100,
                unit="SEK/m²",
                source=CostSource.WIKELLS_2024,
                year=2024,
                confidence=0.8,
            ),
            labor_cost=CostEntry(
                value_sek=50,
                unit="SEK/m²",
                source=CostSource.WIKELLS_2024,
                year=2024,
                confidence=0.8,
            ),
            fixed_cost=CostEntry(
                value_sek=10000,
                unit="SEK/building",
                source=CostSource.ESTIMATED,
                year=2024,
                confidence=0.7,
            ),
            lifetime_years=25,
            rot_eligible=True,
            green_tech_eligible=False,
        )

    def test_basic_calculation(self, sample_model):
        """Basic cost calculation works for private owner (ROT eligible)."""
        cost = sample_model.calculate_cost(
            quantity=100,  # 100 m²
            year=2024,
            region=Region.MEDIUM_CITY,
            floor_area_m2=1000,
            owner_type=OwnerType.PRIVATE,  # ROT only for private!
        )

        # Material: 100 * 100 = 10,000
        # Labor: 100 * 50 = 5,000
        # Fixed: 10,000
        # Total before: 25,000
        # ROT on labor: 50% of 5,000 = 2,500 (only for PRIVATE)
        # Total after: 22,500

        assert cost.material_cost == pytest.approx(10000, rel=0.01)
        assert cost.labor_cost == pytest.approx(5000, rel=0.01)
        assert cost.rot_deduction == pytest.approx(2500, rel=0.01)

    def test_brf_no_rot(self, sample_model):
        """BRF owners do NOT get ROT deduction."""
        cost = sample_model.calculate_cost(
            quantity=100,
            year=2024,
            region=Region.MEDIUM_CITY,
            floor_area_m2=1000,
            owner_type=OwnerType.BRF,  # BRF = no ROT
        )

        # BRF should have NO ROT deduction
        assert cost.rot_deduction == 0
        # Total = material + labor + fixed = 25,000
        assert cost.total_after_deductions == pytest.approx(25000, rel=0.01)

    def test_regional_adjustment(self, sample_model):
        """Regional multiplier applies."""
        cost_base = sample_model.calculate_cost(
            quantity=100,
            year=2024,
            region=Region.MEDIUM_CITY,
            floor_area_m2=1000,
        )
        cost_sthlm = sample_model.calculate_cost(
            quantity=100,
            year=2024,
            region=Region.STOCKHOLM,
            floor_area_m2=1000,
        )

        # Stockholm should be ~18% more
        ratio = cost_sthlm.total_before_deductions / cost_base.total_before_deductions
        assert ratio == pytest.approx(1.18, rel=0.05)

    def test_rot_cap_50k(self):
        """ROT deduction capped at 50,000 SEK (private owners only)."""
        model = ECMCostModel(
            ecm_id="big_project",
            name_sv="Stort projekt",
            material_cost=CostEntry(
                value_sek=500,
                unit="SEK/m²",
                source=CostSource.ESTIMATED,
                year=2024,
                confidence=0.7,
            ),
            labor_cost=CostEntry(
                value_sek=500,
                unit="SEK/m²",
                source=CostSource.ESTIMATED,
                year=2024,
                confidence=0.7,
            ),
            rot_eligible=True,
        )

        # 1000 m² = 500,000 labor, 50% = 250,000 but capped at 50k
        cost = model.calculate_cost(
            quantity=1000,
            year=2024,
            region=Region.MEDIUM_CITY,
            floor_area_m2=1000,
            owner_type=OwnerType.PRIVATE,  # ROT only for private!
        )

        assert cost.rot_deduction == 50000

    def test_green_tech_deduction(self):
        """Green tech deduction applies 15% (private owners only)."""
        model = ECMCostModel(
            ecm_id="solar_test",
            name_sv="Solcellstest",
            material_cost=CostEntry(
                value_sek=10000,
                unit="SEK/kWp",
                source=CostSource.ESTIMATED,
                year=2025,
                confidence=0.8,
            ),
            labor_cost=CostEntry(
                value_sek=4000,
                unit="SEK/kWp",
                source=CostSource.ESTIMATED,
                year=2025,
                confidence=0.8,
            ),
            rot_eligible=True,
            green_tech_eligible=True,
        )

        cost = model.calculate_cost(
            quantity=10,  # 10 kWp
            year=2025,
            region=Region.MEDIUM_CITY,
            floor_area_m2=1000,
            owner_type=OwnerType.PRIVATE,  # Green tech only for private!
        )

        # Green tech: 15% of (material + labor) = 15% of 140,000 = 21,000
        assert cost.green_tech_deduction == pytest.approx(21000, rel=0.01)


class TestCostBreakdown:
    """Test cost breakdown reporting."""

    def test_to_dict(self):
        """Cost breakdown exports to dict."""
        breakdown = CostBreakdown(
            ecm_id="test",
            material_cost=10000,
            labor_cost=5000,
            fixed_cost=2000,
            rot_deduction=2500,
            green_tech_deduction=0,
            maintenance_cost=0,
            total_before_deductions=17000,
            total_after_deductions=14500,
            quantity=100,
            unit="SEK/m²",
            year=2025,
            region=Region.STOCKHOLM,
        )

        d = breakdown.to_dict()

        assert d["ecm_id"] == "test"
        assert d["material_cost_sek"] == 10000
        assert d["region"] == "stockholm"

    def test_summary_string(self):
        """Summary produces readable string."""
        breakdown = CostBreakdown(
            ecm_id="test",
            material_cost=10000,
            labor_cost=5000,
            fixed_cost=2000,
            rot_deduction=2500,
            green_tech_deduction=1000,
            maintenance_cost=500,
            total_before_deductions=17000,
            total_after_deductions=13000,
            quantity=100,
            unit="SEK/m²",
            year=2025,
            region=Region.STOCKHOLM,
        )

        summary = breakdown.summary()

        assert "test" in summary
        assert "Material" in summary
        assert "ROT" in summary
        assert "Green tech" in summary


class TestSwedishCostCalculatorV2:
    """Test the main calculator class."""

    def test_calculator_creation(self):
        """Calculator creates with defaults."""
        calc = SwedishCostCalculatorV2()

        assert calc.region == Region.MEDIUM_CITY
        assert calc.year == 2025
        assert len(calc.cost_database) > 0

    def test_calculate_known_ecm(self):
        """Calculate cost for known ECM."""
        calc = SwedishCostCalculatorV2()

        cost = calc.calculate_ecm_cost(
            ecm_id="air_sealing",
            quantity=1000,  # m² floor
            floor_area_m2=1000,
        )

        assert cost.ecm_id == "air_sealing"
        assert cost.total_after_deductions > 0

    def test_unknown_ecm_raises(self):
        """Unknown ECM raises ValueError."""
        calc = SwedishCostCalculatorV2()

        with pytest.raises(ValueError, match="Unknown ECM"):
            calc.calculate_ecm_cost("nonexistent_ecm", 100)

    def test_list_ecms(self):
        """List ECMs returns all available."""
        calc = SwedishCostCalculatorV2()
        ecms = calc.list_ecms()

        assert len(ecms) > 10
        assert "wall_external_insulation" in ecms
        assert "solar_pv" in ecms

    def test_get_ecm_info(self):
        """Get info for specific ECM."""
        calc = SwedishCostCalculatorV2()
        info = calc.get_ecm_info("ftx_installation")

        assert info is not None
        assert info.ecm_id == "ftx_installation"
        assert info.rot_eligible == True


class TestConvenienceFunctions:
    """Test convenience functions."""

    def test_quick_estimate(self):
        """Quick estimate returns reasonable value."""
        cost = quick_estimate(
            ecm_id="roof_insulation",
            quantity=200,  # m² roof
            region="stockholm",
            floor_area_m2=1000,
        )

        assert cost > 0
        assert cost < 1000000  # Sanity check

    def test_quick_estimate_unknown(self):
        """Unknown ECM returns 0."""
        cost = quick_estimate("nonexistent", 100)
        assert cost == 0

    def test_compare_regions(self):
        """Regional comparison returns all regions."""
        costs = compare_costs_by_region(
            ecm_id="window_replacement",
            quantity=50,  # m² window
            floor_area_m2=1000,
        )

        assert len(costs) == len(Region)
        assert costs["stockholm"] > costs["rural"]


class TestECMCostsV2Database:
    """Test the ECM_COSTS_V2 database."""

    def test_database_not_empty(self):
        """Database has entries."""
        assert len(ECM_COSTS_V2) >= 20

    def test_all_entries_have_required_fields(self):
        """All entries have required fields."""
        for ecm_id, model in ECM_COSTS_V2.items():
            assert model.ecm_id == ecm_id
            assert model.material_cost is not None
            assert model.labor_cost is not None
            assert model.lifetime_years > 0

    def test_sources_are_tracked(self):
        """All costs have traceable sources."""
        for ecm_id, model in ECM_COSTS_V2.items():
            assert model.material_cost.source is not None
            assert model.labor_cost.source is not None

    def test_key_ecms_present(self):
        """Important ECMs are in database."""
        key_ecms = [
            "wall_external_insulation",
            "roof_insulation",
            "window_replacement",
            "solar_pv",
            "ftx_installation",
            "exhaust_air_heat_pump",
            "ground_source_heat_pump",
        ]

        for ecm_id in key_ecms:
            assert ecm_id in ECM_COSTS_V2, f"Missing: {ecm_id}"

    def test_rot_eligibility_set(self):
        """Most measures have ROT set appropriately."""
        rot_eligible_count = sum(
            1 for m in ECM_COSTS_V2.values() if m.rot_eligible
        )

        # Most physical measures should be ROT-eligible
        assert rot_eligible_count > len(ECM_COSTS_V2) * 0.5


class TestIntegration:
    """Integration tests."""

    def test_full_calculation_flow(self):
        """Complete calculation flow works for BRF (default)."""
        # Create calculator for Stockholm - defaults to BRF (no ROT)
        calc = SwedishCostCalculatorV2(
            region=Region.STOCKHOLM,
            year=2025,
        )

        # Calculate wall insulation for large building
        cost = calc.calculate_ecm_cost(
            ecm_id="wall_external_insulation",
            quantity=1000,  # m² wall
            floor_area_m2=5000,  # Large building
            include_maintenance=True,
            analysis_period_years=30,
        )

        # Verify all components
        assert cost.material_cost > 0
        assert cost.labor_cost > 0
        assert cost.fixed_cost > 0
        assert cost.rot_deduction == 0  # BRF = NO ROT (only for private)
        assert cost.owner_type == OwnerType.BRF

        # Export and summary work
        d = cost.to_dict()
        s = cost.summary()

        assert isinstance(d, dict)
        assert len(s) > 100  # Reasonable summary length

    def test_private_owner_gets_deductions(self):
        """Private owner gets ROT and green tech deductions."""
        calc = SwedishCostCalculatorV2(
            region=Region.STOCKHOLM,
            year=2025,
            owner_type=OwnerType.PRIVATE,
        )

        # Solar PV - green tech eligible
        cost = calc.calculate_ecm_cost(
            ecm_id="solar_pv",
            quantity=10,  # 10 kWp
            floor_area_m2=200,
        )

        # Private owner should get green tech deduction
        assert cost.green_tech_deduction > 0
        assert cost.owner_type == OwnerType.PRIVATE
        assert cost.total_after_deductions < cost.total_before_deductions

    def test_package_comparison(self):
        """Can compare package costs."""
        calc = SwedishCostCalculatorV2(region=Region.MALMO)

        # Basic package
        basic_ecms = {
            "air_sealing": 1000,
            "heating_curve_adjustment": 1,
            "led_lighting": 500,
        }

        # Premium package
        premium_ecms = {
            "wall_external_insulation": 800,
            "window_replacement": 100,
            "solar_pv": 50,
        }

        basic_costs = calc.calculate_package_cost(basic_ecms, floor_area_m2=1000)
        premium_costs = calc.calculate_package_cost(premium_ecms, floor_area_m2=1000)

        basic_total = sum(c.total_after_deductions for c in basic_costs.values())
        premium_total = sum(c.total_after_deductions for c in premium_costs.values())

        # Premium should be significantly more expensive
        assert premium_total > basic_total * 5
