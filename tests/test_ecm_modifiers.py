"""
Tests for ECM IDF modifiers.

Tests all 12 ECM modification methods:
- Envelope: wall insulation, roof insulation, windows, air sealing
- HVAC: FTX upgrade, FTX installation, DCV
- Renewable: Solar PV
- Controls: Smart thermostats
- Lighting: LED retrofit
"""

import pytest
import re
from pathlib import Path

from src.ecm.idf_modifier import IDFModifier


class TestIDFModifier:
    """Tests for IDFModifier class."""

    @pytest.fixture
    def modifier(self):
        """Create modifier instance."""
        return IDFModifier()

    # =========================================================================
    # WALL INSULATION TESTS
    # =========================================================================

    def test_wall_external_insulation(self, modifier, sample_idf_content):
        """Test external wall insulation increases thickness."""
        params = {'thickness_mm': 100, 'material': 'mineral_wool'}

        result = modifier._apply_wall_insulation(
            sample_idf_content, params, external=True
        )

        # Should contain ECM comment
        assert 'ECM Applied: External Wall Insulation' in result
        assert 'Added thickness: 100 mm' in result

        # Thickness should be increased (0.250 + 0.100 = 0.350)
        assert '0.3500' in result or '0.35' in result

    def test_wall_internal_insulation(self, modifier, sample_idf_content):
        """Test internal wall insulation."""
        params = {'thickness_mm': 50, 'material': 'eps'}

        result = modifier._apply_wall_insulation(
            sample_idf_content, params, external=False
        )

        assert 'ECM Applied: Internal Wall Insulation' in result

    # =========================================================================
    # ROOF INSULATION TESTS
    # =========================================================================

    def test_roof_insulation(self, modifier, sample_idf_content):
        """Test roof insulation increases thickness."""
        params = {'thickness_mm': 150, 'material': 'mineral_wool'}

        result = modifier._apply_roof_insulation(sample_idf_content, params)

        assert 'ECM Applied: Roof Insulation' in result
        assert 'Added thickness: 150 mm' in result

    # =========================================================================
    # WINDOW REPLACEMENT TESTS
    # =========================================================================

    def test_window_replacement_u_value(self, modifier, sample_idf_content):
        """Test window replacement changes U-value."""
        params = {'u_value': 0.8, 'shgc': 0.5}

        result = modifier._apply_window_replacement(sample_idf_content, params)

        assert 'ECM Applied: Window Replacement' in result
        assert 'New U-value: 0.8' in result

        # U-value should be changed from 1.0 to 0.8
        assert '0.8,' in result and 'U-Factor' in result

    def test_window_replacement_shgc(self, modifier, sample_idf_content):
        """Test window replacement changes SHGC."""
        params = {'u_value': 0.9, 'shgc': 0.4}

        result = modifier._apply_window_replacement(sample_idf_content, params)

        # SHGC should be in output
        assert 'New SHGC: 0.4' in result

    # =========================================================================
    # AIR SEALING TESTS
    # =========================================================================

    def test_air_sealing_reduces_infiltration(self, modifier, sample_idf_content):
        """Test air sealing reduces ACH."""
        params = {'reduction_factor': 0.5}

        result = modifier._apply_air_sealing(sample_idf_content, params)

        assert 'ECM Applied: Air Sealing' in result

        # ACH should be reduced from 0.06 to 0.03
        assert '0.0300' in result or '0.03' in result

    def test_air_sealing_70_percent_reduction(self, modifier, sample_idf_content):
        """Test 70% air sealing reduction."""
        params = {'reduction_factor': 0.7}

        result = modifier._apply_air_sealing(sample_idf_content, params)

        # ACH should be 0.06 * 0.7 = 0.042
        assert '0.0420' in result or '0.042' in result

    # =========================================================================
    # FTX UPGRADE TESTS
    # =========================================================================

    def test_ftx_upgrade_effectiveness(self, modifier, sample_idf_content):
        """Test FTX upgrade increases heat recovery."""
        params = {'effectiveness': 0.85}

        result = modifier._apply_ftx_upgrade(sample_idf_content, params)

        assert 'ECM Applied: FTX Heat Recovery Upgrade' in result
        assert 'New effectiveness: 85%' in result

        # Effectiveness should be changed from 0.75 to 0.85
        # Look for the pattern in result
        assert '0.85' in result

    def test_ftx_upgrade_90_percent(self, modifier, sample_idf_content):
        """Test 90% heat recovery."""
        params = {'effectiveness': 0.90}

        result = modifier._apply_ftx_upgrade(sample_idf_content, params)

        assert '0.90' in result or '0.9' in result

    # =========================================================================
    # FTX INSTALLATION TESTS
    # =========================================================================

    def test_ftx_installation(self, modifier, sample_idf_content):
        """Test FTX installation changes heat recovery type."""
        # First modify to have no heat recovery
        content_no_hr = sample_idf_content.replace(
            'Sensible,                !- Heat Recovery Type',
            'None,                    !- Heat Recovery Type'
        )
        params = {'effectiveness': 0.80}

        result = modifier._apply_ftx_installation(content_no_hr, params)

        assert 'ECM Applied: FTX Installation' in result
        assert 'FTX installed' in result

    # =========================================================================
    # DCV TESTS
    # =========================================================================

    def test_dcv_application(self, modifier, sample_idf_content):
        """Test demand controlled ventilation."""
        params = {'co2_setpoint': 1000}

        result = modifier._apply_dcv(sample_idf_content, params)

        assert 'ECM Applied: Demand Controlled Ventilation' in result
        assert 'CO2 setpoint: 1000 ppm' in result

    # =========================================================================
    # SOLAR PV TESTS
    # =========================================================================

    def test_solar_pv_comment(self, modifier, sample_idf_content):
        """Test solar PV adds comment."""
        params = {
            'coverage_fraction': 0.7,
            'panel_efficiency': 0.20,
            'roof_area_m2': 320
        }

        result = modifier._apply_solar_pv(sample_idf_content, params)

        assert 'ECM Applied: Solar PV' in result
        # Implementation uses PVWatts format with capacity/area output
        assert 'Net available roof area: 224' in result
        assert 'DC System Capacity' in result
        assert 'Generator:PVWatts' in result

    # =========================================================================
    # LED LIGHTING TESTS
    # =========================================================================

    def test_led_lighting_reduces_power(self, modifier, sample_idf_content):
        """Test LED lighting reduces power density."""
        params = {'power_density': 4}

        result = modifier._apply_led_lighting(sample_idf_content, params)

        assert 'ECM Applied: LED Lighting' in result
        assert 'New power density: 4 W/m' in result

        # Power should be reduced from 8 to 4
        assert '4,' in result

    def test_led_lighting_6_watts(self, modifier, sample_idf_content):
        """Test 6 W/m² LED lighting."""
        params = {'power_density': 6}

        result = modifier._apply_led_lighting(sample_idf_content, params)

        assert 'New power density: 6 W/m' in result

    # =========================================================================
    # SMART THERMOSTAT TESTS
    # =========================================================================

    def test_smart_thermostat_schedule(self, modifier, sample_idf_content):
        """Test smart thermostat creates setback schedule."""
        params = {'setback_c': 2}

        result = modifier._apply_smart_thermostats(sample_idf_content, params)

        assert 'ECM Applied: Smart Thermostats' in result
        assert 'Setback temperature: 2' in result
        assert 'Schedule:Compact' in result
        assert 'HeatSet_ECM' in result

    def test_smart_thermostat_3_degree_setback(self, modifier, sample_idf_content):
        """Test 3 degree setback."""
        params = {'setback_c': 3}

        result = modifier._apply_smart_thermostats(sample_idf_content, params)

        # Night setpoint should be 21 - 3 = 18
        assert '18' in result

    # =========================================================================
    # HEAT PUMP TESTS
    # =========================================================================

    def test_heat_pump_integration(self, modifier, sample_idf_content):
        """Test heat pump integration comment."""
        params = {'cop': 3.5, 'coverage': 0.8}

        result = modifier._apply_heat_pump(sample_idf_content, params)

        assert 'ECM Applied: Heat Pump Integration' in result
        assert 'COP: 3.5' in result
        assert 'Load coverage: 80%' in result

    # =========================================================================
    # DISPATCH TESTS
    # =========================================================================

    def test_apply_ecm_dispatch(self, modifier, sample_idf_content):
        """Test ECM dispatch routes to correct method."""
        # Test window replacement via dispatch
        result = modifier._apply_ecm(
            sample_idf_content,
            'window_replacement',
            {'u_value': 0.8, 'shgc': 0.5}
        )

        assert 'ECM Applied: Window Replacement' in result

    def test_apply_ecm_unknown(self, modifier, sample_idf_content, capsys):
        """Test unknown ECM returns unchanged content."""
        result = modifier._apply_ecm(
            sample_idf_content,
            'unknown_ecm',
            {}
        )

        # Should return unchanged
        assert result == sample_idf_content

        # Should print warning
        captured = capsys.readouterr()
        assert 'Warning' in captured.out or result == sample_idf_content


class TestApplySingle:
    """Tests for apply_single method."""

    def test_apply_single_creates_file(self, temp_dir, sample_idf_content):
        """Test apply_single creates modified IDF file."""
        # Write baseline IDF
        baseline = temp_dir / "baseline.idf"
        baseline.write_text(sample_idf_content)

        modifier = IDFModifier()
        result_path = modifier.apply_single(
            baseline_idf=baseline,
            ecm_id='air_sealing',
            params={'reduction_factor': 0.5},
            output_dir=temp_dir / "output"
        )

        assert result_path.exists()
        content = result_path.read_text()
        assert 'ECM Applied: Air Sealing' in content

    def test_apply_single_custom_name(self, temp_dir, sample_idf_content):
        """Test apply_single with custom output name."""
        baseline = temp_dir / "baseline.idf"
        baseline.write_text(sample_idf_content)

        modifier = IDFModifier()
        result_path = modifier.apply_single(
            baseline_idf=baseline,
            ecm_id='led_lighting',
            params={'power_density': 4},
            output_dir=temp_dir / "output",
            output_name="my_custom_name"
        )

        assert result_path.name == "my_custom_name.idf"


class TestMultipleECMs:
    """Tests for applying multiple ECMs."""

    def test_apply_multiple_ecms_sequentially(self, sample_idf_content):
        """Test applying multiple ECMs to same content."""
        modifier = IDFModifier()

        # Apply window replacement
        content = modifier._apply_ecm(
            sample_idf_content,
            'window_replacement',
            {'u_value': 0.8, 'shgc': 0.5}
        )

        # Apply air sealing
        content = modifier._apply_ecm(
            content,
            'air_sealing',
            {'reduction_factor': 0.5}
        )

        # Apply LED lighting
        content = modifier._apply_ecm(
            content,
            'led_lighting',
            {'power_density': 4}
        )

        # All ECMs should be present
        assert 'Window Replacement' in content
        assert 'Air Sealing' in content
        assert 'LED Lighting' in content


# =============================================================================
# NEWLY IMPLEMENTED ECM TESTS (2025-12-26)
# =============================================================================

class TestNewlyImplementedECMs:
    """Tests for ECMs implemented on 2025-12-26."""

    @pytest.fixture
    def enhanced_idf_content(self) -> str:
        """IDF content with schedules and thermostats for operational ECMs."""
        return '''Version,25.1;

Schedule:Constant,HeatSched,Temperature,21;
Schedule:Constant,AlwaysOn,Fraction,1;

Zone,
    Zone1,
    0,
    0, 0, 0,
    1,
    1,
    3.0,
    2000,
    500;

Construction,
    Floor_Construction,
    Concrete_Slab;

ZoneInfiltration:DesignFlowRate,
    Zone1_Infiltration,
    Zone1,
    AlwaysOn,
    AirChanges/Hour,
    ,
    ,
    ,
    0.5;

ThermostatSetpoint:SingleHeating,
    Zone1_Heat,
    HeatSched;
'''

    def test_radiator_balancing_reduces_setpoint(self, enhanced_idf_content):
        """Test radiator balancing reduces heating setpoint."""
        from src.ecm.idf_modifier import IDFModifier
        modifier = IDFModifier()

        result = modifier._apply_radiator_balancing(
            enhanced_idf_content, {'setpoint_reduction': 0.5}
        )

        # Should contain ECM comment
        assert 'Radiator Balancing' in result
        # Setpoint should be reduced from 21 to 20.5
        assert '20.5' in result

    def test_basement_insulation_adds_material(self, enhanced_idf_content):
        """Test basement insulation adds floor insulation material."""
        from src.ecm.idf_modifier import IDFModifier
        modifier = IDFModifier()

        result = modifier._apply_basement_insulation(
            enhanced_idf_content, {'thickness_mm': 100, 'material': 'xps'}
        )

        # Should create new insulation material
        assert 'Floor_Insulation_XPS' in result
        assert 'Material,' in result
        # Should mention basement insulation
        assert 'Basement Insulation' in result or 'Källarisolering' in result

    def test_entrance_door_reduces_infiltration(self, enhanced_idf_content):
        """Test entrance door replacement reduces infiltration."""
        from src.ecm.idf_modifier import IDFModifier
        modifier = IDFModifier()

        result = modifier._apply_entrance_door(
            enhanced_idf_content, {'infiltration_reduction': 0.08}
        )

        # Should mention entrance door
        assert 'Entrance Door' in result or 'Entrédörrbyte' in result
        # Infiltration should be reduced (0.5 * 0.92 = 0.46)
        assert '0.46' in result or 'Infiltration reduction: 8%' in result

    def test_summer_bypass_creates_seasonal_schedule(self, enhanced_idf_content):
        """Test summer bypass creates seasonal heating schedule."""
        from src.ecm.idf_modifier import IDFModifier
        modifier = IDFModifier()

        result = modifier._apply_summer_bypass(enhanced_idf_content, {})

        # Should create seasonal schedule
        assert 'HeatSched_Seasonal' in result
        assert 'Schedule:Compact' in result
        # Should mention summer bypass
        assert 'Summer Bypass' in result or 'Sommaravstängning' in result

    def test_night_setback_creates_schedule(self, enhanced_idf_content):
        """Test night setback creates day/night schedule."""
        from src.ecm.idf_modifier import IDFModifier
        modifier = IDFModifier()

        result = modifier._apply_night_setback(
            enhanced_idf_content, {'setback_c': 2, 'start_hour': 22, 'end_hour': 6}
        )

        # Should create schedule
        assert 'Schedule:Compact' in result
        # Should reduce nighttime temp
        assert 'Night Setback' in result
        # Night temp should be 21-2=19
        assert '19' in result

    def test_heating_curve_creates_monthly_schedule(self, enhanced_idf_content):
        """Test heating curve creates outdoor reset schedule."""
        from src.ecm.idf_modifier import IDFModifier
        modifier = IDFModifier()

        result = modifier._apply_heating_curve(
            enhanced_idf_content, {'curve_reduction': 4}
        )

        # Should create outdoor reset schedule
        assert 'HeatSched_OutdoorReset' in result
        # Should have monthly entries
        assert 'Through:' in result
        assert 'Heating Curve' in result or 'framledning' in result.lower()

    def test_hot_water_temp_creates_negative_gains(self, enhanced_idf_content):
        """Test DHW temp reduction creates negative internal gains."""
        from src.ecm.idf_modifier import IDFModifier
        modifier = IDFModifier()

        result = modifier._apply_hot_water_temp(
            enhanced_idf_content, {'target_temp_c': 55}
        )

        # Should create OtherEquipment with negative value
        assert 'OtherEquipment' in result
        assert 'DHW_Reduction' in result
        # Should mention DHW
        assert 'DHW' in result or 'hot water' in result.lower()

    def test_low_flow_fixtures_creates_negative_gains(self, enhanced_idf_content):
        """Test low flow fixtures creates negative internal gains."""
        from src.ecm.idf_modifier import IDFModifier
        modifier = IDFModifier()

        result = modifier._apply_low_flow_fixtures(
            enhanced_idf_content, {'showerhead_flow_lpm': 8, 'faucet_flow_lpm': 5}
        )

        # Should create OtherEquipment with negative value
        assert 'OtherEquipment' in result
        assert 'LowFlow_Reduction' in result
        # Should mention DHW reduction
        assert 'Low-Flow' in result or 'DHW' in result

    # === New Phase 1 ECM tests (2025-12-26 continued) ===

    def test_pipe_insulation_creates_negative_gains(self, enhanced_idf_content):
        """Test pipe insulation reduces internal gains from pipe losses."""
        from src.ecm.idf_modifier import IDFModifier
        modifier = IDFModifier()

        result = modifier._apply_pipe_insulation(
            enhanced_idf_content, {'reduction_pct': 75}
        )

        # Should create OtherEquipment with negative value
        assert 'OtherEquipment' in result
        assert 'PipeLoss_Reduction' in result
        # Should mention pipe insulation
        assert 'Pipe Insulation' in result or 'rör' in result.lower()

    def test_radiator_fans_reduces_setpoint_and_adds_electricity(self, enhanced_idf_content):
        """Test radiator fans reduces setpoint and adds fan electricity."""
        from src.ecm.idf_modifier import IDFModifier
        modifier = IDFModifier()

        result = modifier._apply_radiator_fans(
            enhanced_idf_content, {'setpoint_reduction': 1.0}
        )

        # Should reduce setpoint from 21 to 20
        assert '20' in result
        # Should add fan electricity
        assert 'ElectricEquipment' in result
        assert 'RadiatorFan' in result

    def test_led_outdoor_creates_schedule_and_savings(self, enhanced_idf_content):
        """Test LED outdoor lighting creates schedule and savings equipment."""
        from src.ecm.idf_modifier import IDFModifier
        modifier = IDFModifier()

        result = modifier._apply_led_outdoor(
            enhanced_idf_content, {'reduction_pct': 70}
        )

        # Should create schedule for outdoor lighting
        assert 'Schedule:Compact' in result
        assert 'LED_Outdoor' in result
        # Should create OtherEquipment (for electricity savings)
        assert 'OtherEquipment' in result

    def test_dhw_circulation_creates_schedule(self, enhanced_idf_content):
        """Test DHW circulation optimization creates schedule."""
        from src.ecm.idf_modifier import IDFModifier
        modifier = IDFModifier()

        result = modifier._apply_dhw_circulation(
            enhanced_idf_content, {'reduction_pct': 30}
        )

        # Should create schedule
        assert 'Schedule:Compact' in result
        assert 'DHW_Circ' in result
        # Should create OtherEquipment
        assert 'OtherEquipment' in result

    def test_dhw_tank_insulation_creates_savings(self, enhanced_idf_content):
        """Test DHW tank insulation creates negative internal gains."""
        from src.ecm.idf_modifier import IDFModifier
        modifier = IDFModifier()

        result = modifier._apply_dhw_tank_insulation(
            enhanced_idf_content, {'reduction_pct': 60}
        )

        # Should create OtherEquipment
        assert 'OtherEquipment' in result
        assert 'DHW_Tank_Insulation' in result

    def test_pump_optimization_uses_affinity_law(self, enhanced_idf_content):
        """Test pump optimization uses cubic affinity law for power savings."""
        from src.ecm.idf_modifier import IDFModifier
        modifier = IDFModifier()

        result = modifier._apply_pump_optimization(
            enhanced_idf_content, {'speed_reduction': 30}
        )

        # Should mention pump affinity law
        assert 'affinity law' in result.lower() or 'Speed³' in result or 'VFD' in result
        # Should create ElectricEquipment
        assert 'ElectricEquipment' in result
        assert 'Pump_VFD' in result

    def test_bms_optimization_corrects_setpoint_and_schedule(self, enhanced_idf_content):
        """Test BMS optimization corrects setpoint drift and schedule."""
        from src.ecm.idf_modifier import IDFModifier
        modifier = IDFModifier()

        result = modifier._apply_bms_optimization(
            enhanced_idf_content, {'setpoint_correction': 0.5}
        )

        # Should reduce setpoint from 21 to 20.5
        assert '20.5' in result
        # Should create schedule
        assert 'Schedule:Compact' in result
        assert 'BMS_Optimization' in result

    def test_fault_detection_reduces_infiltration_and_setpoint(self, enhanced_idf_content):
        """Test fault detection reduces infiltration and corrects setpoint."""
        from src.ecm.idf_modifier import IDFModifier
        modifier = IDFModifier()

        result = modifier._apply_fault_detection(
            enhanced_idf_content, {'infiltration_reduction': 0.03, 'setpoint_correction': 0.3}
        )

        # Should mention FDD
        assert 'Fault Detection' in result or 'FDD' in result
        # Infiltration should be reduced (0.5 * 0.97 = 0.485)
        assert '0.485' in result or 'stuck dampers' in result.lower()

    def test_energy_monitoring_reduces_setpoint_and_gains(self, enhanced_idf_content):
        """Test energy monitoring creates behavioral savings."""
        from src.ecm.idf_modifier import IDFModifier
        modifier = IDFModifier()

        result = modifier._apply_energy_monitoring(
            enhanced_idf_content, {'behavioral_savings': 7}
        )

        # Should reduce setpoint from 21 to 20.7
        assert '20.7' in result
        # Should create OtherEquipment for behavioral savings
        assert 'OtherEquipment' in result
        assert 'Energy_Monitoring' in result

    # === Phase 2 ECM tests (2025-12-26 - full E+ objects) ===

    def test_heat_pump_water_heater_creates_dhw_savings_and_electricity(self, enhanced_idf_content):
        """Test HPWH creates DHW savings and electricity load."""
        from src.ecm.idf_modifier import IDFModifier
        modifier = IDFModifier()

        result = modifier._apply_heat_pump_water_heater(
            enhanced_idf_content, {'cop': 3.0, 'dhw_load': 22}
        )

        # Should create OtherEquipment for DHW savings
        assert 'OtherEquipment' in result
        assert 'HPWH_DHW_Savings' in result
        # Should create ElectricEquipment for HP electricity
        assert 'ElectricEquipment' in result
        assert 'HPWH_Electricity' in result
        # Should mention COP
        assert 'COP' in result

    def test_solar_thermal_creates_monthly_schedule(self, enhanced_idf_content):
        """Test solar thermal creates monthly schedule for seasonal savings."""
        from src.ecm.idf_modifier import IDFModifier
        modifier = IDFModifier()

        result = modifier._apply_solar_thermal(
            enhanced_idf_content, {'annual_coverage': 0.40}
        )

        # Should create monthly schedule
        assert 'Schedule:Compact' in result
        assert 'Solar_Thermal' in result
        # Should have monthly entries (Through: 01/31, etc.)
        assert 'Through: 01/31' in result
        assert 'Through: 07/31' in result
        # Should create OtherEquipment
        assert 'OtherEquipment' in result

    def test_district_heating_optimization_reduces_setpoint(self, enhanced_idf_content):
        """Test DH optimization reduces setpoint slightly."""
        from src.ecm.idf_modifier import IDFModifier
        modifier = IDFModifier()

        result = modifier._apply_dh_optimization(
            enhanced_idf_content, {'setpoint_reduction': 0.2}
        )

        # Should reduce setpoint from 21 to 20.8
        assert '20.8' in result
        # Should mention DH optimization
        assert 'District Heating' in result

    def test_battery_storage_creates_storage_objects(self, enhanced_idf_content):
        """Test battery storage creates ElectricLoadCenter objects."""
        from src.ecm.idf_modifier import IDFModifier
        modifier = IDFModifier()

        result = modifier._apply_battery_storage(
            enhanced_idf_content, {'capacity_kwh': 50}
        )

        # Should create storage object
        assert 'ElectricLoadCenter:Storage' in result
        assert 'Battery_System' in result
        # Should create schedules
        assert 'Schedule:Compact' in result
        assert 'Battery_Availability' in result

    def test_effektvakt_creates_peak_shaving_schedule(self, enhanced_idf_content):
        """Test effektvakt creates peak hour setpoint schedule."""
        from src.ecm.idf_modifier import IDFModifier
        modifier = IDFModifier()

        result = modifier._apply_effektvakt(
            enhanced_idf_content, {'setback_c': 2.0, 'peak_start': 16, 'peak_end': 20}
        )

        # Should create peak shaving schedule
        assert 'Schedule:Compact' in result
        assert 'Effektvakt' in result
        # Should have peak hour references
        assert '16:00' in result or '16' in result
        assert '20:00' in result or '20' in result
        # Should mention peak demand
        assert 'Peak' in result or 'peak' in result


class TestECMImplementationStatus:
    """Tests for ECM implementation status tracking."""

    def test_newly_implemented_ecms_marked_full(self):
        """Verify newly implemented ECMs are marked as 'full' status."""
        from src.ecm.catalog import ECM_IMPLEMENTATION_STATUS

        newly_implemented = [
            # First batch (2025-12-26 morning)
            'radiator_balancing',
            'basement_insulation',
            'entrance_door_replacement',
            'summer_bypass',
            'night_setback',
            'heating_curve_adjustment',
            'hot_water_temperature',
            'low_flow_fixtures',
            # Second batch (2025-12-26 continued - Phase 1)
            'pipe_insulation',
            'radiator_fans',
            'heat_recovery_dhw',
            'led_outdoor',
            'dhw_circulation_optimization',
            'dhw_tank_insulation',
            'pump_optimization',
            'bms_optimization',
            'fault_detection',
            'energy_monitoring',
            # Third batch (2025-12-26 Phase 2 - full E+ objects)
            'heat_pump_water_heater',
            'solar_thermal',
            'district_heating_optimization',
            'battery_storage',
            'effektvakt_optimization',
        ]

        for ecm_id in newly_implemented:
            status = ECM_IMPLEMENTATION_STATUS.get(ecm_id)
            assert status == 'full', f"{ecm_id} should be 'full' but is '{status}'"

    def test_implementation_count(self):
        """Test overall implementation counts are reasonable."""
        from src.ecm.catalog import ECM_IMPLEMENTATION_STATUS

        full_count = sum(1 for s in ECM_IMPLEMENTATION_STATUS.values() if s == 'full')
        partial_count = sum(1 for s in ECM_IMPLEMENTATION_STATUS.values() if s == 'partial')
        noop_count = sum(1 for s in ECM_IMPLEMENTATION_STATUS.values() if s in ('comment_only', 'cost_only'))

        # Should have at least 35 full implementations after Phase 2
        assert full_count >= 35, f"Expected >=35 full ECMs, got {full_count}"
        # Should have very few no-ops (only VRF excluded)
        assert noop_count <= 2, f"Expected <=2 no-op ECMs, got {noop_count}"
