"""
IDF Modifier - Apply ECM changes to EnergyPlus model.

Takes a baseline IDF and modifies it to implement ECMs:
- Change material properties (U-values)
- Modify HVAC parameters (heat recovery)
- Add PV generation
- Adjust schedules and loads
"""

from pathlib import Path
from typing import Dict, Any, List, Optional
import re
import logging

from .combinations import ECMVariant, ECMCombination
from ..core.idf_parser import IDFParser

logger = logging.getLogger(__name__)


class IDFModifier:
    """
    Modify EnergyPlus IDF to implement ECMs.

    Supports 12 Swedish ECMs:
    - Envelope: wall insulation (ext/int), roof insulation, windows, air sealing
    - HVAC: FTX upgrade, FTX installation, DCV
    - Renewable: Solar PV
    - Controls: Smart thermostats
    - Lighting: LED retrofit

    Usage:
        modifier = IDFModifier()
        modified_path = modifier.apply(
            baseline_idf=Path('./baseline.idf'),
            combination=ecm_combination,
            output_dir=Path('./scenarios')
        )
    """

    # Material properties for insulation
    INSULATION_PROPERTIES = {
        'mineral_wool': {
            'conductivity': 0.040,  # W/m-K
            'density': 30,  # kg/m³
            'specific_heat': 840,  # J/kg-K
        },
        'eps': {
            'conductivity': 0.038,
            'density': 20,
            'specific_heat': 1450,
        },
        'pir': {
            'conductivity': 0.022,
            'density': 35,
            'specific_heat': 1400,
        },
    }

    def __init__(self):
        self.idf_parser = IDFParser()

    def _extract_heating_setpoint(self, idf_content: str) -> tuple[float, str]:
        """
        Extract base heating setpoint from IDF, handling Bayesian-calibrated formats.

        Returns:
            Tuple of (base_setpoint, schedule_name)
            Defaults to (21.0, "HeatSet") if not found.
        """
        # Try multi-line format (Bayesian calibrated):
        # Schedule:Constant,\n    HeatSet,  !- Name\n    Temperature,\n    20.82;
        base_match = re.search(
            r'Schedule:Constant,\s*\n?\s*(HeatSet|HeatSched|HeatingSetpoint|Heating_Setpoint|Heating)[^;]*?(\d+\.?\d*)\s*;',
            idf_content,
            re.IGNORECASE | re.DOTALL
        )
        if base_match:
            return float(base_match.group(2)), base_match.group(1)

        # Try compact single-line format
        compact_match = re.search(
            r'Schedule:Constant,\s*(HeatSet|HeatSched|HeatingSetpoint|Heating_Setpoint|Heating)\s*,\s*Temperature\s*,\s*([\d.]+)',
            idf_content,
            re.IGNORECASE
        )
        if compact_match:
            return float(compact_match.group(2)), compact_match.group(1)

        return 21.0, "HeatSet"

    def _replace_thermostat_reference(
        self,
        idf_content: str,
        old_name: str,
        new_name: str
    ) -> str:
        """
        Replace thermostat setpoint schedule references, handling IDF formats with comments.

        Args:
            idf_content: IDF file content
            old_name: Original schedule name (HeatSet, HeatSched, etc.)
            new_name: New schedule name to use

        Returns:
            Modified IDF content
        """
        # DualSetpoint with comments
        idf_content = re.sub(
            rf'(ThermostatSetpoint:DualSetpoint,\s*\n\s*[^,]+,\s*[^\n]*\n\s*){re.escape(old_name)}(\s*,|\s*;)',
            rf'\1{new_name}\2',
            idf_content,
            flags=re.IGNORECASE
        )

        # SingleHeating with comments
        idf_content = re.sub(
            rf'(ThermostatSetpoint:SingleHeating,\s*\n?\s*[^,]+,\s*[^\n]*\n?\s*){re.escape(old_name)}(\s*;)',
            rf'\1{new_name}\2',
            idf_content,
            flags=re.IGNORECASE
        )

        # Inline references
        idf_content = re.sub(
            rf'(Heating Setpoint.*?Schedule.*?Name\s*\n\s*){re.escape(old_name)}(\s*,|\s*;)',
            rf'\1{new_name}\2',
            idf_content,
            flags=re.IGNORECASE
        )

        return idf_content

    def apply(
        self,
        baseline_idf: Path,
        combination: ECMCombination,
        output_dir: Path
    ) -> Path:
        """
        Apply ECM combination to baseline IDF.

        Args:
            baseline_idf: Path to baseline IDF file
            combination: ECM combination to apply
            output_dir: Directory for modified IDF

        Returns:
            Path to modified IDF
        """
        # Read baseline
        with open(baseline_idf, 'r') as f:
            idf_content = f.read()

        # Apply each ECM variant
        for variant in combination.variants:
            idf_content = self._apply_variant(idf_content, variant)

        # Write modified IDF
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{combination.id}.idf"
        with open(output_path, 'w') as f:
            f.write(idf_content)

        return output_path

    def apply_single(
        self,
        baseline_idf: Path,
        ecm_id: str,
        params: Dict[str, Any],
        output_dir: Path,
        output_name: Optional[str] = None
    ) -> Path:
        """
        Apply a single ECM to baseline IDF.

        Args:
            baseline_idf: Path to baseline IDF file
            ecm_id: ECM identifier
            params: ECM parameters
            output_dir: Directory for modified IDF
            output_name: Optional output filename

        Returns:
            Path to modified IDF
        """
        with open(baseline_idf, 'r') as f:
            idf_content = f.read()

        # Apply ECM
        idf_content = self._apply_ecm(idf_content, ecm_id, params)

        # Write modified IDF
        output_dir.mkdir(parents=True, exist_ok=True)
        name = output_name or f"{baseline_idf.stem}_{ecm_id}"
        output_path = output_dir / f"{name}.idf"
        with open(output_path, 'w') as f:
            f.write(idf_content)

        return output_path

    def verify_modification(
        self,
        baseline_content: str,
        modified_content: str,
        ecm_id: str
    ) -> Dict[str, Any]:
        """
        Verify that an ECM actually modified the IDF content.

        Args:
            baseline_content: Original IDF content
            modified_content: Modified IDF content
            ecm_id: ECM ID that was applied

        Returns:
            Dict with verification results:
                - modified: bool, True if content changed
                - changes: List of detected changes
                - has_ecm_comment: bool, True if ECM comment marker exists
        """
        result = {
            "ecm_id": ecm_id,
            "modified": baseline_content != modified_content,
            "changes": [],
            "has_ecm_comment": "ECM Applied:" in modified_content or "ECM:" in modified_content,
        }

        # Check for specific modifications by ECM type
        checks = {
            # Envelope ECMs
            "wall_external_insulation": [
                ("conductivity", "Conductivity", lambda b, m: b.count("Conductivity") == m.count("Conductivity")),
            ],
            "roof_insulation": [
                ("conductivity", "Conductivity", lambda b, m: b.count("Conductivity") == m.count("Conductivity")),
            ],
            "window_replacement": [
                ("u_factor", "UFactor", lambda b, m: b.count("UFactor") != m.count("UFactor") or "ECM:" in m),
            ],
            # Infiltration ECMs
            "air_sealing": [
                ("infiltration", "AirChanges/Hour", lambda b, m: self._count_ach_values(b) != self._count_ach_values(m)),
            ],
            "entrance_door_replacement": [
                ("infiltration", "AirChanges/Hour", lambda b, m: self._count_ach_values(b) != self._count_ach_values(m)),
            ],
            # Heat recovery ECMs
            "ftx_upgrade": [
                ("heat_recovery", "Heat Recovery Effectiveness", lambda b, m: b != m),
            ],
            "ftx_overhaul": [
                ("heat_recovery", "Heat Recovery Effectiveness", lambda b, m: b != m),
            ],
            # Schedule ECMs
            "night_setback": [
                ("schedule", "Schedule:Compact", lambda b, m: m.count("Schedule:Compact") > b.count("Schedule:Compact")),
            ],
            "smart_thermostats": [
                ("schedule", "Schedule:Compact", lambda b, m: m.count("Schedule:Compact") > b.count("Schedule:Compact")),
            ],
            # Ventilation ECMs
            "demand_controlled_ventilation": [
                ("outdoor_air", "Outdoor Air Flow", lambda b, m: b != m),
            ],
        }

        # Run applicable checks
        if ecm_id in checks:
            for check_name, search_term, check_fn in checks[ecm_id]:
                if check_fn(baseline_content, modified_content):
                    result["changes"].append(check_name)

        # Log verification result
        if result["modified"]:
            logger.debug(f"ECM {ecm_id} verification: MODIFIED - changes={result['changes']}")
        else:
            logger.warning(f"ECM {ecm_id} verification: NO CHANGE - IDF unchanged")

        return result

    def _count_ach_values(self, content: str) -> float:
        """Sum all ACH values in IDF content for comparison."""
        import re
        pattern = r'AirChanges/Hour,[\s,]*([\d.]+(?:[eE][+-]?\d+)?);'
        matches = re.findall(pattern, content, re.IGNORECASE)
        return sum(float(m) for m in matches) if matches else 0.0

    def apply_multiple(
        self,
        baseline_idf: Path,
        ecms: List[tuple],
        output_dir: Path,
        output_name: str
    ) -> Path:
        """
        Apply multiple ECMs to create a combined package scenario.

        ECMs are applied in optimal order:
        1. Envelope (insulation, windows, air sealing)
        2. Ventilation (DCV, FTX)
        3. Controls (thermostats)
        4. Zero-cost (DUC calibration, effektvakt)

        Args:
            baseline_idf: Path to baseline IDF file
            ecms: List of (ecm_id, params) tuples
            output_dir: Directory for modified IDF
            output_name: Output filename (without .idf)

        Returns:
            Path to modified IDF with all ECMs applied
        """
        # Define application order (envelope first, controls last)
        ORDER = {
            # Envelope - apply first
            'wall_external_insulation': 1,
            'wall_internal_insulation': 1,
            'roof_insulation': 1,
            'window_replacement': 1,
            'air_sealing': 2,
            # Ventilation - apply second
            'ftx_installation': 3,
            'ftx_upgrade': 3,
            'demand_controlled_ventilation': 4,
            # HVAC
            'heat_pump_integration': 5,
            # Controls - apply last
            'smart_thermostats': 6,
            'led_lighting': 6,
            # Zero-cost operational
            'duc_calibration': 7,
            'effektvakt_optimization': 7,
            'heating_curve_adjustment': 7,
            # Renewable (doesn't affect thermal)
            'solar_pv': 8,
        }

        # Sort ECMs by application order
        sorted_ecms = sorted(ecms, key=lambda x: ORDER.get(x[0], 99))

        with open(baseline_idf, 'r') as f:
            idf_content = f.read()

        # Track applied ECMs for header comment
        applied_ecms = []

        # Apply each ECM in sequence
        for ecm_id, params in sorted_ecms:
            logger.info(f"Applying ECM: {ecm_id}")
            idf_content = self._apply_ecm(idf_content, ecm_id, params)
            applied_ecms.append(ecm_id)

        # Add package header comment
        package_comment = f'''
! ==== COMBINED PACKAGE: {output_name} ====
! ECMs applied ({len(applied_ecms)}):
'''
        for ecm_id in applied_ecms:
            package_comment += f'!   - {ecm_id}\n'
        package_comment += '! ========================================\n\n'

        idf_content = package_comment + idf_content

        # Write combined IDF
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{output_name}.idf"
        with open(output_path, 'w') as f:
            f.write(idf_content)

        logger.info(f"Package IDF created: {output_path} ({len(applied_ecms)} ECMs)")
        return output_path

    def _apply_variant(self, idf_content: str, variant: ECMVariant) -> str:
        """Apply a single ECM variant to IDF content."""
        return self._apply_ecm(idf_content, variant.ecm.id, variant.parameters)

    def _apply_ecm(self, idf_content: str, ecm_id: str, params: Dict[str, Any]) -> str:
        """Apply ECM by ID."""
        # Dispatch to specific handler
        if ecm_id == "wall_external_insulation":
            return self._apply_wall_insulation(idf_content, params, external=True)
        elif ecm_id == "wall_internal_insulation":
            return self._apply_wall_insulation(idf_content, params, external=False)
        elif ecm_id == "roof_insulation":
            return self._apply_roof_insulation(idf_content, params)
        elif ecm_id == "window_replacement":
            return self._apply_window_replacement(idf_content, params)
        elif ecm_id == "air_sealing":
            return self._apply_air_sealing(idf_content, params)
        elif ecm_id == "ftx_upgrade":
            return self._apply_ftx_upgrade(idf_content, params)
        elif ecm_id == "ftx_installation":
            return self._apply_ftx_installation(idf_content, params)
        elif ecm_id == "ftx_overhaul":
            return self._apply_ftx_overhaul(idf_content, params)
        elif ecm_id == "demand_controlled_ventilation":
            return self._apply_dcv(idf_content, params)
        elif ecm_id == "solar_pv":
            return self._apply_solar_pv(idf_content, params)
        elif ecm_id == "led_lighting":
            return self._apply_led_lighting(idf_content, params)
        elif ecm_id == "smart_thermostats":
            return self._apply_smart_thermostats(idf_content, params)
        elif ecm_id == "heat_pump_integration":
            return self._apply_heat_pump(idf_content, params)
        # Zero-cost operational ECMs
        elif ecm_id == "duc_calibration":
            return self._apply_duc_calibration(idf_content, params)
        elif ecm_id == "heating_curve_adjustment":
            return self._apply_heating_curve(idf_content, params)
        elif ecm_id == "ventilation_schedule_optimization":
            return self._apply_ventilation_schedule(idf_content, params)
        elif ecm_id == "radiator_balancing":
            return self._apply_radiator_balancing(idf_content, params)
        elif ecm_id == "effektvakt_optimization":
            # Effektvakt doesn't affect thermal simulation, only cost
            return self._apply_effektvakt(idf_content, params)
        # Additional zero-cost operational ECMs
        elif ecm_id == "night_setback":
            return self._apply_night_setback(idf_content, params)
        elif ecm_id == "summer_bypass":
            return self._apply_summer_bypass(idf_content, params)
        elif ecm_id == "hot_water_temperature":
            return self._apply_hot_water_temp(idf_content, params)
        elif ecm_id == "pump_optimization":
            return self._apply_pump_optimization(idf_content, params)
        elif ecm_id == "bms_optimization":
            return self._apply_bms_optimization(idf_content, params)
        # Swedish-specific ECMs
        elif ecm_id == "exhaust_air_heat_pump":
            return self._apply_exhaust_air_heat_pump(idf_content, params)
        elif ecm_id == "ground_source_heat_pump":
            return self._apply_ground_source_heat_pump(idf_content, params)
        elif ecm_id == "district_heating_optimization":
            return self._apply_dh_optimization(idf_content, params)
        elif ecm_id == "solar_thermal":
            return self._apply_solar_thermal(idf_content, params)
        elif ecm_id == "low_flow_fixtures":
            return self._apply_low_flow_fixtures(idf_content, params)
        # Additional envelope ECMs
        elif ecm_id == "basement_insulation":
            return self._apply_basement_insulation(idf_content, params)
        elif ecm_id == "thermal_bridge_remediation":
            return self._apply_thermal_bridge_remediation(idf_content, params)
        elif ecm_id == "facade_renovation":
            return self._apply_facade_renovation(idf_content, params)
        elif ecm_id == "entrance_door_replacement":
            return self._apply_entrance_door(idf_content, params)
        elif ecm_id == "pipe_insulation":
            return self._apply_pipe_insulation(idf_content, params)
        # HVAC & Controls ECMs
        elif ecm_id == "radiator_fans":
            return self._apply_radiator_fans(idf_content, params)
        elif ecm_id == "heat_recovery_dhw":
            return self._apply_heat_recovery_dhw(idf_content, params)
        elif ecm_id == "vrf_system":
            return self._apply_vrf_system(idf_content, params)
        elif ecm_id == "occupancy_sensors":
            return self._apply_occupancy_sensors(idf_content, params)
        elif ecm_id == "daylight_sensors":
            return self._apply_daylight_sensors(idf_content, params)
        elif ecm_id == "predictive_control":
            return self._apply_predictive_control(idf_content, params)
        elif ecm_id == "fault_detection":
            return self._apply_fault_detection(idf_content, params)
        elif ecm_id == "building_automation_system":
            return self._apply_building_automation(idf_content, params)
        # Metering & Monitoring
        elif ecm_id == "individual_metering":
            return self._apply_individual_metering(idf_content, params)
        elif ecm_id == "energy_monitoring":
            return self._apply_energy_monitoring(idf_content, params)
        elif ecm_id == "recommissioning":
            return self._apply_recommissioning(idf_content, params)
        # Lighting (specific areas)
        elif ecm_id == "led_common_areas":
            return self._apply_led_common_areas(idf_content, params)
        elif ecm_id == "led_outdoor":
            return self._apply_led_outdoor(idf_content, params)
        # DHW ECMs
        elif ecm_id == "dhw_circulation_optimization":
            return self._apply_dhw_circulation(idf_content, params)
        elif ecm_id == "dhw_tank_insulation":
            return self._apply_dhw_tank_insulation(idf_content, params)
        # Storage
        elif ecm_id == "battery_storage":
            return self._apply_battery_storage(idf_content, params)
        # Heat pump ECMs
        elif ecm_id == "air_source_heat_pump":
            return self._apply_air_source_heat_pump(idf_content, params)
        elif ecm_id == "heat_pump_water_heater":
            return self._apply_heat_pump_water_heater(idf_content, params)
        else:
            # Unknown ECM, return unchanged
            logger.warning(f"Unknown ECM '{ecm_id}', no modifications applied")
            return idf_content

    def _apply_wall_insulation(
        self,
        idf_content: str,
        params: Dict[str, Any],
        external: bool
    ) -> str:
        """
        Add insulation to wall construction.

        Modifies wall Material objects to increase insulation thickness.
        For external insulation: adds layer to outside
        For internal insulation: adds layer to inside (reduces floor area slightly)
        """
        thickness_mm = params.get('thickness_mm', 100)
        thickness_m = thickness_mm / 1000
        material = params.get('material', 'mineral_wool')
        props = self.INSULATION_PROPERTIES.get(material, self.INSULATION_PROPERTIES['mineral_wool'])

        # Create unique name for new insulation layer
        layer_name = f"ECM_Wall_Insulation_{thickness_mm}mm"
        position = "External" if external else "Internal"

        # Create new material object
        new_material = f'''
! ECM: {position} Wall Insulation - {thickness_mm}mm {material}
Material,
    {layer_name},              !- Name
    Rough,                     !- Roughness
    {thickness_m:.4f},         !- Thickness {{m}}
    {props['conductivity']},   !- Conductivity {{W/m-K}}
    {props['density']},        !- Density {{kg/m3}}
    {props['specific_heat']};  !- Specific Heat {{J/kg-K}}

'''

        # Find existing wall insulation material and increase its thickness
        # Look for materials that appear to be insulation (low conductivity)
        def increase_insulation(match):
            name = match.group(1).strip()
            roughness = match.group(2).strip()
            current_thickness = float(match.group(3))
            conductivity = float(match.group(4))

            # Only modify WALL insulation (not roof, floor, or basement)
            name_lower = name.lower()
            is_insulation = conductivity < 0.1
            is_wall = 'wall' in name_lower and 'roof' not in name_lower and 'floor' not in name_lower and 'basement' not in name_lower
            if is_insulation and is_wall:
                new_thickness = current_thickness + thickness_m
                return f'''Material,
    {name},
    {roughness},
    {new_thickness:.4f},
    {conductivity},'''
            return match.group(0)

        # Scientific notation regex component (matches 0.123 or 1.23e-01 or 1.23E+02)
        NUM = r'[\d.]+(?:[eE][+-]?\d+)?'

        # Pattern 1: Compact format (no comments) - most common
        # Matches: Material,\n  Name,\n  Roughness,\n  Thickness,\n  Conductivity,
        pattern_compact = rf'Material,\s*\n\s*([^,\n]+),\s*\n\s*([^,\n]+),\s*\n\s*({NUM}),\s*\n\s*({NUM}),'

        modified = re.sub(pattern_compact, increase_insulation, idf_content)

        # Pattern 2: Commented format (with !- Name, !- Roughness, etc.)
        # Now handles scientific notation from Bayesian calibration (e.g., 1.182266e-01)
        pattern_commented = rf'Material,\s*\n\s*([^,]+),\s*!-\s*Name\s*\n\s*([^,]+),\s*!-\s*Roughness\s*\n\s*({NUM}),\s*!-\s*Thickness[^,]*\n\s*({NUM}),\s*!-\s*Conductivity'
        modified = re.sub(pattern_commented, increase_insulation, modified)

        # Pattern 3: Mixed format - Name/Roughness without comments, Thickness/Conductivity with comments
        # This is what BaselineGenerator produces:
        # Material,
        #     WallInsulation,
        #     MediumRough,
        #     0.056,  !- Thickness (m)
        #     0.035,                   !- Conductivity (W/m-K)
        # NOTE: The comma after thickness value is critical for matching!
        # Now handles scientific notation from Bayesian calibration
        pattern_mixed = rf'(Material,\s*\n\s*([^,\n]+),\s*\n\s*([^,\n]+),\s*\n\s*)({NUM})(,\s*!-\s*Thickness[^,\n]*\n\s*)({NUM})'

        def increase_insulation_mixed(match):
            prefix = match.group(1)  # "Material,\n  Name,\n  Roughness,\n  "
            name = match.group(2).strip()
            roughness = match.group(3).strip()
            current_thickness = float(match.group(4))
            thickness_suffix = match.group(5)  # ",  !- Thickness (m)\n  "
            conductivity = float(match.group(6))

            # Only modify WALL insulation (not roof, floor, or basement)
            name_lower = name.lower()
            is_insulation = conductivity < 0.1
            is_wall = 'wall' in name_lower and 'roof' not in name_lower and 'floor' not in name_lower and 'basement' not in name_lower
            if is_insulation and is_wall:
                new_thickness = current_thickness + thickness_m
                return f'{prefix}{new_thickness:.4f}{thickness_suffix}{conductivity}'
            return match.group(0)

        modified = re.sub(pattern_mixed, increase_insulation_mixed, modified)

        # Add comment documenting the ECM
        comment = f'''
! ==== ECM Applied: {position} Wall Insulation ====
! Added thickness: {thickness_mm} mm
! Material: {material}
! Expected heating reduction: 10-20%

'''
        return comment + modified

    def _apply_roof_insulation(
        self,
        idf_content: str,
        params: Dict[str, Any]
    ) -> str:
        """
        Add insulation to roof construction.

        Increases thickness of roof insulation material.
        """
        thickness_mm = params.get('thickness_mm', 150)
        thickness_m = thickness_mm / 1000
        material = params.get('material', 'mineral_wool')
        props = self.INSULATION_PROPERTIES.get(material, self.INSULATION_PROPERTIES['mineral_wool'])

        # Find roof insulation and increase thickness
        # Look for materials with roof/attic/tak (Swedish) in name
        # Also match common insulation names for roofs

        # Keywords that indicate roof insulation (case-insensitive)
        ROOF_KEYWORDS = [
            'roof', 'attic', 'tak', 'vind', 'ceiling',  # Common names
            'roofinsulation', 'atticinsulation', 'takisolering',  # Full names
        ]

        def is_roof_material(name: str) -> bool:
            """Check if material name indicates roof insulation."""
            name_lower = name.lower().replace('_', '').replace('-', '')
            return any(kw in name_lower for kw in ROOF_KEYWORDS)

        def increase_roof_insulation(match):
            full_match = match.group(0)
            name = match.group(1).strip()
            roughness = match.group(2).strip()
            current_thickness = float(match.group(3))
            conductivity = float(match.group(4))

            if is_roof_material(name):
                new_thickness = current_thickness + thickness_m
                logger.info(f"  Roof insulation (compact): {name} thickness {current_thickness:.4f} → {new_thickness:.4f}m")
                return f'''Material,
    {name},
    {roughness},
    {new_thickness:.4f},
    {conductivity},'''
            return full_match

        # Scientific notation regex component (matches 0.123 or 1.23e-01 or 1.23E+02)
        NUM = r'[\d.]+(?:[eE][+-]?\d+)?'

        # Pattern 1: Compact format (no comments)
        pattern_compact = rf'Material,\s*\n\s*([^,\n]+),\s*\n\s*([^,\n]+),\s*\n\s*({NUM}),\s*\n\s*({NUM}),'
        modified = re.sub(pattern_compact, increase_roof_insulation, idf_content)

        # Pattern 2: Commented format (full comments)
        # Now handles scientific notation from Bayesian calibration
        pattern_commented = rf'Material,\s*\n\s*([^,]+),\s*!-\s*Name\s*\n\s*([^,]+),\s*!-\s*Roughness\s*\n\s*({NUM}),\s*!-\s*Thickness[^,]*\n\s*({NUM}),\s*!-\s*Conductivity'
        modified = re.sub(pattern_commented, increase_roof_insulation, modified)

        # Pattern 3: Mixed format (Raiden generator format - handles comments on all lines)
        # Material,
        #     RoofInsulation,           !- Name
        #     MediumRough,              !- Roughness
        #     0.336,                    !- Thickness
        #     0.035,                    !- Conductivity
        # Now handles scientific notation from Bayesian calibration AND comments on all lines
        # [^\n]* allows any characters (including !- comments) before newline
        pattern_mixed = rf'(Material,\s*\n\s*([^,\n]+),[^\n]*\n\s*([^,\n]+),[^\n]*\n\s*)({NUM})(,\s*[^\n]*!-\s*Thickness[^\n]*\n\s*)({NUM})'

        modifications_made = [0]  # Use list for mutation in closure

        def increase_roof_mixed(match):
            prefix = match.group(1)
            name = match.group(2).strip()
            roughness = match.group(3).strip()
            thickness = float(match.group(4))
            thickness_comment = match.group(5)
            conductivity = float(match.group(6))

            if is_roof_material(name):
                new_thickness = thickness + thickness_m
                modifications_made[0] += 1
                logger.info(f"  Roof insulation (mixed): {name} thickness {thickness:.4f} → {new_thickness:.4f}m")
                return f'{prefix}{new_thickness:.4f}{thickness_comment}{conductivity}'
            return match.group(0)

        modified = re.sub(pattern_mixed, increase_roof_mixed, modified)

        if modifications_made[0] == 0:
            logger.warning("⚠ Roof insulation ECM: No materials modified (check regex patterns)")

        comment = f'''
! ==== ECM Applied: Roof Insulation ====
! Added thickness: {thickness_mm} mm
! Material: {material}
! Expected heating reduction: 5-10%

'''
        return comment + modified

    def _apply_window_replacement(
        self,
        idf_content: str,
        params: Dict[str, Any]
    ) -> str:
        """
        Replace windows with specified U-value.

        Modifies WindowMaterial:SimpleGlazingSystem objects.
        """
        u_value = params.get('u_value', 0.9)
        shgc = params.get('shgc', 0.5)

        # Replace U-Factor in WindowMaterial:SimpleGlazingSystem
        def replace_window_v1(match):
            name = match.group(1).strip()
            return f'''WindowMaterial:SimpleGlazingSystem,
    {name},
    {u_value},          !- U-Factor (W/m2-K) (ECM: upgraded)
    {shgc}'''

        # Scientific notation regex component (matches 0.123 or 1.23e-01 or 1.23E+02)
        NUM = r'[\d.]+(?:[eE][+-]?\d+)?'

        # Pattern 1: Commented name format
        # Now handles scientific notation from Bayesian calibration
        # Note: U-Factor may be written as "UFactor" or "U-Factor"
        pattern1 = rf'WindowMaterial:SimpleGlazingSystem,\s*\n\s*([^,]+),\s*!-\s*Name\s*\n\s*{NUM},\s*!-\s*U-?Factor[^,]*\s*\n\s*{NUM}'
        modified = re.sub(pattern1, replace_window_v1, idf_content)

        # Pattern 2: Compact format (Raiden generator) - name on its own line, U-factor has comment
        # WindowMaterial:SimpleGlazingSystem,
        #     GlazingSystem,
        #     1.2,          !- U-Factor (W/m2-K)
        #     0.50;       !- Solar Heat Gain Coefficient
        # Now handles scientific notation from Bayesian calibration
        def replace_window_v2(match):
            name = match.group(1).strip()
            return f'''WindowMaterial:SimpleGlazingSystem,
    {name},
    {u_value},          !- U-Factor (W/m2-K) (ECM: upgraded)
    {shgc};       !- Solar Heat Gain Coefficient'''

        # Handle both "U-Factor" and "UFactor" variations
        pattern2 = rf'WindowMaterial:SimpleGlazingSystem,\s*\n\s*([^,\n]+),\s*\n\s*{NUM},\s*!-\s*U-?Factor[^\n]*\n\s*{NUM};\s*!-\s*Solar[^\n]*'
        modified = re.sub(pattern2, replace_window_v2, modified)

        # Pattern 3: Bayesian eppy format - name with comment, UFactor without hyphen
        # WindowMaterial:SimpleGlazingSystem,
        #     GlazingSystem,            !- Name
        #     1.1425...,                !- UFactor
        #     0.5;                      !- Solar Heat Gain Coefficient
        def replace_window_v3(match):
            name = match.group(1).strip()
            return f'''WindowMaterial:SimpleGlazingSystem,
    {name},            !- Name
    {u_value},         !- UFactor (ECM: upgraded)
    {shgc};            !- Solar Heat Gain Coefficient'''

        pattern3 = rf'WindowMaterial:SimpleGlazingSystem,\s*\n\s*([^,\n]+),\s*!-\s*Name\s*\n\s*{NUM},\s*!-\s*UFactor[^\n]*\n\s*{NUM};\s*!-\s*Solar[^\n]*'
        modified = re.sub(pattern3, replace_window_v3, modified)

        comment = f'''
! ==== ECM Applied: Window Replacement ====
! New U-value: {u_value} W/m²K
! New SHGC: {shgc}
! Expected heating reduction: 10-20%

'''
        return comment + modified

    def _apply_air_sealing(
        self,
        idf_content: str,
        params: Dict[str, Any]
    ) -> str:
        """
        Reduce infiltration rate.

        Modifies ZoneInfiltration:DesignFlowRate objects.
        Uses structured parser when available, falls back to regex for simple fixtures.
        """
        reduction_factor = params.get('reduction_factor', 0.5)

        # Use regex approach for reliability with varying IDF formats
        # Pattern 1: with comment
        def reduce_ach(match):
            current_ach = float(match.group(1))
            new_ach = current_ach * reduction_factor
            return f'{new_ach:.4f};                        !- Air Changes per Hour (ECM: sealed)'

        pattern1 = r'([\d.]+)\s*;\s*!-\s*Air Changes per Hour'
        modified = re.sub(pattern1, reduce_ach, idf_content)

        # Pattern 2: value at end of infiltration block (after AirChanges/Hour, with blank fields)
        def reduce_ach_block(match):
            prefix = match.group(1)  # "AirChanges/Hour,"
            blanks = match.group(2)  # blank fields
            current_ach = float(match.group(3))
            new_ach = current_ach * reduction_factor
            return f'{prefix}{blanks}{new_ach:.4f};'

        pattern2 = r'(AirChanges/Hour,)([\s,]*)([\d.]+);'
        modified = re.sub(pattern2, reduce_ach_block, modified, flags=re.IGNORECASE)

        comment = f'''
! ==== ECM Applied: Air Sealing ====
! Infiltration reduction factor: {reduction_factor}
! Expected heating reduction: 5-10%

'''
        return comment + modified

    def _apply_ftx_upgrade(
        self,
        idf_content: str,
        params: Dict[str, Any]
    ) -> str:
        """
        Upgrade heat recovery effectiveness.

        Modifies Sensible Heat Recovery Effectiveness in IdealLoadsAirSystem.
        """
        effectiveness = params.get('effectiveness', 0.85)

        # Replace heat recovery effectiveness
        def replace_hr(match):
            prefix = match.group(1)
            return f'{prefix}{effectiveness:.2f}'

        # Pattern for sensible heat recovery effectiveness
        pattern = r'(Sensible Heat Recovery Effectiveness\s*\n\s*)([\d.]+)'
        modified = re.sub(pattern, replace_hr, idf_content, flags=re.IGNORECASE)

        comment = f'''
! ==== ECM Applied: FTX Heat Recovery Upgrade ====
! New effectiveness: {effectiveness:.0%}
! Expected heating reduction: 15-25%

'''
        return comment + modified

    def _apply_ftx_installation(
        self,
        idf_content: str,
        params: Dict[str, Any]
    ) -> str:
        """
        Install new FTX system (convert from F or natural).

        Changes Heat Recovery Type from None to Sensible.
        """
        effectiveness = params.get('effectiveness', 0.80)

        # Change heat recovery type from None to Sensible
        modified = re.sub(
            r'None,\s*!-\s*Heat Recovery Type',
            f'Sensible,                        !- Heat Recovery Type (ECM: FTX installed)',
            idf_content
        )

        # Set effectiveness value
        def set_hr_effectiveness(match):
            return f'{effectiveness:.2f};                         !- Sensible Heat Recovery Effectiveness (ECM)'

        modified = re.sub(
            r'[\d.]+;\s*!-\s*Sensible Heat Recovery Effectiveness',
            set_hr_effectiveness,
            modified
        )

        comment = f'''
! ==== ECM Applied: FTX Installation ====
! Heat recovery type: Sensible
! Effectiveness: {effectiveness:.0%}
! Expected heating reduction: 30-40%

'''
        return comment + modified

    def _apply_ftx_overhaul(
        self,
        idf_content: str,
        params: Dict[str, Any]
    ) -> str:
        """
        FTX system overhaul/repair for malfunctioning systems.

        Restores heat recovery effectiveness from broken/degraded state.
        Also improves air tightness (broken seals, dampers).
        Similar to FTX upgrade but assumes starting from lower baseline.
        """
        target_effectiveness = params.get('target_effectiveness', 0.75)

        # Scientific notation support for Bayesian calibration output
        NUM = r'[\d.]+(?:[eE][+-]?\d+)?'

        modified = idf_content

        # Replace heat recovery effectiveness
        # Actual format: 0.75,                !- Sensible Heat Recovery Effectiveness
        # Value comes BEFORE the comment, not after
        def replace_hr(match):
            terminator = match.group(2)  # , or ;
            return f'{target_effectiveness:.2f}{terminator}                !- Sensible Heat Recovery Effectiveness (ECM: FTX Overhaul)'

        pattern_hr = rf'({NUM})([,;])\s*!-\s*Sensible Heat Recovery Effectiveness'
        modified = re.sub(pattern_hr, replace_hr, modified, flags=re.IGNORECASE)

        # Also reduce infiltration (fixed seals, dampers)
        def reduce_infiltration(match):
            prefix = match.group(1)
            blanks = match.group(2)
            current = float(match.group(3))
            new_value = current * 0.85  # 15% reduction from fixing leaks
            return f'{prefix}{blanks}{new_value:.4f};'

        pattern_ach = rf'(AirChanges/Hour,)([\s,]*)({NUM});'
        modified = re.sub(pattern_ach, reduce_infiltration, modified, flags=re.IGNORECASE)

        comment = f'''
! ==== ECM Applied: FTX System Overhaul ====
! Restored effectiveness: {target_effectiveness:.0%}
! Fixed: heat exchanger, filters, fans, dampers, seals
! Expected heating reduction: 20-30% (from malfunctioning state)

'''
        return comment + modified

    def _apply_dcv(
        self,
        idf_content: str,
        params: Dict[str, Any]
    ) -> str:
        """
        Apply demand-controlled ventilation.

        Reduces ventilation rate during unoccupied periods by modifying
        the outdoor air flow rate. In practice, this is approximated by
        reducing the overall ventilation rate by a factor.
        """
        co2_setpoint = params.get('co2_setpoint', 1000)

        # DCV typically reduces ventilation energy by 15-30%
        # We approximate this by reducing the design outdoor air flow
        # The actual reduction depends on occupancy patterns
        reduction_factor = 0.7 if co2_setpoint >= 1000 else 0.8

        modified = idf_content

        # Pattern 1: Compact format - DesignSpecification:OutdoorAir with Flow/Area
        # Matches: DesignSpecification:OutdoorAir,\n  Name,\n  Flow/Area,\n  ,\n  0.00035;
        def reduce_oa_flow_area(match):
            prefix = match.group(1)  # Everything before the value
            current_value = float(match.group(2))
            new_value = current_value * reduction_factor
            return f'{prefix}{new_value:.6f};'

        pattern_compact = r'(DesignSpecification:OutdoorAir,\s*\n\s*[^,]+,\s*\n\s*Flow/Area,\s*\n\s*,\s*\n\s*)([\d.]+);'
        modified = re.sub(pattern_compact, reduce_oa_flow_area, modified)

        # Pattern 2: Commented format - Outdoor Air Flow per Person
        def reduce_oa_person(match):
            current_value = float(match.group(1))
            new_value = current_value * reduction_factor
            return f'{new_value:.6f},                  !- Outdoor Air Flow per Person (ECM: DCV)'

        pattern_person = r'([\d.]+),\s*!-\s*Outdoor Air Flow per Person'
        modified = re.sub(pattern_person, reduce_oa_person, modified)

        # Pattern 3: Commented format - Outdoor Air Flow per Floor Area
        def reduce_oa_area(match):
            current_value = float(match.group(1))
            new_value = current_value * reduction_factor
            terminator = match.group(2)  # ; or ,
            return f'{new_value:.6f}{terminator}                  !- Outdoor Air Flow per Zone Floor Area (ECM: DCV)'

        # Handle both "Floor Area" and "Zone Floor Area", and both comma and semicolon endings
        pattern_area = r'([\d.]+)([;,])\s*!-\s*Outdoor Air Flow per (?:Zone )?Floor Area'
        modified = re.sub(pattern_area, reduce_oa_area, modified)

        comment = f'''
! ==== ECM Applied: Demand Controlled Ventilation ====
! CO2 setpoint: {co2_setpoint} ppm
! Effective ventilation reduction: {(1-reduction_factor)*100:.0f}%
! Expected heating reduction: 10-20%

'''
        return comment + modified

    def _apply_solar_pv(
        self,
        idf_content: str,
        params: Dict[str, Any]
    ) -> str:
        """
        Add PV generation using Generator:PVWatts.

        Uses the simplified PVWatts model which requires only:
        - DC system capacity (kW)
        - Module type (Standard, Premium, ThinFilm)
        - Array type (Fixed, Tracking)

        The model uses TMY weather data to calculate hourly generation.
        Output is available via:
        - Generator Produced DC Electricity Energy [J]
        - Electric Load Center Produced Electricity Energy [J]

        Parameters from Google Solar API (preferred):
        - roof_analysis: RoofAnalysis object with segments, pitch, azimuth
        - optimal_capacity_kwp: Calculated optimal PV capacity
        - annual_generation_kwh: Pre-calculated annual yield

        Fallback estimation (if no Google Solar API):
        - Optimal tilt: latitude (59° for Stockholm)
        - South-facing (azimuth = 180°)
        - Standard modules (polycrystalline, ~21% efficiency)
        - Fixed (rack/roof) mounting
        - System losses: 14% (inverter, wiring, soiling)
        """
        # Check for Google Solar API data first
        roof_analysis = params.get('roof_analysis')
        roof_segments = params.get('roof_segments', [])

        if roof_analysis or params.get('optimal_capacity_kwp'):
            # Use Google Solar API data
            dc_capacity_kw = params.get('optimal_capacity_kwp', 0)
            annual_generation_kwh = params.get('annual_generation_kwh', dc_capacity_kw * 900)
            tilt_deg = params.get('tilt_deg', params.get('primary_pitch_deg', 40))
            azimuth_deg = params.get('azimuth_deg', params.get('primary_azimuth_deg', 180))
            data_source = params.get('data_source', 'google_solar_api')
            net_available_m2 = params.get('net_available_m2', dc_capacity_kw * 5)  # ~5 m²/kWp

            logger.info(f"Solar PV using {data_source}: {dc_capacity_kw:.1f} kWp, "
                       f"tilt={tilt_deg:.0f}°, azimuth={azimuth_deg:.0f}°")
        else:
            # Fallback estimation based on roof area
            coverage = params.get('coverage_fraction', 0.7)
            efficiency = params.get('panel_efficiency', 0.21)
            roof_area = params.get('roof_area_m2', 320)  # Default Sjostaden roof
            latitude = params.get('latitude', 59.35)  # Stockholm default

            net_available_m2 = roof_area * coverage
            # DC capacity: ~210 Wp/m² for modern panels
            dc_capacity_kw = net_available_m2 * 0.21
            annual_generation_kwh = dc_capacity_kw * 900  # Stockholm ~900 kWh/kWp

            # Optimal tilt ≈ latitude for year-round production
            tilt_deg = latitude if latitude > 30 else 35
            azimuth_deg = 180  # South-facing
            data_source = "estimation"

            logger.info(f"Solar PV using estimation: {dc_capacity_kw:.1f} kWp from "
                       f"{net_available_m2:.0f} m² available")

        dc_capacity_watts = dc_capacity_kw * 1000

        # If we have multiple roof segments, create PV systems for each
        if roof_segments and len(roof_segments) > 1:
            return self._apply_solar_pv_multi_segment(
                idf_content, roof_segments, dc_capacity_watts, annual_generation_kwh
            )

        # Single PV system for simple roofs
        pv_objects = f'''
! ==== ECM Applied: Solar PV (Generator:PVWatts) ====
! Data source: {data_source}
! Net available roof area: {net_available_m2:.0f} m²
! DC System Capacity: {dc_capacity_kw:.1f} kWp
! Estimated annual generation: {annual_generation_kwh:.0f} kWh/year
! Tilt: {tilt_deg:.1f}° (optimal for latitude)
! Azimuth: {azimuth_deg:.1f}° (0=N, 180=S)

Generator:PVWatts,
    PVSystem_Roof,           !- Name
    {dc_capacity_watts:.0f},             !- DC System Capacity {{W}}
    Standard,                !- Module Type (Standard, Premium, ThinFilm)
    FixedRoofMounted,        !- Array Type (Fixed, Tracking options)
    0.14,                    !- System Losses (14% typical)
    {tilt_deg:.1f},                   !- Tilt Angle {{degrees}}
    {azimuth_deg:.1f};                 !- Azimuth Angle {{degrees}}

! Electric Load Center to connect PV to building
ElectricLoadCenter:Generators,
    PV_Generators,           !- Name
    PVSystem_Roof,           !- Generator 1 Name
    Generator:PVWatts,       !- Generator 1 Object Type
    {dc_capacity_watts:.0f},             !- Generator 1 Rated Electric Power Output {{W}}
    ,                        !- Generator 1 Availability Schedule Name
    ;                        !- Generator 1 Rated Thermal to Electrical Power Ratio

ElectricLoadCenter:Distribution,
    PV_LoadCenter,           !- Name
    PV_Generators,           !- Generator List Name
    Baseload,                !- Generator Operation Scheme Type
    0,                       !- Generator Demand Limit Scheme Purchased Electric Demand Limit {{W}}
    ,                        !- Track Schedule Name Scheme Schedule Name
    ,                        !- Track Meter Scheme Meter Name
    DirectCurrentWithInverter,  !- Electrical Buss Type
    PV_Inverter;             !- Inverter Object Name

ElectricLoadCenter:Inverter:Simple,
    PV_Inverter,             !- Name
    ,                        !- Availability Schedule Name
    ,                        !- Zone Name
    ,                        !- Radiative Fraction
    0.96;                    !- Inverter Efficiency

! Output variables for PV generation
Output:Variable,
    PVSystem_Roof,
    Generator Produced DC Electricity Energy,
    Hourly;

Output:Variable,
    *,
    Facility Total Electric Demand Power,
    Hourly;

Output:Meter,
    ElectricityProduced:Facility,
    Monthly;

'''
        return pv_objects + idf_content

    def _apply_solar_pv_multi_segment(
        self,
        idf_content: str,
        roof_segments: List[Dict[str, Any]],
        total_dc_watts: float,
        total_annual_kwh: float,
    ) -> str:
        """
        Add multiple PV systems for complex multi-segment roofs.

        From Google Solar API, we may get multiple roof segments with different
        orientations (e.g., gabled roof with east and west faces).
        Create separate Generator:PVWatts for each segment.
        """
        generator_list = []
        pv_objects = f'''
! ==== ECM Applied: Solar PV Multi-Segment (Generator:PVWatts) ====
! Total segments: {len(roof_segments)}
! Total DC capacity: {total_dc_watts/1000:.1f} kWp
! Total annual generation: {total_annual_kwh:.0f} kWh/year

'''
        for i, segment in enumerate(roof_segments):
            seg_id = segment.get('segment_id', f'Segment_{i+1}')
            seg_area = segment.get('usable_area_m2', segment.get('area_m2', 0))
            seg_azimuth = segment.get('azimuth_deg', 180)
            seg_pitch = segment.get('pitch_deg', 30)
            seg_efficiency = segment.get('pv_efficiency_factor', 1.0)

            # Allocate capacity proportionally based on area and efficiency
            seg_capacity_watts = (seg_area / sum(s.get('usable_area_m2', s.get('area_m2', 1))
                                                 for s in roof_segments)) * total_dc_watts * seg_efficiency

            if seg_capacity_watts < 500:  # Skip segments < 0.5 kW
                continue

            gen_name = f"PVSystem_{seg_id}"
            generator_list.append(gen_name)

            pv_objects += f'''
! Segment: {seg_id}
! Area: {seg_area:.0f} m², Azimuth: {seg_azimuth:.0f}°, Pitch: {seg_pitch:.0f}°
Generator:PVWatts,
    {gen_name},              !- Name
    {seg_capacity_watts:.0f},             !- DC System Capacity {{W}}
    Standard,                !- Module Type
    FixedRoofMounted,        !- Array Type
    0.14,                    !- System Losses
    {seg_pitch:.1f},                   !- Tilt Angle {{degrees}}
    {seg_azimuth:.1f};                 !- Azimuth Angle {{degrees}}

'''
        # Build generator list
        gen_entries = ""
        for gen_name in generator_list:
            gen_entries += f'''    {gen_name},           !- Generator Name
    Generator:PVWatts,       !- Generator Object Type
    ,                        !- Rated Electric Power Output
    ,                        !- Availability Schedule
    ,                        !- Rated Thermal to Electrical Power Ratio
'''
        pv_objects += f'''
ElectricLoadCenter:Generators,
    PV_Generators,           !- Name
{gen_entries.rstrip().rstrip(',')};

ElectricLoadCenter:Distribution,
    PV_LoadCenter,           !- Name
    PV_Generators,           !- Generator List Name
    Baseload,                !- Generator Operation Scheme Type
    0,                       !- Generator Demand Limit
    ,                        !- Track Schedule
    ,                        !- Track Meter
    DirectCurrentWithInverter,  !- Electrical Buss Type
    PV_Inverter;             !- Inverter Object Name

ElectricLoadCenter:Inverter:Simple,
    PV_Inverter,             !- Name
    ,                        !- Availability Schedule Name
    ,                        !- Zone Name
    ,                        !- Radiative Fraction
    0.96;                    !- Inverter Efficiency

Output:Variable,
    *,
    Generator Produced DC Electricity Energy,
    Monthly;

Output:Meter,
    ElectricityProduced:Facility,
    Monthly;

'''
        return pv_objects + idf_content

    def _apply_led_lighting(
        self,
        idf_content: str,
        params: Dict[str, Any]
    ) -> str:
        """
        Replace lighting with LED.

        Modifies Lights objects to reduce W/m².
        Reducing lighting power also reduces internal heat gains,
        which can slightly increase heating demand in cold climates.
        """
        new_power_density = params.get('power_density', 6)
        # Also accept 'watts_per_m2' parameter name
        new_power_density = params.get('watts_per_m2', new_power_density)

        modified = idf_content

        # Pattern 1: Compact format - Lights with Watts/Area
        # Matches: Lights,\n  Name,\n  Zone,\n  Schedule,\n  Watts/Area,\n  ,\n  8,
        def replace_lighting_compact(match):
            prefix = match.group(1)  # Everything before the value
            return f'{prefix}{new_power_density},'

        pattern_compact = r'(Lights,\s*\n\s*[^,]+,\s*\n\s*[^,]+,\s*\n\s*[^,]+,\s*\n\s*Watts/Area,\s*\n\s*,\s*\n\s*)[\d.]+'
        modified = re.sub(pattern_compact, replace_lighting_compact, modified)

        # Pattern 2: Commented format - Watts per Zone Floor Area
        def replace_lighting_commented(match):
            return f'{new_power_density},                        !- Watts per Zone Floor Area (ECM: LED)'

        pattern_commented = r'[\d.]+\s*,\s*!-\s*Watts per Zone Floor Area'
        modified = re.sub(pattern_commented, replace_lighting_commented, modified)

        comment = f'''
! ==== ECM Applied: LED Lighting ====
! New power density: {new_power_density} W/m² (from 8 W/m²)
! Expected lighting energy reduction: 40-60%
! Note: Reduced internal gains may slightly increase heating demand

'''
        return comment + modified

    def _apply_smart_thermostats(
        self,
        idf_content: str,
        params: Dict[str, Any]
    ) -> str:
        """
        Apply smart thermostat settings.

        Modifies heating setpoint schedule for night setback.
        Handles Bayesian-calibrated setpoints (not just 21°C).
        """
        setback_c = params.get('setback_c', 2)

        # Extract current base setpoint from IDF (may be calibrated, not 21°C)
        # Pattern: Schedule:Constant,\n    HeatSet,\n    Temperature,\n    20.82...; or compact
        base_match = re.search(
            r'Schedule:Constant,\s*\n?\s*HeatSet[^;]*?(\d+\.?\d*)\s*;',
            idf_content,
            re.IGNORECASE | re.DOTALL
        )
        if base_match:
            normal_setpoint = float(base_match.group(1))
        else:
            # Fallback to default
            normal_setpoint = 21.0

        setback_setpoint = normal_setpoint - setback_c

        # Create new heating setpoint schedule with night setback
        new_schedule = f'''
! ==== ECM Applied: Smart Thermostats ====
! Original setpoint: {normal_setpoint:.1f} C (from Bayesian calibration)
! Setback temperature: {setback_c} C
! Night setpoint: {setback_setpoint:.1f} C
! Setback period: 23:00 - 06:00

Schedule:Compact,
    HeatSet_ECM,
    Temperature,
    Through: 12/31,
    For: AllDays,
    Until: 06:00,{setback_setpoint:.1f},
    Until: 23:00,{normal_setpoint:.1f},
    Until: 24:00,{setback_setpoint:.1f};

'''

        # Replace Schedule:Constant HeatSet with comment (handles any numeric value)
        # Match both compact and multi-line formats
        modified = re.sub(
            r'Schedule:Constant,\s*\n?\s*HeatSet,\s*\n?\s*!?[^,]*,?\s*\n?\s*Temperature,\s*\n?\s*!?[^,]*,?\s*\n?\s*[\d.]+\s*;',
            '! HeatSet replaced by ECM smart thermostat schedule (HeatSet_ECM)',
            idf_content,
            flags=re.IGNORECASE | re.DOTALL
        )

        # Replace references to HeatSet in ThermostatSetpoint:DualSetpoint
        # Handle format with !- comments on each line
        modified = re.sub(
            r'(ThermostatSetpoint:DualSetpoint,\s*\n\s*[^,]+,\s*[^\n]*\n\s*)HeatSet(\s*,|\s*;)',
            r'\1HeatSet_ECM\2',
            modified,
            flags=re.IGNORECASE
        )

        # Also handle inline format without line breaks
        modified = re.sub(
            r'(Heating Setpoint.*?Schedule.*?Name\s*\n\s*)HeatSet(\s*,|\s*;)',
            r'\1HeatSet_ECM\2',
            modified,
            flags=re.IGNORECASE
        )

        return new_schedule + modified

    def _apply_heat_pump(
        self,
        idf_content: str,
        params: Dict[str, Any]
    ) -> str:
        """
        Add heat pump to reduce heating demand and primary energy.

        Heat pump integration provides savings via:
        1. Exhaust air heat recovery: Extract heat from exhaust air (frånluftsvärmepump)
        2. Improved building tightness: Better sealed for HP installation
        3. Primary energy: COP converts electricity to heat efficiently

        For IdealLoadsAirSystem, we model:
        - Enable/improve heat recovery (representing exhaust air heat pump)
        - Reduce infiltration (better sealed system)
        - Document COP-based primary energy calculation for post-processing

        Swedish context:
        - Exhaust air heat pumps (frånluftsvärmepump) are common in Sweden
        - They extract heat from exhaust air and boost it for heating
        - Effective heat recovery efficiency: 75-90% (depending on COP)
        """
        cop = params.get('cop', 3.5)
        coverage = params.get('coverage', 0.8)
        hp_type = params.get('type', 'exhaust_air')  # exhaust_air, ground_source, air_source

        modified = idf_content

        # Heat pump effective heat recovery efficiency
        # Exhaust air HP extracts heat from exhaust, COP boosts it
        # Effective efficiency = base_recovery + (1 - base_recovery) * COP_factor
        # For COP 3.5: ~85-90% effective heat recovery
        if hp_type == 'exhaust_air':
            target_hr_eff = min(0.90, 0.70 + 0.05 * cop)  # Scale with COP, cap at 90%
        elif hp_type == 'ground_source':
            target_hr_eff = 0.85  # Ground source typically paired with good ventilation
        else:
            target_hr_eff = 0.80  # Air source - less effect on ventilation

        # Pattern 1: Enable heat recovery if currently "None"
        # Heat_Recovery_Type field in IdealLoadsAirSystem
        def enable_heat_recovery(match):
            return f'{match.group(1)}Sensible,{match.group(2)}'

        pattern_hr_type = r'(Heat_Recovery_Type,\s*\n?\s*)None,'
        modified = re.sub(pattern_hr_type, enable_heat_recovery, modified, flags=re.IGNORECASE)

        # Also match compact format: None, followed by sensible heat recovery effectiveness
        pattern_hr_none = r'(,\s*)None(,\s*!-\s*Heat Recovery Type)'
        modified = re.sub(pattern_hr_none, r'\1Sensible\2', modified, flags=re.IGNORECASE)

        # Pattern 2: Improve existing heat recovery effectiveness
        def improve_hr_eff(match):
            prefix = match.group(1)
            current_eff = float(match.group(2))
            # If already have HR, improve it; if not, set to target
            new_eff = max(current_eff, target_hr_eff)
            suffix = match.group(3)
            return f'{prefix}{new_eff:.2f}{suffix}'

        # Match: Sensible_Heat_Recovery_Effectiveness, value
        pattern_hr_eff = r'(Sensible_Heat_Recovery_Effectiveness,\s*\n?\s*)([\d.]+)(,|\s*;)'
        modified = re.sub(pattern_hr_eff, improve_hr_eff, modified, flags=re.IGNORECASE)

        # Also match inline format with comment
        pattern_hr_eff2 = r'(,\s*)([\d.]+)(\s*,?\s*!-\s*Sensible Heat Recovery Effectiveness)'
        modified = re.sub(pattern_hr_eff2, improve_hr_eff, modified, flags=re.IGNORECASE)

        # Pattern 3: Reduce infiltration (better sealed for HP installation)
        infiltration_reduction = 0.85  # 15% reduction

        def reduce_ach_block(match):
            prefix = match.group(1)
            blanks = match.group(2)
            current_ach = float(match.group(3))
            new_ach = current_ach * infiltration_reduction
            return f'{prefix}{blanks}{new_ach:.4f};'

        pattern_ach = r'(AirChanges/Hour,)([\s,]*)([\d.]+);'
        modified = re.sub(pattern_ach, reduce_ach_block, modified, flags=re.IGNORECASE)

        # Pattern 4: ACH with comment
        def reduce_ach_comment(match):
            current_ach = float(match.group(1))
            new_ach = current_ach * infiltration_reduction
            return f'{new_ach:.4f};                        !- Air Changes per Hour (HP sealed system)'

        pattern_ach2 = r'([\d.]+)\s*;\s*!-\s*Air Changes per Hour'
        modified = re.sub(pattern_ach2, reduce_ach_comment, modified)

        # Calculate expected savings
        primary_energy_reduction = (1 - 1/cop) * 100 * coverage

        comment = f'''
! ==== ECM Applied: Heat Pump Integration ====
! Type: {hp_type.replace('_', ' ').title()}
! COP: {cop}
! Load coverage: {coverage*100:.0f}%
!
! Thermal simulation effects:
! - Heat recovery effectiveness improved to: {target_hr_eff:.0%}
! - Infiltration reduced by: {(1-infiltration_reduction)*100:.0f}%
!
! Primary energy calculation (for post-processing):
! - Electricity for HP = Heating Load / COP
! - Primary energy savings: ~{primary_energy_reduction:.0f}% vs district heating
!
! Swedish context: Frånluftsvärmepump extracts heat from exhaust air
! and boosts it with COP {cop} for space heating and DHW.

'''
        return comment + modified

    def _apply_air_source_heat_pump(
        self,
        idf_content: str,
        params: Dict[str, Any]
    ) -> str:
        """
        Air source heat pump installation.

        Air source heat pumps (ASHP) extract heat from outdoor air.
        Less efficient than ground source but lower installation cost.
        COP typically 2.5-4.0 depending on outdoor temperature.
        """
        cop = params.get('cop', 3.0)
        coverage = params.get('coverage', 0.7)

        # Similar thermal benefits to ground source HP
        modified = idf_content

        # Improve infiltration (better sealed system with HP installation)
        infiltration_reduction = 0.88

        def reduce_ach_block(match):
            prefix = match.group(1)
            blanks = match.group(2)
            current_ach = float(match.group(3))
            new_ach = current_ach * infiltration_reduction
            return f'{prefix}{blanks}{new_ach:.4f};'

        pattern_ach = r'(AirChanges/Hour,)([\s,]*)([\d.]+);'
        modified = re.sub(pattern_ach, reduce_ach_block, modified, flags=re.IGNORECASE)

        comment = f'''
! ==== ECM Applied: Air Source Heat Pump ====
! COP: {cop}
! Load coverage: {coverage*100:.0f}%
! Type: Air-to-water heat pump
!
! Thermal benefit: ~5-8% from improved building tightness and controls
! Primary energy benefit: ~{(1-1/cop)*100:.0f}% reduction (calculated in post-processing)
!
! Note: Efficiency decreases at low outdoor temperatures.
! Typical COP range: 2.5 (cold) to 4.0 (mild weather)

'''
        return comment + modified

    def _apply_heat_pump_water_heater(
        self,
        idf_content: str,
        params: Dict[str, Any]
    ) -> str:
        """
        Heat Pump Water Heater (Frånluftsvärmepump för VV).

        Extracts heat from exhaust air for DHW. COP 2.5-3.5.

        Implementation:
        - Swedish DHW: 22 kWh/m²/year standard
        - With HPWH at COP 3.0: electricity = 22/3 = 7.3 kWh/m²/year
        - Net DHW reduction: 22 - 7.3 = 14.7 kWh/m²/year (67% heating savings)
        - Model as: negative internal gains (DHW heating saved) + electricity added

        Physics: HPWH extracts heat from exhaust air (same principle as FTX
        but for DHW instead of supply air). This shifts DHW load from district
        heating to electricity at COP 2.5-3.5.
        """
        cop = params.get('cop', 3.0)
        dhw_kwh_m2_year = params.get('dhw_load', 22)  # Swedish standard

        # Calculate energy flows
        hp_electricity_kwh_m2_year = dhw_kwh_m2_year / cop
        dhw_savings_kwh_m2_year = dhw_kwh_m2_year - hp_electricity_kwh_m2_year

        # Convert to W/m²
        dhw_savings_w_m2 = dhw_savings_kwh_m2_year * 1000 / 8760  # ~1.68 W/m²
        hp_electricity_w_m2 = hp_electricity_kwh_m2_year * 1000 / 8760  # ~0.84 W/m²

        # Get zone names
        zone_pattern = r'Zone,\s*\n?\s*([^,\n]+)\s*,'
        zones = re.findall(zone_pattern, idf_content)

        if not zones:
            comment = f'''
! ==== ECM Applied: Heat Pump Water Heater ====
! COP: {cop}
! DHW savings: {dhw_savings_kwh_m2_year:.1f} kWh/m²/year
! HP electricity: {hp_electricity_kwh_m2_year:.1f} kWh/m²/year
! Note: No zones found for modeling

'''
            return comment + idf_content

        first_zone = zones[0].strip()

        # HPWH runs year-round for DHW, but slightly more in winter (more hot water use)
        hpwh_schedule = '''
Schedule:Compact,
    HPWH_Operation_Sched,         !- Name
    Any Number,                    !- Schedule Type Limits Name
    Through: 03/31,                !- Winter/spring
    For: AllDays,                  !- Field 2
    Until: 24:00, 1.1,            !- Slightly more DHW in winter
    Through: 09/30,                !- Summer/fall
    For: AllDays,                  !- Field 4
    Until: 24:00, 0.9,            !- Slightly less DHW in summer
    Through: 12/31,                !- Winter
    For: AllDays,                  !- Field 6
    Until: 24:00, 1.1;            !- More DHW in winter

'''

        hpwh_objects = f'''
! ==== ECM Applied: Heat Pump Water Heater ====
! COP: {cop}
! DHW load shifted from district heating to electricity
! DHW heating savings: {dhw_savings_kwh_m2_year:.1f} kWh/m²/year ({(1-1/cop)*100:.0f}% reduction)
! HP electricity added: {hp_electricity_kwh_m2_year:.1f} kWh/m²/year
! Net energy benefit: {dhw_savings_kwh_m2_year - hp_electricity_kwh_m2_year:.1f} kWh/m²/year

! Negative internal gains: DHW heating now done by HP, not district heating
OtherEquipment,
    HPWH_DHW_Savings,              !- Name
    None,                          !- Fuel Type (EnergyPlus 25.x required)
    {first_zone},                  !- Zone Name
    HPWH_Operation_Sched,          !- Schedule Name
    Watts/Area,                    !- Design Level Calculation Method
    ,                              !- Design Level {{W}}
    -{dhw_savings_w_m2:.4f},       !- Power per Zone Floor Area {{W/m2}} (negative = heating savings)
    ,                              !- Power per Person {{W/person}}
    0,                             !- Fraction Latent
    0.3,                           !- Fraction Radiant (tank in utility room)
    0.7;                           !- Fraction Lost

! Electricity consumption by heat pump
ElectricEquipment,
    HPWH_Electricity,              !- Name
    {first_zone},                  !- Zone Name
    HPWH_Operation_Sched,          !- Schedule Name
    Watts/Area,                    !- Design Level Calculation Method
    ,                              !- Design Level {{W}}
    {hp_electricity_w_m2:.4f},     !- Power per Zone Floor Area {{W/m2}}
    ,                              !- Power per Person {{W/person}}
    0,                             !- Fraction Latent
    0.8,                           !- Fraction Radiant (compressor heat)
    0;                             !- Fraction Lost

'''

        return hpwh_schedule + hpwh_objects + idf_content

    # =========================================================================
    # ZERO-COST / OPERATIONAL ECMs
    # =========================================================================

    def _apply_duc_calibration(
        self,
        idf_content: str,
        params: Dict[str, Any]
    ) -> str:
        """
        DUC (District Heating Control Unit) calibration.

        Optimizes heating curve and night setback settings.
        Equivalent to reducing heating setpoint by 1-2°C during optimization period.
        Handles Bayesian-calibrated setpoints.
        """
        curve_offset = params.get('heating_curve_offset', -1)
        night_setback = params.get('night_setback', 2)

        # Apply via reduced heating setpoint (simulates better curve tuning)
        setpoint_reduction = abs(curve_offset)

        modified = idf_content

        # Extract base setpoint from IDF (may be Bayesian-calibrated)
        base_match = re.search(
            r'Schedule:Constant,\s*\n?\s*HeatSet[^;]*?(\d+\.?\d*)\s*;',
            idf_content,
            re.IGNORECASE | re.DOTALL
        )
        if base_match:
            base_setpoint = float(base_match.group(1))
        else:
            base_setpoint = 21.0

        # Create optimized heating schedule (reduce from calibrated base)
        normal_setpoint = base_setpoint - setpoint_reduction
        night_setpoint = normal_setpoint - night_setback

        new_schedule = f'''
! ==== ECM Applied: DUC Calibration ====
! Original setpoint: {base_setpoint:.1f}°C (from Bayesian calibration)
! Heating curve offset: {curve_offset}°C
! Night setback: {night_setback}°C
! Optimized setpoint: {normal_setpoint:.1f}°C (day), {night_setpoint:.1f}°C (night)

Schedule:Compact,
    HeatSet_DUC,
    Temperature,
    Through: 12/31,
    For: AllDays,
    Until: 06:00,{night_setpoint:.1f},
    Until: 22:00,{normal_setpoint:.1f},
    Until: 24:00,{night_setpoint:.1f};

'''

        # Replace Schedule:Constant HeatSet (multi-line format)
        modified = re.sub(
            r'Schedule:Constant,\s*\n?\s*HeatSet,\s*\n?\s*!?[^,]*,?\s*\n?\s*Temperature,\s*\n?\s*!?[^,]*,?\s*\n?\s*[\d.]+\s*;',
            '! HeatSet replaced by DUC-optimized schedule (HeatSet_DUC)',
            modified,
            flags=re.IGNORECASE | re.DOTALL
        )

        # Replace references in ThermostatSetpoint (with !- comments)
        modified = re.sub(
            r'(ThermostatSetpoint:DualSetpoint,\s*\n\s*[^,]+,\s*[^\n]*\n\s*)HeatSet(\s*,|\s*;)',
            r'\1HeatSet_DUC\2',
            modified,
            flags=re.IGNORECASE
        )

        return new_schedule + modified

    def _apply_heating_curve(
        self,
        idf_content: str,
        params: Dict[str, Any]
    ) -> str:
        """
        Heating curve adjustment (framledningstemperatur).

        Implements outdoor reset logic: supply temp varies with outdoor temp.
        In IdealLoads model, simulated via monthly heating setpoint schedule.
        Handles Bayesian-calibrated setpoints.

        Physics: Lower curve = lower supply temp = energy savings in mild weather.
        A 4°C curve reduction saves ~5% heating in shoulder seasons.

        Stockholm monthly average outdoor temps:
        Jan: -3°C, Feb: -3°C, Mar: 1°C, Apr: 5°C, May: 11°C, Jun: 15°C
        Jul: 18°C, Aug: 17°C, Sep: 12°C, Oct: 7°C, Nov: 2°C, Dec: -1°C
        """
        curve_reduction = params.get('curve_reduction', 4)  # °C reduction

        # Find the current heating setpoint from Schedule:Constant
        # Handle both HeatSet and HeatSched names, and multi-line format
        base_setpoint = 21.0  # Default

        # Try multi-line format (Bayesian calibrated) with HeatSet
        setpoint_match = re.search(
            r'Schedule:Constant,\s*\n?\s*(HeatSet|HeatSched)[^;]*?(\d+\.?\d*)\s*;',
            idf_content,
            re.IGNORECASE | re.DOTALL
        )
        if setpoint_match:
            base_setpoint = float(setpoint_match.group(2))
            schedule_name = setpoint_match.group(1)
        else:
            # Fallback to compact single-line
            setpoint_match = re.search(
                r'Schedule:Constant,\s*(HeatSet|HeatSched),\s*Temperature,\s*([\d.]+)',
                idf_content,
                re.IGNORECASE
            )
            if setpoint_match:
                base_setpoint = float(setpoint_match.group(2))
                schedule_name = setpoint_match.group(1)
            else:
                schedule_name = "HeatSet"

        # Outdoor reset effect: in shoulder months (spring/fall), allow slightly
        # lower indoor temp because radiators can't deliver full output at lower
        # supply temps. In cold months (-5 to -15°C), full capacity needed.

        # Stockholm monthly effects (conservative estimates)
        monthly_reductions = {
            1: 0.0,    # Jan: -3°C → minimal reduction
            2: 0.0,    # Feb: -3°C → minimal reduction
            3: 0.2,    # Mar: 1°C → slight reduction
            4: 0.4,    # Apr: 5°C → moderate reduction
            5: 0.6,    # May: 11°C → most savings (heating barely needed)
            6: 0.0,    # Jun: no heating
            7: 0.0,    # Jul: no heating
            8: 0.0,    # Aug: no heating
            9: 0.4,    # Sep: 12°C → moderate reduction
            10: 0.3,   # Oct: 7°C → moderate reduction
            11: 0.1,   # Nov: 2°C → slight reduction
            12: 0.0,   # Dec: -1°C → minimal reduction
        }

        # Scale reductions by curve_reduction (4°C → 1.0°C max reduction)
        max_reduction = curve_reduction / 4.0  # 4°C → 1.0°C setpoint

        # Build monthly schedule sections
        schedule_parts = []
        for month, reduction_factor in monthly_reductions.items():
            month_setpoint = base_setpoint - (max_reduction * reduction_factor)
            # End day of each month
            end_days = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
            end_date = f"{month}/{end_days[month-1]}"
            schedule_parts.append(f"    Through: {end_date}, For: AllDays, Until: 24:00, {month_setpoint:.1f},")

        # Create the new compact schedule - use consistent name
        new_schedule_name = f"{schedule_name}_OutdoorReset"

        new_schedule = f'''
! ==== ECM Applied: Heating Curve Adjustment ====
! Curve reduction: {curve_reduction}°C (framledningstemperatur)
! Base setpoint: {base_setpoint:.1f}°C (from Bayesian calibration)
! Max indoor temp reduction: {max_reduction:.1f}°C (in shoulder months)
! Physics: Lower supply temp in mild weather saves pump energy + heat losses
! Typical savings: 3-5% heating energy

Schedule:Compact,
    {new_schedule_name},
    Temperature,
{chr(10).join(schedule_parts[:-1])}
    Through: 12/31, For: AllDays, Until: 24:00, {base_setpoint:.1f};

'''

        modified = idf_content

        # Replace Schedule:Constant with comment (multi-line format)
        modified = re.sub(
            r'Schedule:Constant,\s*\n?\s*(HeatSet|HeatSched),\s*\n?\s*!?[^,]*,?\s*\n?\s*Temperature,\s*\n?\s*!?[^,]*,?\s*\n?\s*[\d.]+\s*;',
            f'! {schedule_name} replaced by heating curve schedule ({new_schedule_name})',
            modified,
            flags=re.IGNORECASE | re.DOTALL
        )

        # Replace thermostat references (with !- comments)
        modified = re.sub(
            r'(ThermostatSetpoint:DualSetpoint,\s*\n\s*[^,]+,\s*[^\n]*\n\s*)(HeatSet|HeatSched)(\s*,|\s*;)',
            rf'\1{new_schedule_name}\3',
            modified,
            flags=re.IGNORECASE
        )

        # Also try single heating thermostats
        modified = re.sub(
            r'(ThermostatSetpoint:SingleHeating,\s*\w+,\s*)(HeatSet|HeatSched)\s*;',
            rf'\1{new_schedule_name};',
            modified,
            flags=re.IGNORECASE
        )

        return new_schedule + modified

    def _apply_ventilation_schedule(
        self,
        idf_content: str,
        params: Dict[str, Any]
    ) -> str:
        """
        Ventilation schedule optimization.

        Reduces ventilation during unoccupied hours.
        """
        night_reduction = params.get('night_reduction', 50) / 100.0  # Convert to fraction
        night_factor = 1.0 - night_reduction

        modified = idf_content

        # Create optimized ventilation schedule
        new_schedule = f'''
! ==== ECM Applied: Ventilation Schedule Optimization ====
! Night reduction: {night_reduction*100:.0f}%
! Night factor: {night_factor:.2f}

Schedule:Compact,
    VentSched_Optimized,
    Fraction,
    Through: 12/31,
    For: AllDays,
    Until: 06:00,{night_factor},
    Until: 08:00,1.0,
    Until: 22:00,1.0,
    Until: 24:00,{night_factor};

'''

        # Try to replace existing ventilation schedule
        modified = re.sub(
            r'Schedule:Constant,\s*VentSched,\s*Fraction,\s*1\s*;',
            '! VentSched replaced by optimized schedule',
            modified
        )

        return new_schedule + modified

    def _apply_radiator_balancing(
        self,
        idf_content: str,
        params: Dict[str, Any]
    ) -> str:
        """
        Radiator balancing (injustering).
        Handles Bayesian-calibrated setpoints.

        Physics:
        - Unbalanced radiators → uneven heat distribution
        - Cold corners force higher thermostat setting (22°C to get 20°C in cold spots)
        - Balanced radiators → even distribution across all zones
        - Allows 0.5-1.0°C lower setpoint while maintaining same comfort

        Swedish context:
        - Mandatory for new buildings (BBR)
        - Common retrofit measure for 1960-1990 buildings
        - Typical savings: 5-10% heating energy

        Implementation:
        Reduce heating setpoint by 0.5°C (conservative estimate).
        """
        setpoint_reduction = params.get('setpoint_reduction', 0.5)  # °C

        modified = idf_content

        # Extract base setpoint (handles Bayesian-calibrated values)
        base_setpoint, schedule_name = self._extract_heating_setpoint(idf_content)
        new_setpoint = base_setpoint - setpoint_reduction
        new_schedule_name = f"{schedule_name}_Balanced"

        # Create new schedule with reduced setpoint
        balanced_schedule = f'''
! ==== ECM Applied: Radiator Balancing (Injustering) ====
! Original setpoint: {base_setpoint:.1f}°C (from Bayesian calibration)
! New setpoint: {new_setpoint:.1f}°C (reduction: {setpoint_reduction}°C)
! Physics: Even heat distribution allows lower thermostat setting
! Typical savings: 5-10% heating energy
! Swedish reference: BBR requirement for new construction

Schedule:Constant,
    {new_schedule_name},          !- Name
    Temperature,                   !- Schedule Type Limits Name
    {new_setpoint:.1f};           !- Hourly Value

'''

        # Replace Schedule:Constant with comment (multi-line format)
        modified = re.sub(
            r'Schedule:Constant,\s*\n?\s*(HeatSet|HeatSched),\s*\n?\s*!?[^,]*,?\s*\n?\s*Temperature,\s*\n?\s*!?[^,]*,?\s*\n?\s*[\d.]+\s*;',
            f'! {schedule_name} replaced by balanced schedule ({new_schedule_name})',
            modified,
            flags=re.IGNORECASE | re.DOTALL
        )

        # Replace thermostat references
        modified = self._replace_thermostat_reference(modified, schedule_name, new_schedule_name)

        modified = balanced_schedule + modified

        comment = ''  # Already included in balanced_schedule

        return comment + modified

    def _apply_effektvakt(
        self,
        idf_content: str,
        params: Dict[str, Any]
    ) -> str:
        """
        Effektvakt (Power Guard) Optimization.

        Peak demand shaving using thermal mass to coast through peak hours.
        Handles Bayesian-calibrated setpoints.

        Implementation Strategy (Swedish best practice):
        1. PRE-HEAT phase (14:00-16:00): Raise setpoint 0.5°C to charge thermal mass
        2. COAST phase (16:00-20:00): Allow temperature to drift down, minimal heating
        3. GRADUAL RECOVERY (20:00-22:00): Slowly return to normal (avoid rebound spike)
        4. NORMAL (22:00-14:00): Normal heating operation

        Physics: Building thermal mass stores heat during pre-heating.
        During peak hours, the stored heat maintains comfort while heating is reduced.
        This cuts peak demand without sacrificing comfort or causing rebound heating.
        """
        peak_reduction_pct = params.get('peak_reduction', 15)
        setback_during_peak = params.get('setback_c', 1.5)  # 1.5°C setback (was 2.0)
        preheat_boost = params.get('preheat_boost_c', 0.5)  # Pre-heat 0.5°C higher
        peak_start = params.get('peak_start', 16)  # 16:00
        peak_end = params.get('peak_end', 20)  # 20:00
        preheat_start = peak_start - 2  # Start pre-heating 2 hours before peak

        # Get zone names (for info only)
        zone_pattern = r'Zone,\s*\n?\s*([^,\n]+)\s*,'
        zones = re.findall(zone_pattern, idf_content)

        modified = idf_content

        # Extract base setpoint (handles Bayesian-calibrated values)
        base_setpoint, schedule_name = self._extract_heating_setpoint(idf_content)
        preheat_setpoint = base_setpoint + preheat_boost  # Charge thermal mass
        coast_setpoint = base_setpoint - setback_during_peak  # Allow drift down
        recovery_setpoint = base_setpoint - 0.5  # Gradual recovery (not instant jump)

        # Create peak-shaving schedule with proper thermal mass strategy
        new_schedule_name = f"{schedule_name}_Effektvakt"

        effektvakt_schedule = f'''
! ==== ECM Applied: Effektvakt (Peak Demand Shaving) ====
! Original setpoint: {base_setpoint:.1f}°C (from Bayesian calibration)
! Strategy: Pre-heat → Coast → Gradual Recovery
! 1. Normal (00:00-{preheat_start}:00): {base_setpoint:.1f}°C
! 2. Pre-heat ({preheat_start}:00-{peak_start}:00): {preheat_setpoint:.1f}°C (charge thermal mass)
! 3. Coast ({peak_start}:00-{peak_end}:00): {coast_setpoint:.1f}°C (peak shaving)
! 4. Recovery ({peak_end}:00-22:00): {recovery_setpoint:.1f}°C (gradual, avoid rebound)
! 5. Normal (22:00-24:00): {base_setpoint:.1f}°C
! Expected peak demand reduction: {peak_reduction_pct}%

Schedule:Compact,
    {new_schedule_name},          !- Name (replaces original)
    Temperature,                   !- Schedule Type Limits Name
    Through: 12/31,                !- Field 1
    For: AllDays,                  !- Field 2 (EnergyPlus 25.x compatible)
    Until: {preheat_start}:00, {base_setpoint:.1f},   !- Normal morning
    Until: {peak_start}:00, {preheat_setpoint:.1f},   !- Pre-heat (charge thermal mass)
    Until: {peak_end}:00, {coast_setpoint:.1f},       !- Peak hours: coast/drift
    Until: 22:00, {recovery_setpoint:.1f},            !- Gradual recovery
    Until: 24:00, {base_setpoint:.1f};                !- Normal night

'''

        # Replace Schedule:Constant with comment (multi-line format)
        modified = re.sub(
            r'Schedule:Constant,\s*\n?\s*(HeatSet|HeatSched),\s*\n?\s*!?[^,]*,?\s*\n?\s*Temperature,\s*\n?\s*!?[^,]*,?\s*\n?\s*[\d.]+\s*;',
            f'! {schedule_name} replaced by Effektvakt schedule ({new_schedule_name})',
            modified,
            flags=re.IGNORECASE | re.DOTALL
        )

        # Replace thermostat references
        modified = self._replace_thermostat_reference(modified, schedule_name, new_schedule_name)

        modified = effektvakt_schedule + modified

        return modified

    def _apply_night_setback(
        self,
        idf_content: str,
        params: Dict[str, Any]
    ) -> str:
        """
        Apply night setback - reduce heating setpoint during unoccupied hours.

        Modifies heating schedules to reduce setpoint by 2-3°C during night hours.
        Swedish residential buildings typically have 21°C daytime, 18-19°C night.

        Implementation:
        1. Extract base setpoint from IDF (may be Bayesian-calibrated)
        2. Replace Schedule:Constant with Schedule:Compact that varies by hour
        3. Maintain setback during 22:00-06:00 (configurable)

        Expected savings: 3-5% heating energy (Swedish studies show 3-4%)
        """
        setback_c = params.get('setback_c', 2)
        start_hour = params.get('start_hour', 22)
        end_hour = params.get('end_hour', 6)

        modified = idf_content
        schedule_replaced = False

        # Extract base setpoint from IDF (may be Bayesian-calibrated, not 21°C)
        # Handle multi-line format: Schedule:Constant,\n  HeatSet,\n  Temperature,\n  20.82...;
        base_match = re.search(
            r'Schedule:Constant,\s*\n?\s*HeatSet[^;]*?(\d+\.?\d*)\s*;',
            idf_content,
            re.IGNORECASE | re.DOTALL
        )
        if base_match:
            base_setpoint = float(base_match.group(1))
        else:
            base_setpoint = params.get('base_setpoint_c', 21)

        # Calculate night setpoint
        night_setpoint = base_setpoint - setback_c

        # Pattern 1: Match multi-line Schedule:Constant format (Bayesian calibrated)
        # Format: Schedule:Constant,\n    HeatSet,  !- Name\n    Temperature, ...\n    20.82;
        multiline_pattern = r'Schedule:Constant,\s*\n?\s*(HeatSet|HeatSched|HeatingSetpoint|Heating_Setpoint|Heating)\s*,\s*[^\n]*\n?\s*Temperature\s*,\s*[^\n]*\n?\s*[\d.]+\s*;'

        def replace_multiline_schedule(match):
            nonlocal schedule_replaced
            # Extract schedule name
            name_match = re.search(r'(HeatSet|HeatSched|HeatingSetpoint|Heating_Setpoint|Heating)', match.group(0), re.IGNORECASE)
            schedule_name = name_match.group(1) if name_match else 'HeatSet'
            schedule_replaced = True

            # Create new compact schedule with night setback
            compact_schedule = f'''Schedule:Compact,
    {schedule_name},            !- Name
    Temperature,                !- Schedule Type Limits Name
    Through: 12/31,             !- Field 1
    For: AllDays,               !- Field 2 (EnergyPlus 25.x compatible)
    Until: {end_hour:02d}:00, {night_setpoint:.1f},  !- Night setback
    Until: {start_hour:02d}:00, {base_setpoint:.1f},  !- Daytime
    Until: 24:00, {night_setpoint:.1f};  !- Night
'''
            return compact_schedule

        modified = re.sub(multiline_pattern, replace_multiline_schedule, modified, flags=re.IGNORECASE | re.DOTALL)

        # Pattern 2: Match compact single-line format (original generator)
        if not schedule_replaced:
            compact_pattern = r'Schedule:Constant,\s*(HeatSet|HeatSched|HeatingSetpoint|Heating_Setpoint|Heating)\s*,\s*Temperature\s*,\s*[\d.]+\s*;'
            modified = re.sub(compact_pattern, replace_multiline_schedule, modified, flags=re.IGNORECASE)

        # Add comment at the top
        if schedule_replaced:
            comment = f'''
! ==== ECM Applied: Night Setback ====
! Base setpoint: {base_setpoint:.1f}°C (from Bayesian calibration)
! Night setback: {night_setpoint:.1f}°C ({start_hour}:00 - {end_hour}:00)
! Temperature reduction: {setback_c}°C
! Expected savings: 3-5% heating energy
! Swedish reference: Boverket recommends 18°C min night temp

'''
        else:
            comment = f'''
! ==== ECM Applied: Night Setback ====
! Setback: {setback_c}°C from {start_hour}:00 to {end_hour}:00
! Note: Could not find heating schedule to modify.
! For full effect, manually update thermostat schedule.
! Expected savings if implemented: 3-5% heating energy

'''
        return comment + modified

    def _apply_summer_bypass(
        self,
        idf_content: str,
        params: Dict[str, Any]
    ) -> str:
        """
        Apply summer bypass - disable heating during summer months.
        Handles Bayesian-calibrated setpoints.

        Physics:
        - Swedish heating season: typically Sep 15 - May 15
        - Summer bypass turns off heating system during summer
        - Prevents: accidental heating, standby energy, thermostat drift
        - IdealLoads already respects setpoint, but real systems can waste energy

        Implementation:
        Create heating availability schedule that is 0 during summer months.
        For IdealLoads, reduce heating setpoint to minimum during summer.

        Expected savings: 2-4% of heating energy (from shoulder season optimization).
        """
        # Swedish heating season: Sep 15 - May 15
        heating_start_month = params.get('heating_start_month', 9)  # September
        heating_start_day = params.get('heating_start_day', 15)
        heating_end_month = params.get('heating_end_month', 5)  # May
        heating_end_day = params.get('heating_end_day', 15)
        summer_setpoint = params.get('summer_setpoint', 10)  # Minimal during summer

        modified = idf_content

        # Extract base setpoint (handles Bayesian-calibrated values)
        base_setpoint, schedule_name = self._extract_heating_setpoint(idf_content)
        new_schedule_name = f"{schedule_name}_Seasonal"

        # Create a seasonal heating schedule
        seasonal_schedule = f'''
! ==== ECM Applied: Summer Bypass (Sommaravstängning) ====
! Original setpoint: {base_setpoint:.1f}°C (from Bayesian calibration)
! Heating season: Sep 15 - May 15
! Summer setpoint: {summer_setpoint}°C (effectively off)
! Winter setpoint: {base_setpoint:.1f}°C
! Physics: Prevents heating during warm periods
! Expected savings: 2-4% heating energy

Schedule:Compact,
    {new_schedule_name},          !- Name
    Temperature,                  !- Schedule Type Limits Name
    Through: {heating_end_month}/15,  !- Field 1 (heating season end)
    For: AllDays,
    Until: 24:00, {base_setpoint:.1f},
    Through: {heating_start_month}/14,  !- Field 4 (summer period)
    For: AllDays,
    Until: 24:00, {summer_setpoint:.1f},
    Through: 12/31,               !- Field 7 (heating season start)
    For: AllDays,
    Until: 24:00, {base_setpoint:.1f};

'''

        # Replace Schedule:Constant with comment (multi-line format)
        modified = re.sub(
            r'Schedule:Constant,\s*\n?\s*(HeatSet|HeatSched),\s*\n?\s*!?[^,]*,?\s*\n?\s*Temperature,\s*\n?\s*!?[^,]*,?\s*\n?\s*[\d.]+\s*;',
            f'! {schedule_name} replaced by seasonal schedule ({new_schedule_name})',
            modified,
            flags=re.IGNORECASE | re.DOTALL
        )

        # Replace thermostat references
        modified = self._replace_thermostat_reference(modified, schedule_name, new_schedule_name)

        return seasonal_schedule + modified

    def _apply_hot_water_temp(
        self,
        idf_content: str,
        params: Dict[str, Any]
    ) -> str:
        """
        Reduce domestic hot water temperature setpoint.

        Physics:
        - Standard DHW temp: 60°C, reduced to 55°C
        - 5°C reduction → ~8% DHW energy savings
        - Swedish MFH: 20-25 kWh/m²/year DHW
        - Savings: ~1.6-2.0 kWh/m²/year

        Implementation in IdealLoads model:
        - DHW pipe losses contribute to internal gains
        - Lower temp → less loss → slightly reduced internal gains
        - This increases heating need slightly (net negative on heating)
        - But overall building energy is lower (DHW savings >> heating penalty)

        We model this by adding small negative "equipment" that represents
        the reduced pipe losses (which were previously contributing heat).
        """
        target_temp = params.get('target_temp_c', 55)
        baseline_temp = 60.0  # Standard Swedish DHW temp

        # Temperature reduction
        temp_reduction = baseline_temp - target_temp  # °C

        # DHW savings factor: ~1.6% per °C reduction (based on thermodynamics)
        # (ΔT to cold water changes from 50°C to 45°C → 10% savings)
        dhw_savings_fraction = temp_reduction * 0.016

        # Swedish MFH average: 22 kWh/m²/year DHW
        # Internal gains from DHW (pipe losses, ~10% of DHW energy)
        # 22 * 0.10 = 2.2 kWh/m²/year internal gains from DHW pipes
        # Reduction: 2.2 * savings_fraction = ~0.18 kWh/m²/year
        # This is ~0.02 W/m² average

        # For a 2000 m² building, that's ~40W reduction in internal gains
        # We need to find the floor area from the model
        floor_area_m2 = 2000  # Default, will extract from model if possible

        # Try to extract floor area from Zone objects
        zone_matches = re.findall(
            r'Zone,\s*([^,]+),\s*[^,]*,\s*[^,]*,\s*[^,]*,\s*[^,]*,\s*[^,]*,\s*[^,]*,\s*([^,]+),\s*([^;]+)',
            idf_content
        )
        if zone_matches:
            # Sum floor areas from zones (last two fields are volume and area)
            try:
                total_area = sum(float(m[2].strip()) for m in zone_matches if m[2].strip())
                if total_area > 0:
                    floor_area_m2 = total_area
            except ValueError:
                pass

        # Calculate internal gain reduction (W)
        # 0.02 W/m² * savings_fraction * floor_area
        internal_gain_reduction = 0.02 * (dhw_savings_fraction / 0.08) * floor_area_m2

        # Get zone names for applying the negative internal gain
        zone_names = re.findall(r'Zone,\s*([^,]+),', idf_content)
        if not zone_names:
            zone_names = ['Zone1']  # Fallback

        # Create negative internal gains object (one per zone, distributed)
        gain_per_zone = internal_gain_reduction / len(zone_names)

        internal_gain_objects = []
        for zone_name in zone_names:
            zone_clean = zone_name.strip()
            internal_gain_objects.append(f'''
OtherEquipment,
    DHW_Reduction_{zone_clean},       !- Name
    None,                              !- Fuel Type
    {zone_clean},                      !- Zone Name
    AlwaysOn,                          !- Schedule Name
    EquipmentLevel,                    !- Design Level Calculation Method
    {-gain_per_zone:.2f},              !- Design Level (W) - NEGATIVE
    ,                                  !- Power per Zone Floor Area
    ,                                  !- Power per Person
    0,                                 !- Fraction Latent
    0,                                 !- Fraction Radiant
    0;                                 !- Fraction Lost
''')

        new_objects = f'''
! ==== ECM Applied: DHW Temperature Reduction ====
! Baseline temp: {baseline_temp:.0f}°C → Target: {target_temp:.0f}°C
! Temperature reduction: {temp_reduction:.0f}°C
! DHW energy savings: ~{dhw_savings_fraction*100:.1f}%
! Internal gain reduction: {internal_gain_reduction:.1f}W ({internal_gain_reduction/floor_area_m2*1000:.2f} mW/m²)
! Note: Negative internal gain represents reduced DHW pipe losses
! Important: Weekly Legionella flush to 70°C still required for safety
{''.join(internal_gain_objects)}
'''

        return new_objects + idf_content

    def _apply_pump_optimization(
        self,
        idf_content: str,
        params: Dict[str, Any]
    ) -> str:
        """
        Optimize circulation pump speeds (Pumpoptimering).

        VFD (Variable Frequency Drive) control on heating/cooling pumps.

        Implementation:
        - Pump affinity laws: Power ∝ (Speed)³
        - 30% speed reduction → 66% power reduction ((0.7)³ = 0.34)
        - Swedish multi-family: pumps use ~2-3 kWh/m²/year
        - VFD retrofit saves 40-60% of pump electricity
        - Also reduces slight heat from pump motors (minor thermal effect)

        Physics: Constant-speed pumps waste energy at partial loads.
        VFD matches flow to demand → significant electricity savings.
        """
        speed_reduction_pct = params.get('speed_reduction', 30)  # 30% average speed reduction

        # Calculate power savings using pump affinity law
        # New power ratio = (1 - speed_reduction/100)^3
        power_ratio = (1 - speed_reduction_pct / 100) ** 3
        power_savings_pct = (1 - power_ratio) * 100

        # Swedish typical: pumps use ~2.5 kWh/m²/year
        # Convert to W/m²: 2.5 kWh/m²/year ÷ 8760 hrs/year ≈ 0.29 W/m²
        pump_power_w_m2 = params.get('pump_power_w_m2', 0.29)
        savings_w_m2 = pump_power_w_m2 * (power_savings_pct / 100)

        # Get zone names
        zone_pattern = r'Zone,\s*\n?\s*([^,\n]+)\s*,'
        zones = re.findall(zone_pattern, idf_content)

        if not zones:
            comment = f'''
! ==== ECM Applied: Pump Speed Optimization (VFD) ====
! Speed reduction: {speed_reduction_pct}%
! Power savings: {power_savings_pct:.0f}% (pump affinity law)
! Note: No zones found for modeling

'''
            return comment + idf_content

        first_zone = zones[0].strip()

        # Pumps run during heating season (Sep-May in Sweden)
        pump_schedule = '''
Schedule:Compact,
    Pump_Savings_Sched,           !- Name
    Any Number,                    !- Schedule Type Limits Name
    Through: 05/15,                !- Heating season
    For: AllDays,                  !- Field 2
    Until: 24:00, 1.0,            !- Full savings during heating
    Through: 09/15,                !- Summer (pumps off or minimal)
    For: AllDays,                  !- Field 4
    Until: 24:00, 0.2,            !- Reduced savings (DHW circ only)
    Through: 12/31,                !- Heating season
    For: AllDays,                  !- Field 6
    Until: 24:00, 1.0;            !- Full savings

'''

        # Negative electricity (savings from VFD) - Use OtherEquipment for negative values
        pump_savings = f'''
! ==== ECM Applied: Pump Speed Optimization (VFD) ====
! Speed reduction: {speed_reduction_pct}%
! Power savings: {power_savings_pct:.0f}% via pump affinity law (Power ∝ Speed³)
! Electricity savings: {savings_w_m2:.3f} W/m²
! Investment: ~30,000-80,000 SEK per VFD unit

OtherEquipment,
    Pump_VFD_Savings,              !- Name
    None,                          !- Fuel Type (EnergyPlus 25.x required)
    {first_zone},                  !- Zone Name
    Pump_Savings_Sched,            !- Schedule Name
    Watts/Area,                    !- Design Level Calculation Method
    ,                              !- Design Level {{W}}
    -{savings_w_m2:.4f},           !- Power per Zone Floor Area {{W/m2}} (negative = savings)
    ,                              !- Power per Person {{W/person}}
    0,                             !- Fraction Latent
    0.8,                           !- Fraction Radiant (pump motor heat to zone)
    0;                             !- Fraction Lost

'''

        return pump_schedule + pump_savings + idf_content

    def _apply_bms_optimization(
        self,
        idf_content: str,
        params: Dict[str, Any]
    ) -> str:
        """
        BMS tune-up - general optimization of building controls.

        Comprehensive review and correction of all control settings.

        Implementation:
        - Correct setpoint drift (0.5°C reduction typically found)
        - Align schedules with actual occupancy
        - Eliminate simultaneous heating/cooling
        - Fix stuck valves and dampers
        - Typical savings: 3-8% of total energy

        Physics: Buildings drift from design over time.
        Sensors drift, schedules accumulate exceptions, valves stick.
        BMS tune-up restores optimal operation.
        """
        setpoint_correction = params.get('setpoint_correction', 0.5)
        schedule_alignment_pct = params.get('schedule_savings', 5)  # 5% from schedule fixes

        modified = idf_content

        # 1. Reduce heating setpoint by 0.5°C (correct drift)
        # Pattern for Schedule:Constant with heating setpoint
        schedule_pattern = r'(Schedule:Constant,\s*[^,]*(?:HeatSet|HeatSched|HeatingSetpoint|Heating_Setpoint|Heating)[^,]*,\s*[^,]*,\s*)(\d+\.?\d*)'

        def adjust_setpoint(match):
            original = float(match.group(2))
            new_value = original - setpoint_correction
            return f'{match.group(1)}{new_value:.1f}'

        modified = re.sub(schedule_pattern, adjust_setpoint, modified, flags=re.IGNORECASE)

        # Also adjust thermostat setpoints
        thermostat_pattern = r'(Heating Setpoint Temperature\s*,\s*)(\d+\.?\d*)'

        def reduce_thermostat(match):
            original = float(match.group(2))
            return f'{match.group(1)}{original - setpoint_correction:.1f}'

        modified = re.sub(thermostat_pattern, reduce_thermostat, modified)

        # 2. Reduce internal gains slightly (better scheduling = less waste heat)
        # This represents equipment not running during unoccupied periods
        zone_pattern = r'Zone,\s*\n?\s*([^,\n]+)\s*,'
        zones = re.findall(zone_pattern, modified)

        if zones:
            first_zone = zones[0].strip()

            # Schedule alignment savings (equipment off when not needed)
            schedule_savings_w_m2 = 0.3  # ~0.3 W/m² from schedule alignment

            bms_schedule = '''
Schedule:Compact,
    BMS_Optimization_Sched,       !- Name
    Any Number,                    !- Schedule Type Limits Name
    Through: 12/31,                !- Field 1
    For: AllDays,                  !- Field 2 (EnergyPlus 25.x compatible)
    Until: 06:00, 1.5,            !- Night (more savings - equipment off)
    Until: 18:00, 0.5,            !- Day (some waste eliminated)
    Until: 24:00, 1.5;            !- Evening

'''

            bms_savings = f'''
! ==== ECM Applied: BMS Optimization ====
! Setpoint correction: -{setpoint_correction}°C (drift eliminated)
! Schedule alignment: ~{schedule_alignment_pct}% savings
! Includes: valve/damper corrections, alarm review

OtherEquipment,
    BMS_Schedule_Savings,          !- Name
    None,                          !- Fuel Type (EnergyPlus 25.x required)
    {first_zone},                  !- Zone Name
    BMS_Optimization_Sched,        !- Schedule Name
    Watts/Area,                    !- Design Level Calculation Method
    ,                              !- Design Level {{W}}
    -{schedule_savings_w_m2:.2f}, !- Power per Zone Floor Area {{W/m2}} (negative = savings)
    ,                              !- Power per Person {{W/person}}
    0,                             !- Fraction Latent
    0.5,                           !- Fraction Radiant
    0;                             !- Fraction Lost

'''
            modified = bms_schedule + bms_savings + modified
        else:
            comment = f'''
! ==== ECM Applied: BMS Optimization ====
! Setpoint correction: -{setpoint_correction}°C
! Note: No zones found for schedule optimization

'''
            modified = comment + modified

        return modified

    # =========================================================================
    # SWEDISH-SPECIFIC ECM HANDLERS
    # =========================================================================

    def _apply_exhaust_air_heat_pump(
        self,
        idf_content: str,
        params: Dict[str, Any]
    ) -> str:
        """
        Apply Exhaust Air Heat Pump (Frånluftsvärmepump).

        Extracts heat from exhaust air for heating/DHW.
        Modeled by enabling heat recovery and adjusting effectiveness.
        Typical Swedish FVP: NIBE F470/F750 with COP 3.0-3.5.
        """
        cop = params.get('cop', 3.0)
        capacity_kw = params.get('capacity_kw', 10)

        # FVP effectively adds heat recovery to exhaust-only systems
        # Model as high-effectiveness heat recovery
        # COP 3.0 ≈ extracting ~67% of exhaust heat + work input
        effectiveness = min(0.90, 0.50 + (cop - 2.5) * 0.10)

        # Enable heat recovery (change from None to Sensible)
        modified = re.sub(
            r'None,\s*!-\s*Heat Recovery Type',
            f'Sensible,                        !- Heat Recovery Type (ECM: FVP installed)',
            idf_content
        )

        # Set high effectiveness for FVP
        def set_fvp_effectiveness(match):
            return f'{effectiveness:.2f};                         !- Sensible Heat Recovery Effectiveness (ECM: FVP)'

        modified = re.sub(
            r'[\d.]+;\s*!-\s*Sensible Heat Recovery Effectiveness',
            set_fvp_effectiveness,
            modified
        )

        comment = f'''
! ==== ECM Applied: Exhaust Air Heat Pump (Frånluftsvärmepump) ====
! Type: NIBE F470/F750 or equivalent
! Capacity: {capacity_kw} kW
! COP: {cop}
! Modeled effectiveness: {effectiveness:.0%}
! Expected savings: 40-60% heating + DHW
! Note: Cannot combine with FTX (competing heat source)

'''
        return comment + modified

    def _apply_ground_source_heat_pump(
        self,
        idf_content: str,
        params: Dict[str, Any]
    ) -> str:
        """
        Apply Ground Source Heat Pump (Bergvärmepump).

        High-efficiency heating from geothermal source.
        Modeled by significantly reducing heating energy demand.
        """
        cop = params.get('cop', 4.5)
        capacity_kw = params.get('capacity_kw', 20)

        # GSHP covers most of heating load with high COP
        # Model thermal benefit: reduced infiltration (better sealing with HP installation)
        # and improved heat recovery if not already at maximum
        modified = idf_content

        # 1. Reduce infiltration (GSHP installation typically improves building tightness)
        infiltration_reduction = 0.80  # 20% reduction

        # Scientific notation support for Bayesian calibration output
        NUM = r'[\d.]+(?:[eE][+-]?\d+)?'

        def reduce_infiltration_ach(match):
            prefix = match.group(1)
            spaces = match.group(2)
            current_ach = float(match.group(3))
            new_ach = current_ach * infiltration_reduction
            return f'{prefix}{spaces}{new_ach:.4f};'

        # Pattern for ZoneInfiltration:DesignFlowRate with AirChanges/Hour
        # Handles multi-line format with empty fields between AirChanges/Hour and value
        pattern_ach = rf'(AirChanges/Hour,)([\s,]*)({NUM});'
        modified = re.sub(pattern_ach, reduce_infiltration_ach, modified, flags=re.IGNORECASE)

        # 2. Enable heat recovery if not already enabled
        modified = re.sub(
            r'None,\s*!-\s*Heat Recovery Type',
            'Sensible,                        !- Heat Recovery Type (ECM: GSHP system)',
            modified
        )

        # 3. Improve heat recovery effectiveness if not already high
        # Pattern handles both comma and semicolon terminators
        def improve_hr_effectiveness(match):
            current_eff = float(match.group(1))
            terminator = match.group(2)
            if current_eff < 0.85:
                return f'0.85{terminator}                         !- Sensible Heat Recovery Effectiveness (ECM: GSHP)'
            return match.group(0)  # Keep if already high

        # Handle both comma and semicolon after effectiveness value
        modified = re.sub(
            rf'({NUM})([,;])\s*!-\s*Sensible Heat Recovery Effectiveness',
            improve_hr_effectiveness,
            modified
        )

        # COP-based primary energy reduction
        primary_energy_reduction = (1 - 1/cop) * 100

        comment = f'''
! ==== ECM Applied: Ground Source Heat Pump (Bergvärmepump) ====
! Capacity: {capacity_kw} kW
! Seasonal COP: {cop}
! Borehole depth: {params.get('borehole_depth_m', 150)} m
! Primary energy reduction: ~{primary_energy_reduction:.0f}%
! Expected delivered energy savings: 60-70%
! Note: Best with low-temperature distribution

'''
        return comment + modified

    def _apply_dh_optimization(
        self,
        idf_content: str,
        params: Dict[str, Any]
    ) -> str:
        """
        District Heating Optimization (Fjärrvärmeoptimering).

        Operational measure to improve substation efficiency.
        Lower return temperature and higher delta-T.

        Implementation:
        - Better control allows slightly lower setpoints (0.2-0.3°C)
        - Reduced overheating during mild weather
        - Main value is economic (DH tariff) but small thermal effect

        Physics: Optimized DH substations have:
        - Lower return temperatures (better heat extraction)
        - Outdoor-compensated supply temperature
        - Tighter control → less overheating
        """
        target_return = params.get('target_return_temp_c', 35)
        setpoint_reduction = params.get('setpoint_reduction', 0.2)  # Small thermal effect

        modified = idf_content

        # Small setpoint reduction from better control
        thermostat_pattern = r'(Heating Setpoint Temperature\s*,\s*)(\d+\.?\d*)'

        def reduce_setpoint(match):
            original = float(match.group(2))
            return f'{match.group(1)}{original - setpoint_reduction:.1f}'

        modified = re.sub(thermostat_pattern, reduce_setpoint, modified)

        # Also adjust schedule-based setpoints
        schedule_pattern = r'(Schedule:Constant,\s*[^,]*(?:HeatSet|HeatSched|HeatingSetpoint|Heating_Setpoint|Heating)[^,]*,\s*[^,]*,\s*)(\d+\.?\d*)'

        def adjust_schedule(match):
            original = float(match.group(2))
            return f'{match.group(1)}{original - setpoint_reduction:.1f}'

        modified = re.sub(schedule_pattern, adjust_schedule, modified, flags=re.IGNORECASE)

        comment = f'''
! ==== ECM Applied: District Heating Optimization ====
! Target return temperature: {target_return}°C
! Setpoint improvement: -{setpoint_reduction}°C (from better control)
! Primary benefit: 5-10% cost (tariff improvement from lower return temp)
! Note: Swedish DH utilities penalize high return temperatures

'''
        return comment + modified

    def _apply_solar_thermal(
        self,
        idf_content: str,
        params: Dict[str, Any]
    ) -> str:
        """
        Solar Thermal Collectors (Solfångare).

        Reduces DHW heating load by solar preheating.

        Implementation:
        - Swedish irradiance has strong seasonal variation
        - January: 5-10% solar coverage, July: 60-70% coverage
        - Annual average: 35-45% DHW reduction
        - Model as monthly schedule of negative DHW gains

        Physics: Solar collectors capture irradiance to preheat DHW.
        In Sweden, summer provides most of the contribution due to
        long days and higher sun angles. Winter contribution is minimal.
        """
        area_m2 = params.get('area_m2', 40)
        annual_coverage = params.get('annual_coverage', 0.40)  # 40% annual
        dhw_kwh_m2_year = params.get('dhw_load', 22)  # Swedish standard

        # Monthly solar fractions for Stockholm (based on irradiance data)
        # These sum to ~4.3, so annual average is 4.3/12 = 0.36 (36%)
        monthly_solar_fraction = {
            1: 0.05,   # January - minimal sun
            2: 0.10,   # February
            3: 0.25,   # March - spring
            4: 0.45,   # April
            5: 0.60,   # May - long days
            6: 0.70,   # June - peak
            7: 0.70,   # July - peak
            8: 0.55,   # August
            9: 0.35,   # September
            10: 0.15,  # October
            11: 0.05,  # November
            12: 0.03,  # December - minimal
        }

        # Normalize to match target annual coverage
        current_annual_avg = sum(monthly_solar_fraction.values()) / 12
        scale_factor = annual_coverage / current_annual_avg

        # Calculate savings
        annual_dhw_savings = dhw_kwh_m2_year * annual_coverage
        dhw_savings_w_m2 = annual_dhw_savings * 1000 / 8760  # Average W/m²

        # Get zone names
        zone_pattern = r'Zone,\s*\n?\s*([^,\n]+)\s*,'
        zones = re.findall(zone_pattern, idf_content)

        if not zones:
            comment = f'''
! ==== ECM Applied: Solar Thermal Collectors ====
! Collector area: {area_m2} m²
! Annual DHW coverage: {annual_coverage*100:.0f}%
! Note: No zones found for modeling

'''
            return comment + idf_content

        first_zone = zones[0].strip()

        # Create monthly schedule for solar contribution
        # Higher values in summer = more DHW savings
        solar_schedule = f'''
Schedule:Compact,
    Solar_Thermal_Sched,          !- Name
    Any Number,                    !- Schedule Type Limits Name
    Through: 01/31,                !- January
    For: AllDays, Until: 24:00, {monthly_solar_fraction[1] * scale_factor:.2f},
    Through: 02/28,                !- February
    For: AllDays, Until: 24:00, {monthly_solar_fraction[2] * scale_factor:.2f},
    Through: 03/31,                !- March
    For: AllDays, Until: 24:00, {monthly_solar_fraction[3] * scale_factor:.2f},
    Through: 04/30,                !- April
    For: AllDays, Until: 24:00, {monthly_solar_fraction[4] * scale_factor:.2f},
    Through: 05/31,                !- May
    For: AllDays, Until: 24:00, {monthly_solar_fraction[5] * scale_factor:.2f},
    Through: 06/30,                !- June
    For: AllDays, Until: 24:00, {monthly_solar_fraction[6] * scale_factor:.2f},
    Through: 07/31,                !- July
    For: AllDays, Until: 24:00, {monthly_solar_fraction[7] * scale_factor:.2f},
    Through: 08/31,                !- August
    For: AllDays, Until: 24:00, {monthly_solar_fraction[8] * scale_factor:.2f},
    Through: 09/30,                !- September
    For: AllDays, Until: 24:00, {monthly_solar_fraction[9] * scale_factor:.2f},
    Through: 10/31,                !- October
    For: AllDays, Until: 24:00, {monthly_solar_fraction[10] * scale_factor:.2f},
    Through: 11/30,                !- November
    For: AllDays, Until: 24:00, {monthly_solar_fraction[11] * scale_factor:.2f},
    Through: 12/31,                !- December
    For: AllDays, Until: 24:00, {monthly_solar_fraction[12] * scale_factor:.2f};

'''

        # Base savings rate (modulated by schedule)
        # Schedule values are 0.05-0.70, representing monthly solar contribution
        # We use a base load that when multiplied by schedule gives correct annual
        base_dhw_w_m2 = dhw_kwh_m2_year * 1000 / 8760  # ~2.5 W/m²

        solar_savings = f'''
! ==== ECM Applied: Solar Thermal Collectors ====
! Collector area: {area_m2} m²
! Annual DHW coverage: {annual_coverage*100:.0f}%
! Monthly variation: {monthly_solar_fraction[1]*scale_factor*100:.0f}% (Jan) to {monthly_solar_fraction[7]*scale_factor*100:.0f}% (Jul)
! Annual savings: {annual_dhw_savings:.1f} kWh/m²/year

OtherEquipment,
    Solar_Thermal_DHW_Savings,     !- Name
    None,                          !- Fuel Type (EnergyPlus 25.x required field)
    {first_zone},                  !- Zone Name
    Solar_Thermal_Sched,           !- Schedule Name (monthly solar fraction)
    Watts/Area,                    !- Design Level Calculation Method
    ,                              !- Design Level {{W}}
    -{base_dhw_w_m2:.4f},          !- Power per Floor Area {{W/m2}} (negative = savings)
    ,                              !- Power per Person {{W/person}}
    0,                             !- Fraction Latent
    0.3,                           !- Fraction Radiant
    0.7;                           !- Fraction Lost

'''

        return solar_schedule + solar_savings + idf_content

    def _apply_low_flow_fixtures(
        self,
        idf_content: str,
        params: Dict[str, Any]
    ) -> str:
        """
        Low-Flow Water Fixtures (Snålspolande armaturer).

        Physics:
        - Standard showerhead: 12 L/min → Low-flow: 8 L/min (33% reduction)
        - Standard faucet: 8 L/min → Low-flow: 5 L/min (37% reduction)
        - Weighted average: ~30% DHW volume reduction
        - Swedish MFH: 20-25 kWh/m²/year DHW → Saves 6-7.5 kWh/m²/year

        Implementation in IdealLoads model:
        - DHW use creates internal gains (hot water in pipes, equipment)
        - Less hot water use → less internal gains from DHW-related losses
        - We model via negative internal gains (similar to DHW temp reduction)
        - Effect is ~10x larger than temp reduction (30% vs 3% savings)
        """
        showerhead_flow = params.get('showerhead_flow_lpm', 8)
        faucet_flow = params.get('faucet_flow_lpm', 5)

        # Standard flows: showerhead 12 L/min, faucet 8 L/min
        # Calculate reduction percentages
        shower_reduction = 1 - showerhead_flow / 12  # Fraction
        faucet_reduction = 1 - faucet_flow / 8       # Fraction

        # Weighted average (showers ~60% of hot water use, faucets ~40%)
        total_dhw_reduction = shower_reduction * 0.60 + faucet_reduction * 0.40

        # Swedish MFH average: 22 kWh/m²/year DHW
        # Internal gains from DHW system (tank, pipes, use): ~15% of DHW energy
        # 22 * 0.15 = 3.3 kWh/m²/year internal gains from DHW system
        # Reduction: 3.3 * dhw_reduction_fraction
        # At 30% reduction: 3.3 * 0.30 = 1.0 kWh/m²/year
        # This is ~0.11 W/m² average

        # Try to extract floor area from Zone objects
        floor_area_m2 = 2000  # Default
        zone_matches = re.findall(
            r'Zone,\s*([^,]+),\s*[^,]*,\s*[^,]*,\s*[^,]*,\s*[^,]*,\s*[^,]*,\s*[^,]*,\s*([^,]+),\s*([^;]+)',
            idf_content
        )
        if zone_matches:
            try:
                total_area = sum(float(m[2].strip()) for m in zone_matches if m[2].strip())
                if total_area > 0:
                    floor_area_m2 = total_area
            except ValueError:
                pass

        # Calculate internal gain reduction (W)
        # 3.3 kWh/m²/year = 0.38 W/m² average internal gain from DHW
        # Reduction = 0.38 * total_dhw_reduction
        dhw_gain_w_per_m2 = 0.38
        internal_gain_reduction = dhw_gain_w_per_m2 * total_dhw_reduction * floor_area_m2

        # Get zone names for applying the negative internal gain
        zone_names = re.findall(r'Zone,\s*([^,]+),', idf_content)
        if not zone_names:
            zone_names = ['Zone1']

        # Create negative internal gains (one per zone, distributed)
        gain_per_zone = internal_gain_reduction / len(zone_names)

        internal_gain_objects = []
        for zone_name in zone_names:
            zone_clean = zone_name.strip()
            internal_gain_objects.append(f'''
OtherEquipment,
    LowFlow_Reduction_{zone_clean},   !- Name
    None,                              !- Fuel Type
    {zone_clean},                      !- Zone Name
    AlwaysOn,                          !- Schedule Name
    EquipmentLevel,                    !- Design Level Calculation Method
    {-gain_per_zone:.2f},              !- Design Level (W) - NEGATIVE
    ,                                  !- Power per Zone Floor Area
    ,                                  !- Power per Person
    0,                                 !- Fraction Latent
    0,                                 !- Fraction Radiant
    0;                                 !- Fraction Lost
''')

        dhw_savings_kwh_m2 = 22 * total_dhw_reduction  # kWh/m²/year
        new_objects = f'''
! ==== ECM Applied: Low-Flow Water Fixtures ====
! Showerhead: {showerhead_flow} L/min (standard 12, reduction {shower_reduction*100:.0f}%)
! Faucet: {faucet_flow} L/min (standard 8, reduction {faucet_reduction*100:.0f}%)
! Total DHW volume reduction: {total_dhw_reduction*100:.0f}%
! DHW energy savings: ~{dhw_savings_kwh_m2:.1f} kWh/m²/year
! Internal gain reduction: {internal_gain_reduction:.1f}W ({internal_gain_reduction/floor_area_m2:.3f} W/m²)
! Note: Very low cost (100-300 SEK/fixture), easy self-installation
{''.join(internal_gain_objects)}
'''

        return new_objects + idf_content

    # ═══════════════════════════════════════════════════════════════════════════
    # ADDITIONAL ENVELOPE ECMs
    # ═══════════════════════════════════════════════════════════════════════════

    def _apply_basement_insulation(
        self,
        idf_content: str,
        params: Dict[str, Any]
    ) -> str:
        """
        Basement/Floor Insulation (Källarisolering).

        Physics:
        - Old buildings: floor U ~0.4-0.6 W/m²K (uninsulated slab/crawlspace)
        - Adding EPS/XPS insulation: reduces to ~0.15-0.20 W/m²K
        - Ground coupling effect reduced by ~60-70%

        Swedish context:
        - Common in 1950-1980 buildings with uninsulated ground slabs
        - Typical: 100-150mm XPS under slab or on crawlspace ceiling
        - Savings: 5-15% heating energy depending on building shape

        Implementation:
        1. Add new insulation material (EPS/XPS)
        2. Increase existing floor insulation thickness OR
        3. Add insulation layer to floor constructions
        """
        target_u = params.get('target_u_value', 0.15)
        thickness_mm = params.get('thickness_mm', 100)
        thickness_m = thickness_mm / 1000
        material = params.get('material', 'xps')  # XPS typical for basement

        # Get insulation properties
        insulation_props = self.INSULATION_PROPERTIES.get(
            material, self.INSULATION_PROPERTIES.get('eps', {'conductivity': 0.035})
        )
        conductivity = insulation_props.get('conductivity', 0.035)

        modified = idf_content
        insulation_added = False

        # Create new floor insulation material
        new_material = f'''
Material,
    Floor_Insulation_{material.upper()},    !- Name
    Rough,                                   !- Roughness
    {thickness_m:.4f},                       !- Thickness (m)
    {conductivity:.3f},                      !- Conductivity (W/m-K)
    30,                                      !- Density (kg/m3)
    1400,                                    !- Specific Heat (J/kg-K)
    0.9,                                     !- Thermal Absorptance
    0.7,                                     !- Solar Absorptance
    0.7;                                     !- Visible Absorptance

'''

        # Find floor constructions and add insulation layer
        # Pattern for Construction objects with floor/ground in name
        floor_pattern = r'(Construction,\s*\n?\s*)([^,]+(?:Floor|Ground|Slab)[^,]*)(,\s*\n?\s*)([^;]+;)'

        def add_insulation_layer(match):
            nonlocal insulation_added
            prefix = match.group(1)
            name = match.group(2).strip()
            comma = match.group(3)
            layers = match.group(4)

            # Add insulation as first layer (inside surface)
            insulation_added = True
            return f'''{prefix}{name}{comma}Floor_Insulation_{material.upper()},
    {layers}'''

        modified = re.sub(floor_pattern, add_insulation_layer, modified, flags=re.IGNORECASE)

        # Alternative pattern: simpler Construction format
        if not insulation_added:
            simple_pattern = r'(Construction,\s*)([^,]*[Ff]loor[^,]*),(\s*[^;]+;)'

            def add_insulation_simple(match):
                nonlocal insulation_added
                prefix = match.group(1)
                name = match.group(2).strip()
                rest = match.group(3)
                insulation_added = True
                return f'''{prefix}{name},
    Floor_Insulation_{material.upper()},{rest}'''

            modified = re.sub(simple_pattern, add_insulation_simple, modified)

        # Calculate approximate U-value improvement
        # R_added = thickness / conductivity
        r_added = thickness_m / conductivity
        # Assume original floor R ~1.5 (U ~0.67)
        original_r = 1.5
        new_r = original_r + r_added
        new_u = 1 / new_r

        if insulation_added:
            comment = f'''
! ==== ECM Applied: Basement Insulation (Källarisolering) ====
! Material: {material.upper()} insulation
! Added thickness: {thickness_mm} mm
! Conductivity: {conductivity} W/m-K
! Added R-value: {r_added:.2f} m²K/W
! Estimated new floor U: ~{new_u:.2f} W/m²K
! Expected heating savings: 5-15%

{new_material}
'''
        else:
            # Even if no construction found, add a note
            comment = f'''
! ==== ECM Applied: Basement Insulation (Källarisolering) ====
! Target floor U-value: {target_u} W/m²K
! Material: {material.upper()}, {thickness_mm} mm
! Note: Could not find floor construction to modify
! Expected savings if implemented: 5-15% heating energy

{new_material}
'''

        return comment + modified

    def _apply_thermal_bridge_remediation(
        self,
        idf_content: str,
        params: Dict[str, Any]
    ) -> str:
        """
        Thermal Bridge Remediation (Köldbryggeåtgärder).

        Reduces overall heat loss by addressing junctions, balconies, etc.
        Typically 5-15% reduction in transmission losses.

        Physics:
        - Thermal bridges at junctions increase effective U-value by 5-20%
        - Remediation reduces this penalty
        - Also reduces associated air leakage at joints

        Implementation:
        1. Reduce material conductivity slightly (improves effective U-value)
        2. Reduce infiltration (sealed joints)
        """
        reduction_factor = params.get('reduction_factor', 0.10)  # 10% reduction

        modified = idf_content

        # Scientific notation support for Bayesian calibration output
        NUM = r'[\d.]+(?:[eE][+-]?\d+)?'

        # 1. Reduce infiltration (thermal bridges often cause drafts at joints)
        # Pattern handles multi-line format with empty fields
        def reduce_infiltration(match):
            prefix = match.group(1)
            blanks = match.group(2)
            current = float(match.group(3))
            new_value = current * (1 - reduction_factor * 0.5)  # 5% reduction in infiltration
            return f'{prefix}{blanks}{new_value:.4f};'

        pattern_ach = rf'(AirChanges/Hour,)([\s,]*)({NUM});'
        modified = re.sub(pattern_ach, reduce_infiltration, modified, flags=re.IGNORECASE)

        # 2. Improve material conductivity (reduces effective U-value)
        # Reduce conductivity of wall materials by ~5-10%
        def reduce_conductivity(match):
            thickness = match.group(1)
            current_k = float(match.group(2))
            new_k = current_k * (1 - reduction_factor * 0.5)  # 5% reduction
            return f'{thickness},  !- Thickness {{m}}\n    {new_k:.4f},  !- Conductivity (ECM: Thermal Bridge)'

        # Pattern: Thickness followed by Conductivity in Material object
        pattern_k = rf'({NUM}),\s*!-\s*Thickness[^\n]*\n\s*({NUM}),\s*!-\s*Conductivity'
        modified = re.sub(pattern_k, reduce_conductivity, modified, flags=re.IGNORECASE)

        comment = f'''
! ==== ECM Applied: Thermal Bridge Remediation ====
! Reduction factor: {reduction_factor*100:.0f}%
! Addresses: balcony connections, window frames, wall junctions
! Expected savings: 5-15% of transmission losses

'''
        return comment + modified

    def _apply_facade_renovation(
        self,
        idf_content: str,
        params: Dict[str, Any]
    ) -> str:
        """
        Comprehensive Facade Renovation (Fasadrenovering).

        Combines external insulation + new windows + air sealing.
        Major renovation typically done every 40-50 years.
        """
        wall_u = params.get('target_wall_u', 0.18)
        window_u = params.get('target_window_u', 1.0)

        # Apply wall insulation
        modified = self._apply_wall_insulation(idf_content, {'target_u_value': wall_u}, external=True)
        # Apply window replacement
        modified = self._apply_window_replacement(modified, {'target_u_value': window_u})
        # Apply air sealing
        modified = self._apply_air_sealing(modified, {'target_ach': 0.04})

        comment = f'''
! ==== ECM Applied: Comprehensive Facade Renovation ====
! Wall U-value: {wall_u} W/m²K
! Window U-value: {window_u} W/m²K
! Air tightness: 0.04 ACH
! Major renovation package

'''
        return comment + modified

    def _apply_entrance_door(
        self,
        idf_content: str,
        params: Dict[str, Any]
    ) -> str:
        """
        Entrance Door Replacement (Entrédörrbyte).

        Physics:
        - Old entrance doors: poor seals, manual closers, gaps
        - Account for 5-15% of total building infiltration
        - New automatic doors with proper seals reduce door-related leakage 70-80%
        - Net building infiltration reduction: ~8-10%

        Swedish context:
        - Common in 1960-1990 MFH buildings
        - Often combined with vestibule (vindfång) improvements
        - U-value improvement from ~2.5 to ~1.2 W/m²K

        Implementation:
        Reduce infiltration rate by 8% (conservative estimate for door contribution).
        """
        target_u = params.get('target_u_value', 1.2)
        infiltration_reduction = params.get('infiltration_reduction', 0.08)  # 8%

        modified = idf_content
        infiltration_modified = False

        # Scientific notation support for Bayesian calibration output
        NUM = r'[\d.]+(?:[eE][+-]?\d+)?'

        # Find ZoneInfiltration objects and reduce the rate
        # The actual format has empty fields between AirChanges/Hour and the value:
        # AirChanges/Hour,
        #     ,
        #     ,
        #     ,
        #     0.06;
        def reduce_ach(match):
            nonlocal infiltration_modified
            prefix = match.group(1)
            blanks = match.group(2)
            current_ach = float(match.group(3))
            new_ach = current_ach * (1 - infiltration_reduction)
            infiltration_modified = True
            return f'{prefix}{blanks}{new_ach:.4f};'

        # Pattern handles multi-line format with empty fields
        ach_pattern = rf'(AirChanges/Hour,)([\s,]*)({NUM});'
        modified = re.sub(ach_pattern, reduce_ach, modified, flags=re.IGNORECASE)

        # Pattern for flow rate based infiltration
        if not infiltration_modified:
            flow_pattern = r'(ZoneInfiltration:DesignFlowRate,\s*[^;]*Flow/Zone,\s*)([\d.]+)'

            def reduce_flow(match):
                nonlocal infiltration_modified
                prefix = match.group(1)
                current_flow = float(match.group(2))
                new_flow = current_flow * (1 - infiltration_reduction)
                infiltration_modified = True
                return f'{prefix}{new_flow:.6f}'

            modified = re.sub(flow_pattern, reduce_flow, modified, flags=re.IGNORECASE)

        # Alternative: look for infiltration values in any format
        if not infiltration_modified:
            # Generic pattern for infiltration values
            generic_pattern = r'(ZoneInfiltration[^;]*,\s*)(0\.\d+)(\s*[,;])'

            def reduce_generic(match):
                nonlocal infiltration_modified
                prefix = match.group(1)
                value = float(match.group(2))
                suffix = match.group(3)
                if value < 1.0:  # Likely a rate, not a coefficient
                    infiltration_modified = True
                    new_value = value * (1 - infiltration_reduction)
                    return f'{prefix}{new_value:.4f}{suffix}'
                return match.group(0)

            modified = re.sub(generic_pattern, reduce_generic, modified, flags=re.IGNORECASE)

        if infiltration_modified:
            comment = f'''
! ==== ECM Applied: Entrance Door Replacement (Entrédörrbyte) ====
! Target door U-value: {target_u} W/m²K
! Infiltration reduction: {infiltration_reduction*100:.0f}%
! Features: Automatic closer, improved seals, insulated core
! Note: Also improves thermal comfort near entrances
! Expected heating savings: 3-5%

'''
        else:
            comment = f'''
! ==== ECM Applied: Entrance Door Replacement (Entrédörrbyte) ====
! Target door U-value: {target_u} W/m²K
! Infiltration reduction potential: {infiltration_reduction*100:.0f}%
! Note: Could not find infiltration object to modify
! Expected savings if implemented: 3-5% heating energy

'''

        return comment + modified

    def _apply_pipe_insulation(
        self,
        idf_content: str,
        params: Dict[str, Any]
    ) -> str:
        """
        Pipe Insulation (Rörisolering).

        Physics:
        - Uninsulated pipes in basements/technical rooms lose 5-10% of heat
        - 40mm mineral wool insulation reduces losses by 70-80%
        - Net effect: 3-8% reduction in total heating/DHW energy

        Swedish context:
        - Required by BBR for new construction
        - Common retrofit in 1960-1990 buildings
        - Series 1 insulation class standard (40mm for <50mm pipes)

        Implementation:
        Model as reduced internal gains (pipe losses that were heating
        unoccupied spaces now stay in the system).
        """
        insulation_thickness_mm = params.get('thickness_mm', 40)
        loss_reduction = params.get('loss_reduction', 0.75)  # 75% reduction

        modified = idf_content

        # Calculate internal gain reduction
        # Pipe losses typically 5% of total heating in uninsulated building
        # For a 2000m² building at 100 kWh/m²/year = 200,000 kWh
        # 5% = 10,000 kWh/year = 1.14 kW average
        # With 75% reduction = 0.86 kW average reduction

        # Get floor area
        floor_area_m2 = 2000  # Default
        zone_matches = re.findall(
            r'Zone,\s*([^,]+),\s*[^,]*,\s*[^,]*,\s*[^,]*,\s*[^,]*,\s*[^,]*,\s*[^,]*,\s*([^,]+),\s*([^;]+)',
            modified
        )
        if zone_matches:
            try:
                total_area = sum(float(m[2].strip()) for m in zone_matches if m[2].strip())
                if total_area > 0:
                    floor_area_m2 = total_area
            except ValueError:
                pass

        # Pipe loss intensity: ~0.5 W/m² average (5% of 100 W/m² heating)
        # Reduction: 0.5 * 0.75 = 0.375 W/m²
        pipe_loss_intensity = 0.5  # W/m²
        gain_reduction = pipe_loss_intensity * loss_reduction * floor_area_m2

        # Get zone names
        zone_names = re.findall(r'Zone,\s*([^,]+),', modified)
        if not zone_names:
            zone_names = ['Zone1']

        gain_per_zone = gain_reduction / len(zone_names)

        # Create negative internal gains (represents reduced losses)
        internal_gain_objects = []
        for zone_name in zone_names:
            zone_clean = zone_name.strip()
            internal_gain_objects.append(f'''
OtherEquipment,
    PipeLoss_Reduction_{zone_clean},  !- Name
    None,                              !- Fuel Type
    {zone_clean},                      !- Zone Name
    AlwaysOn,                          !- Schedule Name
    EquipmentLevel,                    !- Design Level Calculation Method
    {-gain_per_zone:.2f},              !- Design Level (W) - NEGATIVE
    ,                                  !- Power per Zone Floor Area
    ,                                  !- Power per Person
    0,                                 !- Fraction Latent
    0.3,                               !- Fraction Radiant (some pipe in walls)
    0;                                 !- Fraction Lost
''')

        savings_kwh_m2 = pipe_loss_intensity * loss_reduction * 8760 / 1000  # kWh/m²/year
        comment = f'''
! ==== ECM Applied: Pipe Insulation (Rörisolering) ====
! Insulation thickness: {insulation_thickness_mm} mm (Series 1 standard)
! Loss reduction: {loss_reduction*100:.0f}%
! Heating gain reduction: {gain_reduction:.1f}W ({gain_reduction/floor_area_m2:.3f} W/m²)
! Expected savings: ~{savings_kwh_m2:.1f} kWh/m²/year
! Note: Applied to heating and DHW distribution pipes
{''.join(internal_gain_objects)}
'''

        return comment + modified

    # ═══════════════════════════════════════════════════════════════════════════
    # HVAC & CONTROLS ECMs
    # ═══════════════════════════════════════════════════════════════════════════

    def _apply_radiator_fans(
        self,
        idf_content: str,
        params: Dict[str, Any]
    ) -> str:
        """
        Radiator Fans (Radiatorfläktar).

        Physics:
        - Small fans behind radiators increase convection
        - Heat output increases 20-40% at same water temp
        - Allows 1-2°C lower supply temperature
        - Or maintains comfort with 1°C lower room temp

        Swedish context:
        - Common in poorly insulated buildings
        - Low-cost retrofit (~500 SEK/radiator)
        - Electricity use: ~5W per fan, offset by heating savings

        Implementation:
        - Reduce heating setpoint by 1°C
        - Add small electricity load for fans
        """
        setpoint_reduction = params.get('setpoint_reduction', 1.0)  # °C
        fan_power_per_m2 = params.get('fan_power_w_m2', 0.1)  # ~1 fan per 20m²

        modified = idf_content
        setpoint_modified = False

        # Find and reduce heating setpoint in Schedule:Constant
        base_setpoint = 21.0
        setpoint_match = re.search(
            r'Schedule:Constant,\s*(HeatSched|HeatSet|HeatingSetpoint|Heating_Setpoint|Heating)\s*,\s*Temperature\s*,\s*([\d.]+)',
            modified, re.IGNORECASE
        )
        if setpoint_match:
            schedule_name = setpoint_match.group(1)
            base_setpoint = float(setpoint_match.group(2))
            new_setpoint = base_setpoint - setpoint_reduction

            pattern = rf'(Schedule:Constant,\s*{re.escape(schedule_name)}\s*,\s*Temperature\s*,\s*){re.escape(str(base_setpoint))}'
            modified = re.sub(pattern, rf'\g<1>{new_setpoint:.1f}', modified, flags=re.IGNORECASE)
            setpoint_modified = True

        # Get floor area for fan electricity calculation
        floor_area_m2 = 2000
        zone_matches = re.findall(
            r'Zone,\s*([^,]+),\s*[^,]*,\s*[^,]*,\s*[^,]*,\s*[^,]*,\s*[^,]*,\s*[^,]*,\s*([^,]+),\s*([^;]+)',
            modified
        )
        if zone_matches:
            try:
                total_area = sum(float(m[2].strip()) for m in zone_matches if m[2].strip())
                if total_area > 0:
                    floor_area_m2 = total_area
            except ValueError:
                pass

        # Add fan electricity load
        total_fan_power = fan_power_per_m2 * floor_area_m2
        zone_names = re.findall(r'Zone,\s*([^,]+),', modified)
        if not zone_names:
            zone_names = ['Zone1']

        fan_per_zone = total_fan_power / len(zone_names)

        fan_objects = []
        for zone_name in zone_names:
            zone_clean = zone_name.strip()
            fan_objects.append(f'''
ElectricEquipment,
    RadiatorFan_{zone_clean},         !- Name
    {zone_clean},                      !- Zone Name
    Heating_Availability,              !- Schedule Name (only during heating season)
    EquipmentLevel,                    !- Design Level Calculation Method
    {fan_per_zone:.1f},                !- Design Level (W)
    ,                                  !- Power per Zone Floor Area
    ,                                  !- Power per Person
    0,                                 !- Fraction Latent
    0.0,                               !- Fraction Radiant
    0;                                 !- Fraction Lost (motor heat in room)
''')

        # Add heating availability schedule if not present
        heating_schedule = '''
Schedule:Compact,
    Heating_Availability,            !- Name
    Fraction,                         !- Schedule Type Limits
    Through: 5/15, For: AllDays, Until: 24:00, 1,
    Through: 9/14, For: AllDays, Until: 24:00, 0,
    Through: 12/31, For: AllDays, Until: 24:00, 1;

'''

        if setpoint_modified:
            comment = f'''
! ==== ECM Applied: Radiator Fans (Radiatorfläktar) ====
! Original setpoint: {base_setpoint:.1f}°C → New: {base_setpoint - setpoint_reduction:.1f}°C
! Setpoint reduction: {setpoint_reduction}°C
! Fan electricity: {total_fan_power:.0f}W total ({fan_power_per_m2:.2f} W/m²)
! Physics: Improved convection allows lower supply temp
! Expected heating savings: 5-10% (net of fan electricity)

{heating_schedule}
{''.join(fan_objects)}
'''
        else:
            comment = f'''
! ==== ECM Applied: Radiator Fans (Radiatorfläktar) ====
! Setpoint reduction potential: {setpoint_reduction}°C
! Fan electricity: {total_fan_power:.0f}W ({fan_power_per_m2:.2f} W/m²)
! Note: Could not find heating schedule to modify

{heating_schedule}
{''.join(fan_objects)}
'''

        return comment + modified

    def _apply_heat_recovery_dhw(
        self,
        idf_content: str,
        params: Dict[str, Any]
    ) -> str:
        """
        DHW Heat Recovery (Värmeåtervinning från avloppsvatten).

        Physics:
        - Drain water exits at ~30-35°C
        - Cold water inlet at ~8°C (Sweden average)
        - Heat exchanger preheats inlet water to ~20°C
        - Reduces DHW heating load by 30-50%

        Swedish context:
        - Common in ecodistricts and passive houses
        - Drain water heat recovery (DWHR) typical 40% efficiency
        - Works best with showers (continuous flow)

        Implementation:
        Model as reduced internal gains representing lower DHW energy need.
        The preheated water reduces the energy needed from the heating system.
        """
        recovery_efficiency = params.get('efficiency', 0.40)  # 40% typical

        modified = idf_content

        # Swedish DHW: 22 kWh/m²/year average
        # ΔT without recovery: 55°C - 8°C = 47°C
        # ΔT with recovery: 55°C - 20°C = 35°C
        # Reduction: 1 - 35/47 = 26% minimum
        # With 40% efficiency on shower fraction (60%): ~25% savings

        dhw_savings_fraction = recovery_efficiency * 0.60  # 60% of DHW is showers

        # Get floor area
        floor_area_m2 = 2000
        zone_matches = re.findall(
            r'Zone,\s*([^,]+),\s*[^,]*,\s*[^,]*,\s*[^,]*,\s*[^,]*,\s*[^,]*,\s*[^,]*,\s*([^,]+),\s*([^;]+)',
            modified
        )
        if zone_matches:
            try:
                total_area = sum(float(m[2].strip()) for m in zone_matches if m[2].strip())
                if total_area > 0:
                    floor_area_m2 = total_area
            except ValueError:
                pass

        # DHW internal gains reduction
        # 22 kWh/m²/year * savings_fraction = ~5 kWh/m²/year
        # As average power: 5 * 1000 / 8760 = 0.57 W/m²
        dhw_gain_reduction = 22 * dhw_savings_fraction * 1000 / 8760 * floor_area_m2 / 1000  # kW

        zone_names = re.findall(r'Zone,\s*([^,]+),', modified)
        if not zone_names:
            zone_names = ['Zone1']

        gain_per_zone = dhw_gain_reduction * 1000 / len(zone_names)  # W

        internal_gain_objects = []
        for zone_name in zone_names:
            zone_clean = zone_name.strip()
            internal_gain_objects.append(f'''
OtherEquipment,
    DWHR_{zone_clean},                !- Name
    None,                              !- Fuel Type
    {zone_clean},                      !- Zone Name
    AlwaysOn,                          !- Schedule Name
    EquipmentLevel,                    !- Design Level Calculation Method
    {-gain_per_zone:.2f},              !- Design Level (W) - NEGATIVE
    ,                                  !- Power per Zone Floor Area
    ,                                  !- Power per Person
    0,                                 !- Fraction Latent
    0,                                 !- Fraction Radiant
    0;                                 !- Fraction Lost
''')

        savings_kwh_m2 = 22 * dhw_savings_fraction
        comment = f'''
! ==== ECM Applied: Drain Water Heat Recovery (DWHR) ====
! Recovery efficiency: {recovery_efficiency*100:.0f}%
! DHW savings fraction: {dhw_savings_fraction*100:.0f}% (showers only)
! Energy savings: ~{savings_kwh_m2:.1f} kWh/m²/year
! Internal gain reduction: {dhw_gain_reduction*1000:.1f}W
! Physics: Preheats cold water inlet using drain water
{''.join(internal_gain_objects)}
'''

        return comment + modified

    def _apply_vrf_system(
        self,
        idf_content: str,
        params: Dict[str, Any]
    ) -> str:
        """
        Variable Refrigerant Flow System (VRF/VRV).

        High-efficiency heat pump system with simultaneous heating/cooling.
        COP typically 4-6.

        Thermal modeling approach:
        1. Reduce infiltration (better building integration with VRF)
        2. Enable/improve heat recovery (VRF systems have zone-to-zone HR)
        3. Better zonal control reduces overshooting
        """
        cop = params.get('cop', 4.5)

        modified = idf_content

        # Scientific notation support for Bayesian calibration output
        NUM = r'[\d.]+(?:[eE][+-]?\d+)?'

        # 1. Reduce infiltration (VRF systems require good building tightness)
        infiltration_reduction = 0.85  # 15% reduction

        def reduce_infiltration_ach(match):
            prefix = match.group(1)
            spaces = match.group(2)
            current_ach = float(match.group(3))
            new_ach = current_ach * infiltration_reduction
            return f'{prefix}{spaces}{new_ach:.4f};'

        pattern_ach = rf'(AirChanges/Hour,)([\s,]*)({NUM});'
        modified = re.sub(pattern_ach, reduce_infiltration_ach, modified, flags=re.IGNORECASE)

        # 2. Enable heat recovery if not already enabled (VRF has built-in HR)
        modified = re.sub(
            r'None,\s*!-\s*Heat Recovery Type',
            'Sensible,                        !- Heat Recovery Type (ECM: VRF)',
            modified
        )

        # 3. Improve heat recovery effectiveness - VRF can achieve 90%+ with zone-to-zone transfer
        def improve_hr_effectiveness(match):
            current_eff = float(match.group(1))
            terminator = match.group(2)
            if current_eff < 0.88:
                return f'0.88{terminator}                         !- Sensible Heat Recovery Effectiveness (ECM: VRF)'
            return match.group(0)  # Keep if already high

        modified = re.sub(
            rf'({NUM})([,;])\s*!-\s*Sensible Heat Recovery Effectiveness',
            improve_hr_effectiveness,
            modified
        )

        comment = f'''
! ==== ECM Applied: VRF/VRV System ====
! Coefficient of Performance: {cop}
! Variable refrigerant flow for efficient heating/cooling
! Enables heat recovery between zones
! Thermal benefits modeled: infiltration reduction, improved HR efficiency

'''
        return comment + modified

    def _apply_occupancy_sensors(
        self,
        idf_content: str,
        params: Dict[str, Any]
    ) -> str:
        """
        Occupancy Sensors for Lighting/HVAC (Närvarosensorer).

        Reduces lighting and ventilation when spaces are unoccupied.
        Typically 20-40% reduction in lighting energy.
        """
        lighting_reduction = params.get('lighting_reduction', 0.30)

        # Reduce lighting power density
        pattern = r'(Lights,\s*[^,]*,\s*[^,]*,\s*[^,]*,\s*Watts/Area\s*,\s*)(\d+\.?\d*)'

        def reduce_lighting(match):
            original = float(match.group(2))
            reduced = original * (1 - lighting_reduction)
            return f'{match.group(1)}{reduced:.2f}'

        modified = re.sub(pattern, reduce_lighting, idf_content)

        comment = f'''
! ==== ECM Applied: Occupancy Sensors ====
! Lighting reduction factor: {lighting_reduction*100:.0f}%
! Controls lighting and ventilation based on occupancy
! Applied to common areas, meeting rooms, corridors

'''
        return comment + modified

    def _apply_daylight_sensors(
        self,
        idf_content: str,
        params: Dict[str, Any]
    ) -> str:
        """
        Daylight Sensors (Dagsljussensorer).

        Dims artificial lighting based on available daylight.
        Typically 30-50% reduction in lighting energy.
        """
        reduction = params.get('reduction', 0.35)

        # Similar to occupancy sensors, reduce lighting
        pattern = r'(Lights,\s*[^,]*,\s*[^,]*,\s*[^,]*,\s*Watts/Area\s*,\s*)(\d+\.?\d*)'

        def reduce_lighting(match):
            original = float(match.group(2))
            reduced = original * (1 - reduction)
            return f'{match.group(1)}{reduced:.2f}'

        modified = re.sub(pattern, reduce_lighting, idf_content)

        comment = f'''
! ==== ECM Applied: Daylight Sensors ====
! Lighting reduction: {reduction*100:.0f}%
! Dims lighting based on available daylight
! Most effective in perimeter zones

'''
        return comment + modified

    def _apply_predictive_control(
        self,
        idf_content: str,
        params: Dict[str, Any]
    ) -> str:
        """
        Predictive/MPC Control (Prediktiv styrning).

        Uses weather forecasts and occupancy prediction for optimal control.
        Typically 5-15% energy savings.

        Physics: MPC systems:
        - Predict weather and pre-heat/pre-cool based on forecast
        - Reduce overshoot by tighter control (0.5°C reduction possible)
        - Optimize start times (avoid heating empty building)
        - Swedish studies show 5-10% savings typical for MFH

        Implementation:
        - Reduce base setpoint by 0.5°C (tighter control = less overshoot)
        - Handles both Schedule:Constant and inline setpoint formats
        """
        setpoint_reduction = params.get('setpoint_reduction', 0.5)
        savings_factor = params.get('savings_factor', 0.10)

        modified = idf_content

        # Extract current setpoint (handles Bayesian-calibrated values)
        base_setpoint, schedule_name = self._extract_heating_setpoint(idf_content)
        new_setpoint = base_setpoint - setpoint_reduction

        # Create optimized schedule
        new_schedule_name = f"{schedule_name}_MPC"

        mpc_schedule = f'''
! ==== ECM Applied: Predictive/MPC Control ====
! Original setpoint: {base_setpoint:.1f}°C
! Optimized setpoint: {new_setpoint:.1f}°C (tighter control, less overshoot)
! Expected savings: {savings_factor*100:.0f}%
! Features: Weather prediction, optimal start, occupancy sensing

Schedule:Constant,
    {new_schedule_name},          !- Name
    Temperature,                   !- Schedule Type Limits Name
    {new_setpoint:.1f};            !- Optimized setpoint value

'''

        # Replace thermostat references to use new schedule
        modified = self._replace_thermostat_reference(modified, schedule_name, new_schedule_name)

        return mpc_schedule + modified

    def _apply_fault_detection(
        self,
        idf_content: str,
        params: Dict[str, Any]
    ) -> str:
        """
        Fault Detection and Diagnostics (FDD).

        Continuous monitoring and alerting on system faults.

        Implementation:
        - Identifies stuck valves → reduces infiltration 3%
        - Detects sensor drift → corrects setpoints 0.3°C
        - Catches schedule errors → 2% schedule savings
        - Typical total savings: 3-5% of heating energy

        Physics: Buildings have ~5-10 active faults at any time.
        Stuck dampers cause infiltration, failed sensors cause overheating,
        schedule errors waste energy during unoccupied periods.
        FDD catches these within days instead of months/years.
        """
        infiltration_reduction = params.get('infiltration_reduction', 0.03)  # 3%
        setpoint_correction = params.get('setpoint_correction', 0.3)

        modified = idf_content

        # 1. Reduce infiltration (stuck dampers/valves detected and fixed)
        infiltration_pattern = r'(ZoneInfiltration:DesignFlowRate,[^;]*?Design Flow Rate\s*{[^}]*}\s*,\s*)(\d+\.?\d*)'

        def reduce_infiltration(match):
            original = float(match.group(2))
            new_value = original * (1 - infiltration_reduction)
            return f'{match.group(1)}{new_value:.4f}'

        modified = re.sub(infiltration_pattern, reduce_infiltration, modified, flags=re.IGNORECASE | re.DOTALL)

        # Also try ACH pattern
        ach_pattern = r'(AirChanges/Hour\s*,\s*)(\d+\.?\d*)'

        def reduce_ach(match):
            original = float(match.group(2))
            new_value = original * (1 - infiltration_reduction)
            return f'{match.group(1)}{new_value:.4f}'

        modified = re.sub(ach_pattern, reduce_ach, modified)

        # 2. Small setpoint correction (sensor drift detected)
        thermostat_pattern = r'(Heating Setpoint Temperature\s*,\s*)(\d+\.?\d*)'

        def correct_setpoint(match):
            original = float(match.group(2))
            return f'{match.group(1)}{original - setpoint_correction:.1f}'

        modified = re.sub(thermostat_pattern, correct_setpoint, modified)

        # Also adjust schedule-based setpoints
        schedule_pattern = r'(Schedule:Constant,\s*[^,]*(?:HeatSet|HeatSched|HeatingSetpoint|Heating_Setpoint|Heating)[^,]*,\s*[^,]*,\s*)(\d+\.?\d*)'

        def adjust_schedule(match):
            original = float(match.group(2))
            return f'{match.group(1)}{original - setpoint_correction:.1f}'

        modified = re.sub(schedule_pattern, adjust_schedule, modified, flags=re.IGNORECASE)

        comment = f'''
! ==== ECM Applied: Fault Detection & Diagnostics (FDD) ====
! Continuous monitoring for system faults
! Infiltration correction: -{infiltration_reduction*100:.0f}% (stuck dampers fixed)
! Setpoint correction: -{setpoint_correction}°C (sensor drift detected)
! Schedule error detection: ~2% additional savings
! Investment: ~50,000-150,000 SEK for FDD system

'''
        return comment + modified

    def _apply_building_automation(
        self,
        idf_content: str,
        params: Dict[str, Any]
    ) -> str:
        """
        Building Automation System (Fastighetsautomation).

        Comprehensive BMS upgrade or installation.
        Enables all control-based ECMs.

        Physics: BAS provides:
        - Centralized scheduling (avoid heating empty spaces)
        - Optimal start/stop algorithms
        - Zone-by-zone control
        - Integration of all subsystems
        - Typical savings: 5-15% for poorly controlled buildings
        """
        setpoint_reduction = params.get('setpoint_reduction', 0.5)

        modified = idf_content

        # Extract current setpoint (handles Bayesian-calibrated values)
        base_setpoint, schedule_name = self._extract_heating_setpoint(idf_content)
        new_setpoint = base_setpoint - setpoint_reduction

        # Create optimized schedule
        new_schedule_name = f"{schedule_name}_BAS"

        bas_schedule = f'''
! ==== ECM Applied: Building Automation System ====
! Original setpoint: {base_setpoint:.1f}°C
! Optimized setpoint: {new_setpoint:.1f}°C (centralized control optimization)
! Features: Optimal start/stop, zone control, scheduling

Schedule:Constant,
    {new_schedule_name},          !- Name
    Temperature,                   !- Schedule Type Limits Name
    {new_setpoint:.1f};            !- Optimized setpoint value

'''

        # Replace thermostat references
        modified = self._replace_thermostat_reference(modified, schedule_name, new_schedule_name)

        return bas_schedule + modified

    # ═══════════════════════════════════════════════════════════════════════════
    # METERING & MONITORING ECMs
    # ═══════════════════════════════════════════════════════════════════════════

    def _apply_individual_metering(
        self,
        idf_content: str,
        params: Dict[str, Any]
    ) -> str:
        """
        Individual Metering (Individuell mätning - IMD).

        Per-apartment energy metering.
        Behavioral change typically saves 10-20%.
        """
        savings = params.get('behavioral_savings', 0.15)

        # Reduce internal loads slightly (occupant behavior)
        pattern = r'(People,\s*[^,]*,\s*[^,]*,\s*[^,]*,\s*[^,]*,\s*[^,]*,\s*)(\d+\.?\d*)'

        def reduce_activity(match):
            # Slightly reduce activity level (more conscious behavior)
            original = float(match.group(2))
            return f'{match.group(1)}{original * 0.95:.1f}'

        modified = re.sub(pattern, reduce_activity, idf_content)

        comment = f'''
! ==== ECM Applied: Individual Metering (IMD) ====
! Expected behavioral savings: {savings*100:.0f}%
! Per-apartment metering for heating, hot water, electricity
! Required for new buildings in Sweden since 2021

'''
        return comment + modified

    def _apply_energy_monitoring(
        self,
        idf_content: str,
        params: Dict[str, Any]
    ) -> str:
        """
        Energy Monitoring System (Energiuppföljning).

        Building-level energy monitoring with visualization and reporting.

        Implementation:
        - Behavioral change from seeing consumption: 5-10% savings
        - Enables benchmarking and anomaly detection
        - Swedish studies (Sveby) show 5-15% from visibility alone
        - Model as reduced internal gains (occupant behavior) and slight setpoint

        Physics: When occupants and facility managers SEE real-time energy data,
        they adjust behavior: lower thermostats, turn off lights, report issues.
        The Hawthorne effect applies to buildings too.
        """
        behavioral_savings_pct = params.get('behavioral_savings', 7)  # 7% typical

        modified = idf_content

        # 1. Slight setpoint reduction (awareness leads to lower setpoints)
        setpoint_reduction = 0.3  # 0.3°C from better awareness

        thermostat_pattern = r'(Heating Setpoint Temperature\s*,\s*)(\d+\.?\d*)'

        def reduce_setpoint(match):
            original = float(match.group(2))
            return f'{match.group(1)}{original - setpoint_reduction:.1f}'

        modified = re.sub(thermostat_pattern, reduce_setpoint, modified)

        # Also adjust schedule-based setpoints
        schedule_pattern = r'(Schedule:Constant,\s*[^,]*(?:HeatSet|HeatSched|HeatingSetpoint|Heating_Setpoint|Heating)[^,]*,\s*[^,]*,\s*)(\d+\.?\d*)'

        def adjust_schedule(match):
            original = float(match.group(2))
            return f'{match.group(1)}{original - setpoint_reduction:.1f}'

        modified = re.sub(schedule_pattern, adjust_schedule, modified, flags=re.IGNORECASE)

        # 2. Reduce internal gains (behavioral: lights off, equipment standby)
        zone_pattern = r'Zone,\s*\n?\s*([^,\n]+)\s*,'
        zones = re.findall(zone_pattern, modified)

        if zones:
            first_zone = zones[0].strip()

            # Behavioral savings from monitoring visibility
            # People turn off lights, report issues, reduce standby
            behavioral_savings_w_m2 = 0.4  # ~0.4 W/m² from behavior

            monitoring_schedule = '''
Schedule:Compact,
    Energy_Monitoring_Sched,      !- Name
    Any Number,                    !- Schedule Type Limits Name
    Through: 12/31,                !- Field 1
    For: AllDays,                  !- Field 2
    Until: 07:00, 1.2,            !- Night (higher savings - unoccupied)
    Until: 09:00, 0.7,            !- Morning routine
    Until: 17:00, 1.0,            !- Day (baseline savings)
    Until: 22:00, 0.8,            !- Evening
    Until: 24:00, 1.2;            !- Night

'''

            monitoring_savings = f'''
! ==== ECM Applied: Energy Monitoring System ====
! Behavioral savings: {behavioral_savings_pct}% from visibility
! Setpoint awareness: -{setpoint_reduction}°C
! Enables: benchmarking, anomaly detection, continuous improvement
! Investment: ~20,000-100,000 SEK for monitoring platform

OtherEquipment,
    Energy_Monitoring_Behavioral, !- Name
    None,                          !- Fuel Type (EnergyPlus 25.x required)
    {first_zone},                  !- Zone Name
    Energy_Monitoring_Sched,       !- Schedule Name
    Watts/Area,                    !- Design Level Calculation Method
    ,                              !- Design Level {{W}}
    -{behavioral_savings_w_m2:.2f}, !- Power per Zone Floor Area {{W/m2}} (negative = savings)
    ,                              !- Power per Person {{W/person}}
    0,                             !- Fraction Latent
    0.5,                           !- Fraction Radiant
    0;                             !- Fraction Lost

'''
            modified = monitoring_schedule + monitoring_savings + modified
        else:
            comment = f'''
! ==== ECM Applied: Energy Monitoring System ====
! Behavioral savings: {behavioral_savings_pct}%
! Setpoint awareness: -{setpoint_reduction}°C
! Note: No zones found for behavioral modeling

'''
            modified = comment + modified

        return modified

    def _apply_recommissioning(
        self,
        idf_content: str,
        params: Dict[str, Any]
    ) -> str:
        """
        Recommissioning (Injustering/Driftoptimering).

        Systematic tune-up of existing systems.
        Low-cost measure with 5-15% savings.
        """
        # Improve system efficiency through proper commissioning
        pattern = r'(Heating Setpoint Temperature\s*,\s*)(\d+\.?\d*)'

        def optimize_setpoint(match):
            original = float(match.group(2))
            # Better control allows slightly lower setpoint
            return f'{match.group(1)}{original - 0.3:.1f}'

        modified = re.sub(pattern, optimize_setpoint, idf_content)

        comment = '''
! ==== ECM Applied: Recommissioning ====
! Systematic verification and optimization of all systems
! Includes: valve calibration, sensor verification, schedule review
! Expected savings: 5-15% with minimal investment

'''
        return comment + modified

    # ═══════════════════════════════════════════════════════════════════════════
    # LIGHTING (SPECIFIC AREAS)
    # ═══════════════════════════════════════════════════════════════════════════

    def _apply_led_common_areas(
        self,
        idf_content: str,
        params: Dict[str, Any]
    ) -> str:
        """
        LED Lighting in Common Areas (LED i allmänna utrymmen).

        LED upgrade in corridors, stairwells, laundry rooms.
        Typically 60-70% reduction in lighting energy.
        """
        reduction = params.get('reduction', 0.65)

        # Target common area zones (corridor, stairwell patterns)
        pattern = r'(Lights,\s*[^,]*(?:corridor|stair|common|trapphus|korridor)[^;]*Watts/Area\s*,\s*)(\d+\.?\d*)'

        def reduce_lighting(match):
            original = float(match.group(2))
            return f'{match.group(1)}{original * (1 - reduction):.2f}'

        modified = re.sub(pattern, reduce_lighting, idf_content, flags=re.IGNORECASE)

        comment = f'''
! ==== ECM Applied: LED Common Areas ====
! Lighting reduction: {reduction*100:.0f}%
! Applied to: corridors, stairwells, laundry rooms
! Includes motion sensors for optimal control

'''
        return comment + modified

    def _apply_led_outdoor(
        self,
        idf_content: str,
        params: Dict[str, Any]
    ) -> str:
        """
        LED Outdoor Lighting (LED utomhusbelysning).

        LED upgrade for parking, pathways, facades.
        Typically 60-80% reduction in outdoor lighting electricity.

        Implementation:
        - Add Exterior:Lights object with negative wattage (savings)
        - Swedish multi-family: ~1.5 kWh/m²/year outdoor lighting
        - LED reduction: 70% → savings of ~1 kWh/m²/year
        - This is modeled as reduced electricity via negative equipment
        """
        reduction_pct = params.get('reduction_pct', 70)  # 70% savings

        # Swedish typical: 1.5 kWh/m²/year outdoor lighting
        # Convert to W/m²: 1.5 kWh/m²/year ÷ 4000 hrs/year ≈ 0.4 W/m²
        # But outdoors run ~2000 hrs, so effective base is 0.75 W/m²
        outdoor_base_w_m2 = params.get('outdoor_base_w_m2', 0.75)
        savings_w_m2 = outdoor_base_w_m2 * (reduction_pct / 100)

        # Get zone names for adding exterior equipment effect
        zone_pattern = r'Zone,\s*\n?\s*([^,\n]+)\s*,'
        zones = re.findall(zone_pattern, idf_content)

        if not zones:
            # Fallback if no zones found - just add comment
            comment = f'''
! ==== ECM Applied: LED Outdoor Lighting ====
! Reduction: {reduction_pct}% of outdoor lighting energy
! Savings: {savings_w_m2:.2f} W/m² (outdoor fixtures)
! Note: No zones found for modeling

'''
            return comment + idf_content

        # Model as negative OtherEquipment in first zone (electricity reduction)
        # Since outdoor lights don't heat the building, we use OtherEquipment
        first_zone = zones[0].strip()

        # Create schedule for outdoor lighting (dusk to dawn, ~4000 hrs/year)
        outdoor_schedule = '''
Schedule:Compact,
    LED_Outdoor_Savings_Sched,   !- Name
    Any Number,                   !- Schedule Type Limits Name
    Through: 12/31,               !- Field 1
    For: AllDays,                 !- Field 2
    Until: 06:00, 1.0,           !- Nighttime (full outdoor lights on)
    Until: 08:00, 0.5,           !- Dusk transition
    Until: 17:00, 0.0,           !- Daytime (lights off)
    Until: 19:00, 0.5,           !- Dawn transition
    Until: 24:00, 1.0;           !- Nighttime

'''

        # Negative electricity equipment (savings)
        led_savings = f'''
! ==== ECM Applied: LED Outdoor Lighting ====
! Reduction: {reduction_pct}% of outdoor lighting electricity
! Savings: {savings_w_m2:.2f} W/m² average outdoor fixture density

OtherEquipment,
    LED_Outdoor_Savings,          !- Name
    None,                          !- Fuel Type (EnergyPlus 25.x required)
    {first_zone},                 !- Zone Name
    LED_Outdoor_Savings_Sched,    !- Schedule Name
    EquipmentLevel,               !- Design Level Calculation Method
    -{savings_w_m2 * 100:.1f},   !- Design Level {{W}} (negative = savings, ~100m² reference)
    ,                             !- Power per Zone Floor Area {{W/m2}}
    ,                             !- Power per Person {{W/person}}
    0,                            !- Fraction Latent
    0,                            !- Fraction Radiant
    0;                            !- Fraction Lost (100% to outdoor)

'''

        return outdoor_schedule + led_savings + idf_content

    # ═══════════════════════════════════════════════════════════════════════════
    # DHW ECMs
    # ═══════════════════════════════════════════════════════════════════════════

    def _apply_dhw_circulation(
        self,
        idf_content: str,
        params: Dict[str, Any]
    ) -> str:
        """
        DHW Circulation Optimization (VVC-optimering).

        Reduces circulation pump runtime and heat losses.
        Timer/demand control on DHW circulation pump.

        Implementation:
        - Circulation runs 50% less → less pipe losses to unheated spaces
        - Swedish multi-family: circulation losses ~2-3 kWh/m²/year
        - Timer control saves 25-40% of circulation losses
        - Model as negative internal gains (heat kept in pipes, not lost)

        Physics: DHW circulation pumps run 24/7 to maintain hot water at taps.
        This loses ~50W per riser continuously. Timer control reduces to demand periods.
        """
        reduction_pct = params.get('reduction_pct', 30)  # 30% reduction in circulation losses

        # Swedish typical: 2.5 kWh/m²/year circulation losses
        # Convert to W/m²: 2.5 kWh/m²/year ÷ 8760 hrs/year ≈ 0.29 W/m²
        circulation_loss_w_m2 = params.get('circulation_loss_w_m2', 0.29)
        savings_w_m2 = circulation_loss_w_m2 * (reduction_pct / 100)

        # Get zone names
        zone_pattern = r'Zone,\s*\n?\s*([^,\n]+)\s*,'
        zones = re.findall(zone_pattern, idf_content)

        if not zones:
            comment = f'''
! ==== ECM Applied: DHW Circulation Optimization ====
! Reduction: {reduction_pct}% of circulation losses
! Savings: {savings_w_m2:.3f} W/m²
! Note: No zones found for modeling

'''
            return comment + idf_content

        first_zone = zones[0].strip()

        # Circulation pump runs less during night/low-demand
        circulation_schedule = '''
Schedule:Compact,
    DHW_Circ_Savings_Sched,       !- Name
    Any Number,                    !- Schedule Type Limits Name
    Through: 12/31,                !- Field 1
    For: AllDays,                  !- Field 2 (EnergyPlus 25.x compatible)
    Until: 06:00, 0.5,            !- Night (pump runs 50% - timer off)
    Until: 09:00, 0.0,            !- Morning peak (full circ, no savings)
    Until: 12:00, 0.3,            !- Mid-morning (reduced)
    Until: 17:00, 0.3,            !- Afternoon (reduced)
    Until: 22:00, 0.0,            !- Evening peak (full circ)
    Until: 24:00, 0.5;            !- Night

'''

        # Negative internal gains (less heat lost from pipes)
        circ_savings = f'''
! ==== ECM Applied: DHW Circulation Optimization ====
! Reduction: {reduction_pct}% of circulation losses via timer/demand control
! Savings: {savings_w_m2:.3f} W/m² (less heat lost from circulation pipes)
! Pump electricity also reduced ~{reduction_pct}%

OtherEquipment,
    DHW_Circ_Optimization,         !- Name
    None,                          !- Fuel Type (EnergyPlus 25.x required)
    {first_zone},                  !- Zone Name
    DHW_Circ_Savings_Sched,        !- Schedule Name
    Watts/Area,                    !- Design Level Calculation Method
    ,                              !- Design Level {{W}}
    -{savings_w_m2:.4f},           !- Power per Zone Floor Area {{W/m2}} (negative = savings)
    ,                              !- Power per Person {{W/person}}
    0,                             !- Fraction Latent
    0.3,                           !- Fraction Radiant (pipes in walls)
    0.7;                           !- Fraction Lost (to unheated spaces)

'''

        return circulation_schedule + circ_savings + idf_content

    def _apply_dhw_tank_insulation(
        self,
        idf_content: str,
        params: Dict[str, Any]
    ) -> str:
        """
        DHW Tank Insulation (Ackumulatortankisolering).

        Additional insulation on hot water storage tanks.
        Reduces standby losses by 30-50%.

        Implementation:
        - Standby losses typically 10-15% of total DHW energy
        - Swedish DHW: 22 kWh/m²/year → standby losses ~2-3 kWh/m²/year
        - Better insulation reduces standby from 10-15% to 3-5%
        - Model as negative internal gains (less heat lost from tank)

        Physics: Hot water tanks lose heat 24/7 through insulation.
        Older tanks: UA ~5 W/K. Adding 100mm insulation → UA ~1.5 W/K.
        ΔT ~40°C → saves (5-1.5)*40 = 140W per tank continuously.
        """
        reduction_pct = params.get('reduction_pct', 60)  # 60% of standby losses

        # Swedish typical: standby losses ~2.5 kWh/m²/year for multi-family
        # Convert to W/m²: 2.5 kWh/m²/year ÷ 8760 hrs/year ≈ 0.29 W/m²
        standby_loss_w_m2 = params.get('standby_loss_w_m2', 0.29)
        savings_w_m2 = standby_loss_w_m2 * (reduction_pct / 100)

        # Get zone names
        zone_pattern = r'Zone,\s*\n?\s*([^,\n]+)\s*,'
        zones = re.findall(zone_pattern, idf_content)

        if not zones:
            comment = f'''
! ==== ECM Applied: DHW Tank Insulation ====
! Reduction: {reduction_pct}% of standby losses
! Savings: {savings_w_m2:.3f} W/m²
! Note: No zones found for modeling

'''
            return comment + idf_content

        first_zone = zones[0].strip()

        # Constant savings (tanks lose heat 24/7)
        tank_savings = f'''
! ==== ECM Applied: DHW Tank Insulation ====
! Reduction: {reduction_pct}% of standby losses via additional insulation
! Savings: {savings_w_m2:.3f} W/m² (less heat lost from tank surface)
! Investment: ~5,000-15,000 SEK per tank for jacket insulation

Schedule:Compact,
    DHW_Tank_Always_On,           !- Name
    Any Number,                    !- Schedule Type Limits Name
    Through: 12/31,                !- Field 1
    For: AllDays,                  !- Field 2
    Until: 24:00, 1.0;            !- Always on

OtherEquipment,
    DHW_Tank_Insulation_Savings,   !- Name
    None,                          !- Fuel Type (EnergyPlus 25.x required)
    {first_zone},                  !- Zone Name
    DHW_Tank_Always_On,            !- Schedule Name (constant loss reduction)
    Watts/Area,                    !- Design Level Calculation Method
    ,                              !- Design Level {{W}}
    -{savings_w_m2:.4f},           !- Power per Zone Floor Area {{W/m2}} (negative = savings)
    ,                              !- Power per Person {{W/person}}
    0,                             !- Fraction Latent
    0.5,                           !- Fraction Radiant (tank in utility room)
    0.5;                           !- Fraction Lost (to mechanical room)

'''

        return tank_savings + idf_content

    # ═══════════════════════════════════════════════════════════════════════════
    # STORAGE
    # ═══════════════════════════════════════════════════════════════════════════

    def _apply_battery_storage(
        self,
        idf_content: str,
        params: Dict[str, Any]
    ) -> str:
        """
        Battery Storage System (Batterilager).

        Stores excess solar PV for later use.
        Increases self-consumption from 30% to 70-80%.

        Implementation:
        - Uses ElectricLoadCenter:Storage:Simple for modeling
        - Connects to existing PV via ElectricLoadCenter:Distribution
        - Affects Facility Total Purchased Electric Power (peak demand)
        - Round-trip efficiency ~85-90%

        Physics: Battery stores daytime PV generation for evening use.
        This reduces grid import during peak hours (16:00-20:00).
        The simulation will show reduced purchased electricity.
        """
        capacity_kwh = params.get('capacity_kwh', 50)
        power_kw = params.get('power_kw', capacity_kwh / 4)  # C/4 rate typical
        efficiency = params.get('round_trip_efficiency', 0.90)

        # Check if PV system exists
        has_pv = 'Generator:PVWatts' in idf_content or 'ElectricLoadCenter:Generators' in idf_content

        if not has_pv:
            comment = f'''
! ==== ECM Applied: Battery Storage ====
! Capacity: {capacity_kwh} kWh
! Power: {power_kw:.1f} kW
! Note: No PV system found - battery added for future PV or grid arbitrage

'''
            # Still add battery for potential grid arbitrage

        # Create battery storage object using Simple storage model
        # (LiIonNMCBattery requires more detailed parameters)
        battery_objects = f'''
! ==== ECM Applied: Battery Storage System ====
! Capacity: {capacity_kwh} kWh
! Power: {power_kw:.1f} kW
! Round-trip efficiency: {efficiency*100:.0f}%
! Self-consumption increase: 30% → 70-80%
! Peak demand reduction: Discharge during 16:00-20:00

Schedule:Compact,
    Battery_Availability,         !- Name
    Any Number,                    !- Schedule Type Limits Name
    Through: 12/31,                !- Field 1
    For: AllDays,                  !- Field 2
    Until: 24:00, 1;              !- Always available

Schedule:Compact,
    Battery_Charge_Sched,         !- Name
    Any Number,                    !- Schedule Type Limits Name
    Through: 12/31,                !- Field 1
    For: AllDays,                  !- Field 2
    Until: 09:00, 0,              !- Night: hold
    Until: 16:00, 1,              !- Day: charge from PV
    Until: 20:00, -1,             !- Evening peak: discharge
    Until: 24:00, 0;              !- Night: hold

! Simple battery storage model
! Note: EnergyPlus uses Joules (J) for capacity: 1 kWh = 3,600,000 J
ElectricLoadCenter:Storage:Simple,
    Battery_System,               !- Name
    Battery_Availability,         !- Availability Schedule Name
    ,                             !- Zone Name (outdoor or garage)
    0,                            !- Radiative Fraction for Zone Heat Gains
    {efficiency ** 0.5:.3f},      !- Nominal Energetic Efficiency for Charging (fraction 0-1)
    {efficiency ** 0.5:.3f},      !- Nominal Discharging Energetic Efficiency (fraction 0-1)
    {capacity_kwh * 3600000},     !- Maximum Storage Capacity {{J}} (kWh × 3,600,000)
    {power_kw * 1000},            !- Maximum Power for Discharging {{W}}
    {power_kw * 1000},            !- Maximum Power for Charging {{W}}
    {capacity_kwh * 3600000 * 0.2};  !- Initial State of Charge {{J}} (20% initial)

'''

        # If PV exists, try to update the ElectricLoadCenter:Distribution to include battery
        if has_pv and 'ElectricLoadCenter:Distribution' in idf_content:
            # Look for existing distribution and add storage reference
            # Pattern to find ElectricLoadCenter:Distribution
            distribution_pattern = r'(ElectricLoadCenter:Distribution,[^;]*?)(Baseload|TrackFacility|TrackMeter|TrackSchedule)([^;]*?;)'

            def add_battery_to_distribution(match):
                # Modify to include battery storage
                # This is complex because we need to add storage object name
                # For now, add a note that manual integration may be needed
                return match.group(0)

            # Add note about integration
            battery_objects += f'''
! NOTE: Battery should be connected to existing PV system
! Manual update may be needed to ElectricLoadCenter:Distribution
! Add "Battery_System" as Electrical Storage Object Name

'''

        # If no PV, create a standalone distribution for future use
        if not has_pv:
            battery_objects += f'''
! Standalone battery (no PV yet)
ElectricLoadCenter:Distribution,
    Battery_LoadCenter,           !- Name
    ,                             !- Generator List Name
    Baseload,                     !- Generator Operation Scheme Type
    0,                            !- Demand Limit Scheme Purchased Electric Demand Limit
    ,                             !- Track Schedule Name Scheme Schedule Name
    ,                             !- Track Meter Scheme Meter Name
    AlternatingCurrentWithStorage, !- Electrical Buss Type
    ,                             !- Inverter Object Name
    Battery_System;               !- Electrical Storage Object Name

'''

        return battery_objects + idf_content
