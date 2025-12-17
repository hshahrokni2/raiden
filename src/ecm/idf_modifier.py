"""
IDF Modifier - Apply ECM changes to EnergyPlus model.

Takes a baseline IDF and modifies it to implement ECMs:
- Change material properties (U-values)
- Modify HVAC parameters (heat recovery)
- Add PV generation
- Adjust schedules and loads
"""

from pathlib import Path
from typing import Dict, Any, List
import re

from .combinations import ECMVariant, ECMCombination


class IDFModifier:
    """
    Modify EnergyPlus IDF to implement ECMs.

    Usage:
        modifier = IDFModifier()
        modified_path = modifier.apply(
            baseline_idf=Path('./baseline.idf'),
            combination=ecm_combination,
            output_dir=Path('./scenarios')
        )
    """

    def __init__(self):
        pass

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

    def _apply_variant(self, idf_content: str, variant: ECMVariant) -> str:
        """Apply a single ECM variant to IDF content."""
        ecm_id = variant.ecm.id
        params = variant.parameters

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
        elif ecm_id == "demand_controlled_ventilation":
            return self._apply_dcv(idf_content, params)
        elif ecm_id == "solar_pv":
            return self._apply_solar_pv(idf_content, params)
        elif ecm_id == "led_lighting":
            return self._apply_led_lighting(idf_content, params)
        elif ecm_id == "smart_thermostats":
            return self._apply_smart_thermostats(idf_content, params)
        else:
            # Unknown ECM, return unchanged
            return idf_content

    def _apply_wall_insulation(
        self,
        idf_content: str,
        params: Dict[str, Any],
        external: bool
    ) -> str:
        """
        Add insulation to wall construction.

        Modifies the Material object for wall insulation to increase thickness.
        """
        thickness_mm = params.get('thickness_mm', 100)
        thickness_m = thickness_mm / 1000

        # TODO: Implement
        # 1. Find wall construction object
        # 2. Add new insulation layer (or increase existing)
        # 3. Update thermal properties
        raise NotImplementedError("Implement wall insulation modification")

    def _apply_roof_insulation(
        self,
        idf_content: str,
        params: Dict[str, Any]
    ) -> str:
        """Add insulation to roof construction."""
        thickness_mm = params.get('thickness_mm', 150)

        # TODO: Implement
        raise NotImplementedError("Implement roof insulation modification")

    def _apply_window_replacement(
        self,
        idf_content: str,
        params: Dict[str, Any]
    ) -> str:
        """
        Replace windows with specified U-value.

        Modifies WindowMaterial:SimpleGlazingSystem objects.
        """
        u_value = params.get('u_value', 1.0)
        shgc = params.get('shgc', 0.5)

        # Find and replace window glazing properties
        pattern = r'(WindowMaterial:SimpleGlazingSystem,\s*[^,]+,\s*)[\d.]+(\s*,\s*)[\d.]+'
        replacement = rf'\g<1>{u_value}\g<2>{shgc}'

        return re.sub(pattern, replacement, idf_content)

    def _apply_air_sealing(
        self,
        idf_content: str,
        params: Dict[str, Any]
    ) -> str:
        """
        Reduce infiltration rate.

        Modifies ZoneInfiltration:DesignFlowRate objects.
        """
        reduction_factor = params.get('reduction_factor', 0.5)

        # Find infiltration objects and multiply ACH by reduction factor
        # Pattern: Air Changes per Hour field
        def reduce_ach(match):
            current_ach = float(match.group(1))
            new_ach = current_ach * reduction_factor
            return f'{new_ach:.4f}'

        pattern = r'(\d+\.?\d*)\s*;\s*!-\s*Air Changes per Hour'
        return re.sub(pattern, lambda m: f'{float(m.group(1)) * reduction_factor:.4f}; !- Air Changes per Hour', idf_content)

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

        # Find and replace heat recovery effectiveness
        pattern = r'(Sensible,\s*!-\s*Heat Recovery Type\s*)[\d.]+'
        replacement = rf'\g<1>{effectiveness}'

        return re.sub(pattern, replacement, idf_content)

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
        idf_content = re.sub(
            r'None,\s*!-\s*Heat Recovery Type',
            f'Sensible,                        !- Heat Recovery Type',
            idf_content
        )

        # Set effectiveness
        idf_content = re.sub(
            r'[\d.]+;\s*!-\s*Sensible Heat Recovery Effectiveness',
            f'{effectiveness};                         !- Sensible Heat Recovery Effectiveness',
            idf_content
        )

        return idf_content

    def _apply_dcv(
        self,
        idf_content: str,
        params: Dict[str, Any]
    ) -> str:
        """Apply demand-controlled ventilation (modify OA specification)."""
        # TODO: Implement - modify Design Specification Outdoor Air
        raise NotImplementedError("Implement DCV modification")

    def _apply_solar_pv(
        self,
        idf_content: str,
        params: Dict[str, Any]
    ) -> str:
        """
        Add PV generation.

        Note: For IdealLoadsAirSystem models, PV doesn't directly affect
        thermal simulation. We track PV generation separately in results.
        """
        # For now, just add a comment noting PV was added
        # Actual generation calculated in post-processing
        coverage = params.get('coverage_fraction', 0.7)
        efficiency = params.get('panel_efficiency', 0.20)

        comment = f'''
! ECM: Solar PV Added
! Coverage: {coverage*100:.0f}%
! Panel efficiency: {efficiency*100:.0f}%
! Generation calculated in post-processing
'''
        return comment + idf_content

    def _apply_led_lighting(
        self,
        idf_content: str,
        params: Dict[str, Any]
    ) -> str:
        """
        Replace lighting with LED.

        Modifies Lights objects to reduce W/m².
        """
        new_power_density = params.get('power_density', 6)

        # Find Watts per Zone Floor Area and replace
        pattern = r'(\d+\.?\d*)\s*,\s*!-\s*Watts per Zone Floor Area'
        replacement = f'{new_power_density},                        !- Watts per Zone Floor Area'

        return re.sub(pattern, replacement, idf_content)

    def _apply_smart_thermostats(
        self,
        idf_content: str,
        params: Dict[str, Any]
    ) -> str:
        """
        Apply smart thermostat settings.

        Modifies heating setpoint schedule for night setback.
        """
        setback_c = params.get('setback_c', 2)

        # TODO: Implement - modify Schedule:Compact for heating setpoint
        # Add night setback period
        raise NotImplementedError("Implement smart thermostat schedule modification")
