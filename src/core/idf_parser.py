"""
Structured IDF Parser using eppy.

Replaces fragile regex-based parsing with proper object-model access.
Provides type-safe getters/setters for common calibration parameters.
"""

from pathlib import Path
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
import tempfile
import logging

from eppy.modeleditor import IDF

logger = logging.getLogger(__name__)


@dataclass
class InfiltrationData:
    """Data for a ZoneInfiltration:DesignFlowRate object."""
    name: str
    zone_name: str
    schedule_name: str
    calculation_method: str
    air_changes_per_hour: Optional[float]
    flow_per_zone_area: Optional[float]
    flow_per_exterior_area: Optional[float]
    design_flow_rate: Optional[float]


@dataclass
class WindowGlazingData:
    """Data for a WindowMaterial:SimpleGlazingSystem object."""
    name: str
    u_factor: float
    shgc: float
    visible_transmittance: Optional[float] = None


@dataclass
class IdealLoadsData:
    """Data for a ZoneHVAC:IdealLoadsAirSystem object."""
    name: str
    zone_name: str
    heat_recovery_type: str
    sensible_heat_recovery_effectiveness: float
    latent_heat_recovery_effectiveness: float


class IDFParser:
    """
    Structured IDF parser using eppy.

    Provides clean API for reading and modifying EnergyPlus IDF files
    without fragile regex patterns.

    Usage:
        parser = IDFParser()
        idf = parser.load(Path("model.idf"))

        # Get parameters
        ach = parser.get_infiltration_ach(idf)
        u_value = parser.get_window_u_value(idf)
        hr_eff = parser.get_heat_recovery_effectiveness(idf)

        # Set parameters
        parser.set_infiltration_ach(idf, 0.04)
        parser.set_window_u_value(idf, 0.9)
        parser.set_heat_recovery_effectiveness(idf, 0.80)

        # Save
        parser.save(idf, Path("modified.idf"))
    """

    _idd_set = False

    @classmethod
    def _ensure_idd(cls):
        """Ensure IDD is set (only needed once per process)."""
        if not cls._idd_set:
            try:
                # eppy needs the IDD file location
                # Try to find it from EnergyPlus installation
                import shutil

                # Common EnergyPlus installation paths
                possible_paths = [
                    "/usr/local/EnergyPlus-25-1-0/Energy+.idd",
                    "/Applications/EnergyPlus-25-1-0/Energy+.idd",
                    "C:\\EnergyPlusV25-1-0\\Energy+.idd",
                    "/usr/local/EnergyPlus-24-2-0/Energy+.idd",
                    "/Applications/EnergyPlus-24-2-0/Energy+.idd",
                ]

                # Also check if energyplus command exists and find IDD from there
                ep_path = shutil.which("energyplus")
                if ep_path:
                    ep_dir = Path(ep_path).parent
                    possible_paths.insert(0, str(ep_dir / "Energy+.idd"))

                idd_path = None
                for path in possible_paths:
                    if Path(path).exists():
                        idd_path = path
                        break

                if idd_path:
                    IDF.setiddname(idd_path)
                    cls._idd_set = True
                    logger.debug(f"Using IDD from: {idd_path}")
                else:
                    logger.warning("EnergyPlus IDD not found, eppy features limited")

            except Exception as e:
                logger.warning(f"Could not set IDD: {e}")

    def load(self, idf_path: Path) -> IDF:
        """
        Load an IDF file.

        Args:
            idf_path: Path to IDF file

        Returns:
            Loaded IDF object
        """
        self._ensure_idd()
        return IDF(str(idf_path))

    def load_string(self, idf_content: str) -> IDF:
        """
        Load IDF from string content.

        Args:
            idf_content: IDF file content as string

        Returns:
            Loaded IDF object
        """
        self._ensure_idd()
        # eppy requires a file path, so write to temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.idf', delete=False) as f:
            f.write(idf_content)
            temp_path = f.name

        try:
            return IDF(temp_path)
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def save(self, idf: IDF, output_path: Path) -> None:
        """
        Save IDF to file.

        Args:
            idf: IDF object to save
            output_path: Path to save to
        """
        idf.saveas(str(output_path))

    def to_string(self, idf: IDF) -> str:
        """
        Convert IDF to string.

        Args:
            idf: IDF object

        Returns:
            IDF content as string
        """
        return idf.idfstr()

    # ========== Infiltration ==========

    def get_infiltration_objects(self, idf: IDF) -> List[Any]:
        """Get all ZoneInfiltration:DesignFlowRate objects."""
        return idf.idfobjects['ZoneInfiltration:DesignFlowRate']

    def get_infiltration_ach(self, idf: IDF) -> Optional[float]:
        """
        Get the infiltration rate in ACH.

        Returns the first infiltration object's ACH value, or None if not found.
        Most models use the same ACH for all zones.
        """
        infiltrations = self.get_infiltration_objects(idf)
        if not infiltrations:
            return None

        # Get the first one (typically all zones have same ACH)
        infil = infiltrations[0]

        # Check calculation method
        method = getattr(infil, 'Design_Flow_Rate_Calculation_Method', '')
        if method == 'AirChanges/Hour':
            return getattr(infil, 'Air_Changes_per_Hour', None)

        return None

    def set_infiltration_ach(self, idf: IDF, ach: float) -> int:
        """
        Set infiltration rate for all zones.

        Args:
            idf: IDF object
            ach: Air changes per hour

        Returns:
            Number of objects modified
        """
        infiltrations = self.get_infiltration_objects(idf)
        count = 0

        for infil in infiltrations:
            method = getattr(infil, 'Design_Flow_Rate_Calculation_Method', '')
            if method == 'AirChanges/Hour':
                infil.Air_Changes_per_Hour = ach
                count += 1

        logger.debug(f"Set infiltration ACH to {ach} for {count} zones")
        return count

    def get_all_infiltration_data(self, idf: IDF) -> List[InfiltrationData]:
        """Get detailed data for all infiltration objects."""
        infiltrations = self.get_infiltration_objects(idf)
        result = []

        for infil in infiltrations:
            data = InfiltrationData(
                name=getattr(infil, 'Name', ''),
                zone_name=getattr(infil, 'Zone_or_ZoneList_or_Space_or_SpaceList_Name', ''),
                schedule_name=getattr(infil, 'Schedule_Name', ''),
                calculation_method=getattr(infil, 'Design_Flow_Rate_Calculation_Method', ''),
                air_changes_per_hour=getattr(infil, 'Air_Changes_per_Hour', None),
                flow_per_zone_area=getattr(infil, 'Flow_per_Zone_Floor_Area', None),
                flow_per_exterior_area=getattr(infil, 'Flow_per_Exterior_Surface_Area', None),
                design_flow_rate=getattr(infil, 'Design_Flow_Rate', None),
            )
            result.append(data)

        return result

    # ========== Windows ==========

    def get_simple_glazing_objects(self, idf: IDF) -> List[Any]:
        """Get all WindowMaterial:SimpleGlazingSystem objects."""
        return idf.idfobjects['WindowMaterial:SimpleGlazingSystem']

    def get_window_u_value(self, idf: IDF) -> Optional[float]:
        """
        Get window U-value from SimpleGlazingSystem.

        Returns the first glazing's U-factor, or None if not found.
        """
        glazings = self.get_simple_glazing_objects(idf)
        if not glazings:
            return None

        return getattr(glazings[0], 'UFactor', None)

    def set_window_u_value(self, idf: IDF, u_value: float, shgc: Optional[float] = None) -> int:
        """
        Set window U-value for all SimpleGlazingSystem materials.

        Args:
            idf: IDF object
            u_value: U-Factor in W/m²K
            shgc: Optional new SHGC value

        Returns:
            Number of objects modified
        """
        glazings = self.get_simple_glazing_objects(idf)
        count = 0

        for glazing in glazings:
            glazing.UFactor = u_value
            if shgc is not None:
                glazing.Solar_Heat_Gain_Coefficient = shgc
            count += 1

        logger.debug(f"Set window U-value to {u_value} for {count} glazing materials")
        return count

    def get_all_glazing_data(self, idf: IDF) -> List[WindowGlazingData]:
        """Get detailed data for all simple glazing objects."""
        glazings = self.get_simple_glazing_objects(idf)
        result = []

        for glazing in glazings:
            data = WindowGlazingData(
                name=getattr(glazing, 'Name', ''),
                u_factor=getattr(glazing, 'UFactor', 0),
                shgc=getattr(glazing, 'Solar_Heat_Gain_Coefficient', 0),
                visible_transmittance=getattr(glazing, 'Visible_Transmittance', None),
            )
            result.append(data)

        return result

    # ========== Heat Recovery ==========

    def get_ideal_loads_objects(self, idf: IDF) -> List[Any]:
        """Get all ZoneHVAC:IdealLoadsAirSystem objects."""
        return idf.idfobjects['ZoneHVAC:IdealLoadsAirSystem']

    def get_heat_recovery_effectiveness(self, idf: IDF) -> Optional[float]:
        """
        Get sensible heat recovery effectiveness.

        Returns the first IdealLoadsAirSystem's HR effectiveness, or None.
        """
        ideal_loads = self.get_ideal_loads_objects(idf)
        if not ideal_loads:
            return None

        system = ideal_loads[0]
        hr_type = getattr(system, 'Heat_Recovery_Type', 'None')

        if hr_type != 'None':
            return getattr(system, 'Sensible_Heat_Recovery_Effectiveness', None)

        return None

    def set_heat_recovery_effectiveness(self, idf: IDF, effectiveness: float) -> int:
        """
        Set sensible heat recovery effectiveness for all IdealLoadsAirSystem.

        Args:
            idf: IDF object
            effectiveness: Heat recovery effectiveness (0-1)

        Returns:
            Number of objects modified
        """
        ideal_loads = self.get_ideal_loads_objects(idf)
        count = 0

        for system in ideal_loads:
            hr_type = getattr(system, 'Heat_Recovery_Type', 'None')
            if hr_type != 'None':
                system.Sensible_Heat_Recovery_Effectiveness = effectiveness
                count += 1

        logger.debug(f"Set HR effectiveness to {effectiveness} for {count} systems")
        return count

    def enable_heat_recovery(self, idf: IDF, effectiveness: float = 0.75) -> int:
        """
        Enable heat recovery on all IdealLoadsAirSystem.

        Args:
            idf: IDF object
            effectiveness: Heat recovery effectiveness

        Returns:
            Number of systems modified
        """
        ideal_loads = self.get_ideal_loads_objects(idf)
        count = 0

        for system in ideal_loads:
            system.Heat_Recovery_Type = 'Sensible'
            system.Sensible_Heat_Recovery_Effectiveness = effectiveness
            count += 1

        logger.debug(f"Enabled HR with effectiveness {effectiveness} for {count} systems")
        return count

    def get_all_ideal_loads_data(self, idf: IDF) -> List[IdealLoadsData]:
        """Get detailed data for all IdealLoadsAirSystem objects."""
        ideal_loads = self.get_ideal_loads_objects(idf)
        result = []

        for system in ideal_loads:
            data = IdealLoadsData(
                name=getattr(system, 'Name', ''),
                zone_name=getattr(system, 'Zone_Supply_Air_Node_Name', '').replace('_Supply', ''),
                heat_recovery_type=getattr(system, 'Heat_Recovery_Type', 'None'),
                sensible_heat_recovery_effectiveness=getattr(
                    system, 'Sensible_Heat_Recovery_Effectiveness', 0
                ),
                latent_heat_recovery_effectiveness=getattr(
                    system, 'Latent_Heat_Recovery_Effectiveness', 0
                ),
            )
            result.append(data)

        return result

    # ========== Materials ==========

    def get_material_objects(self, idf: IDF) -> List[Any]:
        """Get all Material objects."""
        return idf.idfobjects['Material']

    def get_material_by_name(self, idf: IDF, name: str) -> Optional[Any]:
        """Get a material by name."""
        materials = self.get_material_objects(idf)
        for mat in materials:
            if getattr(mat, 'Name', '') == name:
                return mat
        return None

    def set_material_thickness(self, idf: IDF, name: str, thickness: float) -> bool:
        """
        Set material thickness.

        Args:
            idf: IDF object
            name: Material name
            thickness: New thickness in meters

        Returns:
            True if material was found and modified
        """
        material = self.get_material_by_name(idf, name)
        if material:
            material.Thickness = thickness
            logger.debug(f"Set thickness of {name} to {thickness}m")
            return True
        return False

    def add_material_thickness(self, idf: IDF, name: str, additional: float) -> bool:
        """
        Add thickness to existing material.

        Args:
            idf: IDF object
            name: Material name
            additional: Additional thickness in meters

        Returns:
            True if material was found and modified
        """
        material = self.get_material_by_name(idf, name)
        if material:
            current = getattr(material, 'Thickness', 0)
            material.Thickness = current + additional
            logger.debug(f"Increased thickness of {name} from {current}m to {material.Thickness}m")
            return True
        return False

    # ========== Lights ==========

    def get_lights_objects(self, idf: IDF) -> List[Any]:
        """Get all Lights objects."""
        return idf.idfobjects['Lights']

    def set_lighting_power_density(self, idf: IDF, watts_per_m2: float) -> int:
        """
        Set lighting power density for all zones.

        Args:
            idf: IDF object
            watts_per_m2: Power density in W/m²

        Returns:
            Number of lights objects modified
        """
        lights = self.get_lights_objects(idf)
        count = 0

        for light in lights:
            method = getattr(light, 'Design_Level_Calculation_Method', '')
            if method == 'Watts/Area':
                # eppy field name is 'Watts_per_Floor_Area' not 'Watts_per_Zone_Floor_Area'
                light.Watts_per_Floor_Area = watts_per_m2
                count += 1

        logger.debug(f"Set lighting power density to {watts_per_m2} W/m² for {count} zones")
        return count

    # ========== Utility Methods ==========

    def get_building_name(self, idf: IDF) -> Optional[str]:
        """Get the building name."""
        buildings = idf.idfobjects['Building']
        if buildings:
            return getattr(buildings[0], 'Name', None)
        return None

    def get_zone_count(self, idf: IDF) -> int:
        """Get the number of zones."""
        return len(idf.idfobjects['Zone'])

    def get_zone_names(self, idf: IDF) -> List[str]:
        """Get all zone names."""
        zones = idf.idfobjects['Zone']
        return [getattr(z, 'Name', '') for z in zones]

    def extract_calibration_parameters(self, idf: IDF) -> Dict[str, float]:
        """
        Extract all key calibration parameters.

        Returns dict with:
        - infiltration: ACH
        - heat_recovery: effectiveness
        - window_u: U-value
        """
        return {
            'infiltration': self.get_infiltration_ach(idf) or 0.06,
            'heat_recovery': self.get_heat_recovery_effectiveness(idf) or 0.75,
            'window_u': self.get_window_u_value(idf) or 1.0,
        }

    def apply_calibration_parameters(
        self,
        idf: IDF,
        params: Dict[str, float]
    ) -> Dict[str, int]:
        """
        Apply calibration parameters to IDF.

        Args:
            idf: IDF object
            params: Dict with infiltration, heat_recovery, window_u

        Returns:
            Dict with count of objects modified for each parameter
        """
        results = {}

        if 'infiltration' in params:
            results['infiltration'] = self.set_infiltration_ach(idf, params['infiltration'])

        if 'heat_recovery' in params:
            results['heat_recovery'] = self.set_heat_recovery_effectiveness(
                idf, params['heat_recovery']
            )

        if 'window_u' in params:
            results['window_u'] = self.set_window_u_value(idf, params['window_u'])

        return results


# Module-level convenience instance
_parser = None

def get_parser() -> IDFParser:
    """Get the module-level parser instance."""
    global _parser
    if _parser is None:
        _parser = IDFParser()
    return _parser
