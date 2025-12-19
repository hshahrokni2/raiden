"""
Results Parser - Parse EnergyPlus output files.

Extracts:
- Annual energy by end use (heating, cooling, lighting, equipment, fans)
- Peak loads
- Unmet hours
- Energy intensities (kWh/m²)
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, List
import csv
import re
import logging

logger = logging.getLogger(__name__)


@dataclass
class AnnualResults:
    """Parsed annual simulation results."""
    # Energy (kWh)
    heating_kwh: float
    cooling_kwh: float
    lighting_kwh: float
    equipment_kwh: float
    fan_kwh: float
    total_site_energy_kwh: float

    # Intensities (kWh/m²)
    floor_area_m2: float
    heating_kwh_m2: float
    cooling_kwh_m2: float
    lighting_kwh_m2: float
    equipment_kwh_m2: float
    total_electricity_kwh_m2: float

    # Peak loads (W)
    peak_heating_w: Optional[float] = None
    peak_cooling_w: Optional[float] = None

    # Comfort
    heating_unmet_hours: float = 0.0
    cooling_unmet_hours: float = 0.0

    # By zone (optional)
    zone_heating_kwh: Dict[str, float] = field(default_factory=dict)

    # Source info
    source_file: str = ""

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "energy_kwh": {
                "heating": self.heating_kwh,
                "cooling": self.cooling_kwh,
                "lighting": self.lighting_kwh,
                "equipment": self.equipment_kwh,
                "fans": self.fan_kwh,
                "total_site": self.total_site_energy_kwh,
            },
            "intensity_kwh_m2": {
                "heating": self.heating_kwh_m2,
                "cooling": self.cooling_kwh_m2,
                "lighting": self.lighting_kwh_m2,
                "equipment": self.equipment_kwh_m2,
                "total_electricity": self.total_electricity_kwh_m2,
            },
            "floor_area_m2": self.floor_area_m2,
            "peak_loads_w": {
                "heating": self.peak_heating_w,
                "cooling": self.peak_cooling_w,
            },
            "unmet_hours": {
                "heating": self.heating_unmet_hours,
                "cooling": self.cooling_unmet_hours,
            },
        }


class ResultsParser:
    """
    Parse EnergyPlus simulation results.

    Supports:
    - eplustbl.csv (tabular summary - preferred)
    - eplusout.eso (time series data)
    - eplusout.eio (building information)

    Usage:
        parser = ResultsParser()
        results = parser.parse(output_dir=Path('./output'))

        print(f"Heating: {results.heating_kwh_m2:.1f} kWh/m²")
    """

    def __init__(self):
        pass

    def parse(self, output_dir: Path) -> Optional[AnnualResults]:
        """
        Parse results from EnergyPlus output directory.

        Args:
            output_dir: Directory containing eplusout.* files

        Returns:
            AnnualResults if successful, None if parsing failed
        """
        output_dir = Path(output_dir)

        # Try to parse from eplustbl.csv (tabular results) - preferred
        tbl_path = output_dir / 'eplustbl.csv'
        if tbl_path.exists():
            results = self._parse_tabular(tbl_path)
            if results:
                return results

        # Fallback to ESO parsing
        eso_path = output_dir / 'eplusout.eso'
        if eso_path.exists():
            return self._parse_eso(eso_path, output_dir)

        return None

    def _parse_tabular(self, tbl_path: Path) -> Optional[AnnualResults]:
        """
        Parse results from eplustbl.csv.

        The CSV has multiple sections separated by headers.
        Key sections:
        - Building Area: floor area
        - End Uses: energy by category and fuel type
        - Normalized Metrics: energy intensities
        """
        try:
            with open(tbl_path, 'r') as f:
                content = f.read()

            lines = content.split('\n')

            # Initialize values
            floor_area = 0.0
            heating = 0.0
            cooling = 0.0
            lighting = 0.0
            equipment = 0.0
            fans = 0.0

            # Parse Building Area section
            floor_area = self._extract_building_area(lines)

            # Parse End Uses section
            end_uses = self._extract_end_uses(lines)
            if end_uses:
                heating = end_uses.get('heating', 0.0)
                cooling = end_uses.get('cooling', 0.0)
                lighting = end_uses.get('lighting', 0.0)
                equipment = end_uses.get('equipment', 0.0)
                fans = end_uses.get('fans', 0.0)

            # Calculate totals
            total_electricity = lighting + equipment + fans
            total_site = heating + cooling + total_electricity

            # Calculate intensities
            area = floor_area if floor_area > 0 else 1.0
            heating_intensity = heating / area
            cooling_intensity = cooling / area
            lighting_intensity = lighting / area
            equipment_intensity = equipment / area
            electricity_intensity = total_electricity / area

            # Parse unmet hours
            unmet = self._extract_unmet_hours(lines)

            return AnnualResults(
                heating_kwh=heating,
                cooling_kwh=cooling,
                lighting_kwh=lighting,
                equipment_kwh=equipment,
                fan_kwh=fans,
                total_site_energy_kwh=total_site,
                floor_area_m2=floor_area,
                heating_kwh_m2=heating_intensity,
                cooling_kwh_m2=cooling_intensity,
                lighting_kwh_m2=lighting_intensity,
                equipment_kwh_m2=equipment_intensity,
                total_electricity_kwh_m2=electricity_intensity,
                heating_unmet_hours=unmet.get('heating', 0.0),
                cooling_unmet_hours=unmet.get('cooling', 0.0),
                source_file=str(tbl_path),
            )

        except Exception as e:
            logger.error(f"Error parsing {tbl_path}: {e}", exc_info=True)
            return None

    def _extract_building_area(self, lines: List[str]) -> float:
        """Extract building floor area from CSV."""
        # Simple approach: just look for the Total Building Area line anywhere
        for line in lines:
            if 'Total Building Area' in line and 'Energy Per' not in line:
                parts = line.split(',')
                # Format: ,Total Building Area,2240.00
                # Value is at index 2
                if len(parts) >= 3:
                    try:
                        val = float(parts[2].strip())
                        if val > 0:
                            return val
                    except ValueError:
                        pass
                # Also try iterating all parts as fallback
                for part in parts:
                    try:
                        val = float(part.strip())
                        if val > 0:
                            return val
                    except ValueError:
                        continue

        return 0.0

    def _extract_end_uses(self, lines: List[str]) -> Dict[str, float]:
        """Extract end use energy from CSV."""
        end_uses = {
            'heating': 0.0,
            'cooling': 0.0,
            'lighting': 0.0,
            'equipment': 0.0,
            'fans': 0.0,
        }

        in_end_uses = False
        header_indices = {}

        for i, line in enumerate(lines):
            # Find "End Uses" section (but not "End Uses By Subcategory")
            if line.strip() == 'End Uses':
                in_end_uses = True
                continue

            # End of section - blank line after we've found data
            if in_end_uses and line.strip() == '':
                if any(v > 0 for v in end_uses.values()):
                    break
                continue

            # End section on new report section
            if in_end_uses and ('End Uses By' in line or 'REPORT:' in line):
                break

            if in_end_uses:
                parts = [p.strip() for p in line.split(',')]

                # Parse header row to find column indices
                if 'Electricity [kWh]' in line:
                    for j, part in enumerate(parts):
                        part_lower = part.lower()
                        if 'electricity' in part_lower and 'kwh' in part_lower:
                            header_indices['electricity'] = j
                        elif 'district heating water' in part_lower:
                            header_indices['district_heating'] = j
                        elif 'district heating steam' in part_lower:
                            header_indices['district_steam'] = j
                        elif 'natural gas' in part_lower:
                            header_indices['gas'] = j
                        elif 'district cooling' in part_lower:
                            header_indices['district_cooling'] = j
                    continue

                # Parse data rows - they start with comma (empty first field)
                if len(parts) > 2 and parts[0] == '' and parts[1]:
                    row_type = parts[1].lower()

                    # Get energy values from correct columns
                    elec_idx = header_indices.get('electricity', 2)
                    elec_val = self._safe_float(parts, elec_idx)

                    # District heating (water or steam)
                    dh_idx = header_indices.get('district_heating', 13)
                    dh_val = self._safe_float(parts, dh_idx)

                    dhs_idx = header_indices.get('district_steam', 14)
                    dhs_val = self._safe_float(parts, dhs_idx)

                    # District cooling
                    dc_idx = header_indices.get('district_cooling', 12)
                    dc_val = self._safe_float(parts, dc_idx)

                    # Natural gas
                    gas_idx = header_indices.get('gas', 3)
                    gas_val = self._safe_float(parts, gas_idx)

                    # Assign to categories based on row type
                    if row_type == 'heating':
                        end_uses['heating'] = dh_val + dhs_val + gas_val + elec_val
                    elif row_type == 'cooling':
                        end_uses['cooling'] = elec_val + dc_val
                    elif row_type == 'interior lighting':
                        end_uses['lighting'] = elec_val
                    elif row_type == 'interior equipment':
                        end_uses['equipment'] = elec_val
                    elif row_type == 'fans':
                        end_uses['fans'] = elec_val

        return end_uses

    def _extract_unmet_hours(self, lines: List[str]) -> Dict[str, float]:
        """Extract unmet load hours."""
        unmet = {'heating': 0.0, 'cooling': 0.0}

        for line in lines:
            lower = line.lower()
            if 'unmet' in lower and 'hour' in lower:
                parts = line.split(',')
                for j, part in enumerate(parts):
                    try:
                        val = float(part.strip())
                        if 'heating' in lower:
                            unmet['heating'] = val
                        elif 'cooling' in lower:
                            unmet['cooling'] = val
                        break
                    except ValueError:
                        continue

        return unmet

    def _safe_float(self, parts: List[str], idx: int) -> float:
        """Safely extract float from list."""
        try:
            if idx < len(parts):
                return float(parts[idx])
        except (ValueError, IndexError):
            pass
        return 0.0

    def _parse_eso(self, eso_path: Path, output_dir: Path) -> Optional[AnnualResults]:
        """
        Parse results from eplusout.eso (time series).

        ESO format:
        - Header section with variable definitions
        - Data section with timestep values

        This is more complex but gives detailed time series data.
        """
        try:
            # First, try to get floor area from EIO
            floor_area = self._parse_floor_area_from_eio(output_dir / 'eplusout.eio')

            # Initialize accumulators
            heating_j = 0.0
            cooling_j = 0.0
            lighting_j = 0.0
            equipment_j = 0.0

            # Variable ID mappings
            var_ids = {}

            with open(eso_path, 'r') as f:
                in_header = True

                for line in f:
                    line = line.strip()

                    # End of header
                    if line == 'End of Data Dictionary':
                        in_header = False
                        continue

                    if in_header:
                        # Parse variable definitions
                        # Format: id,num_vals,key,var_name,unit,frequency
                        parts = line.split(',')
                        if len(parts) >= 4:
                            try:
                                var_id = int(parts[0])
                                var_name = parts[3].lower() if len(parts) > 3 else ''

                                if 'heating energy' in var_name:
                                    var_ids[var_id] = 'heating'
                                elif 'cooling energy' in var_name:
                                    var_ids[var_id] = 'cooling'
                                elif 'lights' in var_name and 'electric' in var_name:
                                    var_ids[var_id] = 'lighting'
                                elif 'equipment' in var_name and 'electric' in var_name:
                                    var_ids[var_id] = 'equipment'
                            except ValueError:
                                continue
                    else:
                        # Parse data
                        # Format: id,value (for regular timestep data)
                        parts = line.split(',')
                        if len(parts) >= 2:
                            try:
                                var_id = int(parts[0])
                                value = float(parts[1])

                                var_type = var_ids.get(var_id)
                                if var_type == 'heating':
                                    heating_j += value
                                elif var_type == 'cooling':
                                    cooling_j += value
                                elif var_type == 'lighting':
                                    lighting_j += value
                                elif var_type == 'equipment':
                                    equipment_j += value
                            except ValueError:
                                continue

            # Convert J to kWh
            j_to_kwh = 1 / 3600000

            heating = heating_j * j_to_kwh
            cooling = cooling_j * j_to_kwh
            lighting = lighting_j * j_to_kwh
            equipment = equipment_j * j_to_kwh

            total_electricity = lighting + equipment
            total_site = heating + cooling + total_electricity

            area = floor_area if floor_area > 0 else 1.0

            return AnnualResults(
                heating_kwh=heating,
                cooling_kwh=cooling,
                lighting_kwh=lighting,
                equipment_kwh=equipment,
                fan_kwh=0.0,
                total_site_energy_kwh=total_site,
                floor_area_m2=floor_area,
                heating_kwh_m2=heating / area,
                cooling_kwh_m2=cooling / area,
                lighting_kwh_m2=lighting / area,
                equipment_kwh_m2=equipment / area,
                total_electricity_kwh_m2=total_electricity / area,
                source_file=str(eso_path),
            )

        except Exception as e:
            logger.error(f"Error parsing ESO {eso_path}: {e}", exc_info=True)
            return None

    def _parse_floor_area_from_eio(self, eio_path: Path) -> float:
        """Parse floor area from eplusout.eio."""
        floor_area = 0.0

        if not eio_path.exists():
            return floor_area

        try:
            with open(eio_path, 'r') as f:
                for line in f:
                    if 'Zone Information' in line and 'Floor Area' in line:
                        # Parse zone floor areas and sum
                        continue
                    if line.startswith(' Zone Information,'):
                        parts = line.split(',')
                        # Zone floor area is typically at index 9
                        for j, part in enumerate(parts):
                            if j > 5:  # Skip header fields
                                try:
                                    val = float(part.strip())
                                    if val > 10:  # Reasonable floor area
                                        floor_area += val
                                        break
                                except ValueError:
                                    continue
        except Exception:
            pass

        return floor_area


def parse_results(output_dir: Path) -> Optional[AnnualResults]:
    """
    Convenience function to parse simulation results.

    Args:
        output_dir: EnergyPlus output directory

    Returns:
        AnnualResults or None
    """
    parser = ResultsParser()
    return parser.parse(output_dir)
