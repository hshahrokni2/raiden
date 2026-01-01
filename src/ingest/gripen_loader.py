"""
Gripen Energy Declaration Loader.

Loads Swedish government energy declarations from Boverket's Gripen database.
~831,000 unique buildings (deduplicated) with 205 fields per building.

Data source: Boverket Digital Gold export (pipe-delimited text files)
Location: ~/Dropbox/ZeldaGripen/Digtal Gold/v17 (250108)/efter_v260/data/

Record counts by year (raw, before deduplication):
- 2019: 374,752 records
- 2020: 341,884 records
- 2021: 206,231 records
- 2022: 162,455 records
- 2023: 140,711 records
- 2024: 147,880 records
Total: ~1,373,913 records → ~831,000 unique buildings

KEY INSIGHT: Same building can have multiple declarations over time.
This is VALUABLE for detecting renovations:
- Building with 2019: G (160 kWh/m²) → 2024: B (35 kWh/m²) = deep renovation!
- We keep the MOST RECENT declaration + track history for LLM reasoning.
"""

from __future__ import annotations

import csv
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any, Iterator
from functools import lru_cache

logger = logging.getLogger(__name__)

# Default path to Gripen data - use env var or None (requires explicit path)
import os
_gripen_env = os.environ.get("GRIPEN_DATA_PATH")
DEFAULT_GRIPEN_PATH = Path(_gripen_env) if _gripen_env else None


@dataclass
class GripenBuilding:
    """A building from the Gripen energy declaration database."""

    # Identifiers
    formular_id: str
    county_code: str
    county_name: str
    municipality_code: str
    municipality_name: str
    property_designation: str  # Fastighetsbeteckning
    address: str
    postal_code: str
    city: str
    is_main_address: bool

    # Building characteristics
    building_type_code: str
    building_category: str
    building_form: str  # Friliggande, etc.
    construction_year: int
    atemp_m2: float
    num_basement_floors: int
    num_floors: int
    num_staircases: int
    num_apartments: int

    # Area breakdown (% of Atemp)
    residential_area_m2: float
    hotel_area_m2: float
    restaurant_area_m2: float
    office_area_m2: float
    grocery_area_m2: float
    retail_area_m2: float
    shopping_center_area_m2: float
    healthcare_area_m2: float
    school_area_m2: float
    pool_area_m2: float
    theater_area_m2: float
    other_area_m2: float

    # Energy performance
    energy_class: str  # A, B, C, D, E, F, G
    specific_energy_kwh_m2: float
    total_energy_kwh: float
    primary_energy_kwh: float
    reference_value_1: float
    reference_value_2_max: float

    # Ventilation
    has_ftx: bool
    has_f_only: bool
    has_ft: bool
    has_natural_draft: bool
    ventilation_approved: bool
    ventilation_airflow_ls_m2: float

    # Heating (kWh by source)
    district_heating_space_kwh: float
    district_heating_hot_water_kwh: float
    oil_space_kwh: float
    gas_space_kwh: float
    wood_space_kwh: float
    pellets_space_kwh: float
    electric_direct_kwh: float
    electric_water_heater_kwh: float
    ground_source_hp_kwh: float
    exhaust_air_hp_kwh: float
    air_to_air_hp_kwh: float
    air_to_water_hp_kwh: float

    # Solar
    has_solar_thermal: bool
    has_solar_pv: bool
    solar_pv_production_kwh: float

    # Metadata
    declaration_year: int
    declaration_version: str
    approved_date: str

    # Renovation history (previous declarations for same building)
    previous_declarations: List[Dict[str, Any]] = field(default_factory=list)

    # Raw properties for full access
    raw_properties: Dict[str, Any] = field(default_factory=dict)

    @property
    def full_address(self) -> str:
        """Get formatted full address."""
        return f"{self.address}, {self.postal_code} {self.city}"

    def get_primary_heating(self) -> str:
        """Determine primary heating system."""
        heating_sources = {
            "district_heating": self.district_heating_space_kwh + self.district_heating_hot_water_kwh,
            "ground_source_hp": self.ground_source_hp_kwh,
            "exhaust_air_hp": self.exhaust_air_hp_kwh,
            "air_to_water_hp": self.air_to_water_hp_kwh,
            "electric_direct": self.electric_direct_kwh,
            "oil": self.oil_space_kwh,
            "gas": self.gas_space_kwh,
            "wood": self.wood_space_kwh,
            "pellets": self.pellets_space_kwh,
        }

        if not any(heating_sources.values()):
            return "unknown"

        return max(heating_sources, key=heating_sources.get)

    @property
    def has_district_heating(self) -> bool:
        """Check if building uses district heating."""
        return (self.district_heating_space_kwh or 0) + (self.district_heating_hot_water_kwh or 0) > 0

    @property
    def district_heating_kwh(self) -> float:
        """Total district heating consumption (space + hot water)."""
        return (self.district_heating_space_kwh or 0) + (self.district_heating_hot_water_kwh or 0)

    @property
    def has_heat_pump(self) -> bool:
        """Check if building has any heat pump."""
        return (
            (self.ground_source_hp_kwh or 0) +
            (self.exhaust_air_hp_kwh or 0) +
            (self.air_to_air_hp_kwh or 0) +
            (self.air_to_water_hp_kwh or 0)
        ) > 0

    @property
    def electric_heating_kwh(self) -> float:
        """Total electric direct heating."""
        return (self.electric_direct_kwh or 0) + (self.electric_water_heater_kwh or 0)

    @property
    def has_solar(self) -> bool:
        """Check if building has solar PV."""
        return self.has_solar_pv or (self.solar_pv_production_kwh or 0) > 0

    @property
    def solar_pv_kwh(self) -> float:
        """Solar PV production (alias for solar_pv_production_kwh)."""
        return self.solar_pv_production_kwh or 0

    @property
    def ventilation_type(self) -> str:
        """Get ventilation system type as string."""
        if self.has_ftx:
            return "FTX"
        elif self.has_ft:
            return "FT"
        elif self.has_f_only:
            return "F"
        elif self.has_natural_draft:
            return "S"
        return "unknown"

    def get_zone_breakdown(self) -> Dict[str, float]:
        """Get zone breakdown as fractions of Atemp."""
        if self.atemp_m2 <= 0:
            return {"residential": 1.0}

        breakdown = {}
        total = self.atemp_m2

        if self.residential_area_m2 > 0:
            breakdown["residential"] = self.residential_area_m2 / total
        if self.hotel_area_m2 > 0:
            breakdown["hotel"] = self.hotel_area_m2 / total
        if self.restaurant_area_m2 > 0:
            breakdown["restaurant"] = self.restaurant_area_m2 / total
        if self.office_area_m2 > 0:
            breakdown["office"] = self.office_area_m2 / total
        if self.grocery_area_m2 > 0:
            breakdown["grocery"] = self.grocery_area_m2 / total
        if self.retail_area_m2 > 0:
            breakdown["retail"] = self.retail_area_m2 / total
        if self.shopping_center_area_m2 > 0:
            breakdown["retail"] = breakdown.get("retail", 0) + self.shopping_center_area_m2 / total
        if self.healthcare_area_m2 > 0:
            breakdown["healthcare"] = self.healthcare_area_m2 / total
        if self.school_area_m2 > 0:
            breakdown["school"] = self.school_area_m2 / total
        if self.pool_area_m2 > 0:
            breakdown["pool"] = self.pool_area_m2 / total
        if self.theater_area_m2 > 0:
            breakdown["theater"] = self.theater_area_m2 / total
        if self.other_area_m2 > 0:
            breakdown["other"] = self.other_area_m2 / total

        # Default to residential if no breakdown
        if not breakdown:
            breakdown["residential"] = 1.0

        return breakdown

    def is_mixed_use(self) -> bool:
        """Check if building has multiple use types."""
        breakdown = self.get_zone_breakdown()
        return len(breakdown) > 1

    def has_renovation_history(self) -> bool:
        """Check if building has previous declarations (potential renovation)."""
        return len(self.previous_declarations) > 0

    def get_renovation_analysis(self) -> Optional[Dict[str, Any]]:
        """
        Analyze renovation history if available.

        Returns dict with:
        - is_renovated: bool
        - energy_class_improvement: int (e.g., 3 = improved 3 classes)
        - kwh_reduction_percent: float
        - original_year: int
        - original_energy_class: str
        - original_kwh_m2: float
        """
        if not self.previous_declarations:
            return None

        # Energy class ordering (A=1 best, G=7 worst)
        class_order = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7}

        # Find oldest declaration
        oldest = min(self.previous_declarations, key=lambda x: x.get('year', 9999))

        current_class = class_order.get(self.energy_class, 4)
        original_class = class_order.get(oldest.get('energy_class', ''), 4)

        class_improvement = original_class - current_class  # Positive = improved

        original_kwh = oldest.get('kwh_m2', 0)
        if original_kwh and self.specific_energy_kwh_m2:
            kwh_reduction = (original_kwh - self.specific_energy_kwh_m2) / original_kwh * 100
        else:
            kwh_reduction = 0

        return {
            'is_renovated': class_improvement >= 2 or kwh_reduction >= 30,
            'energy_class_improvement': class_improvement,
            'kwh_reduction_percent': kwh_reduction,
            'original_year': oldest.get('year'),
            'original_energy_class': oldest.get('energy_class', ''),
            'original_kwh_m2': original_kwh,
            'current_year': self.declaration_year,
            'current_energy_class': self.energy_class,
            'current_kwh_m2': self.specific_energy_kwh_m2,
        }


class GripenLoader:
    """
    Loader for Gripen energy declaration database.

    Usage:
        loader = GripenLoader()

        # Find by address
        buildings = loader.find_by_address("Kungsgatan 1", city="Stockholm")

        # Find by municipality
        buildings = loader.find_by_municipality("Stockholm")

        # Get statistics
        stats = loader.get_statistics()
    """

    # All available years in Gripen data
    AVAILABLE_YEARS = [2019, 2020, 2021, 2022, 2023, 2024]

    def __init__(
        self,
        data_path: Optional[Path] = None,
        years: Optional[List[int]] = None,
        deduplicate: bool = True,
        track_history: bool = True,
    ):
        """
        Initialize Gripen loader.

        Args:
            data_path: Path to Gripen data directory. Set via GRIPEN_DATA_PATH env var or pass explicitly.
            years: List of years to load (defaults to ALL years for deduplication)
            deduplicate: Keep only most recent declaration per building (default: True)
            track_history: Store previous declarations for renovation detection (default: True)

        Raises:
            ValueError: If no data_path provided and GRIPEN_DATA_PATH env var not set.
        """
        if data_path:
            self.data_path = Path(data_path)
        elif DEFAULT_GRIPEN_PATH:
            self.data_path = DEFAULT_GRIPEN_PATH
        else:
            raise ValueError(
                "Gripen data path not configured. Either pass data_path parameter "
                "or set GRIPEN_DATA_PATH environment variable."
            )
        self.years = years or self.AVAILABLE_YEARS  # Default to ALL years
        self.deduplicate = deduplicate
        self.track_history = track_history

        self._buildings: List[GripenBuilding] = []
        self._raw_records: Dict[str, List[Dict]] = {}  # address_key -> list of records
        self._loaded = False
        self._address_index: Dict[str, List[GripenBuilding]] = {}
        self._municipality_index: Dict[str, List[GripenBuilding]] = {}

    def _ensure_loaded(self):
        """Lazy load buildings on first access."""
        if not self._loaded:
            self._load_all()

    def _load_all(self):
        """Load all buildings from Gripen data files."""
        logger.info(f"Loading Gripen data from {self.data_path}")

        if not self.data_path.exists():
            logger.warning(f"Gripen data path does not exist: {self.data_path}")
            self._loaded = True
            return

        # Phase 1: Load all records grouped by address
        total_records = 0
        for year in sorted(self.years):  # Load chronologically
            file_path = self.data_path / f"std_uttag_efter_260_{year}.txt"
            if file_path.exists():
                count = self._load_file_to_raw(file_path, year)
                total_records += count
                logger.info(f"  {year}: {count} records")
            else:
                logger.warning(f"Gripen file not found: {file_path}")

        logger.info(f"Total raw records: {total_records}")

        # Phase 2: Deduplicate and create GripenBuilding objects
        if self.deduplicate:
            self._deduplicate_and_build()
        else:
            # Just convert all records to buildings (no dedup)
            for addr_key, records in self._raw_records.items():
                for record in records:
                    building = self._parse_record_to_building(record)
                    if building:
                        self._buildings.append(building)

        # Build indexes
        self._build_indexes()
        self._loaded = True

        logger.info(f"Loaded {len(self._buildings)} unique buildings from Gripen")

    def _load_file_to_raw(self, file_path: Path, year: int) -> int:
        """Load a file into raw records grouped by address."""
        count = 0

        try:
            # Try different encodings
            for encoding in ['latin-1', 'iso-8859-1', 'cp1252', 'utf-8']:
                try:
                    with open(file_path, 'r', encoding=encoding) as f:
                        reader = csv.DictReader(f, delimiter='|')
                        for row in reader:
                            # Create address key for grouping
                            addr = row.get('IdAdr', '').strip()
                            city = row.get('IdPostort', '').strip()
                            formular_id = row.get('FormularId', '')

                            if not addr or not city:
                                continue

                            addr_key = f"{addr.lower()}|{city.lower()}"

                            # Store raw record with year and formular_id
                            record = {
                                'year': year,
                                'formular_id': formular_id,
                                'row': dict(row),
                            }

                            if addr_key not in self._raw_records:
                                self._raw_records[addr_key] = []

                            # Avoid duplicate FormularIds
                            existing_ids = {r['formular_id'] for r in self._raw_records[addr_key]}
                            if formular_id not in existing_ids:
                                self._raw_records[addr_key].append(record)
                                count += 1

                    break  # Success, exit encoding loop
                except UnicodeDecodeError:
                    continue
        except Exception as e:
            logger.error(f"Failed to load {file_path}: {e}")

        return count

    def _deduplicate_and_build(self):
        """Deduplicate records and build GripenBuilding objects with history."""
        for addr_key, records in self._raw_records.items():
            if not records:
                continue

            # Sort by year (most recent last)
            sorted_records = sorted(records, key=lambda x: x['year'])

            # Use most recent record as the primary building
            most_recent = sorted_records[-1]
            building = self._parse_record_to_building(most_recent)

            if building and self.track_history and len(sorted_records) > 1:
                # Add previous declarations to history
                for prev_record in sorted_records[:-1]:
                    prev_row = prev_record['row']
                    try:
                        kwh_str = prev_row.get('EgiSpecifikEnergianvandning', '0')
                        kwh_val = float(kwh_str.replace(',', '.')) if kwh_str else 0
                    except (ValueError, TypeError):
                        kwh_val = 0

                    building.previous_declarations.append({
                        'year': prev_record['year'],
                        'formular_id': prev_record['formular_id'],
                        'energy_class': prev_row.get('EgiEnergiklass', ''),
                        'kwh_m2': kwh_val,
                        'approved': prev_row.get('Godkand', ''),
                    })

            if building:
                self._buildings.append(building)

    def _parse_record_to_building(self, record: Dict) -> Optional[GripenBuilding]:
        """Parse a raw record into a GripenBuilding."""
        row = record['row']
        year = record['year']
        return self._parse_row(row, year)

    def _parse_row(self, row: Dict[str, str], year: int) -> Optional[GripenBuilding]:
        """Parse a row into a GripenBuilding."""
        try:
            # Helper to safely parse float
            def safe_float(val: str, default: float = 0.0) -> float:
                if not val or val == '':
                    return default
                try:
                    # Handle Swedish decimal format (comma)
                    return float(val.replace(',', '.'))
                except (ValueError, TypeError):
                    return default

            # Helper to safely parse int
            def safe_int(val: str, default: int = 0) -> int:
                if not val or val == '':
                    return default
                try:
                    return int(float(val.replace(',', '.')))
                except (ValueError, TypeError):
                    return default

            # Helper to check yes/no
            def is_yes(val: str) -> bool:
                return val.lower() in ('ja', 'yes', '1', 'true') if val else False

            building = GripenBuilding(
                # Identifiers
                formular_id=row.get('FormularId', ''),
                county_code=row.get('IdLankod', ''),
                county_name=row.get('IdLan', ''),
                municipality_code=row.get('IdKommunkod', ''),
                municipality_name=row.get('IdKommun', ''),
                property_designation=row.get('IdFastBet', ''),
                address=row.get('IdAdr', ''),
                postal_code=row.get('IdPostnr', ''),
                city=row.get('IdPostort', ''),
                is_main_address=is_yes(row.get('IdHuvudadress', '')),

                # Building characteristics
                building_type_code=row.get('EgenTypkod', ''),
                building_category=row.get('EgenByggnadsKat', ''),
                building_form=row.get('EgenByggnadsTyp', ''),
                construction_year=safe_int(row.get('EgenNybyggAr', '')),
                atemp_m2=safe_float(row.get('EgenAtemp', '')),
                num_basement_floors=safe_int(row.get('EgenAntalKallarplan', '')),
                num_floors=safe_int(row.get('EgenAntalPlan', '')),
                num_staircases=safe_int(row.get('EgenAntalTrapphus', '')),
                num_apartments=safe_int(row.get('EgenAntalBolgh', '')),

                # Area breakdown
                residential_area_m2=safe_float(row.get('EgenAtempBostad', '')),
                hotel_area_m2=safe_float(row.get('EgenAtempHotell', '')),
                restaurant_area_m2=safe_float(row.get('EgenAtempRestaurang', '')),
                office_area_m2=safe_float(row.get('EgenAtempKontor', '')),
                grocery_area_m2=safe_float(row.get('EgenAtempLivsmedel', '')),
                retail_area_m2=safe_float(row.get('EgenAtempButik', '')),
                shopping_center_area_m2=safe_float(row.get('EgenAtempKopcentrum', '')),
                healthcare_area_m2=safe_float(row.get('EgenAtempVard', '')),
                school_area_m2=safe_float(row.get('EgenAtempSkolor', '')),
                pool_area_m2=safe_float(row.get('EgenAtempBad', '')),
                theater_area_m2=safe_float(row.get('EgenAtempTeater', '')),
                other_area_m2=safe_float(row.get('EgenAtempOvrig', '')),

                # Energy performance
                energy_class=row.get('EgiEnergiklass', ''),
                specific_energy_kwh_m2=safe_float(row.get('EgiSpecifikEnergianvandning', '')),
                total_energy_kwh=safe_float(row.get('EgiEnergianvandning', '')),
                primary_energy_kwh=safe_float(row.get('EgiPrimarenergianvandning', '')),
                reference_value_1=safe_float(row.get('EgiRefvarde1', '')),
                reference_value_2_max=safe_float(row.get('EgiRefvarde2Max', '')),

                # Ventilation
                has_ftx=is_yes(row.get('VentTypFTX', '')),
                has_f_only=is_yes(row.get('VentTypF', '')),
                has_ft=is_yes(row.get('VentTypFT', '')),
                has_natural_draft=is_yes(row.get('VentTypSjalvdrag', '')),
                ventilation_approved=is_yes(row.get('VentGruppGodkand', '')),
                ventilation_airflow_ls_m2=safe_float(row.get('EgenProjVentFlode', '')),

                # Heating
                district_heating_space_kwh=safe_float(row.get('EgiFjarrvarmeUPPV', '')),
                district_heating_hot_water_kwh=safe_float(row.get('EgiFjarrvarmeVV', '')),
                oil_space_kwh=safe_float(row.get('EgiOljaUPPV', '')),
                gas_space_kwh=safe_float(row.get('EgiGasUPPV', '')),
                wood_space_kwh=safe_float(row.get('EgiVedUPPV', '')),
                pellets_space_kwh=safe_float(row.get('EgiFlisUPPV', '')),
                electric_direct_kwh=safe_float(row.get('EgiElDirektUPPV', '')),
                electric_water_heater_kwh=safe_float(row.get('EgiElVattenUPPV', '')),
                ground_source_hp_kwh=safe_float(row.get('EgiPumpMarkUPPV', '')),
                exhaust_air_hp_kwh=safe_float(row.get('EgiPumpFranluftUPPV', '')),
                air_to_air_hp_kwh=safe_float(row.get('EgiPumpLuftLuftUPPV', '')),
                air_to_water_hp_kwh=safe_float(row.get('EgiPumpLuftVattenUPPV', '')),

                # Solar
                has_solar_thermal=is_yes(row.get('EgiSolvarme', '')),
                has_solar_pv=is_yes(row.get('EgiSolcell', '')),
                solar_pv_production_kwh=safe_float(row.get('EgiBerElProduktion', '')),

                # Metadata
                declaration_year=year,
                declaration_version=row.get('Version', ''),
                approved_date=row.get('Godkand', ''),

                # Raw properties
                raw_properties=dict(row),
            )

            return building

        except Exception as e:
            logger.warning(f"Failed to parse row: {e}")
            return None

    def _build_indexes(self):
        """Build indexes for fast lookups."""
        for building in self._buildings:
            # Address index (normalized)
            addr_key = self._normalize_address(building.address)
            if addr_key:
                if addr_key not in self._address_index:
                    self._address_index[addr_key] = []
                self._address_index[addr_key].append(building)

            # Municipality index
            muni_key = building.municipality_name.lower()
            if muni_key:
                if muni_key not in self._municipality_index:
                    self._municipality_index[muni_key] = []
                self._municipality_index[muni_key].append(building)

    def _normalize_address(self, address: str) -> str:
        """Normalize address for matching."""
        if not address:
            return ""

        # Lowercase, remove common Swedish address suffixes
        addr = address.lower().strip()
        for suffix in ['gatan', 'vägen', 'stigen', 'platsen', 'torget']:
            # Keep full word for matching
            pass
        return addr

    def find_by_address(
        self,
        address: str,
        city: Optional[str] = None,
        postal_code: Optional[str] = None,
        limit: int = 10
    ) -> List[GripenBuilding]:
        """
        Find buildings by address.

        Args:
            address: Street address to search
            city: Optional city filter
            postal_code: Optional postal code filter
            limit: Maximum results to return

        Returns:
            List of matching buildings
        """
        self._ensure_loaded()

        search_term = self._normalize_address(address)
        if not search_term:
            return []

        results = []

        # Search through address index
        for addr_key, buildings in self._address_index.items():
            if search_term in addr_key or addr_key in search_term:
                for building in buildings:
                    # Apply filters
                    if city and city.lower() not in building.city.lower():
                        continue
                    if postal_code and postal_code != building.postal_code:
                        continue
                    results.append(building)
                    if len(results) >= limit:
                        break
            if len(results) >= limit:
                break

        return results[:limit]

    def find_by_municipality(
        self,
        municipality: str,
        limit: int = 100
    ) -> List[GripenBuilding]:
        """Find buildings by municipality name."""
        self._ensure_loaded()

        muni_key = municipality.lower()

        if muni_key in self._municipality_index:
            return self._municipality_index[muni_key][:limit]

        # Fuzzy match
        results = []
        for key, buildings in self._municipality_index.items():
            if muni_key in key or key in muni_key:
                results.extend(buildings)
                if len(results) >= limit:
                    break

        return results[:limit]

    def find_by_postal_code(self, postal_code: str, limit: int = 100) -> List[GripenBuilding]:
        """Find buildings by postal code."""
        self._ensure_loaded()

        results = []
        for building in self._buildings:
            if building.postal_code == postal_code:
                results.append(building)
                if len(results) >= limit:
                    break

        return results

    def find_by_property(self, property_designation: str) -> List[GripenBuilding]:
        """
        Find ALL buildings/addresses on the same property (fastighet).

        A property can have multiple buildings, each with its own address.
        This returns all of them, which is useful for:
        - Knowing how many buildings to extract from satellite
        - Getting all entrance addresses for geocoding
        - Getting total Atemp across all buildings

        Args:
            property_designation: Fastighetsbeteckning (e.g., "STOCKHOLM SÖDERMALM 1:1")

        Returns:
            List of all buildings on this property
        """
        self._ensure_loaded()

        # Normalize for matching
        search = property_designation.upper().strip()

        results = []
        for building in self._buildings:
            if building.property_designation:
                prop = building.property_designation.upper().strip()
                if prop == search or search in prop or prop in search:
                    results.append(building)

        return results

    def get_all_addresses_for_property(self, property_designation: str) -> List[str]:
        """
        Get all unique addresses for a property.

        These addresses can be geocoded to get building locations,
        which can then be used as prompts for SAM segmentation.

        Args:
            property_designation: Fastighetsbeteckning

        Returns:
            List of unique addresses (e.g., ["Bellmansgatan 16A", "Bellmansgatan 16B"])
        """
        buildings = self.find_by_property(property_designation)

        # Get unique addresses
        addresses = set()
        for b in buildings:
            if b.address:
                full_addr = f"{b.address}, {b.city}" if b.city else b.address
                addresses.add(full_addr)

        return sorted(list(addresses))

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about loaded buildings."""
        self._ensure_loaded()

        if not self._buildings:
            return {"total_buildings": 0, "loaded": False}

        # Energy class distribution
        energy_classes = {}
        for building in self._buildings:
            ec = building.energy_class or "unknown"
            energy_classes[ec] = energy_classes.get(ec, 0) + 1

        # Municipality distribution
        municipalities = {}
        for building in self._buildings:
            muni = building.municipality_name or "unknown"
            municipalities[muni] = municipalities.get(muni, 0) + 1

        # Heating systems
        heating_systems = {}
        for building in self._buildings:
            hs = building.get_primary_heating()
            heating_systems[hs] = heating_systems.get(hs, 0) + 1

        return {
            "total_buildings": len(self._buildings),
            "loaded": True,
            "years": self.years,
            "energy_classes": energy_classes,
            "municipalities": dict(sorted(municipalities.items(), key=lambda x: -x[1])[:20]),
            "heating_systems": heating_systems,
        }

    def __len__(self) -> int:
        self._ensure_loaded()
        return len(self._buildings)

    def __iter__(self) -> Iterator[GripenBuilding]:
        self._ensure_loaded()
        return iter(self._buildings)


# Convenience functions
@lru_cache(maxsize=1)
def load_gripen(years: Optional[tuple] = None) -> GripenLoader:
    """Load Gripen database (cached)."""
    return GripenLoader(years=list(years) if years else None)


def find_gripen_building(
    address: str,
    city: Optional[str] = None,
    postal_code: Optional[str] = None,
) -> Optional[GripenBuilding]:
    """Find a building in Gripen database."""
    loader = load_gripen()
    results = loader.find_by_address(address, city=city, postal_code=postal_code, limit=1)
    return results[0] if results else None
