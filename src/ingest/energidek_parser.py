"""
Swedish Energy Declaration (Energideklaration) PDF Parser.

Extracts structured data from Boverket energy declaration PDFs.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from datetime import date


@dataclass
class VentilationData:
    """Ventilation system data from energy declaration."""
    system_type: str = ""  # FTX, F, FT, S, etc.
    designed_airflow_ls_sqm: Optional[float] = None  # l/s,m²
    measured_airflow_ls_sqm: Optional[float] = None
    inspection_date: Optional[date] = None
    notes: str = ""


@dataclass
class RadonData:
    """Radon measurement data."""
    value_bq_m3: Optional[int] = None  # Bq/m³
    measurement_date: Optional[date] = None
    status: str = ""  # GOOD (<200), WARNING (200-400), HIGH (>400)


@dataclass
class RecommendationData:
    """Energy saving recommendation from declaration."""
    description: str = ""
    annual_savings_kwh: Optional[int] = None
    cost_savings_kr: Optional[float] = None
    implementation_cost_kr: Optional[int] = None
    payback_years: Optional[float] = None


@dataclass
class EnergyDeclarationData:
    """All extracted data from energy declaration PDF."""
    # Identification
    declaration_id: str = ""
    property_name: str = ""
    organization_number: str = ""

    # Addresses (can be multiple)
    addresses: list[str] = field(default_factory=list)
    municipality: str = ""
    postal_code: str = ""

    # Building data
    construction_year: Optional[int] = None
    renovation_year: Optional[int] = None
    building_type: str = ""  # Flerbostadshus, etc.

    # Areas
    atemp_sqm: Optional[float] = None  # Heated area
    aom_sqm: Optional[float] = None  # Other heated area

    # Energy performance
    energy_class: str = ""  # A-G
    primary_energy_kwh_sqm: Optional[float] = None  # Primary energy number
    specific_energy_kwh_sqm: Optional[float] = None  # Actual use per m²
    reference_value_1: Optional[float] = None  # New building requirement
    reference_value_2: Optional[float] = None  # Similar buildings

    # Energy use breakdown (kWh/year)
    heating_kwh: Optional[float] = None
    hot_water_kwh: Optional[float] = None
    property_electricity_kwh: Optional[float] = None
    cooling_kwh: Optional[float] = None
    total_energy_kwh: Optional[float] = None

    # Heat sources
    district_heating_kwh: Optional[float] = None
    ground_source_heat_pump_kwh: Optional[float] = None
    exhaust_air_heat_pump_kwh: Optional[float] = None
    electric_heating_kwh: Optional[float] = None
    gas_kwh: Optional[float] = None
    oil_kwh: Optional[float] = None
    biofuel_kwh: Optional[float] = None

    # Solar
    solar_pv_kwh: Optional[float] = None
    solar_thermal_kwh: Optional[float] = None
    solar_area_sqm: Optional[float] = None

    # Ventilation
    ventilation: VentilationData = field(default_factory=VentilationData)

    # Indoor environment
    radon: RadonData = field(default_factory=RadonData)

    # Recommendations
    recommendations: list[RecommendationData] = field(default_factory=list)

    # Dates
    declaration_date: Optional[date] = None
    valid_until: Optional[date] = None

    # Raw text for debugging
    raw_text: str = ""


class EnergyDeclarationParser:
    """
    Parser for Swedish energy declaration PDFs.

    Uses pdfplumber for text extraction with fallback patterns
    for the standard Boverket format.
    """

    def __init__(self):
        self._patterns = self._compile_patterns()

    def _compile_patterns(self) -> dict:
        """Compile regex patterns for extraction."""
        return {
            # Energy values
            "energy_class": re.compile(r"Energiklass[:\s]*([A-G])", re.IGNORECASE),
            "primary_energy": re.compile(r"Primärenergital[:\s]*([\d,\.]+)\s*kWh", re.IGNORECASE),
            "specific_energy": re.compile(r"Specifik energianvändning[:\s]*([\d,\.]+)\s*kWh", re.IGNORECASE),
            "atemp": re.compile(r"Atemp[:\s]*([\d\s,\.]+)\s*m²", re.IGNORECASE),

            # Reference values
            "ref_value_1": re.compile(r"Referensvärde 1[:\s]*([\d,\.]+)", re.IGNORECASE),
            "ref_value_2": re.compile(r"Referensvärde 2[:\s]*([\d,\.]+)", re.IGNORECASE),

            # Ventilation
            "ventilation_airflow": re.compile(r"([\d,\.]+)\s*l/s[,\s]*m²", re.IGNORECASE),
            "ventilation_type": re.compile(r"(FTX|FT|F|S|Självdrag)", re.IGNORECASE),

            # Radon
            "radon": re.compile(r"Radon[:\s]*([\d]+)\s*Bq/m³", re.IGNORECASE),

            # Dates
            "date_pattern": re.compile(r"(\d{4}-\d{2}-\d{2})"),

            # Address patterns (Swedish format)
            "address": re.compile(r"([A-ZÅÄÖ][a-zåäö]+(?:\s+[A-Za-zåäöÅÄÖ]+)*)\s+(\d+(?:-\d+)?)", re.UNICODE),

            # Energy breakdown
            "district_heating": re.compile(r"Fjärrvärme[:\s]*([\d\s,\.]+)\s*kWh", re.IGNORECASE),
            "heat_pump": re.compile(r"Värmepump[:\s]*([\d\s,\.]+)\s*kWh", re.IGNORECASE),
            "ground_source": re.compile(r"[Bb]erg(?:värme)?[:\s]*([\d\s,\.]+)\s*kWh", re.IGNORECASE),
            "exhaust_air_hp": re.compile(r"[Ff]rånluft(?:s)?värmepump[:\s]*([\d\s,\.]+)\s*kWh", re.IGNORECASE),

            # Savings recommendation
            "savings_kwh": re.compile(r"([\d\s,\.]+)\s*kWh/år", re.IGNORECASE),
            "savings_cost": re.compile(r"([\d\s,\.]+)\s*kr/år", re.IGNORECASE),
            "implementation_cost": re.compile(r"kostnad[:\s]*([\d\s,\.]+)\s*kr", re.IGNORECASE),
        }

    def parse(self, pdf_path: Path | str) -> EnergyDeclarationData:
        """
        Parse energy declaration PDF and extract structured data.

        Args:
            pdf_path: Path to the PDF file

        Returns:
            EnergyDeclarationData with all extracted values
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        # Extract text from PDF
        text = self._extract_text(pdf_path)

        # Parse into structured data
        data = EnergyDeclarationData(raw_text=text)

        # Extract various fields
        self._extract_energy_class(text, data)
        self._extract_energy_values(text, data)
        self._extract_areas(text, data)
        self._extract_ventilation(text, data)
        self._extract_radon(text, data)
        self._extract_addresses(text, data)
        self._extract_heating_sources(text, data)
        self._extract_recommendations(text, data)
        self._extract_dates(text, data)

        return data

    def _extract_text(self, pdf_path: Path) -> str:
        """Extract text from PDF using pdfplumber."""
        try:
            import pdfplumber

            text_parts = []
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text_parts.append(page_text)

                    # Also try extracting tables
                    tables = page.extract_tables()
                    for table in tables:
                        for row in table:
                            if row:
                                row_text = " | ".join(str(cell) for cell in row if cell)
                                text_parts.append(row_text)

            return "\n".join(text_parts)

        except ImportError:
            # Fallback to PyPDF2 if pdfplumber not available
            try:
                import PyPDF2

                text_parts = []
                with open(pdf_path, "rb") as f:
                    reader = PyPDF2.PdfReader(f)
                    for page in reader.pages:
                        text = page.extract_text()
                        if text:
                            text_parts.append(text)

                return "\n".join(text_parts)

            except ImportError:
                raise ImportError(
                    "Neither pdfplumber nor PyPDF2 installed. "
                    "Install with: pip install pdfplumber"
                )

    def _parse_number(self, value: str) -> Optional[float]:
        """Parse Swedish number format (1 234,56) to float."""
        if not value:
            return None
        try:
            # Remove spaces, replace comma with dot
            cleaned = value.replace(" ", "").replace(",", ".")
            return float(cleaned)
        except ValueError:
            return None

    def _extract_energy_class(self, text: str, data: EnergyDeclarationData) -> None:
        """Extract energy class (A-G)."""
        match = self._patterns["energy_class"].search(text)
        if match:
            data.energy_class = match.group(1).upper()

    def _extract_energy_values(self, text: str, data: EnergyDeclarationData) -> None:
        """Extract energy performance values."""
        # Primary energy
        match = self._patterns["primary_energy"].search(text)
        if match:
            data.primary_energy_kwh_sqm = self._parse_number(match.group(1))

        # Specific energy
        match = self._patterns["specific_energy"].search(text)
        if match:
            data.specific_energy_kwh_sqm = self._parse_number(match.group(1))

        # Reference values
        match = self._patterns["ref_value_1"].search(text)
        if match:
            data.reference_value_1 = self._parse_number(match.group(1))

        match = self._patterns["ref_value_2"].search(text)
        if match:
            data.reference_value_2 = self._parse_number(match.group(1))

    def _extract_areas(self, text: str, data: EnergyDeclarationData) -> None:
        """Extract area values."""
        match = self._patterns["atemp"].search(text)
        if match:
            data.atemp_sqm = self._parse_number(match.group(1))

    def _extract_ventilation(self, text: str, data: EnergyDeclarationData) -> None:
        """Extract ventilation data."""
        # Airflow
        match = self._patterns["ventilation_airflow"].search(text)
        if match:
            data.ventilation.designed_airflow_ls_sqm = self._parse_number(match.group(1))

        # System type
        match = self._patterns["ventilation_type"].search(text)
        if match:
            data.ventilation.system_type = match.group(1)

    def _extract_radon(self, text: str, data: EnergyDeclarationData) -> None:
        """Extract radon measurement."""
        match = self._patterns["radon"].search(text)
        if match:
            value = int(match.group(1))
            data.radon.value_bq_m3 = value

            # Classify status
            if value < 200:
                data.radon.status = "GOOD"
            elif value < 400:
                data.radon.status = "WARNING"
            else:
                data.radon.status = "HIGH"

    def _extract_addresses(self, text: str, data: EnergyDeclarationData) -> None:
        """Extract street addresses from declaration."""
        # Common Swedish street suffixes
        street_patterns = [
            r"([A-ZÅÄÖ][a-zåäö]+(?:gatan|vägen|allén|stigen|torget|platsen))\s+(\d+(?:-\d+)?)",
            r"([A-ZÅÄÖ][a-zåäö]+\s+[Aa]llé)\s+(\d+(?:-\d+)?)",
            r"(Hammarby\s+Allé)\s+(\d+(?:-\d+)?)",
            r"(Lugnets\s+Allé)\s+(\d+(?:-\d+)?)",
            r"(Aktergatan)\s+(\d+(?:-\d+)?)",
        ]

        addresses = set()
        for pattern in street_patterns:
            for match in re.finditer(pattern, text, re.UNICODE | re.IGNORECASE):
                street = match.group(1)
                number = match.group(2)
                address = f"{street} {number}"
                addresses.add(address)

        data.addresses = sorted(list(addresses))

    def _extract_heating_sources(self, text: str, data: EnergyDeclarationData) -> None:
        """Extract heating source breakdown."""
        match = self._patterns["district_heating"].search(text)
        if match:
            data.district_heating_kwh = self._parse_number(match.group(1))

        match = self._patterns["ground_source"].search(text)
        if match:
            data.ground_source_heat_pump_kwh = self._parse_number(match.group(1))

        match = self._patterns["exhaust_air_hp"].search(text)
        if match:
            data.exhaust_air_heat_pump_kwh = self._parse_number(match.group(1))

    def _extract_recommendations(self, text: str, data: EnergyDeclarationData) -> None:
        """Extract energy saving recommendations."""
        # Look for recommendation sections
        rec_patterns = [
            r"Rekommendation[:\s]*(.+?)(?:Besparing|kWh/år)",
            r"Åtgärd[:\s]*(.+?)(?:Besparing|kWh/år)",
        ]

        for pattern in rec_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE | re.DOTALL):
                rec = RecommendationData()
                rec.description = match.group(1).strip()[:200]  # Limit length

                # Try to find associated savings
                context = text[match.start():match.end() + 200]

                savings_match = self._patterns["savings_kwh"].search(context)
                if savings_match:
                    rec.annual_savings_kwh = int(self._parse_number(savings_match.group(1)) or 0)

                cost_match = self._patterns["savings_cost"].search(context)
                if cost_match:
                    rec.cost_savings_kr = self._parse_number(cost_match.group(1))

                impl_match = self._patterns["implementation_cost"].search(context)
                if impl_match:
                    rec.implementation_cost_kr = int(self._parse_number(impl_match.group(1)) or 0)

                if rec.description:
                    data.recommendations.append(rec)

    def _extract_dates(self, text: str, data: EnergyDeclarationData) -> None:
        """Extract relevant dates."""
        dates = self._patterns["date_pattern"].findall(text)

        # Try to identify declaration date (usually the most recent)
        if dates:
            try:
                parsed_dates = []
                for d in dates:
                    parts = d.split("-")
                    if len(parts) == 3:
                        parsed_dates.append(date(int(parts[0]), int(parts[1]), int(parts[2])))

                if parsed_dates:
                    data.declaration_date = max(parsed_dates)
            except (ValueError, IndexError):
                pass


def parse_energy_declaration(pdf_path: Path | str) -> EnergyDeclarationData:
    """
    Convenience function to parse energy declaration PDF.

    Args:
        pdf_path: Path to the energy declaration PDF

    Returns:
        EnergyDeclarationData with all extracted values
    """
    parser = EnergyDeclarationParser()
    return parser.parse(pdf_path)
