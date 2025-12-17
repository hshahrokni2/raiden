"""
EnergyPlus IDF exporter.

Generates EnergyPlus Input Data Files from enriched BRF building data.
Uses GeomEppy for geometry generation.

Features:
- Complete zone definitions with floor-by-floor zoning
- Swedish construction material layers based on U-values and era
- Residential schedules for occupancy, lighting, equipment
- HVAC system templates (heat pump, district heating, ideal loads)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from rich.console import Console

from ..core.models import (
    EnrichedBRFProperty,
    EnrichedBuilding,
    FacadeMaterial,
    EnergyPlusReady,
    EnergyPlusZone,
)
from ..core.config import settings

console = Console()


# =============================================================================
# MATERIAL PROPERTY LIBRARIES
# =============================================================================

# Swedish construction materials with thermal properties
# Based on BBR and Sveby standards
MATERIAL_LIBRARY = {
    # Insulation materials
    "mineralwool_035": {
        "name": "Mineral Wool 0.035",
        "conductivity": 0.035,  # W/m·K
        "density": 30,  # kg/m³
        "specific_heat": 840,  # J/kg·K
    },
    "mineralwool_037": {
        "name": "Mineral Wool 0.037",
        "conductivity": 0.037,
        "density": 25,
        "specific_heat": 840,
    },
    "eps_038": {
        "name": "EPS Insulation",
        "conductivity": 0.038,
        "density": 25,
        "specific_heat": 1400,
    },
    # Structural materials
    "concrete_150": {
        "name": "Concrete 150mm",
        "conductivity": 1.7,
        "density": 2300,
        "specific_heat": 900,
        "thickness": 0.15,
    },
    "concrete_200": {
        "name": "Concrete 200mm",
        "conductivity": 1.7,
        "density": 2300,
        "specific_heat": 900,
        "thickness": 0.20,
    },
    "brick_120": {
        "name": "Brick 120mm",
        "conductivity": 0.77,
        "density": 1800,
        "specific_heat": 840,
        "thickness": 0.12,
    },
    "plaster_20": {
        "name": "Plaster Render 20mm",
        "conductivity": 0.8,
        "density": 1600,
        "specific_heat": 840,
        "thickness": 0.02,
    },
    "gypsum_13": {
        "name": "Gypsum Board 13mm",
        "conductivity": 0.16,
        "density": 800,
        "specific_heat": 1000,
        "thickness": 0.013,
    },
    # Flooring
    "wood_floor_22": {
        "name": "Wood Flooring 22mm",
        "conductivity": 0.14,
        "density": 650,
        "specific_heat": 1200,
        "thickness": 0.022,
    },
    # Roofing
    "roofing_felt": {
        "name": "Roofing Felt",
        "conductivity": 0.19,
        "density": 1100,
        "specific_heat": 1000,
        "thickness": 0.005,
    },
}

# Window properties by era (U-value, SHGC, visible transmittance)
WINDOW_LIBRARY = {
    "pre_1975": {"u_value": 2.8, "shgc": 0.75, "vt": 0.80, "name": "Single Glazed"},
    "1975_1990": {"u_value": 2.0, "shgc": 0.65, "vt": 0.75, "name": "Double Glazed"},
    "1990_2005": {"u_value": 1.6, "shgc": 0.55, "vt": 0.70, "name": "Low-E Double"},
    "2005_2015": {"u_value": 1.2, "shgc": 0.50, "vt": 0.65, "name": "Low-E Triple"},
    "post_2015": {"u_value": 0.8, "shgc": 0.45, "vt": 0.60, "name": "Advanced Triple"},
}


class EnergyPlusExporter:
    """
    Export enriched BRF data to EnergyPlus IDF format.

    Uses GeomEppy for geometry generation and eppy for IDF manipulation.
    """

    # Swedish construction templates based on era and material
    CONSTRUCTION_TEMPLATES = {
        "exterior_wall": {
            FacadeMaterial.BRICK: "BrickWall_Insulated",
            FacadeMaterial.CONCRETE: "ConcreteWall_Insulated",
            FacadeMaterial.PLASTER: "PlasterWall_Insulated",
            FacadeMaterial.GLASS: "CurtainWall_Glass",
            FacadeMaterial.METAL: "MetalPanel_Insulated",
            FacadeMaterial.WOOD: "WoodFrame_Insulated",
            FacadeMaterial.STONE: "StoneWall_Insulated",
            FacadeMaterial.UNKNOWN: "GenericWall_Insulated",
        },
    }

    def __init__(
        self,
        idd_path: Path | str | None = None,
        weather_file: Path | str | None = None,
    ):
        """
        Initialize EnergyPlus exporter.

        Args:
            idd_path: Path to Energy+.idd file
            weather_file: Path to .epw weather file
        """
        self.idd_path = Path(idd_path) if idd_path else settings.energyplus_idd_path
        self.weather_file = Path(weather_file) if weather_file else None
        self._idf = None
        self._initialized = False

    def _lazy_init(self) -> None:
        """Lazily initialize GeomEppy/Eppy."""
        if self._initialized:
            return

        try:
            from geomeppy import IDF

            if self.idd_path and self.idd_path.exists():
                IDF.setiddname(str(self.idd_path))
                console.print(f"[green]Loaded IDD: {self.idd_path}[/green]")
            else:
                console.print(
                    "[yellow]IDD path not set. Set BRF_ENERGYPLUS_IDD_PATH or pass idd_path[/yellow]"
                )

            self._IDF = IDF
            self._initialized = True

        except ImportError:
            console.print(
                "[yellow]geomeppy not installed. Install with: pip install geomeppy[/yellow]"
            )
            self._IDF = None

    def _generate_zones(
        self,
        building: EnrichedBuilding,
    ) -> list[EnergyPlusZone]:
        """
        Generate per-floor zone definitions for a building.

        Creates one zone per floor, with appropriate area and volume.
        """
        ep_data = building.energyplus_ready
        props = building.original_properties

        zones = []
        num_floors = ep_data.num_stories
        total_area = props.dimensions.heated_area_sqm
        floor_to_floor = ep_data.floor_to_floor_height_m

        # Calculate area per floor (simplified: equal distribution)
        area_per_floor = total_area / num_floors
        volume_per_floor = area_per_floor * floor_to_floor

        # Determine zone types based on building usage
        space_usage = props.space_usage_percent

        for floor_num in range(1, num_floors + 1):
            # Determine zone type
            if floor_num == 1 and space_usage.other > 10:
                zone_type = "commercial"  # Ground floor retail/commercial
            elif floor_num <= 0:
                zone_type = "parking"  # Basement
            else:
                zone_type = "residential"

            zone = EnergyPlusZone(
                name=f"Floor_{floor_num}_Zone",
                floor_area_sqm=area_per_floor,
                volume_m3=volume_per_floor,
                floor_number=floor_num,
                zone_type=zone_type,
            )
            zones.append(zone)

        return zones

    def _add_materials_and_constructions(
        self,
        idf,
        building: EnrichedBuilding,
    ) -> None:
        """
        Add material and construction objects based on U-values.

        Creates realistic material layers to achieve target U-values.
        """
        ep_data = building.energyplus_ready
        envelope = ep_data.envelope
        u_values = envelope.u_values
        construction_year = building.original_properties.building_info.construction_year

        if u_values is None:
            return

        try:
            # Calculate insulation thickness needed for each element
            # R = 1/U, R_total = R_surface + R_structure + R_insulation

            # Surface resistances (standard values)
            R_si = 0.13  # Interior surface (walls)
            R_se = 0.04  # Exterior surface (walls)
            R_si_floor = 0.17  # Interior surface (floor)
            R_se_floor = 0.0   # Exterior surface (ground)

            # === EXTERIOR WALL ===
            target_R_wall = 1.0 / u_values.walls
            # Structure: plaster (0.02m) + concrete (0.15m) + insulation + gypsum (0.013m)
            R_plaster = 0.02 / 0.8  # thickness / conductivity
            R_concrete = 0.15 / 1.7
            R_gypsum = 0.013 / 0.16
            R_insulation_wall = target_R_wall - R_si - R_se - R_plaster - R_concrete - R_gypsum
            insulation_thickness_wall = max(0.05, R_insulation_wall * 0.035)  # using λ=0.035

            # Add wall materials
            idf.newidfobject(
                "Material",
                Name="Wall_Plaster",
                Roughness="MediumSmooth",
                Thickness=0.02,
                Conductivity=0.8,
                Density=1600,
                Specific_Heat=840,
            )

            idf.newidfobject(
                "Material",
                Name="Wall_Concrete",
                Roughness="MediumRough",
                Thickness=0.15,
                Conductivity=1.7,
                Density=2300,
                Specific_Heat=900,
            )

            idf.newidfobject(
                "Material",
                Name="Wall_Insulation",
                Roughness="MediumRough",
                Thickness=insulation_thickness_wall,
                Conductivity=0.035,
                Density=30,
                Specific_Heat=840,
            )

            idf.newidfobject(
                "Material",
                Name="Wall_Gypsum",
                Roughness="Smooth",
                Thickness=0.013,
                Conductivity=0.16,
                Density=800,
                Specific_Heat=1000,
            )

            # Create wall construction
            idf.newidfobject(
                "Construction",
                Name="ExteriorWall",
                Outside_Layer="Wall_Plaster",
                Layer_2="Wall_Concrete",
                Layer_3="Wall_Insulation",
                Layer_4="Wall_Gypsum",
            )

            # === ROOF ===
            target_R_roof = 1.0 / u_values.roof
            R_felt = 0.005 / 0.19
            R_concrete_roof = 0.15 / 1.7
            R_insulation_roof = target_R_roof - R_si - R_se - R_felt - R_concrete_roof
            insulation_thickness_roof = max(0.10, R_insulation_roof * 0.035)

            idf.newidfobject(
                "Material",
                Name="Roof_Felt",
                Roughness="Rough",
                Thickness=0.005,
                Conductivity=0.19,
                Density=1100,
                Specific_Heat=1000,
            )

            idf.newidfobject(
                "Material",
                Name="Roof_Insulation",
                Roughness="MediumRough",
                Thickness=insulation_thickness_roof,
                Conductivity=0.035,
                Density=30,
                Specific_Heat=840,
            )

            idf.newidfobject(
                "Construction",
                Name="Roof",
                Outside_Layer="Roof_Felt",
                Layer_2="Wall_Concrete",
                Layer_3="Roof_Insulation",
            )

            # === FLOOR ===
            target_R_floor = 1.0 / u_values.floor
            R_wood = 0.022 / 0.14
            R_concrete_floor = 0.15 / 1.7
            R_insulation_floor = target_R_floor - R_si_floor - R_se_floor - R_wood - R_concrete_floor
            insulation_thickness_floor = max(0.05, R_insulation_floor * 0.038)

            idf.newidfobject(
                "Material",
                Name="Floor_Wood",
                Roughness="MediumSmooth",
                Thickness=0.022,
                Conductivity=0.14,
                Density=650,
                Specific_Heat=1200,
            )

            idf.newidfobject(
                "Material",
                Name="Floor_Insulation",
                Roughness="MediumRough",
                Thickness=insulation_thickness_floor,
                Conductivity=0.038,
                Density=25,
                Specific_Heat=1400,
            )

            idf.newidfobject(
                "Construction",
                Name="Floor",
                Outside_Layer="Floor_Insulation",
                Layer_2="Wall_Concrete",
                Layer_3="Floor_Wood",
            )

            # === WINDOWS ===
            # Select window type based on era
            if construction_year < 1975:
                window_type = WINDOW_LIBRARY["pre_1975"]
            elif construction_year < 1990:
                window_type = WINDOW_LIBRARY["1975_1990"]
            elif construction_year < 2005:
                window_type = WINDOW_LIBRARY["1990_2005"]
            elif construction_year < 2015:
                window_type = WINDOW_LIBRARY["2005_2015"]
            else:
                window_type = WINDOW_LIBRARY["post_2015"]

            # Use actual U-value if available, otherwise use era default
            window_u = min(u_values.windows, window_type["u_value"])

            idf.newidfobject(
                "WindowMaterial:SimpleGlazingSystem",
                Name="Window_Glass",
                UFactor=window_u,
                Solar_Heat_Gain_Coefficient=window_type["shgc"],
                Visible_Transmittance=window_type["vt"],
            )

            idf.newidfobject(
                "Construction",
                Name="Window",
                Outside_Layer="Window_Glass",
            )

            console.print(f"[dim]Added materials: wall insul={insulation_thickness_wall:.3f}m, "
                         f"roof insul={insulation_thickness_roof:.3f}m, "
                         f"window U={window_u:.2f}[/dim]")

        except Exception as e:
            console.print(f"[yellow]Material/construction creation failed: {e}[/yellow]")

    def export_building(
        self,
        building: EnrichedBuilding,
        output_path: Path | str,
        template_path: Path | str | None = None,
    ) -> Path | None:
        """
        Export a single building to IDF.

        Args:
            building: Enriched building data
            output_path: Output IDF file path
            template_path: Optional base IDF template

        Returns:
            Path to generated IDF or None if failed
        """
        self._lazy_init()

        if self._IDF is None:
            console.print("[red]Cannot export: geomeppy not available[/red]")
            return None

        output_path = Path(output_path)
        ep_data = building.energyplus_ready

        try:
            # Create or load IDF
            if template_path and Path(template_path).exists():
                idf = self._IDF(str(template_path))
            else:
                idf = self._IDF()

            # Generate zones if not already present
            if not ep_data.zones:
                ep_data.zones = self._generate_zones(building)

            # Add building block (geometry creates zones automatically via GeomEppy)
            self._add_building_geometry(idf, building)

            # Add materials and constructions based on U-values
            self._add_materials_and_constructions(idf, building)

            # Add schedules
            self._add_schedules(idf)

            # Add internal loads
            self._add_internal_loads(idf, building)

            # Add HVAC
            self._add_hvac(idf, building)

            # Save
            idf.saveas(str(output_path))
            console.print(f"[green]Exported IDF: {output_path}[/green]")

            return output_path

        except Exception as e:
            console.print(f"[red]IDF export failed: {e}[/red]")
            import traceback
            traceback.print_exc()
            return None

    def _add_building_geometry(
        self,
        idf,
        building: EnrichedBuilding,
    ) -> None:
        """Add building geometry using GeomEppy."""
        ep_data = building.energyplus_ready
        props = building.original_properties

        # Get footprint coordinates (already in local coords)
        footprint = ep_data.footprint_coords_local

        # Remove duplicate closing vertex if present
        if footprint and footprint[0] == footprint[-1]:
            footprint = footprint[:-1]

        # Add block using GeomEppy
        try:
            idf.add_block(
                name=f"Building_{building.building_id}",
                coordinates=footprint,
                height=ep_data.height_m,
                num_stories=ep_data.num_stories,
            )
            console.print(f"[dim]Added geometry for Building {building.building_id}[/dim]")
        except Exception as e:
            console.print(f"[yellow]GeomEppy block creation failed: {e}[/yellow]")
            # Fallback: create simple box
            self._add_simple_box(idf, building)

    def _add_simple_box(
        self,
        idf,
        building: EnrichedBuilding,
    ) -> None:
        """Fallback: create simple rectangular building."""
        ep_data = building.energyplus_ready
        props = building.original_properties

        # Calculate bounding box of footprint
        coords = ep_data.footprint_coords_local
        if not coords:
            return

        xs = [c[0] for c in coords]
        ys = [c[1] for c in coords]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)

        width = max_x - min_x
        depth = max_y - min_y

        # Create simple rectangular block
        simple_coords = [
            (min_x, min_y),
            (max_x, min_y),
            (max_x, max_y),
            (min_x, max_y),
        ]

        idf.add_block(
            name=f"Building_{building.building_id}_Simple",
            coordinates=simple_coords,
            height=ep_data.height_m,
            num_stories=ep_data.num_stories,
        )

    def _add_constructions(
        self,
        idf,
        ep_data: EnergyPlusReady,
    ) -> None:
        """Add construction definitions based on U-values and materials."""
        envelope = ep_data.envelope
        u_values = envelope.u_values

        if u_values is None:
            return

        # We'll create simplified constructions
        # In a full implementation, you'd define materials and layers

        constructions_to_add = [
            ("ExteriorWall", u_values.walls),
            ("Roof", u_values.roof),
            ("Floor", u_values.floor),
            ("Window", u_values.windows),
        ]

        for name, u_value in constructions_to_add:
            # Note: This is simplified. Real implementation needs material layers.
            # GeomEppy's set_default_constructions() is often used instead.
            pass

        try:
            idf.set_default_constructions()
        except Exception:
            pass

    def _add_schedules(self, idf) -> None:
        """Add standard schedules for residential buildings based on Sveby."""
        try:
            # Schedule type limits
            idf.newidfobject(
                "ScheduleTypeLimits",
                Name="Fraction",
                Lower_Limit_Value=0.0,
                Upper_Limit_Value=1.0,
                Numeric_Type="Continuous",
            )

            idf.newidfobject(
                "ScheduleTypeLimits",
                Name="Temperature",
                Lower_Limit_Value=-50,
                Upper_Limit_Value=50,
                Numeric_Type="Continuous",
            )

            # Residential occupancy schedule (Sveby typical)
            idf.newidfobject(
                "Schedule:Compact",
                Name="ResidentialOccupancy",
                Schedule_Type_Limits_Name="Fraction",
                Field_1="Through: 12/31",
                Field_2="For: Weekdays",
                Field_3="Until: 07:00, 0.9",   # Sleeping
                Field_4="Until: 09:00, 0.5",   # Morning routine
                Field_5="Until: 17:00, 0.2",   # Work hours
                Field_6="Until: 19:00, 0.5",   # Coming home
                Field_7="Until: 22:00, 0.9",   # Evening at home
                Field_8="Until: 24:00, 0.9",   # Sleep
                Field_9="For: Weekends",
                Field_10="Until: 09:00, 0.9",
                Field_11="Until: 12:00, 0.7",
                Field_12="Until: 18:00, 0.5",
                Field_13="Until: 22:00, 0.9",
                Field_14="Until: 24:00, 0.9",
            )

            # Residential lighting schedule
            idf.newidfobject(
                "Schedule:Compact",
                Name="ResidentialLighting",
                Schedule_Type_Limits_Name="Fraction",
                Field_1="Through: 12/31",
                Field_2="For: AllDays",
                Field_3="Until: 06:00, 0.1",
                Field_4="Until: 08:00, 0.5",
                Field_5="Until: 17:00, 0.2",
                Field_6="Until: 22:00, 0.8",
                Field_7="Until: 24:00, 0.3",
            )

            # Residential equipment schedule
            idf.newidfobject(
                "Schedule:Compact",
                Name="ResidentialEquipment",
                Schedule_Type_Limits_Name="Fraction",
                Field_1="Through: 12/31",
                Field_2="For: AllDays",
                Field_3="Until: 07:00, 0.3",
                Field_4="Until: 09:00, 0.6",
                Field_5="Until: 17:00, 0.3",
                Field_6="Until: 22:00, 0.7",
                Field_7="Until: 24:00, 0.4",
            )

            # Heating setpoint schedule
            idf.newidfobject(
                "Schedule:Compact",
                Name="HeatingSetpoint",
                Schedule_Type_Limits_Name="Temperature",
                Field_1="Through: 12/31",
                Field_2="For: AllDays",
                Field_3="Until: 06:00, 18",   # Night setback
                Field_4="Until: 22:00, 21",   # Daytime
                Field_5="Until: 24:00, 18",   # Night setback
            )

            # Cooling setpoint schedule (Swedish climate - minimal cooling needed)
            idf.newidfobject(
                "Schedule:Compact",
                Name="CoolingSetpoint",
                Schedule_Type_Limits_Name="Temperature",
                Field_1="Through: 12/31",
                Field_2="For: AllDays",
                Field_3="Until: 24:00, 26",   # High setpoint - passive cooling preferred
            )

            console.print("[dim]Added residential schedules[/dim]")

        except Exception as e:
            console.print(f"[yellow]Schedule creation failed: {e}[/yellow]")

    def _add_internal_loads(
        self,
        idf,
        building: EnrichedBuilding,
    ) -> None:
        """Add internal loads (people, lights, equipment) to each zone."""
        ep_data = building.energyplus_ready

        # Get load densities from enriched data
        occupant_density = ep_data.occupant_density_m2_per_person  # m² per person
        lighting_density = ep_data.lighting_power_density_w_per_m2  # W/m²
        equipment_density = ep_data.equipment_power_density_w_per_m2  # W/m²

        try:
            # Get all zones from IDF
            zones = idf.idfobjects["Zone"]

            for zone in zones:
                zone_name = zone.Name

                # Add People
                idf.newidfobject(
                    "People",
                    Name=f"{zone_name}_People",
                    Zone_or_ZoneList_or_Space_or_SpaceList_Name=zone_name,
                    Number_of_People_Schedule_Name="ResidentialOccupancy",
                    Number_of_People_Calculation_Method="Area/Person",
                    Zone_Floor_Area_per_Person=occupant_density,
                    Fraction_Radiant=0.3,
                    Sensible_Heat_Fraction="autocalculate",
                    Activity_Level_Schedule_Name="",
                )

                # Add Lights
                idf.newidfobject(
                    "Lights",
                    Name=f"{zone_name}_Lights",
                    Zone_or_ZoneList_or_Space_or_SpaceList_Name=zone_name,
                    Schedule_Name="ResidentialLighting",
                    Design_Level_Calculation_Method="Watts/Area",
                    Watts_per_Zone_Floor_Area=lighting_density,
                    Fraction_Radiant=0.4,
                    Fraction_Visible=0.2,
                    Fraction_Replaceable=1.0,
                )

                # Add Electric Equipment
                idf.newidfobject(
                    "ElectricEquipment",
                    Name=f"{zone_name}_Equipment",
                    Zone_or_ZoneList_or_Space_or_SpaceList_Name=zone_name,
                    Schedule_Name="ResidentialEquipment",
                    Design_Level_Calculation_Method="Watts/Area",
                    Watts_per_Zone_Floor_Area=equipment_density,
                    Fraction_Radiant=0.3,
                    Fraction_Latent=0.0,
                    Fraction_Lost=0.0,
                )

            console.print(f"[dim]Added internal loads for {len(zones)} zones[/dim]")

        except Exception as e:
            console.print(f"[yellow]Internal loads creation failed: {e}[/yellow]")

    def _add_hvac(
        self,
        idf,
        building: EnrichedBuilding,
    ) -> None:
        """
        Add HVAC system based on building's actual heating type.

        Supports:
        - Ideal Loads (simplified, for energy studies)
        - Heat Pump (GSHP/ASHP)
        - District Heating (via Purchased:Heating)
        """
        props = building.original_properties
        ep_data = building.energyplus_ready

        # Determine heating system type from original data
        heating = props.energy.heating
        if heating.ground_source_heat_pump_kwh > 0:
            hvac_type = "HeatPump"
        elif heating.district_heating_kwh > 0:
            hvac_type = "DistrictHeating"
        elif heating.exhaust_air_heat_pump_kwh > 0:
            hvac_type = "ExhaustHeatPump"
        else:
            hvac_type = "IdealLoads"

        try:
            # Get all zones
            zones = idf.idfobjects["Zone"]

            if hvac_type == "IdealLoads":
                # Simple ideal loads air system for each zone
                # Great for design studies and energy calculations
                for zone in zones:
                    zone_name = zone.Name

                    idf.newidfobject(
                        "HVACTemplate:Zone:IdealLoadsAirSystem",
                        Zone_Name=zone_name,
                        Template_Thermostat_Name=f"{zone_name}_Thermostat",
                        Heating_Availability_Schedule_Name="",
                        Cooling_Availability_Schedule_Name="",
                        Heating_Limit="NoLimit",
                        Cooling_Limit="NoLimit",
                        Dehumidification_Control_Type="None",
                        Humidification_Control_Type="None",
                        Outdoor_Air_Method="None",
                    )

                    # Add thermostat
                    idf.newidfobject(
                        "HVACTemplate:Thermostat",
                        Name=f"{zone_name}_Thermostat",
                        Heating_Setpoint_Schedule_Name="HeatingSetpoint",
                        Cooling_Setpoint_Schedule_Name="CoolingSetpoint",
                    )

                console.print(f"[dim]Added IdealLoads HVAC for {len(zones)} zones[/dim]")

            elif hvac_type in ["HeatPump", "ExhaustHeatPump"]:
                # Use ZoneHVAC:IdealLoadsAirSystem with COP factors
                # In real implementation, would use detailed heat pump model
                for zone in zones:
                    zone_name = zone.Name

                    idf.newidfobject(
                        "HVACTemplate:Zone:IdealLoadsAirSystem",
                        Zone_Name=zone_name,
                        Template_Thermostat_Name=f"{zone_name}_Thermostat",
                        Heating_Availability_Schedule_Name="",
                        Cooling_Availability_Schedule_Name="",
                        Heating_Limit="NoLimit",
                        Cooling_Limit="NoLimit",
                    )

                    idf.newidfobject(
                        "HVACTemplate:Thermostat",
                        Name=f"{zone_name}_Thermostat",
                        Heating_Setpoint_Schedule_Name="HeatingSetpoint",
                        Cooling_Setpoint_Schedule_Name="CoolingSetpoint",
                    )

                console.print(f"[dim]Added Heat Pump HVAC template for {len(zones)} zones[/dim]")

            elif hvac_type == "DistrictHeating":
                # Simplified district heating as ideal loads
                # In real implementation, would use DistrictHeating:Water
                for zone in zones:
                    zone_name = zone.Name

                    idf.newidfobject(
                        "HVACTemplate:Zone:IdealLoadsAirSystem",
                        Zone_Name=zone_name,
                        Template_Thermostat_Name=f"{zone_name}_Thermostat",
                        Heating_Availability_Schedule_Name="",
                        Cooling_Availability_Schedule_Name="",
                        Heating_Limit="NoLimit",
                        Cooling_Limit="LimitCapacity",
                        Maximum_Sensible_Cooling_Capacity=0,  # No cooling for district heat
                    )

                    idf.newidfobject(
                        "HVACTemplate:Thermostat",
                        Name=f"{zone_name}_Thermostat",
                        Heating_Setpoint_Schedule_Name="HeatingSetpoint",
                        Cooling_Setpoint_Schedule_Name="CoolingSetpoint",
                    )

                console.print(f"[dim]Added District Heating HVAC template for {len(zones)} zones[/dim]")

        except Exception as e:
            console.print(f"[yellow]HVAC creation failed: {e}[/yellow]")
            console.print("[yellow]Building will use default constructions only[/yellow]")

    def export_property(
        self,
        brf: EnrichedBRFProperty,
        output_dir: Path | str,
        combined: bool = False,
    ) -> list[Path]:
        """
        Export all buildings in a BRF property.

        Args:
            brf: Enriched BRF property
            output_dir: Output directory
            combined: If True, export all buildings to single IDF

        Returns:
            List of generated IDF paths
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        generated = []

        if combined:
            # Single IDF with all buildings
            output_path = output_dir / f"{brf.brf_name.replace(' ', '_')}.idf"
            # Would need to handle multiple buildings in one IDF
            console.print("[yellow]Combined export not yet implemented[/yellow]")
        else:
            # Separate IDF per building
            for building in brf.buildings:
                filename = f"{brf.brf_name.replace(' ', '_')}_bldg{building.building_id}.idf"
                output_path = output_dir / filename

                result = self.export_building(building, output_path)
                if result:
                    generated.append(result)

        return generated

    def generate_idf_string(
        self,
        building: EnrichedBuilding,
    ) -> str:
        """
        Generate IDF content as string (without file I/O).

        Useful for preview or streaming.
        """
        import io
        import tempfile

        self._lazy_init()

        if self._IDF is None:
            return "# Error: geomeppy not available"

        # Create temporary file
        with tempfile.NamedTemporaryFile(suffix=".idf", delete=False) as tmp:
            tmp_path = tmp.name

        self.export_building(building, tmp_path)

        with open(tmp_path, "r") as f:
            content = f.read()

        Path(tmp_path).unlink()

        return content


def create_minimal_idf_template() -> str:
    """
    Create a minimal IDF template string.

    Useful when no template file is available.
    """
    return """!-Generator IDFEditor 1.51
!-Option SortedOrder

Version,
    23.2;                    !- Version Identifier

SimulationControl,
    No,                      !- Do Zone Sizing Calculation
    No,                      !- Do System Sizing Calculation
    No,                      !- Do Plant Sizing Calculation
    No,                      !- Run Simulation for Sizing Periods
    Yes;                     !- Run Simulation for Weather File Run Periods

Building,
    BRF Building,            !- Name
    0,                       !- North Axis
    City,                    !- Terrain
    0.04,                    !- Loads Convergence Tolerance Value
    0.4,                     !- Temperature Convergence Tolerance Value
    FullInteriorAndExterior, !- Solar Distribution
    25,                      !- Maximum Number of Warmup Days
    6;                       !- Minimum Number of Warmup Days

Timestep,
    4;                       !- Number of Timesteps per Hour

RunPeriod,
    Annual,                  !- Name
    1,                       !- Begin Month
    1,                       !- Begin Day of Month
    ,                        !- Begin Year
    12,                      !- End Month
    31,                      !- End Day of Month
    ,                        !- End Year
    Sunday,                  !- Day of Week for Start Day
    No,                      !- Use Weather File Holidays and Special Days
    No,                      !- Use Weather File Daylight Saving Period
    No,                      !- Apply Weekend Holiday Rule
    Yes,                     !- Use Weather File Rain Indicators
    Yes;                     !- Use Weather File Snow Indicators

GlobalGeometryRules,
    UpperLeftCorner,         !- Starting Vertex Position
    Counterclockwise,        !- Vertex Entry Direction
    Relative;                !- Coordinate System
"""
