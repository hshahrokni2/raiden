"""
Swedish HVAC System Templates for EnergyPlus.

Based on:
- BeBo (Beställargruppen Bostäder) documentation
- Sveby 2.0 system definitions
- Swedish building codes (BBR)
- Energimyndigheten statistics

Each template generates complete EnergyPlus IDF objects for a specific
Swedish HVAC system type, including:
- Plant loops (hot water)
- Air loops (ventilation)
- Zone equipment (radiators, etc.)
- Controls and setpoints
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import logging

from .performance_curves import (
    HeatPumpType,
    PerformanceCurve,
    get_heat_pump_performance,
    GSHP_HEATING_CAPACITY_CURVE,
    GSHP_HEATING_COP_CURVE,
    GSHP_PART_LOAD_CURVE,
    ASHP_HEATING_CAPACITY_CURVE,
    ASHP_HEATING_COP_CURVE,
    ASHP_PART_LOAD_CURVE,
    EXHAUST_HP_HEATING_COP_CURVE,
)

logger = logging.getLogger(__name__)


class SwedishHVACSystem(Enum):
    """Swedish HVAC system types mapped to EnergyPlus objects."""

    # District Heating (70% of Swedish MFH)
    DISTRICT_HEATING = "district_heating"

    # Heat Pumps
    EXHAUST_AIR_HP = "exhaust_air_hp"      # FTX-VP (Nibe F730, F750, Thermia)
    GROUND_SOURCE_HP = "ground_source_hp"  # Bergvärme
    AIR_SOURCE_HP = "air_source_hp"        # Luft-vatten

    # Electric
    DIRECT_ELECTRIC = "direct_electric"    # Direktel

    # Hybrid
    DISTRICT_PLUS_HP = "district_plus_hp"  # DH + supplemental HP

    # Legacy (for older buildings)
    OIL_BOILER = "oil_boiler"              # Oljeeldning (being phased out)
    PELLET_BOILER = "pellet_boiler"        # Pelletseldning


class VentilationSystem(Enum):
    """Swedish ventilation system types."""

    FTX = "ftx"              # Balanced with HR (80-90% efficiency)
    F_SYSTEM = "f_system"    # Exhaust only (frånluft)
    FT_SYSTEM = "ft_system"  # Balanced without HR
    NATURAL = "natural"      # Självdrag (pre-1970)
    FTX_VP = "ftx_vp"        # FTX with integrated heat pump


@dataclass
class HVACTemplate:
    """EnergyPlus IDF template for a Swedish HVAC system."""

    system_type: SwedishHVACSystem

    # EnergyPlus objects as format strings
    plant_loop: Optional[str] = None           # Hot water loop
    air_loop: Optional[str] = None             # Ventilation system
    zone_equipment: str = ""                   # Per-zone equipment

    # Performance parameters
    heating_cop: float = 1.0                   # COP at design conditions
    cooling_cop: float = 3.0
    heat_recovery_effectiveness: float = 0.0

    # Swedish-specific
    district_heating_substation: bool = False
    radiator_system: bool = True               # Most Swedish buildings
    underfloor_heating: bool = False           # Newer buildings

    # Temperature setpoints (Swedish standards)
    supply_temp_design_c: float = 55.0         # Radiator supply (modern)
    return_temp_design_c: float = 45.0         # Radiator return
    heating_setpoint_c: float = 21.0           # Indoor setpoint
    cooling_setpoint_c: float = 26.0           # Rarely used in Sweden


@dataclass
class HVACSelection:
    """Selected HVAC system with parameters."""

    primary_heating: SwedishHVACSystem
    ventilation: VentilationSystem

    # Sizing parameters (optional, can be autosized)
    heating_capacity_kw: Optional[float] = None
    hp_cop: float = 3.5
    heat_recovery_eff: float = 0.80

    # DHW
    dhw_system: str = "integrated"  # or "separate_tank", "district"
    dhw_hp_fraction: float = 1.0    # Fraction from HP vs electric backup

    # Source data
    detected_from: str = "archetype"  # or "gripen", "sweden_geojson"
    confidence: float = 0.7

    # Temperature design
    supply_temp_c: float = 55.0
    return_temp_c: float = 45.0


# =============================================================================
# DISTRICT HEATING TEMPLATE
# =============================================================================
# Most common heating system in Swedish multi-family buildings (70%)
# Typically 80/60°C supply/return for older buildings, 55/45°C for newer

DISTRICT_HEATING_TEMPLATE = '''
!- ============================================================
!- DISTRICT HEATING (Fjärrvärme) - {building_name}
!- Swedish standard: {supply_temp}°C/{return_temp}°C supply/return
!- Substation with heat exchanger to building hot water loop
!- ============================================================

!- Hot Water Plant Loop
PlantLoop,
    {plant_name},                !- Name
    Water,                       !- Fluid Type
    ,                            !- User Defined Fluid Type
    {plant_name} Operation,      !- Plant Equipment Operation Scheme Name
    {plant_name} Supply Outlet,  !- Loop Supply Side Outlet Node Name
    {plant_name} Supply Inlet,   !- Loop Supply Side Inlet Node Name
    {plant_name} Demand Outlet,  !- Loop Demand Side Outlet Node Name
    {plant_name} Demand Inlet,   !- Loop Demand Side Inlet Node Name
    autosize,                    !- Maximum Loop Flow Rate
    0,                           !- Minimum Loop Flow Rate
    autosize,                    !- Plant Loop Volume
    {plant_name} Supply Setpoint Nodes,  !- Plant Side Inlet Node Name
    ,                            !- Demand Side Inlet Node Name
    SingleSetpoint,              !- Load Distribution Scheme
    ;                            !- Common Pipe Simulation

!- District Heating Connection (infinite source)
DistrictHeating:Water,
    {plant_name} District Heating,   !- Name
    {plant_name} DH Inlet,           !- Hot Water Inlet Node Name
    {plant_name} DH Outlet,          !- Hot Water Outlet Node Name
    autosize;                        !- Nominal Capacity

!- Plant Operation Scheme
PlantEquipmentOperationSchemes,
    {plant_name} Operation,
    PlantEquipmentOperation:HeatingLoad,
    {plant_name} Heating Operation,
    AlwaysOn;

PlantEquipmentOperation:HeatingLoad,
    {plant_name} Heating Operation,
    0,
    1000000000,
    {plant_name} Heating Equipment;

PlantEquipmentList,
    {plant_name} Heating Equipment,
    DistrictHeating:Water,
    {plant_name} District Heating;

!- Supply side branch
Branch,
    {plant_name} Supply Branch,
    ,
    DistrictHeating:Water,
    {plant_name} District Heating,
    {plant_name} DH Inlet,
    {plant_name} DH Outlet,
    ;

!- Demand side branches (one per zone with radiator)
{demand_branches}

!- Setpoint Manager
SetpointManager:Scheduled,
    {plant_name} Supply Setpoint Manager,
    Temperature,
    HW_Supply_Temp_Schedule,
    {plant_name} Supply Setpoint Nodes;

!- Hot water supply temperature schedule
Schedule:Compact,
    HW_Supply_Temp_Schedule,
    Temperature,
    Through: 12/31,
    For: AllDays,
    Until: 24:00, {supply_temp};

!- Pump
Pump:VariableSpeed,
    {plant_name} Pump,
    {plant_name} Supply Inlet,
    {plant_name} Pump Outlet,
    autosize,                    !- Design Maximum Flow Rate
    179352,                      !- Design Pump Head (Pa) - typical
    autosize,                    !- Design Power Consumption
    0.9,                         !- Motor Efficiency
    0.0,                         !- Fraction of Motor Inefficiencies to Fluid Stream
    0,                           !- Coefficient 1 of the Part Load Performance Curve
    1,                           !- Coefficient 2 of the Part Load Performance Curve
    0,                           !- Coefficient 3 of the Part Load Performance Curve
    0,                           !- Coefficient 4 of the Part Load Performance Curve
    0,                           !- Design Minimum Flow Rate
    Intermittent;                !- Pump Control Type
'''


# =============================================================================
# RADIATOR TEMPLATE (Per-Zone)
# =============================================================================
# Used with district heating, boilers, or heat pumps

RADIATOR_TEMPLATE = '''
!- ============================================================
!- HOT WATER RADIATOR - {zone_name}
!- Standard Swedish panel radiator (70/30 radiant/convective)
!- ============================================================

ZoneHVAC:Baseboard:RadiantConvective:Water,
    {zone_name} Radiator,        !- Name
    AlwaysOn,                    !- Availability Schedule Name
    {zone_name} Radiator Inlet,  !- Inlet Node Name
    {zone_name} Radiator Outlet, !- Outlet Node Name
    HeatingDesignCapacity,       !- Heating Design Capacity Method
    autosize,                    !- Heating Design Capacity
    ,                            !- Heating Design Capacity Per Floor Area
    ,                            !- Fraction of Autosized Heating Design Capacity
    0.3,                         !- Fraction Radiant
    0.012,                       !- Fraction of Radiant Energy Incident on People
    {surface_list};              !- Surface Name/Fraction pairs

!- Zone Equipment List
ZoneHVAC:EquipmentList,
    {zone_name} Equipment,
    SequentialLoad,
    ZoneHVAC:Baseboard:RadiantConvective:Water,
    {zone_name} Radiator,
    1,
    1,
    ,
    ;

!- Zone Equipment Connections
ZoneHVAC:EquipmentConnections,
    {zone_name},                 !- Zone Name
    {zone_name} Equipment,       !- Zone Conditioning Equipment List Name
    ,                            !- Zone Air Inlet Node or NodeList Name
    ,                            !- Zone Air Exhaust Node or NodeList Name
    {zone_name} Air Node,        !- Zone Air Node Name
    ;                            !- Zone Return Air Node or NodeList Name

!- Thermostat
ZoneControl:Thermostat,
    {zone_name} Thermostat,
    {zone_name},
    {zone_name} Thermostat Schedule,
    ThermostatSetpoint:SingleHeating,
    {zone_name} Heating Setpoint;

ThermostatSetpoint:SingleHeating,
    {zone_name} Heating Setpoint,
    Heating_Setpoint_Schedule;

Schedule:Constant,
    {zone_name} Thermostat Schedule,
    Any Number,
    1;  !- SingleHeating

Schedule:Compact,
    Heating_Setpoint_Schedule,
    Temperature,
    Through: 12/31,
    For: AllDays,
    Until: 24:00, 21.0;
'''


# =============================================================================
# EXHAUST AIR HEAT PUMP TEMPLATE (FTX-VP)
# =============================================================================
# Common in renovated buildings: extracts heat from exhaust air
# Examples: Nibe F730, F750, Thermia Mega

EXHAUST_AIR_HP_TEMPLATE = '''
!- ============================================================
!- EXHAUST AIR HEAT PUMP (FTX-VP) - {building_name}
!- Extracts heat from exhaust air, heats supply + DHW
!- COP: {cop} at design conditions
!- Common brands: Nibe F730/F750, Thermia Mega, IVT
!- ============================================================

!- Hot Water Plant Loop
PlantLoop,
    {plant_name},                !- Name
    Water,                       !- Fluid Type
    ,                            !- User Defined Fluid Type
    {plant_name} Operation,      !- Plant Equipment Operation Scheme Name
    {plant_name} Supply Outlet,  !- Loop Supply Side Outlet Node Name
    {plant_name} Supply Inlet,   !- Loop Supply Side Inlet Node Name
    {plant_name} Demand Outlet,  !- Loop Demand Side Outlet Node Name
    {plant_name} Demand Inlet,   !- Loop Demand Side Inlet Node Name
    autosize,                    !- Maximum Loop Flow Rate
    0,                           !- Minimum Loop Flow Rate
    autosize,                    !- Plant Loop Volume
    {plant_name} Supply Setpoint Nodes,
    ,
    SingleSetpoint,
    ;

!- Exhaust Air Heat Pump (modeled as water-to-water with exhaust air source)
HeatPump:WaterToWater:EquationFit:Heating,
    {hp_name},                   !- Name
    {hp_name} Load Inlet,        !- Load Side Inlet Node Name (to radiators)
    {hp_name} Load Outlet,       !- Load Side Outlet Node Name
    {hp_name} Source Inlet,      !- Source Side Inlet Node Name (exhaust air)
    {hp_name} Source Outlet,     !- Source Side Outlet Node Name
    autosize,                    !- Reference Load Side Flow Rate
    autosize,                    !- Reference Source Side Flow Rate
    autosize,                    !- Reference Heating Capacity
    {cop},                       !- Reference Heating Power Consumption (COP basis)
    {hp_name} Heating CAPFTemp,  !- Heating Capacity Modifier Function of Temperature
    {hp_name} Heating EIRFTemp,  !- Heating Power Modifier Function of Temperature
    {hp_name} Heating CAPFPLR,   !- Heating Capacity Modifier Function of Part Load Ratio
    {hp_name} Heating EIRFPLR;   !- Heating Power Modifier Function of Part Load Ratio

!- Performance curves for FTX-VP (exhaust air ~20°C constant)
Curve:Biquadratic,
    {hp_name} Heating CAPFTemp,
    1.0,                         !- Coefficient1 Constant
    0.0,                         !- Coefficient2 x (source temp, relatively constant)
    0.0,                         !- Coefficient3 x**2
    0.0,                         !- Coefficient4 y (load temp)
    0.0,                         !- Coefficient5 y**2
    0.0,                         !- Coefficient6 x*y
    10, 25,                      !- Min/Max X (exhaust air temp range)
    30, 60;                      !- Min/Max Y (water supply temp range)

Curve:Biquadratic,
    {hp_name} Heating EIRFTemp,
    1.0,                         !- Coefficient1 Constant
    0.0, 0.0, 0.0, 0.0, 0.0,     !- Other coefficients
    10, 25,
    30, 60;

Curve:Quadratic,
    {hp_name} Heating CAPFPLR,
    1.0, 0.0, 0.0,               !- Coefficients (linear, no degradation)
    0, 1;                        !- Min/Max PLR

Curve:Quadratic,
    {hp_name} Heating EIRFPLR,
    1.0, 0.0, 0.0,               !- Coefficients
    0, 1;

!- Electric backup heater (for cold days when HP insufficient)
Boiler:HotWater,
    {hp_name} Backup Heater,     !- Name
    Electricity,                 !- Fuel Type
    autosize,                    !- Nominal Capacity
    0.99,                        !- Nominal Thermal Efficiency
    LeavingBoiler,               !- Efficiency Curve Temperature Evaluation Variable
    ,                            !- Normalized Boiler Efficiency Curve Name
    {hp_name} Backup Inlet,
    {hp_name} Backup Outlet,
    ,
    60,                          !- Design Water Outlet Temperature
    autosize,                    !- Design Water Flow Rate
    0,                           !- Minimum Part Load Ratio
    1.1,                         !- Maximum Part Load Ratio
    1,                           !- Optimum Part Load Ratio
    {hp_name} Backup Inlet,
    1,                           !- Sizing Factor
    NotModulated,                !- Boiler Flow Mode
    0;                           !- Parasitic Electric Load

{demand_branches}
'''


# =============================================================================
# GROUND SOURCE HEAT PUMP TEMPLATE (Bergvärme)
# =============================================================================
# Common for single-family and smaller MFH
# Borehole field + water-to-water HP

GSHP_TEMPLATE = '''
!- ============================================================
!- GROUND SOURCE HEAT PUMP (Bergvärme) - {building_name}
!- Borehole field with water-to-water heat pump
!- COP: {cop} at design conditions
!- Swedish standard: 100-200m boreholes in granite/gneiss
!- ============================================================

!- Hot Water Plant Loop (Load Side)
PlantLoop,
    {plant_name},                !- Name
    Water,
    ,
    {plant_name} Operation,
    {plant_name} Supply Outlet,
    {plant_name} Supply Inlet,
    {plant_name} Demand Outlet,
    {plant_name} Demand Inlet,
    autosize,
    0,
    autosize,
    {plant_name} Supply Setpoint Nodes,
    ,
    SingleSetpoint,
    ;

!- Ground Loop (Source Side)
PlantLoop,
    {plant_name} Ground Loop,
    Water,
    ,
    {plant_name} Ground Operation,
    {plant_name} Ground Supply Outlet,
    {plant_name} Ground Supply Inlet,
    {plant_name} Ground Demand Outlet,
    {plant_name} Ground Demand Inlet,
    autosize,
    0,
    autosize,
    ,
    ,
    SingleSetpoint,
    ;

!- Ground Heat Exchanger (Borehole Field)
GroundHeatExchanger:System,
    {ghx_name},                  !- Name
    {ghx_name} Inlet,            !- Inlet Node Name
    {ghx_name} Outlet,           !- Outlet Node Name
    autosize,                    !- Design Flow Rate
    Site:GroundTemperature:Undisturbed:KusudaAchenbach,  !- Undisturbed Ground Temperature Model Type
    {ground_temp_name},          !- Undisturbed Ground Temperature Model Name
    1.8,                         !- Ground Thermal Conductivity (Swedish granite/gneiss)
    2400000,                     !- Ground Thermal Heat Capacity
    {ghx_name} GFunction;        !- GHE:Vertical:ResponseFactors Object Name

!- Swedish ground temperature profile
Site:GroundTemperature:Undisturbed:KusudaAchenbach,
    {ground_temp_name},          !- Name
    1.8,                         !- Soil Thermal Conductivity (W/m-K)
    2400,                        !- Soil Density (kg/m3)
    1000,                        !- Soil Specific Heat (J/kg-K)
    8.5,                         !- Average Soil Surface Temperature (°C) - Stockholm
    3.5,                         !- Average Amplitude of Surface Temperature (°C)
    15;                          !- Phase Shift of Minimum Surface Temperature (days)

!- G-function for borehole field
GroundHeatExchanger:Vertical:Properties,
    {ghx_name} Props,            !- Name
    1,                           !- Depth of Top of Borehole (m)
    {borehole_depth},            !- Borehole Length (m) - typically 100-200m
    0.114,                       !- Borehole Diameter (m) - 4.5 inch
    1.8,                         !- Grout Thermal Conductivity (W/m-K)
    0.389,                       !- Pipe Thermal Conductivity (W/m-K)
    0.032,                       !- Pipe Out Diameter (m)
    0.0262,                      !- U-Tube Distance (m)
    0.00269,                     !- Pipe Thickness (m)
    0,                           !- Maximum Length of Simulation (years)
    ;                            !- G-Function Reference Ratio

!- Ground Source Heat Pump
HeatPump:WaterToWater:EquationFit:Heating,
    {hp_name},                   !- Name
    {hp_name} Load Inlet,        !- Load Side Inlet Node
    {hp_name} Load Outlet,       !- Load Side Outlet Node
    {hp_name} Source Inlet,      !- Source Side Inlet Node (from GHX)
    {hp_name} Source Outlet,     !- Source Side Outlet Node (to GHX)
    autosize,                    !- Reference Load Side Flow Rate
    autosize,                    !- Reference Source Side Flow Rate
    autosize,                    !- Reference Heating Capacity
    {cop},                       !- Reference COP
    {hp_name} Heating CAPFTemp,
    {hp_name} Heating EIRFTemp,
    {hp_name} Heating CAPFPLR,
    {hp_name} Heating EIRFPLR;

!- GSHP Performance curves (source temp 0-10°C typical)
Curve:Biquadratic,
    {hp_name} Heating CAPFTemp,
    1.04,                        !- Based on typical GSHP data
    0.02,                        !- Slight increase with source temp
    0.0,
    -0.01,                       !- Slight decrease with higher load temp
    0.0,
    0.0,
    -5, 15,                      !- Source temp range (brine)
    30, 55;                      !- Load temp range (supply water)

Curve:Biquadratic,
    {hp_name} Heating EIRFTemp,
    0.96,
    -0.02,
    0.0,
    0.02,
    0.0,
    0.0,
    -5, 15,
    30, 55;

Curve:Quadratic,
    {hp_name} Heating CAPFPLR,
    1.0, 0.0, 0.0,
    0.1, 1.0;

Curve:Quadratic,
    {hp_name} Heating EIRFPLR,
    0.1, 0.9, 0.0,               !- Part load penalty
    0.1, 1.0;

{demand_branches}
'''


# =============================================================================
# AIR SOURCE HEAT PUMP TEMPLATE (Luft-vatten)
# =============================================================================
# Outdoor air as source, delivers to hydronic system

AIR_SOURCE_HP_TEMPLATE = '''
!- ============================================================
!- AIR SOURCE HEAT PUMP (Luft-vatten) - {building_name}
!- Outdoor air source with hydronic distribution
!- COP: {cop} at +7°C outdoor (varies significantly with temperature)
!- Typical Swedish brands: Nibe, Thermia, CTC, IVT
!- ============================================================

!- Hot Water Plant Loop
PlantLoop,
    {plant_name},
    Water,
    ,
    {plant_name} Operation,
    {plant_name} Supply Outlet,
    {plant_name} Supply Inlet,
    {plant_name} Demand Outlet,
    {plant_name} Demand Inlet,
    autosize, 0, autosize,
    {plant_name} Supply Setpoint Nodes,
    , SingleSetpoint, ;

!- Air Source Heat Pump (Coil:Heating:WaterToAirHeatPump)
Coil:Heating:WaterToAirHeatPump:EquationFit,
    {hp_name},                   !- Name
    {hp_name} Water Inlet,       !- Water Inlet Node Name
    {hp_name} Water Outlet,      !- Water Outlet Node Name
    {hp_name} Air Inlet,         !- Air Inlet Node Name (outdoor)
    {hp_name} Air Outlet,        !- Air Outlet Node Name
    autosize,                    !- Rated Air Flow Rate
    autosize,                    !- Rated Water Flow Rate
    autosize,                    !- Gross Rated Heating Capacity
    {cop},                       !- Gross Rated Heating COP
    {hp_name} TotCapFTemp,       !- Total Heating Capacity Modifier Function of Temperature
    ,                            !- Total Heating Capacity Modifier Function of Air Flow Fraction
    {hp_name} EIRFTemp;          !- Heating Power Consumption Modifier Function of Temperature

!- ASHP Performance curves (significant temperature dependency)
!- COP drops significantly below 0°C
Curve:Biquadratic,
    {hp_name} TotCapFTemp,
    0.8,                         !- Base capacity
    0.02,                        !- Increases with outdoor temp
    0.0001,                      !- Quadratic outdoor
    -0.005,                      !- Decreases with higher supply temp
    0.0,
    0.0,
    -20, 20,                     !- Outdoor temp range (Swedish winter!)
    25, 55;                      !- Supply temp range

Curve:Biquadratic,
    {hp_name} EIRFTemp,
    1.2,                         !- Base EIR (1/COP proxy)
    -0.03,                       !- Better efficiency at higher outdoor temp
    0.0005,
    0.01,                        !- Worse at higher supply temp
    0.0,
    0.0,
    -20, 20,
    25, 55;

!- Defrost (critical for Swedish climate)
!- Most ASHP lose 5-15% capacity to defrost below 0°C

{demand_branches}
'''


# =============================================================================
# FTX VENTILATION TEMPLATE (WITH HEAT RECOVERY)
# =============================================================================
# Standard Swedish balanced ventilation with heat recovery

FTX_TEMPLATE = '''
!- ============================================================
!- FTX VENTILATION SYSTEM - {building_name}
!- Balanced ventilation with heat recovery
!- Heat recovery effectiveness: {hr_eff}
!- SFP (Specific Fan Power): {sfp} kW/(m³/s)
!- ============================================================

!- Outdoor Air System
AirLoopHVAC:OutdoorAirSystem,
    {oa_system_name},
    {oa_system_name} Controllers,
    {oa_system_name} Equipment List,
    {oa_system_name} Avail List;

!- Heat Recovery Unit (rotary or plate)
HeatExchanger:AirToAir:SensibleAndLatent,
    {hr_name},                   !- Name
    AlwaysOn,                    !- Availability Schedule
    autosize,                    !- Nominal Supply Air Flow Rate
    {hr_eff},                    !- Sensible Effectiveness at 100% Heating
    0.0,                         !- Latent Effectiveness at 100% Heating (sensible only typical)
    {hr_eff},                    !- Sensible Effectiveness at 75% Heating
    0.0,                         !- Latent Effectiveness at 75% Heating
    {hr_eff},                    !- Sensible Effectiveness at 100% Cooling
    0.0,
    {hr_eff},
    0.0,
    {hr_name} OA Inlet,          !- Supply Air Inlet Node Name
    {hr_name} OA Outlet,         !- Supply Air Outlet Node Name
    {hr_name} Exhaust Inlet,     !- Exhaust Air Inlet Node Name
    {hr_name} Exhaust Outlet,    !- Exhaust Air Outlet Node Name
    0,                           !- Nominal Electric Power
    Yes,                         !- Supply Air Outlet Temperature Control
    Plate,                       !- Heat Exchanger Type
    MinimumExhaustTemperature,   !- Frost Control Type
    1.7,                         !- Threshold Temperature (for frost control)
    0.083,                       !- Initial Defrost Time Fraction
    0.012,                       !- Rate of Defrost Time Fraction Increase
    Yes;                         !- Economizer Lockout

!- Supply Fan
Fan:SystemModel,
    {supply_fan_name},           !- Name
    AlwaysOn,                    !- Availability Schedule
    {supply_fan_name} Inlet,     !- Air Inlet Node Name
    {supply_fan_name} Outlet,    !- Air Outlet Node Name
    autosize,                    !- Design Maximum Air Flow Rate
    Continuous,                  !- Speed Control Method
    0.2,                         !- Electric Power Minimum Flow Rate Fraction
    {fan_pressure},              !- Design Pressure Rise (Pa) - from SFP
    0.7,                         !- Motor Efficiency
    1.0,                         !- Motor In Air Stream Fraction
    autosize,                    !- Design Electric Power Consumption
    TotalEfficiencyAndPressure,  !- Design Power Sizing Method
    ,
    ,
    {fan_efficiency};            !- Fan Total Efficiency

!- Exhaust Fan
Fan:SystemModel,
    {exhaust_fan_name},
    AlwaysOn,
    {exhaust_fan_name} Inlet,
    {exhaust_fan_name} Outlet,
    autosize,
    Continuous,
    0.2,
    {fan_pressure},
    0.7,
    1.0,
    autosize,
    TotalEfficiencyAndPressure,
    ,
    ,
    {fan_efficiency};
'''


def _generate_named_curve(base_curve: PerformanceCurve, hp_name: str, suffix: str) -> str:
    """
    Generate EnergyPlus curve with building-specific name.

    Args:
        base_curve: The base performance curve with coefficients
        hp_name: Heat pump name prefix
        suffix: Curve purpose suffix (CAPFTemp, EIRFTemp, etc.)

    Returns:
        IDF curve object string
    """
    curve_name = f"{hp_name}_{suffix}"

    if base_curve.curve_type == "biquadratic":
        return f"""
Curve:Biquadratic,
    {curve_name},                        !- Name
    {base_curve.c1},                     !- Coefficient1 Constant
    {base_curve.c2},                     !- Coefficient2 x
    {base_curve.c3},                     !- Coefficient3 x**2
    {base_curve.c4},                     !- Coefficient4 y
    {base_curve.c5},                     !- Coefficient5 y**2
    {base_curve.c6},                     !- Coefficient6 x*y
    {base_curve.x_min},                  !- Minimum Value of x
    {base_curve.x_max},                  !- Maximum Value of x
    {base_curve.y_min},                  !- Minimum Value of y
    {base_curve.y_max},                  !- Maximum Value of y
    {base_curve.output_min},             !- Minimum Curve Output
    {base_curve.output_max};             !- Maximum Curve Output
"""
    elif base_curve.curve_type in ("quadratic", "cubic"):
        # For PLR curves (quadratic/cubic)
        return f"""
Curve:Quadratic,
    {curve_name},                        !- Name
    {base_curve.c1},                     !- Coefficient1 Constant
    {base_curve.c2},                     !- Coefficient2 x
    {base_curve.c3},                     !- Coefficient3 x**2
    {base_curve.x_min},                  !- Minimum Value of x
    {base_curve.x_max},                  !- Maximum Value of x
    {base_curve.output_min},             !- Minimum Curve Output
    {base_curve.output_max};             !- Maximum Curve Output
"""
    return ""


def _generate_hp_performance_curves(
    system_type: SwedishHVACSystem,
    hp_name: str,
    supply_temp_c: float = 35.0,
) -> tuple[str, float]:
    """
    Generate performance curves for a heat pump and return design COP.

    Args:
        system_type: Swedish HVAC system type
        hp_name: Heat pump name for curve naming
        supply_temp_c: Design supply water temperature

    Returns:
        Tuple of (IDF curves string, design COP)
    """
    curves = []
    design_cop = 3.5  # Default

    if system_type == SwedishHVACSystem.GROUND_SOURCE_HP:
        hp = get_heat_pump_performance(HeatPumpType.GROUND_SOURCE)
        design_cop = hp.get_cop(source_temp_c=0.0, sink_temp_c=supply_temp_c)

        curves.append(_generate_named_curve(GSHP_HEATING_CAPACITY_CURVE, hp_name, "Heating_CAPFTemp"))
        curves.append(_generate_named_curve(GSHP_HEATING_COP_CURVE, hp_name, "Heating_EIRFTemp"))
        curves.append(_generate_named_curve(GSHP_PART_LOAD_CURVE, hp_name, "Heating_CAPFPLR"))
        curves.append(_generate_named_curve(GSHP_PART_LOAD_CURVE, hp_name, "Heating_EIRFPLR"))

    elif system_type == SwedishHVACSystem.AIR_SOURCE_HP:
        hp = get_heat_pump_performance(HeatPumpType.AIR_SOURCE)
        # Use design point at 7°C outdoor (A7 rating)
        design_cop = hp.get_cop(source_temp_c=7.0, sink_temp_c=supply_temp_c)

        curves.append(_generate_named_curve(ASHP_HEATING_CAPACITY_CURVE, hp_name, "TotCapFTemp"))
        curves.append(_generate_named_curve(ASHP_HEATING_COP_CURVE, hp_name, "EIRFTemp"))

    elif system_type == SwedishHVACSystem.EXHAUST_AIR_HP:
        hp = get_heat_pump_performance(HeatPumpType.EXHAUST_AIR)
        # Exhaust air source at 21°C
        design_cop = hp.get_cop(source_temp_c=21.0, sink_temp_c=supply_temp_c)

        curves.append(_generate_named_curve(EXHAUST_HP_HEATING_COP_CURVE, hp_name, "Heating_CAPFTemp"))
        curves.append(_generate_named_curve(EXHAUST_HP_HEATING_COP_CURVE, hp_name, "Heating_EIRFTemp"))
        # Simple linear PLR curves for exhaust air HP
        plr_curve = PerformanceCurve(
            name="PLR", curve_type="quadratic",
            c1=1.0, c2=0.0, c3=0.0,
            x_min=0.0, x_max=1.0,
            output_min=0.8, output_max=1.0,
        )
        curves.append(_generate_named_curve(plr_curve, hp_name, "Heating_CAPFPLR"))
        curves.append(_generate_named_curve(plr_curve, hp_name, "Heating_EIRFPLR"))

    logger.info(f"Generated performance curves for {system_type.value}: design COP = {design_cop:.2f}")

    return "\n".join(curves), design_cop


def generate_hvac_idf(
    system_type: SwedishHVACSystem,
    zone_names: List[str],
    design_heating_load_w: float,
    building_name: str = "Building",
    supply_temp_c: float = 55.0,
    return_temp_c: float = 45.0,
    cop: float = None,  # If None, calculate from performance curves
    heat_recovery_eff: float = 0.80,
    sfp: float = 1.5,
    borehole_depth_m: float = 150.0,
) -> str:
    """
    Generate EnergyPlus HVAC objects for a Swedish building.

    Uses temperature-dependent performance curves from performance_curves.py
    for realistic heat pump modeling.

    Args:
        system_type: Swedish HVAC system type
        zone_names: List of thermal zone names
        design_heating_load_w: Total design heating load (W)
        building_name: Building name for labels
        supply_temp_c: Hot water supply temperature (°C)
        return_temp_c: Hot water return temperature (°C)
        cop: Heat pump COP at design conditions (if None, calculated from curves)
        heat_recovery_eff: FTX heat recovery effectiveness (0-1)
        sfp: Specific Fan Power (kW/(m³/s))
        borehole_depth_m: GSHP borehole depth (m)

    Returns:
        IDF string with all HVAC objects including performance curves
    """
    plant_name = f"{building_name}_HW_Loop"
    hp_name = f"{building_name}_HP"
    ghx_name = f"{building_name}_GHX"
    ground_temp_name = f"{building_name}_Ground_Temp"

    # Generate demand branches (radiators for each zone)
    demand_branches = _generate_demand_branches(zone_names, plant_name)

    # Generate performance curves and get design COP for heat pump systems
    perf_curves = ""
    if system_type in (SwedishHVACSystem.GROUND_SOURCE_HP,
                       SwedishHVACSystem.AIR_SOURCE_HP,
                       SwedishHVACSystem.EXHAUST_AIR_HP):
        perf_curves, calculated_cop = _generate_hp_performance_curves(
            system_type, hp_name, supply_temp_c
        )
        # Use calculated COP if not explicitly provided
        if cop is None:
            cop = calculated_cop
    else:
        if cop is None:
            cop = 3.5  # Default for other systems

    if system_type == SwedishHVACSystem.DISTRICT_HEATING:
        template = DISTRICT_HEATING_TEMPLATE
        return template.format(
            building_name=building_name,
            plant_name=plant_name,
            supply_temp=supply_temp_c,
            return_temp=return_temp_c,
            demand_branches=demand_branches,
        )

    elif system_type == SwedishHVACSystem.EXHAUST_AIR_HP:
        template = EXHAUST_AIR_HP_TEMPLATE
        idf_content = template.format(
            building_name=building_name,
            plant_name=plant_name,
            hp_name=hp_name,
            cop=cop,
            demand_branches=demand_branches,
        )
        return perf_curves + idf_content

    elif system_type == SwedishHVACSystem.GROUND_SOURCE_HP:
        template = GSHP_TEMPLATE
        idf_content = template.format(
            building_name=building_name,
            plant_name=plant_name,
            hp_name=hp_name,
            ghx_name=ghx_name,
            ground_temp_name=ground_temp_name,
            cop=cop,
            borehole_depth=borehole_depth_m,
            demand_branches=demand_branches,
        )
        return perf_curves + idf_content

    elif system_type == SwedishHVACSystem.AIR_SOURCE_HP:
        template = AIR_SOURCE_HP_TEMPLATE
        idf_content = template.format(
            building_name=building_name,
            plant_name=plant_name,
            hp_name=hp_name,
            cop=cop,
            demand_branches=demand_branches,
        )
        return perf_curves + idf_content

    else:
        # Fall back to simple IdealLoads for unsupported systems
        logger.warning(
            f"HVAC system {system_type} not fully implemented, "
            "using IdealLoadsAirSystem"
        )
        return _generate_ideal_loads_fallback(zone_names)


def _generate_demand_branches(zone_names: List[str], plant_name: str) -> str:
    """Generate demand-side branches with radiators for each zone."""
    branches = []

    for zone_name in zone_names:
        branch = RADIATOR_TEMPLATE.format(
            zone_name=zone_name,
            surface_list="",  # Will be populated with actual surfaces
        )
        branches.append(branch)

    return "\n".join(branches)


def _generate_ideal_loads_fallback(zone_names: List[str]) -> str:
    """Generate IdealLoadsAirSystem as fallback for unsupported systems."""
    idf_parts = []

    for zone_name in zone_names:
        idf_parts.append(f'''
ZoneHVAC:IdealLoadsAirSystem,
    {zone_name}_IdealLoads,
    ,
    {zone_name}_Supply,
    {zone_name}_Exhaust,
    ,
    50, 13,
    0.015, 0.009,
    NoLimit, autosize, ,
    NoLimit, autosize, ,
    ,
    ,
    ConstantSupplyHumidityRatio,
    ,
    ConstantSupplyHumidityRatio,
    ,
    ,
    None,
    NoEconomizer,
    Sensible,
    0.80,
    0.0;

ZoneHVAC:EquipmentList,
    {zone_name}_EquipList,
    SequentialLoad,
    ZoneHVAC:IdealLoadsAirSystem,
    {zone_name}_IdealLoads,
    1, 1, , ;

ZoneHVAC:EquipmentConnections,
    {zone_name},
    {zone_name}_EquipList,
    {zone_name}_Supply,
    {zone_name}_Exhaust,
    {zone_name}_AirNode,
    {zone_name}_Return;
''')

    return "\n".join(idf_parts)


# =============================================================================
# SWEDISH HVAC DEFAULTS BY ERA
# =============================================================================
# These are typical HVAC configurations by construction era

HVAC_BY_ERA = {
    # Pre-1945: Natural ventilation, district heating or local boiler
    "pre_1945": {
        "heating": SwedishHVACSystem.DISTRICT_HEATING,
        "ventilation": VentilationSystem.NATURAL,
        "supply_temp_c": 80.0,  # Old radiators need higher temp
        "return_temp_c": 60.0,
    },

    # 1945-1975: Exhaust ventilation, district heating
    "1945_1975": {
        "heating": SwedishHVACSystem.DISTRICT_HEATING,
        "ventilation": VentilationSystem.F_SYSTEM,
        "supply_temp_c": 70.0,
        "return_temp_c": 50.0,
    },

    # 1976-1995: FTX becoming common, district heating
    "1976_1995": {
        "heating": SwedishHVACSystem.DISTRICT_HEATING,
        "ventilation": VentilationSystem.FTX,
        "heat_recovery_eff": 0.70,
        "supply_temp_c": 60.0,
        "return_temp_c": 45.0,
    },

    # 1996-2010: Modern FTX, some heat pumps
    "1996_2010": {
        "heating": SwedishHVACSystem.DISTRICT_HEATING,  # Still dominant
        "ventilation": VentilationSystem.FTX,
        "heat_recovery_eff": 0.80,
        "supply_temp_c": 55.0,
        "return_temp_c": 45.0,
    },

    # 2011+: High-efficiency FTX, heat pumps common
    "2011_plus": {
        "heating": SwedishHVACSystem.EXHAUST_AIR_HP,  # Common in new MFH
        "ventilation": VentilationSystem.FTX_VP,
        "heat_recovery_eff": 0.85,
        "hp_cop": 3.5,
        "supply_temp_c": 45.0,  # Low-temp system
        "return_temp_c": 35.0,
    },
}


def get_hvac_defaults_for_era(construction_year: int) -> Dict:
    """Get typical HVAC configuration for a construction era."""
    if construction_year < 1945:
        return HVAC_BY_ERA["pre_1945"]
    elif construction_year < 1976:
        return HVAC_BY_ERA["1945_1975"]
    elif construction_year < 1996:
        return HVAC_BY_ERA["1976_1995"]
    elif construction_year < 2011:
        return HVAC_BY_ERA["1996_2010"]
    else:
        return HVAC_BY_ERA["2011_plus"]
