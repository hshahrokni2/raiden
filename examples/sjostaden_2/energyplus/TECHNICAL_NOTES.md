# Technical Notes - Sjostaden 2 EnergyPlus Model

## Document Version
- **Created**: 2025-12-16
- **EnergyPlus Version**: 25.1.0
- **Model File**: sjostaden_7zone.idf

---

## 1. Model Architecture

### 1.1 Thermal Zoning Strategy

The building is modeled with **7 thermal zones**, one per floor:

| Zone | Name | Floor Area (m²) | Volume (m³) | Height (m) |
|------|------|-----------------|-------------|------------|
| 1 | Floor1 | 320 | 960 | 3.0 |
| 2 | Floor2 | 320 | 960 | 3.0 |
| 3 | Floor3 | 320 | 960 | 3.0 |
| 4 | Floor4 | 320 | 960 | 3.0 |
| 5 | Floor5 | 320 | 960 | 3.0 |
| 6 | Floor6 | 320 | 960 | 3.0 |
| 7 | Floor7 | 320 | 960 | 3.0 |
| **Total** | | **2,240** | **6,720** | |

**Zoning Rationale**:
- Each floor treated as single zone (acceptable for multi-family with similar apartments)
- Inter-zone heat transfer modeled through floor/ceiling surfaces
- Simplified approach appropriate for ECM analysis (not detailed comfort studies)

### 1.2 Geometry Conventions

```
Building Footprint (Plan View):

    N (Y+)
    ^
    |
    +------------------+
    |                  |
    |    20m x 16m     |
    |                  |
    +------------------+--> E (X+)

Origin: Southwest corner at ground level (0, 0, 0)
```

**Surface Vertex Convention**: Counterclockwise when viewed from outside (EnergyPlus standard)

**Coordinate System**:
- X-axis: West to East (building width: 20m)
- Y-axis: South to North (building depth: 16m)
- Z-axis: Ground to Roof (building height: 21m)

---

## 2. Envelope Construction Details

### 2.1 External Walls

```
Layer Structure (outside to inside):
┌─────────────────────────────────┐
│ Exterior Air Film (Rso=0.04)   │ Not modeled explicitly
├─────────────────────────────────┤
│ Render/Plaster    10mm         │ k=1.0 W/mK
├─────────────────────────────────┤
│ Mineral Wool      250mm        │ k=0.038 W/mK, R=6.58
├─────────────────────────────────┤
│ Concrete          200mm        │ k=1.7 W/mK
├─────────────────────────────────┤
│ Interior Finish   15mm         │ k=0.16 W/mK
├─────────────────────────────────┤
│ Interior Air Film (Rsi=0.13)   │ Not modeled explicitly
└─────────────────────────────────┘

Calculated U-value: ~0.13 W/m²K
```

**Material Properties** (from IDF):
```
Material,
    MineralWool250,          !- Name
    MediumRough,             !- Roughness
    0.250,                   !- Thickness {m}
    0.038,                   !- Conductivity {W/m-K}
    30,                      !- Density {kg/m3}
    840;                     !- Specific Heat {J/kg-K}
```

### 2.2 Roof Assembly

```
Layer Structure (outside to inside):
┌─────────────────────────────────┐
│ Roofing Membrane  10mm         │ k=0.17 W/mK
├─────────────────────────────────┤
│ Mineral Wool      350mm        │ k=0.038 W/mK, R=9.21
├─────────────────────────────────┤
│ Concrete Slab     200mm        │ k=1.7 W/mK
├─────────────────────────────────┤
│ Interior Finish   15mm         │ k=0.16 W/mK
└─────────────────────────────────┘

Calculated U-value: ~0.10 W/m²K
```

### 2.3 Ground Floor (Slab-on-Grade)

```
Layer Structure (ground to interior):
┌─────────────────────────────────┐
│ Ground (not modeled)           │
├─────────────────────────────────┤
│ EPS Insulation    250mm        │ k=0.038 W/mK
├─────────────────────────────────┤
│ Concrete Slab     200mm        │ k=1.7 W/mK
├─────────────────────────────────┤
│ Interior Finish   15mm         │ k=0.16 W/mK
└─────────────────────────────────┘

Calculated U-value: ~0.13 W/m²K
```

**Ground Temperature**: Monthly values from Swedish climate data (Site:GroundTemperature:BuildingSurface)

### 2.4 Internal Floors

```
Layer Structure (lower zone to upper zone):
┌─────────────────────────────────┐
│ Floor Finish      10mm         │
├─────────────────────────────────┤
│ Concrete Slab     200mm        │
├─────────────────────────────────┤
│ Ceiling Finish    15mm         │
└─────────────────────────────────┘

Thermal Mass: 200mm concrete provides significant thermal storage
```

### 2.5 Windows (Fenestration)

**Triple Glazing Configuration**:
```
WindowMaterial:SimpleGlazingSystem,
    TripleGlazing,           !- Name
    1.0,                     !- U-Factor {W/m2-K}
    0.45,                    !- Solar Heat Gain Coefficient
    0.70;                    !- Visible Transmittance
```

**Window Distribution**:
| Facade | Area (m²) | WWR (%) | Notes |
|--------|-----------|---------|-------|
| South | ~40 | 13% | Primary solar gains |
| North | ~40 | 13% | Minimal solar gains |
| East | ~32 | 10% | Morning solar |
| West | ~32 | 10% | Afternoon solar |
| **Total** | **~144** | **~12%** | Conservative Swedish design |

**Window Sizing per Floor**:
- South/North: 2.0m × 1.2m × 2 windows = 4.8 m² per facade
- East/West: 2.0m × 1.2m × 2 windows = 4.8 m² per facade
- Total per floor: ~20 m² glazing

---

## 3. HVAC System Design

### 3.1 System Overview

The model uses `ZoneHVAC:IdealLoadsAirSystem` to represent a Swedish district heating + FTX ventilation system. This approach:
- Captures thermal loads accurately
- Separates ventilation energy from heating energy
- Allows heat recovery modeling
- Avoids complex HVAC system modeling for ECM analysis

### 3.2 Ideal Loads Configuration

```
ZoneHVAC:IdealLoadsAirSystem,
    Floor1_IdealLoads,           !- Name
    ,                            !- Availability Schedule (always on)
    Floor1_Supply,               !- Zone Supply Air Node
    Floor1_Exhaust,              !- Zone Exhaust Air Node
    ,                            !- System Inlet Air Node (for heat recovery)
    50,                          !- Maximum Heating Supply Air Temperature {C}
    13,                          !- Minimum Cooling Supply Air Temperature {C}
    0.015,                       !- Maximum Heating Supply Air Humidity Ratio
    0.009,                       !- Minimum Cooling Supply Air Humidity Ratio
    NoLimit,                     !- Heating Limit
    autosize,                    !- Maximum Heating Air Flow Rate
    ,                            !- Maximum Sensible Heating Capacity
    NoLimit,                     !- Cooling Limit
    autosize,                    !- Maximum Cooling Air Flow Rate
    ,                            !- Maximum Total Cooling Capacity
    ,                            !- Heating Availability Schedule
    ,                            !- Cooling Availability Schedule
    ConstantSupplyHumidityRatio, !- Dehumidification Control Type
    ,                            !- Cooling Sensible Heat Ratio
    ConstantSupplyHumidityRatio, !- Humidification Control Type
    OA_Spec,                     !- Design Specification Outdoor Air Object
    Floor1_OA,                   !- Outdoor Air Inlet Node
    None,                        !- Demand Controlled Ventilation Type
    NoEconomizer,                !- Outdoor Air Economizer Type
    Sensible,                    !- Heat Recovery Type
    0.75,                        !- Sensible Heat Recovery Effectiveness
    0.0;                         !- Latent Heat Recovery Effectiveness
```

### 3.3 Critical Configuration Notes

**IMPORTANT - EnergyPlus 25.1.0 Bug Workaround**:

The following configuration causes a **segmentation fault** in E+ 25.1.0:
```
! DO NOT USE THIS CONFIGURATION:
    None,                        !- Dehumidification Control Type (CRASHES!)
    ,                            !- Cooling Sensible Heat Ratio
    None,                        !- Humidification Control Type (CRASHES!)
```

**Working configuration**:
```
! USE THIS INSTEAD:
    ConstantSupplyHumidityRatio, !- Dehumidification Control Type
    ,                            !- Cooling Sensible Heat Ratio (leave blank!)
    ConstantSupplyHumidityRatio, !- Humidification Control Type
```

### 3.4 Outdoor Air Specification

```
DesignSpecification:OutdoorAir,
    OA_Spec,                     !- Name
    Flow/Area,                   !- Outdoor Air Method
    ,                            !- Outdoor Air Flow per Person
    0.00035,                     !- Outdoor Air Flow per Zone Floor Area {m3/s-m2}
    ,                            !- Outdoor Air Flow per Zone
    ,                            !- Outdoor Air Flow Air Changes per Hour
    ;                            !- Outdoor Air Schedule Name
```

**Calculation**:
- BBR 6:251 requirement: 0.35 l/s/m² = 0.00035 m³/s/m²
- Per floor: 320 m² × 0.00035 = 0.112 m³/s = 112 l/s
- Total building: 0.784 m³/s = 784 l/s

### 3.5 Heat Recovery

**FTX System Parameters**:
- Sensible Heat Recovery Effectiveness: 75%
- Latent Heat Recovery Effectiveness: 0% (sensible-only plate exchanger)
- Type: Cross-flow plate heat exchanger (typical Swedish residential)

**Energy Recovery Calculation**:
```
Q_recovered = η × m_dot × cp × (T_exhaust - T_outdoor)

Where:
η = 0.75 (sensible effectiveness)
m_dot = 0.784 m³/s × 1.2 kg/m³ = 0.94 kg/s
cp = 1005 J/kg·K
```

### 3.6 FTX Fan Electricity

Modeled as `ElectricEquipment` in each zone:

```
ElectricEquipment,
    Floor1_FTX_Fan,              !- Name
    Floor1,                      !- Zone Name
    AlwaysOn,                    !- Schedule Name
    EquipmentLevel,              !- Design Level Calculation Method
    168,                         !- Design Level {W}
    ,                            !- Watts per Zone Floor Area
    ,                            !- Watts per Person
    0,                           !- Fraction Latent
    0,                           !- Fraction Radiant
    0;                           !- Fraction Lost
```

**Fan Power Calculation**:
```
SFP = 1.5 kW/(m³/s)           (Swedish requirement for FTX)
Total airflow = 0.784 m³/s
Total fan power = 1.5 × 0.784 = 1.18 kW
Per zone = 1.18 kW / 7 = 168 W
Annual = 1.18 kW × 8760 h = 10,337 kWh = 4.6 kWh/m²
```

---

## 4. Internal Loads

### 4.1 Occupancy

```
People,
    Floor1_People,               !- Name
    Floor1,                      !- Zone Name
    OccupancySchedule,           !- Number of People Schedule
    People/Area,                 !- Number of People Calculation Method
    ,                            !- Number of People
    0.04,                        !- People per Zone Floor Area {person/m2}
    ,                            !- Zone Floor Area per Person
    0.3,                         !- Fraction Radiant
    autocalculate,               !- Sensible Heat Fraction
    ActivityLevel;               !- Activity Level Schedule Name
```

**Occupancy Calculation**:
- Sveby standard: 25 m²/person
- Density: 1/25 = 0.04 person/m²
- Per floor: 320 m² × 0.04 = 12.8 people
- Total building: ~90 people (but occupancy schedule reduces effective count)

**Activity Level**: 120 W/person (seated, light activity - Sveby residential)

### 4.2 Lighting

```
Lights,
    Floor1_Lights,               !- Name
    Floor1,                      !- Zone Name
    LightingSchedule,            !- Schedule Name
    Watts/Area,                  !- Design Level Calculation Method
    ,                            !- Lighting Level
    8.0,                         !- Watts per Zone Floor Area {W/m2}
    ,                            !- Watts per Person
    0,                           !- Return Air Fraction
    0.4,                         !- Fraction Radiant
    0.2;                         !- Fraction Visible
```

**Lighting Calculation**:
- Installed power: 8 W/m² (Sveby residential standard)
- Per floor: 320 m² × 8 = 2,560 W
- Annual (with schedule): ~5,900 kWh/floor × 7 = 41,300 kWh = 18.4 kWh/m²

### 4.3 Equipment (Plug Loads)

```
ElectricEquipment,
    Floor1_Equipment,            !- Name
    Floor1,                      !- Zone Name
    EquipmentSchedule,           !- Schedule Name
    Watts/Area,                  !- Design Level Calculation Method
    ,                            !- Design Level
    10.0,                        !- Watts per Zone Floor Area {W/m2}
    ,                            !- Watts per Person
    0,                           !- Fraction Latent
    0.3,                         !- Fraction Radiant
    0;                           !- Fraction Lost
```

**Equipment Calculation**:
- Installed power: 10 W/m² (Sveby residential standard)
- Per floor: 320 m² × 10 = 3,200 W
- Annual (with schedule): ~7,000 kWh/floor × 7 = 49,000 kWh = 21.9 kWh/m²

---

## 5. Schedules

### 5.1 Occupancy Schedule

```
Schedule:Compact,
    OccupancySchedule,           !- Name
    Fraction,                    !- Schedule Type Limits Name
    Through: 12/31,              !- Field 1
    For: Weekdays,               !- Field 2
    Until: 07:00, 1.0,           !- Night (sleeping)
    Until: 08:00, 0.8,           !- Morning departure
    Until: 16:00, 0.2,           !- Daytime (work/school)
    Until: 22:00, 0.8,           !- Evening
    Until: 24:00, 1.0,           !- Night
    For: Weekends,               !- Weekend pattern
    Until: 09:00, 1.0,
    Until: 18:00, 0.5,
    Until: 24:00, 0.8;
```

### 5.2 Heating Setpoint Schedule

```
Schedule:Compact,
    HeatingSetpoint,             !- Name
    Temperature,                 !- Schedule Type Limits Name
    Through: 12/31,              !- Field 1
    For: AllDays,                !- Field 2
    Until: 24:00, 21.0;          !- Constant 21°C
```

### 5.3 Lighting Schedule

```
Schedule:Compact,
    LightingSchedule,            !- Name
    Fraction,                    !- Schedule Type Limits Name
    Through: 12/31,
    For: Weekdays,
    Until: 06:00, 0.1,           !- Night lights
    Until: 08:00, 0.6,           !- Morning
    Until: 16:00, 0.1,           !- Daytime (natural light)
    Until: 22:00, 0.8,           !- Evening peak
    Until: 24:00, 0.3,           !- Late evening
    For: Weekends,
    Until: 08:00, 0.1,
    Until: 22:00, 0.5,
    Until: 24:00, 0.2;
```

---

## 6. Output Variables

### 6.1 Standard Outputs

```
Output:Variable,*,Zone Ideal Loads Supply Air Total Heating Energy,Hourly;
Output:Variable,*,Zone Ideal Loads Supply Air Total Cooling Energy,Hourly;
Output:Variable,*,Zone Mean Air Temperature,Hourly;
Output:Variable,*,Zone Air System Sensible Heating Energy,Hourly;
Output:Variable,*,Zone Air System Sensible Cooling Energy,Hourly;
```

### 6.2 Tabular Reports

```
OutputControl:Table:Style,
    CommaAndHTML;                !- Column Separator

Output:Table:SummaryReports,
    AllSummary;                  !- Report 1 Name
```

### 6.3 Key Output Files

| File | Description |
|------|-------------|
| `eplustbl.csv` | Tabular summary (end-use breakdown) |
| `eplustbl.htm` | HTML formatted summary |
| `eplusout.eso` | Time-series data (can be very large) |
| `eplusout.err` | Warnings and errors |
| `eplusout.eio` | Initialization outputs (areas, volumes, constructions) |

---

## 7. Weather Data

### 7.1 Weather File

- **File**: SWE_Stockholm.Arlanda.024600_IWEC.epw
- **Source**: IWEC (International Weather for Energy Calculations)
- **Station**: Stockholm Arlanda (WMO 024600)
- **Latitude**: 59.65°N
- **Longitude**: 17.95°E
- **Elevation**: 61m

### 7.2 Climate Characteristics

| Parameter | Value |
|-----------|-------|
| Heating Degree Days (base 17°C) | ~3,800 HDD |
| Annual Mean Temperature | 6.5°C |
| Coldest Month Mean (Feb) | -3°C |
| Warmest Month Mean (Jul) | 17°C |
| Annual Solar Radiation | ~950 kWh/m² |

### 7.3 Ground Temperatures

```
Site:GroundTemperature:BuildingSurface,
    5, 4, 5, 7, 11, 14, 16, 17, 15, 12, 8, 6;
```

Monthly values (°C) from Swedish ground temperature data.

---

## 8. Model Limitations

### 8.1 Simplifications

1. **Single-zone per floor**: Doesn't capture apartment-level variations
2. **Ideal loads HVAC**: No system efficiency losses beyond heat recovery
3. **No DHW**: Domestic hot water excluded (added in post-processing)
4. **No thermal bridges**: Psi-values not explicitly modeled
5. **Simplified infiltration**: Constant ACH, no wind pressure modeling
6. **No shading devices**: External blinds/shutters not modeled

### 8.2 Appropriate Use Cases

| Use Case | Suitability |
|----------|-------------|
| ECM comparative analysis | Excellent |
| Annual energy estimates | Good |
| Peak load calculations | Good |
| Hourly load profiles | Good |
| Detailed comfort analysis | Limited |
| HVAC system sizing | Limited |

### 8.3 Known Deviations from Reality

1. **Inter-apartment heat transfer**: Real buildings have more complex heat flows
2. **Stairwell/corridor**: Not explicitly modeled (included in floor zones)
3. **Elevator machine room**: Not included
4. **Garage/basement**: Not modeled (assumed unheated)

---

## 9. Validation Notes

### 9.1 Geometry Checks

Run `convert-only` mode to validate IDF syntax:
```bash
energyplus --convert-only sjostaden_7zone.idf
```

### 9.2 Expected Results Ranges

| Metric | Expected Range | Notes |
|--------|----------------|-------|
| Space Heating | 30-50 kWh/m² | Swedish residential typical |
| Fan Electricity | 4-6 kWh/m² | SFP 1.5 kW/(m³/s) |
| Lighting | 15-25 kWh/m² | Residential with LED |
| Equipment | 18-25 kWh/m² | Sveby standard |

### 9.3 Sanity Checks

1. **Peak heating load**: Should be 30-50 W/m² for Swedish climate
2. **Annual heating hours**: ~5,500-6,000 hours
3. **Mean zone temperature**: Should hover around setpoint (21°C)
4. **Heating season**: October - April

---

## 10. References

1. **BBR 29** - Boverkets byggregler (2020)
2. **Sveby** - Standardisera och verifiera energiprestanda i byggnader
3. **TABULA/EPISCOPE** - Typology Approach for Building Stock Energy Assessment
4. **SS-EN ISO 13790** - Energy performance of buildings
5. **SS-EN 15251** - Indoor environmental input parameters
6. **EnergyPlus Documentation** - Input Output Reference (v25.1.0)

---

*Document generated for BRF Energy Toolkit Project*
