# ECM Full Implementation Plan

**Goal**: Implement all remaining ECMs (except VRF) with proper EnergyPlus thermal modeling for ecodistrict accuracy.

**Date**: 2025-12-26

---

## Current Status (Updated 2025-12-26)

| Category | Count | Status |
|----------|-------|--------|
| Full implementation | 35 | ✓ Complete |
| Excluded (VRF) | 1 | Not needed |

**Implementation rate: 97.2%** (35/36 ECMs have thermal/peak effects)

---

## Phase 1: Quick Wins (IdealLoads Approach) - ✓ COMPLETE

All Phase 1 ECMs are now fully implemented with thermal modeling:

| ECM | Implementation | Key Objects |
|-----|---------------|-------------|
| pipe_insulation | ✓ Full | Negative OtherEquipment for pipe losses |
| radiator_fans | ✓ Full | Setpoint reduction + ElectricEquipment |
| led_outdoor | ✓ Full | Schedule:Compact + OtherEquipment savings |
| dhw_circulation_optimization | ✓ Full | Timer schedule + OtherEquipment |
| dhw_tank_insulation | ✓ Full | Negative OtherEquipment for standby |
| pump_optimization | ✓ Full | VFD savings using affinity law (P∝S³) |
| bms_optimization | ✓ Full | Setpoint correction + schedule alignment |
| fault_detection | ✓ Full | Infiltration reduction + setpoint |
| energy_monitoring | ✓ Full | Behavioral savings + setpoint awareness |

---

## Phase 2: DHW System ECMs - ✓ COMPLETE

### 2.1 heat_pump_water_heater (Frånluftsvärmepump för VV) - ✓ IMPLEMENTED
- **Implementation**: COP-based energy shifting
- **E+ Objects Used**:
  - `OtherEquipment` (negative) - DHW thermal savings
  - `ElectricEquipment` - HP electricity consumption
- **Physics**: COP 2.5-3.5 for DHW, models energy shift from thermal to electric
- **Key Parameters**: `cop` (default 3.0), `dhw_load` (default 22 kWh/m²/year)

### 2.2 solar_thermal (Solvärme) - ✓ IMPLEMENTED
- **Implementation**: Monthly DHW savings schedule based on Stockholm irradiance
- **E+ Objects Used**:
  - `Schedule:Compact` - Monthly solar fraction (5% Jan → 70% Jul → 5% Dec)
  - `OtherEquipment` (negative) - DHW load reduction
- **Physics**: Preheats DHW using solar collectors, seasonal variation
- **Key Parameters**: `collector_area_m2` (default 4), `coverage_fraction` (default 0.40)

### 2.3 district_heating_optimization - ✓ IMPLEMENTED
- **Implementation**: Small setpoint reduction (0.2°C) from better DH control
- **E+ Objects Used**: Modifies IdealLoads setpoint
- **Physics**: Lower return temperature → more efficient DH extraction

---

## Phase 3: Storage & Peak Shaving - ✓ COMPLETE

### 3.1 battery_storage - ✓ IMPLEMENTED
- **Implementation**: ElectricLoadCenter:Storage:Simple with charge/discharge schedules
- **E+ Objects Used**:
  - `ElectricLoadCenter:Storage:Simple` - Battery system
  - `Schedule:Compact` - Charge (10:00-16:00) / Discharge (06:00-10:00, 16:00-22:00)
- **Physics**: Store PV generation for later use, affects purchased electric power
- **Key Parameters**: `capacity_kwh` (default 10), `power_kw` (default 5)

### 3.2 effektvakt_optimization - ✓ IMPLEMENTED
- **Implementation**: Peak hour setpoint reduction schedule
- **E+ Objects Used**:
  - `Schedule:Compact` - Peak shaving schedule (16:00-20:00 weekdays)
  - Modifies heating schedule reference
- **Physics**: 2°C setback during peak demand hours, reduces peak heating load
- **Key Parameters**: `setback_c` (default 2.0), `peak_hours` (default 16:00-20:00)

---

## Phase 4: Advanced Controls - ✓ COMPLETE (in Phase 1)

Already implemented in Phase 1:
- bms_optimization
- fault_detection
- energy_monitoring

---

## Implementation Summary

### ECM Thermal Effects Table

| ECM | Primary Effect | Secondary Effect |
|-----|---------------|------------------|
| heat_pump_water_heater | DHW reduction (negative gains) | Electricity increase (COP-based) |
| solar_thermal | DHW reduction (seasonal) | - |
| district_heating_optimization | Heating setpoint -0.2°C | - |
| battery_storage | Peak electric reduction | Storage losses |
| effektvakt_optimization | Peak heating reduction | Thermal mass coasting |

### Modeling Approach Summary

1. **Negative OtherEquipment**: Used for thermal savings (DHW, pipe losses)
2. **ElectricEquipment**: Used for new electricity loads (HPs, fans, pumps)
3. **Schedule:Compact**: Used for time-varying effects (seasonal, peak hours)
4. **Setpoint Modification**: Used for controls optimization
5. **ElectricLoadCenter:Storage:Simple**: Used for battery storage

---

## Testing Results (2025-12-26)

All 46 ECM tests pass:
- 22 core ECM tests
- 24 newly implemented ECM tests

```
tests/test_ecm_modifiers.py ........................ 46 passed
```

---

## Notes for Ecodistricts

Swedish ecodistricts typically feature:
- ✓ District heating with low return temps → `district_heating_optimization`
- ✓ Solar thermal for DHW (30-50% coverage) → `solar_thermal`
- ✓ Battery storage with PV → `battery_storage`
- ✓ Heat pump water heaters → `heat_pump_water_heater`
- ✓ Extensive metering and BMS → `bms_optimization`, `energy_monitoring`
- ✓ Peak shaving (effektvakt) → `effektvakt_optimization`

**All ecodistrict ECMs are now implemented with proper thermal/peak effects.**

---

## Excluded ECM

### VRF (Variable Refrigerant Flow)
- **Status**: Not implemented (comment_only)
- **Reason**: Requires complex PlantLoop modeling, not common in Swedish residential
- **Alternative**: Use `heat_pump_integration` for similar effect
