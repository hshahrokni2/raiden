# ECM Analysis

Apply Energy Conservation Measures to assess renovation potential.

## Available ECMs

### Envelope
- `wall_insulation_external` - Add external insulation
- `wall_insulation_internal` - Add internal insulation
- `roof_insulation` - Improve attic insulation
- `window_replacement` - Upgrade to triple glazing
- `air_sealing` - Reduce infiltration

### HVAC
- `ftx_upgrade` - Improve heat recovery efficiency
- `ftx_installation` - Install FTX system
- `demand_controlled_ventilation` - Add CO2-based DCV

### Other
- `solar_pv` - Rooftop solar PV
- `led_lighting` - LED retrofit
- `smart_thermostats` - Night setback control
- `heat_pump_integration` - Add heat pump

## Applying ECMs

```python
from src.ecm.idf_modifier import IDFModifier

modifier = IDFModifier()

# Single ECM
ecm_idf = modifier.apply_single(
    baseline_idf=Path('baseline.idf'),
    ecm_id='air_sealing',
    params={'reduction_factor': 0.5},
    output_dir=Path('./ecm')
)

# Multiple ECMs (sequentially)
step1 = modifier.apply_single(baseline_idf, 'air_sealing', {...}, output_dir)
step2 = modifier.apply_single(step1, 'window_replacement', {...}, output_dir)
```

## ECM Parameters

### Air Sealing
```python
params = {'reduction_factor': 0.5}  # 50% infiltration reduction
```

### Window Replacement
```python
params = {
    'u_value': 0.8,  # W/m²K
    'shgc': 0.5      # Solar heat gain coefficient
}
```

### FTX Upgrade
```python
params = {'effectiveness': 0.85}  # 85% heat recovery
```
