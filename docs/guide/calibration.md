# Model Calibration

Calibrate baseline models to match actual energy consumption from Swedish energy declarations (energideklarationer).

## Calibration Parameters

The calibrator adjusts three parameters within physically plausible bounds:

| Parameter | Range | Sensitivity |
|-----------|-------|-------------|
| Infiltration (ACH) | 0.02 - 0.15 | ~80 kWh/m² per ACH |
| Heat Recovery | 0.60 - 0.90 | ~50 kWh/m² per 0.1 |
| Window U-value | 0.7 - 1.5 W/m²K | ~8 kWh/m² per W/m²K |

## Usage

```python
from src.baseline.calibrator import BaselineCalibrator

calibrator = BaselineCalibrator()
result = calibrator.calibrate(
    idf_path=Path('baseline.idf'),
    weather_path=Path('stockholm.epw'),
    measured_heating_kwh_m2=33.0,
    output_dir=Path('./calibrated')
)

if result.success:
    print(f"Calibrated to {result.calibrated_kwh_m2:.1f} kWh/m²")
    print(f"Final error: {result.final_error_percent:.1f}%")
    print(f"Adjusted infiltration: {result.adjusted_infiltration_ach:.3f} ACH")
```

## Convergence

- Target: ±10% of measured value
- Maximum iterations: 10
- Damping factor: 0.7 (prevents overshooting)
