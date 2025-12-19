# Baseline Module

The baseline module provides tools for generating and calibrating EnergyPlus baseline models.

## Archetypes

Swedish building archetypes based on TABULA/EPISCOPE building stock studies.

::: src.baseline.archetypes.SwedishArchetype
    options:
      show_root_heading: true
      members:
        - envelope
        - hvac
        - loads

::: src.baseline.archetypes.ArchetypeMatcher
    options:
      show_root_heading: true

::: src.baseline.archetypes.SWEDISH_ARCHETYPES
    options:
      show_root_heading: true

## Generator

Baseline IDF generator from building geometry and archetypes.

::: src.baseline.generator.BaselineGenerator
    options:
      show_root_heading: true

## Calibrator

Model calibration to match energy declarations.

::: src.baseline.calibrator.BaselineCalibrator
    options:
      show_root_heading: true

::: src.baseline.calibrator.CalibrationResult
    options:
      show_root_heading: true
