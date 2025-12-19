# Simulation Module

The simulation module handles EnergyPlus execution and results parsing.

## Simulation Runner

Execute EnergyPlus simulations with error handling and timeout support.

::: src.simulation.runner.SimulationRunner
    options:
      show_root_heading: true

::: src.simulation.runner.SimulationResult
    options:
      show_root_heading: true

::: src.simulation.runner.run_simulation
    options:
      show_root_heading: true

## Results Parser

Parse EnergyPlus output files (eplustbl.csv, eplusout.eso).

::: src.simulation.results.ResultsParser
    options:
      show_root_heading: true

::: src.simulation.results.AnnualResults
    options:
      show_root_heading: true
