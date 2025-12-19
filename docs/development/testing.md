# Testing Guide

## Test Structure

```
tests/
├── conftest.py           # Shared fixtures
├── test_archetypes.py    # Archetype matching tests
├── test_calibrator.py    # Calibration unit tests
├── test_ecm_modifiers.py # ECM modification tests
├── test_integration.py   # Full simulation tests
├── test_pv_potential.py  # PV calculation tests
├── test_results.py       # Results parser tests
├── test_runner.py        # Simulation runner tests
└── fixtures/
    └── stockholm.epw     # Weather file for integration tests
```

## Running Tests

```bash
# Unit tests only (fast, no EnergyPlus needed)
pytest tests/ -m "not integration" -v

# Integration tests only
pytest tests/ -m "integration" -v

# Specific test file
pytest tests/test_calibrator.py -v

# Specific test
pytest tests/test_calibrator.py::TestParameterExtraction -v
```

## Test Markers

- `@pytest.mark.integration` - Requires EnergyPlus installation
- No marker - Unit test, runs without external dependencies

## Writing Tests

### Unit Test Example

```python
def test_archetype_matching(self):
    matcher = ArchetypeMatcher()
    archetype = matcher.match(construction_year=1968)

    assert archetype is not None
    assert 1961 <= 1968 <= archetype.era_end
```

### Integration Test Example

```python
@pytest.mark.integration
def test_simulation_runs(self, model_file, weather_file):
    runner = SimulationRunner()
    result = runner.run(model_file, weather_file, tmp_path)

    assert result.success
```

## Coverage

```bash
# Generate coverage report
pytest tests/ --cov=src --cov-report=html

# View report
open htmlcov/index.html
```
