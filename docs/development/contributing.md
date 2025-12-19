# Contributing

## Development Setup

```bash
# Clone and install
git clone https://github.com/komilion/raiden.git
cd raiden
pip install -e ".[dev,docs]"

# Install pre-commit hooks
pre-commit install
```

## Code Style

- **Formatter**: Ruff
- **Linter**: Ruff
- **Type checker**: mypy

```bash
# Format code
ruff format src/ tests/

# Check linting
ruff check src/ tests/

# Type check
mypy src/
```

## Testing

```bash
# Run unit tests
pytest tests/ -m "not integration" -v

# Run all tests (requires EnergyPlus)
pytest tests/ -v

# With coverage
pytest tests/ --cov=src --cov-report=html
```

## Pull Request Process

1. Create feature branch from `develop`
2. Write tests for new functionality
3. Ensure all tests pass
4. Update documentation if needed
5. Submit PR with clear description
