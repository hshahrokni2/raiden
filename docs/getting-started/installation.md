# Installation

## Requirements

- Python 3.10+
- EnergyPlus 25.1.0

## Install Raiden

```bash
# Clone repository
git clone https://github.com/komilion/raiden.git
cd raiden

# Install with dev dependencies
pip install -e ".[dev]"
```

## Install EnergyPlus

Download EnergyPlus 25.1.0 from the [official website](https://energyplus.net/downloads).

### macOS
```bash
# After downloading the .dmg installer
sudo mv /Applications/EnergyPlus-25-1-0 /usr/local/
```

### Linux
```bash
wget https://github.com/NREL/EnergyPlus/releases/download/v25.1.0/EnergyPlus-25.1.0-68a4a7c774-Linux-Ubuntu22.04-x86_64.tar.gz
tar -xzf EnergyPlus-*.tar.gz
sudo mv EnergyPlus-* /usr/local/EnergyPlus-25-1-0
export PATH=$PATH:/usr/local/EnergyPlus-25-1-0
```

## Verify Installation

```bash
# Check EnergyPlus
energyplus --version

# Run tests
pytest tests/ -m "not integration" -v
```

## Weather Files

Download Swedish weather files from [EnergyPlus Weather](https://energyplus.net/weather):

- Stockholm: `SWE_Stockholm.Arlanda.024600_IWEC.epw`
- Gothenburg: `SWE_Goteborg.Landvetter.025130_IWEC.epw`
- Malmo: `SWE_Malmo.Sturup.026360_IWEC.epw`
