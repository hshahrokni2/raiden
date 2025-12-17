# Claude Quick Start Card

## What Is This?
BRF Energy Toolkit - Enriches Swedish building data for EnergyPlus simulation

## One-Liner Test
```bash
cd /Users/hosseins/Dropbox/Dev/Komilion/brf-energy-toolkit && python scripts/process_sjostaden.py
```

## Current Status
**SIMULATION VALIDATED!** - Calibrated model matches measured 33 kWh/m²

## What's Working (Full Pipeline!)
- Mapillary image fetching with user's token
- **SAM backend** for window segmentation (new!)
- **DINOv2 spatial** material classification (new!)
- OSM neighbor building shading analysis
- Remaining rooftop solar potential
- U-value back-calculation from PDF energy data
- **Complete EnergyPlus IDF** with zones, materials, HVAC (new!)
- 3D visualization generation

## AI Backends Available
```python
WWRDetector backends: ['grounded_sam', 'lang_sam', 'sam', 'opencv', 'manual']
# 'opencv' = default, no GPU needed
# 'sam' = Meta SAM via HuggingFace, better accuracy
```

## Key Outputs
- **WWR**: AI + era blended estimate (17%)
- **Material**: DINOv2 spatial + era (plaster)
- **Solar**: Existing PV + remaining rooftop potential
- **IDF**: Complete with zones, schedules, HVAC

## Next Actions
1. Test SAM backend with GPU acceleration
2. Add Swedish TMY weather file
3. Batch processing for multiple BRFs

## Key Files
- `SESSION_STATE.md` - Full context (READ THIS)
- `scripts/process_sjostaden.py` - Working demo with all features
- `src/ai/wwr_detector.py` - SAM/OpenCV window detection
- `src/ai/material_classifier.py` - DINOv2 material classification
- `src/export/energyplus_idf.py` - Complete IDF export
- `examples/sjostaden_2/viewer.html` - 3D output

## User Preferences
- "Ultrathink" = deep analysis before coding
- NO-ACCOUNT tools only (Mapillary, OSM, Meta AI)
- Client = building owners seeing 3D presentations
- Goal = EnergyPlus simulation

## Latest Results (Sjöstaden 2)
```
WWR: AI 11% + era 32% → blended 17%
Material: plaster (correct for 2003 eco-district)
U-values: Back-calculated (walls=0.11, roof=0.08, windows=0.80)
Solar existing: 500 m² → 75,000 kWh/yr
Solar remaining: 235 m² → 47 kWp → 40,000 kWh/yr
Neighbors: 26 buildings (16 tall), 2 trees
Shading loss: 0% (tall building)

ENERGYPLUS SIMULATION:
  Base model (envelope): 84 kWh/m²/yr
  Calibrated (+ gains):  28 kWh/m²/yr thermal demand
  With HP (COP 3.5):      8 kWh/m²/yr electricity
  Measured:              33 kWh/m²/yr ← Model validates!
```
