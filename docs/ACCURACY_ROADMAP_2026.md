# Raiden Accuracy Roadmap 2026

## Goal: 90% Simulation Accuracy for Swedish BRF Energy Investments

Based on 2025-2026 research literature, this roadmap prioritizes improvements
by ROI (accuracy gain per engineering effort).

---

## Phase 1: Quick Wins (HIGH PRIORITY)

Target: ±10% → ±5% prediction error

| Component | Current | Target | Status | File |
|-----------|---------|--------|--------|------|
| Surrogate Validation | No split | 80/20 + test R² | TODO | `src/calibration/surrogate.py` |
| Calibration Samples | 80 | 200 | TODO | `src/calibration/pipeline.py` |
| Final E+ Verification | None | Run E+ with MAP | TODO | `src/calibration/pipeline.py` |
| MC Uncertainty | sqrt(2) | Full propagation | DONE | `src/calibration/bayesian.py` |
| WWR Detection | OpenCV 70% | SOLOv2 93% | TODO | `src/ai/wwr_detector_v2.py` |

### 1.1 Surrogate Train/Test Validation

**Problem**: Current surrogate shows R²=1.0 which indicates overfitting.

**Solution**:
```python
# In SurrogateBuilder.build()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
gp.fit(X_train, y_train)

train_r2 = gp.score(X_train, y_train)
test_r2 = gp.score(X_test, y_test)

# Flag overfitting
if train_r2 - test_r2 > 0.10:
    logger.warning(f"Surrogate overfitting: train R²={train_r2:.3f}, test R²={test_r2:.3f}")
```

**Acceptance Criteria**:
- [ ] Test R² reported separately from train R²
- [ ] Warning logged if gap > 0.10
- [ ] Test added to `tests/test_calibration.py`

### 1.2 Increase Calibration Samples

**Problem**: 80 samples may be insufficient for 7-parameter calibration.

**Solution**: Change default from 80 to 200 in `CalibrationPipeline`.

**Literature**: Chong & Menberg (2018) recommend 10-20 samples per parameter.
For 7 parameters: 70-140 minimum, 200 recommended.

### 1.3 Final E+ Verification Run

**Problem**: Currently only surrogate prediction is reported, not ground truth.

**Solution**: After calibration, run one E+ simulation with MAP parameters.

```python
# After MCMC completes
map_params = calibrator.get_map_estimate()
final_result = run_energyplus(idf_path, map_params)

# Compare to surrogate prediction
surrogate_pred = surrogate.predict(map_params)
discrepancy = abs(final_result - surrogate_pred) / final_result
if discrepancy > 0.05:
    logger.warning(f"Surrogate-E+ discrepancy: {discrepancy:.1%}")
```

### 1.4 SOLOv2 WWR Detection

**Problem**: OpenCV edge detection achieves ~70% accuracy on window detection.

**Solution**: Upgrade to SOLOv2 instance segmentation (93% mAP in literature).

**Implementation**: `src/ai/wwr_detector_v2.py`

**Hardware**: Requires GPU with 4GB+ VRAM for inference.

---

## Phase 2: Medium Priority (Next Quarter)

Target: Handle high-stakes buildings (>10 MSEK investments)

| Component | Description | Status | File |
|-----------|-------------|--------|------|
| ABC-SMC Calibration | Likelihood-free for complex models | TODO | `src/calibration/abc_smc.py` |
| CLIP+DINOv2 Ensemble | Material accuracy 70%→80% | TODO | `src/ai/material_ensemble.py` |
| Hybrid Calibration | Surrogate + E+ verification | TODO | `src/calibration/hybrid.py` |

### 2.1 ABC-SMC Calibration

**When to use**: Buildings where surrogate may not generalize (unusual construction,
mixed-use, non-standard HVAC).

**Method**: Approximate Bayesian Computation with Sequential Monte Carlo.
Runs actual E+ simulations, accepts samples where |simulated - measured| < epsilon.

**Cost**: 2,000-5,000 E+ runs vs 200 for surrogate approach.

### 2.2 CLIP + DINOv2 Material Ensemble

**Problem**: Single model material classification plateaus at ~70-75%.

**Solution**: Ensemble of:
- DINOv2 (current) - texture/pattern features
- CLIP - semantic understanding ("this looks like concrete")
- Color histogram - simple but effective for brick vs plaster

### 2.3 Hybrid Calibration

**When to use**: Contractual guarantees, ESCO projects, >10 MSEK investments.

**Method**:
1. Train surrogate on 200 E+ runs
2. Run MCMC on surrogate (50,000 samples)
3. Take top 500 posterior samples
4. Run actual E+ on all 500
5. Reweight posterior based on E+/surrogate discrepancy

**Result**: Ground-truth posterior from actual E+ runs.

---

## Phase 3: Future (When Market Demands)

| Component | Description | When |
|-----------|-------------|------|
| Graph Neural Network | Urban-scale (1000+ buildings) | Portfolio product |
| Physics-Informed NN | Embed thermal RC equations | Commercial/industrial |
| Real-time Digital Twin | IoT/BMS integration | Smart building market |

---

## Metrics & Targets

### Calibration Quality (ASHRAE Guideline 14)

| Resolution | NMBE Limit | CV(RMSE) Limit | Current | Target |
|------------|------------|----------------|---------|--------|
| Monthly | ±5% | 15% | ~15% | <10% |
| Annual | ±10% | - | ~10% | <5% |

### Surrogate Quality

| Metric | Current | Target |
|--------|---------|--------|
| Train R² | ~0.99 | 0.95-0.99 |
| Test R² | Unknown | 0.85-0.95 |
| Train-Test Gap | Unknown | <0.10 |

### Computer Vision

| Component | Current | Target | Method |
|-----------|---------|--------|--------|
| WWR Detection | 70% | 93% | SOLOv2 |
| Material Classification | 70% | 80% | CLIP+DINOv2 ensemble |

---

## Implementation Checklist

### Phase 1 (This Sprint)
- [ ] Add train/test split to `SurrogateBuilder`
- [ ] Report test R² in calibration results
- [ ] Change default n_samples from 80 to 200
- [ ] Add final E+ verification after calibration
- [ ] Create `wwr_detector_v2.py` with SOLOv2

### Phase 2 (Next Sprint)
- [ ] Create `abc_smc.py` with ABC-SMC implementation
- [ ] Create `material_ensemble.py` with CLIP+DINOv2
- [ ] Create `hybrid.py` with hybrid calibration
- [ ] Add CLI flag `--calibration-method {surrogate,abc,hybrid}`

### Testing
- [ ] Test surrogate validation catches overfitting
- [ ] Test E+ verification runs after calibration
- [ ] Benchmark SOLOv2 vs OpenCV on test images
- [ ] Integration test for full pipeline

---

## References

1. ASHRAE Guideline 14-2002: Calibration metrics
2. Chong & Menberg (2018): Bayesian calibration guidelines
3. SOLOv2 (Wang et al. 2020): Instance segmentation
4. ABC-SMC (Sisson et al. 2007): Likelihood-free inference
5. Kennedy & O'Hagan (2001): Bayesian calibration framework
