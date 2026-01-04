-- Migration: Add columns needed for store_analysis method
-- These columns support the full FullPipelineAnalyzer result storage

-- ============================================
-- BUILDINGS TABLE ADDITIONS
-- ============================================

-- Data sources used for this building (array of sources like 'sweden_geojson', 'mapillary', etc.)
ALTER TABLE buildings ADD COLUMN IF NOT EXISTS data_sources TEXT[];

-- Data quality/confidence score (0-1)
ALTER TABLE buildings ADD COLUMN IF NOT EXISTS data_confidence REAL;

-- ============================================
-- BASELINE_SIMULATIONS ADDITIONS
-- ============================================

-- Calibration gap percentage (difference from declared energy)
ALTER TABLE baseline_simulations ADD COLUMN IF NOT EXISTS calibration_gap REAL;

-- Calibrated parameters from Bayesian calibration
ALTER TABLE baseline_simulations ADD COLUMN IF NOT EXISTS calibrated_infiltration REAL;
ALTER TABLE baseline_simulations ADD COLUMN IF NOT EXISTS calibrated_heat_recovery REAL;
ALTER TABLE baseline_simulations ADD COLUMN IF NOT EXISTS calibrated_window_u REAL;

-- Uncertainty from calibration
ALTER TABLE baseline_simulations ADD COLUMN IF NOT EXISTS kwh_m2_std REAL;

-- ============================================
-- ECM_RESULTS ADDITIONS
-- ============================================

-- Rename category to match code expectations (create alias)
-- Note: ecm_category already exists, we'll use it

-- Baseline reference for comparison
ALTER TABLE ecm_results ADD COLUMN IF NOT EXISTS baseline_kwh_m2 REAL;

-- Result energy after ECM (renamed from heating_kwh_m2 for clarity)
ALTER TABLE ecm_results ADD COLUMN IF NOT EXISTS result_kwh_m2 REAL;

-- Savings per m²
ALTER TABLE ecm_results ADD COLUMN IF NOT EXISTS savings_kwh_m2 REAL;

-- Savings percentage
ALTER TABLE ecm_results ADD COLUMN IF NOT EXISTS savings_percent REAL;

-- Investment cost (total after deductions)
ALTER TABLE ecm_results ADD COLUMN IF NOT EXISTS investment_sek REAL;

-- ============================================
-- ECM_PACKAGES ADDITIONS
-- ============================================

-- Combined savings as kWh/m² (matches code)
ALTER TABLE ecm_packages ADD COLUMN IF NOT EXISTS combined_kwh_m2 REAL;

-- ============================================
-- UPDATE store_analysis COLUMN MAPPINGS
-- ============================================

-- The store_analysis method uses these column names:
-- buildings:
--   - address ✓
--   - construction_year ✓
--   - heated_area_m2 ✓
--   - energy_class ✓
--   - heating_system ✓
--   - ventilation_system ✓
--   - facade_material ✓
--   - num_floors ✓
--   - num_apartments ✓
--   - declared_energy_kwh_m2 ✓
--   - data_sources (NEW)
--   - data_confidence (NEW)

-- baseline_simulations:
--   - building_id ✓
--   - archetype_id ✓
--   - heating_kwh_m2 ✓
--   - calibration_gap (NEW)
--   - calibrated_infiltration (NEW)
--   - calibrated_heat_recovery (NEW)
--   - calibrated_window_u (NEW)

-- ecm_results:
--   - building_id ✓
--   - ecm_id ✓
--   - ecm_name ✓
--   - category → ecm_category ✓
--   - is_applicable ✓
--   - baseline_kwh_m2 (NEW)
--   - result_kwh_m2 (NEW) or heating_kwh_m2
--   - savings_kwh_m2 (NEW)
--   - savings_percent (NEW)
--   - investment_sek (NEW)
--   - annual_savings_sek ✓
--   - simple_payback_years ✓

-- ecm_packages:
--   - building_id ✓
--   - package_name ✓
--   - ecm_ids ✓
--   - combined_savings_kwh_m2 ✓ or combined_kwh_m2
--   - combined_savings_percent ✓
--   - total_investment_sek → total_cost ✓
--   - annual_savings_sek ✓
--   - simple_payback_years ✓

-- ============================================
-- CREATE INDEXES
-- ============================================

CREATE INDEX IF NOT EXISTS idx_buildings_data_confidence ON buildings(data_confidence);
CREATE INDEX IF NOT EXISTS idx_ecm_results_savings ON ecm_results(savings_percent);
