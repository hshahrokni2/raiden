-- Portfolio Analysis Tables
-- For RaidenOrchestrator portfolio-scale analysis

-- ============================================
-- PORTFOLIOS TABLE
-- ============================================
CREATE TABLE IF NOT EXISTS portfolios (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),

    -- Portfolio info
    name TEXT NOT NULL,
    description TEXT,
    owner_id UUID,  -- User who created the portfolio

    -- Settings
    skip_energy_classes TEXT[] DEFAULT ARRAY['A', 'B'],  -- Energy classes to skip
    standard_workers INTEGER DEFAULT 50,
    deep_workers INTEGER DEFAULT 10,

    -- Aggregated metrics
    total_buildings INTEGER DEFAULT 0,
    analyzed_buildings INTEGER DEFAULT 0,
    skipped_buildings INTEGER DEFAULT 0,
    failed_buildings INTEGER DEFAULT 0,

    -- Portfolio-level savings
    total_savings_potential_kwh REAL DEFAULT 0,
    total_investment_sek REAL DEFAULT 0,
    portfolio_npv_sek REAL DEFAULT 0,
    portfolio_payback_years REAL,

    -- Status
    status TEXT DEFAULT 'pending',  -- pending, in_progress, completed, failed
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    error_message TEXT
);

CREATE INDEX IF NOT EXISTS idx_portfolios_status ON portfolios(status);
CREATE INDEX IF NOT EXISTS idx_portfolios_owner ON portfolios(owner_id);

-- ============================================
-- PORTFOLIO_BUILDINGS (junction table)
-- ============================================
CREATE TABLE IF NOT EXISTS portfolio_buildings (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    portfolio_id UUID REFERENCES portfolios(id) ON DELETE CASCADE,
    building_id UUID REFERENCES buildings(id) ON DELETE CASCADE,
    created_at TIMESTAMPTZ DEFAULT NOW(),

    -- Analysis result
    tier TEXT,  -- skip, fast, standard, deep
    analysis_status TEXT DEFAULT 'pending',  -- pending, in_progress, completed, failed

    -- Building-specific results
    baseline_kwh_m2 REAL,
    savings_kwh_m2 REAL,
    savings_percent REAL,
    investment_sek REAL,
    payback_years REAL,
    npv_sek REAL,

    -- Archetype
    archetype_id TEXT,
    archetype_confidence REAL,

    -- QC flags
    needs_qc BOOLEAN DEFAULT FALSE,
    qc_triggers TEXT[],
    qc_completed BOOLEAN DEFAULT FALSE,
    qc_result JSONB,

    -- Recommended ECMs (stored as JSONB array)
    recommended_ecms JSONB DEFAULT '[]'::JSONB,

    -- Uncertainty
    savings_uncertainty_kwh_m2 REAL,
    savings_ci_90_lower REAL,
    savings_ci_90_upper REAL,

    -- Priority/ranking
    roi_rank INTEGER,
    payback_rank INTEGER,

    -- Processing
    processing_time_sec REAL,
    error_message TEXT,

    UNIQUE(portfolio_id, building_id)
);

CREATE INDEX IF NOT EXISTS idx_portfolio_buildings_portfolio ON portfolio_buildings(portfolio_id);
CREATE INDEX IF NOT EXISTS idx_portfolio_buildings_building ON portfolio_buildings(building_id);
CREATE INDEX IF NOT EXISTS idx_portfolio_buildings_tier ON portfolio_buildings(tier);
CREATE INDEX IF NOT EXISTS idx_portfolio_buildings_status ON portfolio_buildings(analysis_status);

-- ============================================
-- PORTFOLIO_REPORTS
-- ============================================
CREATE TABLE IF NOT EXISTS portfolio_reports (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    portfolio_id UUID REFERENCES portfolios(id) ON DELETE CASCADE,
    created_at TIMESTAMPTZ DEFAULT NOW(),

    -- Report metadata
    report_format TEXT DEFAULT 'html',  -- html, markdown, json, pdf
    language TEXT DEFAULT 'sv',

    -- Content
    content TEXT,  -- For HTML/Markdown
    json_data JSONB,  -- For JSON format
    file_url TEXT,  -- For PDF or external storage

    -- Summary metrics
    summary JSONB  -- Aggregated stats, top buildings, etc.
);

CREATE INDEX IF NOT EXISTS idx_portfolio_reports_portfolio ON portfolio_reports(portfolio_id);

-- ============================================
-- QC_LOGS
-- ============================================
CREATE TABLE IF NOT EXISTS qc_logs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    portfolio_building_id UUID REFERENCES portfolio_buildings(id) ON DELETE CASCADE,
    created_at TIMESTAMPTZ DEFAULT NOW(),

    -- QC details
    trigger_type TEXT NOT NULL,  -- low_wwr_confidence, negative_savings, etc.
    agent_type TEXT NOT NULL,  -- image_qc, ecm_refiner, anomaly

    -- Result
    success BOOLEAN DEFAULT FALSE,
    action_taken TEXT,
    explanation TEXT,

    -- Updates made
    updated_values JSONB,
    new_confidence REAL,

    -- Flags
    needs_human_review BOOLEAN DEFAULT FALSE,
    escalated BOOLEAN DEFAULT FALSE,

    -- Recommendations
    recommendations TEXT[]
);

CREATE INDEX IF NOT EXISTS idx_qc_logs_portfolio_building ON qc_logs(portfolio_building_id);
CREATE INDEX IF NOT EXISTS idx_qc_logs_trigger ON qc_logs(trigger_type);

-- ============================================
-- SURROGATE_MODELS
-- ============================================
CREATE TABLE IF NOT EXISTS surrogate_models (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    created_at TIMESTAMPTZ DEFAULT NOW(),

    -- Model info
    archetype_id TEXT NOT NULL UNIQUE,
    model_type TEXT DEFAULT 'gaussian_process',

    -- Training metrics
    n_samples INTEGER,
    train_r2 REAL,
    test_r2 REAL,
    train_rmse_kwh_m2 REAL,
    test_rmse_kwh_m2 REAL,

    -- Overfitting detection
    has_overfitting_warning BOOLEAN DEFAULT FALSE,

    -- Parameter bounds (JSONB)
    param_bounds JSONB,

    -- Model storage
    model_blob BYTEA,  -- Pickled model
    model_path TEXT,  -- Or path to file

    -- Metadata
    trained_date DATE,
    training_time_sec REAL
);

CREATE INDEX IF NOT EXISTS idx_surrogate_models_archetype ON surrogate_models(archetype_id);

-- ============================================
-- FUNCTIONS
-- ============================================

-- Update portfolio aggregates when portfolio_buildings changes
CREATE OR REPLACE FUNCTION update_portfolio_aggregates()
RETURNS TRIGGER AS $$
BEGIN
    UPDATE portfolios SET
        total_buildings = (
            SELECT COUNT(*) FROM portfolio_buildings
            WHERE portfolio_id = COALESCE(NEW.portfolio_id, OLD.portfolio_id)
        ),
        analyzed_buildings = (
            SELECT COUNT(*) FROM portfolio_buildings
            WHERE portfolio_id = COALESCE(NEW.portfolio_id, OLD.portfolio_id)
            AND analysis_status = 'completed'
        ),
        skipped_buildings = (
            SELECT COUNT(*) FROM portfolio_buildings
            WHERE portfolio_id = COALESCE(NEW.portfolio_id, OLD.portfolio_id)
            AND tier = 'skip'
        ),
        failed_buildings = (
            SELECT COUNT(*) FROM portfolio_buildings
            WHERE portfolio_id = COALESCE(NEW.portfolio_id, OLD.portfolio_id)
            AND analysis_status = 'failed'
        ),
        total_savings_potential_kwh = (
            SELECT COALESCE(SUM(savings_kwh_m2 * b.heated_area_m2), 0)
            FROM portfolio_buildings pb
            JOIN buildings b ON pb.building_id = b.id
            WHERE pb.portfolio_id = COALESCE(NEW.portfolio_id, OLD.portfolio_id)
            AND pb.analysis_status = 'completed'
        ),
        total_investment_sek = (
            SELECT COALESCE(SUM(investment_sek), 0)
            FROM portfolio_buildings
            WHERE portfolio_id = COALESCE(NEW.portfolio_id, OLD.portfolio_id)
            AND analysis_status = 'completed'
        ),
        portfolio_npv_sek = (
            SELECT COALESCE(SUM(npv_sek), 0)
            FROM portfolio_buildings
            WHERE portfolio_id = COALESCE(NEW.portfolio_id, OLD.portfolio_id)
            AND analysis_status = 'completed'
        ),
        updated_at = NOW()
    WHERE id = COALESCE(NEW.portfolio_id, OLD.portfolio_id);

    RETURN COALESCE(NEW, OLD);
END;
$$ LANGUAGE plpgsql;

-- Trigger for updating aggregates
DROP TRIGGER IF EXISTS trigger_update_portfolio_aggregates ON portfolio_buildings;
CREATE TRIGGER trigger_update_portfolio_aggregates
AFTER INSERT OR UPDATE OR DELETE ON portfolio_buildings
FOR EACH ROW EXECUTE FUNCTION update_portfolio_aggregates();

-- Calculate portfolio payback when aggregates change
CREATE OR REPLACE FUNCTION update_portfolio_payback()
RETURNS TRIGGER AS $$
DECLARE
    annual_savings_sek REAL;
    energy_price_sek REAL := 1.50;  -- Average Swedish energy price SEK/kWh
BEGIN
    -- Calculate annual savings from kWh savings
    annual_savings_sek := NEW.total_savings_potential_kwh * energy_price_sek;

    -- Calculate simple payback
    IF annual_savings_sek > 0 THEN
        NEW.portfolio_payback_years := NEW.total_investment_sek / annual_savings_sek;
    ELSE
        NEW.portfolio_payback_years := NULL;
    END IF;

    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trigger_update_portfolio_payback ON portfolios;
CREATE TRIGGER trigger_update_portfolio_payback
BEFORE UPDATE ON portfolios
FOR EACH ROW EXECUTE FUNCTION update_portfolio_payback();
