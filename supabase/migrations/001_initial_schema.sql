-- Raiden ECM Analysis Database Schema
-- Swedish Building Energy Conservation Measure Analysis

-- Enable PostGIS for geometry support (optional, for building footprints)
-- CREATE EXTENSION IF NOT EXISTS postgis;

-- ============================================
-- BUILDINGS TABLE
-- ============================================
CREATE TABLE IF NOT EXISTS buildings (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),

    -- Identification
    address TEXT NOT NULL,
    org_number TEXT,  -- BRF organization number
    property_designation TEXT,  -- Fastighetsbeteckning

    -- Basic info
    name TEXT,
    building_type TEXT DEFAULT 'multi_family',  -- multi_family, single_family, commercial
    construction_year INTEGER,
    renovation_year INTEGER,

    -- Physical characteristics
    heated_area_m2 REAL,
    num_apartments INTEGER,
    num_floors INTEGER,
    facade_material TEXT,  -- brick, concrete, wood, plaster
    roof_type TEXT,

    -- Location
    municipality TEXT,
    region TEXT,  -- stockholm, gothenburg, malmo, etc.
    latitude REAL,
    longitude REAL,

    -- Energy data (from energy declaration)
    energy_class TEXT,  -- A-G
    declared_energy_kwh_m2 REAL,
    heating_system TEXT,  -- district_heating, heat_pump, electric, etc.
    ventilation_system TEXT,  -- ftx, f, fx, natural

    -- Heritage/constraints
    heritage_listed BOOLEAN DEFAULT FALSE,

    -- Owner info
    owner_type TEXT DEFAULT 'brf',  -- private, brf, rental, commercial

    -- Metadata
    data_source TEXT,  -- manual, api, energy_declaration
    data_quality TEXT DEFAULT 'medium'  -- low, medium, high
);

-- Index for common queries
CREATE INDEX IF NOT EXISTS idx_buildings_address ON buildings(address);
CREATE INDEX IF NOT EXISTS idx_buildings_org_number ON buildings(org_number);
CREATE INDEX IF NOT EXISTS idx_buildings_region ON buildings(region);

-- ============================================
-- BASELINE SIMULATIONS
-- ============================================
CREATE TABLE IF NOT EXISTS baseline_simulations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    building_id UUID REFERENCES buildings(id) ON DELETE CASCADE,
    created_at TIMESTAMPTZ DEFAULT NOW(),

    -- Simulation settings
    weather_file TEXT,
    archetype_id TEXT,

    -- Results
    annual_heating_kwh REAL,
    annual_cooling_kwh REAL,
    annual_electricity_kwh REAL,
    heating_kwh_m2 REAL,

    -- Calibration
    is_calibrated BOOLEAN DEFAULT FALSE,
    calibration_gap_percent REAL,

    -- Model file reference
    idf_file_path TEXT,

    -- Metadata
    energyplus_version TEXT,
    simulation_duration_seconds REAL
);

CREATE INDEX IF NOT EXISTS idx_baseline_building ON baseline_simulations(building_id);

-- ============================================
-- ECM ANALYSIS RESULTS
-- ============================================
CREATE TABLE IF NOT EXISTS ecm_results (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    building_id UUID REFERENCES buildings(id) ON DELETE CASCADE,
    baseline_id UUID REFERENCES baseline_simulations(id) ON DELETE CASCADE,
    created_at TIMESTAMPTZ DEFAULT NOW(),

    -- ECM identification
    ecm_id TEXT NOT NULL,
    ecm_name TEXT,
    ecm_category TEXT,

    -- Applied parameters
    parameters JSONB,

    -- Energy results
    annual_heating_kwh REAL,
    annual_cooling_kwh REAL,
    annual_electricity_kwh REAL,
    heating_kwh_m2 REAL,

    -- Savings
    heating_savings_kwh REAL,
    heating_savings_percent REAL,
    electricity_savings_kwh REAL,
    total_savings_kwh REAL,

    -- Costs (SEK)
    installation_cost REAL,
    material_cost REAL,
    labor_cost REAL,
    total_cost REAL,
    annual_maintenance REAL,

    -- Deductions applied
    rot_deduction REAL DEFAULT 0,
    green_tech_deduction REAL DEFAULT 0,
    net_cost REAL,

    -- Financial metrics
    annual_savings_sek REAL,
    simple_payback_years REAL,
    npv_20yr REAL,
    irr_percent REAL,

    -- CO2
    annual_co2_reduction_kg REAL,

    -- Applicability
    is_applicable BOOLEAN DEFAULT TRUE,
    constraint_violations JSONB,

    -- Simulation reference
    simulated BOOLEAN DEFAULT FALSE,
    idf_file_path TEXT
);

CREATE INDEX IF NOT EXISTS idx_ecm_building ON ecm_results(building_id);
CREATE INDEX IF NOT EXISTS idx_ecm_id ON ecm_results(ecm_id);
CREATE INDEX IF NOT EXISTS idx_ecm_payback ON ecm_results(simple_payback_years);

-- ============================================
-- ECM PACKAGES
-- ============================================
CREATE TABLE IF NOT EXISTS ecm_packages (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    building_id UUID REFERENCES buildings(id) ON DELETE CASCADE,
    created_at TIMESTAMPTZ DEFAULT NOW(),

    -- Package info
    package_name TEXT NOT NULL,  -- grundpaket, standardpaket, premiumpaket
    package_type TEXT,  -- basic, standard, premium, custom

    -- ECMs included
    ecm_ids TEXT[],

    -- Combined results (simulated together)
    combined_heating_kwh_m2 REAL,
    combined_savings_percent REAL,

    -- Synergy factor
    synergy_factor REAL DEFAULT 1.0,

    -- Combined costs
    total_cost REAL,
    package_discount REAL DEFAULT 0,  -- Shared scaffolding etc.
    net_cost REAL,

    -- Financial
    annual_savings_sek REAL,
    simple_payback_years REAL,
    npv_20yr REAL,

    -- Validation
    is_valid BOOLEAN DEFAULT TRUE,
    validation_issues JSONB
);

CREATE INDEX IF NOT EXISTS idx_package_building ON ecm_packages(building_id);
CREATE INDEX IF NOT EXISTS idx_package_type ON ecm_packages(package_type);

-- ============================================
-- ENERGY PRICES (for ROI calculations)
-- ============================================
CREATE TABLE IF NOT EXISTS energy_prices (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    valid_from DATE,
    valid_to DATE,

    -- Location
    region TEXT NOT NULL,
    electricity_zone TEXT,  -- SE1, SE2, SE3, SE4

    -- Prices (SEK/kWh)
    district_heating_price REAL,
    electricity_price REAL,
    natural_gas_price REAL,

    -- Components
    network_fee REAL,
    energy_tax REAL,

    -- Carbon intensity (kg CO2/kWh)
    district_heating_co2 REAL,
    electricity_co2 REAL,

    -- Source
    data_source TEXT
);

CREATE INDEX IF NOT EXISTS idx_energy_region ON energy_prices(region);

-- ============================================
-- ANALYSIS REPORTS
-- ============================================
CREATE TABLE IF NOT EXISTS analysis_reports (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    building_id UUID REFERENCES buildings(id) ON DELETE CASCADE,
    created_at TIMESTAMPTZ DEFAULT NOW(),

    -- Report metadata
    report_type TEXT DEFAULT 'full',  -- quick, full, detailed
    language TEXT DEFAULT 'sv',  -- sv, en

    -- Generated content
    html_content TEXT,
    pdf_url TEXT,

    -- Summary data
    baseline_kwh_m2 REAL,
    best_package TEXT,
    best_package_savings_percent REAL,
    best_package_payback_years REAL,

    -- Recommendations
    top_ecms JSONB,
    packages JSONB
);

CREATE INDEX IF NOT EXISTS idx_report_building ON analysis_reports(building_id);

-- ============================================
-- AUDIT LOG
-- ============================================
CREATE TABLE IF NOT EXISTS audit_log (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    created_at TIMESTAMPTZ DEFAULT NOW(),

    -- What happened
    action TEXT NOT NULL,  -- create, update, delete, simulate
    entity_type TEXT NOT NULL,  -- building, ecm_result, package
    entity_id UUID,

    -- Who/what
    user_id UUID,
    api_key_id TEXT,

    -- Details
    details JSONB
);

CREATE INDEX IF NOT EXISTS idx_audit_entity ON audit_log(entity_type, entity_id);
CREATE INDEX IF NOT EXISTS idx_audit_time ON audit_log(created_at);

-- ============================================
-- UPDATED_AT TRIGGER
-- ============================================
CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER buildings_updated_at
    BEFORE UPDATE ON buildings
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at();

-- ============================================
-- ROW LEVEL SECURITY (RLS)
-- ============================================
-- Enable RLS on all tables
ALTER TABLE buildings ENABLE ROW LEVEL SECURITY;
ALTER TABLE baseline_simulations ENABLE ROW LEVEL SECURITY;
ALTER TABLE ecm_results ENABLE ROW LEVEL SECURITY;
ALTER TABLE ecm_packages ENABLE ROW LEVEL SECURITY;
ALTER TABLE analysis_reports ENABLE ROW LEVEL SECURITY;

-- For now, allow all authenticated access (adjust based on your auth setup)
CREATE POLICY "Allow all access" ON buildings FOR ALL USING (true);
CREATE POLICY "Allow all access" ON baseline_simulations FOR ALL USING (true);
CREATE POLICY "Allow all access" ON ecm_results FOR ALL USING (true);
CREATE POLICY "Allow all access" ON ecm_packages FOR ALL USING (true);
CREATE POLICY "Allow all access" ON analysis_reports FOR ALL USING (true);

-- ============================================
-- VIEWS FOR COMMON QUERIES
-- ============================================

-- Building summary with latest analysis
CREATE OR REPLACE VIEW building_summary AS
SELECT
    b.id,
    b.address,
    b.name,
    b.construction_year,
    b.heated_area_m2,
    b.num_apartments,
    b.declared_energy_kwh_m2,
    b.region,
    bs.heating_kwh_m2 as simulated_kwh_m2,
    bs.is_calibrated,
    (SELECT COUNT(*) FROM ecm_results er WHERE er.building_id = b.id) as ecm_count,
    (SELECT MIN(simple_payback_years) FROM ecm_results er WHERE er.building_id = b.id AND er.is_applicable) as best_payback
FROM buildings b
LEFT JOIN baseline_simulations bs ON bs.building_id = b.id
ORDER BY b.created_at DESC;

-- Top ECMs across all buildings
CREATE OR REPLACE VIEW top_ecms AS
SELECT
    ecm_id,
    ecm_name,
    ecm_category,
    COUNT(*) as application_count,
    AVG(heating_savings_percent) as avg_savings_percent,
    AVG(simple_payback_years) as avg_payback_years,
    AVG(net_cost) as avg_net_cost
FROM ecm_results
WHERE is_applicable = true AND simulated = true
GROUP BY ecm_id, ecm_name, ecm_category
ORDER BY avg_savings_percent DESC;
