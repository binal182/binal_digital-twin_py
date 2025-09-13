-- Comprehensive PostgreSQL Schema for Professional Digital Twin
-- Supports structured data, JSONB flexibility, full-text search, and audit trails

-- Enable necessary extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";

-- ===========================================
-- CORE TABLES
-- ===========================================

-- Personal Information Table
CREATE TABLE personal_info (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    full_name VARCHAR(255) NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    phone VARCHAR(50),
    location VARCHAR(255),
    linkedin_url VARCHAR(500),
    github_url VARCHAR(500),
    website_url VARCHAR(500),
    summary TEXT,
    profile_image_url VARCHAR(500),
    raw_data JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    deleted_at TIMESTAMP WITH TIME ZONE NULL,
    version INTEGER DEFAULT 1,
    data_quality_score INTEGER CHECK (data_quality_score >= 0 AND data_quality_score <= 100)
);

-- Experience Table
CREATE TABLE experience (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    person_id UUID REFERENCES personal_info(id) ON DELETE CASCADE,
    company_name VARCHAR(255) NOT NULL,
    position_title VARCHAR(255) NOT NULL,
    start_date DATE NOT NULL,
    end_date DATE,
    is_current_position BOOLEAN DEFAULT FALSE,
    location VARCHAR(255),
    employment_type VARCHAR(100),
    description TEXT,
    achievements TEXT[],
    technologies_used TEXT[],
    raw_data JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    deleted_at TIMESTAMP WITH TIME ZONE NULL,
    sort_order INTEGER DEFAULT 0
);

-- Education Table
CREATE TABLE education (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    person_id UUID REFERENCES personal_info(id) ON DELETE CASCADE,
    institution_name VARCHAR(255) NOT NULL,
    degree VARCHAR(255) NOT NULL,
    field_of_study VARCHAR(255),
    start_date DATE,
    end_date DATE,
    gpa DECIMAL(3,2),
    honors TEXT[],
    relevant_coursework TEXT[],
    raw_data JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    deleted_at TIMESTAMP WITH TIME ZONE NULL,
    sort_order INTEGER DEFAULT 0
);

-- Skills Table
CREATE TABLE skills (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    person_id UUID REFERENCES personal_info(id) ON DELETE CASCADE,
    skill_name VARCHAR(255) NOT NULL,
    category VARCHAR(100),
    proficiency_level VARCHAR(50) CHECK (proficiency_level IN ('Beginner', 'Intermediate', 'Advanced', 'Expert')),
    years_of_experience DECIMAL(4,1),
    last_used DATE,
    is_technical BOOLEAN DEFAULT TRUE,
    raw_data JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    deleted_at TIMESTAMP WITH TIME ZONE NULL,
    UNIQUE(person_id, skill_name)
);

-- Projects Table
CREATE TABLE projects (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    person_id UUID REFERENCES personal_info(id) ON DELETE CASCADE,
    project_name VARCHAR(255) NOT NULL,
    description TEXT,
    start_date DATE,
    end_date DATE,
    is_ongoing BOOLEAN DEFAULT FALSE,
    project_url VARCHAR(500),
    github_url VARCHAR(500),
    technologies_used TEXT[],
    role VARCHAR(255),
    achievements TEXT[],
    raw_data JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    deleted_at TIMESTAMP WITH TIME ZONE NULL,
    sort_order INTEGER DEFAULT 0
);

-- Certifications Table
CREATE TABLE certifications (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    person_id UUID REFERENCES personal_info(id) ON DELETE CASCADE,
    certification_name VARCHAR(255) NOT NULL,
    issuing_organization VARCHAR(255) NOT NULL,
    issue_date DATE,
    expiry_date DATE,
    credential_id VARCHAR(255),
    credential_url VARCHAR(500),
    is_active BOOLEAN DEFAULT TRUE,
    raw_data JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    deleted_at TIMESTAMP WITH TIME ZONE NULL
);

-- Publications Table
CREATE TABLE publications (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    person_id UUID REFERENCES personal_info(id) ON DELETE CASCADE,
    title VARCHAR(500) NOT NULL,
    authors TEXT[],
    publication_date DATE,
    journal_or_conference VARCHAR(255),
    doi VARCHAR(255),
    url VARCHAR(500),
    abstract TEXT,
    keywords TEXT[],
    raw_data JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    deleted_at TIMESTAMP WITH TIME ZONE NULL
);

-- Awards Table
CREATE TABLE awards (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    person_id UUID REFERENCES personal_info(id) ON DELETE CASCADE,
    award_name VARCHAR(255) NOT NULL,
    issuing_organization VARCHAR(255),
    award_date DATE,
    description TEXT,
    category VARCHAR(100),
    raw_data JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    deleted_at TIMESTAMP WITH TIME ZONE NULL
);

-- Professional Network Table
CREATE TABLE professional_network (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    person_id UUID REFERENCES personal_info(id) ON DELETE CASCADE,
    contact_name VARCHAR(255) NOT NULL,
    company VARCHAR(255),
    position VARCHAR(255),
    relationship_type VARCHAR(100),
    linkedin_url VARCHAR(500),
    email VARCHAR(255),
    notes TEXT,
    raw_data JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    deleted_at TIMESTAMP WITH TIME ZONE NULL
);

-- ===========================================
-- RAG AND SEARCH TABLES
-- ===========================================

-- Content Chunks for RAG
CREATE TABLE content_chunks (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    person_id UUID REFERENCES personal_info(id) ON DELETE CASCADE,
    chunk_type VARCHAR(100) NOT NULL,
    content TEXT NOT NULL,
    embedding_vector VECTOR(1536), -- Adjust dimension based on your embedding model
    metadata JSONB,
    source_table VARCHAR(100),
    source_id UUID,
    chunk_index INTEGER,
    total_chunks INTEGER,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    deleted_at TIMESTAMP WITH TIME ZONE NULL
);

-- Search Keywords Table
CREATE TABLE search_keywords (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    person_id UUID REFERENCES personal_info(id) ON DELETE CASCADE,
    keyword VARCHAR(255) NOT NULL,
    category VARCHAR(100),
    weight DECIMAL(3,2) DEFAULT 1.0,
    frequency INTEGER DEFAULT 1,
    raw_data JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    deleted_at TIMESTAMP WITH TIME ZONE NULL,
    UNIQUE(person_id, keyword)
);

-- Industry Focus Table
CREATE TABLE industry_focus (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    person_id UUID REFERENCES personal_info(id) ON DELETE CASCADE,
    industry_name VARCHAR(255) NOT NULL,
    focus_level VARCHAR(50) CHECK (focus_level IN ('Primary', 'Secondary', 'Emerging')),
    description TEXT,
    raw_data JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    deleted_at TIMESTAMP WITH TIME ZONE NULL,
    UNIQUE(person_id, industry_name)
);

-- Career Trajectory Table
CREATE TABLE career_trajectory (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    person_id UUID REFERENCES personal_info(id) ON DELETE CASCADE,
    current_stage VARCHAR(255),
    target_role VARCHAR(255),
    unique_value_proposition TEXT,
    career_progression TEXT,
    raw_data JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    deleted_at TIMESTAMP WITH TIME ZONE NULL
);

-- Metadata Table
CREATE TABLE digital_twin_metadata (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    person_id UUID REFERENCES personal_info(id) ON DELETE CASCADE,
    version VARCHAR(50) DEFAULT '1.0',
    data_quality_score INTEGER CHECK (data_quality_score >= 0 AND data_quality_score <= 100),
    completeness JSONB,
    data_sources TEXT[],
    ai_readiness JSONB,
    last_sync TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    raw_data JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    deleted_at TIMESTAMP WITH TIME ZONE NULL
);

-- ===========================================
-- RELATIONSHIP TABLES
-- ===========================================

-- Experience-Skills Junction Table
CREATE TABLE experience_skills (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    experience_id UUID REFERENCES experience(id) ON DELETE CASCADE,
    skill_id UUID REFERENCES skills(id) ON DELETE CASCADE,
    proficiency_during_experience VARCHAR(50),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(experience_id, skill_id)
);

-- Project-Skills Junction Table
CREATE TABLE project_skills (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    project_id UUID REFERENCES projects(id) ON DELETE CASCADE,
    skill_id UUID REFERENCES skills(id) ON DELETE CASCADE,
    role_in_project VARCHAR(100),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(project_id, skill_id)
);

-- ===========================================
-- INDEXES FOR PERFORMANCE
-- ===========================================

-- Personal Info Indexes
CREATE INDEX idx_personal_info_email ON personal_info(email);
CREATE INDEX idx_personal_info_deleted_at ON personal_info(deleted_at) WHERE deleted_at IS NULL;
CREATE INDEX idx_personal_info_raw_data ON personal_info USING GIN (raw_data);

-- Experience Indexes
CREATE INDEX idx_experience_person_id ON experience(person_id);
CREATE INDEX idx_experience_company ON experience(company_name);
CREATE INDEX idx_experience_dates ON experience(start_date, end_date);
CREATE INDEX idx_experience_current ON experience(is_current_position) WHERE is_current_position = TRUE;
CREATE INDEX idx_experience_raw_data ON experience USING GIN (raw_data);
CREATE INDEX idx_experience_technologies ON experience USING GIN (technologies_used);
CREATE INDEX idx_experience_achievements ON experience USING GIN (achievements);

-- Education Indexes
CREATE INDEX idx_education_person_id ON education(person_id);
CREATE INDEX idx_education_institution ON education(institution_name);
CREATE INDEX idx_education_dates ON education(start_date, end_date);
CREATE INDEX idx_education_raw_data ON education USING GIN (raw_data);

-- Skills Indexes
CREATE INDEX idx_skills_person_id ON skills(person_id);
CREATE INDEX idx_skills_category ON skills(category);
CREATE INDEX idx_skills_proficiency ON skills(proficiency_level);
CREATE INDEX idx_skills_name_trgm ON skills USING GIN (skill_name gin_trgm_ops);
CREATE INDEX idx_skills_raw_data ON skills USING GIN (raw_data);

-- Projects Indexes
CREATE INDEX idx_projects_person_id ON projects(person_id);
CREATE INDEX idx_projects_dates ON projects(start_date, end_date);
CREATE INDEX idx_projects_technologies ON projects USING GIN (technologies_used);
CREATE INDEX idx_projects_achievements ON projects USING GIN (achievements);
CREATE INDEX idx_projects_raw_data ON projects USING GIN (raw_data);

-- Certifications Indexes
CREATE INDEX idx_certifications_person_id ON certifications(person_id);
CREATE INDEX idx_certifications_active ON certifications(is_active) WHERE is_active = TRUE;
CREATE INDEX idx_certifications_expiry ON certifications(expiry_date);
CREATE INDEX idx_certifications_raw_data ON certifications USING GIN (raw_data);

-- Publications Indexes
CREATE INDEX idx_publications_person_id ON publications(person_id);
CREATE INDEX idx_publications_date ON publications(publication_date);
CREATE INDEX idx_publications_keywords ON publications USING GIN (keywords);
CREATE INDEX idx_publications_raw_data ON publications USING GIN (raw_data);

-- Awards Indexes
CREATE INDEX idx_awards_person_id ON awards(person_id);
CREATE INDEX idx_awards_date ON awards(award_date);
CREATE INDEX idx_awards_raw_data ON awards USING GIN (raw_data);

-- Professional Network Indexes
CREATE INDEX idx_professional_network_person_id ON professional_network(person_id);
CREATE INDEX idx_professional_network_company ON professional_network(company);
CREATE INDEX idx_professional_network_relationship ON professional_network(relationship_type);
CREATE INDEX idx_professional_network_raw_data ON professional_network USING GIN (raw_data);

-- Content Chunks Indexes
CREATE INDEX idx_content_chunks_person_id ON content_chunks(person_id);
CREATE INDEX idx_content_chunks_type ON content_chunks(chunk_type);
CREATE INDEX idx_content_chunks_source ON content_chunks(source_table, source_id);
CREATE INDEX idx_content_chunks_metadata ON content_chunks USING GIN (metadata);
CREATE INDEX idx_content_chunks_vector ON content_chunks USING ivfflat (embedding_vector vector_cosine_ops) WITH (lists = 100);

-- Search Keywords Indexes
CREATE INDEX idx_search_keywords_person_id ON search_keywords(person_id);
CREATE INDEX idx_search_keywords_category ON search_keywords(category);
CREATE INDEX idx_search_keywords_weight ON search_keywords(weight);
CREATE INDEX idx_search_keywords_name_trgm ON search_keywords USING GIN (keyword gin_trgm_ops);
CREATE INDEX idx_search_keywords_raw_data ON search_keywords USING GIN (raw_data);

-- Industry Focus Indexes
CREATE INDEX idx_industry_focus_person_id ON industry_focus(person_id);
CREATE INDEX idx_industry_focus_level ON industry_focus(focus_level);
CREATE INDEX idx_industry_focus_raw_data ON industry_focus USING GIN (raw_data);

-- Career Trajectory Indexes
CREATE INDEX idx_career_trajectory_person_id ON career_trajectory(person_id);
CREATE INDEX idx_career_trajectory_raw_data ON career_trajectory USING GIN (raw_data);

-- Metadata Indexes
CREATE INDEX idx_metadata_person_id ON digital_twin_metadata(person_id);
CREATE INDEX idx_metadata_quality ON digital_twin_metadata(data_quality_score);
CREATE INDEX idx_metadata_completeness ON digital_twin_metadata USING GIN (completeness);
CREATE INDEX idx_metadata_raw_data ON digital_twin_metadata USING GIN (raw_data);

-- Junction Table Indexes
CREATE INDEX idx_experience_skills_experience ON experience_skills(experience_id);
CREATE INDEX idx_experience_skills_skill ON experience_skills(skill_id);
CREATE INDEX idx_project_skills_project ON project_skills(project_id);
CREATE INDEX idx_project_skills_skill ON project_skills(skill_id);

-- ===========================================
-- FULL-TEXT SEARCH INDEXES
-- ===========================================

-- Full-text search on experience descriptions and achievements
CREATE INDEX idx_experience_fts ON experience
USING GIN (to_tsvector('english', COALESCE(description, '') || ' ' || array_to_string(achievements, ' ')));

-- Full-text search on project descriptions and achievements
CREATE INDEX idx_projects_fts ON projects
USING GIN (to_tsvector('english', COALESCE(description, '') || ' ' || array_to_string(achievements, ' ')));

-- Full-text search on publications
CREATE INDEX idx_publications_fts ON publications
USING GIN (to_tsvector('english', COALESCE(title, '') || ' ' || COALESCE(abstract, '') || ' ' || array_to_string(keywords, ' ')));

-- Full-text search on content chunks
CREATE INDEX idx_content_chunks_fts ON content_chunks
USING GIN (to_tsvector('english', content));

-- ===========================================
-- TRIGGERS FOR AUDIT TRAILS
-- ===========================================

-- Function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Apply update trigger to all tables
CREATE TRIGGER update_personal_info_updated_at BEFORE UPDATE ON personal_info FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_experience_updated_at BEFORE UPDATE ON experience FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_education_updated_at BEFORE UPDATE ON education FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_skills_updated_at BEFORE UPDATE ON skills FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_projects_updated_at BEFORE UPDATE ON projects FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_certifications_updated_at BEFORE UPDATE ON certifications FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_publications_updated_at BEFORE UPDATE ON publications FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_awards_updated_at BEFORE UPDATE ON awards FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_professional_network_updated_at BEFORE UPDATE ON professional_network FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_content_chunks_updated_at BEFORE UPDATE ON content_chunks FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_search_keywords_updated_at BEFORE UPDATE ON search_keywords FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_industry_focus_updated_at BEFORE UPDATE ON industry_focus FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_career_trajectory_updated_at BEFORE UPDATE ON career_trajectory FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_digital_twin_metadata_updated_at BEFORE UPDATE ON digital_twin_metadata FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- ===========================================
-- VIEWS FOR COMMON QUERIES
-- ===========================================

-- Complete profile view
CREATE VIEW complete_profile AS
SELECT
    pi.*,
    json_build_object(
        'experience', (
            SELECT json_agg(
                json_build_object(
                    'id', e.id,
                    'company_name', e.company_name,
                    'position_title', e.position_title,
                    'start_date', e.start_date,
                    'end_date', e.end_date,
                    'is_current_position', e.is_current_position,
                    'description', e.description,
                    'achievements', e.achievements,
                    'technologies_used', e.technologies_used
                ) ORDER BY e.sort_order, e.start_date DESC
            )
            FROM experience e WHERE e.person_id = pi.id AND e.deleted_at IS NULL
        ),
        'education', (
            SELECT json_agg(
                json_build_object(
                    'id', ed.id,
                    'institution_name', ed.institution_name,
                    'degree', ed.degree,
                    'field_of_study', ed.field_of_study,
                    'start_date', ed.start_date,
                    'end_date', ed.end_date,
                    'gpa', ed.gpa,
                    'honors', ed.honors
                ) ORDER BY ed.sort_order, ed.start_date DESC
            )
            FROM education ed WHERE ed.person_id = pi.id AND ed.deleted_at IS NULL
        ),
        'skills', (
            SELECT json_agg(
                json_build_object(
                    'id', s.id,
                    'skill_name', s.skill_name,
                    'category', s.category,
                    'proficiency_level', s.proficiency_level,
                    'years_of_experience', s.years_of_experience
                ) ORDER BY s.category, s.skill_name
            )
            FROM skills s WHERE s.person_id = pi.id AND s.deleted_at IS NULL
        ),
        'projects', (
            SELECT json_agg(
                json_build_object(
                    'id', p.id,
                    'project_name', p.project_name,
                    'description', p.description,
                    'start_date', p.start_date,
                    'end_date', p.end_date,
                    'technologies_used', p.technologies_used,
                    'achievements', p.achievements
                ) ORDER BY p.sort_order, p.start_date DESC
            )
            FROM projects p WHERE p.person_id = pi.id AND p.deleted_at IS NULL
        )
    ) as profile_data
FROM personal_info pi
WHERE pi.deleted_at IS NULL;

-- Skills summary view
CREATE VIEW skills_summary AS
SELECT
    pi.id as person_id,
    pi.full_name,
    s.category,
    COUNT(*) as skill_count,
    ARRAY_AGG(s.skill_name ORDER BY s.skill_name) as skills,
    AVG(CASE
        WHEN s.proficiency_level = 'Beginner' THEN 1
        WHEN s.proficiency_level = 'Intermediate' THEN 2
        WHEN s.proficiency_level = 'Advanced' THEN 3
        WHEN s.proficiency_level = 'Expert' THEN 4
        ELSE 2
    END) as avg_proficiency_score
FROM personal_info pi
JOIN skills s ON pi.id = s.person_id
WHERE pi.deleted_at IS NULL AND s.deleted_at IS NULL
GROUP BY pi.id, pi.full_name, s.category;

-- Experience timeline view
CREATE VIEW experience_timeline AS
SELECT
    pi.id as person_id,
    pi.full_name,
    e.id as experience_id,
    e.company_name,
    e.position_title,
    e.start_date,
    e.end_date,
    e.is_current_position,
    EXTRACT(YEAR FROM AGE(COALESCE(e.end_date, CURRENT_DATE), e.start_date)) * 12 +
    EXTRACT(MONTH FROM AGE(COALESCE(e.end_date, CURRENT_DATE), e.start_date)) as months_duration,
    ROW_NUMBER() OVER (PARTITION BY pi.id ORDER BY e.start_date DESC) as recency_rank
FROM personal_info pi
JOIN experience e ON pi.id = e.person_id
WHERE pi.deleted_at IS NULL AND e.deleted_at IS NULL
ORDER BY pi.id, e.start_date DESC;