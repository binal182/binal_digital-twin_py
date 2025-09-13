-- Sample INSERT statements for importing data from binal_mytwin.json
-- These statements show how to populate the database with your professional data

-- ===========================================
-- SAMPLE INSERT STATEMENTS
-- ===========================================

-- Insert Personal Information
INSERT INTO personal_info (
    full_name, email, phone, location, linkedin_url, github_url, website_url,
    summary, profile_image_url, raw_data, data_quality_score
) VALUES (
    'Binal shah',
    'binal@example.com',
    '+1-555-0123',
    'San Francisco, CA',
    'https://linkedin.com/in/binal-patel',
    'https://github.com/binal-patel',
    'https://binal-patel.dev',
    'AI Data Engineering Intern with expertise in machine learning, data engineering, and human-centered AI development. Proven track record in leadership and technical innovation.',
    'https://example.com/profile.jpg',
    '{
        "personalInfo": {
            "name": "Binal Patel",
            "email": "binal@example.com",
            "location": "San Francisco, CA",
            "linkedin": "https://linkedin.com/in/binal-patel",
            "github": "https://github.com/binal-patel",
            "website": "https://binal-patel.dev"
        }
    }'::jsonb,
    98
);

-- Get the person_id for subsequent inserts
-- In practice, you'd store this in a variable or use a CTE

-- Insert Experience Records
INSERT INTO experience (
    person_id, company_name, position_title, start_date, end_date,
    is_current_position, location, employment_type, description,
    achievements, technologies_used, raw_data, sort_order
) VALUES
(
    (SELECT id FROM personal_info WHERE email = 'binal@example.com' LIMIT 1),
    'TechCorp Solutions',
    'AI Data Engineering Intern',
    '2024-06-01',
    NULL,
    TRUE,
    'San Francisco, CA',
    'Internship',
    'Leading AI data engineering initiatives, developing machine learning pipelines, and implementing data processing solutions.',
    ARRAY[
        'Developed automated data pipelines processing 1M+ records daily',
        'Implemented ML models with 95% accuracy for customer segmentation',
        'Reduced data processing time by 60% through optimization'
    ],
    ARRAY['Python', 'TensorFlow', 'Apache Spark', 'PostgreSQL', 'AWS', 'Docker'],
    '{"source": "binal_mytwin.json", "section": "experience", "index": 0}'::jsonb,
    1
),
(
    (SELECT id FROM personal_info WHERE email = 'binal@example.com' LIMIT 1),
    'DataFlow Inc',
    'Data Engineering Intern',
    '2024-01-01',
    '2024-05-31',
    FALSE,
    'Remote',
    'Internship',
    'Built ETL pipelines and data warehousing solutions for enterprise clients.',
    ARRAY[
        'Created ETL pipelines handling 500K+ daily transactions',
        'Implemented data quality monitoring reducing errors by 40%',
        'Collaborated with cross-functional teams on data architecture'
    ],
    ARRAY['Python', 'SQL', 'Apache Airflow', 'Snowflake', 'Tableau'],
    '{"source": "binal_mytwin.json", "section": "experience", "index": 1}'::jsonb,
    2
);

-- Insert Education Records
INSERT INTO education (
    person_id, institution_name, degree, field_of_study, start_date, end_date,
    gpa, honors, relevant_coursework, raw_data, sort_order
) VALUES
(
    (SELECT id FROM personal_info WHERE email = 'binal@example.com' LIMIT 1),
    'University of California, Berkeley',
    'Bachelor of Science',
    'Data Science and Engineering',
    '2020-08-01',
    '2024-05-15',
    3.8,
    ARRAY['Dean''s List', 'Data Science Honor Society'],
    ARRAY['Machine Learning', 'Database Systems', 'Statistics', 'Algorithms'],
    '{"source": "binal_mytwin.json", "section": "education", "index": 0}'::jsonb,
    1
);

-- Insert Skills
INSERT INTO skills (
    person_id, skill_name, category, proficiency_level,
    years_of_experience, last_used, is_technical, raw_data
) VALUES
(
    (SELECT id FROM personal_info WHERE email = 'binal@example.com' LIMIT 1),
    'Python',
    'Programming Languages',
    'Expert',
    3.5,
    '2024-12-01',
    TRUE,
    '{"source": "binal_mytwin.json", "section": "skills"}'::jsonb
),
(
    (SELECT id FROM personal_info WHERE email = 'binal@example.com' LIMIT 1),
    'Machine Learning',
    'AI/ML',
    'Advanced',
    2.0,
    '2024-12-01',
    TRUE,
    '{"source": "binal_mytwin.json", "section": "skills"}'::jsonb
),
(
    (SELECT id FROM personal_info WHERE email = 'binal@example.com' LIMIT 1),
    'PostgreSQL',
    'Databases',
    'Advanced',
    2.5,
    '2024-12-01',
    TRUE,
    '{"source": "binal_mytwin.json", "section": "skills"}'::jsonb
),
(
    (SELECT id FROM personal_info WHERE email = 'binal@example.com' LIMIT 1),
    'Leadership',
    'Soft Skills',
    'Expert',
    4.0,
    '2024-12-01',
    FALSE,
    '{"source": "binal_mytwin.json", "section": "skills"}'::jsonb
);

-- Insert Projects
INSERT INTO projects (
    person_id, project_name, description, start_date, end_date,
    is_ongoing, project_url, github_url, technologies_used,
    role, achievements, raw_data, sort_order
) VALUES
(
    (SELECT id FROM personal_info WHERE email = 'binal@example.com' LIMIT 1),
    'AI-Powered Customer Analytics Platform',
    'Developed an end-to-end analytics platform using machine learning to provide customer insights and predictive analytics.',
    '2024-08-01',
    '2024-11-30',
    FALSE,
    'https://github.com/binal-patel/customer-analytics',
    'https://github.com/binal-patel/customer-analytics',
    ARRAY['Python', 'TensorFlow', 'React', 'PostgreSQL', 'AWS'],
    'Lead Developer',
    ARRAY[
        'Built ML models achieving 92% prediction accuracy',
        'Processed 10M+ customer records with 99.9% uptime',
        'Reduced customer churn prediction time by 75%'
    ],
    '{"source": "binal_mytwin.json", "section": "projects", "index": 0}'::jsonb,
    1
);

-- Insert Certifications
INSERT INTO certifications (
    person_id, certification_name, issuing_organization, issue_date,
    expiry_date, credential_id, credential_url, is_active, raw_data
) VALUES
(
    (SELECT id FROM personal_info WHERE email = 'binal@example.com' LIMIT 1),
    'AWS Certified Machine Learning - Specialty',
    'Amazon Web Services',
    '2024-03-15',
    '2027-03-15',
    'AWS-ML-123456',
    'https://aws.amazon.com/verification',
    TRUE,
    '{"source": "binal_mytwin.json", "section": "certifications"}'::jsonb
),
(
    (SELECT id FROM personal_info WHERE email = 'binal@example.com' LIMIT 1),
    'Google Cloud Professional Data Engineer',
    'Google Cloud',
    '2024-01-20',
    '2026-01-20',
    'GCP-DE-789012',
    'https://cloud.google.com/certification',
    TRUE,
    '{"source": "binal_mytwin.json", "section": "certifications"}'::jsonb
);

-- Insert Search Keywords
INSERT INTO search_keywords (
    person_id, keyword, category, weight, frequency, raw_data
) VALUES
(
    (SELECT id FROM personal_info WHERE email = 'binal@example.com' LIMIT 1),
    'machine learning',
    'Technical Skills',
    1.0,
    15,
    '{"source": "binal_mytwin.json", "section": "search_keywords"}'::jsonb
),
(
    (SELECT id FROM personal_info WHERE email = 'binal@example.com' LIMIT 1),
    'data engineering',
    'Technical Skills',
    0.9,
    12,
    '{"source": "binal_mytwin.json", "section": "search_keywords"}'::jsonb
),
(
    (SELECT id FROM personal_info WHERE email = 'binal@example.com' LIMIT 1),
    'AI development',
    'Technical Skills',
    0.8,
    10,
    '{"source": "binal_mytwin.json", "section": "search_keywords"}'::jsonb
);

-- Insert Industry Focus
INSERT INTO industry_focus (
    person_id, industry_name, focus_level, description, raw_data
) VALUES
(
    (SELECT id FROM personal_info WHERE email = 'binal@example.com' LIMIT 1),
    'AI and Data Engineering',
    'Primary',
    'Core expertise in developing AI solutions and data engineering pipelines',
    '{"source": "binal_mytwin.json", "section": "industry_focus"}'::jsonb
),
(
    (SELECT id FROM personal_info WHERE email = 'binal@example.com' LIMIT 1),
    'HR Technology Innovation',
    'Secondary',
    'Experience in HR tech solutions and employee experience platforms',
    '{"source": "binal_mytwin.json", "section": "industry_focus"}'::jsonb
);

-- Insert Career Trajectory
INSERT INTO career_trajectory (
    person_id, current_stage, target_role, unique_value_proposition,
    career_progression, raw_data
) VALUES (
    (SELECT id FROM personal_info WHERE email = 'binal@example.com' LIMIT 1),
    'AI Data Engineering Intern transitioning to full-time role',
    'AI Data Engineer/Architect',
    'Combines cutting-edge AI technical expertise with proven leadership skills and human-centered design approach',
    'Customer Service → Team Leadership → AI Development → Target: AI Data Engineer/Architect',
    '{"source": "binal_mytwin.json", "section": "career_trajectory"}'::jsonb
);

-- Insert Metadata
INSERT INTO digital_twin_metadata (
    person_id, version, data_quality_score, completeness,
    data_sources, ai_readiness, raw_data
) VALUES (
    (SELECT id FROM personal_info WHERE email = 'binal@example.com' LIMIT 1),
    '1.0',
    98,
    '{
        "experience": "100%",
        "skills": "100%",
        "projects": "100%",
        "education": "100%",
        "personal_info": "95%"
    }'::jsonb,
    ARRAY[
        'Direct interview responses',
        'Resume documentation',
        'Project specifications',
        'Technical implementation details',
        'Performance metrics and achievements'
    ],
    '{
        "rag_optimized": true,
        "searchable_content": true,
        "detailed_responses": true,
        "contextual_chunks": true
    }'::jsonb,
    '{"source": "binal_mytwin.json", "section": "metadata"}'::jsonb
);

-- ===========================================
-- COMMON QUERY EXAMPLES
-- ===========================================

-- 1. Get complete profile with all related data
SELECT * FROM complete_profile WHERE full_name = 'Binal Patel';

-- 2. Search for skills by category
SELECT skill_name, proficiency_level, years_of_experience
FROM skills
WHERE person_id = (SELECT id FROM personal_info WHERE email = 'binal@example.com' LIMIT 1)
  AND category = 'Programming Languages'
  AND deleted_at IS NULL
ORDER BY proficiency_level DESC, years_of_experience DESC;

-- 3. Find experience with specific technologies
SELECT company_name, position_title, start_date, end_date,
       technologies_used
FROM experience
WHERE person_id = (SELECT id FROM personal_info WHERE email = 'binal@example.com' LIMIT 1)
  AND 'Python' = ANY(technologies_used)
  AND deleted_at IS NULL
ORDER BY start_date DESC;

-- 4. Full-text search across experience and projects
SELECT 'experience' as source_type, id, company_name as title,
       description, ts_rank_cd(to_tsvector('english', description), query) as rank
FROM experience, plainto_tsquery('english', 'machine learning') query
WHERE person_id = (SELECT id FROM personal_info WHERE email = 'binal@example.com' LIMIT 1)
  AND to_tsvector('english', description) @@ query
  AND deleted_at IS NULL

UNION ALL

SELECT 'project' as source_type, id, project_name as title,
       description, ts_rank_cd(to_tsvector('english', description), query) as rank
FROM projects, plainto_tsquery('english', 'machine learning') query
WHERE person_id = (SELECT id FROM personal_info WHERE email = 'binal@example.com' LIMIT 1)
  AND to_tsvector('english', description) @@ query
  AND deleted_at IS NULL
ORDER BY rank DESC;

-- 5. Get skills summary by category
SELECT * FROM skills_summary
WHERE person_id = (SELECT id FROM personal_info WHERE email = 'binal@example.com' LIMIT 1);

-- 6. Find most recent projects
SELECT project_name, description, start_date, end_date,
       technologies_used, achievements
FROM projects
WHERE person_id = (SELECT id FROM personal_info WHERE email = 'binal@example.com' LIMIT 1)
  AND deleted_at IS NULL
ORDER BY COALESCE(end_date, CURRENT_DATE) DESC, start_date DESC
LIMIT 5;

-- 7. Search for content chunks by semantic similarity (requires pgvector)
-- Note: This requires the pgvector extension and actual embedding vectors
SELECT content, metadata, chunk_type
FROM content_chunks
WHERE person_id = (SELECT id FROM personal_info WHERE email = 'binal@example.com' LIMIT 1)
  AND deleted_at IS NULL
ORDER BY embedding_vector <=> '[your_embedding_vector_here]'::vector
LIMIT 5;

-- 8. Get experience timeline
SELECT * FROM experience_timeline
WHERE person_id = (SELECT id FROM personal_info WHERE email = 'binal@example.com' LIMIT 1)
ORDER BY start_date DESC;

-- 9. Find certifications that are still active
SELECT certification_name, issuing_organization, issue_date, expiry_date
FROM certifications
WHERE person_id = (SELECT id FROM personal_info WHERE email = 'binal@example.com' LIMIT 1)
  AND is_active = TRUE
  AND (expiry_date IS NULL OR expiry_date > CURRENT_DATE)
  AND deleted_at IS NULL
ORDER BY expiry_date ASC NULLS LAST;

-- 10. JSONB queries for flexible data access
-- Search within raw_data JSONB fields
SELECT id, skill_name, raw_data->'source' as data_source
FROM skills
WHERE person_id = (SELECT id FROM personal_info WHERE email = 'binal@example.com' LIMIT 1)
  AND raw_data @> '{"source": "binal_mytwin.json"}'::jsonb
  AND deleted_at IS NULL;

-- 11. Aggregate queries for analytics
SELECT
    COUNT(*) as total_experiences,
    AVG(EXTRACT(YEAR FROM AGE(COALESCE(end_date, CURRENT_DATE), start_date))) as avg_years_per_role,
    SUM(CASE WHEN is_current_position THEN 1 ELSE 0 END) as current_positions
FROM experience
WHERE person_id = (SELECT id FROM personal_info WHERE email = 'binal@example.com' LIMIT 1)
  AND deleted_at IS NULL;

-- 12. Complex search combining multiple tables
WITH search_results AS (
    SELECT 'experience' as type, id, company_name as title, description,
           to_tsvector('english', description) as document
    FROM experience
    WHERE person_id = (SELECT id FROM personal_info WHERE email = 'binal@example.com' LIMIT 1)
      AND deleted_at IS NULL

    UNION ALL

    SELECT 'project' as type, id, project_name as title, description,
           to_tsvector('english', description) as document
    FROM projects
    WHERE person_id = (SELECT id FROM personal_info WHERE email = 'binal@example.com' LIMIT 1)
      AND deleted_at IS NULL

    UNION ALL

    SELECT 'publication' as type, id, title, abstract as description,
           to_tsvector('english', title || ' ' || COALESCE(abstract, '')) as document
    FROM publications
    WHERE person_id = (SELECT id FROM personal_info WHERE email = 'binal@example.com' LIMIT 1)
      AND deleted_at IS NULL
)
SELECT type, id, title, description,
       ts_rank_cd(document, query) as relevance_score
FROM search_results, plainto_tsquery('english', 'data engineering') query
WHERE document @@ query
ORDER BY relevance_score DESC
LIMIT 10;