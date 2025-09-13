-- Migration Script for Professional Digital Twin Database
-- This script handles schema updates, data migrations, and rollbacks

-- ===========================================
-- MIGRATION VERSION CONTROL
-- ===========================================

CREATE TABLE IF NOT EXISTS schema_migrations (
    id SERIAL PRIMARY KEY,
    migration_name VARCHAR(255) NOT NULL UNIQUE,
    applied_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    success BOOLEAN DEFAULT FALSE,
    error_message TEXT
);

-- Function to record migration
CREATE OR REPLACE FUNCTION record_migration(migration_name TEXT, success BOOLEAN, error_msg TEXT DEFAULT NULL)
RETURNS VOID AS $$
BEGIN
    INSERT INTO schema_migrations (migration_name, success, error_message)
    VALUES (migration_name, success, error_msg);
END;
$$ LANGUAGE plpgsql;

-- ===========================================
-- MIGRATION: 001_initial_schema
-- ===========================================

DO $$
DECLARE
    migration_success BOOLEAN := TRUE;
    error_msg TEXT := NULL;
BEGIN
    BEGIN
        -- Check if migration already applied
        IF EXISTS (SELECT 1 FROM schema_migrations WHERE migration_name = '001_initial_schema' AND success = TRUE) THEN
            RAISE NOTICE 'Migration 001_initial_schema already applied, skipping...';
            RETURN;
        END IF;

        -- Enable required extensions
        CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
        CREATE EXTENSION IF NOT EXISTS "pg_trgm";

        -- Create all tables (from comprehensive_schema.sql)
        -- [Tables creation code would go here - abbreviated for brevity]

        -- Record successful migration
        PERFORM record_migration('001_initial_schema', TRUE);

        RAISE NOTICE 'Migration 001_initial_schema completed successfully';

    EXCEPTION WHEN OTHERS THEN
        migration_success := FALSE;
        error_msg := SQLERRM;
        PERFORM record_migration('001_initial_schema', FALSE, error_msg);
        RAISE EXCEPTION 'Migration 001_initial_schema failed: %', error_msg;
    END;
END $$;

-- ===========================================
-- MIGRATION: 002_add_indexes
-- ===========================================

DO $$
DECLARE
    migration_success BOOLEAN := TRUE;
    error_msg TEXT := NULL;
BEGIN
    BEGIN
        IF EXISTS (SELECT 1 FROM schema_migrations WHERE migration_name = '002_add_indexes' AND success = TRUE) THEN
            RAISE NOTICE 'Migration 002_add_indexes already applied, skipping...';
            RETURN;
        END IF;

        -- Add all performance indexes
        -- [Index creation code would go here - abbreviated for brevity]

        PERFORM record_migration('002_add_indexes', TRUE);
        RAISE NOTICE 'Migration 002_add_indexes completed successfully';

    EXCEPTION WHEN OTHERS THEN
        migration_success := FALSE;
        error_msg := SQLERRM;
        PERFORM record_migration('002_add_indexes', FALSE, error_msg);
        RAISE EXCEPTION 'Migration 002_add_indexes failed: %', error_msg;
    END;
END $$;

-- ===========================================
-- MIGRATION: 003_add_triggers
-- ===========================================

DO $$
DECLARE
    migration_success BOOLEAN := TRUE;
    error_msg TEXT := NULL;
BEGIN
    BEGIN
        IF EXISTS (SELECT 1 FROM schema_migrations WHERE migration_name = '003_add_triggers' AND success = TRUE) THEN
            RAISE NOTICE 'Migration 003_add_triggers already applied, skipping...';
            RETURN;
        END IF;

        -- Create update trigger function
        CREATE OR REPLACE FUNCTION update_updated_at_column()
        RETURNS TRIGGER AS $$
        BEGIN
            NEW.updated_at = CURRENT_TIMESTAMP;
            RETURN NEW;
        END;
        $$ language 'plpgsql';

        -- Apply triggers to all tables
        -- [Trigger creation code would go here - abbreviated for brevity]

        PERFORM record_migration('003_add_triggers', TRUE);
        RAISE NOTICE 'Migration 003_add_triggers completed successfully';

    EXCEPTION WHEN OTHERS THEN
        migration_success := FALSE;
        error_msg := SQLERRM;
        PERFORM record_migration('003_add_triggers', FALSE, error_msg);
        RAISE EXCEPTION 'Migration 003_add_triggers failed: %', error_msg;
    END;
END $$;

-- ===========================================
-- MIGRATION: 004_create_views
-- ===========================================

DO $$
DECLARE
    migration_success BOOLEAN := TRUE;
    error_msg TEXT := NULL;
BEGIN
    BEGIN
        IF EXISTS (SELECT 1 FROM schema_migrations WHERE migration_name = '004_create_views' AND success = TRUE) THEN
            RAISE NOTICE 'Migration 004_create_views already applied, skipping...';
            RETURN;
        END IF;

        -- Create views for common queries
        -- [View creation code would go here - abbreviated for brevity]

        PERFORM record_migration('004_create_views', TRUE);
        RAISE NOTICE 'Migration 004_create_views completed successfully';

    EXCEPTION WHEN OTHERS THEN
        migration_success := FALSE;
        error_msg := SQLERRM;
        PERFORM record_migration('004_create_views', FALSE, error_msg);
        RAISE EXCEPTION 'Migration 004_create_views failed: %', error_msg;
    END;
END $$;

-- ===========================================
-- MIGRATION: 005_sample_data_import
-- ===========================================

DO $$
DECLARE
    migration_success BOOLEAN := TRUE;
    error_msg TEXT := NULL;
    person_id UUID;
BEGIN
    BEGIN
        IF EXISTS (SELECT 1 FROM schema_migrations WHERE migration_name = '005_sample_data_import' AND success = TRUE) THEN
            RAISE NOTICE 'Migration 005_sample_data_import already applied, skipping...';
            RETURN;
        END IF;

        -- Insert sample data (from sample_data_and_queries.sql)
        -- This would include all the INSERT statements for populating the database

        PERFORM record_migration('005_sample_data_import', TRUE);
        RAISE NOTICE 'Migration 005_sample_data_import completed successfully';

    EXCEPTION WHEN OTHERS THEN
        migration_success := FALSE;
        error_msg := SQLERRM;
        PERFORM record_migration('005_sample_data_import', FALSE, error_msg);
        RAISE EXCEPTION 'Migration 005_sample_data_import failed: %', error_msg;
    END;
END $$;

-- ===========================================
-- UTILITY FUNCTIONS
-- ===========================================

-- Function to get current schema version
CREATE OR REPLACE FUNCTION get_current_schema_version()
RETURNS TABLE(migration_name TEXT, applied_at TIMESTAMP WITH TIME ZONE) AS $$
BEGIN
    RETURN QUERY
    SELECT sm.migration_name, sm.applied_at
    FROM schema_migrations sm
    WHERE sm.success = TRUE
    ORDER BY sm.applied_at DESC;
END;
$$ LANGUAGE plpgsql;

-- Function to check if migration is applied
CREATE OR REPLACE FUNCTION is_migration_applied(migration_name TEXT)
RETURNS BOOLEAN AS $$
BEGIN
    RETURN EXISTS (
        SELECT 1 FROM schema_migrations
        WHERE schema_migrations.migration_name = is_migration_applied.migration_name
        AND success = TRUE
    );
END;
$$ LANGUAGE plpgsql;

-- ===========================================
-- ROLLBACK FUNCTIONS
-- ===========================================

-- Function to rollback a specific migration
CREATE OR REPLACE FUNCTION rollback_migration(target_migration TEXT)
RETURNS VOID AS $$
DECLARE
    migration_record RECORD;
BEGIN
    -- Find the migration
    SELECT * INTO migration_record
    FROM schema_migrations
    WHERE migration_name = target_migration AND success = TRUE;

    IF NOT FOUND THEN
        RAISE EXCEPTION 'Migration % not found or not successfully applied', target_migration;
    END IF;

    -- Perform rollback based on migration name
    CASE target_migration
        WHEN '005_sample_data_import' THEN
            -- Rollback sample data
            DELETE FROM digital_twin_metadata WHERE version = '1.0';
            DELETE FROM career_trajectory;
            DELETE FROM industry_focus;
            DELETE FROM search_keywords;
            DELETE FROM certifications;
            DELETE FROM projects;
            DELETE FROM skills;
            DELETE FROM education;
            DELETE FROM experience;
            DELETE FROM personal_info WHERE email = 'binal@example.com';

        WHEN '004_create_views' THEN
            -- Drop views
            DROP VIEW IF EXISTS experience_timeline;
            DROP VIEW IF EXISTS skills_summary;
            DROP VIEW IF EXISTS complete_profile;

        WHEN '003_add_triggers' THEN
            -- Drop triggers (simplified - would need to drop each trigger individually)
            DROP FUNCTION IF EXISTS update_updated_at_column();

        WHEN '002_add_indexes' THEN
            -- Drop indexes (simplified - would need to drop each index individually)
            -- This is complex and would require listing all indexes

        WHEN '001_initial_schema' THEN
            -- Drop all tables
            DROP TABLE IF EXISTS project_skills CASCADE;
            DROP TABLE IF EXISTS experience_skills CASCADE;
            DROP TABLE IF EXISTS digital_twin_metadata CASCADE;
            DROP TABLE IF EXISTS career_trajectory CASCADE;
            DROP TABLE IF EXISTS industry_focus CASCADE;
            DROP TABLE IF EXISTS search_keywords CASCADE;
            DROP TABLE IF EXISTS content_chunks CASCADE;
            DROP TABLE IF EXISTS professional_network CASCADE;
            DROP TABLE IF EXISTS awards CASCADE;
            DROP TABLE IF EXISTS publications CASCADE;
            DROP TABLE IF EXISTS certifications CASCADE;
            DROP TABLE IF EXISTS projects CASCADE;
            DROP TABLE IF EXISTS skills CASCADE;
            DROP TABLE IF EXISTS education CASCADE;
            DROP TABLE IF EXISTS experience CASCADE;
            DROP TABLE IF EXISTS personal_info CASCADE;
            DROP TABLE IF EXISTS schema_migrations CASCADE;

        ELSE
            RAISE EXCEPTION 'Unknown migration: %', target_migration;
    END CASE;

    -- Mark migration as rolled back
    UPDATE schema_migrations
    SET success = FALSE, error_message = 'Rolled back on ' || CURRENT_TIMESTAMP
    WHERE migration_name = target_migration;

    RAISE NOTICE 'Successfully rolled back migration: %', target_migration;
END;
$$ LANGUAGE plpgsql;

-- ===========================================
-- DATA VALIDATION FUNCTIONS
-- ===========================================

-- Function to validate data integrity
CREATE OR REPLACE FUNCTION validate_data_integrity()
RETURNS TABLE(table_name TEXT, record_count BIGINT, issues TEXT[]) AS $$
BEGIN
    RETURN QUERY
    SELECT
        'personal_info'::TEXT,
        COUNT(*)::BIGINT,
        ARRAY[]::TEXT[]
    FROM personal_info
    WHERE deleted_at IS NULL

    UNION ALL

    SELECT
        'experience'::TEXT,
        COUNT(*)::BIGINT,
        ARRAY_AGG(CASE WHEN end_date < start_date THEN 'Invalid date range' ELSE NULL END) FILTER (WHERE end_date < start_date IS NOT NULL)
    FROM experience
    WHERE deleted_at IS NULL

    UNION ALL

    SELECT
        'skills'::TEXT,
        COUNT(*)::BIGINT,
        ARRAY[]::TEXT[]
    FROM skills
    WHERE deleted_at IS NULL;
END;
$$ LANGUAGE plpgsql;

-- ===========================================
-- MONITORING AND MAINTENANCE
-- ===========================================

-- Function to get database statistics
CREATE OR REPLACE FUNCTION get_database_stats()
RETURNS TABLE(table_name TEXT, row_count BIGINT, size_mb NUMERIC) AS $$
BEGIN
    RETURN QUERY
    SELECT
        schemaname || '.' || tablename as table_name,
        n_tup_ins - n_tup_del as row_count,
        pg_total_relation_size(schemaname || '.' || tablename)::NUMERIC / 1024 / 1024 as size_mb
    FROM pg_stat_user_tables
    WHERE schemaname = 'public'
    ORDER BY size_mb DESC;
END;
$$ LANGUAGE plpgsql;

-- Function to rebuild all indexes
CREATE OR REPLACE FUNCTION rebuild_all_indexes()
RETURNS VOID AS $$
DECLARE
    index_record RECORD;
BEGIN
    FOR index_record IN
        SELECT indexname, tablename
        FROM pg_indexes
        WHERE schemaname = 'public'
        AND indexname NOT LIKE 'pg_%'
    LOOP
        EXECUTE 'REINDEX INDEX ' || index_record.indexname;
        RAISE NOTICE 'Reindexed: % on table %', index_record.indexname, index_record.tablename;
    END LOOP;
END;
$$ LANGUAGE plpgsql;

-- ===========================================
-- USAGE EXAMPLES
-- =========================================--

-- Check current schema version:
-- SELECT * FROM get_current_schema_version();

-- Check if specific migration is applied:
-- SELECT is_migration_applied('001_initial_schema');

-- Rollback a migration:
-- SELECT rollback_migration('005_sample_data_import');

-- Validate data integrity:
-- SELECT * FROM validate_data_integrity();

-- Get database statistics:
-- SELECT * FROM get_database_stats();

-- Rebuild all indexes:
-- SELECT rebuild_all_indexes();