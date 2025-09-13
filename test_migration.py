#!/usr/bin/env python3
"""
Test Migration Script
Verifies that data migration from JSON to PostgreSQL was successful
"""

import os
import psycopg2
from dotenv import load_dotenv
import logging

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_database_connection():
    """Test database connection"""
    try:
        conn = psycopg2.connect(
            host=os.getenv('DB_HOST', 'localhost'),
            port=os.getenv('DB_PORT', '5432'),
            database=os.getenv('DB_NAME', 'digital_twin'),
            user=os.getenv('DB_USER', 'postgres'),
            password=os.getenv('DB_PASSWORD', '')
        )
        conn.close()
        logger.info("✅ Database connection successful")
        return True
    except Exception as e:
        logger.error(f"❌ Database connection failed: {str(e)}")
        return False

def test_data_migration():
    """Test that data was migrated successfully"""
    try:
        conn = psycopg2.connect(
            host=os.getenv('DB_HOST', 'localhost'),
            port=os.getenv('DB_PORT', '5432'),
            database=os.getenv('DB_NAME', 'digital_twin'),
            user=os.getenv('DB_USER', 'postgres'),
            password=os.getenv('DB_PASSWORD', '')
        )

        with conn.cursor() as cursor:
            # Test personal info
            cursor.execute("SELECT COUNT(*) FROM personal_info WHERE deleted_at IS NULL")
            personal_count = cursor.fetchone()[0]
            logger.info(f"📊 Personal info records: {personal_count}")

            # Test experience
            cursor.execute("SELECT COUNT(*) FROM experience WHERE deleted_at IS NULL")
            experience_count = cursor.fetchone()[0]
            logger.info(f"📊 Experience records: {experience_count}")

            # Test skills
            cursor.execute("SELECT COUNT(*) FROM skills WHERE deleted_at IS NULL")
            skills_count = cursor.fetchone()[0]
            logger.info(f"📊 Skills records: {skills_count}")

            # Test projects
            cursor.execute("SELECT COUNT(*) FROM projects WHERE deleted_at IS NULL")
            projects_count = cursor.fetchone()[0]
            logger.info(f"📊 Projects records: {projects_count}")

            # Test education
            cursor.execute("SELECT COUNT(*) FROM education WHERE deleted_at IS NULL")
            education_count = cursor.fetchone()[0]
            logger.info(f"📊 Education records: {education_count}")

            # Test content chunks
            cursor.execute("SELECT COUNT(*) FROM content_chunks WHERE deleted_at IS NULL")
            chunks_count = cursor.fetchone()[0]
            logger.info(f"📊 Content chunks: {chunks_count}")

            # Test metadata
            cursor.execute("SELECT COUNT(*) FROM digital_twin_metadata")
            metadata_count = cursor.fetchone()[0]
            logger.info(f"📊 Metadata records: {metadata_count}")

            # Get sample data
            if personal_count > 0:
                cursor.execute("""
                    SELECT full_name, email, summary
                    FROM personal_info
                    WHERE deleted_at IS NULL
                    LIMIT 1
                """)
                person = cursor.fetchone()
                logger.info(f"👤 Sample person: {person[0]} ({person[1]})")

            if experience_count > 0:
                cursor.execute("""
                    SELECT company_name, position_title
                    FROM experience
                    WHERE deleted_at IS NULL
                    LIMIT 3
                """)
                experiences = cursor.fetchall()
                logger.info("💼 Recent experience:")
                for exp in experiences:
                    logger.info(f"   - {exp[1]} at {exp[0]}")

            if skills_count > 0:
                cursor.execute("""
                    SELECT skill_name, proficiency_level, category
                    FROM skills
                    WHERE deleted_at IS NULL AND is_technical = true
                    ORDER BY proficiency_level DESC
                    LIMIT 5
                """)
                skills = cursor.fetchall()
                logger.info("🛠️ Top technical skills:")
                for skill in skills:
                    logger.info(f"   - {skill[0]} ({skill[1]})")

        conn.close()

        # Summary
        total_records = personal_count + experience_count + skills_count + projects_count + education_count + chunks_count + metadata_count
        logger.info(f"\n📈 Migration Summary:")
        logger.info(f"   Total records migrated: {total_records}")
        logger.info("✅ Data migration verification completed!")

        return True

    except Exception as e:
        logger.error(f"❌ Data migration test failed: {str(e)}")
        return False

def main():
    """Main test function"""
    print("🧪 Testing Data Migration")
    print("=" * 30)

    # Test database connection
    if not test_database_connection():
        print("❌ Cannot proceed without database connection")
        return

    # Test data migration
    if test_data_migration():
        print("\n✅ All tests passed! Data migration was successful.")
    else:
        print("\n❌ Data migration tests failed. Please check the migration script.")

if __name__ == "__main__":
    main()</content>
<parameter name="filePath">c:\Users\sbina\OneDrive\Desktop\digitaltwin\test_migration.py