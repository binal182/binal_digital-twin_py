#!/usr/bin/env python3
"""
Database Setup Script for Professional Digital Twin
This script creates the comprehensive PostgreSQL schema and populates it with data from binal_mytwin.json
"""

import json
import os
import psycopg2
from psycopg2.extras import Json
from datetime import datetime
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DigitalTwinDatabase:
    def __init__(self, connection_string=None):
        """Initialize database connection"""
        if connection_string:
            self.connection_string = connection_string
        else:
            # Try to get from environment variables
            self.connection_string = os.getenv('DATABASE_URL')

        if not self.connection_string:
            raise ValueError("Database connection string not provided. Set DATABASE_URL environment variable or pass as parameter.")

        self.conn = None
        self.cursor = None

    def connect(self):
        """Establish database connection"""
        try:
            self.conn = psycopg2.connect(self.connection_string)
            self.cursor = self.conn.cursor()
            logger.info("Database connection established")
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            raise

    def disconnect(self):
        """Close database connection"""
        if self.cursor:
            self.cursor.close()
        if self.conn:
            self.conn.close()
        logger.info("Database connection closed")

    def execute_sql_file(self, file_path):
        """Execute SQL statements from a file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                sql_content = f.read()

            # Split SQL into individual statements
            statements = [stmt.strip() for stmt in sql_content.split(';') if stmt.strip()]

            for statement in statements:
                if statement:
                    logger.info(f"Executing: {statement[:100]}...")
                    self.cursor.execute(statement)

            self.conn.commit()
            logger.info(f"Successfully executed SQL from {file_path}")

        except Exception as e:
            self.conn.rollback()
            logger.error(f"Failed to execute SQL from {file_path}: {e}")
            raise

    def create_schema(self):
        """Create the database schema"""
        logger.info("Creating database schema...")
        self.execute_sql_file('comprehensive_schema.sql')

    def load_sample_data(self):
        """Load sample data into the database"""
        logger.info("Loading sample data...")
        self.execute_sql_file('sample_data_and_queries.sql')

    def load_from_json(self, json_file_path):
        """Load data from the JSON file into the database"""
        logger.info(f"Loading data from {json_file_path}...")

        try:
            with open(json_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Insert personal info
            personal_info = data.get('personalInfo', {})
            self._insert_personal_info(personal_info, data)

            # Insert experience
            for exp in data.get('experience', []):
                self._insert_experience(exp)

            # Insert education
            for edu in data.get('education', []):
                self._insert_education(edu)

            # Insert skills
            for skill in data.get('skills', []):
                self._insert_skill(skill)

            # Insert projects
            for project in data.get('projects', []):
                self._insert_project(project)

            # Insert other data...
            self._insert_certifications(data.get('certifications', []))
            self._insert_publications(data.get('publications', []))
            self._insert_awards(data.get('awards', []))
            self._insert_network(data.get('professional_network', []))
            self._insert_metadata(data.get('metadata', {}))

            self.conn.commit()
            logger.info("Successfully loaded data from JSON file")

        except Exception as e:
            self.conn.rollback()
            logger.error(f"Failed to load data from JSON: {e}")
            raise

    def _insert_personal_info(self, personal_info, full_data):
        """Insert personal information"""
        sql = """
        INSERT INTO personal_info (
            full_name, email, phone, location, linkedin_url, github_url,
            website_url, summary, raw_data, data_quality_score
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        RETURNING id
        """

        values = (
            personal_info.get('name', ''),
            personal_info.get('email', ''),
            personal_info.get('phone', ''),
            personal_info.get('location', ''),
            personal_info.get('linkedin', ''),
            personal_info.get('github', ''),
            personal_info.get('website', ''),
            full_data.get('summary', ''),
            Json(full_data),
            full_data.get('metadata', {}).get('data_quality_score', 95)
        )

        self.cursor.execute(sql, values)
        return self.cursor.fetchone()[0]

    def _insert_experience(self, exp_data):
        """Insert experience record"""
        sql = """
        INSERT INTO experience (
            person_id, company_name, position_title, start_date, end_date,
            is_current_position, location, description, achievements,
            technologies_used, raw_data
        ) VALUES (
            (SELECT id FROM personal_info LIMIT 1),
            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
        )
        """

        values = (
            exp_data.get('company', ''),
            exp_data.get('position', ''),
            exp_data.get('start_date', None),
            exp_data.get('end_date', None),
            exp_data.get('is_current', False),
            exp_data.get('location', ''),
            exp_data.get('description', ''),
            exp_data.get('achievements', []),
            exp_data.get('technologies', []),
            Json(exp_data)
        )

        self.cursor.execute(sql, values)

    def _insert_education(self, edu_data):
        """Insert education record"""
        sql = """
        INSERT INTO education (
            person_id, institution_name, degree, field_of_study,
            start_date, end_date, gpa, raw_data
        ) VALUES (
            (SELECT id FROM personal_info LIMIT 1),
            %s, %s, %s, %s, %s, %s, %s
        )
        """

        values = (
            edu_data.get('institution', ''),
            edu_data.get('degree', ''),
            edu_data.get('field_of_study', ''),
            edu_data.get('start_date', None),
            edu_data.get('end_date', None),
            edu_data.get('gpa', None),
            Json(edu_data)
        )

        self.cursor.execute(sql, values)

    def _insert_skill(self, skill_data):
        """Insert skill record"""
        sql = """
        INSERT INTO skills (
            person_id, skill_name, category, proficiency_level,
            raw_data
        ) VALUES (
            (SELECT id FROM personal_info LIMIT 1),
            %s, %s, %s, %s
        )
        """

        values = (
            skill_data.get('name', ''),
            skill_data.get('category', 'General'),
            skill_data.get('proficiency', 'Intermediate'),
            Json(skill_data)
        )

        self.cursor.execute(sql, values)

    def _insert_project(self, project_data):
        """Insert project record"""
        sql = """
        INSERT INTO projects (
            person_id, project_name, description, start_date, end_date,
            technologies_used, achievements, raw_data
        ) VALUES (
            (SELECT id FROM personal_info LIMIT 1),
            %s, %s, %s, %s, %s, %s, %s
        )
        """

        values = (
            project_data.get('name', ''),
            project_data.get('description', ''),
            project_data.get('start_date', None),
            project_data.get('end_date', None),
            project_data.get('technologies', []),
            project_data.get('achievements', []),
            Json(project_data)
        )

        self.cursor.execute(sql, values)

    def _insert_certifications(self, certifications):
        """Insert certifications"""
        for cert in certifications:
            sql = """
            INSERT INTO certifications (
                person_id, certification_name, issuing_organization,
                issue_date, raw_data
            ) VALUES (
                (SELECT id FROM personal_info LIMIT 1),
                %s, %s, %s, %s
            )
            """

            values = (
                cert.get('name', ''),
                cert.get('issuer', ''),
                cert.get('date', None),
                Json(cert)
            )

            self.cursor.execute(sql, values)

    def _insert_publications(self, publications):
        """Insert publications"""
        for pub in publications:
            sql = """
            INSERT INTO publications (
                person_id, title, authors, publication_date,
                journal_or_conference, raw_data
            ) VALUES (
                (SELECT id FROM personal_info LIMIT 1),
                %s, %s, %s, %s, %s
            )
            """

            values = (
                pub.get('title', ''),
                pub.get('authors', []),
                pub.get('date', None),
                pub.get('venue', ''),
                Json(pub)
            )

            self.cursor.execute(sql, values)

    def _insert_awards(self, awards):
        """Insert awards"""
        for award in awards:
            sql = """
            INSERT INTO awards (
                person_id, award_name, issuing_organization,
                award_date, raw_data
            ) VALUES (
                (SELECT id FROM personal_info LIMIT 1),
                %s, %s, %s, %s
            )
            """

            values = (
                award.get('name', ''),
                award.get('issuer', ''),
                award.get('date', None),
                Json(award)
            )

            self.cursor.execute(sql, values)

    def _insert_network(self, network):
        """Insert professional network"""
        for contact in network:
            sql = """
            INSERT INTO professional_network (
                person_id, contact_name, company, position,
                raw_data
            ) VALUES (
                (SELECT id FROM personal_info LIMIT 1),
                %s, %s, %s, %s
            )
            """

            values = (
                contact.get('name', ''),
                contact.get('company', ''),
                contact.get('position', ''),
                Json(contact)
            )

            self.cursor.execute(sql, values)

    def _insert_metadata(self, metadata):
        """Insert metadata"""
        sql = """
        INSERT INTO digital_twin_metadata (
            person_id, version, data_quality_score, completeness,
            data_sources, ai_readiness, raw_data
        ) VALUES (
            (SELECT id FROM personal_info LIMIT 1),
            %s, %s, %s, %s, %s, %s
        )
        """

        values = (
            metadata.get('version', '1.0'),
            metadata.get('data_quality_score', 95),
            Json(metadata.get('completeness', {})),
            metadata.get('data_sources', []),
            Json(metadata.get('ai_readiness', {})),
            Json(metadata)
        )

        self.cursor.execute(sql, values)

    def verify_setup(self):
        """Verify that the database setup is correct"""
        logger.info("Verifying database setup...")

        # Check table counts
        tables = [
            'personal_info', 'experience', 'education', 'skills',
            'projects', 'certifications', 'publications', 'awards',
            'professional_network', 'digital_twin_metadata'
        ]

        for table in tables:
            self.cursor.execute(f"SELECT COUNT(*) FROM {table} WHERE deleted_at IS NULL")
            count = self.cursor.fetchone()[0]
            logger.info(f"Table {table}: {count} records")

        # Test a sample query
        self.cursor.execute("""
        SELECT pi.full_name, COUNT(e.*) as experience_count, COUNT(s.*) as skills_count
        FROM personal_info pi
        LEFT JOIN experience e ON pi.id = e.person_id AND e.deleted_at IS NULL
        LEFT JOIN skills s ON pi.id = s.person_id AND s.deleted_at IS NULL
        WHERE pi.deleted_at IS NULL
        GROUP BY pi.id, pi.full_name
        """)

        results = self.cursor.fetchall()
        logger.info(f"Sample query results: {results}")

def main():
    """Main function to set up the database"""
    print("Professional Digital Twin Database Setup")
    print("=" * 50)

    # Initialize database connection
    db = DigitalTwinDatabase()

    try:
        # Connect to database
        db.connect()

        # Create schema
        print("\n1. Creating database schema...")
        db.create_schema()

        # Load data from JSON
        print("\n2. Loading data from binal_mytwin.json...")
        db.load_from_json('binal_mytwin.json')

        # Load sample data
        print("\n3. Loading additional sample data...")
        db.load_sample_data()

        # Verify setup
        print("\n4. Verifying database setup...")
        db.verify_setup()

        print("\n✅ Database setup completed successfully!")
        print("\nYou can now run queries against your professional digital twin database.")
        print("Check the sample_data_and_queries.sql file for example queries.")

    except Exception as e:
        logger.error(f"Database setup failed: {e}")
        print(f"\n❌ Database setup failed: {e}")
        return 1

    finally:
        db.disconnect()

    return 0

if __name__ == "__main__":
    exit(main())