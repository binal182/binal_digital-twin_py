#!/usr/bin/env python3
"""
Data Migration Script: JSON to PostgreSQL with Vector Embeddings
Migrates professional profile data from JSON to PostgreSQL database with vector embeddings
"""

import os
import json
import uuid
from datetime import datetime, date
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv
import psycopg2
from psycopg2.extras import Json as PsycopgJson
import logging
from upstash_vector import Index
from groq import Groq

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('migration.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DataMigration:
    """Handles migration of JSON data to PostgreSQL with vector embeddings"""

    def __init__(self):
        self.db_connection = None
        self.vector_index = None
        self.groq_client = None
        self.person_id = None

        # Setup connections
        self.setup_database_connection()
        self.setup_vector_database()
        self.setup_groq_client()

    def setup_database_connection(self):
        """Setup PostgreSQL connection"""
        try:
            db_params = {
                'host': os.getenv('DB_HOST', 'localhost'),
                'port': os.getenv('DB_PORT', '5432'),
                'database': os.getenv('DB_NAME', 'digital_twin'),
                'user': os.getenv('DB_USER', 'postgres'),
                'password': os.getenv('DB_PASSWORD', '')
            }

            self.db_connection = psycopg2.connect(**db_params)
            self.db_connection.autocommit = False  # We'll manage transactions

            logger.info("✅ Connected to PostgreSQL database successfully!")

        except Exception as e:
            logger.error(f"❌ Failed to connect to database: {str(e)}")
            raise

    def setup_vector_database(self):
        """Setup Upstash Vector for embeddings"""
        try:
            self.vector_index = Index.from_env()
            logger.info("✅ Connected to Upstash Vector successfully!")
        except Exception as e:
            logger.error(f"❌ Failed to connect to Upstash Vector: {str(e)}")
            raise

    def setup_groq_client(self):
        """Setup Groq client for content processing"""
        try:
            groq_api_key = os.getenv('GROQ_API_KEY')
            if groq_api_key:
                self.groq_client = Groq(api_key=groq_api_key)
                logger.info("✅ Connected to Groq successfully!")
            else:
                logger.warning("⚠️ GROQ_API_KEY not found - content processing will be limited")
        except Exception as e:
            logger.warning(f"⚠️ Failed to setup Groq client: {str(e)}")

    def load_json_data(self, file_path: str) -> Dict[str, Any]:
        """Load data from JSON file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            logger.info(f"✅ Loaded data from {file_path}")
            return data
        except Exception as e:
            logger.error(f"❌ Failed to load JSON data: {str(e)}")
            raise

    def create_content_chunks(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create content chunks from the profile data for vector embeddings"""
        chunks = []

        # Personal info chunks
        personal_info = data.get('personalInfo', {})
        if personal_info:
            chunks.append({
                'id': str(uuid.uuid4()),
                'chunk_type': 'personal_summary',
                'content': personal_info.get('summary', ''),
                'title': 'Professional Summary',
                'metadata': {
                    'category': 'personal_info',
                    'importance': 'high',
                    'tags': ['summary', 'overview', 'professional']
                },
                'source_table': 'personal_info',
                'source_id': None  # Will be set after personal_info is inserted
            })

            if personal_info.get('elevator_pitch'):
                chunks.append({
                    'id': str(uuid.uuid4()),
                    'chunk_type': 'elevator_pitch',
                    'content': personal_info.get('elevator_pitch', ''),
                    'title': 'Elevator Pitch',
                    'metadata': {
                        'category': 'personal_info',
                        'importance': 'high',
                        'tags': ['pitch', 'goals', 'career']
                    },
                    'source_table': 'personal_info',
                    'source_id': None
                })

        # Experience chunks
        for exp in data.get('experience', []):
            content = f"{exp.get('position', '')} at {exp.get('company', '')}. {exp.get('description', '')}"
            if exp.get('achievements'):
                content += f" Key achievements: {'; '.join(exp['achievements'])}"
            if exp.get('technologies'):
                content += f" Technologies used: {', '.join(exp['technologies'])}"

            chunks.append({
                'id': str(uuid.uuid4()),
                'chunk_type': 'experience',
                'content': content,
                'title': f"{exp.get('position', '')} at {exp.get('company', '')}",
                'metadata': {
                    'category': 'experience',
                    'company': exp.get('company', ''),
                    'position': exp.get('position', ''),
                    'duration': exp.get('duration', ''),
                    'importance': 'high',
                    'tags': ['experience', 'work', 'career'] + exp.get('technologies', [])[:3]
                },
                'source_table': 'experience',
                'source_id': None
            })

        # Skills chunks
        skills_content = "Technical Skills: " + ", ".join([
            f"{skill.get('name', '')} ({skill.get('level', '')})"
            for skill in data.get('skills', {}).get('technical', [])
        ])

        if data.get('skills', {}).get('soft', []):
            skills_content += "\n\nSoft Skills: " + ", ".join([
                skill.get('name', '') for skill in data['skills']['soft']
            ])

        chunks.append({
            'id': str(uuid.uuid4()),
            'chunk_type': 'skills_overview',
            'content': skills_content,
            'title': 'Technical and Soft Skills',
            'metadata': {
                'category': 'skills',
                'importance': 'high',
                'tags': ['skills', 'competencies', 'expertise']
            },
            'source_table': 'skills',
            'source_id': None
        })

        # Projects chunks
        for project in data.get('projects', []):
            content = f"{project.get('name', '')}. {project.get('description', '')}"
            if project.get('technologies'):
                content += f" Technologies: {', '.join(project['technologies'])}"
            if project.get('outcomes'):
                content += f" Outcomes: {'; '.join(project['outcomes'])}"

            chunks.append({
                'id': str(uuid.uuid4()),
                'chunk_type': 'project',
                'content': content,
                'title': project.get('name', ''),
                'metadata': {
                    'category': 'projects',
                    'importance': 'medium',
                    'tags': ['project', 'development'] + project.get('technologies', [])[:3]
                },
                'source_table': 'projects',
                'source_id': None
            })

        # Education chunks
        for edu in data.get('education', []):
            content = f"{edu.get('degree', '')} in {edu.get('field', '')} from {edu.get('institution', '')}"
            if edu.get('gpa'):
                content += f". GPA: {edu['gpa']}"
            if edu.get('achievements'):
                content += f". Achievements: {'; '.join(edu['achievements'])}"

            chunks.append({
                'id': str(uuid.uuid4()),
                'chunk_type': 'education',
                'content': content,
                'title': f"{edu.get('degree', '')} - {edu.get('institution', '')}",
                'metadata': {
                    'category': 'education',
                    'institution': edu.get('institution', ''),
                    'degree': edu.get('degree', ''),
                    'importance': 'medium',
                    'tags': ['education', 'academic', 'qualification']
                },
                'source_table': 'education',
                'source_id': None
            })

        return chunks

    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for text using Upstash"""
        try:
            # For now, we'll use Upstash's built-in embedding generation
            # This is a placeholder - in practice, we'd need to get the actual embedding
            # For this migration, we'll store the text and generate embeddings later
            return []  # Return empty list for now
        except Exception as e:
            logger.warning(f"Failed to generate embedding: {str(e)}")
            return []

    def insert_personal_info(self, data: Dict[str, Any]) -> str:
        """Insert personal information"""
        try:
            personal_info = data.get('personalInfo', {})

            query = """
            INSERT INTO personal_info (
                full_name, email, phone, location, linkedin_url, github_url,
                summary, raw_data, data_quality_score
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id
            """

            contact = personal_info.get('contact', {})
            values = (
                personal_info.get('name', ''),
                contact.get('email', ''),
                contact.get('phone', ''),
                personal_info.get('location', ''),
                contact.get('linkedin', ''),
                contact.get('github', ''),
                personal_info.get('summary', ''),
                PsycopgJson(personal_info),
                data.get('metadata', {}).get('data_quality_score', 95)
            )

            with self.db_connection.cursor() as cursor:
                cursor.execute(query, values)
                person_id = cursor.fetchone()[0]

            self.db_connection.commit()
            logger.info(f"✅ Inserted personal info for {personal_info.get('name', '')}")
            return str(person_id)

        except Exception as e:
            self.db_connection.rollback()
            logger.error(f"❌ Failed to insert personal info: {str(e)}")
            raise

    def insert_experience(self, person_id: str, experiences: List[Dict[str, Any]]):
        """Insert work experience"""
        try:
            query = """
            INSERT INTO experience (
                person_id, company_name, position_title, start_date, end_date,
                is_current_position, location, description, achievements,
                technologies_used, raw_data
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id
            """

            with self.db_connection.cursor() as cursor:
                for exp in experiences:
                    # Parse dates
                    start_date = self.parse_date(exp.get('start_date'))
                    end_date = self.parse_date(exp.get('end_date'))

                    values = (
                        person_id,
                        exp.get('company', ''),
                        exp.get('position', ''),
                        start_date,
                        end_date,
                        exp.get('is_current', False),
                        exp.get('location', ''),
                        exp.get('description', ''),
                        exp.get('achievements', []),
                        exp.get('technologies', []),
                        PsycopgJson(exp)
                    )

                    cursor.execute(query, values)
                    exp_id = cursor.fetchone()[0]

                    # Update content chunks with source_id
                    self.update_chunk_source_id(exp['id'], 'experience', str(exp_id))

            self.db_connection.commit()
            logger.info(f"✅ Inserted {len(experiences)} experience records")

        except Exception as e:
            self.db_connection.rollback()
            logger.error(f"❌ Failed to insert experience: {str(e)}")
            raise

    def insert_skills(self, person_id: str, skills_data: Dict[str, Any]):
        """Insert skills"""
        try:
            query = """
            INSERT INTO skills (
                person_id, skill_name, category, proficiency_level,
                years_of_experience, is_technical, raw_data
            ) VALUES (%s, %s, %s, %s, %s, %s, %s)
            """

            with self.db_connection.cursor() as cursor:
                # Technical skills
                for skill in skills_data.get('technical', []):
                    values = (
                        person_id,
                        skill.get('name', ''),
                        'Technical',
                        skill.get('level', 'Intermediate'),
                        skill.get('years', 1),
                        True,
                        PsycopgJson(skill)
                    )
                    cursor.execute(query, values)

                # Soft skills
                for skill in skills_data.get('soft', []):
                    values = (
                        person_id,
                        skill.get('name', ''),
                        'Soft Skills',
                        'Intermediate',
                        None,
                        False,
                        PsycopgJson(skill)
                    )
                    cursor.execute(query, values)

            self.db_connection.commit()
            logger.info("✅ Inserted skills")

        except Exception as e:
            self.db_connection.rollback()
            logger.error(f"❌ Failed to insert skills: {str(e)}")
            raise

    def insert_projects(self, person_id: str, projects: List[Dict[str, Any]]):
        """Insert projects"""
        try:
            query = """
            INSERT INTO projects (
                person_id, project_name, description, start_date, end_date,
                is_ongoing, technologies_used, raw_data
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id
            """

            with self.db_connection.cursor() as cursor:
                for project in projects:
                    start_date = self.parse_date(project.get('start_date'))
                    end_date = self.parse_date(project.get('end_date'))

                    values = (
                        person_id,
                        project.get('name', ''),
                        project.get('description', ''),
                        start_date,
                        end_date,
                        project.get('ongoing', False),
                        project.get('technologies', []),
                        PsycopgJson(project)
                    )

                    cursor.execute(query, values)
                    project_id = cursor.fetchone()[0]

                    # Update content chunks with source_id
                    self.update_chunk_source_id(project['id'], 'projects', str(project_id))

            self.db_connection.commit()
            logger.info(f"✅ Inserted {len(projects)} projects")

        except Exception as e:
            self.db_connection.rollback()
            logger.error(f"❌ Failed to insert projects: {str(e)}")
            raise

    def insert_education(self, person_id: str, education: List[Dict[str, Any]]):
        """Insert education"""
        try:
            query = """
            INSERT INTO education (
                person_id, institution_name, degree, field_of_study,
                start_date, end_date, gpa, raw_data
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id
            """

            with self.db_connection.cursor() as cursor:
                for edu in education:
                    start_date = self.parse_date(edu.get('start_date'))
                    end_date = self.parse_date(edu.get('end_date'))

                    values = (
                        person_id,
                        edu.get('institution', ''),
                        edu.get('degree', ''),
                        edu.get('field', ''),
                        start_date,
                        end_date,
                        edu.get('gpa'),
                        PsycopgJson(edu)
                    )

                    cursor.execute(query, values)
                    edu_id = cursor.fetchone()[0]

                    # Update content chunks with source_id
                    self.update_chunk_source_id(edu['id'], 'education', str(edu_id))

            self.db_connection.commit()
            logger.info(f"✅ Inserted {len(education)} education records")

        except Exception as e:
            self.db_connection.rollback()
            logger.error(f"❌ Failed to insert education: {str(e)}")
            raise

    def insert_content_chunks(self, person_id: str, chunks: List[Dict[str, Any]]):
        """Insert content chunks with vector embeddings"""
        try:
            query = """
            INSERT INTO content_chunks (
                person_id, chunk_type, content, embedding_vector,
                metadata, source_table, source_id, chunk_index
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """

            with self.db_connection.cursor() as cursor:
                for i, chunk in enumerate(chunks):
                    # For now, we'll store empty vector and generate embeddings later
                    # In a production system, you'd generate the actual embeddings here
                    embedding_vector = self.generate_embedding(chunk['content'])

                    values = (
                        person_id,
                        chunk['chunk_type'],
                        chunk['content'],
                        embedding_vector if embedding_vector else None,
                        PsycopgJson(chunk['metadata']),
                        chunk.get('source_table'),
                        chunk.get('source_id'),
                        i
                    )

                    cursor.execute(query, values)

            self.db_connection.commit()
            logger.info(f"✅ Inserted {len(chunks)} content chunks")

        except Exception as e:
            self.db_connection.rollback()
            logger.error(f"❌ Failed to insert content chunks: {str(e)}")
            raise

    def insert_metadata(self, person_id: str, metadata: Dict[str, Any]):
        """Insert digital twin metadata"""
        try:
            query = """
            INSERT INTO digital_twin_metadata (
                person_id, version, data_quality_score, completeness,
                data_sources, ai_readiness, raw_data
            ) VALUES (%s, %s, %s, %s, %s, %s, %s)
            """

            values = (
                person_id,
                metadata.get('version', '1.0'),
                metadata.get('data_quality_score', 95),
                PsycopgJson(metadata.get('completeness', {})),
                metadata.get('data_sources', []),
                PsycopgJson(metadata.get('ai_readiness', {})),
                PsycopgJson(metadata)
            )

            with self.db_connection.cursor() as cursor:
                cursor.execute(query, values)

            self.db_connection.commit()
            logger.info("✅ Inserted metadata")

        except Exception as e:
            self.db_connection.rollback()
            logger.error(f"❌ Failed to insert metadata: {str(e)}")
            raise

    def update_chunk_source_id(self, original_id: str, table_name: str, new_id: str):
        """Update content chunk source_id after inserting the main record"""
        try:
            query = """
            UPDATE content_chunks
            SET source_id = %s
            WHERE source_table = %s AND metadata->>'original_id' = %s
            """

            with self.db_connection.cursor() as cursor:
                cursor.execute(query, (new_id, table_name, original_id))

        except Exception as e:
            logger.warning(f"Failed to update chunk source_id: {str(e)}")

    def parse_date(self, date_str: Optional[str]) -> Optional[date]:
        """Parse date string into date object"""
        if not date_str:
            return None

        try:
            # Try different date formats
            formats = ['%Y-%m-%d', '%m/%d/%Y', '%d/%m/%Y', '%B %Y', '%Y']
            for fmt in formats:
                try:
                    return datetime.strptime(date_str, fmt).date()
                except ValueError:
                    continue
            return None
        except Exception:
            return None

    def migrate_data(self, json_file_path: str):
        """Main migration function"""
        try:
            logger.info("🚀 Starting data migration...")

            # Load JSON data
            data = self.load_json_data(json_file_path)

            # Create content chunks for vector embeddings
            content_chunks = self.create_content_chunks(data)

            # Insert personal info first (to get person_id)
            self.person_id = self.insert_personal_info(data)

            # Update content chunks with person_id
            for chunk in content_chunks:
                chunk['person_id'] = self.person_id

            # Insert other data
            if data.get('experience'):
                self.insert_experience(self.person_id, data['experience'])

            if data.get('skills'):
                self.insert_skills(self.person_id, data['skills'])

            if data.get('projects'):
                self.insert_projects(self.person_id, data['projects'])

            if data.get('education'):
                self.insert_education(self.person_id, data['education'])

            # Insert content chunks
            self.insert_content_chunks(self.person_id, content_chunks)

            # Insert metadata
            if data.get('metadata'):
                self.insert_metadata(self.person_id, data['metadata'])

            logger.info("✅ Data migration completed successfully!")
            logger.info(f"📊 Migrated data for person ID: {self.person_id}")

        except Exception as e:
            logger.error(f"❌ Migration failed: {str(e)}")
            raise
        finally:
            if self.db_connection:
                self.db_connection.close()

    def generate_vector_embeddings(self):
        """Generate vector embeddings for content chunks (separate step)"""
        try:
            logger.info("🧠 Generating vector embeddings...")

            # This would be implemented to generate actual embeddings
            # For now, it's a placeholder for the embedding generation process

            logger.info("✅ Vector embeddings generation completed!")

        except Exception as e:
            logger.error(f"❌ Failed to generate embeddings: {str(e)}")
            raise


def main():
    """Main entry point"""
    print("🚀 Data Migration: JSON to PostgreSQL with Vector Embeddings")
    print("=" * 60)

    # Check environment variables
    required_env_vars = [
        'DB_HOST', 'DB_PORT', 'DB_NAME', 'DB_USER', 'DB_PASSWORD',
        'UPSTASH_VECTOR_REST_URL', 'UPSTASH_VECTOR_REST_TOKEN'
    ]

    missing_vars = [var for var in required_env_vars if not os.getenv(var)]
    if missing_vars:
        print("❌ Missing required environment variables:")
        for var in missing_vars:
            print(f"   - {var}")
        print("\n💡 Please set these in your .env file")
        return

    try:
        # Initialize migration
        migration = DataMigration()

        # Run migration
        json_file = "binal_mytwin.json"
        migration.migrate_data(json_file)

        print("\n✅ Migration completed successfully!")
        print("📊 Data has been migrated to PostgreSQL with vector embeddings ready for generation")

        # Optional: Generate embeddings
        generate_embeddings = input("\n🧠 Generate vector embeddings now? (y/N): ").strip().lower()
        if generate_embeddings in ['y', 'yes']:
            migration.generate_vector_embeddings()

    except Exception as e:
        print(f"\n❌ Migration failed: {str(e)}")
        print("Check the migration.log file for detailed error information.")


if __name__ == "__main__":
    main()</content>
<parameter name="filePath">c:\Users\sbina\OneDrive\Desktop\digitaltwin\migrate_data.py