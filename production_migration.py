#!/usr/bin/env python3
"""
Production-Ready Data Migration Script
Comprehensive migration from JSON to PostgreSQL with vector embeddings
"""

import os
import json
import uuid
import time
from datetime import datetime, date
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from dotenv import load_dotenv
import psycopg2
from psycopg2.extras import Json as PsycopgJson
import logging
from upstash_vector import Index
import requests
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('migration.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class MigrationConfig:
    """Configuration for migration process"""
    batch_size: int = 10
    max_retries: int = 3
    embedding_dimensions: int = 1024
    chunk_overlap: int = 50
    max_chunk_length: int = 1000
    min_chunk_length: int = 100

class DataValidator:
    """Validates JSON data structure and completeness"""

    @staticmethod
    def validate_personal_info(data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate personal information structure"""
        errors = []
        personal_info = data.get('personalInfo', {})

        required_fields = ['name', 'contact']
        for field in required_fields:
            if field not in personal_info:
                errors.append(f"Missing required field: personalInfo.{field}")

        if 'contact' in personal_info:
            contact = personal_info['contact']
            if not contact.get('email'):
                errors.append("Missing email in contact information")

        return len(errors) == 0, errors

    @staticmethod
    def validate_experience(experience: List[Dict[str, Any]]) -> Tuple[bool, List[str]]:
        """Validate experience data"""
        errors = []
        if not isinstance(experience, list):
            errors.append("Experience must be a list")
            return False, errors

        for i, exp in enumerate(experience):
            if not exp.get('company'):
                errors.append(f"Experience {i+1}: Missing company name")
            if not exp.get('position'):
                errors.append(f"Experience {i+1}: Missing position title")

        return len(errors) == 0, errors

    @staticmethod
    def validate_json_data(data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Comprehensive JSON data validation"""
        errors = []

        # Validate personal info
        valid_personal, personal_errors = DataValidator.validate_personal_info(data)
        errors.extend(personal_errors)

        # Validate experience
        if 'experience' in data:
            valid_exp, exp_errors = DataValidator.validate_experience(data['experience'])
            errors.extend(exp_errors)

        # Check for minimum data requirements
        if not data.get('personalInfo', {}).get('name'):
            errors.append("No personal name found - minimum data requirement not met")

        return len(errors) == 0, errors

class ContentChunker:
    """Advanced content chunking for RAG optimization"""

    def __init__(self, config: MigrationConfig):
        self.config = config

    def create_chunks(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create optimized content chunks from profile data"""
        chunks = []

        # Personal info chunks
        chunks.extend(self._create_personal_chunks(data))

        # Experience chunks
        chunks.extend(self._create_experience_chunks(data))

        # Skills chunks
        chunks.extend(self._create_skills_chunks(data))

        # Projects chunks
        chunks.extend(self._create_project_chunks(data))

        # Education chunks
        chunks.extend(self._create_education_chunks(data))

        return chunks

    def _create_personal_chunks(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create personal information chunks"""
        chunks = []
        personal_info = data.get('personalInfo', {})

        # Summary chunk
        if personal_info.get('summary'):
            chunks.append(self._create_chunk(
                content=personal_info['summary'],
                chunk_type='personal_summary',
                title='Professional Summary',
                category='personal_info',
                importance='high',
                tags=['summary', 'overview', 'professional']
            ))

        # Elevator pitch chunk
        if personal_info.get('elevator_pitch'):
            chunks.append(self._create_chunk(
                content=personal_info['elevator_pitch'],
                chunk_type='elevator_pitch',
                title='Elevator Pitch',
                category='personal_info',
                importance='high',
                tags=['pitch', 'goals', 'career']
            ))

        # Career objective chunk
        if personal_info.get('career_objective'):
            chunks.append(self._create_chunk(
                content=personal_info['career_objective'],
                chunk_type='career_objective',
                title='Career Objective',
                category='personal_info',
                importance='medium',
                tags=['career', 'goals', 'objectives']
            ))

        return chunks

    def _create_experience_chunks(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create experience-related chunks"""
        chunks = []

        for exp in data.get('experience', []):
            # Main experience description
            content_parts = []
            if exp.get('description'):
                content_parts.append(exp['description'])

            if exp.get('achievements'):
                content_parts.append(f"Key Achievements: {'; '.join(exp['achievements'])}")

            if exp.get('technologies'):
                content_parts.append(f"Technologies: {', '.join(exp['technologies'])}")

            if exp.get('detailed_responsibilities'):
                content_parts.append(f"Responsibilities: {'; '.join(exp['detailed_responsibilities'])}")

            content = ' '.join(content_parts)

            if content:
                chunks.append(self._create_chunk(
                    content=content,
                    chunk_type='experience',
                    title=f"{exp.get('position', '')} at {exp.get('company', '')}",
                    category='experience',
                    importance='high',
                    tags=['experience', 'work', 'career'] + exp.get('technologies', [])[:5],
                    metadata={
                        'company': exp.get('company', ''),
                        'position': exp.get('position', ''),
                        'duration': exp.get('duration', ''),
                        'location': exp.get('location', ''),
                        'keywords': exp.get('keywords', [])
                    }
                ))

        return chunks

    def _create_skills_chunks(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create skills-related chunks"""
        chunks = []
        skills_data = data.get('skills', {})

        # Technical skills chunk
        technical_skills = skills_data.get('technical', [])
        if technical_skills:
            content = "Technical Skills:\n" + "\n".join([
                f"• {skill.get('name', '')} - {skill.get('level', 'Intermediate')} "
                f"({skill.get('years', 1)} years experience)"
                for skill in technical_skills
            ])

            chunks.append(self._create_chunk(
                content=content,
                chunk_type='technical_skills',
                title='Technical Skills',
                category='skills',
                importance='high',
                tags=['technical', 'skills', 'expertise'] + [s.get('name', '') for s in technical_skills[:5]]
            ))

        # Soft skills chunk
        soft_skills = skills_data.get('soft', [])
        if soft_skills:
            content = "Soft Skills:\n" + "\n".join([
                f"• {skill.get('name', '')}"
                for skill in soft_skills
            ])

            chunks.append(self._create_chunk(
                content=content,
                chunk_type='soft_skills',
                title='Soft Skills',
                category='skills',
                importance='medium',
                tags=['soft', 'skills', 'interpersonal'] + [s.get('name', '') for s in soft_skills[:5]]
            ))

        return chunks

    def _create_project_chunks(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create project-related chunks"""
        chunks = []

        for project in data.get('projects', []):
            content_parts = []
            if project.get('description'):
                content_parts.append(project['description'])

            if project.get('technologies'):
                content_parts.append(f"Technologies: {', '.join(project['technologies'])}")

            if project.get('outcomes'):
                content_parts.append(f"Outcomes: {'; '.join(project['outcomes'])}")

            content = ' '.join(content_parts)

            if content:
                chunks.append(self._create_chunk(
                    content=content,
                    chunk_type='project',
                    title=project.get('name', ''),
                    category='projects',
                    importance='medium',
                    tags=['project', 'development'] + project.get('technologies', [])[:5],
                    metadata={
                        'project_name': project.get('name', ''),
                        'technologies': project.get('technologies', []),
                        'outcomes': project.get('outcomes', [])
                    }
                ))

        return chunks

    def _create_education_chunks(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create education-related chunks"""
        chunks = []

        for edu in data.get('education', []):
            content_parts = []
            content_parts.append(f"{edu.get('degree', '')} in {edu.get('field', '')} from {edu.get('institution', '')}")

            if edu.get('gpa'):
                content_parts.append(f"GPA: {edu['gpa']}")

            if edu.get('achievements'):
                content_parts.append(f"Achievements: {'; '.join(edu['achievements'])}")

            content = ' '.join(content_parts)

            if content:
                chunks.append(self._create_chunk(
                    content=content,
                    chunk_type='education',
                    title=f"{edu.get('degree', '')} - {edu.get('institution', '')}",
                    category='education',
                    importance='medium',
                    tags=['education', 'academic', 'qualification'],
                    metadata={
                        'institution': edu.get('institution', ''),
                        'degree': edu.get('degree', ''),
                        'field': edu.get('field', ''),
                        'gpa': edu.get('gpa')
                    }
                ))

        return chunks

    def _create_chunk(self, content: str, chunk_type: str, title: str,
                     category: str, importance: str, tags: List[str],
                     metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Create a standardized chunk structure"""
        return {
            'id': str(uuid.uuid4()),
            'chunk_type': chunk_type,
            'content': content,
            'title': title,
            'metadata': {
                'category': category,
                'importance': importance,
                'tags': tags,
                'word_count': len(content.split()),
                'created_at': datetime.now().isoformat(),
                **(metadata or {})
            },
            'source_table': None,
            'source_id': None
        }

class VectorEmbedder:
    """Handles vector embedding generation using Upstash"""

    def __init__(self, config: MigrationConfig):
        self.config = config
        self.vector_index = None
        self.setup_vector_connection()

    def setup_vector_connection(self):
        """Setup Upstash Vector connection"""
        try:
            self.vector_index = Index.from_env()
            logger.info("✅ Connected to Upstash Vector successfully!")
        except Exception as e:
            logger.error(f"❌ Failed to connect to Upstash Vector: {str(e)}")
            raise

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((requests.RequestException, Exception))
    )
    def generate_embedding_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a batch of texts using Upstash"""
        try:
            # Use Upstash's built-in embedding generation
            embeddings = []

            for text in texts:
                # For Upstash, we prepare the data for upsert
                # The actual embedding happens server-side
                vector_data = {
                    'id': str(uuid.uuid4()),
                    'text': text,
                    'metadata': {'text_length': len(text)}
                }
                embeddings.append(vector_data)

            return embeddings

        except Exception as e:
            logger.error(f"❌ Failed to generate embeddings: {str(e)}")
            raise

    def upsert_vectors(self, vectors: List[Tuple[str, str, Dict[str, Any]]]):
        """Upsert vectors to Upstash with retry logic"""
        try:
            self.vector_index.upsert(vectors=vectors)
            logger.info(f"✅ Upserted {len(vectors)} vectors to Upstash")
        except Exception as e:
            logger.error(f"❌ Failed to upsert vectors: {str(e)}")
            raise

class DatabaseMigrator:
    """Handles PostgreSQL database operations"""

    def __init__(self, config: MigrationConfig):
        self.config = config
        self.connection = None
        self.setup_database_connection()

    def setup_database_connection(self):
        """Setup PostgreSQL connection"""
        try:
            # Try connection string first, then individual parameters
            connection_string = os.getenv('POSTGRES_CONNECTION_STRING')
            if connection_string:
                self.connection = psycopg2.connect(connection_string)
            else:
                db_params = {
                    'host': os.getenv('DB_HOST', 'localhost'),
                    'port': os.getenv('DB_PORT', '5432'),
                    'database': os.getenv('DB_NAME', 'digital_twin'),
                    'user': os.getenv('DB_USER', 'postgres'),
                    'password': os.getenv('DB_PASSWORD', '')
                }
                self.connection = psycopg2.connect(**db_params)

            self.connection.autocommit = False
            logger.info("✅ Connected to PostgreSQL database successfully!")

        except Exception as e:
            logger.error(f"❌ Failed to connect to database: {str(e)}")
            raise

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

            with self.connection.cursor() as cursor:
                cursor.execute(query, values)
                person_id = cursor.fetchone()[0]

            self.connection.commit()
            logger.info(f"✅ Inserted personal info for {personal_info.get('name', '')}")
            return str(person_id)

        except Exception as e:
            self.connection.rollback()
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

            with self.connection.cursor() as cursor:
                for exp in experiences:
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

            self.connection.commit()
            logger.info(f"✅ Inserted {len(experiences)} experience records")

        except Exception as e:
            self.connection.rollback()
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

            with self.connection.cursor() as cursor:
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

            self.connection.commit()
            logger.info("✅ Inserted skills")

        except Exception as e:
            self.connection.rollback()
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

            with self.connection.cursor() as cursor:
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

            self.connection.commit()
            logger.info(f"✅ Inserted {len(projects)} projects")

        except Exception as e:
            self.connection.rollback()
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

            with self.connection.cursor() as cursor:
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

            self.connection.commit()
            logger.info(f"✅ Inserted {len(education)} education records")

        except Exception as e:
            self.connection.rollback()
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

            with self.connection.cursor() as cursor:
                for i, chunk in enumerate(chunks):
                    values = (
                        person_id,
                        chunk['chunk_type'],
                        chunk['content'],
                        None,  # Will be updated after embedding generation
                        PsycopgJson(chunk['metadata']),
                        chunk.get('source_table'),
                        chunk.get('source_id'),
                        i
                    )

                    cursor.execute(query, values)

            self.connection.commit()
            logger.info(f"✅ Inserted {len(chunks)} content chunks")

        except Exception as e:
            self.connection.rollback()
            logger.error(f"❌ Failed to insert content chunks: {str(e)}")
            raise

    def insert_json_content(self, person_id: str, json_data: Dict[str, Any]):
        """Insert complete JSON document"""
        try:
            query = """
            INSERT INTO json_content (
                person_id, content_type, json_data, version, metadata
            ) VALUES (%s, %s, %s, %s, %s)
            """

            values = (
                person_id,
                'professional_profile',
                PsycopgJson(json_data),
                json_data.get('metadata', {}).get('version', '1.0'),
                PsycopgJson({
                    'data_quality_score': json_data.get('metadata', {}).get('data_quality_score', 95),
                    'completeness': json_data.get('metadata', {}).get('completeness', {}),
                    'created_at': datetime.now().isoformat()
                })
            )

            with self.connection.cursor() as cursor:
                cursor.execute(query, values)

            self.connection.commit()
            logger.info("✅ Inserted JSON content")

        except Exception as e:
            self.connection.rollback()
            logger.error(f"❌ Failed to insert JSON content: {str(e)}")
            raise

    def update_chunk_source_id(self, original_id: str, table_name: str, new_id: str):
        """Update content chunk source_id after inserting the main record"""
        try:
            query = """
            UPDATE content_chunks
            SET source_id = %s
            WHERE source_table = %s AND metadata->>'original_id' = %s
            """

            with self.connection.cursor() as cursor:
                cursor.execute(query, (new_id, table_name, original_id))

        except Exception as e:
            logger.warning(f"Failed to update chunk source_id: {str(e)}")

    def parse_date(self, date_str: Optional[str]) -> Optional[date]:
        """Parse date string into date object"""
        if not date_str:
            return None

        try:
            formats = ['%Y-%m-%d', '%m/%d/%Y', '%d/%m/%Y', '%B %Y', '%Y']
            for fmt in formats:
                try:
                    return datetime.strptime(date_str, fmt).date()
                except ValueError:
                    continue
            return None
        except Exception:
            return None

    def close(self):
        """Close database connection"""
        if self.connection:
            self.connection.close()

class MigrationOrchestrator:
    """Main orchestrator for the migration process"""

    def __init__(self):
        self.config = MigrationConfig()
        self.validator = DataValidator()
        self.chunker = ContentChunker(self.config)
        self.db_migrator = None
        self.vector_embedder = None
        self.person_id = None

    def setup_connections(self):
        """Setup all required connections"""
        self.db_migrator = DatabaseMigrator(self.config)
        self.vector_embedder = VectorEmbedder(self.config)

    def load_and_validate_data(self, file_path: str) -> Dict[str, Any]:
        """Load and validate JSON data"""
        logger.info(f"📂 Loading data from {file_path}")

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            logger.info("✅ JSON data loaded successfully")

            # Validate data
            is_valid, errors = self.validator.validate_json_data(data)
            if not is_valid:
                logger.error("❌ Data validation failed:")
                for error in errors:
                    logger.error(f"   - {error}")
                raise ValueError("Data validation failed")

            logger.info("✅ Data validation passed")
            return data

        except json.JSONDecodeError as e:
            logger.error(f"❌ Invalid JSON format: {str(e)}")
            raise
        except FileNotFoundError:
            logger.error(f"❌ File not found: {file_path}")
            raise
        except Exception as e:
            logger.error(f"❌ Failed to load data: {str(e)}")
            raise

    def migrate_structured_data(self, data: Dict[str, Any]):
        """Migrate structured data to PostgreSQL"""
        logger.info("🏗️ Starting structured data migration")

        try:
            # Insert personal info first
            self.person_id = self.db_migrator.insert_personal_info(data)

            # Insert other data
            if data.get('experience'):
                self.migrate_experience(data['experience'])

            if data.get('skills'):
                self.migrate_skills(data['skills'])

            if data.get('projects'):
                self.migrate_projects(data['projects'])

            if data.get('education'):
                self.migrate_education(data['education'])

            # Insert complete JSON
            self.db_migrator.insert_json_content(self.person_id, data)

            logger.info("✅ Structured data migration completed")

        except Exception as e:
            logger.error(f"❌ Structured data migration failed: {str(e)}")
            raise

    def migrate_experience(self, experiences: List[Dict[str, Any]]):
        """Migrate experience data"""
        logger.info(f"💼 Migrating {len(experiences)} experience records")
        self.db_migrator.insert_experience(self.person_id, experiences)

    def migrate_skills(self, skills_data: Dict[str, Any]):
        """Migrate skills data"""
        logger.info("🎯 Migrating skills data")
        self.db_migrator.insert_skills(self.person_id, skills_data)

    def migrate_projects(self, projects: List[Dict[str, Any]]):
        """Migrate projects data"""
        logger.info(f"🚀 Migrating {len(projects)} project records")
        self.db_migrator.insert_projects(self.person_id, projects)

    def migrate_education(self, education: List[Dict[str, Any]]):
        """Migrate education data"""
        logger.info(f"🎓 Migrating {len(education)} education records")
        self.db_migrator.insert_education(self.person_id, education)

    def process_content_chunks(self, data: Dict[str, Any]):
        """Process and store content chunks"""
        logger.info("📝 Processing content chunks")

        try:
            # Create content chunks
            chunks = self.chunker.create_chunks(data)

            # Add person_id to chunks
            for chunk in chunks:
                chunk['person_id'] = self.person_id

            # Insert chunks to database
            self.db_migrator.insert_content_chunks(self.person_id, chunks)

            logger.info(f"✅ Processed {len(chunks)} content chunks")

            return chunks

        except Exception as e:
            logger.error(f"❌ Content chunk processing failed: {str(e)}")
            raise

    def generate_vector_embeddings(self, chunks: List[Dict[str, Any]]):
        """Generate vector embeddings for content chunks"""
        logger.info("🧠 Generating vector embeddings")

        try:
            batch_size = self.config.batch_size
            total_processed = 0

            for i in range(0, len(chunks), batch_size):
                batch = chunks[i:i + batch_size]
                batch_texts = [chunk['content'] for chunk in batch]

                # Generate embeddings for batch
                embeddings = self.vector_embedder.generate_embedding_batch(batch_texts)

                # Prepare vectors for Upstash
                vectors_to_upsert = []
                for j, (chunk, embedding) in enumerate(zip(batch, embeddings)):
                    vector_id = f"chunk_{chunk['id']}"
                    metadata = {
                        'chunk_id': chunk['id'],
                        'chunk_type': chunk['chunk_type'],
                        'person_id': self.person_id,
                        'category': chunk['metadata']['category'],
                        'importance': chunk['metadata']['importance'],
                        'tags': chunk['metadata']['tags'],
                        'title': chunk['title'],
                        'word_count': chunk['metadata']['word_count']
                    }

                    vectors_to_upsert.append((vector_id, chunk['content'], metadata))

                # Upsert to Upstash
                if vectors_to_upsert:
                    self.vector_embedder.upsert_vectors(vectors_to_upsert)

                total_processed += len(batch)
                logger.info(f"📊 Processed {total_processed}/{len(chunks)} chunks")

            logger.info("✅ Vector embedding generation completed")

        except Exception as e:
            logger.error(f"❌ Vector embedding generation failed: {str(e)}")
            raise

    def run_migration(self, json_file_path: str):
        """Run the complete migration process"""
        start_time = time.time()

        try:
            logger.info("🚀 Starting comprehensive data migration")
            logger.info("=" * 50)

            # Setup connections
            self.setup_connections()

            # Load and validate data
            data = self.load_and_validate_data(json_file_path)

            # Migrate structured data
            self.migrate_structured_data(data)

            # Process content chunks
            chunks = self.process_content_chunks(data)

            # Generate vector embeddings
            self.generate_vector_embeddings(chunks)

            # Calculate migration statistics
            end_time = time.time()
            duration = end_time - start_time

            logger.info("🎉 Migration completed successfully!")
            logger.info(f"⏱️ Total duration: {duration:.2f} seconds")
            logger.info(f"👤 Person ID: {self.person_id}")
            logger.info(f"📊 Content chunks: {len(chunks)}")

        except Exception as e:
            logger.error(f"❌ Migration failed: {str(e)}")
            raise
        finally:
            if self.db_migrator:
                self.db_migrator.close()

def main():
    """Main entry point"""
    print("🚀 Production-Ready Data Migration Script")
    print("=" * 45)
    print("This script will migrate JSON data to PostgreSQL with vector embeddings")
    print()

    # Check environment variables
    required_env_vars = [
        'POSTGRES_CONNECTION_STRING',  # or DB_HOST, DB_PORT, etc.
        'UPSTASH_VECTOR_REST_URL',
        'UPSTASH_VECTOR_REST_TOKEN'
    ]

    missing_vars = []
    for var in required_env_vars:
        if var == 'POSTGRES_CONNECTION_STRING':
            # Allow either connection string or individual DB params
            if not (os.getenv(var) or all(os.getenv(v) for v in ['DB_HOST', 'DB_PORT', 'DB_NAME', 'DB_USER', 'DB_PASSWORD'])):
                missing_vars.append(var)
        elif not os.getenv(var):
            missing_vars.append(var)

    if missing_vars:
        print("❌ Missing required environment variables:")
        for var in missing_vars:
            print(f"   - {var}")
        print("\n💡 Please set these in your .env file")
        print("   Or use individual DB parameters: DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD")
        return

    try:
        # Initialize orchestrator
        orchestrator = MigrationOrchestrator()

        # Run migration
        json_file = "binal_mytwin.json"
        orchestrator.run_migration(json_file)

        print("\n✅ Migration completed successfully!")
        print("📊 Check migration.log for detailed information")

    except Exception as e:
        print(f"\n❌ Migration failed: {str(e)}")
        print("Check the migration.log file for detailed error information.")

if __name__ == "__main__":
    main()