#!/usr/bin/env python3
"""
Production Migration Test Script
Comprehensive validation of the data migration process
"""

import os
import psycopg2
from psycopg2.extras import Json as PsycopgJson
from dotenv import load_dotenv
from upstash_vector import Index
import logging
import json
from typing import Dict, List, Any, Tuple

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('migration_test.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class MigrationValidator:
    """Comprehensive validation of migration results"""

    def __init__(self):
        self.db_connection = None
        self.vector_index = None
        self.setup_connections()

    def setup_connections(self):
        """Setup database and vector connections"""
        try:
            # Database connection
            connection_string = os.getenv('POSTGRES_CONNECTION_STRING')
            if connection_string:
                self.db_connection = psycopg2.connect(connection_string)
            else:
                db_params = {
                    'host': os.getenv('DB_HOST', 'localhost'),
                    'port': os.getenv('DB_PORT', '5432'),
                    'database': os.getenv('DB_NAME', 'digital_twin'),
                    'user': os.getenv('DB_USER', 'postgres'),
                    'password': os.getenv('DB_PASSWORD', '')
                }
                self.db_connection = psycopg2.connect(**db_params)

            # Vector connection
            self.vector_index = Index.from_env()

            logger.info("✅ Connected to all services successfully")

        except Exception as e:
            logger.error(f"❌ Connection failed: {str(e)}")
            raise

    def validate_personal_info(self) -> Tuple[bool, Dict[str, Any]]:
        """Validate personal information migration"""
        try:
            with self.db_connection.cursor() as cursor:
                cursor.execute("SELECT COUNT(*) FROM personal_info")
                count = cursor.fetchone()[0]

                if count == 0:
                    return False, {"error": "No personal info records found"}

                cursor.execute("""
                    SELECT full_name, email, summary, data_quality_score
                    FROM personal_info LIMIT 1
                """)
                record = cursor.fetchone()

                result = {
                    "count": count,
                    "name": record[0],
                    "email": record[1],
                    "summary_length": len(record[2] or ""),
                    "data_quality_score": record[3]
                }

                return True, result

        except Exception as e:
            return False, {"error": str(e)}

    def validate_experience_data(self) -> Tuple[bool, Dict[str, Any]]:
        """Validate experience data migration"""
        try:
            with self.db_connection.cursor() as cursor:
                cursor.execute("SELECT COUNT(*) FROM experience")
                count = cursor.fetchone()[0]

                cursor.execute("""
                    SELECT company_name, position_title, description,
                           array_length(achievements, 1) as achievement_count,
                           array_length(technologies_used, 1) as tech_count
                    FROM experience
                    ORDER BY start_date DESC
                """)
                records = cursor.fetchall()

                result = {
                    "count": count,
                    "companies": [r[0] for r in records],
                    "positions": [r[1] for r in records],
                    "avg_description_length": sum(len(r[2] or "") for r in records) / len(records) if records else 0,
                    "total_achievements": sum(r[3] or 0 for r in records),
                    "total_technologies": sum(r[4] or 0 for r in records)
                }

                return True, result

        except Exception as e:
            return False, {"error": str(e)}

    def validate_skills_data(self) -> Tuple[bool, Dict[str, Any]]:
        """Validate skills data migration"""
        try:
            with self.db_connection.cursor() as cursor:
                cursor.execute("""
                    SELECT category, COUNT(*) as count
                    FROM skills
                    GROUP BY category
                """)
                skill_counts = dict(cursor.fetchall())

                cursor.execute("""
                    SELECT skill_name, proficiency_level, is_technical
                    FROM skills
                    ORDER BY skill_name
                """)
                skills = cursor.fetchall()

                result = {
                    "total_skills": sum(skill_counts.values()),
                    "categories": skill_counts,
                    "technical_count": sum(1 for s in skills if s[2]),
                    "soft_count": sum(1 for s in skills if not s[2]),
                    "proficiency_levels": list(set(s[1] for s in skills))
                }

                return True, result

        except Exception as e:
            return False, {"error": str(e)}

    def validate_projects_data(self) -> Tuple[bool, Dict[str, Any]]:
        """Validate projects data migration"""
        try:
            with self.db_connection.cursor() as cursor:
                cursor.execute("SELECT COUNT(*) FROM projects")
                count = cursor.fetchone()[0]

                cursor.execute("""
                    SELECT project_name, description,
                           array_length(technologies_used, 1) as tech_count
                    FROM projects
                """)
                records = cursor.fetchall()

                result = {
                    "count": count,
                    "project_names": [r[0] for r in records],
                    "avg_description_length": sum(len(r[1] or "") for r in records) / len(records) if records else 0,
                    "total_technologies": sum(r[2] or 0 for r in records)
                }

                return True, result

        except Exception as e:
            return False, {"error": str(e)}

    def validate_education_data(self) -> Tuple[bool, Dict[str, Any]]:
        """Validate education data migration"""
        try:
            with self.db_connection.cursor() as cursor:
                cursor.execute("SELECT COUNT(*) FROM education")
                count = cursor.fetchone()[0]

                cursor.execute("""
                    SELECT institution_name, degree, field_of_study, gpa
                    FROM education
                    ORDER BY start_date DESC
                """)
                records = cursor.fetchall()

                result = {
                    "count": count,
                    "institutions": [r[0] for r in records],
                    "degrees": [r[1] for r in records],
                    "fields": [r[2] for r in records],
                    "gpa_values": [r[3] for r in records if r[3] is not None]
                }

                return True, result

        except Exception as e:
            return False, {"error": str(e)}

    def validate_content_chunks(self) -> Tuple[bool, Dict[str, Any]]:
        """Validate content chunks migration"""
        try:
            with self.db_connection.cursor() as cursor:
                cursor.execute("SELECT COUNT(*) FROM content_chunks")
                count = cursor.fetchone()[0]

                cursor.execute("""
                    SELECT chunk_type, COUNT(*) as count,
                           AVG(LENGTH(content)) as avg_length,
                           AVG((metadata->>'word_count')::int) as avg_words
                    FROM content_chunks
                    GROUP BY chunk_type
                """)
                chunk_stats = cursor.fetchall()

                cursor.execute("""
                    SELECT metadata->>'category' as category,
                           metadata->>'importance' as importance,
                           COUNT(*) as count
                    FROM content_chunks
                    GROUP BY metadata->>'category', metadata->>'importance'
                """)
                category_stats = cursor.fetchall()

                result = {
                    "total_chunks": count,
                    "chunk_types": {row[0]: row[1] for row in chunk_stats},
                    "avg_chunk_length": sum(row[2] for row in chunk_stats if row[2]) / len(chunk_stats) if chunk_stats else 0,
                    "avg_word_count": sum(row[3] for row in chunk_stats if row[3]) / len(chunk_stats) if chunk_stats else 0,
                    "categories": {}
                }

                # Organize category stats
                for row in category_stats:
                    category, importance, count = row
                    if category not in result["categories"]:
                        result["categories"][category] = {}
                    result["categories"][category][importance] = count

                return True, result

        except Exception as e:
            return False, {"error": str(e)}

    def validate_vector_embeddings(self) -> Tuple[bool, Dict[str, Any]]:
        """Validate vector embeddings in Upstash"""
        try:
            # Query vectors with chunk metadata
            query_result = self.vector_index.query(
                vector=[0.0] * 1024,  # Dummy vector for metadata query
                top_k=1000,
                include_metadata=True,
                include_vectors=False
            )

            if hasattr(query_result, 'results'):
                vectors = query_result.results
            else:
                vectors = query_result

            chunk_vectors = [v for v in vectors if v.get('metadata', {}).get('chunk_id')]

            result = {
                "total_vectors": len(vectors),
                "chunk_vectors": len(chunk_vectors),
                "vector_dimensions": 1024,  # Expected dimension
                "metadata_fields": set()
            }

            # Collect metadata fields
            for vector in chunk_vectors[:10]:  # Sample first 10
                if 'metadata' in vector:
                    result["metadata_fields"].update(vector['metadata'].keys())

            result["metadata_fields"] = list(result["metadata_fields"])

            return True, result

        except Exception as e:
            return False, {"error": str(e)}

    def validate_json_content(self) -> Tuple[bool, Dict[str, Any]]:
        """Validate JSON content storage"""
        try:
            with self.db_connection.cursor() as cursor:
                cursor.execute("SELECT COUNT(*) FROM json_content")
                count = cursor.fetchone()[0]

                cursor.execute("""
                    SELECT content_type, version, metadata
                    FROM json_content LIMIT 1
                """)
                record = cursor.fetchone()

                result = {
                    "count": count,
                    "content_type": record[0] if record else None,
                    "version": record[1] if record else None,
                    "has_metadata": record[2] is not None if record else False
                }

                return True, result

        except Exception as e:
            return False, {"error": str(e)}

    def run_comprehensive_validation(self) -> Dict[str, Any]:
        """Run all validation checks"""
        logger.info("🧪 Starting comprehensive migration validation")
        logger.info("=" * 50)

        validation_results = {
            "timestamp": str(datetime.now()),
            "overall_status": "pending",
            "validations": {}
        }

        # Define validation checks
        checks = [
            ("personal_info", self.validate_personal_info),
            ("experience", self.validate_experience_data),
            ("skills", self.validate_skills_data),
            ("projects", self.validate_projects_data),
            ("education", self.validate_education_data),
            ("content_chunks", self.validate_content_chunks),
            ("vector_embeddings", self.validate_vector_embeddings),
            ("json_content", self.validate_json_content)
        ]

        passed_checks = 0
        total_checks = len(checks)

        for check_name, check_func in checks:
            logger.info(f"🔍 Validating {check_name}...")
            try:
                success, result = check_func()
                validation_results["validations"][check_name] = {
                    "status": "PASSED" if success else "FAILED",
                    "data": result
                }

                if success:
                    passed_checks += 1
                    logger.info(f"✅ {check_name}: PASSED")
                else:
                    logger.error(f"❌ {check_name}: FAILED - {result.get('error', 'Unknown error')}")

            except Exception as e:
                validation_results["validations"][check_name] = {
                    "status": "ERROR",
                    "data": {"error": str(e)}
                }
                logger.error(f"❌ {check_name}: ERROR - {str(e)}")

        # Overall status
        validation_results["overall_status"] = "PASSED" if passed_checks == total_checks else "FAILED"
        validation_results["summary"] = {
            "total_checks": total_checks,
            "passed_checks": passed_checks,
            "failed_checks": total_checks - passed_checks,
            "success_rate": f"{(passed_checks/total_checks)*100:.1f}%"
        }

        logger.info(f"📊 Validation Summary: {passed_checks}/{total_checks} checks passed")
        logger.info(f"🎯 Success Rate: {validation_results['summary']['success_rate']}")

        return validation_results

    def generate_validation_report(self, results: Dict[str, Any]) -> str:
        """Generate a detailed validation report"""
        report = []
        report.append("📋 MIGRATION VALIDATION REPORT")
        report.append("=" * 50)
        report.append(f"Timestamp: {results['timestamp']}")
        report.append(f"Overall Status: {results['overall_status']}")
        report.append("")

        summary = results['summary']
        report.append("📊 SUMMARY:")
        report.append(f"  Total Checks: {summary['total_checks']}")
        report.append(f"  Passed: {summary['passed_checks']}")
        report.append(f"  Failed: {summary['failed_checks']}")
        report.append(f"  Success Rate: {summary['success_rate']}")
        report.append("")

        report.append("🔍 DETAILED RESULTS:")
        for check_name, check_result in results['validations'].items():
            status = check_result['status']
            status_icon = "✅" if status == "PASSED" else "❌" if status == "FAILED" else "⚠️"

            report.append(f"  {status_icon} {check_name}: {status}")

            if status == "PASSED":
                data = check_result['data']
                if check_name == "personal_info":
                    report.append(f"    - Records: {data['count']}")
                    report.append(f"    - Name: {data['name']}")
                    report.append(f"    - Summary Length: {data['summary_length']} chars")
                elif check_name == "experience":
                    report.append(f"    - Records: {data['count']}")
                    report.append(f"    - Companies: {', '.join(data['companies'][:3])}")
                    report.append(f"    - Total Achievements: {data['total_achievements']}")
                elif check_name == "content_chunks":
                    report.append(f"    - Total Chunks: {data['total_chunks']}")
                    report.append(f"    - Chunk Types: {list(data['chunk_types'].keys())}")
                elif check_name == "vector_embeddings":
                    report.append(f"    - Total Vectors: {data['total_vectors']}")
                    report.append(f"    - Chunk Vectors: {data['chunk_vectors']}")

        return "\n".join(report)

    def close(self):
        """Close connections"""
        if self.db_connection:
            self.db_connection.close()

def main():
    """Main entry point"""
    print("🧪 Production Migration Validation")
    print("=" * 35)

    # Check environment variables
    required_vars = ['UPSTASH_VECTOR_REST_URL', 'UPSTASH_VECTOR_REST_TOKEN']
    missing_vars = [var for var in required_vars if not os.getenv(var)]

    # Check database connection
    has_db = bool(os.getenv('POSTGRES_CONNECTION_STRING') or
                  all(os.getenv(v) for v in ['DB_HOST', 'DB_PORT', 'DB_NAME', 'DB_USER', 'DB_PASSWORD']))

    if missing_vars or not has_db:
        print("❌ Missing required configuration:")
        if missing_vars:
            print("  Vector Database:")
            for var in missing_vars:
                print(f"    - {var}")
        if not has_db:
            print("  Database:")
            print("    - POSTGRES_CONNECTION_STRING or DB_* variables")
        return

    try:
        # Run validation
        validator = MigrationValidator()
        results = validator.run_comprehensive_validation()

        # Generate and save report
        report = validator.generate_validation_report(results)

        with open('migration_validation_report.txt', 'w', encoding='utf-8') as f:
            f.write(report)

        print("\n" + report)
        print(f"\n📄 Detailed report saved to: migration_validation_report.txt")

        # Overall result
        if results['overall_status'] == 'PASSED':
            print("\n🎉 All validation checks PASSED!")
            print("✅ Migration completed successfully")
        else:
            print(f"\n⚠️ Some validation checks FAILED")
            print("❌ Please review the detailed report and fix issues")

    except Exception as e:
        print(f"\n❌ Validation failed: {str(e)}")
        print("Check the migration_test.log file for detailed error information.")
    finally:
        if 'validator' in locals():
            validator.close()

if __name__ == "__main__":
    main()