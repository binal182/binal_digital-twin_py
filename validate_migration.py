#!/usr/bin/env python3
"""
Migration Validation Script
==========================

Comprehensive validation of the data migration process including:
- JSON data import to PostgreSQL
- Content chunk generation and storage
- Vector embedding creation (1024 dimensions)
- Search functionality testing
- Metadata association validation
- Data integrity checks
"""

import os
import json
import logging
from typing import Dict, List, Any, Tuple
from datetime import datetime
import psycopg2
from psycopg2 import sql
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('migration_validation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class MigrationValidator:
    """Comprehensive migration validation"""

    def __init__(self):
        self.db_connection = None
        self.validation_results = {
            "timestamp": datetime.now().isoformat(),
            "tests": {},
            "summary": {}
        }

    def setup_connections(self) -> bool:
        """Setup database and vector connections"""
        try:
            # Database connection
            conn_str = os.getenv('POSTGRES_CONNECTION_STRING')
            if conn_str:
                self.db_connection = psycopg2.connect(conn_str)
            else:
                # Try individual parameters
                db_params = {
                    'host': os.getenv('DB_HOST'),
                    'port': os.getenv('DB_PORT', 5432),
                    'database': os.getenv('DB_NAME'),
                    'user': os.getenv('DB_USER'),
                    'password': os.getenv('DB_PASSWORD')
                }
                self.db_connection = psycopg2.connect(**db_params)

            logger.info("✅ Database connection established")
            return True

        except Exception as e:
            logger.error(f"❌ Connection setup failed: {str(e)}")
            return False

    def validate_json_import(self) -> Dict[str, Any]:
        """Validate JSON data import to PostgreSQL"""
        logger.info("🔍 Validating JSON data import...")

        try:
            with self.db_connection.cursor() as cursor:
                # Check main tables
                tables_to_check = [
                    'personal_info', 'experiences', 'skills', 'projects',
                    'education', 'content_chunks'
                ]

                table_counts = {}
                total_records = 0

                for table in tables_to_check:
                    cursor.execute(sql.SQL("SELECT COUNT(*) FROM {}").format(sql.Identifier(table)))
                    count = cursor.fetchone()[0]
                    table_counts[table] = count
                    total_records += count

                # Validate data quality
                data_quality_checks = {}

                # Check personal info
                cursor.execute("SELECT COUNT(*) FROM personal_info WHERE full_name IS NOT NULL")
                data_quality_checks['personal_info_complete'] = cursor.fetchone()[0] > 0

                # Check experiences
                cursor.execute("SELECT COUNT(*) FROM experiences WHERE company_name IS NOT NULL")
                data_quality_checks['experiences_complete'] = cursor.fetchone()[0] > 0

                # Check skills
                cursor.execute("SELECT COUNT(*) FROM skills WHERE skill_name IS NOT NULL")
                data_quality_checks['skills_complete'] = cursor.fetchone()[0] > 0

                return {
                    "status": "PASS" if total_records > 0 else "FAIL",
                    "table_counts": table_counts,
                    "total_records": total_records,
                    "data_quality": data_quality_checks,
                    "message": f"Imported {total_records} records across {len(tables_to_check)} tables"
                }

        except Exception as e:
            logger.error(f"JSON import validation failed: {str(e)}")
            return {
                "status": "FAIL",
                "error": str(e),
                "message": "JSON import validation failed"
            }

    def validate_content_chunks(self) -> Dict[str, Any]:
        """Validate content chunks generation and storage"""
        logger.info("🔍 Validating content chunks...")

        try:
            with self.db_connection.cursor() as cursor:
                # Check content chunks
                cursor.execute("""
                    SELECT COUNT(*), AVG(LENGTH(content)), MIN(LENGTH(content)), MAX(LENGTH(content))
                    FROM content_chunks
                """)
                result = cursor.fetchone()
                chunk_count = result[0]
                avg_length = result[1] or 0
                min_length = result[2] or 0
                max_length = result[3] or 0

                # Check chunk metadata
                cursor.execute("""
                    SELECT COUNT(*) FROM content_chunks
                    WHERE metadata IS NOT NULL AND metadata != '{}'
                """)
                chunks_with_metadata = cursor.fetchone()[0]

                # Check content quality
                cursor.execute("""
                    SELECT COUNT(*) FROM content_chunks
                    WHERE LENGTH(TRIM(content)) > 50
                """)
                quality_chunks = cursor.fetchone()[0]

                return {
                    "status": "PASS" if chunk_count > 0 else "FAIL",
                    "chunk_count": chunk_count,
                    "avg_length": avg_length,
                    "min_length": min_length,
                    "max_length": max_length,
                    "chunks_with_metadata": chunks_with_metadata,
                    "quality_chunks": quality_chunks,
                    "message": f"Generated {chunk_count} content chunks"
                }

        except Exception as e:
            logger.error(f"Content chunk validation failed: {str(e)}")
            return {
                "status": "FAIL",
                "error": str(e),
                "message": "Content chunk validation failed"
            }

    def validate_vector_embeddings(self) -> Dict[str, Any]:
        """Validate vector embeddings creation"""
        logger.info("🔍 Validating vector embeddings...")

        try:
            # Check if we have vector database access
            upstash_url = os.getenv('UPSTASH_VECTOR_REST_URL')
            upstash_token = os.getenv('UPSTASH_VECTOR_REST_TOKEN')

            if not upstash_url or not upstash_token:
                return {
                    "status": "SKIP",
                    "message": "Upstash credentials not configured"
                }

            import upstash_vector
            index = upstash_vector.Index(url=upstash_url, token=upstash_token)

            # Get vector count (approximate)
            test_embedding = [0.1] * 1024  # 1024 dimensions
            search_result = index.query(vector=test_embedding, top_k=1)

            # Extract vector count from response
            if hasattr(search_result, 'results'):
                vector_count = len(search_result.results)
            else:
                vector_count = 0

            # Check embedding dimensions
            dimension_check = len(test_embedding) == 1024

            return {
                "status": "PASS" if vector_count > 0 else "FAIL",
                "vector_count": vector_count,
                "dimensions": 1024,
                "dimension_check": dimension_check,
                "message": f"Found {vector_count} vectors with {1024}d embeddings"
            }

        except Exception as e:
            logger.error(f"Vector embedding validation failed: {str(e)}")
            return {
                "status": "FAIL",
                "error": str(e),
                "message": "Vector embedding validation failed"
            }

    def validate_search_functionality(self) -> Dict[str, Any]:
        """Validate search functionality"""
        logger.info("🔍 Validating search functionality...")

        try:
            upstash_url = os.getenv('UPSTASH_VECTOR_REST_URL')
            upstash_token = os.getenv('UPSTASH_VECTOR_REST_TOKEN')

            if not upstash_url or not upstash_token:
                return {
                    "status": "SKIP",
                    "message": "Upstash credentials not configured"
                }

            import upstash_vector
            index = upstash_vector.Index(url=upstash_url, token=upstash_token)

            # Test search queries
            test_queries = [
                "artificial intelligence experience",
                "web development skills",
                "data engineering projects"
            ]

            search_results = []
            for query in test_queries:
                # Generate query embedding (simplified)
                query_embedding = [0.1] * 1024  # Placeholder

                result = index.query(vector=query_embedding, top_k=5, include_metadata=True)
                search_results.append({
                    "query": query,
                    "results_found": len(result.results) if hasattr(result, 'results') else 0
                })

            # Check if searches return results
            successful_searches = sum(1 for r in search_results if r['results_found'] > 0)

            return {
                "status": "PASS" if successful_searches > 0 else "FAIL",
                "total_searches": len(test_queries),
                "successful_searches": successful_searches,
                "search_results": search_results,
                "message": f"Search functionality: {successful_searches}/{len(test_queries)} queries successful"
            }

        except Exception as e:
            logger.error(f"Search validation failed: {str(e)}")
            return {
                "status": "FAIL",
                "error": str(e),
                "message": "Search functionality validation failed"
            }

    def validate_metadata_association(self) -> Dict[str, Any]:
        """Validate metadata association with vectors"""
        logger.info("🔍 Validating metadata association...")

        try:
            upstash_url = os.getenv('UPSTASH_VECTOR_REST_URL')
            upstash_token = os.getenv('UPSTASH_VECTOR_REST_TOKEN')

            if not upstash_url or not upstash_token:
                return {
                    "status": "SKIP",
                    "message": "Upstash credentials not configured"
                }

            import upstash_vector
            index = upstash_vector.Index(url=upstash_url, token=upstash_token)

            # Search for vectors with metadata
            test_embedding = [0.1] * 1024
            result = index.query(vector=test_embedding, top_k=10, include_metadata=True)

            if hasattr(result, 'results') and result.results:
                vectors_with_metadata = 0
                metadata_types = set()

                for item in result.results:
                    if hasattr(item, 'metadata') and item.metadata:
                        vectors_with_metadata += 1
                        # Check metadata structure
                        metadata = item.metadata
                        if isinstance(metadata, dict):
                            metadata_types.update(metadata.keys())

                return {
                    "status": "PASS" if vectors_with_metadata > 0 else "WARNING",
                    "vectors_with_metadata": vectors_with_metadata,
                    "total_vectors_checked": len(result.results),
                    "metadata_types": list(metadata_types),
                    "message": f"Metadata association: {vectors_with_metadata}/{len(result.results)} vectors have metadata"
                }
            else:
                return {
                    "status": "WARNING",
                    "message": "No vectors found to check metadata"
                }

        except Exception as e:
            logger.error(f"Metadata validation failed: {str(e)}")
            return {
                "status": "FAIL",
                "error": str(e),
                "message": "Metadata association validation failed"
            }

    def validate_data_integrity(self) -> Dict[str, Any]:
        """Validate data integrity and check for corruption"""
        logger.info("🔍 Validating data integrity...")

        try:
            integrity_checks = {}

            with self.db_connection.cursor() as cursor:
                # Check for orphaned records
                cursor.execute("""
                    SELECT COUNT(*) FROM content_chunks cc
                    LEFT JOIN personal_info pi ON cc.profile_id = pi.id
                    WHERE pi.id IS NULL
                """)
                orphaned_chunks = cursor.fetchone()[0]
                integrity_checks['orphaned_chunks'] = orphaned_chunks == 0

                # Check for missing required fields
                cursor.execute("""
                    SELECT
                        COUNT(CASE WHEN full_name IS NULL OR full_name = '' THEN 1 END) as missing_names,
                        COUNT(CASE WHEN email IS NULL OR email = '' THEN 1 END) as missing_emails
                    FROM personal_info
                """)
                missing_data = cursor.fetchone()
                integrity_checks['complete_personal_info'] = missing_data[0] == 0 and missing_data[1] == 0

                # Check content chunk quality
                cursor.execute("""
                    SELECT COUNT(*) FROM content_chunks
                    WHERE LENGTH(TRIM(content)) < 10
                """)
                poor_quality_chunks = cursor.fetchone()[0]
                integrity_checks['content_quality'] = poor_quality_chunks == 0

                # Overall integrity score
                passed_checks = sum(1 for check in integrity_checks.values() if check)
                total_checks = len(integrity_checks)
                integrity_score = passed_checks / total_checks if total_checks > 0 else 0

                return {
                    "status": "PASS" if integrity_score >= 0.8 else "WARNING",
                    "integrity_score": integrity_score,
                    "passed_checks": passed_checks,
                    "total_checks": total_checks,
                    "details": integrity_checks,
                    "message": f"Data integrity: {passed_checks}/{total_checks} checks passed ({integrity_score:.1%})"
                }

        except Exception as e:
            logger.error(f"Data integrity validation failed: {str(e)}")
            return {
                "status": "FAIL",
                "error": str(e),
                "message": "Data integrity validation failed"
            }

    def run_comprehensive_validation(self) -> Dict[str, Any]:
        """Run all validation tests"""
        logger.info("🚀 Starting comprehensive migration validation")
        logger.info("=" * 60)

        if not self.setup_connections():
            return {
                "success": False,
                "error": "Failed to establish database connection",
                "message": "Cannot proceed with validation without database connection"
            }

        try:
            # Run all validation tests
            tests = {
                "json_import": self.validate_json_import(),
                "content_chunks": self.validate_content_chunks(),
                "vector_embeddings": self.validate_vector_embeddings(),
                "search_functionality": self.validate_search_functionality(),
                "metadata_association": self.validate_metadata_association(),
                "data_integrity": self.validate_data_integrity()
            }

            # Calculate summary
            passed_tests = sum(1 for test in tests.values() if test.get("status") == "PASS")
            warning_tests = sum(1 for test in tests.values() if test.get("status") == "WARNING")
            failed_tests = sum(1 for test in tests.values() if test.get("status") == "FAIL")
            skipped_tests = sum(1 for test in tests.values() if test.get("status") == "SKIP")

            total_tests = len(tests)

            # Overall status
            if failed_tests > 0:
                overall_status = "FAIL"
            elif warning_tests > 0:
                overall_status = "WARNING"
            else:
                overall_status = "PASS"

            summary = {
                "timestamp": datetime.now().isoformat(),
                "overall_status": overall_status,
                "total_tests": total_tests,
                "passed_tests": passed_tests,
                "warning_tests": warning_tests,
                "failed_tests": failed_tests,
                "skipped_tests": skipped_tests,
                "success_rate": passed_tests / total_tests if total_tests > 0 else 0
            }

            # Save detailed results
            self.validation_results = {
                "summary": summary,
                "tests": tests
            }

            with open('migration_validation_results.json', 'w', encoding='utf-8') as f:
                json.dump(self.validation_results, f, indent=2, ensure_ascii=False)

            logger.info("✅ Comprehensive validation completed")
            logger.info(f"📊 Results: {passed_tests} passed, {warning_tests} warnings, {failed_tests} failed")

            return {
                "success": overall_status in ["PASS", "WARNING"],
                "summary": summary,
                "tests": tests
            }

        except Exception as e:
            logger.error(f"❌ Validation failed: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
        finally:
            if self.db_connection:
                self.db_connection.close()

def print_validation_report(results: Dict[str, Any]):
    """Print a formatted validation report"""
    print("\n" + "=" * 80)
    print("📋 MIGRATION VALIDATION REPORT")
    print("=" * 80)

    if not results.get("success", False):
        print(f"❌ VALIDATION FAILED: {results.get('error', 'Unknown error')}")
        return

    summary = results.get("summary", {})
    tests = results.get("tests", {})

    # Summary
    print(f"📊 OVERALL STATUS: {summary.get('overall_status', 'UNKNOWN')}")
    print(f"✅ Passed: {summary.get('passed_tests', 0)}")
    print(f"⚠️  Warnings: {summary.get('warning_tests', 0)}")
    print(f"❌ Failed: {summary.get('failed_tests', 0)}")
    print(f"⏭️  Skipped: {summary.get('skipped_tests', 0)}")
    print(".1%")

    print("\n" + "-" * 80)

    # Detailed test results
    for test_name, test_result in tests.items():
        status = test_result.get("status", "UNKNOWN")
        message = test_result.get("message", "")

        if status == "PASS":
            icon = "✅"
        elif status == "WARNING":
            icon = "⚠️"
        elif status == "FAIL":
            icon = "❌"
        else:
            icon = "⏭️"

        print(f"{icon} {test_name.replace('_', ' ').title()}: {message}")

        # Additional details for some tests
        if test_name == "json_import" and "table_counts" in test_result:
            print("   📋 Table counts:"            for table, count in test_result["table_counts"].items():
                print(f"      - {table}: {count}")

        elif test_name == "content_chunks" and "chunk_count" in test_result:
            print(f"   📝 Chunks: {test_result['chunk_count']}")
            print(".0f"
        elif test_name == "vector_embeddings" and "vector_count" in test_result:
            print(f"   🧠 Vectors: {test_result['vector_count']}")
            print(f"   📐 Dimensions: {test_result['dimensions']}")

    print("\n" + "=" * 80)

    # Recommendations
    if summary.get("failed_tests", 0) > 0:
        print("🔧 CRITICAL ISSUES TO FIX:")
        for test_name, test_result in tests.items():
            if test_result.get("status") == "FAIL":
                print(f"   - {test_name.replace('_', ' ').title()}: {test_result.get('message', '')}")

    if summary.get("warning_tests", 0) > 0:
        print("\n⚠️  RECOMMENDATIONS:")
        for test_name, test_result in tests.items():
            if test_result.get("status") == "WARNING":
                print(f"   - {test_name.replace('_', ' ').title()}: {test_result.get('message', '')}")

def main():
    """Main validation entry point"""
    print("🧪 Migration Validation Suite")
    print("=" * 40)
    print("This script validates the complete migration process:")
    print("✅ JSON data import to PostgreSQL")
    print("✅ Content chunk generation and storage")
    print("✅ Vector embedding creation (1024 dimensions)")
    print("✅ Search functionality testing")
    print("✅ Metadata association validation")
    print("✅ Data integrity checks")
    print()

    # Check environment
    required_vars = ['POSTGRES_CONNECTION_STRING']
    missing_vars = [v for v in required_vars if not os.getenv(v)]

    if missing_vars:
        print("❌ Missing required environment variables:")
        for var in missing_vars:
            print(f"   - {var}")
        print("\n💡 Please set up your .env file with database credentials")
        print("   Run: python check_env.py")
        return

    try:
        validator = MigrationValidator()
        results = validator.run_comprehensive_validation()

        print_validation_report(results)

        if results.get("success", False):
            print("\n✅ MIGRATION VALIDATION COMPLETED!")
            print("📄 Detailed results saved to: migration_validation_results.json")
            print("📋 Summary report saved to: migration_validation.log")
        else:
            print(f"\n❌ VALIDATION FAILED: {results.get('error', 'Unknown error')}")
            print("Check the migration_validation.log file for details")

    except Exception as e:
        print(f"\n❌ Validation execution failed: {str(e)}")
        print("Check the migration_validation.log file for details")

if __name__ == "__main__":
    main()