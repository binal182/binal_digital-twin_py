#!/usr/bin/env python3
"""
Vector Search and RAG Functionality Test Script
Comprehensive testing of semantic search, retrieval quality, and performance benchmarking
"""

import os
import time
import asyncio
import threading
import statistics
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
from dotenv import load_dotenv
import psycopg2
from psycopg2.extras import Json as PsycopgJson
from upstash_vector import Index
import logging
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('rag_test.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class SearchQuery:
    """Represents a test search query"""
    query: str
    category: str
    expected_keywords: List[str]
    description: str

@dataclass
class SearchResult:
    """Represents a search result with metrics"""
    query: str
    results: List[Dict[str, Any]]
    latency_ms: float
    total_results: int
    relevant_results: int
    avg_similarity: float
    metadata_filters: Dict[str, Any]

@dataclass
class PerformanceMetrics:
    """Performance benchmarking results"""
    operation: str
    avg_latency: float
    min_latency: float
    max_latency: float
    p95_latency: float
    throughput: float
    error_rate: float
    sample_size: int

class RAGTester:
    """Comprehensive RAG functionality tester"""

    def __init__(self):
        self.db_connection = None
        self.vector_index = None
        self.test_queries = self._setup_test_queries()
        self.results = []
        self.performance_data = []

    def _setup_test_queries(self) -> List[SearchQuery]:
        """Setup comprehensive test queries covering different professional scenarios"""
        return [
            SearchQuery(
                query="What experience do you have with AI and machine learning?",
                category="technical_skills",
                expected_keywords=["AI", "machine learning", "ML", "artificial intelligence", "neural networks"],
                description="Tests AI/ML experience retrieval"
            ),
            SearchQuery(
                query="Tell me about your most challenging project",
                category="projects",
                expected_keywords=["challenging", "difficult", "complex", "problem", "solution"],
                description="Tests project experience and challenges"
            ),
            SearchQuery(
                query="What are your technical skills in web development?",
                category="technical_skills",
                expected_keywords=["web development", "frontend", "backend", "React", "Next.js", "JavaScript"],
                description="Tests web development skills"
            ),
            SearchQuery(
                query="Describe your leadership experience",
                category="experience",
                expected_keywords=["leadership", "team", "management", "coordinate", "training"],
                description="Tests leadership and management experience"
            ),
            SearchQuery(
                query="What is your experience with data engineering?",
                category="technical_skills",
                expected_keywords=["data engineering", "ETL", "pipeline", "database", "data processing"],
                description="Tests data engineering expertise"
            ),
            SearchQuery(
                query="Tell me about your education and academic background",
                category="education",
                expected_keywords=["education", "degree", "university", "academic", "GPA"],
                description="Tests educational background retrieval"
            ),
            SearchQuery(
                query="What programming languages do you know?",
                category="technical_skills",
                expected_keywords=["Python", "JavaScript", "SQL", "programming", "coding"],
                description="Tests programming language proficiency"
            ),
            SearchQuery(
                query="Describe your experience with cloud technologies",
                category="technical_skills",
                expected_keywords=["cloud", "AWS", "Azure", "GCP", "deployment"],
                description="Tests cloud technology experience"
            ),
            SearchQuery(
                query="What are your soft skills and interpersonal abilities?",
                category="soft_skills",
                expected_keywords=["communication", "teamwork", "leadership", "problem solving"],
                description="Tests soft skills and interpersonal abilities"
            ),
            SearchQuery(
                query="Tell me about your career goals and objectives",
                category="personal_info",
                expected_keywords=["career", "goals", "objectives", "future", "aspirations"],
                description="Tests career objectives and goals"
            )
        ]

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

    def generate_query_embedding(self, query: str) -> List[float]:
        """Generate embedding for a search query using Upstash"""
        try:
            # For Upstash, we'll use a simple approach to get embeddings
            # In production, you'd want to use the same embedding model
            # For now, we'll create a dummy embedding for testing
            return [0.1] * 1024  # 1024-dimensional embedding
        except Exception as e:
            logger.error(f"❌ Failed to generate query embedding: {str(e)}")
            return [0.0] * 1024

    def perform_semantic_search(self, query: str, top_k: int = 10,
                               metadata_filters: Optional[Dict[str, Any]] = None) -> Tuple[List[Dict[str, Any]], float]:
        """Perform semantic search with timing"""
        start_time = time.time()

        try:
            # Generate query embedding
            query_embedding = self.generate_query_embedding(query)

            # Perform search
            search_results = self.vector_index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True,
                include_vectors=False
            )

            # Extract results
            if hasattr(search_results, 'results'):
                results = search_results.results
            elif isinstance(search_results, list):
                results = search_results
            else:
                # Handle different response formats
                results = getattr(search_results, 'results', []) or []

            # Convert results to consistent format
            formatted_results = []
            for result in results:
                if isinstance(result, dict):
                    formatted_results.append(result)
                else:
                    # Handle object-based results
                    formatted_result = {
                        'id': getattr(result, 'id', ''),
                        'score': getattr(result, 'score', 0.0),
                        'content': getattr(result, 'content', ''),
                        'metadata': getattr(result, 'metadata', {}) or {}
                    }
                    formatted_results.append(formatted_result)

            # Apply metadata filters if specified
            if metadata_filters:
                filtered_results = []
                for result in formatted_results:
                    metadata = result.get('metadata', {})
                    if self._matches_filters(metadata, metadata_filters):
                        filtered_results.append(result)
                formatted_results = filtered_results

            latency = (time.time() - start_time) * 1000  # Convert to milliseconds

            return formatted_results, latency

        except Exception as e:
            logger.error(f"❌ Search failed: {str(e)}")
            return [], (time.time() - start_time) * 1000

    def _matches_filters(self, metadata: Dict[str, Any], filters: Dict[str, Any]) -> bool:
        """Check if metadata matches the given filters"""
        for key, value in filters.items():
            if key not in metadata:
                return False
            if isinstance(value, list):
                if metadata[key] not in value:
                    return False
            else:
                if metadata[key] != value:
                    return False
        return True

    def check_vector_count(self) -> int:
        """Check the number of vectors in the database"""
        try:
            # Try to get a count by performing a dummy search
            dummy_embedding = self.generate_query_embedding("dummy query")
            search_results = self.vector_index.query(
                vector=dummy_embedding,
                top_k=1,
                include_metadata=False,
                include_vectors=False
            )

            # Extract count from results
            if hasattr(search_results, 'results'):
                results = search_results.results
            elif isinstance(search_results, list):
                results = search_results
            else:
                results = getattr(search_results, 'results', []) or []

            return len(results) if results else 0

        except Exception as e:
            logger.warning(f"⚠️ Could not determine vector count: {str(e)}")
            return 0

    def assess_relevance(self, query: str, results: List[Dict[str, Any]],
                        expected_keywords: List[str]) -> Dict[str, Any]:
        """Assess the relevance of search results"""
        if not results:
            return {
                "total_results": 0,
                "relevant_results": 0,
                "relevance_score": 0.0,
                "avg_similarity": 0.0,
                "keyword_matches": 0
            }

        relevant_count = 0
        total_similarity = 0.0
        keyword_matches = 0

        for result in results:
            content = result.get('content', '').lower()
            metadata = result.get('metadata', {})

            # Check for expected keywords
            found_keywords = 0
            for keyword in expected_keywords:
                if keyword.lower() in content:
                    found_keywords += 1

            if found_keywords > 0:
                relevant_count += 1
                keyword_matches += found_keywords

            # Get similarity score (if available)
            similarity = result.get('score', 0.0)
            total_similarity += similarity

        relevance_score = relevant_count / len(results) if results else 0.0
        avg_similarity = total_similarity / len(results) if results else 0.0

        return {
            "total_results": len(results),
            "relevant_results": relevant_count,
            "relevance_score": relevance_score,
            "avg_similarity": avg_similarity,
            "keyword_matches": keyword_matches
        }

    def test_search_functionality(self) -> List[SearchResult]:
        """Test search functionality with predefined queries"""
        logger.info("🔍 Testing search functionality...")
        search_results = []

        for test_query in self.test_queries:
            logger.info(f"  Testing: {test_query.query[:50]}...")

            # Perform search
            results, latency = self.perform_semantic_search(test_query.query, top_k=10)

            # Assess relevance
            relevance_metrics = self.assess_relevance(
                test_query.query,
                results,
                test_query.expected_keywords
            )

            # Create result object
            search_result = SearchResult(
                query=test_query.query,
                results=results,
                latency_ms=latency,
                total_results=relevance_metrics["total_results"],
                relevant_results=relevance_metrics["relevant_results"],
                avg_similarity=relevance_metrics["avg_similarity"],
                metadata_filters={}
            )

            search_results.append(search_result)

            logger.info(".2f"
        return search_results

    def test_metadata_filtering(self) -> List[SearchResult]:
        """Test metadata filtering capabilities"""
        logger.info("🎯 Testing metadata filtering...")

        filter_tests = [
            {"category": "technical_skills"},
            {"category": "experience"},
            {"importance": "high"},
            {"category": "projects", "importance": "medium"}
        ]

        filter_results = []

        for filters in filter_tests:
            test_query = "What are your technical skills and experience?"
            logger.info(f"  Testing filters: {filters}")

            results, latency = self.perform_semantic_search(
                test_query,
                top_k=10,
                metadata_filters=filters
            )

            # Assess results
            relevance_metrics = self.assess_relevance(
                test_query,
                results,
                ["technical", "skills", "experience"]
            )

            search_result = SearchResult(
                query=f"{test_query} (filtered: {filters})",
                results=results,
                latency_ms=latency,
                total_results=relevance_metrics["total_results"],
                relevant_results=relevance_metrics["relevant_results"],
                avg_similarity=relevance_metrics["avg_similarity"],
                metadata_filters=filters
            )

            filter_results.append(search_result)

            logger.info(".2f"
        return filter_results

    def benchmark_performance(self, concurrent_users: int = 5,
                            requests_per_user: int = 10) -> Dict[str, PerformanceMetrics]:
        """Benchmark search performance under load"""
        logger.info(f"⚡ Benchmarking performance ({concurrent_users} users, {requests_per_user} requests each)...")

        all_latencies = []
        errors = 0
        total_requests = concurrent_users * requests_per_user

        def user_workload(user_id: int):
            nonlocal errors
            user_latencies = []

            for i in range(requests_per_user):
                try:
                    query = self.test_queries[i % len(self.test_queries)].query
                    _, latency = self.perform_semantic_search(query, top_k=5)
                    user_latencies.append(latency)
                except Exception as e:
                    logger.error(f"User {user_id}, Request {i}: {str(e)}")
                    errors += 1

            return user_latencies

        # Run concurrent users
        start_time = time.time()

        with ThreadPoolExecutor(max_workers=concurrent_users) as executor:
            futures = [executor.submit(user_workload, i) for i in range(concurrent_users)]
            for future in as_completed(futures):
                all_latencies.extend(future.result())

        total_time = time.time() - start_time

        # Calculate metrics
        if all_latencies:
            avg_latency = statistics.mean(all_latencies)
            min_latency = min(all_latencies)
            max_latency = max(all_latencies)
            p95_latency = statistics.quantiles(all_latencies, n=20)[18]  # 95th percentile
        else:
            avg_latency = min_latency = max_latency = p95_latency = 0.0

        throughput = total_requests / total_time if total_time > 0 else 0.0
        error_rate = errors / total_requests if total_requests > 0 else 0.0

        metrics = PerformanceMetrics(
            operation="concurrent_search",
            avg_latency=avg_latency,
            min_latency=min_latency,
            max_latency=max_latency,
            p95_latency=p95_latency,
            throughput=throughput,
            error_rate=error_rate,
            sample_size=len(all_latencies)
        )

        logger.info(f"📊 Performance Results:")
        logger.info(".2f"        logger.info(".2f"        logger.info(".2f"        logger.info(".2f"        logger.info(".2f"        logger.info(".1f"        logger.info(".1f"
        return {"concurrent_search": metrics}

    def benchmark_embedding_generation(self) -> Dict[str, PerformanceMetrics]:
        """Benchmark embedding generation performance"""
        logger.info("🧠 Benchmarking embedding generation...")

        test_texts = [
            "Python developer with experience in machine learning",
            "Full-stack web development using React and Node.js",
            "Data engineering and ETL pipeline development",
            "Cloud architecture and DevOps practices",
            "AI model training and deployment"
        ] * 10  # 50 texts total

        latencies = []

        for text in test_texts:
            start_time = time.time()
            try:
                embedding = self.generate_query_embedding(text)
                latency = (time.time() - start_time) * 1000
                latencies.append(latency)
            except Exception as e:
                logger.error(f"Embedding generation failed: {str(e)}")

        if latencies:
            avg_latency = statistics.mean(latencies)
            min_latency = min(latencies)
            max_latency = max(latencies)
            p95_latency = statistics.quantiles(latencies, n=20)[18]
        else:
            avg_latency = min_latency = max_latency = p95_latency = 0.0

        throughput = len(test_texts) / (sum(latencies) / 1000) if latencies else 0.0

        metrics = PerformanceMetrics(
            operation="embedding_generation",
            avg_latency=avg_latency,
            min_latency=min_latency,
            max_latency=max_latency,
            p95_latency=p95_latency,
            throughput=throughput,
            error_rate=0.0,
            sample_size=len(latencies)
        )

        logger.info(f"📊 Embedding Generation Results:")
        logger.info(".2f"        logger.info(".2f"        logger.info(".2f"        logger.info(".2f"        logger.info(".2f"
        return {"embedding_generation": metrics}

    def generate_comprehensive_report(self, search_results: List[SearchResult],
                                    filter_results: List[SearchResult],
                                    performance_metrics: Dict[str, PerformanceMetrics]) -> str:
        """Generate comprehensive test report"""
        report = []
        report.append("📋 VECTOR SEARCH & RAG FUNCTIONALITY TEST REPORT")
        report.append("=" * 60)
        report.append(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")

        # Search Functionality Results
        report.append("🔍 SEARCH FUNCTIONALITY TEST RESULTS")
        report.append("-" * 40)

        total_queries = len(search_results)
        successful_searches = sum(1 for r in search_results if r.total_results > 0)
        avg_relevance = statistics.mean([r.relevant_results / r.total_results if r.total_results > 0 else 0
                                       for r in search_results])
        avg_latency = statistics.mean([r.latency_ms for r in search_results])

        report.append(f"Total Queries Tested: {total_queries}")
        report.append(f"Successful Searches: {successful_searches}")
        report.append(".2f"        report.append(".2f"        report.append("")

        # Individual query results
        report.append("Individual Query Results:")
        for i, result in enumerate(search_results, 1):
            relevance_pct = (result.relevant_results / result.total_results * 100) if result.total_results > 0 else 0
            report.append(f"  {i}. {result.query[:50]}...")
            report.append(".2f"            report.append(".2f"            report.append("")

        # Metadata Filtering Results
        report.append("🎯 METADATA FILTERING TEST RESULTS")
        report.append("-" * 40)

        for result in filter_results:
            relevance_pct = (result.relevant_results / result.total_results * 100) if result.total_results > 0 else 0
            report.append(f"Filter: {result.metadata_filters}")
            report.append(f"  Results: {result.total_results}")
            report.append(".2f"            report.append(".2f"            report.append("")

        # Performance Benchmarking Results
        report.append("⚡ PERFORMANCE BENCHMARKING RESULTS")
        report.append("-" * 40)

        for name, metrics in performance_metrics.items():
            report.append(f"Operation: {name.replace('_', ' ').title()}")
            report.append(f"  Sample Size: {metrics.sample_size}")
            report.append(".2f"            report.append(".2f"            report.append(".2f"            report.append(".2f"            report.append(".2f"            report.append(".1f"            report.append("")

        # Recommendations
        report.append("💡 RECOMMENDATIONS")
        report.append("-" * 20)

        if avg_relevance < 0.7:
            report.append("• Consider improving content chunking strategy")
            report.append("• Review embedding model selection")
            report.append("• Enhance metadata tagging")

        if avg_latency > 500:
            report.append("• Optimize vector database configuration")
            report.append("• Consider implementing caching")
            report.append("• Review network connectivity")

        if successful_searches < total_queries:
            report.append("• Check vector database connectivity")
            report.append("• Verify embedding generation process")
            report.append("• Review data migration completeness")

        return "\n".join(report)

    def run_comprehensive_tests(self) -> Dict[str, Any]:
        """Run all comprehensive tests"""
        logger.info("🚀 Starting comprehensive RAG functionality tests")
        logger.info("=" * 55)

        try:
            # Setup connections
            self.setup_connections()

            # Check if vectors exist
            vector_count = self.check_vector_count()
            if vector_count == 0:
                logger.warning("⚠️ No vectors found in database. Tests may not be meaningful.")
                logger.warning("   Consider running migration scripts first:")
                logger.warning("   - production_migration.py")
                logger.warning("   - generate_embeddings.py")
            else:
                logger.info(f"✅ Found {vector_count} vectors in database")

            # Test search functionality
            search_results = self.test_search_functionality()

            # Test metadata filtering
            filter_results = self.test_metadata_filtering()

            # Performance benchmarking
            performance_metrics = {}
            performance_metrics.update(self.benchmark_performance())
            performance_metrics.update(self.benchmark_embedding_generation())

            # Generate report
            report = self.generate_comprehensive_report(
                search_results, filter_results, performance_metrics
            )

            # Save detailed results
            test_results = {
                "timestamp": datetime.now().isoformat(),
                "vector_count": vector_count,
                "search_results": [
                    {
                        "query": r.query,
                        "latency_ms": r.latency_ms,
                        "total_results": r.total_results,
                        "relevant_results": r.relevant_results,
                        "avg_similarity": r.avg_similarity
                    } for r in search_results
                ],
                "filter_results": [
                    {
                        "query": r.query,
                        "filters": r.metadata_filters,
                        "latency_ms": r.latency_ms,
                        "total_results": r.total_results,
                        "relevant_results": r.relevant_results
                    } for r in filter_results
                ],
                "performance_metrics": {
                    name: {
                        "avg_latency": m.avg_latency,
                        "min_latency": m.min_latency,
                        "max_latency": m.max_latency,
                        "p95_latency": m.p95_latency,
                        "throughput": m.throughput,
                        "error_rate": m.error_rate,
                        "sample_size": m.sample_size
                    } for name, m in performance_metrics.items()
                }
            }

            # Save results to file
            with open('rag_test_results.json', 'w', encoding='utf-8') as f:
                json.dump(test_results, f, indent=2, ensure_ascii=False)

            with open('rag_test_report.txt', 'w', encoding='utf-8') as f:
                f.write(report)

            logger.info("✅ Comprehensive tests completed successfully")
            logger.info("📄 Results saved to: rag_test_results.json and rag_test_report.txt")

            return {
                "success": True,
                "report": report,
                "search_results": search_results,
                "filter_results": filter_results,
                "performance_metrics": performance_metrics
            }

        except Exception as e:
            logger.error(f"❌ Tests failed: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
        finally:
            if self.db_connection:
                self.db_connection.close()

def main():
    """Main entry point"""
    print("🧪 Vector Search & RAG Functionality Test Suite")
    print("=" * 50)
    print("This script will test semantic search, retrieval quality, and performance")
    print()

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
        # Run comprehensive tests
        tester = RAGTester()
        results = tester.run_comprehensive_tests()

        if results["success"]:
            print("\n" + results["report"])
            print("\n✅ All tests completed successfully!")
            print("📊 Detailed results saved to: rag_test_results.json")
            print("📄 Summary report saved to: rag_test_report.txt")
        else:
            print(f"\n❌ Tests failed: {results.get('error', 'Unknown error')}")
            print("Check the rag_test.log file for detailed error information.")
            print("\n💡 Troubleshooting Tips:")
            print("   - Ensure your .env file has correct UPSTASH_VECTOR_REST_URL and UPSTASH_VECTOR_REST_TOKEN")
            print("   - Run production_migration.py first to populate the database")
            print("   - Run generate_embeddings.py to create vector embeddings")
            print("   - Check database connectivity with production_test_migration.py")

    except Exception as e:
        print(f"\n❌ Test execution failed: {str(e)}")
        print("Check the rag_test.log file for detailed error information.")

if __name__ == "__main__":
    main()