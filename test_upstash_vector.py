#!/usr/bin/env python3
"""
Upstash Vector Database Test Script
Comprehensive testing of Upstash Vector database connection and functionality
"""

import os
import time
import json
from typing import List, Dict, Any
from dotenv import load_dotenv
from upstash_vector import Index
import logging

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class UpstashVectorTester:
    """Comprehensive tester for Upstash Vector database functionality"""

    def __init__(self):
        """Initialize the tester with environment variables"""
        self.url = os.getenv('UPSTASH_VECTOR_REST_URL')
        self.token = os.getenv('UPSTASH_VECTOR_REST_TOKEN')
        self.readonly_token = os.getenv('UPSTASH_VECTOR_REST_READONLY_TOKEN')

        if not all([self.url, self.token]):
            raise ValueError("Missing required environment variables: UPSTASH_VECTOR_REST_URL, UPSTASH_VECTOR_REST_TOKEN")

        # Initialize the vector index
        self.index = Index(url=self.url, token=self.token)

        # Test data
        self.test_content = [
            {
                'id': 'test_1',
                'content': 'Experienced software engineer with expertise in AI and machine learning',
                'metadata': {'category': 'skills', 'type': 'technical', 'experience_years': 3}
            },
            {
                'id': 'test_2',
                'content': 'Led development of scalable web applications using Next.js and React',
                'metadata': {'category': 'experience', 'type': 'project', 'technologies': ['Next.js', 'React']}
            },
            {
                'id': 'test_3',
                'content': 'Implemented RAG systems for improved document search and retrieval',
                'metadata': {'category': 'projects', 'type': 'ai_ml', 'technologies': ['Python', 'Vector DB']}
            },
            {
                'id': 'test_4',
                'content': 'Data engineering specialist with expertise in ETL pipelines and data warehousing',
                'metadata': {'category': 'skills', 'type': 'technical', 'experience_years': 2}
            },
            {
                'id': 'test_5',
                'content': 'Full-stack developer experienced in cloud deployment and DevOps practices',
                'metadata': {'category': 'experience', 'type': 'general', 'technologies': ['AWS', 'Docker']}
            }
        ]

        self.test_results = {}

    def run_all_tests(self) -> Dict[str, Any]:
        """Run all test suites"""
        print("🚀 Starting Upstash Vector Database Tests")
        print("=" * 60)

        try:
            # Test 1: Connection and Basic Operations
            self.test_connection()

            # Test 2: Database Information
            self.test_database_info()

            # Test 3: Embedding Generation
            self.test_embedding_generation()

            # Test 4: Vector Storage
            self.test_vector_storage()

            # Test 5: Similarity Search
            self.test_similarity_search()

            # Test 6: Metadata Filtering
            self.test_metadata_filtering()

            # Test 7: Performance Validation
            self.test_performance()

            # Test 8: Cleanup
            self.test_cleanup()

            print("\n" + "=" * 60)
            print("✅ All tests completed!")

        except Exception as e:
            logger.error(f"Test suite failed: {e}")
            self.test_results['overall_success'] = False
            self.test_results['error'] = str(e)

        return self.test_results

    def test_connection(self):
        """Test basic connection to Upstash Vector"""
        print("\n🔌 Test 1: Connection Test")
        print("-" * 30)

        try:
            # Test basic connection by getting database info
            info = self.index.info()
            print("✅ Successfully connected to Upstash Vector")
            print(f"   Database URL: {self.url}")
            print(f"   Connection Status: OK")

            self.test_results['connection'] = {
                'status': 'success',
                'info': info
            }

        except Exception as e:
            print(f"❌ Connection failed: {e}")
            self.test_results['connection'] = {
                'status': 'failed',
                'error': str(e)
            }
            raise

    def test_database_info(self):
        """Test database information and configuration"""
        print("\n📊 Test 2: Database Information")
        print("-" * 30)

        try:
            info = self.index.info()

            print("✅ Database information retrieved:")
            print(f"   Vector Count: {info.get('vectorCount', 'N/A')}")
            print(f"   Dimension: {info.get('dimension', 'N/A')}")
            print(f"   Similarity Function: {info.get('similarityFunction', 'N/A')}")
            print(f"   Embedding Model: {info.get('embeddingModel', 'N/A')}")

            # Validate expected configuration
            expected_dimension = 1024
            expected_similarity = 'COSINE'
            expected_model = 'mixbread-large'

            dimension_ok = info.get('dimension') == expected_dimension
            similarity_ok = info.get('similarityFunction') == expected_similarity
            model_ok = info.get('embeddingModel') == expected_model

            print("
🔍 Configuration Validation:"            print(f"   Dimension (1024): {'✅' if dimension_ok else '❌'} {info.get('dimension')}")
            print(f"   Similarity (COSINE): {'✅' if similarity_ok else '❌'} {info.get('similarityFunction')}")
            print(f"   Model (mixbread-large): {'✅' if model_ok else '❌'} {info.get('embeddingModel')}")

            self.test_results['database_info'] = {
                'status': 'success',
                'info': info,
                'validation': {
                    'dimension_ok': dimension_ok,
                    'similarity_ok': similarity_ok,
                    'model_ok': model_ok
                }
            }

        except Exception as e:
            print(f"❌ Database info test failed: {e}")
            self.test_results['database_info'] = {
                'status': 'failed',
                'error': str(e)
            }

    def test_embedding_generation(self):
        """Test embedding generation with built-in model"""
        print("\n🧠 Test 3: Embedding Generation")
        print("-" * 30)

        try:
            # Test embedding generation for sample text
            test_text = "Experienced AI engineer with expertise in machine learning and data science"
            print(f"Testing embedding for: '{test_text[:50]}...'")

            # Use upsert to generate embedding
            result = self.index.upsert(
                vectors=[
                    {
                        'id': 'embedding_test',
                        'data': test_text
                    }
                ]
            )

            print("✅ Embedding generated successfully")

            # Query to get the vector back and check dimensions
            query_result = self.index.query(
                data=test_text,
                top_k=1,
                include_vectors=True
            )

            if query_result and len(query_result) > 0:
                vector = query_result[0].get('vector', [])
                dimension = len(vector)
                print(f"   Embedding dimension: {dimension}")

                if dimension == 1024:
                    print("✅ Correct embedding dimension (1024)")
                else:
                    print(f"⚠️  Unexpected dimension: {dimension} (expected 1024)")

                self.test_results['embedding'] = {
                    'status': 'success',
                    'dimension': dimension,
                    'expected_dimension': 1024,
                    'dimension_correct': dimension == 1024
                }
            else:
                print("❌ Could not retrieve embedding for verification")
                self.test_results['embedding'] = {
                    'status': 'failed',
                    'error': 'Could not retrieve embedding'
                }

        except Exception as e:
            print(f"❌ Embedding test failed: {e}")
            self.test_results['embedding'] = {
                'status': 'failed',
                'error': str(e)
            }

    def test_vector_storage(self):
        """Test vector storage and retrieval"""
        print("\n💾 Test 4: Vector Storage")
        print("-" * 30)

        try:
            # Clear any existing test data
            self.index.delete(ids=['storage_test_1', 'storage_test_2'])

            # Store test vectors
            test_vectors = [
                {
                    'id': 'storage_test_1',
                    'data': 'Python developer with Django experience',
                    'metadata': {'skill': 'Python', 'level': 'expert'}
                },
                {
                    'id': 'storage_test_2',
                    'data': 'Machine learning engineer specializing in NLP',
                    'metadata': {'skill': 'ML', 'level': 'advanced'}
                }
            ]

            result = self.index.upsert(vectors=test_vectors)
            print("✅ Test vectors stored successfully")

            # Retrieve and verify
            query_result = self.index.query(
                data='Python programming',
                top_k=2,
                include_metadata=True
            )

            if query_result and len(query_result) >= 1:
                print("✅ Vector retrieval successful")
                print(f"   Retrieved {len(query_result)} results")

                for i, res in enumerate(query_result[:2]):
                    score = res.get('score', 0)
                    metadata = res.get('metadata', {})
                    print(".3f")

                self.test_results['vector_storage'] = {
                    'status': 'success',
                    'stored_count': len(test_vectors),
                    'retrieved_count': len(query_result)
                }
            else:
                print("❌ Vector retrieval failed")
                self.test_results['vector_storage'] = {
                    'status': 'failed',
                    'error': 'No results retrieved'
                }

        except Exception as e:
            print(f"❌ Vector storage test failed: {e}")
            self.test_results['vector_storage'] = {
                'status': 'failed',
                'error': str(e)
            }

    def test_similarity_search(self):
        """Test similarity search functionality"""
        print("\n🔍 Test 5: Similarity Search")
        print("-" * 30)

        try:
            # Clear existing test data
            test_ids = [f'search_test_{i}' for i in range(1, 6)]
            self.index.delete(ids=test_ids)

            # Insert test data
            vectors = []
            for item in self.test_content:
                vectors.append({
                    'id': item['id'],
                    'data': item['content'],
                    'metadata': item['metadata']
                })

            self.index.upsert(vectors=vectors)
            print("✅ Test data inserted for similarity search")

            # Test similarity search
            search_queries = [
                'artificial intelligence and machine learning expertise',
                'web development with React and Next.js',
                'data engineering and ETL processes'
            ]

            search_results = {}
            for i, query in enumerate(search_queries):
                print(f"\n   Search {i+1}: '{query[:40]}...'")
                results = self.index.query(
                    data=query,
                    top_k=3,
                    include_metadata=True
                )

                if results:
                    for j, result in enumerate(results):
                        score = result.get('score', 0)
                        content = result.get('data', '')[:50]
                        print(".3f")

                    search_results[f'query_{i+1}'] = {
                        'query': query,
                        'results_count': len(results),
                        'top_score': results[0].get('score', 0) if results else 0
                    }
                else:
                    print("     No results found")

            self.test_results['similarity_search'] = {
                'status': 'success',
                'queries_tested': len(search_queries),
                'results': search_results
            }

        except Exception as e:
            print(f"❌ Similarity search test failed: {e}")
            self.test_results['similarity_search'] = {
                'status': 'failed',
                'error': str(e)
            }

    def test_metadata_filtering(self):
        """Test metadata filtering capabilities"""
        print("\n🏷️  Test 6: Metadata Filtering")
        print("-" * 30)

        try:
            # Test metadata filtering
            print("Testing metadata filtering...")

            # Search with metadata filter for technical skills
            results = self.index.query(
                data='programming and development',
                top_k=5,
                include_metadata=True,
                filter="category = 'skills' AND type = 'technical'"
            )

            if results:
                print("✅ Metadata filtering successful")
                print(f"   Filtered results: {len(results)}")
                for result in results:
                    metadata = result.get('metadata', {})
                    print(f"     - {metadata.get('category', 'N/A')}: {metadata.get('type', 'N/A')}")
            else:
                print("⚠️  No results with metadata filter")

            # Test another filter
            results_2 = self.index.query(
                data='project experience',
                top_k=3,
                include_metadata=True,
                filter="category = 'projects'"
            )

            self.test_results['metadata_filtering'] = {
                'status': 'success',
                'filter_tests': 2,
                'results_1_count': len(results) if results else 0,
                'results_2_count': len(results_2) if results_2 else 0
            }

        except Exception as e:
            print(f"❌ Metadata filtering test failed: {e}")
            self.test_results['metadata_filtering'] = {
                'status': 'failed',
                'error': str(e)
            }

    def test_performance(self):
        """Test performance of various operations"""
        print("\n⚡ Test 7: Performance Validation")
        print("-" * 30)

        try:
            performance_results = {}

            # Test embedding generation speed
            print("Testing embedding generation speed...")
            test_texts = [
                'Software engineering professional',
                'Machine learning specialist',
                'Data engineering expert',
                'Full-stack web developer',
                'AI research scientist'
            ]

            start_time = time.time()
            for text in test_texts:
                self.index.upsert(vectors=[{'id': f'perf_test_{len(performance_results)}', 'data': text}])
            embedding_time = time.time() - start_time

            performance_results['embedding_generation'] = {
                'texts_processed': len(test_texts),
                'total_time': embedding_time,
                'avg_time_per_text': embedding_time / len(test_texts)
            }

            print(".3f"
            # Test batch operations
            print("\nTesting batch operations...")
            batch_vectors = [
                {'id': f'batch_test_{i}', 'data': f'Batch test content {i}'}
                for i in range(10)
            ]

            start_time = time.time()
            self.index.upsert(vectors=batch_vectors)
            batch_time = time.time() - start_time

            performance_results['batch_operations'] = {
                'batch_size': len(batch_vectors),
                'total_time': batch_time,
                'avg_time_per_vector': batch_time / len(batch_vectors)
            }

            print(".3f"
            # Test search performance
            print("\nTesting search performance...")
            search_queries = ['AI engineer', 'web developer', 'data scientist']

            search_times = []
            for query in search_queries:
                start_time = time.time()
                self.index.query(data=query, top_k=5)
                search_times.append(time.time() - start_time)

            avg_search_time = sum(search_times) / len(search_times)
            performance_results['search_performance'] = {
                'queries_tested': len(search_queries),
                'avg_search_time': avg_search_time,
                'total_search_time': sum(search_times)
            }

            print(".3f"
            self.test_results['performance'] = {
                'status': 'success',
                'metrics': performance_results
            }

        except Exception as e:
            print(f"❌ Performance test failed: {e}")
            self.test_results['performance'] = {
                'status': 'failed',
                'error': str(e)
            }

    def test_cleanup(self):
        """Clean up test data"""
        print("\n🧹 Test 8: Cleanup")
        print("-" * 30)

        try:
            # Clean up test data
            test_ids = []
            test_ids.extend([f'test_{i}' for i in range(1, 6)])
            test_ids.extend([f'search_test_{i}' for i in range(1, 6)])
            test_ids.extend([f'storage_test_{i}' for i in range(1, 3)])
            test_ids.extend([f'embedding_test'])
            test_ids.extend([f'perf_test_{i}' for i in range(5)])
            test_ids.extend([f'batch_test_{i}' for i in range(10)])

            self.index.delete(ids=test_ids)
            print("✅ Test data cleaned up successfully")

            self.test_results['cleanup'] = {
                'status': 'success',
                'cleaned_ids': test_ids
            }

        except Exception as e:
            print(f"❌ Cleanup failed: {e}")
            self.test_results['cleanup'] = {
                'status': 'failed',
                'error': str(e)
            }

    def print_summary(self):
        """Print test summary"""
        print("\n" + "=" * 60)
        print("📋 TEST SUMMARY")
        print("=" * 60)

        success_count = 0
        total_tests = 0

        for test_name, result in self.test_results.items():
            if test_name in ['overall_success', 'error']:
                continue

            total_tests += 1
            status = result.get('status', 'unknown')
            if status == 'success':
                success_count += 1
                print(f"✅ {test_name.replace('_', ' ').title()}: PASSED")
            else:
                print(f"❌ {test_name.replace('_', ' ').title()}: FAILED")

        print(f"\n📊 Overall: {success_count}/{total_tests} tests passed")

        if success_count == total_tests:
            print("🎉 All tests passed! Your Upstash Vector database is ready.")
        else:
            print("⚠️  Some tests failed. Please check the configuration.")

        return success_count == total_tests


def main():
    """Main function to run all tests"""
    try:
        tester = UpstashVectorTester()
        results = tester.run_all_tests()
        success = tester.print_summary()

        # Save results to file
        with open('upstash_test_results.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)

        print("\n💾 Detailed results saved to 'upstash_test_results.json'")

        return 0 if success else 1

    except Exception as e:
        print(f"\n❌ Test suite failed to initialize: {e}")
        print("\nTroubleshooting:")
        print("1. Check your .env file has correct UPSTASH_VECTOR_REST_URL and UPSTASH_VECTOR_REST_TOKEN")
        print("2. Verify your Upstash Vector database is accessible")
        print("3. Ensure the database is configured with 1024 dimensions and mixbread-large model")
        return 1


if __name__ == "__main__":
    exit(main())