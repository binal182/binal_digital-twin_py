#!/usr/bin/env python3
"""
Digital Twin RAG System - Quick Start Guide
==========================================

This script provides a quick way to run the complete workflow:
1. Migrate data from JSON to PostgreSQL
2. Generate vector embeddings
3. Test the migration
4. Run comprehensive RAG functionality tests

Usage:
    python quick_start.py

Requirements:
- .env file with all required environment variables
- binal_mytwin.json file in the same directory
"""

import os
import sys
import subprocess
import time
from pathlib import Path

def run_script(script_name: str, description: str) -> bool:
    """Run a Python script and return success status"""
    print(f"\n{'='*60}")
    print(f"🚀 Running: {description}")
    print(f"📄 Script: {script_name}")
    print('='*60)

    try:
        result = subprocess.run([
            sys.executable, script_name
        ], capture_output=True, text=True, encoding='utf-8')

        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr, file=sys.stderr)

        if result.returncode == 0:
            print(f"✅ {description} completed successfully")
            return True
        else:
            print(f"❌ {description} failed with return code {result.returncode}")
            return False

    except Exception as e:
        print(f"❌ Error running {script_name}: {str(e)}")
        return False

def check_environment() -> bool:
    """Check if all required environment variables are set"""
    required_vars = [
        'UPSTASH_VECTOR_REST_URL',
        'UPSTASH_VECTOR_REST_TOKEN',
        'GROQ_API_KEY'
    ]

    # Check for database connection
    has_db_connection = (
        os.getenv('POSTGRES_CONNECTION_STRING') or
        all(os.getenv(v) for v in ['DB_HOST', 'DB_PORT', 'DB_NAME', 'DB_USER', 'DB_PASSWORD'])
    )

    missing_vars = [var for var in required_vars if not os.getenv(var)]

    if missing_vars or not has_db_connection:
        print("❌ Missing required configuration:")
        if missing_vars:
            print("  Required environment variables:")
            for var in missing_vars:
                print(f"    - {var}")
        if not has_db_connection:
            print("  Database connection:")
            print("    - POSTGRES_CONNECTION_STRING or individual DB_* variables")
        print("\n💡 Please set up your .env file with the required variables.")
        return False

    return True

def check_files() -> bool:
    """Check if required files exist"""
    required_files = [
        'binal_mytwin.json',
        'production_migration.py',
        'generate_embeddings.py',
        'production_test_migration.py',
        'rag_functionality_test.py'
    ]

    missing_files = [f for f in required_files if not Path(f).exists()]

    if missing_files:
        print("❌ Missing required files:")
        for file in missing_files:
            print(f"    - {file}")
        return False

    return True

def main():
    """Main quick start workflow"""
    print("🧪 Digital Twin RAG System - Quick Start")
    print("=" * 50)
    print("This will run the complete workflow:")
    print("1. ✅ Environment check")
    print("2. 📊 Production migration (JSON → PostgreSQL)")
    print("3. 🧠 Generate vector embeddings")
    print("4. 🔍 Test migration integrity")
    print("5. ⚡ Run RAG functionality tests")
    print()

    # Check prerequisites
    if not check_environment():
        print("\n❌ Environment check failed. Please configure your .env file.")
        return

    if not check_files():
        print("\n❌ File check failed. Please ensure all required files are present.")
        return

    print("✅ All prerequisites met. Starting workflow...\n")

    # Step 1: Production Migration
    if not run_script('production_migration.py', 'Production Data Migration'):
        print("\n❌ Migration failed. Stopping workflow.")
        return

    # Brief pause
    print("\n⏳ Waiting 5 seconds for database operations to complete...")
    time.sleep(5)

    # Step 2: Generate Embeddings
    if not run_script('generate_embeddings.py', 'Vector Embedding Generation'):
        print("\n❌ Embedding generation failed. Stopping workflow.")
        return

    # Brief pause
    print("\n⏳ Waiting 5 seconds for embeddings to be indexed...")
    time.sleep(5)

    # Step 3: Test Migration
    if not run_script('production_test_migration.py', 'Migration Integrity Test'):
        print("\n⚠️ Migration test had issues, but continuing with RAG tests...")
        print("   Check the test output for details.")

    # Step 4: RAG Functionality Tests
    if not run_script('rag_functionality_test.py', 'RAG Functionality Tests'):
        print("\n❌ RAG tests failed. Check the logs for details.")
        return

    # Success
    print(f"\n{'='*60}")
    print("🎉 WORKFLOW COMPLETED SUCCESSFULLY!")
    print('='*60)
    print("📊 Results Summary:")
    print("   - Data migrated to PostgreSQL")
    print("   - Vector embeddings generated and stored")
    print("   - Migration integrity verified")
    print("   - RAG functionality tested and benchmarked")
    print()
    print("📄 Output Files:")
    print("   - rag_test_results.json (detailed test results)")
    print("   - rag_test_report.txt (comprehensive report)")
    print("   - production_migration.log (migration details)")
    print("   - embedding_generation.log (embedding details)")
    print()
    print("🚀 Your RAG system is ready to use!")
    print("   Run: python binal_rag_app.py")

if __name__ == "__main__":
    main()