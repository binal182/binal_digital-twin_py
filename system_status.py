#!/usr/bin/env python3
"""
System Status Checker
=====================

Quick diagnostic script to check the health of your Digital Twin RAG system.
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, List

def check_file_exists(filename: str) -> bool:
    """Check if a file exists"""
    return Path(filename).exists()

def check_env_var(var_name: str) -> bool:
    """Check if environment variable is set"""
    return bool(os.getenv(var_name))

def check_database_connection() -> bool:
    """Check database connectivity"""
    try:
        import psycopg2
        from psycopg2 import sql

        # Try connection string first
        conn_str = os.getenv('POSTGRES_CONNECTION_STRING')
        if conn_str:
            conn = psycopg2.connect(conn_str)
        else:
            # Try individual parameters
            conn = psycopg2.connect(
                host=os.getenv('DB_HOST'),
                port=os.getenv('DB_PORT', 5432),
                database=os.getenv('DB_NAME'),
                user=os.getenv('DB_USER'),
                password=os.getenv('DB_PASSWORD')
            )

        conn.close()
        return True
    except Exception:
        return False

def check_upstash_connection() -> bool:
    """Check Upstash Vector connectivity"""
    try:
        import upstash_vector

        url = os.getenv('UPSTASH_VECTOR_REST_URL')
        token = os.getenv('UPSTASH_VECTOR_REST_TOKEN')

        if not url or not token:
            return False

        # Try to create index (this will fail if credentials are wrong)
        index = upstash_vector.Index(url=url, token=token)
        # Try a simple operation
        index.query(vector=[0.0] * 1024, top_k=1)
        return True
    except Exception:
        return False

def check_groq_connection() -> bool:
    """Check Groq API connectivity"""
    try:
        import groq

        api_key = os.getenv('GROQ_API_KEY')
        if not api_key:
            return False

        client = groq.Groq(api_key=api_key)
        # Try a simple completion
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=10
        )
        return bool(response.choices)
    except Exception:
        return False

def get_vector_count() -> int:
    """Get number of vectors in database"""
    try:
        import upstash_vector

        url = os.getenv('UPSTASH_VECTOR_REST_URL')
        token = os.getenv('UPSTASH_VECTOR_REST_TOKEN')

        index = upstash_vector.Index(url=url, token=token)
        # This is approximate - Upstash doesn't provide direct count
        result = index.query(vector=[0.0] * 1024, top_k=1)
        return len(result) if hasattr(result, '__len__') else 0
    except Exception:
        return 0

def get_database_stats() -> Dict:
    """Get database statistics"""
    stats = {"tables": 0, "records": 0}
    try:
        import psycopg2

        conn_str = os.getenv('POSTGRES_CONNECTION_STRING')
        if conn_str:
            conn = psycopg2.connect(conn_str)
        else:
            conn = psycopg2.connect(
                host=os.getenv('DB_HOST'),
                port=os.getenv('DB_PORT', 5432),
                database=os.getenv('DB_NAME'),
                user=os.getenv('DB_USER'),
                password=os.getenv('DB_PASSWORD')
            )

        cursor = conn.cursor()

        # Count tables
        cursor.execute("""
            SELECT COUNT(*) FROM information_schema.tables
            WHERE table_schema = 'public'
        """)
        stats["tables"] = cursor.fetchone()[0]

        # Count total records across all tables
        cursor.execute("""
            SELECT SUM(n_tup_ins - n_tup_del) as total_records
            FROM pg_stat_user_tables
        """)
        result = cursor.fetchone()
        stats["records"] = result[0] if result[0] else 0

        conn.close()
    except Exception:
        pass

    return stats

def main():
    """Main status check"""
    print("🔍 Digital Twin RAG System - Status Check")
    print("=" * 50)

    checks = []
    issues = []

    # File checks
    print("\n📁 File System:")
    required_files = [
        'binal_mytwin.json',
        '.env',
        'binal_rag_app.py',
        'production_migration.py',
        'generate_embeddings.py'
    ]

    for file in required_files:
        exists = check_file_exists(file)
        status = "✅" if exists else "❌"
        print(f"  {status} {file}")
        checks.append(("File: " + file, exists))
        if not exists:
            issues.append(f"Missing file: {file}")

    # Environment checks
    print("\n⚙️ Environment Variables:")
    env_vars = [
        'UPSTASH_VECTOR_REST_URL',
        'UPSTASH_VECTOR_REST_TOKEN',
        'GROQ_API_KEY'
    ]

    db_vars = [
        'POSTGRES_CONNECTION_STRING',
        'DB_HOST', 'DB_PORT', 'DB_NAME', 'DB_USER', 'DB_PASSWORD'
    ]

    for var in env_vars:
        set_var = check_env_var(var)
        status = "✅" if set_var else "❌"
        print(f"  {status} {var}")
        checks.append(("Env: " + var, set_var))
        if not set_var:
            issues.append(f"Missing environment variable: {var}")

    # Check database connection method
    has_db_conn = check_env_var('POSTGRES_CONNECTION_STRING') or \
                  all(check_env_var(v) for v in db_vars[1:])
    status = "✅" if has_db_conn else "❌"
    print(f"  {status} Database Connection")
    checks.append(("Database Config", has_db_conn))
    if not has_db_conn:
        issues.append("No database connection configured")

    # Service connectivity checks
    print("\n🔗 Service Connectivity:")

    # Database
    db_ok = check_database_connection()
    status = "✅" if db_ok else "❌"
    print(f"  {status} PostgreSQL Database")
    checks.append(("PostgreSQL", db_ok))
    if not db_ok:
        issues.append("Cannot connect to PostgreSQL database")

    # Upstash
    upstash_ok = check_upstash_connection()
    status = "✅" if upstash_ok else "❌"
    print(f"  {status} Upstash Vector")
    checks.append(("Upstash Vector", upstash_ok))
    if not upstash_ok:
        issues.append("Cannot connect to Upstash Vector")

    # Groq
    groq_ok = check_groq_connection()
    status = "✅" if groq_ok else "❌"
    print(f"  {status} Groq API")
    checks.append(("Groq API", groq_ok))
    if not groq_ok:
        issues.append("Cannot connect to Groq API")

    # Data status
    print("\n📊 Data Status:")
    if db_ok:
        db_stats = get_database_stats()
        print(f"  📋 Tables: {db_stats['tables']}")
        print(f"  📝 Records: {db_stats['records']}")

    if upstash_ok:
        vector_count = get_vector_count()
        print(f"  🧠 Vectors: {vector_count}")

    # Summary
    print("\n" + "=" * 50)
    total_checks = len(checks)
    passed_checks = sum(1 for _, passed in checks if passed)

    if issues:
        print("❌ Issues Found:")
        for issue in issues:
            print(f"   - {issue}")
        print()
    else:
        print("✅ All basic checks passed!")

    print(f"📊 Status: {passed_checks}/{total_checks} checks passed")

    if passed_checks == total_checks:
        print("\n🚀 System is ready! You can:")
        print("   - Run: python quick_start.py")
        print("   - Or run individual components")
    else:
        print("\n💡 Next Steps:")
        print("   1. Fix the issues listed above")
        print("   2. Run: python quick_start.py")
        print("   3. Check logs for detailed error information")

if __name__ == "__main__":
    main()