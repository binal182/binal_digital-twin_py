#!/usr/bin/env python3
"""
Database Connection Test Script
Tests the connection to your PostgreSQL database and verifies the schema setup
"""

import os
import psycopg2
from dotenv import load_dotenv
import logging

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_database_connection():
    """Test database connection and basic operations"""
    database_url = os.getenv('DATABASE_URL')

    if not database_url:
        print("❌ DATABASE_URL not found in environment variables")
        print("Please add DATABASE_URL to your .env file")
        print("Example: DATABASE_URL='postgresql://username:password@localhost:5432/digitaltwin'")
        return False

    try:
        print("🔌 Testing database connection...")
        conn = psycopg2.connect(database_url)
        cursor = conn.cursor()

        # Test basic connection
        cursor.execute("SELECT version();")
        version = cursor.fetchone()
        print(f"✅ Connected to PostgreSQL: {version[0][:50]}...")

        # Check if our tables exist
        print("\n📊 Checking database schema...")

        tables_to_check = [
            'personal_info', 'experience', 'education', 'skills',
            'projects', 'certifications', 'publications', 'awards',
            'professional_network', 'digital_twin_metadata'
        ]

        existing_tables = []
        for table in tables_to_check:
            cursor.execute("""
            SELECT EXISTS (
                SELECT 1 FROM information_schema.tables
                WHERE table_schema = 'public' AND table_name = %s
            );
            """, (table,))

            if cursor.fetchone()[0]:
                existing_tables.append(table)

        if existing_tables:
            print(f"✅ Found {len(existing_tables)} tables: {', '.join(existing_tables)}")

            # Get record counts
            print("\n📈 Record counts:")
            for table in existing_tables:
                cursor.execute(f"SELECT COUNT(*) FROM {table} WHERE deleted_at IS NULL")
                count = cursor.fetchone()[0]
                print(f"  {table}: {count} records")
        else:
            print("⚠️  No tables found. Run 'python setup_database.py' to create the schema.")

        # Test a sample query if data exists
        cursor.execute("SELECT COUNT(*) FROM personal_info WHERE deleted_at IS NULL")
        if cursor.fetchone()[0] > 0:
            print("\n🔍 Testing sample query...")
            cursor.execute("""
            SELECT pi.full_name, COUNT(e.*) as experience_count, COUNT(s.*) as skills_count
            FROM personal_info pi
            LEFT JOIN experience e ON pi.id = e.person_id AND e.deleted_at IS NULL
            LEFT JOIN skills s ON pi.id = s.person_id AND s.deleted_at IS NULL
            WHERE pi.deleted_at IS NULL
            GROUP BY pi.id, pi.full_name
            """)

            results = cursor.fetchall()
            print("✅ Sample query successful:")
            for row in results:
                print(f"  {row[0]}: {row[1]} experiences, {row[2]} skills")

        # Check extensions
        print("\n🔧 Checking extensions...")
        cursor.execute("SELECT name FROM pg_available_extensions WHERE name IN ('uuid-ossp', 'pg_trgm') AND installed_version IS NOT NULL")
        extensions = cursor.fetchall()
        if extensions:
            print(f"✅ Extensions installed: {', '.join([ext[0] for ext in extensions])}")
        else:
            print("⚠️  Required extensions (uuid-ossp, pg_trgm) not found")

        cursor.close()
        conn.close()

        print("\n🎉 Database connection test completed successfully!")
        return True

    except psycopg2.Error as e:
        print(f"❌ Database error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def main():
    """Main function"""
    print("PostgreSQL Database Connection Test")
    print("=" * 40)

    success = test_database_connection()

    if success:
        print("\n✅ Your database is ready for the digital twin application!")
        print("\nNext steps:")
        print("1. Run 'python setup_database.py' if you haven't already")
        print("2. Start your RAG application with 'python binal_rag_app.py'")
        print("3. Ask questions about your professional profile!")
    else:
        print("\n❌ Database connection failed.")
        print("\nTroubleshooting:")
        print("1. Check your DATABASE_URL in .env file")
        print("2. Ensure PostgreSQL is running")
        print("3. Verify database credentials")
        print("4. Run 'python setup_database.py' to set up the schema")

    return 0 if success else 1

if __name__ == "__main__":
    exit(main())