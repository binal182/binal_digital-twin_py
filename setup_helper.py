#!/usr/bin/env python3
"""
Quick Setup Helper
==================

This script helps you quickly set up your credentials and test the system.
"""

import os
import re
from dotenv import load_dotenv

def validate_groq_key(api_key: str) -> bool:
    """Validate GROQ API key format"""
    return api_key.startswith('gsk_') and len(api_key) > 20

def validate_postgres_connection(conn_str: str) -> bool:
    """Validate PostgreSQL connection string format"""
    pattern = r'postgresql://[^:]+:[^@]+@[^/]+/[^/]+'
    return bool(re.match(pattern, conn_str))

def setup_credentials():
    """Interactive credential setup"""
    print("🔧 Digital Twin RAG System - Quick Setup")
    print("=" * 50)

    # Load current .env
    load_dotenv()
    env_file = '.env'

    # Get GROQ API key
    current_groq = os.getenv('GROQ_API_KEY', '')
    if current_groq and current_groq != 'your_groq_api_key_here':
        print(f"✅ GROQ API Key: Already configured (ends with ...{current_groq[-4:]})")
    else:
        print("\n🔑 Step 1: GROQ API Key")
        print("   Get your API key from: https://console.groq.com/")
        groq_key = input("   Enter your GROQ API key: ").strip()

        if validate_groq_key(groq_key):
            # Update .env file
            with open(env_file, 'r') as f:
                content = f.read()

            content = content.replace(
                'GROQ_API_KEY=your_groq_api_key_here',
                f'GROQ_API_KEY={groq_key}'
            )

            with open(env_file, 'w') as f:
                f.write(content)

            print("✅ GROQ API Key configured successfully")
        else:
            print("❌ Invalid GROQ API key format")
            return False

    # Get database connection
    current_db = os.getenv('POSTGRES_CONNECTION_STRING', '')
    if current_db and current_db != 'postgresql://your_username:your_password@your_host:5432/your_database':
        print(f"✅ Database: Already configured ({current_db.split('@')[1] if '@' in current_db else 'configured'})")
    else:
        print("\n🗄️  Step 2: Database Connection")
        print("   Choose your database option:")
        print("   1. Neon (Cloud PostgreSQL - Recommended)")
        print("   2. Local PostgreSQL")
        print("   3. Other PostgreSQL")

        choice = input("   Enter choice (1-3): ").strip()

        if choice == '1':
            print("\n   🌐 Neon Setup:")
            print("      1. Go to https://console.neon.tech/")
            print("      2. Select your project")
            print("      3. Go to Dashboard > Connection Details")
            print("      4. Copy the connection string")
            conn_str = input("   Paste your Neon connection string: ").strip()

        elif choice == '2':
            print("\n   💻 Local PostgreSQL Setup:")
            print("   Make sure PostgreSQL is installed and running")
            host = input("   Host (default: localhost): ").strip() or 'localhost'
            port = input("   Port (default: 5432): ").strip() or '5432'
            database = input("   Database name (default: digital_twin): ").strip() or 'digital_twin'
            username = input("   Username (default: postgres): ").strip() or 'postgres'
            password = input("   Password: ").strip()

            conn_str = f"postgresql://{username}:{password}@{host}:{port}/{database}"

        else:
            conn_str = input("   Enter your PostgreSQL connection string: ").strip()

        if validate_postgres_connection(conn_str):
            # Update .env file
            with open(env_file, 'r') as f:
                content = f.read()

            content = content.replace(
                'POSTGRES_CONNECTION_STRING=postgresql://your_username:your_password@your_host:5432/your_database',
                f'POSTGRES_CONNECTION_STRING={conn_str}'
            )

            with open(env_file, 'w') as f:
                f.write(content)

            print("✅ Database connection configured successfully")
        else:
            print("❌ Invalid PostgreSQL connection string format")
            print("   Expected format: postgresql://username:password@host:port/database")
            return False

    print("\n🎉 Setup completed successfully!")
    print("\n📋 Next Steps:")
    print("   1. Run system status check: python system_status.py")
    print("   2. Run the complete workflow: python quick_start.py")
    print("   3. Or validate migration: python validate_migration.py")

    return True

def test_configuration():
    """Test the current configuration"""
    print("\n🧪 Testing Configuration...")

    load_dotenv()

    # Test GROQ
    groq_key = os.getenv('GROQ_API_KEY')
    if groq_key and groq_key != 'your_groq_api_key_here':
        print("✅ GROQ API Key: Configured")
    else:
        print("❌ GROQ API Key: Not configured")

    # Test Database
    db_conn = os.getenv('POSTGRES_CONNECTION_STRING')
    if db_conn and db_conn != 'postgresql://your_username:your_password@your_host:5432/your_database':
        print("✅ Database: Configured")
    else:
        print("❌ Database: Not configured")

    # Test Upstash (already configured)
    upstash_url = os.getenv('UPSTASH_VECTOR_REST_URL')
    upstash_token = os.getenv('UPSTASH_VECTOR_REST_TOKEN')
    if upstash_url and upstash_token:
        print("✅ Upstash Vector: Configured")
    else:
        print("❌ Upstash Vector: Not configured")

if __name__ == "__main__":
    try:
        # Test current configuration
        test_configuration()

        # Ask if user wants to update
        response = input("\n❓ Do you want to update your credentials? (y/n): ").strip().lower()

        if response in ['y', 'yes']:
            if setup_credentials():
                print("\n🔄 Reloading configuration...")
                test_configuration()
        else:
            print("\n📋 To update credentials later, run: python setup_helper.py")

    except KeyboardInterrupt:
        print("\n\n👋 Setup cancelled")
    except Exception as e:
        print(f"\n❌ Setup failed: {str(e)}")