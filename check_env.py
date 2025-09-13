#!/usr/bin/env python3
"""Check environment variables status"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

print("🔍 Environment Variables Status:")
print("=" * 40)

# Check key variables
vars_to_check = [
    'UPSTASH_VECTOR_REST_URL',
    'UPSTASH_VECTOR_REST_TOKEN',
    'GROQ_API_KEY',
    'POSTGRES_CONNECTION_STRING',
    'DB_HOST',
    'DB_USER',
    'DB_PASSWORD'
]

for var in vars_to_check:
    value = os.getenv(var)
    if value:
        # Hide sensitive information
        if 'token' in var.lower() or 'key' in var.lower() or 'password' in var.lower():
            display_value = "***" + value[-4:] if len(value) > 4 else "***"
        else:
            display_value = value
        print(f"✅ {var}: {display_value}")
    else:
        print(f"❌ {var}: NOT SET")

print("\n💡 Next Steps:")
if not os.getenv('GROQ_API_KEY'):
    print("   - Get GROQ API key from https://console.groq.com/")
if not os.getenv('POSTGRES_CONNECTION_STRING') and not all(os.getenv(v) for v in ['DB_HOST', 'DB_USER', 'DB_PASSWORD']):
    print("   - Set up PostgreSQL database connection (Neon or local)")
    print("   - Either set POSTGRES_CONNECTION_STRING or individual DB_* variables")