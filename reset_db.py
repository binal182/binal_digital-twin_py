"""
Script to reset the Upstash Vector database
"""

import os
from dotenv import load_dotenv
from upstash_vector import Index

# Load environment variables
load_dotenv()

def reset_database():
    """Reset the Upstash Vector database"""
    try:
        # Initialize index from environment variables
        index = Index.from_env()
        print("✅ Connected to Upstash Vector successfully!")
        
        # Reset the database
        print("🗑️ Clearing existing vectors...")
        index.reset()
        print("✅ Database cleared successfully!")
        
    except Exception as e:
        print(f"❌ Error resetting database: {str(e)}")

if __name__ == "__main__":
    reset_database()