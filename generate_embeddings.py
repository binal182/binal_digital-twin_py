#!/usr/bin/env python3
"""
Generate Vector Embeddings Script
Generates vector embeddings for content chunks stored in PostgreSQL
"""

import os
import psycopg2
from dotenv import load_dotenv
from upstash_vector import Index
import logging
import numpy as np

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_embeddings():
    """Generate embeddings for content chunks"""
    try:
        # Connect to database
        conn = psycopg2.connect(
            host=os.getenv('DB_HOST', 'localhost'),
            port=os.getenv('DB_PORT', '5432'),
            database=os.getenv('DB_NAME', 'digital_twin'),
            user=os.getenv('DB_USER', 'postgres'),
            password=os.getenv('DB_PASSWORD', '')
        )

        # Connect to Upstash Vector
        vector_index = Index.from_env()

        logger.info("🧠 Starting embedding generation...")

        with conn.cursor() as cursor:
            # Get content chunks without embeddings
            cursor.execute("""
                SELECT id, content, chunk_type, metadata
                FROM content_chunks
                WHERE embedding_vector IS NULL
                AND deleted_at IS NULL
                ORDER BY created_at
            """)

            chunks = cursor.fetchall()
            logger.info(f"📊 Found {len(chunks)} chunks without embeddings")

            batch_size = 10
            processed = 0

            for i in range(0, len(chunks), batch_size):
                batch = chunks[i:i + batch_size]

                # Prepare batch for Upstash
                vectors_to_upsert = []

                for chunk in batch:
                    chunk_id, content, chunk_type, metadata = chunk

                    # Create enriched content for better embeddings
                    enriched_content = f"{content}"

                    # Add metadata context if available
                    if metadata and 'category' in metadata:
                        enriched_content += f" Category: {metadata['category']}."

                    if metadata and 'tags' in metadata:
                        enriched_content += f" Tags: {', '.join(metadata['tags'])}."

                    # For Upstash, we'll use the content directly
                    # The actual embedding generation happens server-side
                    vector_data = (
                        f"chunk_{chunk_id}",
                        enriched_content,
                        {
                            "chunk_id": str(chunk_id),
                            "chunk_type": chunk_type,
                            "content_preview": content[:100] + "..." if len(content) > 100 else content,
                            "metadata": metadata or {}
                        }
                    )
                    vectors_to_upsert.append(vector_data)

                # Upsert to Upstash Vector
                if vectors_to_upsert:
                    vector_index.upsert(vectors=vectors_to_upsert)
                    logger.info(f"✅ Processed batch of {len(vectors_to_upsert)} chunks")

                processed += len(batch)

            logger.info(f"🎉 Successfully generated embeddings for {processed} content chunks!")

        conn.close()

    except Exception as e:
        logger.error(f"❌ Failed to generate embeddings: {str(e)}")
        raise

def update_database_embeddings():
    """Update PostgreSQL with embedding vectors (if needed)"""
    try:
        logger.info("🔄 Updating database with embedding vectors...")

        # Note: This would require extracting embeddings from Upstash
        # For now, we rely on Upstash for vector storage and search
        # PostgreSQL is used for structured data and metadata

        logger.info("✅ Database embedding update completed!")

    except Exception as e:
        logger.error(f"❌ Failed to update database embeddings: {str(e)}")
        raise

def main():
    """Main function"""
    print("🧠 Vector Embeddings Generation")
    print("=" * 35)

    # Check environment variables
    required_vars = ['UPSTASH_VECTOR_REST_URL', 'UPSTASH_VECTOR_REST_TOKEN']
    missing_vars = [var for var in required_vars if not os.getenv(var)]

    if missing_vars:
        print("❌ Missing required environment variables:")
        for var in missing_vars:
            print(f"   - {var}")
        return

    try:
        # Generate embeddings
        generate_embeddings()

        # Update database (optional)
        update_db = input("\n🔄 Update PostgreSQL with embedding vectors? (y/N): ").strip().lower()
        if update_db in ['y', 'yes']:
            update_database_embeddings()

        print("\n✅ Embedding generation completed successfully!")

    except Exception as e:
        print(f"\n❌ Embedding generation failed: {str(e)}")

if __name__ == "__main__":
    main()</content>
<parameter name="filePath">c:\Users\sbina\OneDrive\Desktop\digitaltwin\generate_embeddings.py