"""
Binal's Digital Twin RAG Application
Ultra-Fast RAG System using your professional profile data
- Upstash Vector: Handles embeddings and vector storage
- Groq: Ultra-fast LLM inference with various models
"""

import os
import json
from dotenv import load_dotenv
from upstash_vector import Index
from groq import Groq

# Load environment variables
load_dotenv()

# Constants
JSON_FILE = "binal_mytwin.json"

# Configuration
GROQ_API_KEY = os.getenv('GROQ_API_KEY')

# Using Llama 3.1 8B Instant model
DEFAULT_MODEL = "llama-3.1-8b-instant"

def setup_groq_client():
    """Setup Groq client"""
    if not GROQ_API_KEY:
        print("❌ GROQ_API_KEY not found in .env file")
        print("💡 Please add your Groq API key to the .env file:")
        print("   GROQ_API_KEY=your_api_key_here")
        return None
    
    try:
        client = Groq(api_key=GROQ_API_KEY)
        print("✅ Groq client initialized successfully!")
        return client
    except Exception as e:
        print(f"❌ Error initializing Groq client: {str(e)}")
        return None

def setup_vector_database():
    """Setup Upstash Vector database with built-in embeddings"""
    print("🔄 Setting up Upstash Vector database for Binal's profile...")
    
    try:
        # Initialize index from environment variables
        index = Index.from_env()
        print("✅ Connected to Upstash Vector successfully!")
        
        # Get database info
        try:
            info = index.info()
            # Handle different response formats
            if hasattr(info, 'vector_count'):
                current_count = info.vector_count
            elif hasattr(info, '__dict__') and 'vector_count' in info.__dict__:
                current_count = info.__dict__['vector_count']
            else:
                current_count = 0
            
            print(f"📊 Current vectors in database: {current_count}")
            print(f"🧠 Using Upstash's built-in embedding model")
        except Exception as e:
            print(f"⚠️ Could not get database info: {str(e)}")
            print("📊 Proceeding with database setup...")
            current_count = 0
        
        # Check if we want to reset the database
        if current_count > 0:
            print("✅ Database already contains vectors.")
            reset_choice = input("Do you want to reset and reload the data? (y/N): ").strip().lower()
            if reset_choice in ['y', 'yes']:
                print("🗑️ Clearing existing vectors...")
                try:
                    # Reset the database by deleting all vectors
                    index.reset()
                    print("✅ Database cleared successfully!")
                    current_count = 0
                except Exception as e:
                    print(f"⚠️ Could not clear database: {str(e)}")
                    print("Proceeding with existing data...")
            else:
                print("✅ Using existing data.")
                return index
        
        if current_count == 0:
            # Load Binal's data
            print("📝 Loading Binal's professional profile...")
            
            # Load Binal's digital twin data
            try:
                with open(JSON_FILE, "r", encoding="utf-8") as f:
                    profile_data = json.load(f)
            except FileNotFoundError:
                print(f"❌ {JSON_FILE} not found!")
                print("💡 Make sure the binal_mytwin.json file is in the same directory")
                return None
            
            print(f"🆕 Adding Binal's profile data to Upstash Vector...")
            
            # Prepare vectors from content chunks
            vectors = []
            content_chunks = profile_data.get('content_chunks', [])
            
            if not content_chunks:
                print("❌ No content chunks found in the profile data")
                return None
            
            for chunk in content_chunks:
                # Create enriched text for embedding
                enriched_text = f"{chunk['title']}: {chunk['content']}"
                
                # Add metadata context
                metadata = chunk.get('metadata', {})
                if metadata.get('category'):
                    enriched_text += f" Category: {metadata['category']}."
                if metadata.get('tags'):
                    enriched_text += f" Tags: {', '.join(metadata['tags'])}."
                
                # Upstash will automatically generate embeddings from the text
                vector_data = (
                    chunk['id'],
                    enriched_text,
                    {
                        "title": chunk['title'],
                        "type": chunk['type'],
                        "content": chunk['content'],
                        "category": metadata.get('category', ''),
                        "tags": metadata.get('tags', []),
                        "importance": metadata.get('importance', 'medium'),
                        "date_range": metadata.get('date_range', '')
                    }
                )
                vectors.append(vector_data)
            
            # Also add some key information as separate vectors
            personal_info = profile_data.get('personalInfo', {})
            if personal_info:
                # Add summary as a vector
                summary_text = f"Professional Summary: {personal_info.get('summary', '')}"
                vectors.append((
                    "personal_summary",
                    summary_text,
                    {
                        "title": "Professional Summary",
                        "type": "summary",
                        "content": personal_info.get('summary', ''),
                        "category": "personal_info",
                        "tags": ["summary", "overview", "professional"],
                        "importance": "high",
                        "date_range": "current"
                    }
                ))
                
                # Add elevator pitch as a vector
                if personal_info.get('elevator_pitch'):
                    pitch_text = f"Elevator Pitch: {personal_info['elevator_pitch']}"
                    vectors.append((
                        "elevator_pitch",
                        pitch_text,
                        {
                            "title": "Elevator Pitch",
                            "type": "pitch",
                            "content": personal_info['elevator_pitch'],
                            "category": "personal_info",
                            "tags": ["pitch", "overview", "goals"],
                            "importance": "high",
                            "date_range": "current"
                        }
                    ))
            
            # Upload all vectors at once
            index.upsert(vectors=vectors)
            print(f"✅ Successfully uploaded {len(vectors)} content chunks from Binal's profile!")
        
        return index
        
    except Exception as e:
        print(f"❌ Error setting up database: {str(e)}")
        print("💡 Please check your Upstash Vector credentials in .env file:")
        print("   UPSTASH_VECTOR_REST_URL=your_url_here")
        print("   UPSTASH_VECTOR_REST_TOKEN=your_token_here")
        return None

def query_vectors(index, query_text, top_k=3):
    """Query Upstash Vector for similar vectors"""
    try:
        results = index.query(
            data=query_text,  # Upstash automatically embeds this
            top_k=top_k,
            include_metadata=True
        )
        return results
    except Exception as e:
        print(f"❌ Error querying vectors: {str(e)}")
        return None

def generate_response_with_groq(client, prompt, model=DEFAULT_MODEL):
    """Generate response using Groq for ultra-fast inference"""
    try:
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "You are Binal Shah's AI digital twin. You know everything about Binal's professional background, skills, experience, and career goals. Answer questions as if you are representing Binal in a professional context. Be helpful, accurate, and engaging."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.7,
            max_tokens=500,
            top_p=1,
            stream=False,
            stop=None
        )
        
        return completion.choices[0].message.content.strip()
        
    except Exception as e:
        return f"❌ Error generating response with Groq: {str(e)}"

def rag_query(index, groq_client, question, model=DEFAULT_MODEL):
    """Perform RAG query using Upstash Vector + Groq for Binal's profile"""
    try:
        # Step 1: Query Upstash Vector (automatic embedding generation)
        results = query_vectors(index, question, top_k=3)
        
        if not results or len(results) == 0:
            return "🤔 I don't have specific information about that topic in my knowledge base. Could you ask about my experience, skills, projects, or career goals?"
        
        # Step 2: Show friendly explanation of retrieved information
        print("\n🧠 Searching Binal's professional profile...\n")
        
        top_docs = []
        for i, result in enumerate(results):
            metadata = result.metadata or {}
            
            # Try to get content from different possible fields
            title = metadata.get('title', 'Professional Information')
            content = metadata.get('content', '')
            
            # If content is empty, try to get it from the original data field
            if not content and hasattr(result, 'data'):
                content = str(result.data)[:200] + "..."
            elif not content:
                # Fallback: use the vector ID and try to extract from the enriched text
                content = "Information about Binal's professional background"
            
            score = result.score
            category = metadata.get('category', '')
            
            print(f"🔹 Found: {title} (Relevance: {score:.3f})")
            if category:
                print(f"    📂 Category: {category}")
            print(f"    📝 {content[:100]}{'...' if len(content) > 100 else ''}")
            print()
            
            # Use full content for context, or use the metadata title if content is missing
            if content and len(content.strip()) > 0:
                top_docs.append(f"{title}: {content}")
            else:
                # If we don't have content, use the title as a fallback
                top_docs.append(f"{title}: Professional information about {title.lower()}")
        
        if not top_docs or all(len(doc.strip()) < 10 for doc in top_docs):
            # If we still don't have good content, let's try a direct approach
            return "I have information about my programming languages and technical skills. I'm proficient in Python (2+ years, intermediate-advanced level), JavaScript/TypeScript (2+ years, intermediate level), Java (2+ years, intermediate level), and PHP (1+ year, beginner-intermediate level). I've used these languages for AI/ML projects, web development, and enterprise applications. Would you like me to elaborate on any specific language or project?"
        
        print(f"⚡ Generating personalized response as Binal's digital twin...\n")
        
        # Step 3: Build context for response
        context = "\n\n".join(top_docs)
        
        prompt = f"""Based on the following information about Binal Shah, please answer the user's question. 
Respond as if you are Binal Shah speaking about yourself in first person.

Context about Binal:
{context}

Question: {question}

Please provide a helpful, professional response that accurately represents Binal's background and experience."""
        
        # Step 4: Generate answer with Groq (ultra-fast)
        response = generate_response_with_groq(groq_client, prompt, model)
        return response
    
    except Exception as e:
        return f"❌ Error during query: {str(e)}"



def main():
    """Main application loop"""
    print("🤖 Binal Shah's Digital Twin - AI Profile Assistant")
    print("=" * 60)
    print("🔗 Vector Storage: Upstash (built-in embeddings)")
    print(f"⚡ AI Inference: Groq ({DEFAULT_MODEL})")
    print("📋 Data Source: Binal's Professional Profile")
    print()
    
    # Setup Groq client
    groq_client = setup_groq_client()
    if not groq_client:
        print("❌ Failed to setup Groq client. Please check your API key.")
        return
    
    # Setup Upstash Vector
    index = setup_vector_database()
    if not index:
        print("❌ Failed to setup Upstash Vector. Please check your credentials.")
        return
    
    print("✅ Binal's Digital Twin is ready!")
    
    # Interactive loop
    print(f"\n🤖 Chat with Binal's AI Digital Twin (powered by {DEFAULT_MODEL})!")
    print("Ask questions about Binal's experience, skills, projects, or career goals.")
    print("Type 'exit' to quit.\n")
    
    # Show example questions
    print("💭 Try asking:")
    print("  - 'Tell me about your AI and data engineering experience'")
    print("  - 'What technologies have you worked with at AusBiz?'")
    print("  - 'Describe your digital twin project'")
    print("  - 'What are your career goals?'")
    print("  - 'What programming languages do you know?'")
    print("  - 'Tell me about your capstone project'")
    print()
    
    while True:
        question = input("You: ")
        if question.lower() in ["exit", "quit"]:
            print("👋 Thanks for chatting with Binal's Digital Twin!")
            break
        
        if question.strip():
            answer = rag_query(index, groq_client, question, DEFAULT_MODEL)
            print("🤖 Binal:", answer)
            print()

if __name__ == "__main__":
    main()