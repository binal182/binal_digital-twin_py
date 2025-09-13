# Binal's Digital Twin RAG Application

An AI-powered conversational interface that represents Binal Shah's professional profile using cutting-edge RAG (Retrieval-Augmented Generation) technology.

## 🚀 Quick Start

The fastest way to get started:

```bash
# 1. Set up your environment variables in .env
# 2. Check system status
python system_status.py

# 3. Run the complete workflow
python quick_start.py
```

This will automatically:
- ✅ Check all system components
- 🔄 Migrate your JSON data to PostgreSQL
- 🧠 Generate vector embeddings
- 🔍 Test data integrity
- ⚡ Run comprehensive RAG functionality tests

## 🛠️ Technology Stack

- **Vector Database**: Upstash Vector (built-in embeddings)
- **AI Inference**: Groq (`llama-3.1-8b-instant` model)
- **Relational Database**: PostgreSQL with pgvector extension
- **Language**: Python 3.8+
- **Data Source**: `binal_mytwin.json` (comprehensive professional profile)

## 📋 Prerequisites

1. **Python 3.8+** installed on your system
2. **PostgreSQL** database with pgvector extension
3. **Upstash Vector Database** account and credentials
4. **Groq API** key for AI inference
5. The `binal_mytwin.json` file in the project directory

## 🔧 Setup Instructions

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Database Setup

1. **Install PostgreSQL** and create a database
2. **Enable pgvector extension**:
   ```sql
   CREATE EXTENSION IF NOT EXISTS vector;
   CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
   CREATE EXTENSION IF NOT EXISTS "pg_trgm";
   ```
3. **Run the schema**:
   ```bash
   psql -d your_database -f comprehensive_schema.sql
   ```

### 3. Environment Configuration

Your `.env` file should contain:

```env
# Database Configuration (choose one option)
# Option 1: Connection String (recommended for production)
POSTGRES_CONNECTION_STRING="postgresql://username:password@host:port/database"

# Option 2: Individual Parameters
DB_HOST=localhost
DB_PORT=5432
DB_NAME=digital_twin
DB_USER=postgres
DB_PASSWORD=your_password_here

# Upstash Vector Database Configuration
UPSTASH_VECTOR_REST_TOKEN="your_token_here"
UPSTASH_VECTOR_REST_URL="your_upstash_url_here"

# Groq API Configuration
GROQ_API_KEY="your_groq_api_key_here"
```

### 4. Data Migration

Migrate your JSON data to PostgreSQL with vector embeddings:

```bash
python migrate_data.py
```

This script will:
- Read data from `binal_mytwin.json`
- Insert all profile information into PostgreSQL tables
- Create content chunks for vector search
- Generate embeddings using Upstash Vector
- Store everything in the database

## 🚀 Production Migration Script

For a robust, production-ready migration with comprehensive error handling:

```bash
python production_migration.py
```

**Features:**
- ✅ Comprehensive data validation
- ✅ Advanced content chunking for RAG optimization
- ✅ Batch processing with retry logic
- ✅ 1024-dimensional embeddings using Upstash Vector
- ✅ Detailed logging and progress tracking
- ✅ Transaction safety with rollback support
- ✅ Connection pooling and resource management

### 5. Generate Vector Embeddings

After migration, generate vector embeddings for content chunks:

```bash
python generate_embeddings.py
```

This script will:
- Process all content chunks in the database
- Generate vector embeddings using Upstash Vector
- Store embeddings for semantic search capabilities

### 6. Verify Migration

Test that the migration was successful:

```bash
python test_migration.py
```

This will verify that all data was properly migrated and provide a summary of records.

## 🧪 Production Validation

For comprehensive validation of the production migration:

```bash
python production_test_migration.py
```

**Validation Checks:**
- ✅ Personal information integrity
- ✅ Experience data completeness
- ✅ Skills categorization and counts
- ✅ Projects data validation
- ✅ Education records verification
- ✅ Content chunks quality and metadata
- ✅ Vector embeddings in Upstash
- ✅ JSON content storage
- 📊 Detailed validation report generation

## 🔍 RAG Functionality Testing

Test vector search and RAG functionality with comprehensive benchmarks:

```bash
python rag_functionality_test.py
```

**Testing Features:**
- ✅ **Search Functionality Testing** with 10 professional queries
- ✅ **Retrieval Quality Assessment** with relevance scoring
- ✅ **Metadata Filtering Tests** by category, importance, etc.
- ✅ **Performance Benchmarking** for latency and throughput
- ✅ **Concurrent Operations Testing** with multiple users
- ✅ **Embedding Generation Performance** measurement
- 📊 Comprehensive test reports and metrics

### 7. Data File

Ensure `binal_mytwin.json` is in the same directory as the Python script.

## 🚀 Running the Application

```bash
python binal_rag_app.py
```

### First-Time Setup
- The application will automatically detect if the database is empty
- If needed, it will offer to reset and reload data for optimal performance
- Choose 'y' if you want fresh data or 'N' to use existing embeddings

## 💬 How to Use

1. **Automatic Setup**: The application handles all embedding and database setup
2. **Start Chatting**: Begin asking questions about Binal's professional background
3. **Natural Queries**: Ask in natural language - the AI understands context

### Example Questions

- "Tell me about your AI and data engineering experience"
- "What technologies have you worked with at AusBiz?"
- "Describe your digital twin project"
- "What are your career goals?"
- "What programming languages do you know?"
- "Tell me about your capstone project"
- "Explain your experience with RAG pipelines"
- "What's your background in computer vision?"

## 🧠 How It Works

1. **Data Processing**: Reads `binal_mytwin.json` and extracts pre-structured content chunks
2. **Vector Embedding**: Upstash automatically generates high-quality embeddings
3. **Query Processing**: User questions are embedded and matched against the knowledge base
4. **Response Generation**: Groq's `llama-3.1-8b-instant` generates natural, contextual responses

## 📊 Data Structure

The application uses these key components from your profile:

- **Content Chunks**: 8 optimized sections covering all professional aspects
- **Experience**: AI/Data internship at AusBiz and customer service roles
- **Technical Skills**: Programming languages, AI/ML tools, frameworks
- **Projects**: Digital Twin platform and Smart Attendance System
- **Education**: Information Technology degree from Victoria University
- **Career Goals**: AI Data Engineer/Architect aspirations

## 🔍 Performance Optimizations

### Vector Search
- **High Relevance**: Semantic similarity matching (scores typically 0.6-0.8+)
- **Top-K Retrieval**: Gets 3 most relevant content pieces per query
- **Smart Fallbacks**: Handles edge cases with graceful degradation

### AI Response Generation
- **Fixed Model**: `llama-3.1-8b-instant` for consistent, fast responses
- **Optimized Prompts**: First-person responses maintaining Binal's voice
- **Error Handling**: Robust fallback mechanisms for reliability

### Database Management
- **Reset Capability**: `reset_db.py` for clean data reloads
- **Metadata Integrity**: Proper content preservation and retrieval
- **Efficient Storage**: 8 content chunks vs. previous 151 (optimized)

## 🎯 Use Cases

- **Recruiter Interactions**: Instant, detailed responses about experience and skills
- **Portfolio Showcase**: Interactive complement to traditional resumes
- **Interview Preparation**: Practice explaining your professional background
- **Networking Events**: Quick professional introductions and background sharing
- **Career Development**: Reflect on and articulate your professional journey

## 📈 Technical Specifications

- **Embedding Model**: Upstash's built-in high-performance embeddings
- **Inference Speed**: Ultra-fast with Groq's optimized infrastructure
- **Data Quality**: 92% completeness score across all profile sections
- **Response Accuracy**: First-person responses based on actual professional data
- **Scalability**: Cloud-native vector database and AI inference

## 🛠️ Utilities

### System Status Checker
```bash
python system_status.py
```
Comprehensive diagnostic tool that checks:
- ✅ File system integrity
- ✅ Environment variable configuration
- ✅ Database connectivity (PostgreSQL)
- ✅ Upstash Vector connection
- ✅ Groq API accessibility
- 📊 Data statistics and counts

### Database Reset
```bash
python reset_db.py
```
Use this to clear the vector database and start fresh if needed.

### PostgreSQL Database Setup
```bash
python setup_database.py
```
Sets up the comprehensive PostgreSQL schema and populates it with your professional data.

### Upstash Vector Database Testing
```bash
python test_upstash_simple.py
```
Comprehensive test suite for validating Upstash Vector database connection and functionality.

#### What It Tests:
- **Connection Test**: Verifies database connectivity and configuration
- **Embedding Test**: Validates 1024-dimension embedding generation using mixbread-large
- **Vector Operations**: Tests storage, retrieval, and search functionality
- **Performance**: Measures query speed and batch operation efficiency

#### Expected Results:
- ✅ Database connection successful (1024 dimensions, COSINE similarity)
- ✅ Embedding generation working (1024-dimension vectors)
- ✅ Vector storage and retrieval operational
- ✅ Search functionality returning relevant results
- ✅ Performance metrics within acceptable ranges

#### Success Criteria:
- All 4 test suites pass
- Database dimensions confirmed as 1024
- Embedding generation produces correct vector dimensions
- CRUD operations function properly
- Query performance is adequate for production use

### Requirements
- `upstash-vector`: Vector database operations
- `groq`: AI inference
- `python-dotenv`: Environment configuration
- `requests`: HTTP utilities
- `psycopg2-binary`: PostgreSQL database connectivity

## 🗄️ PostgreSQL Database Schema

This project now includes a comprehensive PostgreSQL database schema for your professional digital twin:

### Features
- **23+ Tables**: Complete coverage of professional data
- **JSONB Support**: Flexible storage for complex data structures
- **Full-Text Search**: Advanced search capabilities across all content
- **GIN Indexes**: Optimized for JSONB and text search operations
- **Audit Trails**: Created/updated/deleted timestamps on all records
- **Soft Delete**: Maintains data integrity while hiding deleted records
- **Data Validation**: Constraints and checks for data quality

### Database Tables
- `personal_info`: Basic profile information
- `experience`: Work experience with achievements and technologies
- `education`: Educational background and qualifications
- `skills`: Technical and soft skills with proficiency levels
- `projects`: Project portfolio with technologies and outcomes
- `certifications`: Professional certifications and credentials
- `publications`: Academic and professional publications
- `awards`: Recognition and achievements
- `professional_network`: Professional connections and relationships
- `content_chunks`: RAG-optimized content for AI interactions
- `search_keywords`: Keywords for enhanced searchability
- `industry_focus`: Industry expertise and focus areas
- `career_trajectory`: Career goals and progression
- `digital_twin_metadata`: Data quality and completeness tracking

### Setup Instructions
1. **Install PostgreSQL**: Ensure you have PostgreSQL running (local or cloud)
2. **Update Environment**: Add `DATABASE_URL` to your `.env` file
3. **Run Setup Script**: Execute `python setup_database.py`
4. **Verify Installation**: Check table creation and data population

### Environment Variables
```env
# Existing variables...
DATABASE_URL="postgresql://username:password@localhost:5432/digitaltwin"
# or for Neon: "postgresql://username:password@ep-xxx-xxx.us-east-1.aws.neon.tech/neondb"
```

### Sample Queries
The database supports complex queries including:
- Full-text search across experience and projects
- JSONB queries for flexible data access
- Semantic similarity search with vector embeddings
- Multi-table joins for comprehensive profile views
- Analytics and reporting queries

See `sample_data_and_queries.sql` for detailed examples.

## 🚨 Troubleshooting

### Common Issues

1. **"No content available" in search results**
   - Run `python reset_db.py` to clear corrupted data
   - Restart the main application to reload with correct metadata

2. **API Key Errors**
   - Verify `GROQ_API_KEY` in `.env` file
   - Check Upstash credentials are properly set

3. **Empty Database**
   - Ensure `binal_mytwin.json` is in the correct directory
   - Check file permissions and JSON validity

### Performance Tips
- First run may take longer due to initial embedding
- Subsequent runs are instant with persistent data
- Reset database if experiencing retrieval issues

## 🔐 Security & Privacy

- **Environment Variables**: API keys secured in `.env` file
- **Professional Data Only**: No sensitive personal information
- **Public Career Information**: Only professional background and achievements
- **No Data Persistence**: Conversations are not stored

## � Recent Updates

- **Fixed Metadata Issues**: Resolved content retrieval problems
- **Optimized Model**: Switched to `llama-3.1-8b-instant` for better performance
- **Streamlined Interface**: Removed model selection for faster startup
- **Enhanced Error Handling**: Better fallbacks and user feedback
- **Database Reset Tool**: Easy way to refresh data when needed

## 🤝 Support

If you encounter any issues:

1. **Check Environment**: Verify all API keys and credentials
2. **Reset Database**: Use `reset_db.py` if experiencing data issues
3. **File Validation**: Ensure `binal_mytwin.json` is properly formatted
4. **Network**: Verify internet connection for API access
5. **Dependencies**: Confirm all packages are installed with `pip install -r requirements.txt`

---

**Built with ❤️ using cutting-edge AI technology to showcase the future of professional interaction.**

*Last updated: September 11, 2025*