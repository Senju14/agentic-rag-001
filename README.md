# LEARN RAG 🚀

A modular **Retrieval-Augmented Generation (RAG)** pipeline for intelligent document search and Q&A chatbot, leveraging semantic chunking, advanced embeddings, Pinecone vector database, and multi-agent architecture with FastAPI.

## 📋 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Technologies Used](#technologies-used)
- [Architecture](#architecture)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [API Endpoints](#api-endpoints)
- [Key Components](#key-components)
- [Multi-Agent System](#multi-agent-system)
- [Example Queries](#example-queries)
- [Project Structure](#project-structure)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## 📖 Overview

**LEARN RAG** is an advanced Retrieval-Augmented Generation system designed to intelligently process, store, and retrieve information from documents. The system combines state-of-the-art NLP techniques with a multi-agent architecture to provide accurate, context-aware responses to user queries.

### Purpose

This project demonstrates how to build a production-ready RAG pipeline that can:
- **Index and search** through company documents, legal texts, and other materials
- **Answer questions** using retrieved context and LLM reasoning
- **Execute complex tasks** through function calling (web search, email, translation, calculations)
- **Coordinate multiple AI agents** for sophisticated query handling
- **Build knowledge graphs** from unstructured data

### Use Cases

- Corporate document search and Q&A
- Legal document analysis
- Knowledge base construction
- Multi-step task automation
- Conversational AI with memory

## ✨ Key Features

- 🔍 **Semantic Search**: Advanced embedding-based retrieval using Qwen3-Embedding model
- 🎯 **Cross-Encoder Reranking**: Improved relevance with ms-marco-MiniLM-L6-v2
- 🧠 **Multi-Agent Architecture**: Supervisor + Public/Private agents for complex task coordination
- 💬 **Conversational Memory**: Context-aware conversations with session management
- 🔧 **Function Calling**: Weather, web search, email, translation, calculator, database search
- 🌐 **Knowledge Graph**: Neo4j integration for structured knowledge representation
- 🚀 **FastAPI Backend**: RESTful API with automatic documentation
- 📊 **MCP Protocol**: Model Context Protocol for agent communication

## 🛠️ Technologies Used

### Core Framework
- **Python 3.10+**: Main programming language
- **FastAPI**: Modern web framework for building APIs
- **Pydantic**: Data validation using Python type annotations
- **Uvicorn**: ASGI server for running FastAPI

### AI/ML Components
- **Embedding Model**: Qwen/Qwen3-Embedding-0.6B for semantic embeddings
- **Cross-Encoder**: ms-marco-MiniLM-L6-v2 for reranking
- **LLM**: Groq (llama3-70b-versatile, llama3-8b-chat) for reasoning and generation
- **Sentence Transformers**: For generating embeddings

### Databases & Storage
- **Pinecone**: Vector database for semantic search
- **Neo4j**: Graph database for knowledge graph storage

### Agent Framework
- **FastMCP**: Model Context Protocol for agent communication
- **LangChain**: Framework for LLM applications and agents

### External APIs
- **Tavily API**: Web search functionality
- **SMTP/Gmail**: Email sending capabilities
- **MyMemory API**: Translation services

### Document Processing
- **pdfplumber**: PDF text extraction
- **python-docx**: Word document processing
- **NLTK**: Natural language processing utilities

## 🏗️ Architecture

The system follows a modular architecture with three main layers:

1. **API Layer**: FastAPI endpoints for client interaction
2. **Agent Layer**: Multi-agent system for task coordination
   - **Supervisor Agent**: Routes tasks to appropriate agents
   - **Public Agent**: Handles general tasks (search, weather, calculations)
   - **Private Agent**: Manages sensitive operations (database access, email)
3. **Core Layer**: Embedding, retrieval, chunking, and memory management

```
┌─────────────┐
│   Client    │
└──────┬──────┘
       │
┌──────▼──────────────────────────────────────┐
│          FastAPI Application                 │
│  (/chat, /search, /ingest, /build-graph)    │
└──────┬──────────────────────────────────────┘
       │
┌──────▼──────────────────────────────────────┐
│         Multi-Agent System                   │
│  ┌────────────────────────────────────┐    │
│  │      Supervisor Agent              │    │
│  └────────┬──────────────┬────────────┘    │
│           │              │                  │
│  ┌────────▼────────┐ ┌──▼──────────────┐   │
│  │  Public Agent   │ │  Private Agent  │   │
│  │  (MCP Server)   │ │  (MCP Server)   │   │
│  └─────────────────┘ └─────────────────┘   │
└──────┬──────────────────────────────────────┘
       │
┌──────▼──────────────────────────────────────┐
│            Core Services                     │
│  - Embedding Generator                       │
│  - Retriever & Reranker                      │
│  - Text Chunker                              │
│  - Conversation Memory                       │
└──────┬──────────────────────────────────────┘
       │
┌──────▼──────────────────────────────────────┐
│         Data Storage                         │
│  - Pinecone (Vectors)                        │
│  - Neo4j (Knowledge Graph)                   │
└──────────────────────────────────────────────┘
```

## 📋 Prerequisites

Before you begin, ensure you have the following:

- **Python 3.10 or higher** installed on your system
- **Pinecone account** and API key ([Sign up here](https://www.pinecone.io/))
- **Groq API key** ([Get it here](https://console.groq.com/))
- **Tavily API key** for web search ([Sign up here](https://tavily.com/))
- **Neo4j database** (optional, for knowledge graph features) ([AuraDB free tier](https://neo4j.com/cloud/aura/))
- **Gmail App Password** (optional, for email functionality)

## 🚀 Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/Senju14/agentic-rag-001.git
cd agentic-rag-001
```

### Step 2: Create Virtual Environment

**On Windows:**
```bash
python -m venv venv
.\venv\Scripts\activate
```

**On Linux/MacOS:**
```bash
python -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

Using `uv` (recommended for faster installation):
```bash
pip install uv
uv pip install -r requirements.txt
```

Or using standard pip:
```bash
pip install -r requirements.txt
```

### Step 4: Download NLTK Data

```bash
python -c "import nltk; nltk.download('punkt')"
```

## ⚙️ Configuration

### Step 1: Set Up Environment Variables

Copy the example environment file:
```bash
cp .env.example .env
```

### Step 2: Configure API Keys

Edit the `.env` file and add your API keys:

```bash
# Pinecone Configuration
PINECONE_API_KEY=your_pinecone_api_key_here
PINECONE_INDEX=your_index_name_here
PINECONE_ENV=us-east-1-aws
PINECONE_HOST=your_pinecone_host_here

# Groq/LLM Models
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=llama3-70b-versatile
GROQ_MODEL_PRIVATE_AGENT=llama3-8b-chat
GROQ_MODEL_PUBLIC_AGENT=llama3-70b-chat

# Tavily Search API
TAVILY_API_KEY=your_tavily_api_key_here

# Email/SMTP Configuration (Optional)
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
SENDER_EMAIL=your_email@gmail.com
SENDER_PASSWORD=your_app_password_here

# MCP (Multi-Agent Control Protocol)
MCP_PUBLIC_URL=http://127.0.0.1:9001/mcp
MCP_PRIVATE_URL=http://127.0.0.1:9002/mcp

# Neo4j Database (Optional - for Knowledge Graph)
NEO4J_URI=neo4j+s://xxxxxxxx.databases.neo4j.io
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_password_here
```

**Important Notes:**
- For Gmail, use an [App Password](https://support.google.com/accounts/answer/185833), not your regular password
- Create a Pinecone index with dimension 1024 (matching Qwen3-Embedding model)
- Neo4j configuration is optional unless you use the knowledge graph feature

## 📝 Usage

### Running the Application

The application consists of three components that need to be started separately:

#### 1. Start Public MCP Server (Terminal 1)
```bash
python -m src.agents.public.mcp_server_public
```
This will start on `http://127.0.0.1:9001/mcp`

#### 2. Start Private MCP Server (Terminal 2)
```bash
python -m src.agents.private.mcp_server_private
```
This will start on `http://127.0.0.1:9002/mcp`

#### 3. Start Main FastAPI Application (Terminal 3)
```bash
python -m src.main
```
This will start on `http://127.0.0.1:8000`

### Accessing the API Documentation

Once the application is running, you can access:
- **Interactive API Docs (Swagger)**: http://127.0.0.1:8000/docs
- **Alternative API Docs (ReDoc)**: http://127.0.0.1:8000/redoc

### Quick Start Workflow

1. **Ingest Documents**: Upload and process your documents
   ```bash
   curl -X POST http://127.0.0.1:8000/ingest-folder
   ```

2. **Search Documents**: Perform semantic search
   ```bash
   curl -X POST http://127.0.0.1:8000/search \
     -H "Content-Type: application/json" \
     -d '{"question": "When was GreenGrow Innovations founded?", "top_k": 5}'
   ```

3. **Chat with Function Calling**: Ask questions that trigger tool usage
   ```bash
   curl -X POST http://127.0.0.1:8000/chat-function-calling \
     -H "Content-Type: application/json" \
     -d '{"session_id": "session_123", "message": "What is the weather in Singapore?"}'
   ```

4. **Multi-Agent Chat**: Use the supervisor agent for complex queries
   ```bash
   curl -X POST http://127.0.0.1:8000/chat-multi-ai \
     -H "Content-Type: application/json" \
     -d '{"session_id": "session_456", "message": "Search latest AI news and tell me about GreenGrow Innovations"}'
   ```

## 🔌 API Endpoints

### Document Management

#### `POST /ingest-folder`
Loads all files from `src/data/`, chunks them, generates embeddings, and stores in Pinecone.

**Response:**
```json
{
  "status": "Ingestion completed successfully",
  "ingested": [
    {"file": "document.pdf", "chunks": 42}
  ]
}
```

#### `POST /build-knowledge-graph`
Builds a knowledge graph from documents using Neo4j (runs in background).

**Response:**
```json
{
  "status": "Graph building started in background",
  "message": "Please check logs or Neo4j Browser for progress."
}
```

### Search & Retrieval

#### `POST /search`
Performs hybrid semantic search with reranking.

**Request Body:**
```json
{
  "question": "What is the vision of Greenfield Architects?",
  "top_k": 5
}
```

**Response:**
```json
{
  "query": "What is the vision of Greenfield Architects?",
  "results": [
    {
      "chunk_text": "...",
      "score": 0.89,
      "metadata": {...}
    }
  ]
}
```

### Conversational Endpoints

#### `POST /chat-function-calling`
Chat with automatic function calling (weather, search, email, translate, calculator, database).

**Request Body:**
```json
{
  "session_id": "session_123",
  "message": "What's the weather in Singapore and search for AI news?"
}
```

#### `POST /chat-mcp`
Chat using Model Context Protocol with public/private agents.

**Request Body:**
```json
{
  "session_id": "session_456",
  "message": "Search in database for GreenGrow Innovations and generate a password"
}
```

#### `POST /chat-multi-ai`
Chat using supervisor + multi-agent system for complex task coordination.

**Request Body:**
```json
{
  "session_id": "session_789",
  "message": "Search latest news about AI, solve (5+3*5)/2, and search GreenGrow history"
}
```

#### `DELETE /chat/{session_id}`
Clear conversation history for a specific session.

**Response:**
```json
{
  "status": "cleared",
  "session_id": "session_123"
}
```

## 🔑 Key Components

### 1. Embedding Generator (`src/core/embedding_generator.py`)
Generates semantic embeddings using the **Qwen3-Embedding-0.6B** model. This model converts text into 1024-dimensional vectors that capture semantic meaning, enabling similarity-based search.

**Key Features:**
- Batch processing for efficiency
- Automatic device selection (CPU/GPU)
- Sentence-level embeddings

### 2. Text Chunker (`src/core/text_chunker.py`)
Implements **semantic chunking** to split documents into meaningful segments while preserving context.

**Strategy:**
- Sentence-based splitting using NLTK
- Semantic coherence preservation
- Configurable chunk sizes

### 3. Pinecone Vector Database (`src/database/pineconedb.py`)
Manages vector storage and retrieval using Pinecone's cloud-native vector database.

**Features:**
- Serverless index management
- Efficient similarity search
- Metadata filtering capabilities

### 4. Retriever & Reranker (`src/core/retriever.py`)
Two-stage retrieval process for optimal accuracy:

1. **Initial Retrieval**: Semantic search using embeddings in Pinecone
2. **Reranking**: Cross-encoder (ms-marco-MiniLM-L6-v2) reranks results for relevance

**Benefits:**
- Improved precision
- Context-aware ranking
- Reduced false positives

### 5. Conversation Memory (`src/core/conversation_memory.py`)
Manages session-based conversation history with a planning-execution framework.

**Components:**
- **Planner**: Analyzes queries and creates execution plans
- **Executor**: Executes plans using appropriate tools
- **Memory**: Maintains conversation context

### 6. Function Calling (`src/functions_calling/tool_registry.py`)
Enables the LLM to use external tools:

- **weather**: Get current weather for any city
- **web_search**: Search the web using Tavily API
- **send_mail**: Send emails via SMTP
- **translate**: Translate text between languages
- **search_db**: Search internal document database
- **calculator**: Evaluate mathematical expressions

## 🤖 Multi-Agent System

The system uses a **supervisor-worker pattern** with three types of agents:

### Supervisor Agent (`src/agents/supervisor_agent.py`)
- **Role**: Task orchestration and routing
- **Responsibilities**: 
  - Analyzes user queries
  - Breaks down complex tasks
  - Routes subtasks to appropriate agents
  - Aggregates results

### Public Agent (`src/agents/public/public_agent.py`)
- **Role**: General-purpose operations
- **Available Tools**:
  - Web search (Tavily)
  - Weather lookup
  - Password generation
  - Mathematical calculations

### Private Agent (`src/agents/private/private_agent.py`)
- **Role**: Secure operations requiring authentication
- **Available Tools**:
  - Database search
  - Email sending
  - Translation
  - Sensitive data access

### Agent Communication (MCP)
Agents communicate via **Model Context Protocol (FastMCP)**:
- Standardized tool interface
- Dynamic tool discovery
- Streaming support
- HTTP transport layer

## 💡 Example Queries

### Basic Document Search
```
"When was GreenGrow Innovations founded?"
"What industries does Anderson & Lee Consulting Group serve?"
"What is the vision of Greenfield Architects & Planners?"
```

### Function Calling Examples
```
"What is the weather in Singapore?"
"Search for LangChain vs LangGraph"
"Translate 'How are you today?' to French"
"Send an email to user@example.com with subject 'Test'"
"Calculate (5^2 + 3*4) / 2"
```

### Multi-Agent Complex Queries
```
"Search the latest AI news, solve (5+3*5)/2, and search GreenGrow history"
"Find information about GreenGrow Innovations and what's the weather at their HQ?"
"Tell me about Banh Mi origin and the weather there"
```

### Conversational Context
```
User: "When was GreenGrow Innovations founded?"
AI: "GreenGrow Innovations was founded in..."
User: "Where is it headquartered?"
AI: "Based on our previous discussion..."
User: "What's the weather there?"
AI: "At the headquarters location..."
```

More examples can be found in `tests/example_test_questions.txt`.

## 📁 Project Structure

```
agentic-rag-001/
│
├── src/                                    # Main source code
│   ├── main.py                            # FastAPI application entry point
│   │
│   ├── agents/                            # Multi-agent system
│   │   ├── mcp_client.py                 # MCP client for agent communication
│   │   ├── supervisor_agent.py           # Supervisor agent for task routing
│   │   ├── private/
│   │   │   ├── mcp_server_private.py    # Private MCP server
│   │   │   └── private_agent.py         # Private agent implementation
│   │   └── public/
│   │       ├── mcp_server_public.py     # Public MCP server
│   │       └── public_agent.py          # Public agent implementation
│   │
│   ├── core/                              # Core RAG components
│   │   ├── conversation_memory.py        # Session management & memory
│   │   ├── embedding_generator.py        # Embedding model (Qwen3)
│   │   ├── retriever.py                  # Retrieval & reranking logic
│   │   └── text_chunker.py              # Semantic text chunking
│   │
│   ├── database/                          # Database interactions
│   │   ├── knowledge_graph_builder.py    # Neo4j knowledge graph builder
│   │   ├── pineconedb.py                # Pinecone vector DB operations
│   │   └── schema.py                     # Pydantic models/schemas
│   │
│   ├── functions_calling/                 # Tool/function definitions
│   │   └── tool_registry.py             # Available tools and their schemas
│   │
│   ├── prompts/                           # LLM prompt templates
│   │   └── conversation_prompts.py       # System prompts
│   │
│   ├── utils/                             # Utility functions
│   │   ├── clear_all_data.py            # Data cleanup utilities
│   │   ├── file_loader.py               # Document loaders (PDF, DOCX, TXT)
│   │   └── archive_data/                 # Archived sample documents
│   │
│   └── data/                              # Active document storage
│       ├── Company_ GreenFields BioTech.docx
│       ├── Company_ QuantumNext Systems.docx
│       ├── Company_ TechWave Innovations.docx
│       ├── GreenGrow Innovations_ Company History.docx
│       └── GreenGrow's EcoHarvest System_ A Revolution in Farming.pdf
│
├── outputs_sample/                        # Sample output responses
│   ├── response_chatmcp.json             # MCP chat examples
│   ├── response_multi_ai_agents.json     # Multi-agent examples
│   └── response_tool_callings.json       # Function calling examples
│
├── tests/                                 # Test files
│   └── example_test_questions.txt        # Sample queries for testing
│
├── .env.example                           # Environment variable template
├── .gitignore                            # Git ignore rules
├── README.md                             # This file
├── requirements.txt                       # Python dependencies
└── pyproject.toml                        # Project metadata
```

## 🔧 Troubleshooting

### Common Issues

#### 1. Import Errors
**Problem**: `ModuleNotFoundError` when running the application

**Solution**:
```bash
# Make sure you're in the project root
cd agentic-rag-001

# Run with Python module syntax
python -m src.main
```

#### 2. Pinecone Connection Issues
**Problem**: "Invalid API key" or connection timeout

**Solution**:
- Verify your `PINECONE_API_KEY` in `.env`
- Ensure your Pinecone index exists with dimension=1024
- Check `PINECONE_HOST` matches your index host

#### 3. MCP Server Not Running
**Problem**: "Connection refused" for MCP servers

**Solution**:
- Start both MCP servers before the main app:
  ```bash
  # Terminal 1
  python -m src.agents.public.mcp_server_public
  
  # Terminal 2
  python -m src.agents.private.mcp_server_private
  
  # Terminal 3
  python -m src.main
  ```

#### 4. Embedding Model Download
**Problem**: Model downloading fails or is slow

**Solution**:
- Ensure stable internet connection
- Model downloads automatically on first use
- Stored in `~/.cache/huggingface/`

#### 5. NLTK Data Missing
**Problem**: `LookupError: Resource punkt not found`

**Solution**:
```bash
python -c "import nltk; nltk.download('punkt')"
```

#### 6. GPU/CUDA Issues
**Problem**: "CUDA out of memory" or GPU not detected

**Solution**:
- The system works on CPU (slower but functional)
- For GPU: Install appropriate PyTorch version with CUDA support
- Reduce batch size in embedding generation if needed

### Getting Help

If you encounter issues not listed here:
1. Check the [Issues](https://github.com/Senju14/agentic-rag-001/issues) page
2. Review logs for error messages
3. Ensure all environment variables are correctly set
4. Verify all three components (2 MCP servers + main app) are running

## 🤝 Contributing

We welcome contributions to improve this project! Here's how you can help:

### How to Contribute

1. **Fork the Repository**
   ```bash
   git clone https://github.com/YOUR_USERNAME/agentic-rag-001.git
   cd agentic-rag-001
   ```

2. **Create a Feature Branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Make Your Changes**
   - Write clear, documented code
   - Follow existing code style and patterns
   - Add comments for complex logic
   - Update documentation if needed

4. **Test Your Changes**
   - Ensure all endpoints work correctly
   - Test with various query types
   - Verify no existing functionality is broken

5. **Commit and Push**
   ```bash
   git add .
   git commit -m "Add: Brief description of your changes"
   git push origin feature/your-feature-name
   ```

6. **Open a Pull Request**
   - Provide a clear description of changes
   - Reference any related issues
   - Wait for review and address feedback

### Areas for Contribution

- 🐛 **Bug Fixes**: Report and fix bugs
- 📚 **Documentation**: Improve docs, add examples
- ✨ **Features**: Add new capabilities or tools
- 🧪 **Testing**: Add test coverage
- 🎨 **UI/UX**: Improve API design or add frontend
- 🚀 **Performance**: Optimize retrieval or embedding generation
- 🔧 **Tools**: Add new function calling tools

### Code Guidelines

- Use meaningful variable and function names
- Add docstrings to functions and classes
- Keep functions focused and modular
- Handle errors gracefully
- Follow PEP 8 style guidelines

### Reporting Issues

When reporting bugs, please include:
- Description of the issue
- Steps to reproduce
- Expected vs actual behavior
- Environment details (OS, Python version)
- Error messages/logs

## 📄 License

This project is licensed under the **MIT License**.

### MIT License Summary

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

## 📞 Contact

### Project Maintainer

- **GitHub**: [@Senju14](https://github.com/Senju14)
- **Project Repository**: [agentic-rag-001](https://github.com/Senju14/agentic-rag-001)

### Get in Touch

- 💬 **Issues & Bugs**: [GitHub Issues](https://github.com/Senju14/agentic-rag-001/issues)
- 🌟 **Feature Requests**: [GitHub Discussions](https://github.com/Senju14/agentic-rag-001/discussions)
- 📧 **Email**: For private inquiries, create an issue and we'll respond

### Community

- ⭐ **Star this repo** if you find it useful
- 🔄 **Share** with others who might benefit
- 🐦 **Follow** for updates and improvements

---

<div align="center">

**Built with ❤️ using FastAPI, Groq, Pinecone, and LangChain**

[⬆ Back to Top](#learn-rag-)

</div>
