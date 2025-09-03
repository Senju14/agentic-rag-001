agentic-rag-001/
│── data/
│   ├─ corporate_law.md
│   ├─ intellectual_property.md
│   ├─ law_firm_intro.txt
│   └─ legal_tips.html
│
│── app/
│   ├── main.py            # FastAPI entry
│   ├── schema.py          # Pydantic models (schema + metadata)
│   ├── chunking.py        # Semantic chunking
│   ├── preprocessing.py   # Clean text
│   ├── embeddings.py      # SBERT 1024 dim (Qwen/Qwen3-Embedding-0.6B) + Cross-encoder (cross-encoder/ms-marco-MiniLM-L6-v2)
│   ├── vectordb.py        # Pinecone handler
│   ├── postgres.py        # Postgres handler
│   ├── search.py          # Full-text Search + Top-k + rerank
│   ├── rag.py             # RAG pipeline (with Groq GPT-OSS 20B)
│   ├── postprocess.py     # Cleanup output
│   ├── mcp.py             # Simulated MCP
│   └── chat_history.py    # Store history 
│
│── app/
│   ├── clear_all_data.py  # Delete all data on Pinecone and PostgreSQL
|
│── .env                   # Config: API keys, DB, Pinecone
│── requirements.txt       # Python dependencies
│── README.md              # Setup & run instructions
│── rag_test_questions.txt # Demo test questions for RAG




