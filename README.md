project_demo/
│── data/
│   ├─ ticket_wifi_issue.txt
│   ├─ ticket_slow_computer.txt
│   ├─ faq_account.md
│   ├─ faq_billing.md
│   ├─ troubleshooting_printer.html
│   ├─ knowledge_reset_password.md
│   └─ about_support.html
│
│── db/
│   └── init_postgres.sql
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
│── .env                   # Config: API keys, DB, Pinecone
│── requirements.txt       # Python dependencies
│── README.md              # Setup & run instructions

