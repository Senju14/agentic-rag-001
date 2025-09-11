# LEARN RAG

A modular Retrieval-Augmented Generation (RAG) pipeline for legal document search and Q&A, leveraging semantic chunking, SBERT embeddings, Pinecone vector DB, PostgreSQL, and FastAPI.

## Technologies Used

- Python 3.10+
- FastAPI
- Pydantic
- SBERT (Qwen/Qwen3-Embedding-0.6B)
- Cross-encoder (ms-marco-MiniLM-L6-v2)
- Pinecone
- PostgreSQL
- Groq GPT-OSS 20B

## Requirements

- Python 3.10 or higher
- Pinecone account & API key
- PostgreSQL database
- All dependencies in `requirements.txt`

## Installation Instructions

1. Clone the repository:
	```bash
	git clone https://github.com/Senju14/agentic-rag-001.git
	cd agentic-rag-001
	```
2. Install dependencies:
	```bash
	pip install -r requirements.txt
	```
3. Configure environment variables in `.env` (API keys, DB credentials, Pinecone).
4. Run the FastAPI app:
	```bash
	python app/main.py
	```

## Project Structure

```text
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
│   ├── postprocess.py     # Cleanup output
│   ├── mcp.py             # Simulated MCP
│   └── chat_history.py    # Store history (with Groq GPT-OSS 20B)
│
│── utils/
│   └── clear_all_data.py  # Delete all data on Pinecone and PostgreSQL
│
│── .env                   # Config: API keys, DB, Pinecone
│── requirements.txt       # Python dependencies
│── README.md              # Setup & run instructions
│── rag_test_questions.txt # Demo test questions for RAG
```

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License

MIT License
