# LEARN RAG

A modular Retrieval-Augmented Generation (RAG) pipeline for synthetic legal document search and Q&A chatbot, leveraging semantic chunking, SBERT embeddings, Pinecone vector DB, and FastAPI.

## Technologies Used

- Python 3.10+
- FastAPI
- Pydantic
- Embedding Model (Qwen/Qwen3-Embedding-0.6B)
- Cross-encoder (ms-marco-MiniLM-L6-v2)
- Pinecone
- Groq GPT-OSS 20B

## Requirements

- Python 3.10 or higher
- Pinecone account & API key
- All dependencies in `requirements.txt`

## Installation Instructions

1. Clone the repository:
   ```bash
   git clone https://github.com/Senju14/agentic-rag-001.git
   cd agentic-rag-001
   ```
2. Install dependencies:
   ```bash
   python -m venv venv
   .\venv\Scripts\activate
   pip install uv
   uv pip install -r requirements.txt
   ```
3. Configure environment variables in `.env` (API keys, DB credentials, Pinecone).
4. Run the FastAPI app:
   ```bash
   python app/main.py
   ```

## Project Structure

```
	agentic-rag-001/
│
├── app/
│   ├── main.py                 		# FastAPI entrypoint (e.g., starts the API server)
│   ├── schema.py               		# Pydantic models
│   ├── chunking.py             		# Semantic chunking
│   ├── embeddings.py           		# Embedding generation (e.g., Model Qwen)
│   ├── pineconedb.py           		# Pinecone DB         		
│   ├── search.py               		# Semantic search and rerank model
│   ├── chat_history.py         		# Chat history management
│   ├── file_loader.py          		# File loading from data
│   └── function_calling/
│       └── tool_registry.py    		# Agent tool registry
│
├── data/
│   ├── architecture_firm.pdf
│   ├── consulting_firm.docx
│   └── law_firm_intro.txt
│
├── utils/
│   ├── clear_all_data.py       		# clear all data from Pinecone 
│   └── test.pdf
│
├── rag_test_questions.txt
├── README.md
├── requirements.txt
```

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License

MIT License
