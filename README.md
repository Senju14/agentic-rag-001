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
│   ├── main.py                 		# FastAPI entrypoint (starts API server)
│   ├── schema.py               		# Pydantic models
│   ├── chunking.py             		# Semantic chunking
│   ├── embeddings.py           		# Embedding generation
│   ├── pineconedb.py           		# Pinecone DB
│   ├── search.py               		# Semantic search and rerank
│   ├── chat_history.py         		# Chat history management
│   ├── file_loader.py          		# Load files from data/
│   ├── function_calling/
│   │   └── tool_registry.py    		# Agent tool registry
│   ├── mcp/
│   │   ├── mcp_client.py       		# MCP client
│   │   ├── mcp_server_private.py		# Private MCP server
│   │   └── mcp_server_public.py		# Public MCP server
│   └── multi_ai_agents/
│       ├── private_agent.py    		# Private agent
│       ├── public_agent.py     		# Public agent
│       └── supervisor_agent.py 		# Supervisor agent
│
├── data/
│   ├── Company_ GreenFields BioTech.docx
│   ├── Company_ QuantumNext Systems.docx
│   ├── Company_ TechWave Innovations.docx
│   ├── GreenGrow Innovations_ Company History.docx
│   └── GreenGrow's EcoHarvest System_ A Revolution in Farming.pdf
│
├── utils/
│   ├── clear_all_data.py       		# Clear Pinecone data 
│   ├── test.pdf
│   └── archive/
│       ├── architecture_firm.pdf
│       ├── consulting_firm.docx
│       └── law_firm_intro.txt
│
├── rag_test_questions.txt
├── README.md
├── requirements.txt
```

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License

MIT License
