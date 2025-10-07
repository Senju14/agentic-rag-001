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
├── src/
│   ├── main.py
│   ├── agents/
│   │   ├── mcp_client.py
│   │   ├── supervisor_agent.py
│   │   ├── private/
│   │   │   ├── mcp_server_private.py
│   │   │   └── private_agent.py
│   │   └── public/
│   │       ├── mcp_server_public.py
│   │       └── public_agent.py
│   ├── core/
│   │   ├── conversation_memory.py
│   │   ├── embedding_generator.py
│   │   ├── retriever.py
│   │   └── text_chunker.py
│   ├── database/
│   │   ├── knowledge_graph_builder.py
│   │   ├── pineconedb.py
│   │   └── schema.py
│   ├── functions_calling/
│   │   └── tool_registry.py
│   ├── prompts/
│   │   └── conversation_prompts.py
│   ├── utils/
│   │   ├── clear_all_data.py
│   │   ├── file_loader.py
│   │   ├── test.pdf
│   │   └── archive_data/
│   │       ├── architecture_firm.pdf
│   │       ├── consulting_firm.docx
│   │       └── law_firm_intro.txt
│   └── data/
│       ├── Company_ GreenFields BioTech.docx
│       ├── Company_ QuantumNext Systems.docx
│       ├── Company_ TechWave Innovations.docx
│       ├── GreenGrow Innovations_ Company History.docx
│       └── GreenGrow's EcoHarvest System_ A Revolution in Farming.pdf
│
├── outputs_sample/
│   ├── response_chatmcp.json
│   ├── response_multi_ai_agents.json
│   └── response_tool_callings.json
│
├── tests/
│   └── example_test_questions.txt
│
├── README.md
└── requirements.txt
```

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License

MIT License
