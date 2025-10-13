# src/main.py
import os
import uvicorn
from src.core.embedding_generator import embed_chunks
from src.database.pineconedb import upsert_vectors
from src.core.text_chunker import semantic_chunk
from src.core.conversation_memory import get_history_tools_calling, clear_history, reply, check_or_create_session_id
from src.core.retriever import retrieve_and_rerank
from src.utils.file_loader import read_file
from src.database.schema import SearchResult, ConversationRequest
from fastapi import FastAPI, HTTPException
from src.agents.mcp_client import mcp_reply, get_history, get_session_id
from src.agents.supervisor_agent import SupervisorAgent, get_history_agents

from src.database.knowledge_graph_builder import build_graph
import asyncio

from dotenv import load_dotenv
load_dotenv()
 

# -------------------------
app = FastAPI(title="RAG Demo")
DATA_FOLDER = "src/data/"


# -------------------------
@app.get("/")
def read_root():
    return {"message": "FastAPI server is running! Use /docs để test API."}


# -------------------------
@app.post("/ingest-folder")
def ingest_folder():
    """Load all files, chunk, embed, store in Pinecone"""
    if not os.path.exists(DATA_FOLDER):
        raise HTTPException(status_code=400, detail="Data folder not found")

    ingested = []
    for fname in os.listdir(DATA_FOLDER):
        path = os.path.join(DATA_FOLDER, fname)

        text = read_file(path)
        chunks = semantic_chunk(text)
        chunk_texts = [chunk["chunk_text"] for chunk in chunks]
        embeddings = embed_chunks(chunk_texts)
 
        vectors = []
        for chunk, embedding in zip(chunks, embeddings):
            vectors.append({
                "id": f"{fname}_{chunk['chunk_index']}", 
                "embedding": embedding["embedding"],
                "metadata": {
                    "chunk_text": chunk["chunk_text"],
                    "chunk_index": chunk["chunk_index"],
                    "file_name": fname
                }
            })
        if vectors:
            upsert_vectors(vectors)
        ingested.append({"file": fname, "chunks": len(chunks)})
    return {"status": "Ingestion completed successfully", "ingested": ingested}


# -------------------------
@app.post("/build-knowledge-graph")
async def build_knowledge_graph():
    """
    Run the knowledge graph builder using Groq and push data to Neo4j.
    """
    try:
        asyncio.create_task(build_graph())
        return {
            "status": "Graph building started in background",
            "message": "Please check logs or Neo4j Browser for progress.",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# -------------------------
@app.post("/search")
def search(req: SearchResult):
    """Hybrid search: semantic + full-text"""
    semantic_hits = retrieve_and_rerank(
        query=req.question,
        top_k=req.top_k
    ) 

    return {
        "query": req.question,
        "results": semantic_hits
    }
  

# -------------------------
@app.post("/chat-function-calling")
def chat_function_calling(req: ConversationRequest):
    session_id = check_or_create_session_id(getattr(req, 'session_id', None))
    answer, trace = reply(session_id, req.user_input)

    selected_tool = None
    for step in trace:
        if step.get("action"):  
            selected_tool = step["action"]
            break
 
    return {
        "session_id": session_id,
        "reply": answer,
        "trace": trace,
        "history": get_history_tools_calling(session_id),
        "selected_tool": selected_tool
    }


# -------------------------
@app.post("/chat-mcp")
async def chat_mcp(req: ConversationRequest):
    session_id = get_session_id(getattr(req, 'session_id', None))
    reply = await mcp_reply(req.user_input, session_id=session_id)

    return {
        "session_id": session_id,
        "history": get_history(session_id),
        "reply": reply
    }


# -------------------------
sup = SupervisorAgent()
@app.post("/chat-multi-ai")
async def chat_multi_ai(req: ConversationRequest):
    """Multi-Agent Chat endpoint (Supervisor + Public/Private agents)"""

    session_id = check_or_create_session_id(getattr(req, 'session_id', None))
    session_id, answer = await sup.run(req.user_input, session_id=session_id)

    return {
        "session_id": session_id,
        "reply": answer,
        "history": get_history_agents(session_id)
    }

# -------------------------
@app.delete("/chat/{session_id}")
def clear_chat(session_id: str):
    clear_history(session_id)
    return {
        "status": "cleared", 
        "session_id": session_id
    }

 
# -------------------------
if __name__ == "__main__":
    # python -m src.agents.public.mcp_server_public
    # python -m src.agents.private.mcp_server_private
    # python -m src.main
    uvicorn.run("src.main:app", host="127.0.0.1", port=8000, reload=True)
    