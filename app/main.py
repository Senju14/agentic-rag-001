# app/main.py
import os
import uvicorn
from app.embeddings import embed_chunks
from app.pineconedb import upsert_vectors
from app.chunking import semantic_chunk
from app.chat_history import get_history, clear_history, reply, check_or_create_session_id
from app.search import retrieve_and_rerank
from app.file_loader import read_file
from app.schema import SearchResult, ConversationRequest
from fastapi import FastAPI, HTTPException
from app.mcp.mcp_client import mcp_reply, get_history, get_session_id
from app.multi_ai_agents.supervisor_agent import SupervisorAgent, get_history_agents
from dotenv import load_dotenv
load_dotenv()

 

# -------------------------
app = FastAPI(title="RAG Demo")
DATA_FOLDER = os.path.join(os.path.dirname(__file__), "..", "data")


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
        "history": get_history(session_id),
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
    # python -m app.mcp.mcp_server_public
    # python -m app.mcp.mcp_server_private
    # python -m app.mcp.mcp_client
    # python -m app.main
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)
    