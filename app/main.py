import os
from fastapi import FastAPI, HTTPException
from embeddings import embed_text, embed_chunks
from pineconedb import upsert_vectors, query_vector
from postgres import create_tables, insert_document, insert_chunk, fetch_chunks_by_text
from chunking import semantic_chunk
from chat_history import get_history, clear_history, reply, check_or_create_session_id
from schema import SearchResult, ConversationRequest
from function_calling.tool_registry import tool_registry, custom_functions
import uuid
from file_loader import read_file

# -------------------------
app = FastAPI(title="RAG Demo")

# CREATE TABLES IF NOT EXIST
create_tables()

# -------------------------
DATA_FOLDER = os.path.join(os.path.dirname(__file__), "..", "data")

# -------------------------
@app.post("/ingest-folder")
def ingest_folder():
    """Load all files, chunk, embed, store in Postgres + Pinecone"""
    if not os.path.exists(DATA_FOLDER):
        raise HTTPException(status_code=400, detail="Data folder not found")

    ingested = []
    for fname in os.listdir(DATA_FOLDER):
        path = os.path.join(DATA_FOLDER, fname)
        
        text = read_file(path)

        doc_id = insert_document(fname, fname, os.path.splitext(fname)[1])          # Get file extension (e.g. .txt, .pdf, .docx)
        chunks = semantic_chunk(text)
        chunk_texts = [chunk["chunk_text"] for chunk in chunks]
        embeddings = embed_chunks(chunk_texts)

        vectors = []
        for chunk, embedding in zip(chunks, embeddings):
            # Insert chunk into PostgreSQL
            chunk_id = insert_chunk(doc_id, chunk["chunk_text"], chunk["chunk_index"], {"file_name": fname})
            # Upload to Pinecone
            vectors.append({
                "id": f"{doc_id}_{chunk_id}",
                "embedding": embedding["embedding"],
                "metadata": {
                    "document_id": doc_id,
                    "chunk_id": chunk_id,
                    "chunk_text": chunk["chunk_text"],
                    "file_name": fname,
                    "chunk_index": chunk["chunk_index"],
                    "file_type": os.path.splitext(fname)[1]          # Get file extension (e.g. .txt, .pdf, .docx)
                }
            })
        if vectors:
            upsert_vectors(vectors) 
        ingested.append({"file": fname, "doc_id": doc_id, "chunks": len(chunks)})
    return {"status": "Ingestion completed successfully", "ingested": ingested}

# -------------------------
@app.post("/search")
def search(req: SearchResult):
    """Hybrid search: semantic + full-text"""
    question_emb = embed_text(req.question)
    # 1. Semantic search
    semantic_hits = query_vector(question_emb, top_k=req.top_k * 2)
    # 2. Full-text search
    keyword_hits = fetch_chunks_by_text(req.question, limit=req.top_k * 2)
    # 3. Merge
    results = []
    for hit in semantic_hits:
        results.append({
            "source": "semantic",
            "text": hit.get("metadata", {}).get("chunk_text") or hit.get("text"),
            "raw_score": hit.get("score")
        })
    for hit in keyword_hits:
        results.append({ 
            "source": "keyword",
            "text": hit["chunk_text"],
            "raw_score": hit["rank"]
        })
    return {"query": req.question, "results": results[:req.top_k]}


# -------------------------
@app.post("/chat")
def chat(req: ConversationRequest):
    session_id = check_or_create_session_id(getattr(req, 'session_id', None))
    answer, trace = reply(session_id, req.user_input, custom_functions, tool_registry)
    
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
        "tools": [function["function"]["name"] for function in custom_functions],
        "selected_tool": selected_tool
    }

@app.delete("/chat/{session_id}")
def clear_chat(session_id: str):
    clear_history(session_id)
    return {
        "status": "cleared", 
        "session_id": session_id
    }

# -------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
    