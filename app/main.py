import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from embeddings import embed_text, embed_chunks
from vectordb import upsert_vectors, query_vector
from postgres import create_tables, insert_document, insert_chunk, fetch_chunks_by_text
from chunking import semantic_chunk
from postprocess import postprocess
from chat_history import generate_reply, get_history, clear_history
from schema import Document, Chunk, ChatRequest, ChatResponse, ChatbotRequest

# -------------------------
DATA_FOLDER = os.path.join(os.path.dirname(__file__), "..", "data")

# -------------------------
app = FastAPI(title="RAG Demo")

# CREATE TABLES IF NOT EXIST
create_tables()

# -------------------------
@app.post("/ingest-folder")
def ingest_folder():
    """Load all files, chunk, embed, store in Postgres + Pinecone"""
    if not os.path.exists(DATA_FOLDER):
        raise HTTPException(status_code=400, detail="Data folder not found")

    ingested = []
    for fname in os.listdir(DATA_FOLDER):
        path = os.path.join(DATA_FOLDER, fname)
        if not os.path.isfile(path):
            continue
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()

        doc_id = insert_document(fname, fname, "txt")
        chunks = semantic_chunk(text)
        chunk_texts = [c["chunk_text"] for c in chunks]
        embeddings = embed_chunks(chunk_texts)

        vectors = []
        for c, emb in zip(chunks, embeddings):
            chunk_id = insert_chunk(doc_id, c["chunk_text"], c["chunk_index"], {"file_name": fname})
            vectors.append({
                "id": f"{doc_id}_{chunk_id}",
                "embedding": emb["embedding"],
                "metadata": {
                    "document_id": doc_id,
                    "chunk_id": chunk_id,
                    "chunk_text": c["chunk_text"],
                    "file_name": fname,
                    "chunk_index": c["chunk_index"],
                    "file_type": "txt"
                }
            })

        if vectors:
            upsert_vectors(vectors)

        ingested.append({"file": fname, "doc_id": doc_id, "chunks": len(chunks)})

    return {"status": "ok", "ingested": ingested}

# -------------------------
@app.post("/search")
def search(req: ChatRequest):
    """Simple hybrid search: semantic + full-text"""
    q_emb = embed_text(req.question)

    # 1. Semantic search
    semantic_hits = query_vector(q_emb, top_k=req.top_k * 2)

    # 2. Full-text search
    keyword_hits = fetch_chunks_by_text(req.question, limit=req.top_k * 2)

    # 3. Merge
    results = []
    for h in semantic_hits:
        results.append({
            "source": "semantic",
            "text": h.get("metadata", {}).get("chunk_text") or h.get("text"),
            "raw_score": h.get("score")
        })
    for h in keyword_hits:
        results.append({
            "source": "keyword",
            "text": h["chunk_text"],
            "raw_score": h["rank"]
        })

    return {"query": req.question, "results": results[:req.top_k]}

# -------------------------
# @app.post("/chat")
# def chat(req: ChatbotRequest):
#     """Chatbot with history (Groq model)"""
#     reply = generate_reply(req.session_id, req.user_input)
#     return {"session_id": req.session_id, "reply": reply, "history": get_history(req.session_id)}


@app.post("/chat")
def chat(req: ChatbotRequest):
    """Chatbot with history (Groq model)"""
    reply = generate_reply(req.session_id, req.user_input)
    return {"session_id": req.session_id, "reply": reply, "history": get_history(req.session_id)}


@app.delete("/chat/{session_id}")
def clear_chat(session_id: str):
    clear_history(session_id)
    return {"status": "cleared", "session_id": session_id}

# -------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
    