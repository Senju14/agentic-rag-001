import os
from fastapi import FastAPI, HTTPException
from embeddings import embed_text, embed_chunks
from vectordb import upsert_vectors, query_vector
from postgres import create_tables, insert_document, insert_chunk, fetch_chunks_by_text
from chunking import semantic_chunk
from postprocess import postprocess
from chat_history import get_history, clear_history, reply
from schema import Document, Chunk, ChatRequest, ChatResponse, ChatbotRequest, FunctionCallRequest, FunctionCallResponse, ToolDescription
from function_calling.tool_registry import tool_registry, custom_functions
import uuid

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
        with open(path, "r", encoding="utf-8") as file:
            text = file.read()

        doc_id = insert_document(fname, fname, "txt")
        chunks = semantic_chunk(text)
        chunk_texts = [chunk["chunk_text"] for chunk in chunks]
        embeddings = embed_chunks(chunk_texts)

        vectors = []
        for chunk, emb in zip(chunks, embeddings):
            chunk_id = insert_chunk(doc_id, chunk["chunk_text"], chunk["chunk_index"], {"file_name": fname})
            vectors.append({
                "id": f"{doc_id}_{chunk_id}",
                "embedding": emb["embedding"],
                "metadata": {
                    "document_id": doc_id,
                    "chunk_id": chunk_id,
                    "chunk_text": chunk["chunk_text"],
                    "file_name": fname,
                    "chunk_index": chunk["chunk_index"],
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
def summarize_answers_llm(answers: list, user_input: str) -> str:
    """
    Use LLM to summarize raw answers into a single, natural, and coherent response, making sure to link references (like 'there', 'it', etc.) to the correct entities if the questions are related.
    """
    from chat_history import groq_client, GROQ_MODEL
    prompt = (
        "You are an AI assistant. Combine the following answers into a single, natural, and coherent response. "
        "If the questions are related (e.g., using words like 'there', 'it', etc.), make sure to link them to the correct entities and keep the answer smooth and clear.\n"
        f"Original user input: {user_input}\n"
        f"Answers to combine: " + " ".join(f"- {a}" for a in answers)
    )
    response = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "system", "content": prompt}],
        max_tokens=512,
        temperature=0.5
    )
    return response.choices[0].message.content.strip()

# -------------------------
@app.post("/chat")
def chat(req: ChatbotRequest):
    """Chatbot with history, hỗ trợ cả function calling nếu cần"""
    session_id = uuid.uuid4().hex
    answer, trace = reply(session_id, req.user_input, custom_functions, tool_registry, None)
    # If returning multiple answers (list), summarize them using LLM.
    if isinstance(answer, list):
        answer = summarize_answers_llm(answer, req.user_input)
    return {"session_id": session_id, "reply": answer, "trace": trace, "history": get_history(session_id)}


@app.delete("/chat/{session_id}")
def clear_chat(session_id: str):
    clear_history(session_id)
    return {"status": "cleared", "session_id": session_id}

# -------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
    