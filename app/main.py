import os
import asyncio
from fastapi import FastAPI, HTTPException
from embeddings import embed_chunks
from pineconedb import upsert_vectors
from chunking import semantic_chunk
from chat_history import get_history, clear_history, reply, check_or_create_session_id
from schema import SearchResult, ConversationRequest
from function_calling.tool_registry import tool_registry, custom_functions
from search import retrieve_and_rerank
import uvicorn
from file_loader import read_file
from fastmcp import Client
from fastmcp.client.transports import StreamableHttpTransport
from dotenv import load_dotenv
load_dotenv()



# # --- MCP Setup ---
# MCP_PUBLIC_URL = os.environ.get("MCP_PUBLIC_URL")    
# MCP_PRIVATE_URL = os.environ.get("MCP_PRIVATE_URL")  
# public_client = Client(StreamableHttpTransport(url=MCP_PUBLIC_URL))
# private_client = Client(StreamableHttpTransport(url=MCP_PRIVATE_URL))
 
# async def call_mcp_tool(client: Client, tool_name: str, params: dict):
#     async with client:
#         return await client.call_tool(tool_name, params)

# async def list_mcp_tools(client: Client):
#     async with client:
#         return await client.list_tools()


# -------------------------
app = FastAPI(title="RAG Demo")
DATA_FOLDER = os.path.join(os.path.dirname(__file__), "..", "data")


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
    """Hybrid search: semantic + full-text + rrf + rerank"""
    semantic_hits = retrieve_and_rerank(
        query=req.question,
        top_k=req.top_k
    )

    return {
        "query": req.question,
        "results": semantic_hits
    }

 
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
# @app.post("/chat_mcp")
# async def chat_mcp(req: ConversationRequest):
#     """
#     Chat endpoint chỉ dùng MCP tools:
#     - Xác định tool dựa trên input
#     - Gọi MCP tool
#     - Feed output về LLM để rewrite reply
#     - Trả về kết quả final
#     """
#     session_id = check_or_create_session_id(getattr(req, 'session_id', None))
#     user_input_lower = req.user_input.lower()
#     final_reply = ""
#     selected_tool = None
#     mcp_result = None

#     # --- 1. List MCP tools ---
#     public_tools = await list_mcp_tools(public_client)
#     private_tools = await list_mcp_tools(private_client)
#     mcp_tools_available = [t.name for t in public_tools + private_tools]

#     # --- 2. Mapping from keywords to tool ---
#     tool_keywords = {
#         "solve": "math_solver",
#         "password": "password_generator",
#         "search topic": "search_topic",
#         "search in database": "search_in_database",
#         "send mail": "send_mail"
#     }

#     # --- 3. Select tool ---
#     for keyword, tool_name in tool_keywords.items():
#         if keyword in user_input_lower and tool_name in mcp_tools_available:
#             selected_tool = tool_name
#             break

#     # --- 4. Call MCP tool if available ---
#     if selected_tool:
#         client_to_use = public_client if selected_tool in [t.name for t in public_tools] else private_client

#         # --- Prepare params ---
#         params = {}
#         import re
#         if selected_tool == "math_solver":
#             match = re.search(r"solve (.+)", user_input_lower)
#             if match:
#                 params["expression"] = match.group(1)
#         elif selected_tool == "password_generator":
#             params = {"length": 12, "use_special": True}
#         elif selected_tool in ["search_topic", "search_in_database"]:
#             params = {"query": req.user_input, "top_k": 5}
#         elif selected_tool == "send_mail":
#             to_match = re.search(r"to ([\w.@]+)", user_input_lower)
#             subject_match = re.search(r"subject (.+?) with", user_input_lower)
#             body_match = re.search(r"body (.+)", user_input_lower)
#             if to_match: params["to_email"] = to_match.group(1)
#             if subject_match: params["subject"] = subject_match.group(1)
#             if body_match: params["body"] = body_match.group(1)

#         # --- Call MCP tool ---
#         mcp_result = await call_mcp_tool(client_to_use, selected_tool, params)

#         # --- Rewrite by LLM ---
#         combined_prompt = (
#             f"MCP tool output: {mcp_result}\n"
#             f"Rewrite a final answer combining the tool output nicely."
#         )
#         final_reply, _ = reply(session_id, combined_prompt, custom_functions, tool_registry)
#     else:
#         final_reply = "No MCP tool matched your request."

#     # --- 5. Return response ---
#     return {
#         "session_id": session_id,
#         "reply": final_reply,
#         "mcp_tools_available": mcp_tools_available,
#         "selected_tool": selected_tool,
#         "mcp_result": mcp_result,
#         "history": get_history(session_id)
#     }


# -------------------------
if __name__ == "__main__":
    # python -m app.mcp.mcp_server_private
    # python -m app.mcp.mcp_server_public
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
    