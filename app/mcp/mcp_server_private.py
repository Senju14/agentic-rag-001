# app/mcp/mcp_server_private.py
import os
from typing import Dict, List
from fastmcp import FastMCP
from app.search import retrieve_and_rerank
from app.function_calling.tool_registry import send_mail_tool
from dotenv import load_dotenv
load_dotenv()


mcp = FastMCP("MCP Server Private")

# --- Tool: search in database ---
@mcp.tool(annotations={"title": "Search in Database"})
def search_in_database(query: str, top_k: int = 5) -> List[Dict]:
    """
    Use this tool when the user asks about companies, documents, 
    (e.g. GreenGrow Innovations, GreenFields BioTech, QuantumNext Systems, etc.)
    or information that may exist inside the internal database (docx, pdf). 
    It performs semantic search on the internal company dataset and reranks with a cross-encoder.
    """
    try:
        semantic_hits = retrieve_and_rerank(query, top_k=top_k)
        return semantic_hits
    except Exception as e:
        return [{"error": str(e)}]
  
# --- Tool: send mail ---
@mcp.tool(annotations={"title": "Send Mail"})
def send_mail(to_email: str, subject: str, body: str) -> Dict:
    """
    Send an email to nng.ai.intern01@gmail.com using configured SMTP credentials.
    """
    try:
        result = send_mail_tool(to_email, subject, body)
        return {"result": result}
    except Exception as e:
        return {"error": str(e)}

# python -m app.mcp.mcp_server_private

if __name__ == "__main__":
    mcp.run(transport="http", port=9002)
