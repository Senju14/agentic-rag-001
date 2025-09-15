import os
from typing import Dict, List
from fastmcp import FastMCP
from dotenv import load_dotenv
from app.search import retrieve_and_rerank
from app.function_calling.tool_registry import send_mail_tool

load_dotenv()
mcp = FastMCP("MCP Server Private")

# --- Tool: search in database ---
@mcp.tool(annotations={"title": "Search in Database"})
def search_in_database(query: str, top_k: int = 5) -> List[Dict]:
    """
    Search for text in the database (hybrid + rerank).
    """
    try:
        semantic_hits, keyword_hits, candidates = retrieve_and_rerank(query, top_k=top_k)
        return candidates
    except Exception as e:
        return [{"error": str(e)}]

# --- Tool: send mail ---
@mcp.tool(annotations={"title": "Send Mail"})
def send_mail(to_email: str, subject: str, body: str) -> Dict:
    """
    Send an email using configured SMTP credentials.
    """
    try:
        result = send_mail_tool(to_email, subject, body)
        return {"result": result}
    except Exception as e:
        return {"error": str(e)}

# python -m app.mcp.mcp_server_private

if __name__ == "__main__":
    mcp.run(transport="http", port=9002)
