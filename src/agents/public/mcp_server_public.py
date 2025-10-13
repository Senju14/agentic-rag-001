# src/agents/public/mcp_server_public.py
import os
from typing import Dict, List
from fastmcp import FastMCP
from tavily import TavilyClient
import sympy
import random
import string
from dotenv import load_dotenv

load_dotenv()


# Load API Key
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY")
tavily = TavilyClient(api_key=TAVILY_API_KEY)
mcp = FastMCP("TavilyExplorer")


# --- Resource: tech trends ---
@mcp.resource("resource://tech/trends")
def tech_trends() -> List[str]:
    return [
        "Artificial Intelligence",
        "Quantum Computing",
        "Blockchain",
        "5G and IoT",
        "Cybersecurity",
    ]


# --- Tool: search topic ---
@mcp.tool(annotations={"title": "Search Topic"})
def search_topic(query: str, max_results: int = 3) -> List[Dict]:
    """
    Use this tool when the user asks about general knowledge, news, current events,
    or real-time information (e.g., today's weather, recent updates).
    It searches the web using the Tavily API and returns results from the internet.
    """
    try:
        resp = tavily.search(query=query, max_results=max_results)
        results = resp.get("results", [])
        return [
            {
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "content": r.get("content", ""),
            }
            for r in results
        ]
    except Exception as e:
        return {"error": str(e)}


# --- Tool: math solver ---
@mcp.tool(annotations={"title": "Math Solver"})
def math_solver(expression: str) -> Dict:
    """
    Solve a math expression safely using sympy.
    Example: "2 + 3 * (7 - 2)"
    """
    try:
        expr = sympy.sympify(expression)
        result = expr.evalf()
        return {"result": str(result)}
    except Exception as e:
        return {"error": str(e)}


# --- Tool: password generator ---
@mcp.tool(annotations={"title": "Password Generator"})
def password_generator(length: int = 12, use_special: bool = True) -> Dict:
    """
    Generate a random strong password.
    """
    try:
        chars = string.ascii_letters + string.digits
        if use_special:
            chars += string.punctuation
        password = "".join(random.choice(chars) for _ in range(length))
        return {"password": password}
    except Exception as e:
        return {"error": str(e)}


# --- Prompt: explore topic ---
@mcp.prompt
def explore_topic_prompt(topic: str) -> str:
    return (
        f"I want to explore the topic '{topic}'.\n"
        f"1. Use the 'Search Topic' tool to gather 3-5 relevant sources.\n"
        f"2. Summarize the key points from each.\n"
        f"3. Provide an overview combining all insights."
    )


# python -m src.agents.public.mcp_server_public

if __name__ == "__main__":
    import sys

    try:
        mcp.run(transport="http", port=9001)
    except KeyboardInterrupt:
        print("\n[Server stopped] MCP Public Agent has been shut down cleanly.")
        sys.exit(0)
