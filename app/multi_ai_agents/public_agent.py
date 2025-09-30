# app/multi_ai_agents/public_agent.py
import os
import json
import asyncio
from groq import Groq
from fastmcp import Client
from fastmcp.client.transports import StreamableHttpTransport
from dotenv import load_dotenv
load_dotenv()



PUBLIC_URL = os.getenv("MCP_PUBLIC_URL")
GROQ_MODEL_PUBLIC_AGENT = os.getenv("GROQ_MODEL_PUBLIC_AGENT")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
groq_client = Groq(api_key=GROQ_API_KEY)


class PublicAgent:
    def __init__(self):
        self.tools = []

    async def load_tools(self):
        """Dynamically fetch list of tools from MCP Public server"""
        transport = StreamableHttpTransport(url=PUBLIC_URL)
        client = Client(transport)
        async with client:
            self.tools = await client.list_tools()  # List of Tools objects

    async def handle_task(self, task: str):
        if not self.tools:
            await self.load_tools()

        # Build dynamic tool list for LLM prompt
        tool_lines = []
        for t in self.tools:
            name = t.name
            desc = t.description
            schema = json.dumps(t.inputSchema, indent=2, ensure_ascii=False)
            tool_lines.append(f"- {name}{schema}: {desc}")
        tools_text = "\n".join(tool_lines)

        system_prompt = f"""
        You are PublicAgent. Choose exactly ONE tool for the subtask.
        Available tools:
        {tools_text}

        Return JSON ONLY:
        {{"tool": "<tool_name>", "args": {{...}}}}
        """

        response = groq_client.chat.completions.create(
            model=GROQ_MODEL_PUBLIC_AGENT,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": task},
            ],
            temperature=0,
        )

        raw = response.choices[0].message.content
        cleaned = raw.strip().strip("```").replace("json", "").strip()
        try:
            tool_call = json.loads(cleaned)
        except Exception as e:
            return f"Error parsing PublicAgent plan: {e}\nRaw: {raw}"

        # Call MCP Public server
        transport = StreamableHttpTransport(url=PUBLIC_URL)
        client = Client(transport)
        async with client:
            result = await client.call_tool(tool_call["tool"], tool_call["args"])
        return result


# =========================
# Test section
# =========================
async def test():
    agent = PublicAgent()

    tasks = [
        "Search for the latest news about quantum computing.",
        "Solve the math expression: (5^2 + 3*4) / 2.",
        "Generate a password of length 16 with special characters."
    ]

    for t in tasks:
        print("\n==============================")
        print(f"[Task] {t}")
        result = await agent.handle_task(t)
        print(f"[Result] {result}")

# python -m app.multi_ai_agents.public_agent
if __name__ == "__main__":
    asyncio.run(test())
