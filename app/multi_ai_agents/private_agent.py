# app/multi_ai_agents/private_agent.py
import os
import json
import asyncio
from groq import Groq
from fastmcp import Client
from fastmcp.client.transports import StreamableHttpTransport
from dotenv import load_dotenv
load_dotenv()



PRIVATE_URL = os.getenv("MCP_PRIVATE_URL")
GROQ_MODEL_PRIVATE_AGENT = os.getenv("GROQ_MODEL_PRIVATE_AGENT")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
groq_client = Groq(api_key=GROQ_API_KEY)


class PrivateAgent:
    def __init__(self):
        self.tools = []

    async def load_tools(self):
        """Dynamically fetch list of tools from MCP Private server"""
        transport = StreamableHttpTransport(url=PRIVATE_URL)
        client = Client(transport)
        async with client:
            self.tools = await client.list_tools()  # List of Tool objects

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
        You are PrivateAgent. Choose exactly ONE tool for the subtask.
        Available tools:
        {tools_text}

        Return JSON ONLY:
        {{"tool": "<tool_name>", "args": {{...}}}}
        """

        # Call Groq LLM to decide which tool and args
        response = groq_client.chat.completions.create(
            model=GROQ_MODEL_PRIVATE_AGENT,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": task},
            ],
            temperature=0,
        )

        raw = response.choices[0].message.content
        try:
            tool_call = json.loads(raw)
        except Exception as e:
            return f"Error parsing PrivateAgent plan: {e}\nRaw: {raw}"

        # Call MCP Private server
        transport = StreamableHttpTransport(url=PRIVATE_URL)
        client = Client(transport)
        async with client:
            result = await client.call_tool(tool_call["tool"], tool_call["args"])
        return result


# =========================
# Test section
# =========================
async def test():
    agent = PrivateAgent()

    tasks = [
        "Search in database for GreenGrow Innovations company history.",
        # "Send an email to nng.ai.intern01@gmail.com with subject 'Test' and body 'Hello from PrivateAgent!'"
    ]

    for t in tasks:
        print("\n==============================")
        print(f"[Task] {t}")
        result = await agent.handle_task(t)
        print(f"[Result] {result}")


# python -m app.multi_ai_agents.private_agent
if __name__ == "__main__":
    asyncio.run(test())
