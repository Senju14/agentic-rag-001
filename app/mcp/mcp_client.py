# app/mcp/mcp_client.py
import asyncio
import os
import json
import uuid
from fastmcp import Client
from fastmcp.client.transports import StreamableHttpTransport
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

PUBLIC_URL = os.environ.get("MCP_PUBLIC_URL")
PRIVATE_URL = os.environ.get("MCP_PRIVATE_URL")
GROQ_MODEL = os.getenv("GROQ_MODEL")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
groq_client = Groq(api_key=GROQ_API_KEY)

# ===============================
# In-memory session storage
# ===============================

SESSIONS: dict[str, list[dict]] = {}  
# {session_id: [{"role": "user/assistant", "content": "..."}]}


def get_session_id(session_id: str) -> str:
    if session_id and isinstance(session_id, str):
        return session_id
    return uuid.uuid4().hex


def get_history(session_id: str) -> list[dict]:
    return SESSIONS.get(session_id, [])


def add_message(session_id: str, role: str, content: str):
    SESSIONS.setdefault(session_id, []).append({"role": role, "content": content})


# ===============================
# Planner
# ===============================
def planner(user_input: str, session_id: str) -> list[dict]:
    history = get_history(session_id)
    history_text = "\n".join(f"{m['role']}: {m['content']}" for m in history)

    system_prompt = f"""
    You are a task planner. Your job is to break down the user request 
    into one or more MCP tool calls.

    Conversation so far:
    {history_text}

    Available tools:
    - Public: search_topic(query, max_results), math_solver(expression), password_generator(length, use_special)
    - Private: search_in_database(query, top_k), send_mail(to_email, subject, body)

    Rules:
    - If the task is simple -> return just one step.
    - If the task is complex -> return multiple steps in order.
    - DO NOT call tools yourself.
    - Return ONLY valid JSON with this format:

    [
      {{
        "server": "public" or "private",
        "tool": "<tool_name>",
        "args": {{ ... }}
      }},
      ...
    ]
    """

    response = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input},
        ],
        temperature=0,
    )

    raw_plan = response.choices[0].message.content
    print("\n[Groq Plan Raw]", raw_plan)

    try:
        plan = json.loads(raw_plan)
        if not isinstance(plan, list):
            raise ValueError("Planner must return a list of steps")
    except Exception as e:
        raise ValueError(f"Failed to parse plan: {e}\nRaw: {raw_plan}")

    add_message(session_id, "user", user_input)
    add_message(session_id, "assistant", f"Plan: {json.dumps(plan, ensure_ascii=False)}")
    return plan


# ===============================
# Executor
# ===============================
async def executor(plan: list[dict]) -> list[dict]:
    results = []
    for step in plan:
        url = PUBLIC_URL if step["server"] == "public" else PRIVATE_URL
        transport = StreamableHttpTransport(url=url)
        client = Client(transport)
        async with client:
            result = await client.call_tool(step["tool"], step["args"])
            results.append({
                "step": step,
                "result": str(result)
            })
    return results


# ===============================
# Rewriter
# ===============================
def rewrite_output(user_input: str, plan: list[dict], exec_results: list[dict], session_id: str) -> str:
    prompt = f"""
    User asked: {user_input}

    Plan executed:
    {json.dumps(plan, indent=2)}

    Results:
    {json.dumps(exec_results, indent=2)}

    Please combine and rewrite the results into a natural, helpful, 
    final answer for the user.
    """

    response = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )

    answer = response.choices[0].message.content
    add_message(session_id, "assistant", answer)
    return answer


# ===============================
# Full pipeline
# ===============================
async def mcp_reply(user_input: str, session_id: str = "default_session") -> str:
    plan = planner(user_input, session_id)
    exec_results = await executor(plan)
    final_answer = rewrite_output(user_input, plan, exec_results, session_id)
    return final_answer


# ===============================
# TEST
# ===============================
async def main():
    session_id = get_session_id("session_quantum123")



    user_inputs = [
        "Search for the latest news about quantum computing and Solve this math expression: (5^2 + 3*4) / 2.",
        "Generate a password and what is the weather in Ho Chi Minh City",
        "Search in database for GreenGrow Innovations company history and Where it is headquartered?.",
        "Find recent AI news and then send me an email summary to nng.ai.intern01@gmail.com with a summary." 
    ]


    for user_input in user_inputs:
        print("\n==============================")
        print(f"[User] {user_input}")
        reply = await mcp_reply(user_input, session_id=session_id)
        print(f"[AI Reply] {reply}")

    print("\n=== Conversation History ===")
    for m in get_history(session_id):
        print(f"{m['role']}: {m['content']}")


# python -m app.mcp.mcp_server_public
# python -m app.mcp.mcp_server_private
# python -m app.mcp.mcp_client
if __name__ == "__main__":
    asyncio.run(main())
