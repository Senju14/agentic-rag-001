# app/mcp/mcp_client.py
import asyncio
import os
import json
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


# ================== STEP 1: Planner ==================
def planner(user_input: str) -> list[dict]:
    """
    Use Groq LLM to plan a sequence of tool calls if needed.
    Returns a list of steps (each step = {server, tool, args}).
    """

    system_prompt = """
    You are a task planner. Your job is to break down the user request 
    into one or more MCP tool calls.

    Available tools:
    - Public: search_topic(query, max_results), math_solver(expression), password_generator(length, use_special)
    - Private: search_in_database(query, top_k), send_mail(to_email, subject, body)

    Rules:
    - If the task is simple -> return just one step.
    - If the task is complex -> return multiple steps in order.
    - DO NOT call tools yourself.
    - Return ONLY valid JSON with this format:

    [
      {
        "server": "public" or "private",
        "tool": "<tool_name>",
        "args": { ... }
      },
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

    return plan


# ================== STEP 2: Executor ==================
async def executor(plan: list[dict]) -> list[dict]:
    """
    Execute each step in the plan via MCP and collect results.
    Always convert results to string for JSON serialization.
    """
    results = []
    for step in plan:
        url = PUBLIC_URL if step["server"] == "public" else PRIVATE_URL
        transport = StreamableHttpTransport(url=url)
        client = Client(transport)
        async with client:
            result = await client.call_tool(step["tool"], step["args"])
            # Convert result to safe string
            results.append({
                "step": step,
                "result": str(result)  # <-- fix here
            })
    return results



# ================== STEP 3: Rewriter ==================
def rewrite_output(user_input: str, plan: list[dict], exec_results: list[dict]) -> str:
    """
    Rewrite execution results into a natural helpful response.
    """
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

    return response.choices[0].message.content


# ================== STEP 4: Orchestrator ==================
async def run_mcp_agent(user_input: str) -> str:
    """
    Full pipeline:
    1. Planner (LLM creates plan)
    2. Executor (MCP executes steps)
    3. Rewriter (LLM rewrites final answer)
    """
    plan = planner(user_input)
    exec_results = await executor(plan)
    final_answer = rewrite_output(user_input, plan, exec_results)
    return final_answer


# ================== MAIN TEST ==================
async def main():
    user_inputs = [
        "Search for the latest news about quantum computing.",
        "Solve this math expression: (5^2 + 3*4) / 2",
        "Generate a 20-character password without special characters.",
        "Search in database for GreenGrow Innovations company history.",
        "Find recent AI news and then send me an email with a summary."  # <-- complex multi-step
    ]

    for user_input in user_inputs:
        print("\n==============================")
        print(f"[User] {user_input}")
        reply = await run_mcp_agent(user_input)
        print(f"[AI Reply] {reply}")


# python -m app.mcp.mcp_client
if __name__ == "__main__":
    asyncio.run(main())
