# app/chat_history.py
import os
import uuid
import json
from typing import List, Dict
from dotenv import load_dotenv
from groq import Groq
from app.function_calling.tool_registry import tool_registry, custom_functions

# -------------------------
# Load config
load_dotenv()
GROQ_MODEL = os.getenv("GROQ_MODEL")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
groq_client = Groq(api_key=GROQ_API_KEY)

# -------------------------
# In-memory chat store
chat_store: Dict[str, List[Dict[str, str]]] = {}

def get_history(session_id: str) -> List[Dict[str, str]]:
    return chat_store.get(session_id, [])

def clear_history(session_id: str):
    chat_store.pop(session_id, None)

def add_message(session_id: str, role: str, content: str):
    chat_store.setdefault(session_id, []).append({"role": role, "content": content})

def check_or_create_session_id(session_id: str = None) -> str:
    if session_id and isinstance(session_id, str):
        return session_id
    return uuid.uuid4().hex

# -------------------------
# Prompts 
PLANNER_PROMPT = """You are a task planner. 
Given a user task, break it down into a JSON list of steps. 
Each step must include: step number, action (tool name), and input.

Available tools: {tool_registry}
Tools description: {custom_functions}

Task: {input}

Return strictly in JSON with this format:
{{
  "steps": [
    {{"step": 1, "action": "tool_name", "input": "..." }},
    {{"step": 2, "action": "tool_name", "input": "..." }}
  ]
}}
"""

EXECUTOR_PROMPT = """You are a task executor.
You must return reasoning strictly as plain text.
Do NOT call or auto-execute any tool.
Respond ONLY in this format:

Current step: {current_step}

Respond in the format:
Thought: reasoning about this step
Action: the tool to call
Action Input: the input to provide
"""

# -------------------------
def planner(user_input: str, session_id: str) -> Dict:
    """Generate JSON execution plan based on user input and conversation history."""
    tools_list = ", ".join(tool_registry.keys())
    funcs_desc = json.dumps(custom_functions, indent=2, ensure_ascii=False)
    history = get_history(session_id)
    history_text = "\n".join(f"{m['role']}: {m['content']}" for m in history)

    prompt = PLANNER_PROMPT.format(
        input=user_input,
        tool_registry=tools_list,
        custom_functions=funcs_desc
    )

    # Implicitly contextual query - follow-up queries
    if history_text:
        prompt = f"Conversation so far:\n{history_text}\n\nNow user asks: {user_input}\n\n{prompt}"

    resp = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    raw = resp.choices[0].message.content.strip()
 
    try:
        plan = json.loads(raw)
    except Exception:
        raw_json = raw[raw.find("{"): raw.rfind("}") + 1]
        plan = json.loads(raw_json)

    add_message(session_id, "user", user_input)
    add_message(session_id, "assistant", f"Plan: {json.dumps(plan, ensure_ascii=False)}")
    return plan

# -------------------------
def executor(plan: Dict, session_id: str) -> Dict[int, str]:
    """Execute each step in the plan using LLM to reason and select tool."""
    results = {}
    tools_list = ", ".join(tool_registry.keys())
    funcs_desc = json.dumps(custom_functions, indent=2, ensure_ascii=False)

    for step in plan.get("steps", []):
        step_no = step["step"]

        reasoning_prompt = EXECUTOR_PROMPT.format(
            tool_registry=tools_list,
            custom_functions=funcs_desc,
            plan=json.dumps(plan, indent=2, ensure_ascii=False),
            current_step=json.dumps(step, ensure_ascii=False),
            previous_results=json.dumps(results, indent=2, ensure_ascii=False)
        )

        resp = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": reasoning_prompt}],
            temperature=0
        )
        reasoning = resp.choices[0].message.content.strip()
        print(f"\n[Executor Reasoning for Step {step_no}]\n{reasoning}")

        action, action_input = None, ""
        for line in reasoning.splitlines():
            if line.startswith("Action:"):
                action = line.replace("Action:", "").strip()
            elif line.startswith("Action Input:"):
                action_input = line.replace("Action Input:", "").strip()

        if action and action in tool_registry:
            try:
                tool_fn = tool_registry[action]
                output = tool_fn(action_input)
            except Exception as e:
                output = f"Error running {action}: {e}"
        else:
            output = f"Unknown or missing tool: {action}"

        results[step_no] = output
        print(f"[Step {step_no}] {action}({action_input}) -> {output}")

    add_message(session_id, "assistant", f"Results: {json.dumps(results, ensure_ascii=False)}")
    return results

# -------------------------
def reply(session_id: str, user_input: str):
    """Orchestrates planning + execution + final response with final rewrite by LLM."""
    plan = planner(user_input, session_id)
    results = executor(plan, session_id)

    combined_results = "\n".join(
        f"Step {step['step']} ({step['action']}): {results.get(step['step'], 'No result')}"
        for step in plan.get("steps", [])
    )

    rewrite_prompt = f"""
    You are a helpful assistant. Summarize the following multi-step results
    into a clear, concise, natural final answer for the user.  

    Rules:
    - Cover ALL parts of the user question.
    - If the action is "retrieve" or "search", KEEP the original output as-is (do not shorten).
    - If it's a math expression, provide the simplified numeric answer.
    - If it's about facts (e.g. origin), answer briefly and clearly.
    - If it's weather, just state the city with its condition/temperature.
    - The final answer should read like a natural paragraph, not a list of steps.

    User question: {user_input}

    Step results:
    {combined_results}

    Final answer:
    """

    rewrite_resp = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": rewrite_prompt}],
        temperature=0.3,
    )
    final_answer = rewrite_resp.choices[0].message.content.strip()

    trace = [
        {
            "step": step["step"],
            "action": step["action"],
            "input": step.get("input", ""),
            "output": results.get(step["step"], None)
        }
        for step in plan.get("steps", [])
    ]
    
    add_message(session_id, "assistant", f"Final Answer: {final_answer}")
    return final_answer, trace


# -------------------------
# Main test
if __name__ == "__main__":
    session_id = check_or_create_session_id("session_demo123")
    queries = [
        # "Solve this math expression: (5^2 + 3*4)/15.",
        "When was GreenGrow Innovations founded?",
        "Where it is headquartered? and what is the weather there",
        "Do you know Banh Mi, where it come from and what is the weather there and solve this math expression: 5 + (5 * 2) / 10.",
        "What is the previous question that I asked you, what is the weather in Singapore and solve this math expression: (5^2 + 3*4)/15.",
    ]
 
    for query in queries:
        print("=" * 80)
        print(f"User Input: {query}")

        # plan = planner(query, session_id)
        # execute = executor(plan, session_id)
        # print(f"Plan: {plan}")
        # print(f"Executor: {execute}")

        answer, trace = reply(session_id, query)
        print("\n--- Trace ---")
        print(json.dumps(trace, indent=2, ensure_ascii=False))
        print("\n--- Final Answer ---")
        print(answer)
        print("=" * 80)

    # print("\n=== Full Conversation Context ===")
    # for m in get_history(session_id):
    #     print(f"{m['role']}: {m['content']}")
