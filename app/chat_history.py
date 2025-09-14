import os
from typing import List, Dict
from groq import Groq
from dotenv import load_dotenv
from function_calling.tool_registry import tool_registry
from search import retrieve_and_rerank
import json
import re
import uuid
  
# -------------------------
# Load config
load_dotenv()
GROQ_MODEL = os.getenv("GROQ_MODEL")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
groq_client = Groq(api_key=GROQ_API_KEY)

# -------------------------
# In-memory chat store
chat_store: Dict[str, List[Dict[str, str]]] = {}

# "session_xyz": [
#     {"role": "user", "content": "What's the weather today?"},
# ]
 

# -------------------------

def get_history(session_id: str) -> List[Dict[str, str]]:
    return chat_store.get(session_id, [])

def clear_history(session_id: str):  
    chat_store.pop(session_id, None)

def add_message(session_id: str, role: str, content: str):
    chat_store.setdefault(session_id, []).append({"role": role, "content": content})

def check_or_create_session_id(session_id: str = None) -> str:
    # If not transmitted or transmitted as 'string' (default Swagger), then create new.
    if session_id and isinstance(session_id, str) and session_id != 'string':
        return session_id
    return uuid.uuid4().hex

# -------------------------
def reply(session_id, user_input, tools, tool_registry, max_tokens=512):
    trace = []
    answers = []

    # 1. Try to retrieve from database (RAG)
    semantic_hits, keyword_hits, candidates = retrieve_and_rerank(user_input, top_k=1)
    if candidates and candidates[0]['final_score'] > 0.8:
        raw_answer = candidates[0]['text']
        trace.append({
            "thought": "Found answer in database.",
            "action": "database",
            "action_input": None,
            "observation": raw_answer
        })

        # Rewrite database answer
        natural_response = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful assistant. Rewrite this answer in a natural, user-friendly way."},
                {"role": "user", "content": f"Answer: {raw_answer}"}
            ],
            max_tokens=max_tokens,
            temperature=0.3
        )
        answer = natural_response.choices[0].message.content.strip()

        add_message(session_id, "assistant", answer)
        answers.append(answer)
        return answers, trace  

    # 2. Call LLM to check if a tool is needed
    add_message(session_id, "user", user_input)
    response = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=get_history(session_id),
        tools=tools,
        tool_choice='auto',
        max_tokens=max_tokens,
        temperature=0.2
    )
    msg = response.choices[0].message
    tool_calls = getattr(msg, 'tool_calls', None)

    if tool_calls:
        for call in tool_calls:
            tool_name = call.function.name
            arguments = call.function.arguments

            # Convert string args to dict (e.g. user query "weather in Paris" → {"city":"Paris"})
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except Exception:
                    arguments = {"city": arguments}
            result = tool_registry[tool_name](**arguments)

            trace.append({
                "thought": f"Calling tool: {tool_name} with {arguments}",
                "action": tool_name,
                "action_input": arguments,
                "observation": result
            })

            # Rewrite tool output
            natural_response = groq_client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant. Rewrite the tool output into a natural, user-friendly answer."},
                    {"role": "user", "content": f"Tool output: {result}"}
                ],
                max_tokens=max_tokens,
                temperature=0.5
            )
            answer = natural_response.choices[0].message.content.strip()

            add_message(session_id, "assistant", answer)
            answers.append(answer)
        return answers, trace

    # 3. Fallback: not in DB and no tool
    raw_answer = "Sorry, I don’t have this information in my knowledge base."
    trace.append({
        "thought": "No data in DB and no tool used.",
        "action": None,
        "action_input": None,
        "observation": raw_answer
    })

    # Rewrite fallback answer
    natural_response = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": "You are a helpful assistant. Rewrite this fallback answer politely."},
            {"role": "user", "content": raw_answer}
        ],
        max_tokens=max_tokens,
        temperature=0.2
    )
    answer = natural_response.choices[0].message.content.strip()

    add_message(session_id, "assistant", answer)
    answers.append(answer)
    return answers, trace




# -------------------------
# if __name__ == "__main__":
#     sid = "demo-123"

#     print("User: What is deep learning?")
#     print("Bot:", generate_reply(sid, "What is deep learning?"))

#     print("\nUser: Explain it simply like I'm 10 years old.")
#     print("Bot:", generate_reply(sid, "Explain it simply like I'm 10 years old."))

#     print("\nHistory:", get_history(sid))