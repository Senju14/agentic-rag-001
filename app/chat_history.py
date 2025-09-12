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
    
    # 1. Retrieve from database
    candidates = retrieve_and_rerank(user_input, top_k=1)
    if candidates and candidates[0]['rerank_score'] > 0.7:
        answer = candidates[0]['text']
        trace.append({
            "thought": "Found answer in database.",
            "action": "database",
            "action_input": None,
            "observation": answer
        })
        add_message(session_id, "assistant", answer)
        answers.append(answer)

    # 2. Call LLM to answer
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
            answer = result
            add_message(session_id, "assistant", answer)
            answers.append(answer)
    else:
        answer = msg.content
        trace.append({
            "thought": f"LLM answered directly for: {user_input}",
            "action": None,
            "action_input": None,
            "observation": answer
        })
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