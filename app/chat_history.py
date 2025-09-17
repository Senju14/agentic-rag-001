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
def reply(session_id: str, user_input: str, tool_registry: dict, max_tokens: int = 128):
    trace = []
    answers = []

    if "weather" in user_input.lower():
        tool_name = "weather"
        arguments = {
            "city": user_input.replace("weather", "").strip()
        }

    elif "search" in user_input.lower():
        tool_name = "web_search"
        arguments = {
            "query": user_input
        }

    elif "mail" in user_input.lower() or "email" in user_input.lower():
        tool_name = "send_mail"
        arguments = {
            "to_email": "nng.ai.intern01@gmail.com",
            "subject": "Test Function Calling",
            "body": "Hello from Function Calling",
        }

    elif "translate" in user_input.lower():
        tool_name = "translate"
        arguments = {
            "text": user_input,
            "target_lang": "fr",
        }

    elif "database" in user_input.lower():
        tool_name = "search_db"
        arguments = {
            "query": user_input
        }

    else:
        tool_name, arguments = None, None

    # --- Tool execution or fallback ---
    if tool_name:
        # Use registry dict to call the mapped tool function
        tool_func = tool_registry.get(tool_name)
        result = tool_func(**arguments) if tool_func else f"Tool {tool_name} not implemented."

        trace.append({
            "thought": f"Routing to tool: {tool_name} with {arguments}",
            "action": tool_name,
            "action_input": arguments,
            "observation": result,
        })

        natural_response = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful assistant. Rewrite the tool output into a natural, user-friendly answer."},
                {"role": "user", "content": f"Tool output: {result}"},
            ],
            max_tokens=max_tokens,
            temperature=0.5,
        )
        answer = natural_response.choices[0].message.content.strip()

    else:
        raw_answer = "Sorry, I cannot find a suitable tool for this request."
        trace.append({
            "thought": "No tool matched.",
            "action": None,
            "action_input": None,
            "observation": raw_answer,
        })

        natural_response = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful assistant. Rewrite this fallback answer politely."},
                {"role": "user", "content": raw_answer},
            ],
            max_tokens=max_tokens,
            temperature=0.2,
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