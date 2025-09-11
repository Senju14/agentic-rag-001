import os
from typing import List, Dict
from pydantic import BaseModel
from groq import Groq
from dotenv import load_dotenv
from schema import ChatResponse, ChatbotRequest
from function_calling.tool_registry import custom_functions, tool_registry
from search import retrieve_and_rerank
import json

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
def add_message(session_id: str, role: str, content: str):
    chat_store.setdefault(session_id, []).append({"role": role, "content": content})

def get_history(session_id: str) -> List[Dict[str, str]]:
    return chat_store.get(session_id, [])

def clear_history(session_id: str):  
    chat_store.pop(session_id, None)

# -------------------------
def generate_reply(session_id: str, user_input: str, max_tokens: int = 512) -> str:
    """Generate chatbot reply using Groq model with history context"""
    add_message(session_id, "user", user_input)

    response = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=get_history(session_id),
        max_tokens=max_tokens,
        temperature=0.5,
    )
    print(response)

    reply = response.choices[0].message.content
    add_message(session_id, "assistant", reply)
    return reply

# -------------------------
def reply(session_id: str, user_input: str, tools: list, tool_registry, llm_func, max_tokens: int = 512):
    trace = []
    # 1. Truy xuất vectordb
    _, _, candidates = retrieve_and_rerank(user_input, top_k=1)
    if candidates and candidates[0]['rerank_score'] > 0.7:
        answer = candidates[0]['text']
        trace.append({
            "thought": "Found answer in database.",
            "action": None,
            "action_input": None,
            "observation": answer
        })
        return answer, trace
    # 2. Nếu không có, gọi LLM function calling
    add_message(session_id, "user", user_input)
    response = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=get_history(session_id),
        tools=custom_functions,
        tool_choice='auto',
        max_tokens=max_tokens,
        temperature=0.5
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
            thought = f"Calling tool: {tool_name} with {arguments}"
            result = tool_registry[tool_name](**arguments)
            trace.append({
                "thought": thought,
                "action": tool_name,
                "action_input": arguments,
                "observation": result
            })
            # Trả về luôn observation cho user (ví dụ: 'Ha Noi: +29°C')
            answer = result
            add_message(session_id, "assistant", answer)
            return answer, trace
        # Nếu có nhiều tool_call, chỉ trả về câu trả lời đầu tiên
    else:
        answer = msg.content
        trace.append({
            "thought": "LLM answered directly.",
            "action": None,
            "action_input": None,
            "observation": answer
        })
        add_message(session_id, "assistant", answer)
        return answer, trace



# -------------------------
# if __name__ == "__main__":
#     sid = "demo-123"

#     print("User: What is deep learning?")
#     print("Bot:", generate_reply(sid, "What is deep learning?"))

#     print("\nUser: Explain it simply like I'm 10 years old.")
#     print("Bot:", generate_reply(sid, "Explain it simply like I'm 10 years old."))

#     print("\nHistory:", get_history(sid))
