import os
from typing import List, Dict
from pydantic import BaseModel
from groq import Groq
from dotenv import load_dotenv
from schema import ChatResponse, ChatbotRequest

# -------------------------
# Load config
load_dotenv()
GROQ_MODEL = os.getenv("GROQ_MODEL")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

groq_client = Groq(api_key=GROQ_API_KEY)

# -------------------------
# In-memory chat store
chat_store: Dict[str, List[Dict[str, str]]] = {}

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

    reply = response.choices[0].message.content
    add_message(session_id, "assistant", reply)
    return reply

# -------------------------
# if __name__ == "__main__":
#     sid = "demo-123"

#     print("User: What is deep learning?")
#     print("Bot:", generate_reply(sid, "What is deep learning?"))

#     print("\nUser: Explain it simply like I'm 10 years old.")
#     print("Bot:", generate_reply(sid, "Explain it simply like I'm 10 years old."))

#     print("\nHistory:", get_history(sid))
