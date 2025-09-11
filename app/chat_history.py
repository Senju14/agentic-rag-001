import os
from typing import List, Dict
from pydantic import BaseModel
from groq import Groq
from dotenv import load_dotenv
from schema import ChatResponse, ChatbotRequest
from function_calling.tool_registry import custom_functions, tool_registry
from search import retrieve_and_rerank
import json
import re
 
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

def split_questions(text: str) -> list:
    """
    Split input into sub-questions based on '?', keep context if any.
    """
    # Split by '?' and remove extra spaces
    parts = [q.strip() for q in re.split(r'[?？]', text) if q.strip()]
    # Add '?' back to questions (except the last one if not a question)
    questions = [q + '?' for q in parts[:-1]]
    # If the last one is not a question, keep it
    if parts:
        if text.strip()[-1] not in ['?', '？']:
            questions.append(parts[-1])
    return questions

# -------------------------
def reply(session_id: str, user_input: str, tools: list, tool_registry, llm_func, max_tokens: int = 512):
    trace = []
    questions = split_questions(user_input)
    answers = []
    for idx, q in enumerate(questions):
        add_message(session_id, "user", q)
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
                # Explicitly prompt LLM with tool result for a natural answer
                tool_prompt = (
                    f"User asked: {q}\n"
                    f"Tool result: {result}\n"
                    "Please answer the user's question naturally and conversationally, using the tool result above."
                )
                followup = groq_client.chat.completions.create(
                    model=GROQ_MODEL,
                    messages=[{"role": "system", "content": tool_prompt}],
                    max_tokens=max_tokens,
                    temperature=0.5
                )
                answer = followup.choices[0].message.content.strip()
                add_message(session_id, "assistant", answer)
                answers.append(answer)
                break  # Only return the first tool_call answer if multiple
        else:
            answer = msg.content
            trace.append({
                "thought": f"LLM answered directly for: {q}",
                "action": None,
                "action_input": None,
                "observation": answer
            })
            add_message(session_id, "assistant", answer)
            answers.append(answer)
    if len(answers) == 1:
        return answers[0], trace
    return answers, trace



# -------------------------
# if __name__ == "__main__":
#     sid = "demo-123"

#     print("User: What is deep learning?")
#     print("Bot:", generate_reply(sid, "What is deep learning?"))

#     print("\nUser: Explain it simply like I'm 10 years old.")
#     print("Bot:", generate_reply(sid, "Explain it simply like I'm 10 years old."))

#     print("\nHistory:", get_history(sid))
