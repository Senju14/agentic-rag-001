# rag.py
from search import retrieve_and_rerank
from typing import List
import os
import requests

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL")

def build_prompt(question: str, contexts: List[str], tools: str = "Web Search, Weather API", tool_names: str = "web_search, weather") -> str:
    system_prompt = f"""
You are an AI assistant that can answer questions using provided CONTEXTS and by calling external TOOLS.

You have access to the following tools:
{tools}

When answering, always follow this format:

Question: the input question you must answer
Thought: you should always think about what to do next
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: [your answer here]

Guidelines:
- If the answer can be found in CONTEXTS, use it and cite the source.
- If not, use the appropriate TOOL.
- If the question is about the weather, use the Weather API.
- If the question is about current events or general knowledge not in CONTEXTS, use Web Search.
- If you cannot answer, say: "I could not find an exact answer."
- Always be clear, concise, and helpful.
"""

    ctx_text = "\n\n---\n\n".join(contexts)

    prompt = f"""{system_prompt}

CONTEXTS:
{ctx_text}

QUESTION:
{question}

Now write the best possible answer following the above rules.
"""
    return prompt



def call_groq(prompt: str):
    if not GROQ_API_KEY:
        return "DEMO ANSWER (no model):\n\n" + (prompt[:1000] + "...")
    try:
        resp = requests.post(
            "https://api.groq.ai/v1/engines/{}/completions".format(GROQ_MODEL),
            headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
            json={"prompt": prompt, "max_tokens": 512}
        )
        if resp.status_code == 200:
            data = resp.json()
            return data.get("choices", [{}])[0].get("text", str(data))
        else:
            return f"Model error: {resp.status_code} {resp.text}"
    except Exception as e:
        return f"Model call failed: {e}"


def answer_question(question: str, top_k: int = 5):
    hits = retrieve_and_rerank(question, top_k=top_k)
    contexts = [h.get("text") or (h.get("metadata") or {}).get("chunk_text","") for h in hits]
    prompt = build_prompt(question, contexts)
    answer = call_groq(prompt)
    return {"answer": answer, "source_chunks": hits}
