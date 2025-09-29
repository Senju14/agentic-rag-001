# app/multi_ai_agents/supervisor_agent.py
import os
import asyncio
import json
import uuid
from typing import Dict, List
from groq import Groq
from dotenv import load_dotenv
from app.multi_ai_agents.public_agent import PublicAgent
from app.multi_ai_agents.private_agent import PrivateAgent
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
    if session_id and isinstance(session_id, str) and session_id.strip():
        return session_id
    return uuid.uuid4().hex


# -------------------------
class SupervisorAgent:
    def __init__(self):
        self.public_agent = PublicAgent()
        self.private_agent = PrivateAgent()

    def plan(self, user_input: str):
        """Supervisor analyze user request and split subtasks"""
        system_prompt = """
            You are a supervisor_agent. 
            Split the user task into subtasks and assign to:
            - PublicAgent: general knowledge, math, password, open web/news.
            - PrivateAgent: company dataset, internal knowledge, email.

            Return STRICT JSON array. Example:
            [
                {"agent": "public", "task": "Search the latest AI news"},
                {"agent": "private", "task": "Find company headquarters"}
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
        plan = json.loads(raw_plan)
        return plan
        
    def rewrite_answer(self, user_input: str, raw_results: List[Dict[str, str]]) -> str:
        """Rewrite raw agent results into a clear user-friendly answer"""
        system_prompt = """
            You are a helpful supervisor agent.
            Your job: take results from multiple agents and rewrite them 
            into a single clear, concise, user-friendly answer.
            Do not list agents or tasks, just provide a natural final response.
        """

        # Build raw text summary
        summary = "\n".join([f"{r['task']}: {r['result']}" for r in raw_results])

        response = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"User asked: {user_input}\n\nResults:\n{summary}"},
            ],
            temperature=0.3,
        )
        return response.choices[0].message.content.strip()


    async def run(self, user_input: str, session_id: str = None):
        session_id = check_or_create_session_id(session_id)
        add_message(session_id, "user", user_input)
        plan = self.plan(user_input)
        
        print("\n[Supervisor Plan]")
        for i, step in enumerate(plan, 1):
            print(f"  {i}. {step['task']}  →  {step['agent'].title()}Agent")

        results = []

        for step in plan:
            if step["agent"] == "public":
                print(f" → PublicAgent handling: {step['task']}")
                result = await self.public_agent.handle_task(step["task"])
            else:
                print(f" → PrivateAgent handling: {step['task']}")
                result = await self.private_agent.handle_task(step["task"])

            results.append({
                "agent": step["agent"],
                "task": step["task"], 
                "result": result
            })

        detailed_answer = "\n".join([f"- ({r['agent']}) {r['task']}: {r['result']}" for r in results])
        user_answer = self.rewrite_answer(user_input, results)
        add_message(session_id, "assistant", user_answer)
        return session_id, user_answer




# python -m app.multi_ai_agents.supervisor_agent
# Test
if __name__ == "__main__":
    sup = SupervisorAgent()
    async def test():
        session_id, answer1 = await sup.run(
            "Search the latest news about AI, solve (5+3*5)/2 and search in database for GreenGrow Innovations company history.",
        )

        print("\n[Session ID]", session_id)
        print("[Answer 1]\n", answer1)

        # Call again with the same session_id (keep history)
        # _, answer2 = await sup.run(
        #     "Where it is headquartered? and do you know Banh Mi and what is the weather there.", session_id=session_id
        # )

        # print("\n[Answer 2]\n", answer2)
        # print("\n[Session History]")
        # print(json.dumps(get_history(session_id), indent=2, ensure_ascii=False))
    asyncio.run(test())