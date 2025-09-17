from pydantic import BaseModel

class SearchResult(BaseModel):
    question: str
    top_k: int = 5

class ConversationRequest(BaseModel):
    session_id: str
    user_input: str
