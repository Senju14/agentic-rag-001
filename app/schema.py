from pydantic import BaseModel
from typing import Any, Dict, List, Optional

class SearchResult(BaseModel):
    question: str
    top_k: int = 5
    min_score: Optional[float] = None
    allowed_sources: Optional[List[str]] = None             
    allowed_types: Optional[List[str]] = None

class ConversationRequest(BaseModel):
    session_id: str
    user_input: str
