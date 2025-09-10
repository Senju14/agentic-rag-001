from pydantic import BaseModel
from typing import Any, Dict, List, Optional

 
class Document(BaseModel):
    id: Optional[int]
    title: str
    file_name: str
    file_type: str

class Chunk(BaseModel):
    id: Optional[int]
    document_id: int
    chunk_text: str
    chunk_index: int
    metadata: Optional[dict] = {}

class ChatRequest(BaseModel):
    question: str
    top_k: int = 5
    min_score: Optional[float] = None
    allowed_sources: Optional[List[str]] = None
    allowed_types: Optional[List[str]] = None

class ChatResponse(BaseModel):
    answer: str
    source_chunks: List[dict]

class ChatbotRequest(BaseModel):
    session_id: str
    user_input: str

class ToolDescription(BaseModel):
    name: str
    description: str
    parameters: Dict[str, Any]

class FunctionCallRequest(BaseModel):
    user_input: str
    session_id: Optional[str] = None

class ToolCallStep(BaseModel):
    thought: str
    action: Optional[str] = None
    action_input: Optional[Dict[str, Any]] = None
    observation: Optional[str] = None

class FunctionCallResponse(BaseModel):
    answer: str
    tool_trace: List[ToolCallStep]
    tools: List[ToolDescription]
    source_chunks: Optional[List[dict]] = None
