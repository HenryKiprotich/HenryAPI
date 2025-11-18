from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
import uuid
from datetime import datetime

class AIConversationBase(BaseModel):
    prompt: str
    response: str
    model: Optional[str] = None
    temperature: Optional[float] = 0.7
    tokens_prompt: Optional[int] = None
    tokens_response: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None

class AIConversationCreate(AIConversationBase):
    user_id: Optional[uuid.UUID] = None
    session_id: Optional[uuid.UUID] = None

class AIConversation(AIConversationBase):
    id: int
    user_id: Optional[uuid.UUID] = None
    session_id: Optional[uuid.UUID] = None
    feedback: Optional[int] = None
    created_at: datetime

    class Config:
        from_attributes = True

class FeedbackUpdate(BaseModel):
    conversation_id: int
    feedback: int
    comment: Optional[str] = None

class ConversationHistory(BaseModel):
    conversations: List[AIConversation]
    total: int
    limit: int
    offset: int