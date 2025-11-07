from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
import uuid
from datetime import datetime

class AIConversationBase(BaseModel):
    Prompt: str
    Response: str
    Model: Optional[str] = None
    Temperature: Optional[float] = 0.7
    TokensPrompt: Optional[int] = None
    TokensResponse: Optional[int] = None
    Metadata: Optional[Dict[str, Any]] = None

class AIConversationCreate(AIConversationBase):
    UserID: Optional[uuid.UUID] = None
    SessionID: Optional[uuid.UUID] = None

class AIConversation(AIConversationBase):
    ID: int
    UserID: Optional[uuid.UUID] = None
    SessionID: Optional[uuid.UUID] = None
    Feedback: Optional[int] = None
    CreatedAt: datetime
    
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