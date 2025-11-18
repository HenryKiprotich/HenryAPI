from sqlalchemy import Column, Integer, String, Text, ForeignKey, TIMESTAMP, Numeric, JSON, SmallInteger
import uuid
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.sql import func
from app.database.database import Base

class AIConversation(Base):
    __tablename__ = "AIConversations"
    
    id = Column("ID", Integer, primary_key=True, index=True)
    user_id = Column("UserID",UUID(as_uuid=True), index=True)
    session_id = Column("SessionID",UUID(as_uuid=True), index=True)
    prompt = Column("Prompt", Text, nullable=False)
    response = Column("Response", Text, nullable=False)
    model = Column("Model", String(100))
    temperature = Column("Temperature", Numeric(4, 2))
    tokens_prompt = Column("TokensPrompt", Integer)
    tokens_response = Column("TokensResponse", Integer)
    metadata = Column("Metadata", JSON)
    feedback = Column("Feedback", SmallInteger)
    created_at = Column("CreatedAt", TIMESTAMP, server_default=func.now())