from sqlalchemy import Column, Integer, String, Text, ForeignKey, TIMESTAMP, Numeric, JSON, SmallInteger
import uuid
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.sql import func
from app.database.database import Base

class AIConversation(Base):
    __tablename__ = "AIConversations"
    
    ID = Column(Integer, primary_key=True, index=True)
    UserID = Column(UUID(as_uuid=True), index=True)
    SessionID = Column(UUID(as_uuid=True), index=True)
    Prompt = Column(Text, nullable=False)
    Response = Column(Text, nullable=False)
    Model = Column(String(100))
    Temperature = Column(Numeric(4, 2))
    TokensPrompt = Column(Integer)
    TokensResponse = Column(Integer)
    Metadata = Column(JSON)
    Feedback = Column(SmallInteger)
    CreatedAt = Column(TIMESTAMP, server_default=func.now())