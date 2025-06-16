from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime

Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, index=True)
    email = Column(String(100), unique=True, index=True)
    hashed_password = Column(String(100))
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    faqs = relationship("FAQ", back_populates="author")

class FAQ(Base):
    __tablename__ = "faqs"
    
    id = Column(Integer, primary_key=True, index=True)
    question = Column(Text, index=True)
    answer = Column(Text)
    category = Column(String(50), index=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    author_id = Column(Integer, ForeignKey("users.id"))
    
    author = relationship("User", back_populates="faqs")
    embeddings = relationship("FAQEmbedding", back_populates="faq", uselist=False)

class FAQEmbedding(Base):
    __tablename__ = "faq_embeddings"
    
    id = Column(Integer, primary_key=True, index=True)
    faq_id = Column(Integer, ForeignKey("faqs.id"))
    embedding = Column(Text)  # Store as JSON string
    created_at = Column(DateTime, default=datetime.utcnow)
    
    faq = relationship("FAQ", back_populates="embeddings") 