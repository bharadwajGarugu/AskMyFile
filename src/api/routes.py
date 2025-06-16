from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from typing import List, Optional
import json
import os
from datetime import datetime, timedelta
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel
import numpy as np
import faiss

from ..database.database import get_db
from ..database.models import User, FAQ, FAQEmbedding
from ..bot.faq_bot import FaQBot

# Initialize APIRouter
router = APIRouter()

# Security
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-here")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Initialize FaQBot
faq_bot = FaQBot()

# Pydantic models
class UserBase(BaseModel):
    username: str
    email: str

class UserCreate(UserBase):
    password: str

class User(UserBase):
    id: int
    is_active: bool
    created_at: datetime

    class Config:
        orm_mode = True

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    username: Optional[str] = None

class FAQBase(BaseModel):
    question: str
    answer: str
    category: str

class FAQCreate(FAQBase):
    pass

class FAQ(FAQBase):
    id: int
    created_at: datetime
    updated_at: datetime
    author_id: int

    class Config:
        orm_mode = True

# Security functions
def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username)
    except JWTError:
        raise credentials_exception
    user = db.query(User).filter(User.username == token_data.username).first()
    if user is None:
        raise credentials_exception
    return user

# Routes
@router.post("/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == form_data.username).first()
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token = create_access_token(data={"sub": user.username})
    return {"access_token": access_token, "token_type": "bearer"}

@router.post("/users/", response_model=User)
def create_user(user: UserCreate, db: Session = Depends(get_db)):
    db_user = db.query(User).filter(User.username == user.username).first()
    if db_user:
        raise HTTPException(status_code=400, detail="Username already registered")
    hashed_password = get_password_hash(user.password)
    db_user = User(
        username=user.username,
        email=user.email,
        hashed_password=hashed_password
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user

@router.post("/faqs/", response_model=FAQ)
def create_faq(faq: FAQCreate, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    db_faq = FAQ(**faq.dict(), author_id=current_user.id)
    db.add(db_faq)
    db.commit()
    db.refresh(db_faq)
    
    # Create and store embedding
    embedding = faq_bot.create_embeddings([f"{faq.question} {faq.answer}"])[0]
    db_embedding = FAQEmbedding(faq_id=db_faq.id, embedding=json.dumps(embedding.tolist()))
    db.add(db_embedding)
    db.commit()
    
    return db_faq

@router.get("/faqs/", response_model=List[FAQ])
def read_faqs(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    faqs = db.query(FAQ).offset(skip).limit(limit).all()
    return faqs

@router.get("/faqs/search/")
def search_faqs(query: str, db: Session = Depends(get_db)):
    # Get all FAQ embeddings
    embeddings = db.query(FAQEmbedding).all()
    if not embeddings:
        return {"answer": "No FAQs available in the system."}
    
    # Convert stored embeddings back to numpy arrays
    vectors = np.array([json.loads(e.embedding) for e in embeddings])
    
    # Create FAISS index
    index = faiss.IndexFlatL2(vectors.shape[1])
    index.add(vectors)
    
    # Search
    query_vector = faq_bot.create_embeddings([query])[0]
    _, indices = index.search(np.array([query_vector]), k=1)
    
    # Get the most relevant FAQ
    relevant_faq = db.query(FAQ).filter(FAQ.id == embeddings[indices[0][0]].faq_id).first()
    
    return {
        "answer": relevant_faq.answer,
        "question": relevant_faq.question,
        "category": relevant_faq.category
    }

@router.post("/upload/")
async def upload_document(file: UploadFile = File(...), db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    # Save the file
    file_path = f"docs/{file.filename}"
    with open(file_path, "wb") as buffer:
        content = await file.read()
        buffer.write(content)
    
    # Process the document
    texts = faq_bot.process_document(file_path)
    
    # Create FAQs from the document
    for text in texts:
        # Simple heuristic to split text into Q&A pairs
        # You might want to implement more sophisticated parsing
        if "Q:" in text and "A:" in text:
            qa_pairs = text.split("Q:")
            for pair in qa_pairs[1:]:  # Skip the first empty split
                if "A:" in pair:
                    question, answer = pair.split("A:", 1)
                    faq = FAQ(
                        question=question.strip(),
                        answer=answer.strip(),
                        category="Document",
                        author_id=current_user.id
                    )
                    db.add(faq)
                    db.commit()
                    db.refresh(faq)
                    
                    # Create and store embedding
                    embedding = faq_bot.create_embeddings([f"{question} {answer}"])[0]
                    db_embedding = FAQEmbedding(faq_id=faq.id, embedding=json.dumps(embedding.tolist()))
                    db.add(db_embedding)
                    db.commit()
    
    return {"message": "Document processed and FAQs created."} 