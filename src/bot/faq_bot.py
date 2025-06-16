from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
import faiss
import numpy as np
import ollama
import json
from typing import List, Dict, Any
import os

class FaQBot:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        self.index = None
        self.id_to_text = {}
        
    def create_embeddings(self, texts: List[str]) -> np.ndarray:
        """Create embeddings for a list of texts."""
        return self.model.encode(texts)
    
    def create_index(self, texts: List[str]):
        """Create a FAISS index from texts."""
        # Split texts into chunks
        chunks = []
        for text in texts:
            chunks.extend(self.splitter.split_text(text))
        
        # Create embeddings
        vectors = self.create_embeddings(chunks)
        
        # Create and populate FAISS index
        self.index = faiss.IndexFlatL2(vectors.shape[1])
        self.index.add(np.array(vectors))
        self.id_to_text = {i: chunk for i, chunk in enumerate(chunks)}
        
    def search(self, query: str, k: int = 1) -> List[str]:
        """Search for similar texts to the query."""
        if not self.index:
            raise ValueError("Index not created. Call create_index first.")
            
        query_vector = self.model.encode([query])
        _, indices = self.index.search(np.array(query_vector), k=k)
        
        return [self.id_to_text[idx] for idx in indices[0]]
    
    def get_answer(self, query: str, context: str) -> str:
        """Get an answer from Ollama using the query and context."""
        response = ollama.chat(
            model="mistral",
            messages=[
                {"role": "system", "content": "You are a helpful FAQ assistant. Use the context provided to answer questions accurately and concisely."},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
            ]
        )
        return response['message']['content']
    
    def process_document(self, file_path: str) -> List[str]:
        """Process a document (PDF or DOCX) and return its text content."""
        if file_path.endswith('.pdf'):
            import fitz
            doc = fitz.open(file_path)
            return ["\n".join([page.get_text() for page in doc])]
        elif file_path.endswith('.docx'):
            import docx
            doc = docx.Document(file_path)
            return ["\n".join([para.text for para in doc.paragraphs])]
        else:
            raise ValueError(f"Unsupported file format: {file_path}")
    
    def load_documents(self, directory: str) -> List[str]:
        """Load all documents from a directory."""
        texts = []
        for filename in os.listdir(directory):
            if filename.endswith(('.pdf', '.docx')):
                file_path = os.path.join(directory, filename)
                texts.extend(self.process_document(file_path))
        return texts 