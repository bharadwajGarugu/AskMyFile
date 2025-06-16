import logging
import os
import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
import faiss
import numpy as np
import ollama
from typing import List, Dict, Any
import json
from datetime import datetime
import time
from functools import lru_cache

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('askmyfile.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class AskMyFile:
    def __init__(self):
        self.model = None
        self.index = None
        self.id_to_text = {}
        self.splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        self.initialize_model()
        self.chat_history = []
        
    def initialize_model(self):
        """Initialize the sentence transformer model"""
        try:
            logger.info("Initializing sentence transformer model...")
            self.model = SentenceTransformer("all-MiniLM-L6-v2")
            logger.info("Model initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing model: {str(e)}")
            raise

    def get_cached_embedding(self, text: str) -> np.ndarray:
        """Get embedding for text with caching"""
        return self.model.encode([text])[0]

    def load_data(self, data_source: str) -> List[str]:
        """Load data from various sources"""
        try:
            if data_source.endswith('.csv'):
                df = pd.read_csv(data_source)
                return df.to_string()
            elif data_source.endswith('.xlsx'):
                df = pd.read_excel(data_source)
                return df.to_string()
            elif data_source.endswith('.json'):
                with open(data_source, 'r') as f:
                    return json.dumps(json.load(f))
            elif data_source.endswith('.txt'):
                with open(data_source, 'r') as f:
                    return f.read()
            else:
                logger.warning(f"Unsupported file format: {data_source}")
                return []
        except Exception as e:
            logger.error(f"Error loading data from {data_source}: {str(e)}")
            return []

    def process_documents(self, documents: List[str], progress_bar):
        """Process documents and create embeddings with progress tracking"""
        try:
            logger.info("Processing documents...")
            chunks = []
            total_chunks = 0
            
            # First pass: count total chunks
            for doc in documents:
                doc_chunks = self.splitter.split_text(doc)
                total_chunks += len(doc_chunks)
            
            # Second pass: process chunks with progress
            processed_chunks = 0
            for doc in documents:
                doc_chunks = self.splitter.split_text(doc)
                chunks.extend(doc_chunks)
                processed_chunks += len(doc_chunks)
                progress_bar.progress(min(processed_chunks / total_chunks, 1.0), text="Processing chunks...")
            
            logger.info(f"Created {len(chunks)} chunks")
            
            # Create embeddings with progress
            vectors = []
            for i, chunk in enumerate(chunks):
                vector = self.get_cached_embedding(chunk)
                vectors.append(vector)
                progress_bar.progress(min((i + 1) / len(chunks), 1.0), text="Creating embeddings...")
            
            vectors = np.array(vectors)
            
            # Create FAISS index
            self.index = faiss.IndexFlatL2(vectors.shape[1])
            self.index.add(vectors)
            self.id_to_text = {i: chunk for i, chunk in enumerate(chunks)}
            logger.info("Documents processed successfully")
        except Exception as e:
            logger.error(f"Error processing documents: {str(e)}")
            raise

    def query(self, question: str) -> str:
        """Query the knowledge base"""
        try:
            if not self.index:
                return "Please load some documents first."
            
            query_vector = self.get_cached_embedding(question)
            _, indices = self.index.search(np.array([query_vector]), k=1)
            relevant_chunk = self.id_to_text[indices[0][0]]
            
            response = ollama.chat(
                model="mistral",
                messages=[
                    {"role": "system", "content": "You are a helpful internal assistant. Use the context provided."},
                    {"role": "user", "content": f"Context:\n{relevant_chunk}\n\nQuestion: {question}"}
                ]
            )
            
            return response['message']['content']
        except Exception as e:
            logger.error(f"Error querying knowledge base: {str(e)}")
            return f"An error occurred: {str(e)}"

    def add_to_history(self, question: str, answer: str):
        """Add a Q&A pair to chat history"""
        self.chat_history.append({
            "question": question,
            "answer": answer,
            "timestamp": datetime.now().strftime("%H:%M:%S")
        })

def main():
    st.set_page_config(
        page_title="AskMyFile",
        page_icon="📄",
        layout="wide"
    )
    
    # Initialize session state
    if 'bot' not in st.session_state:
        st.session_state.bot = AskMyFile()
    
    # Custom CSS for dark theme
    st.markdown("""
        <style>
        .stApp {
            background-color: #1E1E1E;
            color: #FFFFFF;
        }
        .stTextInput>div>div>input {
            background-color: #2D2D2D;
            color: #FFFFFF;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
        }
        .chat-message {
            padding: 1rem;
            border-radius: 0.5rem;
            margin-bottom: 1rem;
            display: flex;
            flex-direction: column;
        }
        .chat-message.user {
            background-color: #2D2D2D;
        }
        .chat-message.bot {
            background-color: #1E1E1E;
            border: 1px solid #4CAF50;
        }
        .file-types {
            font-size: 0.8em;
            color: #888;
            margin-top: -1rem;
            margin-bottom: 1rem;
        }
        .app-title {
            font-size: 2.5em;
            font-weight: bold;
            margin-bottom: 1rem;
            color: #4CAF50;
        }
        </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="app-title">📄 AskMyFile</div>', unsafe_allow_html=True)
    
    # File uploader with progress
    st.markdown("📁 Got a document? Toss it in — we'll handle the thinking.")
    st.markdown('<div class="file-types">Supported file types: CSV, Excel, JSON, TXT</div>', unsafe_allow_html=True)
    
    uploaded_files = st.file_uploader(
        "",
        accept_multiple_files=True,
        type=['csv', 'xlsx', 'json', 'txt']
    )
    
    if uploaded_files:
        progress_bar = st.progress(0, text="Starting document processing...")
        documents = []
        
        for i, file in enumerate(uploaded_files):
            # Save uploaded file temporarily
            temp_path = f"temp_{file.name}"
            with open(temp_path, "wb") as f:
                f.write(file.getvalue())
            
            # Load and process the file
            content = st.session_state.bot.load_data(temp_path)
            if content:
                documents.append(content)
            
            # Clean up temporary file
            os.remove(temp_path)
            
            # Update progress
            progress_bar.progress((i + 1) / len(uploaded_files), text=f"Processing {file.name}...")
        
        if documents:
            st.session_state.bot.process_documents(documents, progress_bar)
            st.success("Documents processed successfully!")
    
    # Chat interface
    st.markdown("### Chat")
    
    # Display chat history
    for message in st.session_state.bot.chat_history:
        st.markdown(f"""
            <div class="chat-message user">
                <strong>You ({message['timestamp']}):</strong><br>
                {message['question']}
            </div>
            <div class="chat-message bot">
                <strong>AskMyFile ({message['timestamp']}):</strong><br>
                {message['answer']}
            </div>
        """, unsafe_allow_html=True)
    
    # Query input with keyboard shortcut (Enter)
    query = st.text_input("💬 What's on your mind? Let's interrogate that document.", key="query_input")
    
    if query:
        with st.spinner("📚 Reading every line like it's a thriller novel… hang tight!"):
            response = st.session_state.bot.query(query)
            st.session_state.bot.add_to_history(query, response)
            st.rerun()  # Refresh to show new message
    elif st.session_state.bot.chat_history:
        st.markdown("🎯 Got more questions? Your file's still chatting.")
    else:
        st.markdown("🤖 I can read minds (sort of)… but you still need to type your question!")

if __name__ == "__main__":
    main() 