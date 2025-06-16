# AskMyFile

AskMyFile is an intelligent document analysis tool that allows you to ask questions about your documents and get AI-powered responses. It uses advanced natural language processing and machine learning techniques to understand and analyze your documents.

## Features

- 📄 Document Processing: Supports multiple file formats (CSV, Excel, JSON, TXT)
- 🤖 AI-Powered Q&A: Ask questions about your documents in natural language
- 🔍 Smart Search: Uses semantic search to find relevant information
- 💬 Interactive Chat Interface: Easy-to-use chat-like interface
- 🌙 Dark Theme: Modern, eye-friendly dark theme interface
- ⚡ Performance Optimized: Fast document processing and response times

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/askmyfile.git
cd askmyfile
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

3. Install Ollama:
   - Download and install Ollama from [ollama.ai](https://ollama.ai)
   - Pull the Mistral model:
   ```bash
   ollama pull mistral
   ```

## Usage

1. Start the application:
```bash
streamlit run enhanced_bot.py
```

2. Open your web browser and navigate to the provided URL (usually http://localhost:8501)

3. Upload your documents using the file uploader

4. Start asking questions about your documents!

## Supported File Types

- CSV files (.csv)
- Excel files (.xlsx)
- JSON files (.json)
- Text files (.txt)

## Performance Optimizations

The application includes several performance optimizations:
- LRU caching for embeddings
- Optimized vector operations
- Efficient document chunking
- Progress tracking for long operations

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.



## Acknowledgments

- [Streamlit](https://streamlit.io/) for the web interface
- [Sentence Transformers](https://www.sbert.net/) for text embeddings
- [FAISS](https://github.com/facebookresearch/faiss) for efficient similarity search
- [Ollama](https://ollama.ai/) for the LLM backend
