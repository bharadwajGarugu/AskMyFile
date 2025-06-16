# internal_bot.py

from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
import faiss
import numpy as np
import ollama
import os

# Step 1: Your internal Q&A and knowledge base
custom_data = [
    # SFTP Info
    "Q: How to access Sandbox Alacriti SFTP or Sandbox SFTP SB? A: Use the following command:\n"
    "sftp -oIdentityFile=/home/jboss/alacrititest/id_rsa orbipftest@awssftpng.alacriti.com\n"
    "Username: orbipftest\nServer: awssftpng.alacriti.com\nPath: /home/jboss/alacrititest/id_rsa",

    # Developer portal
    "Q: What is the URL for Alacriti Developer API portal? A: https://developers.orbipay.com",




    # SB COM Application
    "Q: What is the login URL for SB COM application? A: https://sbums.orbipay.com/umsui/#/login?partnerKey=bcm&app_id=28\n"
    "Enter this site and log in with your credentials.",

    # Decrypting process
    "Q: How do I run the internal decrypting process? A:\n"
    "Step 1: DB Configuration:\nGo to ~/utility/yaf/tests/dbutility/conf/common/\n\n"
    "Step 2: Modify SQL Queries:\nOpen sql-queries.xml in ~/utility/yaf/tests/dbutility/conf/common/.\n"
    "Modify select and update queries:\n"
    "Select: SELECT record_id,acct_no FROM support_obps354028_temp_tbl WHERE record_id between ? and ?\n"
    "Update: UPDATE support_obps354028_temp_tbl SET acct_no_dec=? WHERE record_id=?\n\n"
    "Step 3: Run the script:\nNavigate to ~/utility/yaf/tests/dbutility and run: sh run.sh\n"
    "Select option 5 for encryption/decryption, enter:\n"
    "- Source query: 8021\n- Target query: 8022\n- Choose (e) or (d)\n- Start and end record IDs.",

    # You can keep adding more Q&As below
]

# Optional: Load more data from docs/
def load_text_from_pdf(path):
    import fitz
    doc = fitz.open(path)
    return "\n".join([page.get_text() for page in doc])

def load_text_from_docx(path):
    import docx
    doc = docx.Document(path)
    return "\n".join([para.text for para in doc.paragraphs])

external_texts = []
doc_folder = "docs"
if os.path.exists(doc_folder):
    for filename in os.listdir(doc_folder):
        full_path = os.path.join(doc_folder, filename)
        if filename.endswith(".pdf"):
            external_texts.append(load_text_from_pdf(full_path))
        elif filename.endswith(".docx"):
            external_texts.append(load_text_from_docx(full_path))

# Step 2: Chunk the documents
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = []
for text in custom_data + external_texts:
    chunks.extend(splitter.split_text(text))

# Step 3: Embed the chunks
print("Creating embeddings...")
model = SentenceTransformer("all-MiniLM-L6-v2")
vectors = model.encode(chunks)

# Step 4: Index in FAISS
index = faiss.IndexFlatL2(vectors.shape[1])
index.add(np.array(vectors))
id_to_text = {i: chunk for i, chunk in enumerate(chunks)}

# Step 5: User query input
print("\nInternal Knowledge Assistant")
query = input("Ask a question: ")

# Step 6: Embed query and search
query_vector = model.encode([query])
_, indices = index.search(np.array(query_vector), k=1)
relevant_chunk = id_to_text[indices[0][0]]

# Step 7: Ask Mistral (Ollama)
print("\n--- Mistral's Answer ---\n")
response = ollama.chat(
    model="mistral",
    messages=[
        {"role": "system", "content": "You are a helpful internal assistant. Use the context provided."},
        {"role": "user", "content": f"Context:\n{relevant_chunk}\n\nQuestion: {query}"}
    ]
)

print(response['message']['content'])
