# app.py
import streamlit as st
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import google.generativeai as genai
import os

# Configure Gemini API Key
genai.configure(api_key="AIzaSyBAuFMnOQ-ZrWPVeLwU_cPPwHB-mEaJTqM")  # <-- paste your key here

# Initialize embedding model and Gemini model
embedder = SentenceTransformer("all-MiniLM-L6-v2")
model = genai.GenerativeModel("gemini-2.0-flash")

# Load PDFs and create chunks
@st.cache_data(show_spinner=True)
def load_and_chunk_pdfs(pdf_files, chunk_size=500, overlap=50):
    all_chunks = []
    for pdf_file in pdf_files:
        reader = PdfReader(pdf_file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        words = text.split()
        i = 0
        while i < len(words):
            chunk = words[i:i+chunk_size]
            all_chunks.append(" ".join(chunk))
            i += chunk_size - overlap
    return all_chunks

# Compute embeddings
@st.cache_data(show_spinner=True)
def compute_embeddings(chunks):
    embeddings = embedder.encode(chunks, convert_to_tensor=False, show_progress_bar=True)
    return np.array(embeddings).astype('float32')

# Build FAISS index
def build_faiss_index(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index

# Retrieve top-k relevant chunks
def get_top_chunks(query, k, index, chunks):
    query_embedding = embedder.encode([query])
    D, I = index.search(np.array(query_embedding).astype('float32'), k)
    return [chunks[i] for i in I[0]]

# Generate answer using Gemini
def generate_answer_gemini(query, retrieved_chunks):
    context = "\n\n".join(retrieved_chunks)
    prompt = f"Use the context below to answer the question.\n\nContext:\n{context}\n\nQuestion: {query}\nAnswer:"
    response = model.generate_content(prompt)
    return response.text.strip()

# Streamlit UI
st.set_page_config(page_title="Document QA Chatbot", layout="wide")
st.title("📄 Intelligent Document QA Chatbot")

uploaded_files = st.file_uploader("Upload one or more PDF files", type="pdf", accept_multiple_files=True)

if uploaded_files:
    st.success(f"{len(uploaded_files)} file(s) uploaded successfully.")

    # Load and embed
    with st.spinner("Processing documents..."):
        chunks = load_and_chunk_pdfs(uploaded_files)
        embeddings = compute_embeddings(chunks)
        index = build_faiss_index(embeddings)

    st.success(f"Indexed {len(chunks)} text chunks.")

    query = st.text_input("Enter your question:")

    if query:
        with st.spinner("Retrieving relevant information..."):
            top_chunks = get_top_chunks(query, k=3, index=index, chunks=chunks)
            answer = generate_answer_gemini(query, top_chunks)

        st.subheader("Answer")
        st.write(answer)

        with st.expander("Show retrieved context"):
            for i, chunk in enumerate(top_chunks, start=1):
                st.markdown(f"**Chunk {i}:**\n{chunk}")

else:
    st.info("Please upload at least one PDF file to start.")
