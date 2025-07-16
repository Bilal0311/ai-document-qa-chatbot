# app.py
import streamlit as st
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import google.generativeai as genai
import os
import pandas as pd
import re

# Configure Gemini API Key
genai.configure(api_key="AIzaSyBAuFMnOQ-ZrWPVeLwU_cPPwHB-mEaJTqM")

# Initialize embedding and Gemini
embedder = SentenceTransformer("all-MiniLM-L6-v2")
model = genai.GenerativeModel("gemini-2.0-flash")

# Load PDFs and create chunks with metadata
@st.cache_data(show_spinner=True)
def load_and_chunk_pdfs(pdf_files, chunk_size=500, overlap=50):
    all_chunks = []
    for pdf_file in pdf_files:
        reader = PdfReader(pdf_file)
        for page_num, page in enumerate(reader.pages, start=1):
            page_text = page.extract_text()
            if page_text:
                words = page_text.split()
                i = 0
                while i < len(words):
                    chunk = words[i:i+chunk_size]
                    all_chunks.append({
                        "text": " ".join(chunk),
                        "page": page_num,
                        "file": pdf_file.name
                    })
                    i += chunk_size - overlap
    return all_chunks

# Compute embeddings
@st.cache_data(show_spinner=True)
def compute_embeddings(chunks):
    texts = [c["text"] for c in chunks]
    embeddings = embedder.encode(texts, convert_to_tensor=False, show_progress_bar=True)
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

# Highlight query keywords in text
def highlight_keywords(text, query):
    keywords = set(re.findall(r'\w+', query.lower()))
    def repl(match):
        word = match.group(0)
        if word.lower() in keywords:
            return f"<mark>{word}</mark>"
        else:
            return word
    return re.sub(r'\w+', repl, text)

# Generate answer using Gemini
def generate_answer_gemini(query, retrieved_chunks):
    context = "\n\n".join([c["text"] for c in retrieved_chunks])
    prompt = f"Use the context below to answer the question.\n\nContext:\n{context}\n\nQuestion: {query}\nAnswer:"
    response = model.generate_content(prompt)
    return response.text.strip()

# Streamlit Page Config
st.set_page_config(
    page_title="📄 AI Document QA Chatbot",
    page_icon="🤖",
    layout="wide"
)

# Sidebar
with st.sidebar:
    st.title("⚙️ Settings")
    chunk_size = st.slider("Chunk size (words)", 100, 1000, 500, 50)
    top_k = st.slider("Top-K Chunks to Retrieve", 1, 10, 3)
    st.markdown("---")
    st.markdown("📘 **Instructions**")
    st.markdown("1. Upload PDFs.\n2. Preview and ask.\n3. Download Q&A.")
    st.markdown("---")
    st.caption("Built with ❤️ using Streamlit + Gemini")

# Custom CSS
st.markdown("""
<style>
.answer-box {
    background-color: #f9f9f9;
    padding: 1rem;
    border-radius: 0.5rem;
    border: 1px solid #ddd;
    font-size: 1.1rem;
}
.chunk-box {
    background-color: #f0f0f0;
    padding: 0.6rem;
    border-radius: 0.5rem;
    margin-bottom:0.5rem;
    border: 1px solid #ddd;
    font-size:0.95rem;
}
.preview-box {
    background-color: #ffffff;
    padding: 0.6rem;
    border: 1px solid #ccc;
    border-radius: 0.5rem;
    margin-bottom: 1rem;
    font-size: 0.9rem;
    max-height: 300px;
    overflow-y: auto;
}
</style>
""", unsafe_allow_html=True)

# Title
st.title("📄 AI Document QA Chatbot")

# File uploader
uploaded_files = st.file_uploader(
    "📂 Upload PDF files",
    type="pdf",
    accept_multiple_files=True
)

if uploaded_files:
    st.success(f"{len(uploaded_files)} file(s) uploaded.")
    with st.spinner("Processing documents..."):
        chunks = load_and_chunk_pdfs(uploaded_files, chunk_size=chunk_size)
        embeddings = compute_embeddings(chunks)
        index = build_faiss_index(embeddings)
    st.info(f"Indexed {len(chunks)} chunks.")

    # Show previews
    st.subheader("👀 Document Previews")
    for file in uploaded_files:
        preview = ""
        reader = PdfReader(file)
        for page in reader.pages[:2]:
            t = page.extract_text()
            if t:
                preview += t[:1000] + "\n"
        with st.expander(f"📄 {file.name} Preview"):
            st.markdown(f"<div class='preview-box'>{preview}</div>", unsafe_allow_html=True)

    # Initialize session state
    if "history" not in st.session_state:
        st.session_state["history"] = []

    # Question input
    query = st.text_input("💬 Ask your question:")

    if query:
        with st.spinner("Retrieving and generating answer..."):
            top_chunks = get_top_chunks(query, k=top_k, index=index, chunks=chunks)
            answer = generate_answer_gemini(query, top_chunks)

        st.subheader("✅ Answer")
        st.markdown(f"<div class='answer-box'>{answer}</div>", unsafe_allow_html=True)

        # Show context with highlights
        with st.expander("📖 Retrieved Context"):
            for c in top_chunks:
                highlighted = highlight_keywords(c["text"], query)
                st.markdown(
                    f"<div class='chunk-box'><strong>{c['file']} — Page {c['page']}</strong><br>{highlighted}</div>",
                    unsafe_allow_html=True
                )

        # Save to session history
        st.session_state["history"].append({
            "question": query,
            "answer": answer
        })

    # Show Q&A History
    if st.session_state["history"]:
        st.subheader("📝 Q&A History")
        for i, qa in enumerate(st.session_state["history"], 1):
            with st.expander(f"Q{i}: {qa['question']}"):
                st.markdown(f"<b>Answer:</b> {qa['answer']}")

        # Download as CSV
        df = pd.DataFrame(st.session_state["history"])
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="💾 Download Q&A History as CSV",
            data=csv,
            file_name="qa_history.csv",
            mime="text/csv"
        )
else:
    st.info("📌 Upload PDF files to get started.")


