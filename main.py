import streamlit as st
from rag_bot.utils.document_loader import process_file
from rag_bot.utils.embedding import get_embedding, get_batch_embeddings
from rag_bot.utils.qdrant_manager import insert_documents, search
from rag_bot.utils.llm import generate_answer
import tempfile
import GPUtil
import psutil
import time

st.set_page_config(page_title="Offline RAG Chatbot", layout="wide")
st.title("📚 Offline RAG Chatbot with Qdrant + Phi-1.5")

# Session State
if "history" not in st.session_state:
    st.session_state.history = []

# File Upload
uploaded_files = st.file_uploader("Upload PDF or DOCX files", type=["pdf", "docx"], accept_multiple_files=True)

if uploaded_files:
    all_chunks = []
    for file in uploaded_files:
        chunks = process_file(file)
        all_chunks.extend(chunks)

    if all_chunks:
        with st.spinner("Embedding and indexing documents..."):
            embeddings = get_batch_embeddings(all_chunks)
            inserted = insert_documents(all_chunks, embeddings)
            st.success(f"✅ Indexed {inserted} chunks from {len(uploaded_files)} document(s).")

# Chat Input
user_input = st.text_input("Ask a question:", placeholder="What does the document say about XYZ?")

if user_input:
    start_time = time.time()
    query_embedding = get_embedding(user_input)
    retrieved_chunks = search(query_embedding, top_k=3)
    if not retrieved_chunks:
        st.warning("No relevant documents found.")
    else:
        newstart = time.time()
        answer = generate_answer(user_input,retrieved_chunks )
        end_time = time.time()- newstart
        st.success(f"Answer generated in {end_time:.2f} seconds.")
        duration = time.time() - start_time

        # Show Answer
        st.markdown("### 🧠 Answer")
        st.markdown(answer)

        # Show Source
        st.markdown("### 📄 Source(s) Used")
        for chunk in retrieved_chunks:
            st.markdown(f"- **{chunk['source']}**, page {chunk['page']}, chunk `{chunk['chunk_id']}`")

        # Add to chat history
        st.session_state.history.append({"q": user_input, "a": answer, "chunks": retrieved_chunks, "time": duration})

# Chat History
if st.session_state.history:
    st.markdown("---")
    st.markdown("## 💬 Chat History")
    for i, entry in enumerate(st.session_state.history[::-1]):
        st.markdown(f"**Q{i+1}:** {entry['q']}")
        st.markdown(f"**A{i+1}:** {entry['a']}")
        for chunk in entry["chunks"]:
            st.markdown(f"📄 **{chunk['source']}**, page {chunk['page']}, chunk `{chunk['chunk_id']}`")
        st.markdown(f"⏱ *Response Time: {entry['time']:.2f} sec*")
        st.markdown("---")

# Memory Usage
gpu = GPUtil.getGPUs()[0] if GPUtil.getGPUs() else None
cpu_mem = psutil.virtual_memory()
st.sidebar.markdown("### ⚙️ System Resource Usage")

st.sidebar.markdown(f"**CPU Usage:** {cpu_mem.percent}%")
st.sidebar.markdown(f"**Available RAM:** {cpu_mem.available // (1024**2)} MB")

if gpu:
    st.sidebar.markdown(f"**GPU Usage:** {gpu.memoryUtil*100:.2f}%")
    st.sidebar.markdown(f"**GPU Memory Used:** {gpu.memoryUsed} MB / {gpu.memoryTotal} MB")
