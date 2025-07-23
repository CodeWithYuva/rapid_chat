# 🧠 RAG Chatbot with Qdrant + Phi-1.5

This is an offline Retrieval-Augmented Generation (RAG) chatbot built with:

- ✅ Qdrant as the vector store
- ✅ BGE Small for fast embeddings
- ✅ Phi-1.5 (local LLM) for answering queries
- ✅ Streamlit UI
- ✅ Full offline operation (no OpenAI, no LangChain)

## 🔧 Features

- PDF/DOCX file ingestion
- Embedding with SentenceTransformer
- Semantic search via Qdrant
- LLM response with contextual awareness
- GPU/CPU memory tracking

## 🖼 Example Output

**Query:** What does the first paragraph talk about?

**Answer:**
> The first paragraph introduces the core idea of X and discusses its relevance in Y context.

**Source:** my-doc.pdf, page 1, chunk p1_c1

---

## 🚀 Run Locally

```bash
git clone https://github.com/yourname/rag-chatbot.git
cd rag-chatbot
pip install -r requirements.txt
streamlit run main.py
