
````markdown
# 🧠 RAG-based Document Chatbot

A **Retrieval-Augmented Generation (RAG)** chatbot that can accurately answer questions based solely on the contents of uploaded PDF and DOCX files. This system is designed to run locally (offline) using only **open-source models and tools**.

---

## 📋 Features

✅ Accurate answers sourced only from provided documents  
✅ File support: `.pdf` and `.docx`  
✅ Smart hybrid chunking strategy  
✅ Uses **Qdrant** as the vector DB  
✅ Displays exact **source filename, page, and chunk ID**  
✅ Built for speed: ⏱️ <15s response time on GPU and average of 50s and <100s time on CPU  
✅ Completely **offline**

---

## 🚀 Setup Instructions

### 🧩 1. Install Dependencies

```bash
pip install -r requirements.txt
````

<details>
<summary><code>requirements.txt</code></summary>

```
transformers
torch
sentence-transformers
qdrant-client
python-docx
PyMuPDF
streamlit
```

</details>

---

### 📦 2. Download Models

Download and cache models locally:

📥 Pre-download Models (for offline usage)
**LLM**: `microsoft/phi-1_5`
**Embedding Model**: `BAAI/bge-small-en-v1.5`

---

### 🧠 3. Run the Chatbot

```bash
streamlit run app.py
```

Upload your files and begin asking document-based questions!

---

## 💡 Architecture Overview

```
┌────────────┐        ┌─────────────────────┐       ┌─────────────┐
│ PDF / DOCX │ ───▶   │ Chunking +          │ ───▶  │ Qdrant DB   │
│ Upload     │        │ Embedding (offline) │       │ (Vector DB) │
└────────────┘        └─────────────────────┘       └─────────────┘
                                                        ▲
                                                        │
                                                 Single Retrieval
                                                        │
                                                ┌────────────────┐
                                                │ Similarity     │
                                                │ Search (1 call)│
                                                └────────────────┘
                                                        │
┌────────────┐        ┌────────────────┐       ┌──────────────┐
│ User Query │ ───▶   │ Tokenizer +    │ ───▶  │ LLM (phi-1_5)│
│ (Streamlit)│        │ Context Merge  │       │              │
└────────────┘        └────────────────┘       └─────┬────────┘
                                                      │
                                               ┌──────▼──────┐
                                               │ Answer +    │
                                               │ Source Info │
                                               └─────────────┘
```

---

## 📐 Chunking Strategy

### 🎯 Goal

The goal of chunking is to convert raw document text into semantically meaningful, context-rich, and token-size-efficient units (“chunks”) that can be embedded and searched accurately — while strictly preserving the meaning of the original content.

### 🔍 Why Chunking Matters in RAG

If chunks are too large, they can:

* Overflow model context limits
* Cause irrelevant context to be retrieved
* Slow down inference and embedding

If chunks are too small, they:

* Lose context
* Lead to fragmented or misleading answers
* Increase semantic ambiguity during retrieval

To strike the right balance, we designed a custom hybrid chunking strategy.

### 📐 Strategy Design

We combined paragraph-based splitting with token-based sliding windows, making it both structure-aware and token-efficient.

### 🔢 Parameters

| Parameter   | Value | Description                                            |
| ----------- | ----- | ------------------------------------------------------ |
| MAX\_TOKENS | 100   | Max token length per chunk (optimizes speed + context) |
| MIN\_TOKENS | 60    | Min token length to avoid fragmentation                |
| OVERLAP     | 20    | Token overlap between chunks to preserve continuity    |

### ⚙️ How It Works (Step-by-Step)

1. **Split document into paragraphs**
   We first split the document by newline characters and remove empty lines.

2. **Check token length of each paragraph**

   * If paragraph has < MIN\_TOKENS (60):
     Merge it with the next paragraph to form a more complete context.

   * If paragraph has > MAX\_TOKENS (100):
     Apply sliding window token chunking:

     * Break into 100-token windows
     * Maintain 20-token overlap between consecutive chunks

   * If paragraph is within 60–100 tokens:
     Use it as-is.

3. **Edge-Case Handling**

   * If merging short paragraphs exceeds 100 tokens → break it back into sub-chunks.
   * If a document has long lists or tables, chunk them based on token count alone.

### 🧠 Why It Works Well

✅ **Semantic Preservation**
Using paragraph structure maintains natural boundaries between thoughts, topics, or arguments.

✅ **Avoids Hallucinations**
Strict token caps prevent large, irrelevant context from being included in the answer.

✅ **Efficient Retrieval**
Small, dense chunks are better suited for embedding models to accurately map query-to-context.

✅ **Contextual Continuity**
Sliding window ensures that no important phrase or sentence is cut off abruptly.

✅ **Scalable**
Efficient for both small documents and large multi-pages. Also suitable for batching across multiple files.

---

## 🔍 Retrieval Strategy

* **Embedding model**: `all-MiniLM-L6-v2`

* **Vector store**: [Qdrant](https://qdrant.tech)

  > ✅ Qdrant is hosted locally via official Windows `.exe` — no Docker or API keys required.
  > ✅ Ensures full offline compatibility as required.

* **1 query per question** → as required

* Only top `k=1–3` chunks retrieved

---

## 🧠 Generation (LLM)

* **Model used**: `microsoft/phi-1_5`
* Loaded with `transformers`
* Takes in retrieved chunk(s) + user query
* If context does not contain an answer → LLM says **"I don't know"**

---

## 💻 Hardware Usaged

| Component       | Optimization Strategy                  |
| --------------- | -------------------------------------- |
| LLM Inference   | CPU                                    |
| Tokenizer       | Batched tokenization                   |
| Embedding Model | Pre-loaded and shared across requests  |
| Chunking        | On-load, not per-query                 |
| Qdrant          | Fast local vector search (runs on CPU) |

---

## 🧪 Example Queries & Results

Available in the [`response_samples.pdf`](#) (See repo)
Includes 10 test queries with:

* Query text
* Screenshot of chatbot response
* Source (filename, page, chunk ID)

---

## 🎥 Demo Video

📹 [Watch the live demo (Unedited, full run)](https://example.com/demo-video)
✅ Face visible
✅ Real-time questions
✅ Full app run from scratch

---

## ❌ Limitation: Response Time Exceeds 15 Seconds (CPU Only)

This solution currently exceeds the 15-second response limit for query processing because it runs entirely on a CPU.

**Expected:** ≤15s (on GPU as per Hackathon rule)
**Actual:** \~50–60s per query (on CPU)

### 📌 Reason for Failure

The model inference and embedding search are slow on CPU due to the LLM's size and lack of parallel processing.

### 🛠 How to Fix

With access to a GPU (e.g., Tesla T4), the issue can be resolved by:

* Running both the LLM and embedding model on GPU (`device="cuda"`)
* Preloading models into memory
* Optimizing chunk size and batching

**⏱ Estimated Fix Time: 1–2 hours (with GPU)**

---

**Made with 💻 + ☕ by Yuvaraj M & YogeshWaran**


