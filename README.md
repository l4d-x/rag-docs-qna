# 🤖 RAG-Based Document Q&A Chatbot

An end-to-end Retrieval-Augmented Generation (RAG) pipeline that lets users upload PDFs and ask natural language questions — eliminating manual document search.

## 🚀 Live Demo

**[Try it on HuggingFace Spaces →](https://huggingface.co/spaces/l4d-x/rag-document-qa)**

## 🎯 Features

- **PDF Upload** — Upload any PDF document for instant Q&A
- **Natural Language Queries** — Ask questions in plain English
- **Smart Retrieval** — Finds the most relevant chunks using semantic search
- **REST API** — Clean FastAPI backend with Swagger docs
- **Free Deployment** — Fully hosted on HuggingFace Spaces

## 🏗️ Architecture

```
PDF Upload → Text Extraction → Chunking (500 chars, 10% overlap)
                                        ↓
                              Embedding (all-MiniLM-L6-v2)
                                        ↓
                              FAISS Vector Store (384-dim)
                                        ↓
User Query → Embed Query → Top-4 Retrieval → LLaMA 3.3 70B (Groq) → Answer
```

## 🔧 Technical Details

| Component | Technology |
|-----------|-----------|
| **Framework** | LangChain |
| **API Backend** | FastAPI (REST API with Swagger docs) |
| **Embeddings** | all-MiniLM-L6-v2 (384-dim, HuggingFace) |
| **Vector Store** | FAISS |
| **LLM** | LLaMA 3.3 70B via Groq API |
| **Frontend** | Custom HTML |
| **Deployment** | HuggingFace Spaces |

## 💡 Key Implementation Details

- **Chunking Strategy**: 500-character chunks with 10% overlap to preserve context across chunk boundaries
- **Retrieval**: Top-4 most relevant chunks fetched from FAISS for each query
- **API Design**: Clean upload/ask endpoints with auto-generated Swagger documentation
- **Zero Cost**: Fully deployed on HuggingFace Spaces — no infrastructure cost

## 🛠️ Setup

```bash
# Clone the repo
git clone https://github.com/l4d-x/rag-docs-qna.git
cd rag-document-qa-chatbot

# Install dependencies
pip install -r requirements.txt

# Set environment variables
export GROQ_API_KEY=your_groq_api_key

# Run the app
uvicorn app:app --reload
```

## 📝 Usage

```python
import requests

# Upload a PDF
files = {"file": open("document.pdf", "rb")}
requests.post("http://localhost:8000/upload", files=files)

# Ask a question
response = requests.post("http://localhost:8000/ask", json={"question": "What is the main topic?"})
print(response.json()["answer"])
```

## 🧠 What I Learned

- Chunk size and overlap significantly impact retrieval quality
- FAISS provides efficient similarity search even on CPU
- Groq API enables fast inference with large models (LLaMA 3.3 70B) at zero cost
- FastAPI's auto-generated Swagger docs improve API usability
