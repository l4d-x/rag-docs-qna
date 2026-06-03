from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from rag_pipeline import build_rag_chain
from dotenv import load_dotenv
import shutil, os

load_dotenv()

app = FastAPI(title="RAG Document Q&A API")

chain = None

class QuestionRequest(BaseModel):
    question: str
    history: list[str] = []

@app.get("/", response_class=HTMLResponse)
async def home():
    return """
<!DOCTYPE html>
<html>
<head>
    <title>RAG Document Q&A</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { font-family: sans-serif; background: #0f0f0f; color: #eee; display: flex; flex-direction: column; align-items: center; padding: 40px 20px; min-height: 100vh; }
        h1 { font-size: 1.8rem; margin-bottom: 30px; color: #fff; }
        .card { background: #1a1a1a; border-radius: 12px; padding: 24px; width: 100%; max-width: 700px; margin-bottom: 20px; }
        label { font-size: 0.85rem; color: #aaa; display: block; margin-bottom: 8px; }
        input[type="file"] { width: 100%; padding: 10px; background: #2a2a2a; border: 1px solid #333; border-radius: 8px; color: #eee; cursor: pointer; }
        input[type="text"] { width: 100%; padding: 12px; background: #2a2a2a; border: 1px solid #333; border-radius: 8px; color: #eee; font-size: 1rem; }
        button { margin-top: 12px; width: 100%; padding: 12px; background: #2563eb; border: none; border-radius: 8px; color: #fff; font-size: 1rem; cursor: pointer; }
        button:hover { background: #1d4ed8; }
        #status { font-size: 0.85rem; margin-top: 10px; color: #4ade80; }
        #answer-box { margin-top: 16px; background: #2a2a2a; border-radius: 8px; padding: 16px; min-height: 80px; white-space: pre-wrap; line-height: 1.6; color: #e2e8f0; display: none; }
        #pages { font-size: 0.8rem; color: #94a3b8; margin-top: 8px; }
    </style>
</head>
<body>
    <h1>📄 RAG Document Q&A</h1>

    <div class="card">
        <label>Upload PDF</label>
        <input type="file" id="pdf-input" accept=".pdf">
        <button onclick="uploadPDF()">Upload</button>
        <div id="status"></div>
    </div>

    <div class="card">
        <label>Ask a question</label>
        <input type="text" id="question" placeholder="What is this document about?" onkeydown="if(event.key==='Enter') ask()">
        <button onclick="ask()">Ask</button>
        <div id="answer-box"></div>
        <div id="pages"></div>
    </div>

    <script>
        async function uploadPDF() {
            const file = document.getElementById('pdf-input').files[0];
            if (!file) return;
            document.getElementById('status').innerText = 'Uploading...';
            const form = new FormData();
            form.append('file', file);
            const res = await fetch('/upload', { method: 'POST', body: form });
            const data = await res.json();
            document.getElementById('status').innerText = data.status === 'loaded' ? '✅ ' + data.filename + ' loaded!' : '❌ Failed';
        }

        async function ask() {
            const q = document.getElementById('question').value.trim();
            if (!q) return;
            const box = document.getElementById('answer-box');
            box.style.display = 'block';
            box.innerText = 'Thinking...';
            document.getElementById('pages').innerText = '';
            const res = await fetch('/ask', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ question: q, history: [] })
            });
            const data = await res.json();
            box.innerText = data.answer || data.error;
            if (data.source_pages) document.getElementById('pages').innerText = '📄 Source pages: ' + data.source_pages.join(', ');
        }
    </script>
</body>
</html>
"""

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    global chain
    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    chain, _ = build_rag_chain([temp_path])
    os.remove(temp_path)
    return {"status": "loaded", "filename": file.filename}

@app.post("/ask")
async def ask_question(body: QuestionRequest):
    if chain is None:
        return {"error": "No document uploaded yet."}
    result = chain.invoke(body.question)
    pages = sorted(set(d.metadata.get("page", 0) + 1 for d in result["source_documents"]))
    return {"answer": result["answer"], "source_pages": pages}

@app.get("/health")
async def health():
    return {"status": "ok"}