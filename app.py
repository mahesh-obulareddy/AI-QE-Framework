"""
app.py
------
FastAPI server for the HR Policy RAG Chatbot.

Serves the chat UI at http://localhost:8000 and exposes a POST /chat API
that wraps hr_rag_chatbot.query_hr_bot().

Run:
    source .venv/bin/activate
    python app.py
    # or
    uvicorn app:app --reload --port 8000
"""

import warnings
warnings.filterwarnings("ignore")

import os
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from hr_rag_chatbot import build_rag_chain, query_hr_bot

# ---------------------------------------------------------------------------
# Globals — RAG chain built once at startup, reused for every request
# ---------------------------------------------------------------------------

_rag_chain: Any = None
_retriever: Any = None
_model_name: str = os.getenv("OLLAMA_MODEL", "llama3.1")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Build the RAG chain once when the server starts."""
    global _rag_chain, _retriever
    print(f"\n[STARTUP] Building RAG chain with model: {_model_name}")
    print("[STARTUP] This may take ~30 seconds on first run (embedding the PDFs)...\n")
    _rag_chain, _retriever = build_rag_chain(model_name=_model_name)
    print("[STARTUP] Server ready! Open http://localhost:8000 in your browser.\n")
    yield
    print("[SHUTDOWN] Shutting down.")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="HR Policy Chatbot",
    description="RAG-powered HR Policy Q&A using LangChain + Ollama",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static frontend files
if os.path.isdir("frontend"):
    app.mount("/static", StaticFiles(directory="frontend"), name="static")


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    message: str
    model: str = _model_name   # allows per-request model switching


class SourceChunk(BaseModel):
    text: str
    source: str
    page: int


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceChunk]
    model: str


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    """Serve the chat UI."""
    ui_path = os.path.join("frontend", "index.html")
    if os.path.exists(ui_path):
        with open(ui_path, "r") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content="<h1>Frontend not found. Place index.html in ./frontend/</h1>")


@app.get("/health")
async def health():
    """Health check — confirms model loaded."""
    return {
        "status": "ok",
        "model": _model_name,
        "rag_ready": _rag_chain is not None,
    }


@app.get("/documents")
async def list_documents():
    """List all HR policy documents loaded into the vector store."""
    docs_dir = "./hr_documents"
    if not os.path.isdir(docs_dir):
        return {"documents": []}
    files = sorted([f for f in os.listdir(docs_dir) if f.endswith(".pdf")])
    # Clean up filenames for display
    display = [f.replace("-", " ").replace(".pdf", "") for f in files]
    return {"documents": display, "count": len(files)}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Main chat endpoint.

    Accepts a user question and returns the grounded answer
    plus the raw retrieved source chunks.
    """
    if not _rag_chain or not _retriever:
        raise HTTPException(status_code=503, detail="RAG chain not ready yet.")

    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    try:
        result = query_hr_bot(request.message.strip(), _rag_chain, _retriever)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RAG error: {str(e)}")

    # Build structured source list
    sources = []
    for text, meta in zip(result["source_documents"], result["source_metadata"]):
        raw_source = meta.get("source", "Unknown")
        # Strip path prefix and .pdf for clean display
        display_source = os.path.basename(raw_source).replace(".pdf", "").replace("-", " ")
        sources.append(SourceChunk(
            text=text[:500] + "..." if len(text) > 500 else text,
            source=display_source,
            page=int(meta.get("page", 0)) + 1,   # 1-indexed for display
        ))

    return ChatResponse(
        answer=result["answer"],
        sources=sources,
        model=_model_name,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=False,          # set True during development
        log_level="warning",   # suppress uvicorn info spam
    )
