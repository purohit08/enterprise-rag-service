"""FastAPI backend for the PDF RAG agent. Run with:

    uv run uvicorn app.main:app --reload
"""
from __future__ import annotations

import shutil
from typing import List, Optional

from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel

from app import config, rag

app = FastAPI(title="PDF RAG Agent", version="1.0.0")


class QueryRequest(BaseModel):
    question: str
    top_k: Optional[int] = None
    generate: bool = True


@app.get("/health")
def health():
    return {
        "status": "ok",
        "device": rag.DEVICE,
        "embed_model": config.EMBED_MODEL,
        "gen_model": config.GEN_MODEL,
        "pdf_dir": str(config.PDF_DIR),
    }


@app.post("/ingest")
def ingest():
    """Read every PDF in the configured folder, embed, and store in Chroma."""
    config.PDF_DIR.mkdir(parents=True, exist_ok=True)
    return rag.ingest_pdfs()


@app.post("/upload")
async def upload(files: List[UploadFile] = File(...)):
    """Save uploaded PDFs into the configured folder (then call /ingest)."""
    config.PDF_DIR.mkdir(parents=True, exist_ok=True)
    saved = []
    for f in files:
        dest = config.PDF_DIR / f.filename
        with dest.open("wb") as out:
            shutil.copyfileobj(f.file, out)
        saved.append(f.filename)
    return {"saved": saved}


@app.post("/query")
def query(req: QueryRequest):
    return rag.answer_question(req.question, top_k=req.top_k, generate=req.generate)
