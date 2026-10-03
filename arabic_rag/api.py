"""واجهة HTTP بسيطة فوق خط RAG — FastAPI."""
from __future__ import annotations

import os

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .pipeline import RAGPipeline

app = FastAPI(title="Arabic RAG API", version="0.1.0")
_pipeline: RAGPipeline | None = None


class AskRequest(BaseModel):
    question: str
    top_k: int | None = None


class IngestRequest(BaseModel):
    doc_id: str
    text: str


def get_pipeline() -> RAGPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = RAGPipeline()
        docs = os.getenv("ARABIC_RAG_DOCS", "data/sample_docs")
        if os.path.isdir(docs):
            _pipeline.add_directory(docs)
    return _pipeline


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "chunks": len(get_pipeline().index)}


@app.post("/ingest")
def ingest(req: IngestRequest) -> dict:
    chunks = get_pipeline().add_text(req.doc_id, req.text)
    return {"doc_id": req.doc_id, "chunks": len(chunks)}


@app.post("/ask")
def ask(req: AskRequest) -> dict:
    pipeline = get_pipeline()
    if not len(pipeline.index):
        raise HTTPException(status_code=400, detail="الفهرس فارغ — استوعب مستندات أولاً عبر /ingest")
    return pipeline.ask(req.question, top_k=req.top_k).to_dict()
