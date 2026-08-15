from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from fastapi import FastAPI, HTTPException, Request, Header
from pydantic import BaseModel
import os
import time
from starlette.responses import JSONResponse

from legal_ai.service import RAGService
from legal_ai.generation import LLMManager
from legal_rag_engine import RuntimeConfig, PipelineConfig, load_json, set_seed

app = FastAPI(title="Legal RAG API", version="0.2")

# Simple in-memory rate limiter: per-key or per-IP counts with a rolling window
_RATE_LIMIT_WINDOW = 60  # seconds
_RATE_LIMIT_MAX = int(os.environ.get("API_RATE_LIMIT", "30"))  # requests per window
_rate_state: dict = {}


def _rate_limited(key: str) -> bool:
    now = int(time.time())
    window_start = now - _RATE_LIMIT_WINDOW
    data = _rate_state.get(key)
    if data is None or data[0] < window_start:
        _rate_state[key] = (now, 1)
        return False
    ts, count = data
    if count >= _RATE_LIMIT_MAX:
        return True
    _rate_state[key] = (ts, count + 1)
    return False


API_KEY = os.environ.get("API_KEY")


class QueryRequest(BaseModel):
    query: str
    top_k: int = 5


@app.get("/v1/health")
def health() -> Dict[str, Any]:
    return {"status": "ok"}


@app.post("/v1/query")
async def query(req: QueryRequest, request: Request, x_api_key: str | None = Header(None)):
    # API key enforcement (if API_KEY is set)
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

    # Rate limit by API key or client IP
    key = x_api_key or request.client.host
    if _rate_limited(key):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")

    # Input limits
    if len(req.query) > 8000:
        raise HTTPException(status_code=400, detail="Query too long")

    try:
        set_seed()
        docs_path = Path("legal_documents (1).json")
        if not docs_path.exists():
            raise HTTPException(status_code=404, detail="Documents file not found; ingest first or place legal_documents (1).json in the app root")
        documents = load_json(docs_path)
        runtime = RuntimeConfig()
        pipeline_cfg = PipelineConfig()
        rag = RAGService(documents, runtime, pipeline_cfg, Path("artifacts_api"), load_reranker=False)
        retrieval = rag.retrieve(req.query, top_k=req.top_k)
        context = rag.build_context(retrieval["results"], max_chars=pipeline_cfg.max_context_chars)

        llm = LLMManager()
        llm.load()
        answer = llm.generate(req.query, context)
        return JSONResponse({
            "query": req.query,
            "answer": answer,
            "sources": retrieval["results"],
        })
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/v1/ingest")
def ingest(payload: Dict[str, Any], x_api_key: str | None = Header(None)):
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

    docs = payload.get("documents")
    if not isinstance(docs, list):
        raise HTTPException(status_code=400, detail="'documents' must be a list of document objects")
    out = Path("ingested_documents.json")
    out.write_text(__import__("json").dumps(docs, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"status": "ok", "saved_to": str(out)}
