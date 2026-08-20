from __future__ import annotations

import os
import time
<<<<<<< HEAD
from starlette.responses import JSONResponse
import httpx
=======
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any
>>>>>>> origin/agents/devops-mlops-streamlit-integration

import logging
from fastapi import FastAPI, Header, HTTPException, Request
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel
from starlette.responses import JSONResponse, Response

# basic logger to stdout (structured logs are handled by src/legal_ai/services/logging_utils if desired)
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger("legal_rag_api")

from src.legal_ai.core.config import load_json, set_seed
from src.legal_ai.core.models import PipelineConfig, RuntimeConfig
from src.legal_ai.evidence import build_grounded_context, select_evidence
from src.legal_ai.generation.manager import LLMManager
from src.legal_ai.services.query_service import QueryService

app = FastAPI(title="Legal RAG API", version="0.2")

REQUEST_COUNT = Counter("legal_rag_requests_total", "Total requests received by the API")
REQUEST_LATENCY = Histogram("legal_rag_request_latency_seconds", "Request latency in seconds")

# Simple in-memory rate limiter: per-key or per-IP counts with a rolling window
_RATE_LIMIT_WINDOW = 60  # seconds
_RATE_LIMIT_MAX = int(os.environ.get("API_RATE_LIMIT", "30"))  # requests per window
_rate_state: dict = {}


@app.middleware("http")
async def metrics_middleware(
    request: Request,
    call_next: Callable[[Request], Awaitable[Response]],
) -> Response:
    start = time.perf_counter()
    response = await call_next(request)
    duration = time.perf_counter() - start
    REQUEST_LATENCY.observe(duration)
    REQUEST_COUNT.inc()
    return response


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
def health() -> dict[str, Any]:
    return {"status": "ok"}


@app.get("/metrics")
def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/v1/query")
async def query(
    req: QueryRequest,
    request: Request,
    x_api_key: str | None = Header(None),
) -> JSONResponse:
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
        # Accept multiple possible locations for the documents file (robust against copy/paste names)
        candidate_paths = [Path("legal_documents.json"), Path("data") / "legal_documents.json", Path("legal_documents (1).json")]
        docs_path = None
        for candidate in candidate_paths:
            if candidate.exists():
                docs_path = candidate
                break
        if docs_path is None:
            raise HTTPException(
                status_code=404,
                detail=(
                    "Documents file not found; ingest first or place legal_documents.json (or data/legal_documents.json) in the app root"
                ),
            )
        documents = load_json(docs_path)
        runtime = RuntimeConfig()
        pipeline_cfg = PipelineConfig()
        rag = QueryService(
            documents,
            runtime,
            pipeline_cfg,
            Path("artifacts_api"),
            load_reranker=False,
        )
        retrieval = rag.retrieve(req.query, top_k=req.top_k)
        evidence = select_evidence(
            retrieval["results"],
            max_chars=pipeline_cfg.max_context_chars,
        )
        context = build_grounded_context(
            evidence,
            max_chars=pipeline_cfg.max_context_chars,
        )

        llm = LLMManager()
        llm.load()
        answer = llm.generate(req.query, context)
        return JSONResponse(
            {
                "query": req.query,
                "answer": answer,
                "sources": retrieval["results"],
            }
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/v1/ingest")
def ingest(
    payload: dict[str, Any],
    x_api_key: str | None = Header(None),
) -> dict[str, Any]:
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

    docs = payload.get("documents")
    if not isinstance(docs, list):
        raise HTTPException(
            status_code=400,
            detail="'documents' must be a list of document objects",
        )
    out = Path("ingested_documents.json")
    out.write_text(__import__("json").dumps(docs, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"status": "ok", "saved_to": str(out)}


@app.post("/v1/llm/gemini")
async def gemini_proxy(payload: Dict[str, Any], x_gemini_key: str | None = Header(None)) -> Dict[str, Any]:
    """Proxy endpoint for Google Generative Language (Gemini) interactions.

    Authentication precedence (highest -> lowest): X-Gemini-Key header, GEMINI_API_KEY env, api_key in JSON body.

    Request JSON shape:
      {"model": "gemini-3.7-flash", "input": "Explain how airplanes fly.", "api_key": "..."}

    Returns the raw Gemini JSON response.
    """
    # Determine API key
    api_key = x_gemini_key or os.environ.get("GEMINI_API_KEY") or payload.get("api_key")
    if not api_key:
        raise HTTPException(status_code=401, detail="Missing GEMINI API key. Set GEMINI_API_KEY or provide X-Gemini-Key or api_key in body.")

    model = payload.get("model", "gemini-3.7-flash")
    user_input = payload.get("input") or payload.get("prompt") or payload.get("text")
    if not user_input:
        raise HTTPException(status_code=400, detail="Missing 'input' in request body.")

    url = "https://generativelanguage.googleapis.com/v1beta/interactions"
    body = {"model": model, "input": user_input}
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(url, json=body, headers=headers)
            resp.raise_for_status()
            return resp.json()
    except httpx.HTTPStatusError as exc:
        # Forward Gemini HTTP error
        status = exc.response.status_code if exc.response is not None else 502
        detail = exc.response.text if exc.response is not None else str(exc)
        raise HTTPException(status_code=status, detail=detail)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
