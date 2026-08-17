#!/usr/bin/env python3
"""
update_knowledge.py

Simple ingestion-to-index pipeline driver for local testing.

Usage:
  # Quick simulated run (fast, no real embeddings):
  python scripts/update_knowledge.py --input new_laws/sample_new_law.json --out artifacts_local --simulate

  # Full run (will use project's DenseEncoder and build FAISS index) - may be slow/require large downloads
  python scripts/update_knowledge.py --input new_laws/sample_new_law.json --out artifacts_local --real

The script performs:
- Validation
- Normalization
- Chunking (article-aware)
- Attach metadata and version
- (Simulated) Embedding generation OR real embedding+index build via prepare_pipeline
- Save knowledge manifest for rollback/versioning
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from src.legal_ai.ingestion.normalization import normalize_arabic
from src.legal_ai.ingestion.validation import validate_documents, hash_document
from src.legal_ai.ingestion.chunker import article_aware_chunk
from src.legal_ai.knowledge.versioning import KnowledgeVersion

# If user asks for real run, we'll import prepare_pipeline lazily


def load_documents(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and "documents" in data:
        docs = data["documents"]
    elif isinstance(data, list):
        docs = data
    else:
        raise RuntimeError("Input JSON must be a list of documents or contain a 'documents' key")
    return docs


def normalize_and_chunk(docs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for doc in docs:
        # Apply minimal canonical fields
        raw = str(doc.get("raw_text") or doc.get("content") or "").strip()
        if not raw:
            continue
        normalized = normalize_arabic(raw)
        chunks = article_aware_chunk({**doc, "raw_text": raw})
        for i, c in enumerate(chunks):
            canonical = {
                "document_id": doc.get("document_id") or doc.get("law_id") or f"doc-{int(time.time())}-{i}",
                "jurisdiction": doc.get("jurisdiction", "unknown"),
                "law_id": doc.get("law_id", doc.get("document_id", "unknown")),
                "law_name": doc.get("law_name", "unknown"),
                "article_id": c.get("article_id", str(i)),
                "raw_text": c.get("content", raw)[:10000],
                "normalized_text": normalize_arabic(c.get("content", raw))[:10000],
                "embedding_text": c.get("content", raw)[:10000],
                "metadata": doc.get("metadata", {}),
            }
            out.append(canonical)
    return out


def simulate_embeddings(docs: list[dict[str, Any]], dim: int = 1536) -> np.ndarray:
    rng = np.random.default_rng(seed=42)
    return rng.standard_normal((len(docs), dim)).astype(np.float32)


def save_documents_and_manifest(docs: list[dict[str, Any]], out_dir: Path, embeddings: np.ndarray | None, version_id: str | None = None):
    out_dir.mkdir(parents=True, exist_ok=True)
    normalized_path = out_dir / "documents_normalized.json"
    normalized_path.write_text(json.dumps(docs, ensure_ascii=False, indent=2), encoding="utf-8")

    if embeddings is not None:
        emb_dir = out_dir / "embeddings"
        emb_dir.mkdir(parents=True, exist_ok=True)
        emb_path = emb_dir / "dense_embeddings.npy"
        np.save(emb_path, embeddings)

    # compute dataset hash
    h = hash_document({"raw_text": json.dumps([d.get("raw_text","") for d in docs])})
    kv = KnowledgeVersion(
        version_id=version_id or f"v{int(time.time())}",
        dataset_hash=h,
        document_count=len(docs),
        embedding_model="SIMULATED" if embeddings is not None else "NONE",
    )
    kv.save(out_dir / "knowledge_version.json")
    return kv


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="input JSON file with documents or {documents:[...]} structure")
    p.add_argument("--out", required=True, help="output artifacts dir")
    p.add_argument("--simulate", action="store_true", help="simulate embeddings (fast)")
    p.add_argument("--real", action="store_true", help="run real embedding/index build (may be slow)")
    p.add_argument("--dim", type=int, default=1536, help="embedding dim for simulation")
    args = p.parse_args()

    inp = Path(args.input)
    if not inp.exists():
        raise SystemExit(f"Input file not found: {inp}")
    docs_raw = load_documents(inp)

    try:
        validate_documents(docs_raw)
    except Exception as exc:
        raise SystemExit(f"Validation failed: {exc}")

    docs_norm = normalize_and_chunk(docs_raw)
    out_dir = Path(args.out)

    if args.simulate:
        embeddings = simulate_embeddings(docs_norm, dim=args.dim)
        kv = save_documents_and_manifest(docs_norm, out_dir, embeddings, version_id=None)
        print("Simulated knowledge build complete:", kv.version_id)
        print("Artifacts saved to", out_dir)
        return

    if args.real:
        # Real pipeline: call prepare_pipeline to build encoder and index
        from src.legal_ai.retrieval.pipeline import prepare_pipeline
        from src.legal_ai.core.models import RuntimeConfig, PipelineConfig

        runtime = RuntimeConfig(device="cpu", precision="auto", dense_batch_size=8, num_threads=2)
        pipeline_cfg = PipelineConfig()
        retriever, info = prepare_pipeline(docs_norm, runtime, pipeline_cfg, out_dir, load_reranker=True)
        print("Real pipeline complete. Runtime info saved to", out_dir / "reports" / "runtime_info.json")
        return

    raise SystemExit("Specify --simulate or --real")


if __name__ == "__main__":
    main()
