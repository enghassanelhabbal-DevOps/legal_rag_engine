from __future__ import annotations

from pathlib import Path
from typing import Any

from legal_rag_engine import (
    PipelineConfig,
    RuntimeConfig,
    build_grounded_context,
    load_json,
    prepare_pipeline,
    validate_documents,
)


class RAGService:
    """High-level service wrapper providing a stable interface for retrieval and context building.

    This class is a thin wrapper around the existing retrieval implementation in
    legal_rag_engine.py to provide a package-style API for the refactor.
    """

    def __init__(self, documents: list[dict[str, Any]], runtime: RuntimeConfig, pipeline_cfg: PipelineConfig, artifact_dir: Path, load_reranker: bool = True):
        validate_documents(documents)
        self.documents = documents
        self.runtime = runtime
        self.pipeline_cfg = pipeline_cfg
        self.artifact_dir = artifact_dir
        self.retriever, self.runtime_info = prepare_pipeline(documents, runtime, pipeline_cfg, artifact_dir, load_reranker=load_reranker)

    @classmethod
    def from_documents_file(cls, path: Path, runtime: RuntimeConfig, pipeline_cfg: PipelineConfig, artifact_dir: Path, load_reranker: bool = True) -> RAGService:
        docs = load_json(path)
        return cls(docs, runtime, pipeline_cfg, artifact_dir, load_reranker=load_reranker)

    def retrieve(self, query: str, top_k: int = 5) -> dict[str, Any]:
        return self.retriever.retrieve(query, top_k=top_k)

    def build_context(self, results: list[dict[str, Any]], max_chars: int) -> str:
        return build_grounded_context(results, max_chars=max_chars)

    def close(self) -> None:
        # best-effort resource release
        try:
            if getattr(self.retriever, "encoder", None) is not None:
                try:
                    self.retriever.encoder.model.to("cpu")
                    del self.retriever.encoder.model
                except Exception:
                    pass
            if getattr(self.retriever, "reranker", None) is not None:
                try:
                    self.retriever.reranker.model.model.to("cpu")
                    del self.retriever.reranker.model
                except Exception:
                    pass
        finally:
            import gc

            import torch

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


__all__ = ["RAGService"]
