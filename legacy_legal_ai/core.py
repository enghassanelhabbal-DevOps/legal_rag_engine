from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class RuntimeConfigLite:
    """Lightweight runtime config used during refactor. Kept intentionally
    minimal to avoid conflicting with existing RuntimeConfig in the repo.
    """
    device: str = "auto"
    gpu_id: int = 0
    precision: str = "auto"
    dense_batch_size: int = 32
    rerank_batch_size: int = 32


@dataclass
class PipelineConfigLite:
    dense_candidates: int = 30
    bm25_candidates: int = 10
    rerank_candidates: int = 40
    final_k: int = 5
    rerank_max_chars: int = 3500
    alpha: float = 0.75
    max_context_chars: int = 14000


@dataclass
class RetrievalHitLite:
    id: str
    index: int
    content: str
    metadata: dict[str, Any]
    dense_score: float | None = None
    bm25_score: float | None = None
    reranker_score: float | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class LLMBackend(Protocol):
    def load(self) -> None: ...
    def unload(self) -> None: ...
    def generate(self, query: str, context: str) -> str: ...
    def info(self) -> dict[str, Any]: ...


class RAGService(Protocol):
    """Minimal RAGService protocol for the refactor bootstrap.
    Concrete implementation should provide retrieve(), index(), and ingest().
    """
    def retrieve(self, query: str, top_k: int = 5) -> dict[str, Any]: ...
    def ingest(self, documents: Sequence[dict[str, Any]]) -> None: ...
    def index(self) -> None: ...


__all__ = [
    "RuntimeConfigLite",
    "PipelineConfigLite",
    "RetrievalHitLite",
    "LLMBackend",
    "RAGService",
]
