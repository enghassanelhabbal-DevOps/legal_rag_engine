"""models.py — Runtime/pipeline configs and shared protocols.

Migrated and unified from the old legal_ai/core.py (RuntimeConfigLite,
PipelineConfigLite, RetrievalHitLite) + legal_rag_engine.py (RuntimeConfig,
PipelineConfig).  The 'Lite' suffix is dropped; these are now the canonical classes.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional, Protocol, Sequence


# ---------------------------------------------------------------------------
# Runtime configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RuntimeConfig:
    """Hardware / runtime settings.

    Hardware target (ARCHITECTURE_CONTRACT.md):
      Windows 10, NVIDIA Quadro M2200, 4 GB VRAM, Compute Capability 5.2, Python 3.12.
    """

    device: str = "auto"           # "auto" | "cpu" | "cuda"
    gpu_id: int = 0
    precision: str = "auto"        # "auto" | "fp32" | "fp16" | "bf16"
    dense_batch_size: int = 32
    rerank_batch_size: int = 32
    num_threads: int = 0           # 0 = PyTorch default
    enable_tf32: bool = True
    compile_reranker: bool = False  # torch.compile — off by default (CC 5.2)
    max_seq_length: int = 1024


# ---------------------------------------------------------------------------
# Pipeline / retrieval configuration
# ---------------------------------------------------------------------------

@dataclass
class PipelineConfig:
    """Hyper-parameters for the retrieval → reranking → evidence pipeline."""

    dense_candidates: int = 30     # BGE-M3 top-k
    bm25_candidates: int = 10      # BM25 supplementary candidates
    rerank_candidates: int = 40    # candidates passed to reranker
    final_k: int = 5               # candidates returned after evidence selection
    rerank_max_chars: int = 3_500  # max chars per candidate for reranker
    alpha: float = 0.75            # dense weight in dense-preserving fusion
    max_context_chars: int = 14_000  # max chars of grounded context for LLM


# ---------------------------------------------------------------------------
# Shared hit type
# ---------------------------------------------------------------------------

@dataclass
class RetrievalHit:
    """Internal representation of a retrieval candidate (pre-contract conversion)."""

    id: str
    index: int
    content: str
    metadata: Dict[str, Any]
    dense_score: Optional[float] = None
    bm25_score: Optional[float] = None
    reranker_score: Optional[float] = None

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Protocols (structural typing — no runtime overhead)
# ---------------------------------------------------------------------------

class LLMBackend(Protocol):
    """Protocol every LLM adapter must satisfy."""

    def load(self) -> None: ...
    def unload(self) -> None: ...
    def generate(self, query: str, context: str) -> str: ...
    def info(self) -> Dict[str, Any]: ...


class RAGServiceProtocol(Protocol):
    """Protocol for the high-level RAG service."""

    def retrieve(self, query: str, top_k: int = 5) -> Dict[str, Any]: ...
    def ingest(self, documents: Sequence[Dict[str, Any]]) -> None: ...
    def index(self) -> None: ...


__all__ = [
    "RuntimeConfig",
    "PipelineConfig",
    "RetrievalHit",
    "LLMBackend",
    "RAGServiceProtocol",
]
