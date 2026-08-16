"""retrieval sub-package — BGE-M3 dense retrieval, FAISS, BM25, candidate fusion.

Public API:
    from src.legal_ai.retrieval import DenseEncoder, DenseIndex, BM25
    from src.legal_ai.retrieval import HybridRetriever
    from src.legal_ai.retrieval import build_index, prepare_pipeline
"""

from src.legal_ai.retrieval.bm25 import BM25
from src.legal_ai.retrieval.dense import DenseEncoder, DenseIndex
from src.legal_ai.retrieval.hybrid import HybridRetriever
from src.legal_ai.retrieval.pipeline import build_index, prepare_pipeline

__all__ = [
    "DenseEncoder",
    "DenseIndex",
    "BM25",
    "HybridRetriever",
    "build_index",
    "prepare_pipeline",
]
