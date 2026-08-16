"""baseline.py — Protected retrieval baseline definition.

The values here are the LOCKED baseline from ARCHITECTURE_CONTRACT.md.
Any change to retrieval or reranking must be measured against these numbers
and the results reported before merging.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RetrievalBaseline:
    """Snapshot of known-good retrieval performance metrics."""

    mrr: float
    recall_at_1: float
    recall_at_3: float
    recall_at_5: float
    recall_at_10: float
    description: str = ""

    def as_dict(self) -> dict[str, float]:
        return {
            "MRR": self.mrr,
            "Recall@1": self.recall_at_1,
            "Recall@3": self.recall_at_3,
            "Recall@5": self.recall_at_5,
            "Recall@10": self.recall_at_10,
        }

    def check(self, measured: RetrievalBaseline, tol: float = 0.005) -> bool:
        """Return True if *measured* is within *tol* of this baseline on all metrics."""
        return all([
            measured.mrr >= self.mrr - tol,
            measured.recall_at_1 >= self.recall_at_1 - tol,
            measured.recall_at_3 >= self.recall_at_3 - tol,
            measured.recall_at_5 >= self.recall_at_5 - tol,
            measured.recall_at_10 >= self.recall_at_10 - tol,
        ])


# LOCKED — do not modify without before/after evidence (ARCHITECTURE_CONTRACT.md)
PROTECTED_BASELINE = RetrievalBaseline(
    mrr=0.835,
    recall_at_1=0.75,
    recall_at_3=0.90,
    recall_at_5=0.95,
    recall_at_10=1.00,
    description="BGE-M3 + BM25 + BGE-reranker-v2-m3, dense-preserving fusion, alpha=0.75",
)

__all__ = ["RetrievalBaseline", "PROTECTED_BASELINE"]
