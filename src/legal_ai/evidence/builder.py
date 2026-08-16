"""builder.py — Evidence selection and grounded context construction.

Extracted from legal_rag_engine.py (build_grounded_context).
Responsibility: turn ranked retrieval hits into a compact, cited LLM context.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

from src.legal_ai.core.contracts import RetrievalResult


def build_grounded_context(
    results: Sequence[Dict[str, Any]],
    max_chars: int = 14_000,
) -> str:
    """Build the grounded context string passed to the LLM.

    Iterates results in order, appending SOURCE blocks until *max_chars* is
    reached.  Each block references its source so the LLM can produce citations.

    Args:
        results: Ordered retrieval result dicts (as returned by HybridRetriever.retrieve).
        max_chars: Hard character budget for the context window.

    Returns:
        A formatted string with numbered [SOURCE N] blocks.
    """
    blocks: List[str] = []
    current = 0
    for i, item in enumerate(results, start=1):
        metadata = item.get("metadata") or {}
        title = " ".join(str(metadata.get("title", "")).split()) or "مصدر قانوني"
        content = " ".join(str(item.get("content", "")).split())
        block = (
            f"[SOURCE {i}]\n"
            f"العنوان: {title}\n"
            f"النص: {content}\n"
            f"معرّف المصدر: {item.get('id')}"
        )
        if current + len(block) > max_chars:
            break
        blocks.append(block)
        current += len(block)
    return "\n\n".join(blocks)


def select_evidence(
    results: Sequence[Dict[str, Any]],
    max_chars: int = 14_000,
    min_score: float = 0.0,
) -> List[Dict[str, Any]]:
    """Select and filter evidence items from retrieval results.

    Filters out items below *min_score* and respects *max_chars* budget.
    Returns a list of result dicts that will be used in context building.
    """
    selected: List[Dict[str, Any]] = []
    current = 0
    for item in results:
        score = item.get("reranker_score") or item.get("dense_score") or 0.0
        if float(score) < min_score:
            continue
        content = str(item.get("content", ""))
        if current + len(content) > max_chars:
            break
        selected.append(item)
        current += len(content)
    return selected


__all__ = ["build_grounded_context", "select_evidence"]
