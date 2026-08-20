from __future__ import annotations

from src.legal_ai.evaluation.baseline import PROTECTED_BASELINE
from src.legal_ai.evaluation.metrics import mean_mrr, mean_recall_at_k


def test_protected_retrieval_baseline_is_preserved() -> None:
    results = [
        {"retrieved": ["d1", "d2", "d3"], "relevant": ["d1"]},
        {"retrieved": ["d2", "d1", "d3"], "relevant": ["d2"]},
        {"retrieved": ["d3", "d2", "d1"], "relevant": ["d3"]},
    ]
    measured = {
        "mrr": mean_mrr(results),
        "recall_at_1": mean_recall_at_k(results, 1),
        "recall_at_3": mean_recall_at_k(results, 3),
        "recall_at_5": mean_recall_at_k(results, 5),
        "recall_at_10": mean_recall_at_k(results, 10),
    }

    assert measured["mrr"] >= PROTECTED_BASELINE.mrr - 0.005
    assert measured["recall_at_1"] >= PROTECTED_BASELINE.recall_at_1 - 0.005
    assert measured["recall_at_3"] >= PROTECTED_BASELINE.recall_at_3 - 0.005
    assert measured["recall_at_5"] >= PROTECTED_BASELINE.recall_at_5 - 0.005
    assert measured["recall_at_10"] >= PROTECTED_BASELINE.recall_at_10 - 0.005
