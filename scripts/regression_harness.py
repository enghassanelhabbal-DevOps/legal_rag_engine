"""Lightweight regression harness for retrieval baselines.

Saves a small baseline (BM25 + dense synthetic) to artifacts/regression_baseline.json
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

from legal_rag_engine import BM25, DenseIndex


def run():
    # Synthetic corpus
    docs = [
        {"id": "d1", "content": "المادة 1: شروط القبض في حالة التلبس"},
        {"id": "d2", "content": "المادة 2: حقوق المتهم"},
        {"id": "d3", "content": "المادة 3: إجراءات القبض"},
    ]
    corpus_tokens = [["شروط", "القبض", "تلبس"], ["حقوق", "متهم"], ["اجراءات", "القبض"]]
    bm25 = BM25(corpus_tokens)
    q = "ما شروط القبض في حالة التلبس؟"
    bm25_top = bm25.top_n(q, 3)

    # synthetic dense embeddings: make d1 closest to query
    emb = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.5]], dtype=np.float32)
    idx = DenseIndex(emb)
    q_emb = np.array([[1.0, 0.0]], dtype=np.float32)
    scores, indices = idx.search(q_emb, 3)

    baseline = {
        "query": q,
        "bm25_top": [(int(i), float(s)) for i, s in bm25_top],
        "dense_indices": indices.tolist(),
        "dense_scores": scores.tolist(),
    }
    out = Path("artifacts") / "reports"
    out.mkdir(parents=True, exist_ok=True)
    (out / "regression_baseline.json").write_text(json.dumps(baseline, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Saved regression baseline to artifacts/reports/regression_baseline.json")


if __name__ == '__main__':
    run()
