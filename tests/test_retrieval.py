from __future__ import annotations

import unittest
import numpy as np
from legal_rag_engine import BM25, DenseIndex, RetrievalHit, HybridRetriever


class TestRetrievalComponents(unittest.TestCase):
    def test_bm25_basic(self):
        corpus = [["قانون", "المادة", "عقوبات"], ["مدني", "عقد", "التزام"], ["جنائي", "جريمة", "قانون"]]
        bm25 = BM25(corpus)
        top = bm25.top_n("قانون", 2)
        # ensure indices returned and score for first > second (if multiple)
        self.assertTrue(len(top) >= 1)
        idx, score = top[0]
        self.assertIn(idx, {0, 2})
        self.assertGreaterEqual(score, 0)

    def test_denseindex_search(self):
        # create simple 3-d embeddings
        emb = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
        idx = DenseIndex(emb)
        q = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
        scores, indices = idx.search(q, 1)
        self.assertEqual(indices.shape[1], 1)
        self.assertEqual(int(indices[0, 0]), 0)

    def test_dense_preserving_union(self):
        # Prepare dummy RetrievalHit objects
        d1 = RetrievalHit(id="1", index=0, content="A", metadata={}, dense_score=0.9, dense_rank=1, sources=["dense"]) 
        d2 = RetrievalHit(id="2", index=1, content="B", metadata={}, dense_score=0.8, dense_rank=2, sources=["dense"]) 
        b1 = RetrievalHit(id="2", index=1, content="B", metadata={}, bm25_score=1.0, bm25_rank=1, sources=["bm25"]) 
        b2 = RetrievalHit(id="3", index=2, content="C", metadata={}, bm25_score=0.5, bm25_rank=2, sources=["bm25"]) 
        merged = HybridRetriever.dense_preserving_union([d1, d2], [b1, b2], max_candidates=10)
        # dense items should be first in order
        self.assertGreaterEqual(len(merged), 3)
        self.assertEqual(merged[0].id, "1")
        self.assertEqual(merged[1].id, "2")
        # unique bm25 item appended after dense
        ids = [h.id for h in merged]
        self.assertIn("3", ids)


if __name__ == "__main__":
    unittest.main()
