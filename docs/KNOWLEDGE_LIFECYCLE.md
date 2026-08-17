Knowledge lifecycle & retrieval pipeline — mapping to repository

Goal
----
This document maps the operational workflow you described to the code in this repository and shows the concrete local commands to test everything in WSL first.

Workflow (user-facing retrieval)
--------------------------------
User Query
  ↓
Query Understanding
  ↓
Jurisdiction / Date / Filters
  ↓
Dense Retrieval (BGE-M3)
+
Lexical Retrieval (BM25)
  ↓
Candidate Union / Fusion
  ↓
Legal Metadata Filtering
  ↓
Cross-Encoder Reranker
  ↓
Evidence Selection
  ↓
Grounded Context Builder
  ↓
LLM Adapter (Qwen / API / Other)
  ↓
Structured Answer
  ↓
Citations + Evidence + Metrics

Where in the code
------------------
- Query Understanding, Jurisdiction/Date/Filters: src/legal_ai/services/query_service.py — orchestrates query processing, builds context, and applies metadata filters
- Dense Retrieval (BGE-M3): src/legal_ai/retrieval/dense.py and pipeline.prepare_pipeline — DenseEncoder loads the dense model and produces embeddings
- Lexical Retrieval (BM25): src/legal_ai/retrieval/bm25.py and ingestion.normalization.tokenize
- Candidate union & hybrid retrieval: src/legal_ai/retrieval/hybrid.py and retrieval.pipeline.prepare_pipeline
- Cross-Encoder Reranker: src/legal_ai/reranking (Reranker) and pipeline (optional lazy load)
- Evidence selection & grounded context: src/legal_ai/evidence.build_grounded_context and select_evidence
- LLM Adapter: src/legal_ai/generation/manager.py and generation/backends/* — adapts local Qwen or cloud providers
- Structured answer / citations: query_service.answer() returns answer + citations + timing

Knowledge lifecycle (ingestion → publish)
-----------------------------------------
When laws change we follow a knowledge-versioned pipeline (no LLM retraining):

1) New law / modification received
   ↓
2) Ingestion (validate, normalize, chunk)
   - src/legal_ai/ingestion/validation.py
   - src/legal_ai/ingestion/normalization.py
   - src/legal_ai/ingestion/chunker.py

3) Validation (schema checks, dedup/hash)
   - src/legal_ai/ingestion/validation.py

4) Normalization (Arabic normalization/tokenization)
   - src/legal_ai/ingestion/normalization.py

5) Metadata + Versioning
   - src/legal_ai/knowledge/versioning.py (KnowledgeVersion)

6) Embedding
   - src/legal_ai/retrieval/dense.DenseEncoder (uses DENSE_MODEL_NAME)
   - Embedding cache: src/legal_ai/knowledge/cache.py

7) Index update
   - src/legal_ai/retrieval/pipeline.build_index — builds/updates FAISS index and saves embeddings/index under artifacts directory

8) Evaluation
   - scripts/quality_gate.py and tests/test_regression_quality.py

9) Publish Knowledge Release
   - Save knowledge_version.json manifest under artifacts/ (ready for rollback and release tagging)

Repository helpers
------------------
- scripts/update_knowledge.py
  - Fast simulate mode: generate simulated embeddings and save a knowledge_version manifest (quick local smoke test)
  - Real mode: runs prepare_pipeline which builds encoder, embeddings, FAISS index and saves reports (slow; requires model download/CPU/GPU)

- src/legal_ai/knowledge/versioning.py
  - list_versions(), rollback_to() help discover and restore earlier knowledge releases

Local test plan (WSL) — end-to-end (fast smoke test)
--------------------------------------------------
1) Prepare environment in WSL (Ubuntu 22.04/24.04 recommended)
   - Install Docker and Docker Compose OR ensure python deps available

2) Create a sample new law json (example new_laws/sample_new_law.json)
   {
     "documents": [
       {
         "document_id": "law-2026-xyz",
         "jurisdiction": "EG",
         "law_id": "law-2026-xyz",
         "law_name": "قانون تجريبي",
         "raw_text": "نص قانون جديد يتعلق بالمسؤولية الجنائية..."
       }
     ]
   }

3) Simulated quick run (no heavy downloads)
   python scripts/update_knowledge.py --input new_laws/sample_new_law.json --out artifacts_local --simulate
   - Expected: artifacts_local/documents_normalized.json and artifacts_local/embeddings/dense_embeddings.npy and artifacts_local/knowledge_version.json

4) Run Streamlit UI + API locally (see WSL_TESTING.md)
   - Start API: docker compose -f docker-compose.example.yml up -d --build
   - Set LEGAL_API_URL=http://localhost:8000/v1/query and API_KEY=test-key
   - Run Streamlit: python -m streamlit run app.py --server.port 8501
   - Query via UI selecting Provider=Custom API

5) Real run (full embedding + index — can be slow)
   python scripts/update_knowledge.py --input new_laws/sample_new_law.json --out artifacts_real --real
   - Requires dense encoder model downloads and CPU cycles

6) Run regression/evaluation
   python -m pytest tests/test_regression_quality.py
   python scripts/quality_gate.py --artifacts artifacts_real

Operational notes
-----------------
- KnowledgeVersion manifests are saved alongside artifacts and are used for rollback (no retraining required).
- When answering historical queries, QueryService should consult the KnowledgeVersion manifests and pick the version whose created_at / dataset_hash applies to the requested date (this repo stores version metadata for this purpose).
- The architecture supports publishing a new knowledge release by copying artifacts to a versioned directory (e.g., artifacts/releases/v20260817) and saving the manifest.

Next steps (optional automation)
--------------------------------
- Add hooks to automatically run the ingestion pipeline when new law documents are pushed to a repo or S3 (e.g., GitHub Actions or serverless function triggers). The CI should run quality_gate and block publishing if metrics drop.
- Integrate MLflow to track embedding model hashes and artifact versions.
- Add a small admin UI to create and view Knowledge Releases and rollbacks.

If you want, I can:
- Add a sample new_laws/sample_new_law.json to the repo
- Run the simulated update_knowledge pipeline here and show the produced artifacts (I can run scripts locally in WSL only on your machine; I can show commands and parse outputs you paste)
- Add an admin endpoint in api/app.py to list knowledge versions and trigger ingestion (secured by API_KEY)

Tell me which of the above I should do next (suggestion: add sample input + run simulate and add admin endpoint).