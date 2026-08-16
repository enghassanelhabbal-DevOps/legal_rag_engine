Master Engineering Prompt — Phase 1 architecture map

Goal
----
Phase 1: repository audit + architecture map for evolving the existing legal_rag_engine into a production-grade, LLM-agnostic Legal AI Platform (see supplied Master Engineering Prompt).

High-level architecture (one-page)
----------------------------------
USER QUERY -> Query Understanding -> (Dense Retrieval | Lexical Retrieval) -> Candidate Generation -> Metadata/Temporal/Jurisdiction Filters -> Candidate Fusion -> Reranker -> Evidence Selection -> Grounded Context Builder -> LLM Adapter -> Structured Answer + Citations

Key modules (planned)
----------------------
- legal_ai/core: core dataclasses and interfaces (PipelineConfig, RuntimeConfig, RetrievalHit, LLMBackend protocol, RAGService)
- legal_ai/config: configuration loaders (.env example, YAML profiles)
- legal_ai/ingestion: parsers, OCR wrappers, normalizers, article-aware chunkers, metadata extraction, versioning
- legal_ai/normalization: Arabic normalization utilities and configurable analyzers
- legal_ai/metadata: legal-aware metadata models (jurisdiction, law_id, article_id, effective_date, repeal_date, versioning)
- legal_ai/retrieval: dense encoder wrappers, FAISS index management, BM25 implementation, candidate generation and fusion
- legal_ai/reranking: cross-encoder/reranker wrapper with batch inference
- legal_ai/knowledge: knowledge versioning, embedding cache, dataset manifest
- legal_ai/generation: grounded prompt templates, LLM adapters, prompt-safety and injection defenses
- legal_ai/models: model adapter implementations (transformers, Ollama, OpenAI-compatible)
- legal_ai/storage: persistent storage abstraction for indexes, embeddings, artifacts
- legal_ai/evaluation: retrieval + generation metrics, regression tests, datasets
- legal_ai/monitoring: metrics and tracing hooks (Prometheus/OpenTelemetry abstractions)
- app/: example CLI apps (app_qwen_m2200.py -> move to app/) and API adapters
- api/: REST API server skeleton (FastAPI) with health/metrics/query endpoints
- tests/: unit and integration tests

Constraints & policies
----------------------
- Target Windows 10/11 + Linux. Use CPU FAISS by default on Windows.
- GPU use: embeddings, reranking, LLM inference only. Explicit load/unload/offload strategies.
- Do NOT store legal knowledge in LLM weights. Implement knowledge versioning and reproducible indexing.
- Arabic normalization must be configurable and reversible for preserving exact legal text.

Deliverables for Phase 1
------------------------
1. Repository audit report (this file references the report in REPO_AUDIT.md).
2. Architecture map (this file).
3. Minimal non-invasive scaffolding (docs + lightweight audit script) to allow Phase 2 to start safely.

Next steps (Phase 2 preview)
----------------------------
- Implement core abstractions in legal_ai/core.py, keeping compatibility wrappers for current legal_rag_engine.py to avoid breaking existing scripts.
- Add configuration loader and .env.example.
- Add tests for normalization and small retrieval smoke tests.

Notes
-----
This is an initial architecture map for Phase 1. Implementation will be incremental and non-destructive: existing entry points (legal_rag_engine.py, app_qwen_m2200.py) will keep working while the refactor introduces the new package and adapters.

(English/Arabic bilingual comments can be added where helpful during Phase 2.)
