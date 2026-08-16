Repository audit — legal_rag_engine (Phase 1)

Summary (quick)
---------------
- Repository root: c:\Users\H Elhabbal\Downloads\legal_rag_engine
- Key existing files:
  - legal_rag_engine.py        (main retrieval pipeline implementation)
  - qwen_transformers_backend.py (LLM Transformers backend tuned for M2200)
  - app_qwen_m2200.py         (example CLI orchestration using RAG pipeline + Qwen)
  - llm_backend_template.py   (LLM backend Protocol / adapter template)
  - check_gpu.py              (environment GPU diagnostic)
  - README_portable_windows_linux.md, README_qwen_m2200.md, INSTALL_GPU.md
- A virtual environment .venv exists in the repo (do NOT commit).

What works today (observed)
---------------------------
- There is a working retrieval pipeline implemented inside legal_rag_engine.py that provides:
  - BM25 implementation
  - Dense encoder using SentenceTransformers
  - FAISS index usage (CPU by default)
  - Reranker integration
  - Retrieval -> context builder -> LLM integration path
- A Qwen Transformers backend exists and is tuned for Quadro M2200 memory constraints.
- Example app (app_qwen_m2200.py) demonstrates full flow: ingest -> retrieve -> build context -> unload retrieval -> load LLM -> generate.

Gaps vs Master Engineering Prompt (high level)
---------------------------------------------
- Missing packaged modules and clear core interfaces: there is no legal_ai/ package yet.
- No config profiles (.env.example, config.dev.yaml, etc.).
- No knowledge versioning artifacts, embedding cache manifest, or reproducible indexing pipeline.
- No test suite or CI workflow in repo.
- No API (FastAPI) skeleton.
- Limited evaluation framework and metrics logging.

Risks / constraints discovered
-----------------------------
- A .venv is present in repo root; ensure it is not committed to future changes and add .gitignore (if repo will be versioned).
- Many heavy dependencies are in venv; Phase 1 will avoid installing or modifying runtime dependencies.
- Files currently directly execute heavy model loads (careful when running on small VRAM devices).

Phase 1 actions performed now
-----------------------------
- Added legal_ai/ARCHITECTURE.md  (this high-level architecture map)
- Added legal_ai/REPO_AUDIT.md    (this audit summary)
- Added scripts/repo_audit.py     (lightweight script to print key files and basic checks)

Planned Phase 1 next actions (if user OK)
-----------------------------------------
- Run scripts/repo_audit.py to collect runtime facts (optional; non-destructive).
- Create .env.example and config loader skeleton.
- Create legal_ai/core.py with stable dataclasses and Protocols that wrap current pipeline types while preserving compatibility.
- Create tests/ directory and add a small normalization unit test.

Requested decisions from user
-----------------------------
1) Proceed to create legal_ai/core.py skeleton now (safe, minimal, will not break existing scripts)?
2) Or run the audit script and a smoke test of current pipeline first (requires careful runtime checks)?

If no reply, next default will be: create legal_ai/core.py (minimal interfaces) and .env.example so Phase 2 can start.
