Legal RAG Engine - Deployment & DevOps Runbook

This runbook contains step-by-step instructions for deploying, testing, and operating the Legal RAG Engine in production-like environments (Streamlit Cloud / Linux servers). It also contains recipes for enabling local inference with Qwen when running on a dedicated Linux host.

Contents
- Streamlit Cloud deployment
- Environment variables and secrets
- Provider health checks
- CI / GitHub Actions checklist
- DVC remote and rollback
- Running FastAPI backend (Custom API) in Docker
- Local Qwen on Linux: Docker recipe and notes
- Monitoring and logging
- Troubleshooting


1. Streamlit Cloud deployment
-----------------------------
1. Repository: github.com/enghassanelhabbal-DevOps/legal_rag_engine
2. Branch: agents/devops-mlops-streamlit-integration
3. Main module (entrypoint): app.py
4. Python: use a supported version compatible with requirements.txt (3.10-3.12 recommended)
5. Requirements: ensure requirements.txt present at repository root
6. Environment variables: add the secrets below (see section "Environment variables and secrets")
7. ALLOW_LOCAL_MODEL_RUNTIME must remain "0" (default) on Streamlit Cloud unless you intentionally run local models in a dedicated Linux host.

After filling secrets and selecting branch, click Deploy.


2. Environment variables and secrets
-----------------------------------
Add the following secrets in Streamlit Cloud or GitHub Secrets (names are case-sensitive):

- GOOGLE_API_KEY: API key for Google Generative Language (Gemini)
- GEMINI_MODEL: e.g., gemini-2.5-flash (default fallback if not set in UI)
- OPENAI_API_KEY: optional for OpenAI provider
- OPENAI_MODEL: optional default model (e.g., gpt-5-mini or gpt-4o-mini)
- HF_TOKEN: optional Hugging Face token for private model downloads
- ALLOW_LOCAL_MODEL_RUNTIME: "0" on Cloud; set to "1" only on controlled Linux host
- LEGAL_API_URL: optional FastAPI backend URL (for Custom API provider)
- DVC_REMOTE_URL: e.g., gs://your-bucket/path or s3://bucket/path (for DVC remote storage)
- MLFLOW_TRACKING_URI: optional MLflow server URI
- SENTRY_DSN: optional DSN for Sentry
- LOGFLARE_SOURCE_TOKEN: optional token for Logflare ingestion

Set the same secrets in GitHub repository secrets for CI usage.


3. Provider health checks (quick)
--------------------------------
- Google list models probe:
  GET https://generativelanguage.googleapis.com/v1beta/models?key=GOOGLE_API_KEY

- Google generation quick-test:
  POST https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key=GOOGLE_API_KEY
  Body example:
  {
    "contents": [{"parts": [{"text": "Test: اجب بالعربية: ما حكم التلبس؟"}]}],
    "generationConfig": {"maxOutputTokens": 80, "temperature": 0.2}
  }

- OpenAI minimal probe:
  GET https://api.openai.com/v1/models  (Authorization: Bearer OPENAI_API_KEY)


4. CI / GitHub Actions checklist
--------------------------------
- Required checks for PRs:
  - ruff or flake8 lint
  - mypy type checks
  - pytest (unit tests + regression quality tests)
  - scripts/quality_gate.py (enforces retrieval metric thresholds)

- Recommended: create a protected branch policy requiring all checks to pass before merge.


5. DVC remote and rollback
--------------------------
- Use cloud object storage as DVC remote (GCS or S3). Example:
  dvc remote add -d storage gs://my-org-legal-rag/models
  dvc remote modify storage credentialpath /path/to/key.json

- Common pattern to rollback an artifact:
  git checkout <commit-or-tag>
  dvc checkout


6. Running FastAPI backend (Custom API) in Docker
------------------------------------------------
Example Dockerfile (placed in api/Dockerfile):

FROM python:3.10-slim
WORKDIR /app
COPY ./api/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt
COPY ./api /app
EXPOSE 8000
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]

Run locally:
  docker build -t legal-rag-api:latest -f api/Dockerfile .
  docker run -p 8000:8000 -e LEGAL_API_KEY="..." legal-rag-api:latest

Then in the Streamlit UI choose provider=Custom API and set LEGAL_API_URL to http://<host>:8000/query


7. Local Qwen on Linux: Docker recipe and notes
----------------------------------------------
Notes:
- Local Qwen models + FAISS/dense encoder require native dependencies and are not supported on Streamlit Cloud.
- Use a dedicated Linux VM (Ubuntu 22.04 LTS recommended) with sufficient disk and optionally GPU drivers.

Example dev Dockerfile (host must provide GPU drivers for GPU usage):

# Dev Dockerfile for local Qwen runtime (CPU-based)
FROM python:3.10-slim
WORKDIR /app
RUN apt-get update && apt-get install -y build-essential git wget curl libsndfile1
COPY requirements-local.txt /app/requirements-local.txt
RUN pip install --no-cache-dir -r /app/requirements-local.txt
COPY ./api /app
EXPOSE 8000
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]

Requirements-local should include faiss-cpu (or faiss-gpu), transformers, accelerate, sentence-transformers, bge-client (if used), and your local model loader.

Run:
  docker build -t legal-rag-local:dev -f api/Dockerfile .
  docker run --rm -p 8000:8000 -e ALLOW_LOCAL_MODEL_RUNTIME=1 legal-rag-local:dev

After bringing up this backend, set LEGAL_API_URL in Streamlit to http://<host-ip>:8000


8. Monitoring and logging
-------------------------
Minimum: structured logging to STDOUT in JSON format, capturing:
- request_id, timestamp, provider, model, retrieval_ms, generation_ms, error (if any)

Optional integrations:
- Sentry: use sentry-sdk.init(dsn=SENTRY_DSN)
- Logflare: send logs via HTTP using LOGFLARE_SOURCE_TOKEN

Example log record:
{"ts":"2026-08-17T09:00:00Z","request_id":"abc123","provider":"Google Gemini","model":"gemini-2.5-flash","retrieval_ms":12,"generation_ms":210,"error":null}


9. Troubleshooting
------------------
- If Streamlit fails to start on Cloud: verify main module = app.py and requirements.txt is correct.
- If provider returns 403/401: check API key and model name.
- If local runtime errors: ensure ALLOW_LOCAL_MODEL_RUNTIME=1 only on Linux host with required native libs and drivers.


Appendix: Useful commands
------------------------
- Run Streamlit locally:
  export GOOGLE_API_KEY="..."
  export GEMINI_MODEL="gemini-2.5-flash"
  export ALLOW_LOCAL_MODEL_RUNTIME="0"
  python -m streamlit run app.py --server.headless true

- Run tests:
  python -m pytest -q

End of runbook.
