# Legal Intelligence Engine

This repository implements a legal retrieval and generation pipeline for Arabic legal content. It follows the architecture contract in `ARCHITECTURE_CONTRACT.md` and adds production-grade DevOps and MLOps controls around data, model, evaluation, and release quality.

## What is included

- Retrieval stack with BM25 + dense search + reranking
- Evidence-aware generation and citation validation
- Protected baseline enforcement for retrieval quality
- DVC-based data/model/index versioning
- Docker and Docker Compose for API + Streamlit deployment
- GitHub Actions CI/CD pipeline
- Monitoring stack via Prometheus + Grafana
- Regression gate before merge
- Rollback guidance for knowledge and index artifacts

## Streamlit app

Run locally with:

```bash
streamlit run ui/streamlit_app.py --server.port 8501
```

If you use Docker Compose, the Streamlit UI is available at http://localhost:8501 and the API at http://localhost:8000.

## Quick start

```bash
python -m venv .venv
. .venv/bin/activate  # PowerShell: .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
pytest
python scripts/regression_harness.py
python scripts/quality_gate.py
```

## DVC workflow

Track data and model assets with DVC:

```bash
dvc init
# Use a shared remote such as Google Drive / Dropbox for dev only
# Example:
dvc remote add -d drive ../legal-rag-remote
```

Then track large artifacts:

```bash
dvc add data/
# or dvc add artifacts/indexes artifacts/embeddings
```

To restore a previous version:

```bash
dvc checkout
```

## CI and quality gates

Every pull request runs:

- Ruff linting
- mypy type checking
- pytest unit tests
- regression quality gate
- DVC pull/check status

The regression gate refuses merges when retrieval metrics fall below the protected baseline in `src/legal_ai/evaluation/baseline.py`.

## Docker

```bash
docker compose up --build
```

## Monitoring

```bash
docker compose -f monitoring/docker-compose.monitoring.yml up -d
```

Grafana is exposed on http://localhost:3000 and Prometheus on http://localhost:9090.

## Rollback

Use the Git tag plus the knowledge/version manifest and DVC artifacts to restore an earlier state:

```bash
git checkout <tag>
dvc checkout
```

## Architecture contract

This project should be read together with `ARCHITECTURE_CONTRACT.md` and `COPILOT.md`.
