FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY pyproject.toml ./
COPY . .

RUN python -m pip install --upgrade pip setuptools wheel && \
    python -m pip install -e ".[dev]"

EXPOSE 8000 8501

CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
