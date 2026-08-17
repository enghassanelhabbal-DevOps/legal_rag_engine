# Dockerfile for FastAPI backend (api/)
FROM python:3.10-slim

WORKDIR /app

# Install backend runtime dependencies (including the native FAISS package)
COPY ./api/requirements.txt /app/api/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /app/api/requirements.txt

# Copy repository root so imports like `api.app` and `src.legal_ai...` resolve correctly
COPY . /app
ENV PYTHONPATH=/app

EXPOSE 8000
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
