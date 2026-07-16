# ── FetalGuard AI — FastAPI Service ───────────────────────────────
# Phase 08: MLOps Deployment
#
# Build:   docker build -t ctg-api:1.0.0 .
# Run:     docker run -p 8000:8000 ctg-api:1.0.0
# Health:  curl http://localhost:8000/health

FROM python:3.11-slim

# Metadata labels
LABEL maintainer="CTG Capstone Project"
LABEL version="1.0.0"
LABEL description="FetalGuard AI FastAPI Service"

# System deps (minimal — no GUI, no Jupyter)
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy and install Python dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY api/            ./api/
COPY config/         ./config/
COPY deployment_artifact.pkl .

# Non-root user for security
RUN useradd --create-home --shell /bin/bash appuser
RUN chown -R appuser:appuser /app
USER appuser

# Liveness probe path
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
