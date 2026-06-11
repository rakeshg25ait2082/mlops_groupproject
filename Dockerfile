# ---- Base ----
FROM python:3.11-slim

# ---- Image Metadata ----
LABEL maintainer="MLOps Team"
LABEL description="DistilBERT IMDb sentiment analysis inference container"
LABEL version="1.0"

# ---- OCI Standard Labels ----
LABEL org.opencontainers.image.title="IMDb Sentiment Analysis"
LABEL org.opencontainers.image.description="DistilBERT inference container"
LABEL org.opencontainers.image.licenses="MIT"

# ---- Build arg: HF model name (overridable at build time) ----
ARG HF_MODEL_NAME=Nlp0187/distilbert-imdb-reviews-v1
ENV HF_MODEL_NAME=${HF_MODEL_NAME}

# ---- Env hygiene ----
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/app/.cache/huggingface \
    TRANSFORMERS_CACHE=/app/.cache/huggingface

# ---- System deps (minimal) ----
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ---- Install Python deps (CPU-only torch to keep image small) ----
COPY requirements.txt .
RUN pip install --no-cache-dir \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        -r requirements.txt

# ---- Copy source code ----
COPY src/ ./src/

# ---- Pre-download model weights at build time (faster cold start) ----
RUN python -c "from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification; \
    DistilBertTokenizerFast.from_pretrained('${HF_MODEL_NAME}'); \
    DistilBertForSequenceClassification.from_pretrained('${HF_MODEL_NAME}')"

# ---- Run inference ----
CMD ["python", "src/inference.py"]
