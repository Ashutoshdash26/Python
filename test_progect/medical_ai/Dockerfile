# ── MedicalAI Production Dockerfile ──────────────────────────────────────
#
# Multi-stage build:
#   Stage 1 (builder): install all deps in a venv
#   Stage 2 (runtime): copy venv only — smaller final image
#
# GPU support: use nvidia/cuda base image for GPU inference
# CPU only:    use python:3.11-slim (swap the FROM line below)
#
# Build:
#   docker build -t medical-ai:1.0 .
#
# Run (CPU):
#   docker run -p 8000:8000 \
#     -e DEV_TOKEN=change_me \
#     -e MODEL_PATH=/models/calibrated_model.pt \
#     -v /path/to/models:/models \
#     -v /path/to/data:/app/data \
#     medical-ai:1.0
#
# Run (GPU):
#   docker run --gpus all -p 8000:8000 ... medical-ai:1.0

# ── Stage 1: Build ───────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

# Install build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Create venv
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies (cached layer unless requirements change)
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ── Stage 2: Runtime ─────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

# Runtime system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

# Non-root user for security
RUN groupadd -r medai && useradd -r -g medai medai

# Copy venv from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /app

# Copy source code
COPY src/       ./src/
COPY configs/   ./configs/
COPY scripts/   ./scripts/

# Create required directories
RUN mkdir -p outputs/logs outputs/models data/guidelines && \
    chown -R medai:medai /app

USER medai

# ── Health check ─────────────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

# ── Environment ──────────────────────────────────────────────────────────
ENV CONFIG_PATH="configs/config.yaml"
ENV MODEL_PATH="outputs/models/calibrated_model.pt"
ENV ICD_VOCAB_PATH="data/icd10_vocab.json"
ENV GUIDELINE_DIR="data/guidelines"
ENV PYTHONPATH="/app"
ENV PYTHONUNBUFFERED=1

# ── Expose ────────────────────────────────────────────────────────────────
EXPOSE 8000

# ── Entrypoint ────────────────────────────────────────────────────────────
CMD ["uvicorn", "src.api.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "4", \
     "--timeout-keep-alive", "30", \
     "--access-log"]
