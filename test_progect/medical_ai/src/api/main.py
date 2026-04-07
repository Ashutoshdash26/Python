"""
src/api/main.py
────────────────
FastAPI application factory.

Startup sequence:
  1. Load config
  2. Build MedicalPredictor (model + preprocessor + RAG)
  3. Register routers (prediction, health, metrics)
  4. Attach middleware (auth, rate limiting, CORS, audit logging)
  5. Expose Prometheus /metrics endpoint

This file is the entrypoint for:
  uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --workers 4
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path

import yaml
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes import router
from src.api.middleware import AuditMiddleware, RateLimitMiddleware
from src.utils.logger import configure_logging, get_logger

log = get_logger(__name__)

# ── Globals ───────────────────────────────────────────────────────────────
# Stored on app.state so routes can access them without circular imports
_CONFIG_PATH = os.getenv("CONFIG_PATH", "configs/config.yaml")


def _load_config() -> dict:
    with open(_CONFIG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Async context manager for startup / shutdown.
    All heavy initialisation (model loading) happens here so the
    app only starts serving after the predictor is ready.
    """
    cfg = _load_config()
    configure_logging(
        level=cfg["monitoring"]["log_level"],
        json_output=not os.getenv("DEV_MODE"),
    )

    log.info("MedicalAI API starting", version=cfg["project"]["version"])

    # Determine device
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("Inference device", device=device)

    # Lazy import to avoid circular deps
    from src.inference.predictor import build_predictor

    model_path    = os.getenv("MODEL_PATH",    "outputs/models/best_model.pt")
    icd_vocab_path= os.getenv("ICD_VOCAB_PATH","data/icd10_vocab.json")
    guideline_dir = os.getenv("GUIDELINE_DIR", cfg["paths"]["guideline_dir"])

    predictor = build_predictor(
        model_path=model_path,
        cfg=cfg,
        icd_vocab_path=icd_vocab_path,
        guideline_dir=guideline_dir,
        device=device,
    )

    # Attach to app state
    app.state.predictor = predictor
    app.state.config    = cfg

    log.info("MedicalAI API ready")
    yield

    log.info("MedicalAI API shutting down")


def create_app() -> FastAPI:
    cfg = _load_config()

    app = FastAPI(
        title="MedicalAI — Disease Prediction API",
        description=(
            "Clinical decision support system that predicts diseases from patient "
            "case history and recommends evidence-based confirmatory investigations. "
            "**All outputs are AI-generated opinions and do not constitute a diagnosis.**"
        ),
        version=cfg["project"]["version"],
        docs_url="/api/docs",
        redoc_url="/api/redoc",
        openapi_url="/api/openapi.json",
        lifespan=lifespan,
    )

    # ── CORS ──────────────────────────────────────────────────────────────
    # In production: restrict origins to your hospital EHR domain
    app.add_middleware(
        CORSMiddleware,
        allow_origins=os.getenv("ALLOWED_ORIGINS", "*").split(","),
        allow_credentials=True,
        allow_methods=["GET", "POST"],
        allow_headers=["Authorization", "Content-Type"],
    )

    # ── Custom middleware ─────────────────────────────────────────────────
    app.add_middleware(AuditMiddleware)
    app.add_middleware(
        RateLimitMiddleware,
        max_requests=cfg["api"]["rate_limit_per_minute"],
        window_seconds=60,
    )

    # ── Routers ───────────────────────────────────────────────────────────
    app.include_router(router, prefix="/api/v1")

    # ── Prometheus metrics ────────────────────────────────────────────────
    if cfg["monitoring"]["enable_prometheus"]:
        try:
            from prometheus_client import make_asgi_app
            metrics_app = make_asgi_app()
            app.mount("/metrics", metrics_app)
            log.info("Prometheus metrics mounted at /metrics")
        except ImportError:
            log.warning("prometheus_client not installed; metrics disabled")

    return app


# Create the ASGI app
app = create_app()
