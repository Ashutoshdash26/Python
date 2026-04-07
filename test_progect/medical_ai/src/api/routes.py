"""
src/api/routes.py
──────────────────
All API route handlers.

Endpoints:
  POST /api/v1/predict          — Main disease prediction
  POST /api/v1/predict/batch    — Batch prediction (up to 20 patients)
  GET  /api/v1/health           — Health check (liveness + readiness)
  GET  /api/v1/diseases         — List all supported ICD-10 codes
  GET  /api/v1/model/info       — Model metadata and parameter counts

Authentication:
  Bearer JWT token in Authorization header.
  In production: integrate with hospital LDAP / OAuth2.
  Development: use the static token from env var DEV_TOKEN.
"""

from __future__ import annotations

import os
import time
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from src.data.schema import PatientInput, PredictionResponse
from src.utils.logger import get_logger

log = get_logger(__name__)
router = APIRouter()
security = HTTPBearer()

# ── Simple dev-token auth ─────────────────────────────────────────────────
_DEV_TOKEN = os.getenv("DEV_TOKEN", "medical-ai-dev-token-change-in-production")


def verify_token(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> str:
    """
    Verify Bearer token. Returns user_id string.
    Production: decode JWT, verify signature, check expiry, return sub claim.
    """
    token = credentials.credentials
    # Development: accept static token
    if token == _DEV_TOKEN:
        return "dev_user"
    # Production JWT verification would go here:
    # payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
    # return payload["sub"]
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or expired token",
        headers={"WWW-Authenticate": "Bearer"},
    )


# ── Prometheus counters (optional) ────────────────────────────────────────
try:
    from prometheus_client import Counter, Histogram
    PREDICTION_COUNTER = Counter(
        "medical_ai_predictions_total", "Total prediction requests",
        ["status"]
    )
    PREDICTION_LATENCY = Histogram(
        "medical_ai_prediction_latency_seconds",
        "Prediction latency in seconds",
        buckets=[0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
    )
    HAS_PROMETHEUS = True
except ImportError:
    HAS_PROMETHEUS = False


# ─────────────────────────────────────────────────────────────────────────
#  POST /predict
# ─────────────────────────────────────────────────────────────────────────

@router.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Predict diseases from patient case history",
    description=(
        "Submit a patient case history and receive ranked disease predictions "
        "with calibrated probabilities, supporting evidence, and recommended "
        "confirmatory investigations. **AI opinion only — not a clinical diagnosis.**"
    ),
    responses={
        200: {"description": "Successful prediction"},
        400: {"description": "Invalid patient data"},
        401: {"description": "Unauthorised"},
        422: {"description": "Validation error in input schema"},
        500: {"description": "Internal inference error"},
    },
)
async def predict(
    request: Request,
    patient: PatientInput,
    user_id: str = Depends(verify_token),
) -> PredictionResponse:
    """Main prediction endpoint."""
    t_start = time.perf_counter()

    predictor = request.app.state.predictor
    if predictor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Predictor not initialised. Model may still be loading.",
        )

    try:
        response = predictor.predict(patient)

        if HAS_PROMETHEUS:
            PREDICTION_COUNTER.labels(status="success").inc()
            PREDICTION_LATENCY.observe(time.perf_counter() - t_start)

        log.info(
            "Prediction served",
            user_id=user_id,
            patient_id=patient.patient_id,
            top_disease=response.top_predictions[0].disease_name
                        if response.top_predictions else "none",
            latency_ms=response.processing_time_ms,
        )
        return response

    except ValueError as e:
        if HAS_PROMETHEUS:
            PREDICTION_COUNTER.labels(status="validation_error").inc()
        log.warning("Validation error", error=str(e), patient_id=patient.patient_id)
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        if HAS_PROMETHEUS:
            PREDICTION_COUNTER.labels(status="error").inc()
        log.error("Inference error", error=str(e), patient_id=patient.patient_id)
        raise HTTPException(
            status_code=500,
            detail="Inference failed. Please check logs and try again.",
        )


# ─────────────────────────────────────────────────────────────────────────
#  POST /predict/batch
# ─────────────────────────────────────────────────────────────────────────

class BatchPredictRequest:
    patients: list[PatientInput]


@router.post(
    "/predict/batch",
    response_model=list[PredictionResponse],
    summary="Batch disease prediction for multiple patients",
)
async def predict_batch(
    request: Request,
    patients: list[PatientInput],
    user_id: str = Depends(verify_token),
) -> list[PredictionResponse]:
    """Run predictions for up to 20 patients in a single call."""
    if len(patients) > 20:
        raise HTTPException(
            status_code=400,
            detail="Batch size limited to 20 patients per request.",
        )
    predictor = request.app.state.predictor
    responses = []
    for patient in patients:
        try:
            responses.append(predictor.predict(patient))
        except Exception as e:
            log.error("Batch inference error", patient_id=patient.patient_id, error=str(e))
            raise HTTPException(status_code=500, detail=f"Error on patient {patient.patient_id}: {e}")
    return responses


# ─────────────────────────────────────────────────────────────────────────
#  GET /health
# ─────────────────────────────────────────────────────────────────────────

@router.get("/health", summary="Health check", include_in_schema=False)
async def health(request: Request) -> dict:
    """
    Kubernetes liveness + readiness probe.
    Returns 200 when the predictor is loaded and ready.
    """
    predictor = getattr(request.app.state, "predictor", None)
    cfg       = getattr(request.app.state, "config", {})

    ready = predictor is not None

    return {
        "status":  "healthy" if ready else "starting",
        "ready":   ready,
        "version": cfg.get("project", {}).get("version", "unknown"),
    }


# ─────────────────────────────────────────────────────────────────────────
#  GET /diseases
# ─────────────────────────────────────────────────────────────────────────

@router.get(
    "/diseases",
    summary="List all supported ICD-10 disease codes",
)
async def list_diseases(
    request: Request,
    system: Optional[str] = None,
    user_id: str = Depends(verify_token),
) -> dict:
    """
    Return the complete list of ICD-10 codes the model can predict.
    Optionally filter by body system (e.g. ?system=Cardiovascular).
    """
    predictor = request.app.state.predictor
    codes = predictor.icd_registry.codes

    if system:
        codes = [c for c in codes if c.get("system", "").lower() == system.lower()]

    return {
        "total": len(codes),
        "filter_system": system,
        "diseases": codes,
    }


# ─────────────────────────────────────────────────────────────────────────
#  GET /model/info
# ─────────────────────────────────────────────────────────────────────────

@router.get(
    "/model/info",
    summary="Model metadata and parameter counts",
)
async def model_info(
    request: Request,
    user_id: str = Depends(verify_token),
) -> dict:
    """Return model architecture info, version, and parameter counts."""
    predictor = request.app.state.predictor
    cfg       = request.app.state.config

    param_counts = predictor.model.parameter_count()

    return {
        "model_version":    predictor.model_version,
        "nlp_backbone":     cfg["nlp"]["model_name"],
        "num_diseases":     predictor.model.num_diseases,
        "parameter_counts": param_counts,
        "device":           predictor.device,
        "calibrated":       True,
        "disclaimer": (
            "This model is a clinical decision support tool. "
            "All outputs require clinical validation by a qualified medical professional."
        ),
    }
