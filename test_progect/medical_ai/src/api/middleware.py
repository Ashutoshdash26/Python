"""
src/api/middleware.py
──────────────────────
Production API middleware stack.

1. AuditMiddleware  — logs every request/response to the audit trail
                      (HIPAA requirement: all PHI access must be logged)
2. RateLimitMiddleware — in-memory sliding window rate limiter
                         In production: use Redis for multi-worker safety
"""

from __future__ import annotations

import time
import uuid
from collections import defaultdict, deque

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from src.utils.logger import get_logger

log = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────
#  Audit Middleware
# ─────────────────────────────────────────────────────────────────────────

class AuditMiddleware(BaseHTTPMiddleware):
    """
    Records every API request with:
      - timestamp, method, path, status code
      - request_id (injected into response headers)
      - processing latency
      - client IP (for access tracing)

    PHI note: request body is NOT logged here.
    Patient-level detail is logged separately by the predictor.
    """

    async def dispatch(self, request: Request, call_next) -> Response:
        request_id = str(uuid.uuid4())
        t_start    = time.perf_counter()
        client_ip  = request.client.host if request.client else "unknown"

        # Inject request_id so downstream handlers can reference it
        request.state.request_id = request_id

        response = await call_next(request)

        latency_ms = round((time.perf_counter() - t_start) * 1000, 1)
        response.headers["X-Request-ID"]      = request_id
        response.headers["X-Processing-Time"] = f"{latency_ms}ms"
        response.headers["X-AI-Disclaimer"]   = (
            "AI clinical decision support - not a diagnosis"
        )

        # Skip health checks from audit to keep logs clean
        if request.url.path != "/api/v1/health":
            log.info(
                "API request",
                request_id=request_id,
                method=request.method,
                path=request.url.path,
                status=response.status_code,
                latency_ms=latency_ms,
                client_ip=client_ip,
            )

        return response


# ─────────────────────────────────────────────────────────────────────────
#  Rate Limit Middleware
# ─────────────────────────────────────────────────────────────────────────

class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Sliding window rate limiter keyed by client IP + Authorization token.

    Args:
        max_requests   : maximum requests per window
        window_seconds : window size in seconds (default 60)

    In production with multiple workers: replace the deque store with
    Redis using ZADD/ZREMRANGEBYSCORE for atomic sliding window counting.
    """

    def __init__(
        self,
        app: ASGIApp,
        max_requests: int = 120,
        window_seconds: int = 60,
    ):
        super().__init__(app)
        self.max_requests    = max_requests
        self.window_seconds  = window_seconds
        # {client_key: deque of request timestamps}
        self._store: dict[str, deque] = defaultdict(deque)

    def _get_client_key(self, request: Request) -> str:
        ip    = request.client.host if request.client else "unknown"
        token = request.headers.get("Authorization", "")[:20]
        return f"{ip}:{token}"

    async def dispatch(self, request: Request, call_next) -> Response:
        # Skip rate limiting for health checks
        if request.url.path in ("/api/v1/health", "/metrics"):
            return await call_next(request)

        key = self._get_client_key(request)
        now = time.time()
        window_start = now - self.window_seconds

        # Prune old timestamps
        dq = self._store[key]
        while dq and dq[0] < window_start:
            dq.popleft()

        if len(dq) >= self.max_requests:
            retry_after = int(self.window_seconds - (now - dq[0])) + 1
            log.warning(
                "Rate limit exceeded",
                client_key=key,
                requests_in_window=len(dq),
            )
            return Response(
                content=f"Rate limit exceeded. Try again in {retry_after} seconds.",
                status_code=429,
                headers={
                    "Retry-After": str(retry_after),
                    "X-RateLimit-Limit": str(self.max_requests),
                    "X-RateLimit-Window": f"{self.window_seconds}s",
                },
            )

        dq.append(now)
        response = await call_next(request)
        response.headers["X-RateLimit-Limit"]     = str(self.max_requests)
        response.headers["X-RateLimit-Remaining"] = str(self.max_requests - len(dq))
        return response
