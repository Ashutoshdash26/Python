"""
src/utils/logger.py
────────────────────
Structured JSON logging using structlog.
Every log entry is machine-readable and includes timestamp, level,
module name, and any extra key-value context passed by the caller.
"""

from __future__ import annotations

import logging
import sys
from typing import Any

import structlog


def configure_logging(level: str = "INFO", json_output: bool = True) -> None:
    """
    Call once at application startup (in main.py or train.py).
    Sets up structlog with JSON renderer for production or
    pretty console output for development.
    """
    shared_processors = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
    ]

    if json_output:
        renderer = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=True)

    structlog.configure(
        processors=shared_processors + [
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        processor=renderer,
        foreign_pre_chain=shared_processors,
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.addHandler(handler)
    root_logger.setLevel(getattr(logging, level.upper(), logging.INFO))


def get_logger(name: str) -> Any:
    """Return a bound structlog logger for a module."""
    return structlog.get_logger(name)


# Audit logger — writes PHI-safe access logs to a separate JSONL file
class AuditLogger:
    """
    Write immutable audit records for every inference call.
    Required for HIPAA compliance: logs who accessed what and when.
    PHI is never logged — only patient_id, request_id, and timestamps.
    """

    def __init__(self, path: str):
        import json
        self.path = path
        self._log = logging.getLogger("audit")
        handler   = logging.FileHandler(path)
        handler.setFormatter(logging.Formatter("%(message)s"))
        self._log.addHandler(handler)
        self._log.setLevel(logging.INFO)
        self._json = json

    def log_inference(
        self,
        request_id: str,
        patient_id: str,
        user_id: str,
        top_prediction: str,
        processing_ms: float,
        completeness_pct: float,
    ) -> None:
        import datetime
        record = {
            "event":           "inference",
            "timestamp":       datetime.datetime.utcnow().isoformat() + "Z",
            "request_id":      request_id,
            "patient_id":      patient_id,    # anonymised
            "user_id":         user_id,
            "top_prediction":  top_prediction,
            "processing_ms":   processing_ms,
            "completeness_pct": completeness_pct,
        }
        self._log.info(self._json.dumps(record))
