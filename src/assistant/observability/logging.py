"""Structured JSON logging (plan/009 §1).

Routes the package's logs to ``logs/agent.jsonl`` as one JSON object per line,
each automatically stamped with the active turn's ``run_id`` / ``user_id`` /
``thread_id`` (pulled from the tracer context var) so a single conversation is
greppable locally and queryable in Cloud Logging in production. Keeping logs in a
file (not stdout) leaves the CLI chat output clean.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

from assistant.observability.tracing import get_tracer


class JsonFormatter(logging.Formatter):
    """Render a log record as a single JSON line, enriched with the run context."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "level": record.levelname,
            "logger": record.name,
            "event": record.getMessage(),
        }
        tracer = get_tracer()
        if tracer is not None:
            payload["run_id"] = tracer.run_id
            payload["user_id"] = tracer.user_id
            payload["thread_id"] = tracer.thread_id
        return json.dumps(payload, default=str)


def configure_logging(level: str = "INFO", log_dir: str = "logs") -> None:
    """Send ``assistant.*`` logs as JSON lines to ``<log_dir>/agent.jsonl``."""
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(Path(log_dir) / "agent.jsonl", encoding="utf-8")
    handler.setFormatter(JsonFormatter())

    logger = logging.getLogger("assistant")
    logger.handlers = [handler]
    logger.setLevel(level.upper())
    logger.propagate = False


def enable_langsmith(api_key: str | None) -> bool:
    """Turn on LangSmith auto-tracing when a key is present (plan/009 §2).

    LangGraph traces each node as a span automatically once these env vars are set;
    without a key we fall back to the local trace files (graceful degradation of
    observability itself). Returns True if LangSmith was enabled.
    """
    if not api_key:
        return False
    os.environ.setdefault("LANGSMITH_TRACING", "true")
    os.environ.setdefault("LANGSMITH_API_KEY", api_key)
    return True
