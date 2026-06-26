"""Per-turn run traces — the deep-dive artifact (plan/009 §2).

Every turn carries a ``run_id``; a :class:`Tracer` records the ordered step
timeline for that turn and persists it to ``traces/<run_id>.json`` so any
"the answer looked wrong" complaint can be reconstructed: intent, retrieved Trio
ids, every SQL attempt + error, self-correction, masked-cell counts, and the
outcome.

The active tracer is held in a :class:`~contextvars.ContextVar` so nodes (and the
inner analysis pipeline) can record events without threading it through state.

**Safety (plan/007 §3):** traces store only counts / ids / errors / the SQL text —
**never** ``raw_rows`` or ``masked_rows`` — so no PII (email/address/geo) can leak
into a trace or log. :func:`summarize_delta` whitelists fields per node to enforce
this; a unit test asserts no raw PII survives.
"""

from __future__ import annotations

import contextvars
import json
import time
from pathlib import Path
from typing import Any

_current: contextvars.ContextVar[Tracer | None] = contextvars.ContextVar(
    "current_tracer", default=None
)

# Per-node whitelist of trace-safe fields extracted from a node's state delta.
# Deliberately omits raw_rows / masked_rows so row data never reaches a trace.
_NODE_FIELDS: dict[str, Any] = {
    "retrieve_golden": lambda d: {
        "trio_ids": [t.id for t in (d.get("retrieved_trios") or [])],
        "cold": d.get("retrieval_cold"),
    },
    "generate_sql": lambda d: {"attempt": d.get("sql_attempts"), "sql": d.get("generated_sql")},
    "validate_sql": lambda d: {"ok": not d.get("last_error"), "error": d.get("last_error")},
    "execute_sql": lambda d: {"rows": d.get("row_count"), "error": d.get("last_error")},
    "self_correct": lambda d: {"empty_retry": bool(d.get("empty_retried"))},
    "mask_pii": lambda d: {"pii_masked": d.get("pii_masked_count")},
    "synthesize_report": lambda d: {"report_chars": len(d.get("report") or "")},
}


def summarize_delta(node: str, delta: Any) -> dict:
    """Extract trace-safe fields for ``node`` from its state delta (never row data)."""
    extractor = _NODE_FIELDS.get(node)
    if extractor is None or not isinstance(delta, dict):
        return {}
    return {key: value for key, value in extractor(delta).items() if value is not None}


class Tracer:
    """Accumulates the ordered event timeline for one turn."""

    def __init__(self, run_id: str, user_id: str, thread_id: str, raw_question: str) -> None:
        self.run_id = run_id
        self.user_id = user_id
        self.thread_id = thread_id
        self.raw_question = raw_question
        self.header: dict = {}
        self.events: list[dict] = []
        self.outcome: dict = {}
        self._t0 = time.perf_counter()
        self._last = self._t0

    def event(self, node: str, **fields: Any) -> None:
        now = time.perf_counter()
        self.events.append(
            {"node": node, "latency_ms": round((now - self._last) * 1000, 1), **fields}
        )
        self._last = now

    def set_header(self, **fields: Any) -> None:
        self.header.update({k: v for k, v in fields.items() if v is not None})

    def finalize(self, **outcome: Any) -> None:
        self.outcome = {
            **{k: v for k, v in outcome.items() if v is not None},
            "total_ms": round((time.perf_counter() - self._t0) * 1000, 1),
        }

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "user_id": self.user_id,
            "thread_id": self.thread_id,
            "raw_question": self.raw_question,
            "header": self.header,
            "events": self.events,
            "outcome": self.outcome,
        }

    def save(self, traces_dir: str = "traces") -> str:
        directory = Path(traces_dir)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{self.run_id}.json"
        path.write_text(json.dumps(self.to_dict(), indent=2, default=str), encoding="utf-8")
        return str(path)

    def render(self) -> str:
        """A human-readable timeline for the CLI's ``/trace`` command."""
        lines = [
            f"run_id={self.run_id}  user={self.user_id}  thread={self.thread_id}",
            f'raw_question="{self.raw_question}"',
        ]
        if self.header:
            lines.append("header: " + "  ".join(f"{k}={v}" for k, v in self.header.items()))
        for event in self.events:
            extra = "  ".join(
                f"{k}={v}" for k, v in event.items() if k not in ("node", "latency_ms")
            )
            lines.append(f"  └─ {event['node']:<18} {extra}  ({event['latency_ms']}ms)")
        if self.outcome:
            lines.append("outcome: " + "  ".join(f"{k}={v}" for k, v in self.outcome.items()))
        return "\n".join(lines)


def start_run(run_id: str, user_id: str, thread_id: str, raw_question: str) -> Tracer:
    """Create the turn's tracer and make it the active one for this context."""
    tracer = Tracer(run_id, user_id, thread_id, raw_question)
    _current.set(tracer)
    return tracer


def get_tracer() -> Tracer | None:
    """Return the active tracer, or None if tracing isn't running (e.g. in tests)."""
    return _current.get()
