"""Agent-level metrics — the counters you'd alert on (plan/009 §3).

A session accumulator updated once per completed turn from that turn's
:class:`~assistant.observability.tracing.Tracer`. Exposed via ``/metrics`` in the
CLI (Cloud Monitoring dashboards in production). Derived from trace events +
outcome, so metrics and the deep-dive trace can never disagree.
"""

from __future__ import annotations

from collections import Counter

from assistant.observability.tracing import Tracer


class Metrics:
    """Session-level counters/timers, accumulated per turn."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.turns = 0
        self.by_intent: Counter = Counter()
        self.success = 0
        self.degraded = 0
        self.clarified = 0
        self.rejected = 0
        self.self_corrections = 0
        self.total_attempts = 0
        self.attempt_turns = 0
        self.empty_results = 0
        self.cold_retrievals = 0
        self.pii_masked = 0
        self.pii_leak_prevented = 0
        self.total_ms = 0.0

    def record(self, tracer: Tracer) -> None:
        self.turns += 1
        if tracer.header.get("intent"):
            self.by_intent[tracer.header["intent"]] += 1

        status = tracer.outcome.get("status")
        if status == "success":
            self.success += 1
        elif status == "degraded":
            self.degraded += 1
        elif status == "clarification":
            self.clarified += 1
        elif status == "rejected":
            self.rejected += 1

        gen_events = [e for e in tracer.events if e["node"] == "generate_sql"]
        if gen_events:
            attempts = max((e.get("attempt") or 0) for e in gen_events) or len(gen_events)
            self.total_attempts += attempts
            self.attempt_turns += 1
            if attempts > 1:
                self.self_corrections += 1
        for event in tracer.events:
            if event["node"] == "retrieve_golden" and event.get("cold"):
                self.cold_retrievals += 1
            if event["node"] == "execute_sql" and event.get("rows") == 0 and not event.get("error"):
                self.empty_results += 1
            if event["node"] == "mask_pii":
                self.pii_masked += event.get("pii_masked") or 0

        self.pii_leak_prevented += tracer.outcome.get("pii_leak_prevented") or 0
        self.total_ms += tracer.outcome.get("total_ms") or 0.0

    def summary(self) -> str:
        if not self.turns:
            return "No turns recorded yet."
        success_rate = 100 * self.success / self.turns
        avg_attempts = self.total_attempts / self.attempt_turns if self.attempt_turns else 0.0
        avg_ms = self.total_ms / self.turns
        return "\n".join(
            [
                f"Turns: {self.turns}  by intent: {dict(self.by_intent)}",
                f"Success: {self.success}  Degraded: {self.degraded}  "
                f"Clarify: {self.clarified}  Rejected: {self.rejected}  "
                f"(success rate {success_rate:.0f}%)",
                f"Self-corrections: {self.self_corrections}  avg attempts/query: {avg_attempts:.2f}  "
                f"empty results: {self.empty_results}  cold retrievals: {self.cold_retrievals}",
                f"PII masked cells: {self.pii_masked}  pii_leak_prevented: {self.pii_leak_prevented}",
                f"Avg latency/turn: {avg_ms:.0f}ms",
            ]
        )


# Process-wide session metrics (the CLI updates and prints these).
SESSION_METRICS = Metrics()
