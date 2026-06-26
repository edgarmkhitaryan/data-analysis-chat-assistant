"""Unit tests for observability (plan/009, plan/011 §1.1).

Pure logic, no network. Covers the trace recorder, the per-node field whitelist
(no raw rows / PII in a trace), trace persistence, and the session metrics.
"""

import json
from pathlib import Path

from assistant.observability.metrics import Metrics
from assistant.observability.tracing import Tracer, summarize_delta

# --- summarize_delta: only trace-safe fields, never row data -----------------


def test_summarize_execute_keeps_count_drops_rows():
    delta = {"row_count": 3, "last_error": None, "raw_rows": [{"email": "jane@example.com"}]}
    out = summarize_delta("execute_sql", delta)
    assert out == {"rows": 3}
    assert "jane@example.com" not in str(out)


def test_summarize_mask_keeps_only_count():
    delta = {"pii_masked_count": 5, "masked_rows": [{"email": "j***@e***.com"}]}
    assert summarize_delta("mask_pii", delta) == {"pii_masked": 5}


def test_summarize_retrieve_extracts_trio_ids():
    class _T:
        id = "trio_0007"

    out = summarize_delta("retrieve_golden", {"retrieved_trios": [_T()], "retrieval_cold": False})
    assert out["trio_ids"] == ["trio_0007"]


# --- Tracer ------------------------------------------------------------------


def test_tracer_records_saves_and_renders(tmp_path):
    tracer = Tracer("rid123", "manager_a", "th1", "top products")
    tracer.event("guard", intent="analysis")
    tracer.event("generate_sql", attempt=1, sql="SELECT 1")
    tracer.set_header(intent="analysis", is_compound=False)
    tracer.finalize(status="success", rows=10)

    path = tracer.save(str(tmp_path))
    data = json.loads(Path(path).read_text())
    assert data["run_id"] == "rid123"
    assert [e["node"] for e in data["events"]] == ["guard", "generate_sql"]
    assert data["outcome"]["status"] == "success"
    assert "total_ms" in data["outcome"]
    assert "guard" in tracer.render() and "generate_sql" in tracer.render()


def test_trace_never_contains_raw_pii(tmp_path):
    """The headline safety guarantee: no raw email/address/geo in a persisted trace."""
    tracer = Tracer("rid", "u", "th", "emails of top customers")
    tracer.event(
        "execute_sql",
        **summarize_delta(
            "execute_sql", {"row_count": 2, "raw_rows": [{"email": "jane.doe@example.com"}]}
        ),
    )
    tracer.event(
        "mask_pii",
        **summarize_delta(
            "mask_pii", {"pii_masked_count": 2, "masked_rows": [{"x": "742 Evergreen Terrace"}]}
        ),
    )
    tracer.finalize(status="success")
    blob = Path(tracer.save(str(tmp_path))).read_text()
    assert "jane.doe@example.com" not in blob
    assert "742 Evergreen Terrace" not in blob
    assert "execute_sql" in blob and "mask_pii" in blob  # the steps are still recorded


# --- Metrics -----------------------------------------------------------------


def _trace(events, intent, status, **outcome):
    tracer = Tracer("r", "u", "th", "q")
    for node, fields in events:
        tracer.event(node, **fields)
    tracer.set_header(intent=intent)
    tracer.finalize(status=status, **outcome)
    return tracer


def test_metrics_success_and_self_correction():
    metrics = Metrics()
    metrics.record(
        _trace(
            [
                ("generate_sql", {"attempt": 1}),
                ("generate_sql", {"attempt": 2}),
                ("mask_pii", {"pii_masked": 4}),
            ],
            "analysis",
            "success",
        )
    )
    metrics.record(_trace([("generate_sql", {"attempt": 1})], "analysis", "degraded"))

    assert metrics.turns == 2
    assert metrics.success == 1 and metrics.degraded == 1
    assert metrics.self_corrections == 1  # the 2-attempt turn
    assert metrics.pii_masked == 4
    assert metrics.by_intent["analysis"] == 2
    assert "Turns: 2" in metrics.summary()


def test_metrics_cold_and_empty_and_leak():
    metrics = Metrics()
    metrics.record(
        _trace(
            [("retrieve_golden", {"cold": True}), ("execute_sql", {"rows": 0})],
            "analysis",
            "success",
            pii_leak_prevented=1,
        )
    )
    assert metrics.cold_retrievals == 1
    assert metrics.empty_results == 1
    assert metrics.pii_leak_prevented == 1


def test_metrics_empty_summary():
    assert "No turns" in Metrics().summary()
