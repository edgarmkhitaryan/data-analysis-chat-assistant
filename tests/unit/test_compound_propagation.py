"""Integration tests for sub-graph -> parent state propagation (audit findings H1, M5).

The analysis pipeline is a compiled subgraph invoked inside ``run_compound``; its
outputs must be lifted back into the parent turn's state. These tests assert the
*wiring* (not just a function), which the harness-level unit tests miss because they
hand-build states. Two layers:
- a recording fake pipeline (no LLM) that proves ``oneoff_preference`` reaches the
  subgraph and ``masked_rows`` are captured + surfaced;
- the REAL analysis pipeline with a fake LLM + a fake BigQuery returning a PII column,
  proving the masked (PII-free) rows reach the top-level state end-to-end.
"""

import json
from types import SimpleNamespace

import pandas as pd
import pytest
from langchain_core.messages import AIMessage

from assistant.agent.graph import build_analysis_pipeline
from assistant.agent.nodes import schema as schema_node
from assistant.agent.nodes.decompose import run_compound
from assistant.agent.nodes.synthesize import synthesize
from assistant.bigquery.runner import ColumnInfo, QueryResult

_DATASET = "bigquery-public-data.thelook_ecommerce"


def _deps(runner=None):
    settings = SimpleNamespace(
        max_sql_attempts=3,
        bq_dataset=_DATASET,
        sql_max_limit=1000,
        pii_mask_columns=["email"],
        pii_mask_style="partial",
        llm_max_retries=2,
        llm_retry_base_delay=0.0,
        circuit_breaker_threshold=5,
        circuit_breaker_cooldown_seconds=30.0,
    )
    return SimpleNamespace(
        runner=runner, retriever=_FakeRetriever(), settings=settings, profiles=None, reports=None
    )


class _FakeRetriever:
    def retrieve(self, _question):
        return []


# --- Layer 1: recording fake pipeline (no LLM, no I/O) -----------------------


class _RecordingPipeline:
    """Captures the sub_state it was invoked with; returns a fixed subgraph output."""

    def __init__(self):
        self.seen = []

    def invoke(self, sub_state):
        self.seen.append(sub_state)
        return {
            "generated_sql": "SELECT 1",
            "report": "A report.",
            "row_count": 1,
            "masked_rows": [{"value": 42}],
            "pii_masked_count": 0,
        }


def test_oneoff_preference_reaches_the_subgraph():
    """M5: a one-off preference must be passed into the per-question subgraph."""
    pipeline = _RecordingPipeline()
    state = {
        "question": "top products",
        "run_id": "r",
        "is_compound": False,
        "sub_questions": ["top products"],
        "oneoff_preference": "as bullet points just this once",
    }
    run_compound(state, pipeline)
    assert pipeline.seen[0]["oneoff_preference"] == "as bullet points just this once"


def test_masked_rows_captured_and_surfaced_single_question():
    """H1: masked_rows from the subgraph must reach the top-level state."""
    pipeline = _RecordingPipeline()
    state = {"question": "q", "run_id": "r", "is_compound": False, "sub_questions": ["q"]}
    compound_out = run_compound(state, pipeline)
    assert compound_out["sub_results"][0]["masked_rows"] == [{"value": 42}]

    merged = synthesize({**state, **compound_out}, _deps())
    assert merged["masked_rows"] == [{"value": 42}]


def test_masked_rows_surfaced_for_compound(monkeypatch):
    """H1: a compound turn surfaces the union of its parts' masked rows."""
    # The compound path runs an LLM merge — stub it so the test spends no quota.
    monkeypatch.setattr(
        "assistant.agent.nodes.synthesize.get_chat_model",
        lambda **_kw: SimpleNamespace(invoke=lambda _m: AIMessage(content="Merged briefing.")),
    )
    pipeline = _RecordingPipeline()
    state = {
        "question": "two parts",
        "run_id": "r",
        "is_compound": True,
        "sub_questions": ["part a", "part b"],
    }
    compound_out = run_compound(state, pipeline)
    merged = synthesize({**state, **compound_out}, _deps())
    assert len(merged["masked_rows"]) == 2  # one row per part


# --- Layer 2: the REAL pipeline with fake LLM + fake BigQuery (PII column) ----

_SQL = f"SELECT u.email AS email, u.id AS id FROM `{_DATASET}.users` AS u"


class _FakeSQLChat:
    def invoke(self, _messages):
        return AIMessage(content=_SQL)


class _FakeReportChat:
    def invoke(self, _messages):
        return AIMessage(content="Here is the report.")


class _FakeRunnerWithPII:
    def __init__(self):
        self.dataset_id = _DATASET

    def get_table_schema(self, _table):
        return [
            ColumnInfo(name="email", type="STRING", mode="NULLABLE"),
            ColumnInfo(name="id", type="INTEGER", mode="NULLABLE"),
        ]

    def execute_query(self, _sql, **_kwargs):
        df = pd.DataFrame([{"email": "jane@example.com", "id": 1}])
        return QueryResult(dataframe=df, bytes_processed=0, bytes_billed=0, duration_ms=0.0)


@pytest.fixture
def fake_llm(monkeypatch):
    monkeypatch.setattr(
        "assistant.agent.nodes.generate_sql.get_chat_model", lambda **_kw: _FakeSQLChat()
    )
    monkeypatch.setattr(
        "assistant.agent.nodes.report.get_chat_model", lambda **_kw: _FakeReportChat()
    )
    schema_node._cache.clear()


def test_real_pipeline_surfaces_masked_pii_free_rows(fake_llm):
    """End-to-end: the rows that reach the parent state are present AND masked."""
    deps = _deps(_FakeRunnerWithPII())
    pipeline = build_analysis_pipeline(deps)
    state = {"question": "list users", "run_id": "r", "is_compound": False,
             "sub_questions": ["list users"]}

    compound_out = run_compound(state, pipeline)
    merged = synthesize({**state, **compound_out}, deps)

    rows = merged["masked_rows"]
    assert rows and "email" in rows[0]  # rows reached the top level, schema intact
    assert "jane@example.com" not in json.dumps(rows, default=str)  # PII was masked
    assert merged["pii_masked_count"] >= 1
