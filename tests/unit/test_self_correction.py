"""Tests for the bounded self-correction loop (plan/008 §1, plan/011 §1.2).

Two layers:
- pure tests of the loop routers + the self_correct node (no graph, no I/O);
- a component test that drives the REAL analysis pipeline with a **fake BigQuery**
  (errors once, then succeeds) and a **fake LLM** — proving recovery within budget
  and honest degradation when the budget is exhausted, with zero quota.
"""

from types import SimpleNamespace

import pandas as pd
import pytest
from langchain_core.messages import AIMessage

from assistant.agent.graph import (
    _route_after_execute,
    _route_after_validate,
    build_analysis_pipeline,
)
from assistant.agent.nodes import schema as schema_node
from assistant.agent.nodes.self_correct import self_correct
from assistant.bigquery.runner import ColumnInfo, QueryResult

# --- Pure router / node tests ------------------------------------------------


def test_validate_routes_valid_to_execute():
    assert _route_after_validate({}, 3) == "execute"


def test_validate_retries_then_degrades_on_budget():
    assert _route_after_validate({"last_error": "bad", "sql_attempts": 1}, 3) == "retry"
    assert _route_after_validate({"last_error": "bad", "sql_attempts": 3}, 3) == "degrade"


def test_execute_routes_rows_to_mask():
    assert _route_after_execute({"row_count": 5}, 3) == "mask"


def test_execute_error_retries_then_degrades():
    assert _route_after_execute({"last_error": "boom", "sql_attempts": 1}, 3) == "retry"
    assert _route_after_execute({"last_error": "boom", "sql_attempts": 3}, 3) == "degrade"


def test_execute_empty_retries_once_then_accepts():
    assert _route_after_execute({"row_count": 0, "sql_attempts": 1}, 3) == "retry"
    # already retried -> accept the empty result (honest no-data report), don't loop
    assert (
        _route_after_execute({"row_count": 0, "sql_attempts": 2, "empty_retried": True}, 3)
        == "mask"
    )
    # budget exhausted -> accept
    assert _route_after_execute({"row_count": 0, "sql_attempts": 3}, 3) == "mask"


def test_self_correct_passthrough_on_error():
    assert self_correct({"last_error": "x", "sql_attempts": 1}) == {}


def test_self_correct_injects_empty_hint_once():
    out = self_correct({"sql_attempts": 1})
    assert out["empty_retried"] is True
    assert "0 rows" in out["last_error"]


# --- Component test: real pipeline, fake BigQuery + fake LLM ------------------

_DATASET = "bigquery-public-data.thelook_ecommerce"
_VALID_SQL = f"SELECT status, COUNT(*) AS n FROM `{_DATASET}.orders` GROUP BY status"


class _FakeSQLChat:
    def invoke(self, _messages):
        return AIMessage(content=_VALID_SQL)


class _FakeReportChat:
    def invoke(self, _messages):
        return AIMessage(content="Orders by status: Complete leads.")


class _FakeRetriever:
    def retrieve(self, _question):
        return []


class _FakeRunner:
    """Errors on the first ``fail_times`` execute calls, then returns one row."""

    def __init__(self, fail_times):
        self.dataset_id = _DATASET
        self.fail_times = fail_times
        self.calls = 0

    def get_table_schema(self, _table):
        return [
            ColumnInfo(name="status", type="STRING", mode="NULLABLE"),
            ColumnInfo(name="order_id", type="INTEGER", mode="NULLABLE"),
        ]

    def execute_query(self, _sql, **_kwargs):
        self.calls += 1
        if self.calls <= self.fail_times:
            raise Exception("404 Unrecognized name: revenue at [1:8]")
        df = pd.DataFrame([{"status": "Complete", "n": 10}])
        return QueryResult(dataframe=df, bytes_processed=0, bytes_billed=0, duration_ms=0.0)


@pytest.fixture
def fake_llm(monkeypatch):
    monkeypatch.setattr(
        "assistant.agent.nodes.generate_sql.get_chat_model", lambda **_kw: _FakeSQLChat()
    )
    monkeypatch.setattr(
        "assistant.agent.nodes.report.get_chat_model", lambda **_kw: _FakeReportChat()
    )
    schema_node._cache.clear()  # don't reuse a real cached schema


def _deps(runner):
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


def _sub_state():
    return {
        "question": "How many orders are in each status?",
        "raw_question": "How many orders are in each status?",
        "sql_attempts": 0,
        "last_error": None,
        "messages": [],
    }


def test_pipeline_self_corrects_after_one_failure(fake_llm):
    runner = _FakeRunner(fail_times=1)
    pipeline = build_analysis_pipeline(_deps(runner))
    out = pipeline.invoke(_sub_state())

    assert runner.calls == 2  # failed once, retried, succeeded
    assert out["sql_attempts"] == 2
    assert out.get("last_error") is None
    assert out["row_count"] == 1
    assert "Complete leads" in out["report"]


def test_pipeline_degrades_when_budget_exhausted(fake_llm):
    runner = _FakeRunner(fail_times=99)  # never succeeds
    pipeline = build_analysis_pipeline(_deps(runner))
    out = pipeline.invoke(_sub_state())

    assert runner.calls == 3  # bounded by max_sql_attempts
    assert "wasn't able to complete" in out["report"].lower()  # graceful degradation, no crash
