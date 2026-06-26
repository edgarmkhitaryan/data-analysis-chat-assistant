"""Unit tests for the non-interrupt report nodes (plan/007 §4, plan/011 §1.2).

Pure logic with a temp store and a stub deps — no LLM, no network. The interrupt
confirm/delete path is exercised end-to-end through the graph in live verification.
"""

from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from assistant.agent.nodes.reports_cmd import (
    list_reports,
    resolve_targets,
    respond_none,
    save_report,
    view_report,
)
from assistant.reports.store import SavedReportStore


@pytest.fixture
def deps(tmp_path):
    return SimpleNamespace(reports=SavedReportStore(str(tmp_path / "reports.db")))


def test_save_report_persists_last_ai_message(deps):
    state = {
        "user_id": "manager_a",
        "messages": [
            HumanMessage(content="top products by revenue"),
            AIMessage(content="Zenith leads with $5,000.00 in revenue."),
        ],
    }
    out = save_report(state, deps)
    saved = deps.reports.list("manager_a")
    assert len(saved) == 1
    assert saved[0].content.startswith("Zenith")
    assert saved[0].title == "top products by revenue"  # titled from the preceding question
    assert "saved" in out["report"].lower()


def test_save_report_uses_requested_name(deps):
    state = {
        "user_id": "manager_a",
        "report_filters": {"name": "My Q2 Review"},
        "messages": [
            HumanMessage(content="top products by revenue"),
            AIMessage(content="Zenith leads with $5,000.00 in revenue."),
        ],
    }
    save_report(state, deps)
    saved = deps.reports.list("manager_a")
    assert saved[0].title == "My Q2 Review"  # the user-given name, not the question


def test_save_report_without_prior_report(deps):
    out = save_report({"user_id": "manager_a", "messages": [HumanMessage(content="save it")]}, deps)
    assert "no report" in out["report"].lower()
    assert deps.reports.list("manager_a") == []


def test_list_reports_empty_then_nonempty(deps):
    out = list_reports({"user_id": "manager_a"}, deps)
    assert "haven't saved" in out["report"].lower()

    deps.reports.save("manager_a", title="R1", content="x")
    out = list_reports({"user_id": "manager_a"}, deps)
    assert "R1" in out["report"]


def test_view_report_by_id_shows_content(deps):
    r = deps.reports.save("manager_a", title="Q2", content="Revenue was $5,000.")
    out = view_report({"user_id": "manager_a", "report_filters": {"name": r.id}}, deps)
    assert "Revenue was $5,000." in out["report"]
    assert r.id in out["report"]


def test_view_report_by_title(deps):
    deps.reports.save("manager_a", title="Globex monthly", content="Globex did well.")
    out = view_report({"user_id": "manager_a", "report_filters": {"name": "globex"}}, deps)
    assert "Globex did well." in out["report"]


def test_view_report_not_found(deps):
    out = view_report({"user_id": "manager_a", "report_filters": {"name": "nope"}}, deps)
    assert "no saved report" in out["report"].lower()


def test_view_report_is_owner_scoped(deps):
    r = deps.reports.save("manager_b", title="Secret", content="b only")
    out = view_report({"user_id": "manager_a", "report_filters": {"name": r.id}}, deps)
    assert "no saved report" in out["report"].lower()  # A cannot view B's report


def test_resolve_targets_sets_pending_action(deps):
    deps.reports.save("manager_a", title="Acme review", content="acme", clients=["Acme"])
    deps.reports.save("manager_a", title="Other", content="x")
    out = resolve_targets(
        {"user_id": "manager_a", "report_filters": {"client": "acme", "today": False}}, deps
    )
    pending = out["pending_action"]
    assert pending and len(pending["target_ids"]) == 1
    assert "delete 1 report" in pending["summary"].lower()


def test_resolve_targets_no_match_returns_none(deps):
    out = resolve_targets(
        {"user_id": "manager_a", "report_filters": {"client": "zzz", "today": False}}, deps
    )
    assert out["pending_action"] is None


def test_resolve_targets_by_name(deps):
    deps.reports.save("manager_a", title="My Q2 Review", content="x")
    deps.reports.save("manager_a", title="Globex monthly", content="y")
    out = resolve_targets(
        {"user_id": "manager_a", "report_filters": {"name": "q2 review"}}, deps
    )
    pending = out["pending_action"]
    assert pending and len(pending["target_ids"]) == 1


def test_unqualified_delete_never_targets_all(deps):
    # Two reports exist, but a delete with NO qualifier must not propose deleting them all.
    deps.reports.save("manager_a", title="R1", content="x")
    deps.reports.save("manager_a", title="R2", content="y")
    out = resolve_targets({"user_id": "manager_a", "report_filters": {}}, deps)
    assert out["pending_action"] is None  # safety: vague delete targets nothing


def test_explicit_delete_all_targets_everything(deps):
    deps.reports.save("manager_a", title="R1", content="x")
    deps.reports.save("manager_a", title="R2", content="y")
    out = resolve_targets({"user_id": "manager_a", "report_filters": {"all": True}}, deps)
    pending = out["pending_action"]
    assert pending and len(pending["target_ids"]) == 2


def test_respond_none_mentions_filter(deps):
    out = respond_none({"report_filters": {"client": "Acme"}})
    assert "no saved reports" in out["report"].lower()
    assert "Acme" in out["report"]


def test_respond_none_vague_delete_asks_to_be_specific(deps):
    out = respond_none({"report_filters": {}})
    assert "which report" in out["report"].lower()
