"""Saved-reports management nodes — the High-Stakes Oversight path (plan/007 §4).

Flow: ``parse_report_command`` extracts the action (+ delete filters), then:
- **save / list** run directly (non-destructive);
- **delete** goes ``resolve_targets`` (ownership-scoped) -> if nothing matches,
  respond plainly; otherwise ``confirm_delete`` raises a LangGraph ``interrupt()``
  with a blast-radius summary and **only** mutates on an explicit affirmative reply.

Every read/write is ``owner_id``-scoped, and the delete is recorded in the audit
log. The destructive path is completely separate from the read-only analysis path.
"""

import logging
from typing import Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.types import interrupt
from pydantic import BaseModel, Field

from assistant.agent.dependencies import AgentDeps
from assistant.agent.nodes.common import as_text
from assistant.agent.state import AgentState
from assistant.llm import get_chat_model, resilient_invoke

logger = logging.getLogger(__name__)

_AFFIRMATIVE = {"confirm", "yes", "y", "yes, delete", "delete", "confirm delete", "ok", "proceed"}


class ReportCommand(BaseModel):
    """Parsed saved-reports command."""

    action: Literal["save", "list", "delete"]
    client: str | None = Field(
        default=None, description="for delete: a client/entity name to match, if mentioned"
    )
    scope_today: bool = Field(
        default=False, description="for delete: true if it targets reports made today"
    )


_PARSE_SYSTEM = (
    "Parse a command about a manager's SAVED REPORTS library into a structured form.\n"
    '- action: "save" (save the last report), "list" (show saved reports), or "delete".\n'
    "- For delete only: set client to the client/entity name if the user names one "
    '(e.g. "delete reports mentioning Acme" -> client="Acme"); set scope_today=true if '
    'they target reports made today (e.g. "delete the reports we made today"). If they '
    'say "delete all my reports" with no qualifier, leave client null and scope_today false.'
)


def parse_report_command(state: AgentState, deps: AgentDeps) -> dict:
    """Classify the saved-reports command and extract any delete filters."""
    question = state.get("question", "")
    chat = get_chat_model(temperature=0.0, settings=deps.settings)
    try:
        cmd: ReportCommand = resilient_invoke(
            chat.with_structured_output(ReportCommand),
            [SystemMessage(content=_PARSE_SYSTEM), HumanMessage(content=question)],
            settings=deps.settings,
        )
    except Exception as exc:  # noqa: BLE001 — default to the safe, non-destructive action
        logger.warning("Report-command parse failed (%s); defaulting to list", exc)
        return {"report_action": "list", "report_filters": {}}
    return {
        "report_action": cmd.action,
        "report_filters": {"client": cmd.client, "today": cmd.scope_today},
    }


def _preceding_report(state: AgentState) -> tuple[str | None, str | None]:
    """Return (report_text, question_title) for the most recent analysis answer in history."""
    messages = state.get("messages", [])
    report_text = None
    report_index = None
    for index in range(len(messages) - 1, -1, -1):
        if isinstance(messages[index], AIMessage):
            report_text = as_text(messages[index].content)
            report_index = index
            break
    if report_text is None:
        return None, None
    for index in range(report_index - 1, -1, -1):
        if isinstance(messages[index], HumanMessage):
            return report_text, as_text(messages[index].content)
    return report_text, None


def save_report(state: AgentState, deps: AgentDeps) -> dict:
    """Persist the most recent analysis report to the user's library."""
    content, title = _preceding_report(state)
    if not content:
        message = "There's no report to save yet — ask a data question first, then save the answer."
        return {"report": message, "messages": [AIMessage(content=message)]}

    title = (title or content.splitlines()[0])[:80]
    report = deps.reports.save(state["user_id"], title=title, content=content)
    deps.reports.record_audit(state["user_id"], "save", f"id={report.id} title={report.title!r}")
    message = f"Saved your report as **{report.title}** (id `{report.id}`)."
    return {"report": message, "messages": [AIMessage(content=message)]}


def list_reports(state: AgentState, deps: AgentDeps) -> dict:
    """List the user's saved reports."""
    reports = deps.reports.list(state["user_id"])
    if not reports:
        message = "You haven't saved any reports yet."
    else:
        lines = [f"- **{r.title}** — id `{r.id}` ({r.created_at[:10]})" for r in reports]
        message = "Your saved reports:\n" + "\n".join(lines)
    return {"report": message, "messages": [AIMessage(content=message)]}


def resolve_targets(state: AgentState, deps: AgentDeps) -> dict:
    """Resolve the reports a delete would affect (owner-scoped) into ``pending_action``."""
    filters = state.get("report_filters") or {}
    matches = deps.reports.find(
        state["user_id"], client=filters.get("client"), today=bool(filters.get("today"))
    )
    if not matches:
        return {"pending_action": None}

    preview = "\n".join(f"  • {r.title} ({r.created_at[:10]})" for r in matches[:10])
    if len(matches) > 10:
        preview += f"\n  …and {len(matches) - 10} more"
    summary = (
        f"⚠️ This will permanently delete {len(matches)} report(s) owned by you:\n"
        f"{preview}\n\nReply **confirm** to proceed, or **cancel** to abort."
    )
    return {
        "pending_action": {
            "action": "delete",
            "filters": filters,
            "target_ids": [r.id for r in matches],
            "summary": summary,
        }
    }


def respond_none(state: AgentState) -> dict:
    """No reports matched the delete request — say so plainly (no interrupt)."""
    filters = state.get("report_filters") or {}
    if filters.get("client"):
        scope = f" mentioning {filters['client']}"
    elif filters.get("today"):
        scope = " from today"
    else:
        scope = ""
    message = f"You have no saved reports{scope} to delete."
    return {"report": message, "messages": [AIMessage(content=message)]}


def confirm_delete(state: AgentState, deps: AgentDeps) -> dict:
    """Pause for explicit confirmation, then delete (owner-scoped) or cancel safely."""
    pending = state["pending_action"]

    # The graph pauses here (state persisted by the checkpointer) and returns the
    # summary to the CLI; on resume, ``interrupt`` yields the user's reply.
    reply = interrupt({"summary": pending["summary"], "count": len(pending["target_ids"])})
    answer = (reply if isinstance(reply, str) else str(reply)).strip().lower()
    affirmative = answer in _AFFIRMATIVE or answer.startswith(("confirm", "yes"))

    if not affirmative:
        message = "Cancelled — nothing was deleted."
        return {"report": message, "messages": [AIMessage(content=message)], "pending_action": None}

    deleted = deps.reports.delete(pending["target_ids"], state["user_id"])
    deps.reports.record_audit(
        state["user_id"], "delete", f"deleted={deleted} ids={pending['target_ids']}"
    )
    message = f"Done — deleted {deleted} report(s)."
    return {"report": message, "messages": [AIMessage(content=message)], "pending_action": None}
