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

    action: Literal["save", "list", "delete", "view"]
    name: str | None = Field(
        default=None,
        description="a specific report the user gives or references (title or id) — for save, "
        'the name to store it under ("save it as Q2 Review"); for view, the report to show '
        '("show me the Globex report", "open report 7ab3e20a"); for delete, the report they '
        'name ("delete the report called Q2 Review")',
    )
    client: str | None = Field(
        default=None, description="for delete: a client/entity name to match, if mentioned"
    )
    scope_today: bool = Field(
        default=False, description="for delete: true if it targets reports made today"
    )
    all_reports: bool = Field(
        default=False,
        description='for delete ONLY: true if the user EXPLICITLY asks to delete ALL their '
        'reports with no other qualifier (e.g. "delete all my reports", "delete everything")',
    )


_PARSE_SYSTEM = (
    "Parse a command about a manager's SAVED REPORTS library into a structured form.\n"
    '- action: "save" (save the last report), "list" (show the saved-reports library), '
    '"view" (show the full content of one named/identified report, e.g. "show me the Globex '
    'report", "open report 7ab3e20a"), or "delete".\n'
    "- name: the specific report title or id the user gives or references. For save, the name to "
    'store it under (e.g. "save it as Q2 Review" -> name="Q2 Review"). For delete, the title '
    'they name (e.g. "delete the report called Q2 Review" -> name="Q2 Review").\n'
    '- client (delete only): a client/entity to match if mentioned (e.g. "delete reports '
    'mentioning Acme" -> client="Acme").\n'
    "- scope_today (delete only): true if they target reports made today.\n"
    '- all_reports (delete only): true ONLY if they explicitly ask to delete ALL their reports '
    'with no other qualifier (e.g. "delete all my reports", "delete everything"). If they name '
    "a specific report or client, leave all_reports false."
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
        "report_filters": {
            "name": cmd.name,
            "client": cmd.client,
            "today": cmd.scope_today,
            "all": cmd.all_reports,
        },
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
    content, fallback_title = _preceding_report(state)
    if not content:
        message = "There's no report to save yet — ask a data question first, then save the answer."
        return {"report": message, "messages": [AIMessage(content=message)]}

    # Prefer the name the user asked for ("save it as X"); fall back to the question.
    requested = (state.get("report_filters") or {}).get("name")
    title = (requested or fallback_title or content.splitlines()[0]).strip()[:80]
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


def view_report(state: AgentState, deps: AgentDeps) -> dict:
    """Show the full content of one saved report, resolved by id or title (owner-scoped)."""
    ref = (state.get("report_filters") or {}).get("name")
    if not ref:
        message = 'Which report? Name it (e.g. "show me the Globex report") or use its id.'
        return {"report": message, "messages": [AIMessage(content=message)]}

    report = deps.reports.get(ref, state["user_id"])  # exact id first
    if report is None:
        matches = deps.reports.find(state["user_id"], name=ref)  # then title match
        if len(matches) == 1:
            report = matches[0]
        elif len(matches) > 1:
            lines = [f"- **{r.title}** — id `{r.id}`" for r in matches]
            message = f"Several reports match {ref!r} — which one?\n" + "\n".join(lines)
            return {"report": message, "messages": [AIMessage(content=message)]}
        else:
            message = f"You have no saved report matching {ref!r}."
            return {"report": message, "messages": [AIMessage(content=message)]}

    message = (
        f"**{report.title}** — id `{report.id}` ({report.created_at[:10]})\n\n{report.content}"
    )
    return {"report": message, "messages": [AIMessage(content=message)]}


def resolve_targets(state: AgentState, deps: AgentDeps) -> dict:
    """Resolve the reports a delete would affect (owner-scoped) into ``pending_action``.

    Safety: a delete with NO qualifier (no name/client/today and not an explicit "all")
    must never silently target every report. It resolves to no match so the user is asked
    to be specific — preventing a vague request from proposing a delete-all.
    """
    filters = state.get("report_filters") or {}
    name = filters.get("name")
    client = filters.get("client")
    today = bool(filters.get("today"))
    delete_all = bool(filters.get("all"))

    if delete_all:
        matches = deps.reports.list(state["user_id"])
    elif name or client or today:
        matches = deps.reports.find(state["user_id"], name=name, client=client, today=today)
    else:
        matches = []  # unqualified delete: refuse to target everything by default
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
    """No reports matched (or the request was too vague) — say so plainly (no interrupt)."""
    filters = state.get("report_filters") or {}
    name = filters.get("name")
    client = filters.get("client")
    today = filters.get("today")
    delete_all = filters.get("all")
    if not (name or client or today or delete_all):
        # Vague delete with no target: ask the user to be specific (never delete-all).
        message = (
            'Which report would you like to delete? Name it (e.g. "delete the report called '
            'Q2 Review"), or say "delete all my reports" to remove them all.'
        )
    else:
        if name:
            scope = f" named {name!r}"
        elif client:
            scope = f" mentioning {client}"
        elif today:
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
