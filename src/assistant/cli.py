"""Interactive CLI chat loop (deliverable 4).

A thin REPL: it marshals user input into the graph and renders the result. It
holds no business logic — every decision lives in the graph's nodes. Each turn
runs on a persisted ``thread_id``; the ``contextualize`` node uses that history
to resolve follow-ups, and clarifications are rendered distinctly.
"""

import argparse
import json
import logging
import sys
import threading
import uuid

from langchain_core.messages import HumanMessage
from langgraph.types import Command
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from assistant.agent.dependencies import AgentDeps
from assistant.agent.graph import build_graph
from assistant.config import Settings, get_settings
from assistant.memory import Candidate, promote_if_qualified
from assistant.observability import (
    SESSION_METRICS,
    configure_logging,
    enable_langsmith,
    start_run,
)
from assistant.observability.tracing import Tracer

_HELP = """\
Commands:
  /login <id>  switch the current manager (drives preferences + ownership)
  /prefs       show the current user's saved preferences
  /reports     list your saved reports
  /report <id> show a saved report's full content
  /save        save the last report to your library
  /trace       show the trace (step timeline) of the last turn
  /metrics     show session metrics (success, self-correction, PII, cost)
  /whoami      show the current user and thread
  /new         start a new conversation thread
  /help        show this help
  /exit        quit"""

# The most recent turn's trace, surfaced by /trace.
_last_tracer: Tracer | None = None


def _new_thread_id() -> str:
    return uuid.uuid4().hex[:12]


def _promote_async(candidate: Candidate, deps: AgentDeps) -> None:
    """Run the automatic learning-loop gate on a worker thread (never crashes the app)."""
    try:
        promote_if_qualified(candidate, deps)
    except Exception:  # noqa: BLE001 — the learning loop must never break the REPL
        logging.getLogger(__name__).warning("learning loop failed", exc_info=True)


def _turn_status(result: dict) -> str:
    """Classify a finished turn for the trace outcome and the --json output (shared)."""
    if result.get("needs_clarification"):
        return "clarification"
    if result.get("intent") == "rejected":
        return "rejected"
    if result.get("intent") == "update_preference":
        return "preference"
    if result.get("intent") == "manage_reports":
        return "reports"
    if result.get("last_error"):
        return "degraded"
    return "success"


def _finalize_trace(tracer: Tracer, result: dict, settings: Settings, deps: AgentDeps) -> None:
    """Record the turn's header + outcome, persist the trace, metrics, and a candidate."""
    tracer.set_header(
        question=result.get("question"),
        history_used=result.get("history_used"),
        intent=result.get("intent"),
        is_compound=result.get("is_compound"),
    )
    status = _turn_status(result)
    tracer.finalize(
        status=status,
        rows=result.get("row_count"),
        pii_leak_prevented=result.get("pii_leak_prevented"),
    )
    tracer.save(settings.traces_dir)
    SESSION_METRICS.record(tracer)

    # Learning loop (plan/010 §3): on an analysis turn, run the automatic gate
    # (deterministic metrics -> dedup -> LLM-judge) on a BACKGROUND thread so it never
    # blocks the answer. No user feedback, no manual trigger, no human in the loop.
    if result.get("intent") == "analysis" and deps.settings.learning_loop_enabled:
        attempts = [e.get("attempt") or 0 for e in tracer.events if e["node"] == "generate_sql"]
        candidate = Candidate(
            run_id=tracer.run_id,
            question=result.get("question") or tracer.raw_question,
            sql=result.get("generated_sql") or "",
            report=result.get("report") or "",
            succeeded=status == "success",
            attempts=max(attempts) if attempts else 0,
            row_count=result.get("row_count") or 0,
            pii_leak_prevented=result.get("pii_leak_prevented") or 0,
            rows=result.get("masked_rows") or [],
        )
        threading.Thread(target=_promote_async, args=(candidate, deps), daemon=True).start()


def _render(console: Console, result: dict, source: str) -> None:
    """Render a completed turn (rewrite hint, clarification, SQL, report)."""
    if result.get("history_used") and result.get("question") and result["question"] != source:
        console.print(f"[dim]↳ interpreted as: {result['question']}[/]")

    if result.get("needs_clarification"):
        console.print(
            Panel(
                result.get("report", ""),
                title="Clarification",
                title_align="left",
                border_style="yellow",
            )
        )
        return

    sql = result.get("generated_sql")
    if sql and not result.get("last_error"):
        console.print(Panel(sql, title="SQL", title_align="left", border_style="grey50"))

    report = result.get("report", "(no report produced)")
    console.print(Panel(Markdown(report), title="Report", title_align="left", border_style="green"))


def _show_interrupt(console: Console, result: dict) -> bool:
    """If the graph paused for confirmation, show the summary; return True if awaiting."""
    interrupts = result.get("__interrupt__")
    if not interrupts:
        return False
    payload = interrupts[0].value
    summary = payload.get("summary") if isinstance(payload, dict) else str(payload)
    console.print(Panel(summary, title="⚠ Confirm", title_align="left", border_style="red"))
    return True


def _initial_state(question: str, user_id: str, thread_id: str, run_id: str) -> dict:
    """Per-turn graph input, reset to a clean slate.

    LangGraph input overwrites the LastValue channels (including with None), so we
    explicitly clear every transient per-turn field here. This matters because a turn
    that doesn't reach the guard (e.g. ``contextualize`` routes straight to ``clarify``)
    would otherwise re-emit the *previous* turn's ``intent``/``report``/rows from the
    checkpoint. Only ``messages`` (accumulated via the reducer) and the durable identity
    fields carry across turns.
    """
    return {
        "messages": [HumanMessage(content=question)],
        "raw_question": question,
        "question": question,
        "user_id": user_id,
        "thread_id": thread_id,
        "run_id": run_id,
        # Routing / classification (reset so non-guard paths can't leak a stale intent).
        "intent": None,
        "rejection_reason": None,
        "needs_clarification": False,
        "clarifying_question": None,
        "history_used": False,
        # Preference transients.
        "also_analysis": False,
        "oneoff_preference": None,
        "pref_update": None,
        "pref_saved_note": None,
        # Compound + analysis outputs.
        "is_compound": False,
        "sub_questions": None,
        "sub_results": [],
        "sql_attempts": 0,
        "last_error": None,
        "empty_retried": False,
        "generated_sql": None,
        "row_count": 0,
        "raw_rows": [],
        "masked_rows": [],
        "pii_masked_count": 0,
        "retrieved_trios": [],
        "retrieval_cold": False,
        # Output + oversight.
        "report": None,
        "pending_action": None,
    }


def _run_turn(
    graph,
    console: Console,
    question: str,
    user_id: str,
    thread_id: str,
    settings: Settings,
    deps: AgentDeps,
) -> bool:
    """Run one turn; return True if it paused awaiting a confirm/cancel reply."""
    global _last_tracer
    run_id = _new_thread_id()
    _last_tracer = start_run(run_id, user_id, thread_id, question)
    initial = _initial_state(question, user_id, thread_id, run_id)
    config = {"configurable": {"thread_id": thread_id}}
    with console.status("[dim]thinking…[/]", spinner="dots"):
        result = graph.invoke(initial, config=config)
    if _show_interrupt(console, result):
        _last_tracer.finalize(status="awaiting_confirmation")
        _last_tracer.save(settings.traces_dir)
        return True
    _finalize_trace(_last_tracer, result, settings, deps)
    _render(console, result, question)
    console.print(f"[dim]run_id={run_id} · /trace for the step timeline[/]")
    return False


def _resume_turn(
    graph, console: Console, reply: str, thread_id: str, settings: Settings, deps: AgentDeps
) -> bool:
    """Resume a paused (interrupted) turn with the user's confirm/cancel reply."""
    global _last_tracer
    run_id = _new_thread_id()
    _last_tracer = start_run(run_id, "resume", thread_id, reply)
    config = {"configurable": {"thread_id": thread_id}}
    with console.status("[dim]working…[/]", spinner="dots"):
        result = graph.invoke(Command(resume=reply), config=config)
    if _show_interrupt(console, result):
        return True
    _finalize_trace(_last_tracer, result, settings, deps)
    _render(console, result, reply)
    return False


def _emit(obj: dict) -> None:
    """Write exactly one JSON object as a line to stdout, flushed (stream-friendly)."""
    print(json.dumps(obj, default=str), flush=True)


def _turn_obj(result: dict, run_id: str) -> dict:
    """Machine-readable form of a finished turn — the --json counterpart of _render."""
    return {
        "type": "turn",
        "run_id": run_id,
        "question": result.get("question"),
        "intent": result.get("intent"),
        "status": _turn_status(result),
        "needs_clarification": bool(result.get("needs_clarification")),
        "history_used": bool(result.get("history_used")),
        "sql": result.get("generated_sql"),
        "row_count": result.get("row_count"),
        "report": result.get("report"),
    }


def _interrupt_obj(result: dict, run_id: str) -> dict | None:
    """Return the awaiting-confirmation object if the turn paused, else None."""
    interrupts = result.get("__interrupt__")
    if not interrupts:
        return None
    payload = interrupts[0].value
    summary = payload.get("summary") if isinstance(payload, dict) else str(payload)
    return {"type": "awaiting_confirmation", "run_id": run_id, "summary": summary}


def _run_batch_json(graph, settings: Settings, deps: AgentDeps, user_id: str) -> None:
    """Non-interactive batch mode for agents/automation (``assistant --json``).

    One process, one ``thread_id`` → the SAME conversation memory as the REPL
    (follow-ups resolve via history); durable preferences/reports/golden apply as
    usual. Reads one turn per stdin line and writes exactly one JSON object per line
    to stdout, flushed, so a reader-then-writer driver can go turn-by-turn. Every
    object carries a ``type``: turn | awaiting_confirmation | prefs | whoami |
    metrics | trace | new_thread | login | help | error.
    """
    global _last_tracer
    thread_id = _new_thread_id()
    awaiting_confirm = False
    while True:
        raw = sys.stdin.readline()
        if raw == "":  # EOF — end of the batch
            break
        text = raw.strip()
        if not text:
            continue
        if text in ("/exit", "/quit"):
            break

        # While a destructive op is pending, this line is the confirm/cancel reply.
        if awaiting_confirm:
            try:
                run_id = _new_thread_id()
                _last_tracer = start_run(run_id, "resume", thread_id, text)
                config = {"configurable": {"thread_id": thread_id}}
                result = graph.invoke(Command(resume=text), config=config)
                pending = _interrupt_obj(result, run_id)
                if pending:
                    _emit(pending)
                else:
                    _finalize_trace(_last_tracer, result, settings, deps)
                    _emit(_turn_obj(result, run_id))
                    awaiting_confirm = False
            except Exception as exc:  # noqa: BLE001 — keep the batch loop alive
                awaiting_confirm = False
                _emit({"type": "error", "message": str(exc)})
            continue

        # Inspection / control commands, answered as JSON (no graph turn).
        if text == "/help":
            _emit({"type": "help", "commands": _HELP})
            continue
        if text == "/new":
            thread_id = _new_thread_id()
            _emit({"type": "new_thread", "thread": thread_id})
            continue
        if text == "/whoami":
            _emit({"type": "whoami", "user": user_id, "thread": thread_id})
            continue
        if text == "/metrics":
            _emit({"type": "metrics", "summary": SESSION_METRICS.summary()})
            continue
        if text == "/trace":
            _emit({"type": "trace", "render": _last_tracer.render() if _last_tracer else None})
            continue
        if text == "/prefs":
            prefs = deps.profiles.get(user_id)
            _emit(
                {
                    "type": "prefs",
                    "user": user_id,
                    "preferences": prefs.preferences,
                    "updated_at": prefs.updated_at,
                }
            )
            continue
        if text.startswith("/login"):
            parts = text.split(maxsplit=1)
            if len(parts) == 2 and parts[1].strip():
                user_id = parts[1].strip()
                _emit({"type": "login", "user": user_id})
            else:
                _emit({"type": "error", "message": "usage: /login <user_id>"})
            continue
        if text == "/report" or text.startswith("/report "):
            parts = text.split(maxsplit=1)
            if len(parts) == 2 and parts[1].strip():
                report = deps.reports.get(parts[1].strip(), user_id)
                if report is None:
                    _emit({"type": "error", "message": f"no report with id {parts[1].strip()}"})
                else:
                    _emit(
                        {
                            "type": "report_view",
                            "id": report.id,
                            "title": report.title,
                            "created_at": report.created_at,
                            "content": report.content,
                        }
                    )
            else:
                _emit({"type": "error", "message": "usage: /report <id>"})
            continue
        if text == "/save":
            text = "save the last report"
        elif text == "/reports":
            text = "list my saved reports"
        elif text.startswith("/"):
            _emit({"type": "error", "message": f"unknown command: {text}"})
            continue

        # A normal turn through the graph.
        try:
            run_id = _new_thread_id()
            _last_tracer = start_run(run_id, user_id, thread_id, text)
            config = {"configurable": {"thread_id": thread_id}}
            result = graph.invoke(_initial_state(text, user_id, thread_id, run_id), config=config)
            pending = _interrupt_obj(result, run_id)
            if pending:
                _last_tracer.finalize(status="awaiting_confirmation")
                _last_tracer.save(settings.traces_dir)
                _emit(pending)
                awaiting_confirm = True
            else:
                _finalize_trace(_last_tracer, result, settings, deps)
                _emit(_turn_obj(result, run_id))
        except Exception as exc:  # noqa: BLE001 — keep the batch loop alive
            _emit({"type": "error", "message": str(exc)})


def main() -> None:
    parser = argparse.ArgumentParser(prog="assistant", description="Data Analysis Chat Assistant")
    parser.add_argument("--user", default=None, help="manager identity (default from config)")
    parser.add_argument(
        "--json",
        action="store_true",
        dest="json_mode",
        help="batch mode for agents/automation: read one turn per stdin line, "
        "emit one JSON object per line on stdout (the interactive REPL is unchanged)",
    )
    args = parser.parse_args()

    settings = get_settings()
    configure_logging(settings.log_level, settings.logs_dir)
    langsmith_key = (
        settings.langsmith_api_key.get_secret_value() if settings.langsmith_api_key else None
    )
    langsmith_on = enable_langsmith(langsmith_key)
    user_id = args.user or settings.default_user
    deps = AgentDeps.create(settings)
    graph = build_graph(deps=deps)

    # Non-interactive batch mode for agents/automation: JSON in, JSON out, with the
    # same single-process conversation memory as the REPL. Returns before any chrome.
    if args.json_mode:
        _run_batch_json(graph, settings, deps, user_id)
        return

    console = Console()
    console.print(
        Panel.fit(
            "[bold]Data Analysis Chat Assistant[/]\n"
            "Ask a business question about the retail data. Type [cyan]/help[/] for commands.",
            border_style="cyan",
        )
    )
    thread_id = _new_thread_id()
    obs = f"traces={settings.traces_dir}/" + ("  langsmith=on" if langsmith_on else "")
    console.print(
        f"[dim]user={user_id}  persona={settings.default_persona}  thread={thread_id}  {obs}[/]"
    )

    awaiting_confirm = False
    while True:
        prompt = "[bold red]confirm/cancel ›[/] " if awaiting_confirm else "[bold cyan]you ›[/] "
        try:
            text = console.input("\n" + prompt).strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]bye[/]")
            break

        if not text:
            continue
        if text in ("/exit", "/quit"):
            console.print("[dim]bye[/]")
            break

        # While a destructive op is pending, the next message is the confirm/cancel reply.
        if awaiting_confirm:
            try:
                awaiting_confirm = _resume_turn(graph, console, text, thread_id, settings, deps)
            except Exception as exc:  # noqa: BLE001 — keep the REPL alive on any error
                awaiting_confirm = False
                console.print(f"[red]Error:[/] {exc}")
            continue

        if text == "/help":
            console.print(_HELP)
            continue
        if text == "/trace":
            if _last_tracer is None:
                console.print("[dim]no turn to trace yet[/]")
            else:
                console.print(
                    Panel(
                        _last_tracer.render(),
                        title="Trace",
                        title_align="left",
                        border_style="blue",
                    )
                )
            continue
        if text == "/metrics":
            console.print(
                Panel(
                    SESSION_METRICS.summary(),
                    title="Metrics",
                    title_align="left",
                    border_style="blue",
                )
            )
            continue
        if text == "/new":
            thread_id = _new_thread_id()
            console.print(f"[dim]started new thread {thread_id}[/]")
            continue
        if text == "/whoami":
            console.print(f"[dim]user={user_id}  thread={thread_id}[/]")
            continue
        if text.startswith("/login"):
            parts = text.split(maxsplit=1)
            if len(parts) == 2 and parts[1].strip():
                user_id = parts[1].strip()
                console.print(f"[dim]now acting as {user_id}[/]")
            else:
                console.print("[yellow]usage: /login <user_id>[/]")
            continue
        if text == "/prefs":
            prefs = deps.profiles.get(user_id)
            console.print(
                f"[dim]preferences: {prefs.preferences or '(none set)'}"
                f"  · updated={prefs.updated_at or 'never'}[/]"
            )
            continue
        # /report <id> is a read-only convenience: fetch + show directly (owner-scoped,
        # no LLM), since the id is already explicit. NL ("show me the X report") still
        # goes through the graph's view path.
        if text == "/report" or text.startswith("/report "):
            parts = text.split(maxsplit=1)
            if len(parts) == 2 and parts[1].strip():
                report = deps.reports.get(parts[1].strip(), user_id)
                if report is None:
                    console.print(f"[yellow]No report with id {parts[1].strip()} (or not yours).[/]")
                else:
                    console.print(
                        Panel(
                            Markdown(report.content),
                            title=f"{report.title} · {report.id} · {report.created_at[:10]}",
                            title_align="left",
                            border_style="green",
                        )
                    )
            else:
                console.print("[yellow]usage: /report <id>  (see /reports for ids)[/]")
            continue
        # /save and /reports are conveniences that run the same graph path as the
        # equivalent natural-language commands (the CLI holds no business logic).
        if text == "/save":
            text = "save the last report"
        elif text == "/reports":
            text = "list my saved reports"
        elif text.startswith("/"):
            console.print(f"[yellow]unknown command:[/] {text}  (try /help)")
            continue

        try:
            awaiting_confirm = _run_turn(graph, console, text, user_id, thread_id, settings, deps)
        except Exception as exc:  # noqa: BLE001 — keep the REPL alive on any error
            console.print(f"[red]Error:[/] {exc}")


if __name__ == "__main__":
    main()
