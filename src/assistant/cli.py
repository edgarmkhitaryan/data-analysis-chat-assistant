"""Interactive CLI chat loop (deliverable 4).

A thin REPL: it marshals user input into the graph and renders the result. It
holds no business logic — every decision lives in the graph's nodes. Each turn
runs on a persisted ``thread_id``; the ``contextualize`` node uses that history
to resolve follow-ups, and clarifications are rendered distinctly.
"""

import argparse
import uuid

from langchain_core.messages import HumanMessage
from langgraph.types import Command
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from assistant.agent.dependencies import AgentDeps
from assistant.agent.graph import build_graph
from assistant.config import get_settings

_HELP = """\
Commands:
  /login <id>  switch the current manager (drives preferences + ownership)
  /prefs       show the current user's saved preferences
  /reports     list your saved reports
  /save        save the last report to your library
  /whoami      show the current user and thread
  /new         start a new conversation thread
  /help        show this help
  /exit        quit"""


def _new_thread_id() -> str:
    return uuid.uuid4().hex[:12]


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


def _run_turn(graph, console: Console, question: str, user_id: str, thread_id: str) -> bool:
    """Run one turn; return True if it paused awaiting a confirm/cancel reply."""
    initial = {
        "messages": [HumanMessage(content=question)],
        "raw_question": question,
        "question": question,
        "user_id": user_id,
        "thread_id": thread_id,
        "run_id": _new_thread_id(),
        "sql_attempts": 0,
        "last_error": None,
    }
    config = {"configurable": {"thread_id": thread_id}}
    with console.status("[dim]thinking…[/]", spinner="dots"):
        result = graph.invoke(initial, config=config)
    if _show_interrupt(console, result):
        return True
    _render(console, result, question)
    return False


def _resume_turn(graph, console: Console, reply: str, thread_id: str) -> bool:
    """Resume a paused (interrupted) turn with the user's confirm/cancel reply."""
    config = {"configurable": {"thread_id": thread_id}}
    with console.status("[dim]working…[/]", spinner="dots"):
        result = graph.invoke(Command(resume=reply), config=config)
    if _show_interrupt(console, result):
        return True
    _render(console, result, reply)
    return False


def main() -> None:
    parser = argparse.ArgumentParser(prog="assistant", description="Data Analysis Chat Assistant")
    parser.add_argument("--user", default=None, help="manager identity (default from config)")
    args = parser.parse_args()

    settings = get_settings()
    user_id = args.user or settings.default_user
    console = Console()

    console.print(
        Panel.fit(
            "[bold]Data Analysis Chat Assistant[/]\n"
            "Ask a business question about the retail data. Type [cyan]/help[/] for commands.",
            border_style="cyan",
        )
    )
    deps = AgentDeps.create(settings)
    graph = build_graph(deps=deps)
    thread_id = _new_thread_id()
    console.print(f"[dim]user={user_id}  persona={settings.default_persona}  thread={thread_id}[/]")

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
                awaiting_confirm = _resume_turn(graph, console, text, thread_id)
            except Exception as exc:  # noqa: BLE001 — keep the REPL alive on any error
                awaiting_confirm = False
                console.print(f"[red]Error:[/] {exc}")
            continue

        if text == "/help":
            console.print(_HELP)
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
                f"[dim]format={prefs.format}  verbosity={prefs.verbosity}  "
                f"updated={prefs.updated_at or 'never'}[/]"
            )
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
            awaiting_confirm = _run_turn(graph, console, text, user_id, thread_id)
        except Exception as exc:  # noqa: BLE001 — keep the REPL alive on any error
            console.print(f"[red]Error:[/] {exc}")


if __name__ == "__main__":
    main()
