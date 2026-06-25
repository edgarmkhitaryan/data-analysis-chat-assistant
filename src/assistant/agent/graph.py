r"""Assembles the agent graph.

Main flow (plan/005 §3): contextualize -> guard -> load -> route.

    START -> contextualize --(ambiguous)--> clarify -> END
                  |
              (resolved / first turn)
                  v
    guard_input --(rejected)--> respond_reject -> END
                  |
                (ok)
                  v
    load_context -> route --(update_preference)--> update_prefs --(preference only)--> END
                      |                                  |
                  (analysis)                     (also asks a question)
                      v                                  |
                  decompose <----------------------------'
                  decompose -> run_compound -> synthesize -> END

The guard refuses off-topic/injection turns (``respond_reject``). A standing
preference is persisted by ``update_prefs`` as a side-effect: a preference-only
turn ends there, but a combined "set preference + ask analysis" turn continues
into ``decompose`` with the new preference applied. ``run_compound`` runs each
sub-question (one for a simple question, several for a compound one) through the
reusable analysis pipeline below; ``synthesize`` merges the results.

The ``manage_reports`` intent enters the oversight path (plan/007 §4):

    route --(manage_reports)--> parse_report_command --(save)--> save_report -> END
                                       |        \--(list)--> list_reports -> END
                                    (delete)
                                       v
    resolve_targets --(none)--> respond_none -> END
        \--(matched)--> confirm_delete  [interrupt() -> CLI confirm] -> END

Only **delete** is destructive: it resolves owner-scoped targets, then pauses on a
``interrupt()`` and mutates only on an explicit confirm (audited). save/list run
directly.

Analysis pipeline (a compiled subgraph, invoked once per sub-question):

    START -> retrieve_golden -> get_schema -> generate_sql -> validate_sql
                                                  |              |
                                              (invalid)      (error)
                                                  v              v
    validate_sql --(valid)--> execute_sql       degrade <--------'
    execute_sql --(rows)--> mask_pii -> synthesize_report -> END
    degrade -> END

``mask_pii`` (Phase 5) deterministically strips PII from the rows so the report
LLM only ever sees ``masked_rows``; the output guard in ``synthesize`` re-scans
the final report text as a last line of defense.

Phase 6 extends ``guard_input`` (injection pre-filter + manage_reports/rejected).
"""

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph import END, START, StateGraph

from assistant.agent.dependencies import AgentDeps
from assistant.agent.nodes.clarify import clarify
from assistant.agent.nodes.contextualize import contextualize
from assistant.agent.nodes.decompose import decompose, run_compound
from assistant.agent.nodes.degrade import degrade
from assistant.agent.nodes.execute_sql import execute_sql
from assistant.agent.nodes.generate_sql import generate_sql
from assistant.agent.nodes.guard import guard_input
from assistant.agent.nodes.load_context import load_context
from assistant.agent.nodes.mask_pii import mask_pii
from assistant.agent.nodes.reject import respond_reject
from assistant.agent.nodes.report import synthesize_report
from assistant.agent.nodes.reports_cmd import (
    confirm_delete,
    list_reports,
    parse_report_command,
    resolve_targets,
    respond_none,
    save_report,
)
from assistant.agent.nodes.retrieve import retrieve_golden
from assistant.agent.nodes.schema import get_schema
from assistant.agent.nodes.synthesize import synthesize
from assistant.agent.nodes.update_prefs import update_prefs
from assistant.agent.nodes.validate_sql import validate_sql
from assistant.agent.state import AgentState

# Our typed state values (pydantic) must be explicitly allow-listed for the
# checkpointer's msgpack (de)serialization — otherwise LangGraph warns on every
# multi-turn read and will block them in a future version.
_ALLOWED_STATE_TYPES = {
    ("assistant.persona.loader", "Persona"),
    ("assistant.memory.profiles", "UserPrefs"),
    ("assistant.golden.models", "Trio"),
}


def _state_serde() -> JsonPlusSerializer:
    return JsonPlusSerializer(allowed_msgpack_modules=_ALLOWED_STATE_TYPES)


def _after_contextualize(state: AgentState) -> str:
    return "clarify" if state.get("needs_clarification") else "guard"


def _after_guard(state: AgentState) -> str:
    return "reject" if state.get("intent") == "rejected" else "ok"


def _route_by_intent(state: AgentState) -> str:
    intent = state.get("intent")
    if intent == "update_preference":
        return "update_preference"
    if intent == "manage_reports":
        return "manage_reports"
    return "analysis"


def _after_update_prefs(state: AgentState) -> str:
    # Combined intent: a standing preference that also asked a question continues
    # into the analysis; a preference-only turn ends here.
    return "analysis" if state.get("also_analysis") else "end"


def _after_parse_report(state: AgentState) -> str:
    return state.get("report_action") or "list"


def _after_resolve(state: AgentState) -> str:
    return "confirm" if state.get("pending_action") else "none"


def _after_validate(state: AgentState) -> str:
    return "degrade" if state.get("last_error") else "execute"


def _after_execute(state: AgentState) -> str:
    return "degrade" if state.get("last_error") else "mask"


def build_analysis_pipeline(deps: AgentDeps):
    """Compile the per-question analysis pipeline (retrieve -> ... -> report).

    Reused by ``run_compound`` once per sub-question. Compiled without a
    checkpointer: each invocation is a transient, isolated sub-run within a turn.
    """
    builder = StateGraph(AgentState)
    builder.add_node("retrieve_golden", lambda state: retrieve_golden(state, deps))
    builder.add_node("get_schema", lambda state: get_schema(state, deps))
    builder.add_node("generate_sql", lambda state: generate_sql(state, deps))
    builder.add_node("validate_sql", lambda state: validate_sql(state, deps))
    builder.add_node("execute_sql", lambda state: execute_sql(state, deps))
    builder.add_node("mask_pii", lambda state: mask_pii(state, deps))
    builder.add_node("synthesize_report", lambda state: synthesize_report(state, deps))
    builder.add_node("degrade", degrade)

    builder.add_edge(START, "retrieve_golden")
    builder.add_edge("retrieve_golden", "get_schema")
    builder.add_edge("get_schema", "generate_sql")
    builder.add_edge("generate_sql", "validate_sql")
    builder.add_conditional_edges(
        "validate_sql", _after_validate, {"execute": "execute_sql", "degrade": "degrade"}
    )
    builder.add_conditional_edges(
        "execute_sql", _after_execute, {"mask": "mask_pii", "degrade": "degrade"}
    )
    builder.add_edge("mask_pii", "synthesize_report")
    builder.add_edge("synthesize_report", END)
    builder.add_edge("degrade", END)
    return builder.compile()


def build_graph(
    deps: AgentDeps | None = None,
    checkpointer: BaseCheckpointSaver | None = None,
):
    """Build and compile the agent graph.

    Args:
        deps: Injected resources (runner + retriever + profiles + settings).
        checkpointer: Conversation-state store. Defaults to an in-memory saver so
            multi-turn follow-ups work within a session.
    """
    deps = deps or AgentDeps.create()
    if checkpointer is None:
        checkpointer = InMemorySaver(serde=_state_serde())

    pipeline = build_analysis_pipeline(deps)

    builder = StateGraph(AgentState)
    builder.add_node("contextualize", lambda state: contextualize(state, deps))
    builder.add_node("clarify", clarify)
    builder.add_node("guard_input", lambda state: guard_input(state, deps))
    builder.add_node("respond_reject", respond_reject)
    builder.add_node("load_context", lambda state: load_context(state, deps))
    builder.add_node("update_prefs", lambda state: update_prefs(state, deps))
    builder.add_node("parse_report_command", lambda state: parse_report_command(state, deps))
    builder.add_node("save_report", lambda state: save_report(state, deps))
    builder.add_node("list_reports", lambda state: list_reports(state, deps))
    builder.add_node("resolve_targets", lambda state: resolve_targets(state, deps))
    builder.add_node("respond_none", respond_none)
    builder.add_node("confirm_delete", lambda state: confirm_delete(state, deps))
    builder.add_node("decompose", lambda state: decompose(state, deps))
    builder.add_node("run_compound", lambda state: run_compound(state, pipeline))
    builder.add_node("synthesize", lambda state: synthesize(state, deps))

    builder.add_edge(START, "contextualize")
    builder.add_conditional_edges(
        "contextualize", _after_contextualize, {"clarify": "clarify", "guard": "guard_input"}
    )
    builder.add_edge("clarify", END)
    builder.add_conditional_edges(
        "guard_input", _after_guard, {"reject": "respond_reject", "ok": "load_context"}
    )
    builder.add_edge("respond_reject", END)
    builder.add_conditional_edges(
        "load_context",
        _route_by_intent,
        {
            "analysis": "decompose",
            "update_preference": "update_prefs",
            "manage_reports": "parse_report_command",
        },
    )
    builder.add_conditional_edges(
        "update_prefs", _after_update_prefs, {"analysis": "decompose", "end": END}
    )

    # Report-management (oversight) path.
    builder.add_conditional_edges(
        "parse_report_command",
        _after_parse_report,
        {"save": "save_report", "list": "list_reports", "delete": "resolve_targets"},
    )
    builder.add_edge("save_report", END)
    builder.add_edge("list_reports", END)
    builder.add_conditional_edges(
        "resolve_targets", _after_resolve, {"confirm": "confirm_delete", "none": "respond_none"}
    )
    builder.add_edge("respond_none", END)
    builder.add_edge("confirm_delete", END)

    builder.add_edge("decompose", "run_compound")
    builder.add_edge("run_compound", "synthesize")
    builder.add_edge("synthesize", END)

    return builder.compile(checkpointer=checkpointer)
