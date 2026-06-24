"""Assembles the agent graph.

Main flow (plan/005 §3): contextualize -> guard -> load -> route.

    START -> contextualize --(ambiguous)--> clarify -> END
                  |
              (resolved / first turn)
                  v
    guard_input -> load_context -> route --(update_preference)--> update_prefs -> END
                                     |
                                 (analysis)
                                     v
                  decompose -> run_compound -> synthesize -> END

``run_compound`` runs each sub-question (one for a simple question, several for a
compound one) through the reusable analysis pipeline below; ``synthesize`` then
merges the results (a single question passes straight through).

Analysis pipeline (a compiled subgraph, invoked once per sub-question):

    START -> retrieve_golden -> get_schema -> generate_sql -> validate_sql
                                                  |              |
                                              (invalid)      (error)
                                                  v              v
    validate_sql --(valid)--> execute_sql       degrade <--------'
    execute_sql --(rows)--> synthesize_report -> END
    degrade -> END

Phase 6 extends ``guard_input`` (injection pre-filter + manage_reports/rejected).
PII masking (Phase 5) inserts a ``mask_pii`` node on the execute->report edge.
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
from assistant.agent.nodes.report import synthesize_report
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


def _route_by_intent(state: AgentState) -> str:
    return "update_preference" if state.get("intent") == "update_preference" else "analysis"


def _after_validate(state: AgentState) -> str:
    return "degrade" if state.get("last_error") else "execute"


def _after_execute(state: AgentState) -> str:
    return "degrade" if state.get("last_error") else "report"


def build_analysis_pipeline(deps: AgentDeps):
    """Compile the per-question analysis pipeline (retrieve -> ... -> report).

    Reused by ``run_compound`` once per sub-question. Compiled without a
    checkpointer: each invocation is a transient, isolated sub-run within a turn.
    """
    builder = StateGraph(AgentState)
    builder.add_node("retrieve_golden", lambda state: retrieve_golden(state, deps))
    builder.add_node("get_schema", lambda state: get_schema(state, deps))
    builder.add_node("generate_sql", lambda state: generate_sql(state, deps))
    builder.add_node("validate_sql", validate_sql)
    builder.add_node("execute_sql", lambda state: execute_sql(state, deps))
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
        "execute_sql", _after_execute, {"report": "synthesize_report", "degrade": "degrade"}
    )
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
    builder.add_node("load_context", lambda state: load_context(state, deps))
    builder.add_node("update_prefs", lambda state: update_prefs(state, deps))
    builder.add_node("decompose", lambda state: decompose(state, deps))
    builder.add_node("run_compound", lambda state: run_compound(state, pipeline))
    builder.add_node("synthesize", lambda state: synthesize(state, deps))

    builder.add_edge(START, "contextualize")
    builder.add_conditional_edges(
        "contextualize", _after_contextualize, {"clarify": "clarify", "guard": "guard_input"}
    )
    builder.add_edge("clarify", END)
    builder.add_edge("guard_input", "load_context")
    builder.add_conditional_edges(
        "load_context",
        _route_by_intent,
        {"analysis": "decompose", "update_preference": "update_prefs"},
    )
    builder.add_edge("update_prefs", END)
    builder.add_edge("decompose", "run_compound")
    builder.add_edge("run_compound", "synthesize")
    builder.add_edge("synthesize", END)

    return builder.compile(checkpointer=checkpointer)
