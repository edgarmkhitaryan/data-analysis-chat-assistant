"""Assembles the analysis graph (Phase 2 happy path).

Flow:  retrieve -> get_schema -> generate_sql -> validate_sql -> execute_sql -> report
with a graceful-degradation exit if validation or execution fails.

    START -> retrieve_golden -> get_schema -> generate_sql -> validate_sql --(valid)--> execute_sql
                                                  |                      |
                                              (invalid)              (error)
                                                  v                      v
                                               degrade <----------------'
    execute_sql --(rows)--> synthesize_report -> END
    degrade -> END

Later phases insert nodes around this core (contextualize/clarify before it,
retrieve before generate_sql, mask before report, routing/oversight branches)
without changing the pieces defined here.
"""

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from assistant.agent.dependencies import AgentDeps
from assistant.agent.nodes.degrade import degrade
from assistant.agent.nodes.execute_sql import execute_sql
from assistant.agent.nodes.generate_sql import generate_sql
from assistant.agent.nodes.report import synthesize_report
from assistant.agent.nodes.retrieve import retrieve_golden
from assistant.agent.nodes.schema import get_schema
from assistant.agent.nodes.validate_sql import validate_sql
from assistant.agent.state import AgentState


def _after_validate(state: AgentState) -> str:
    return "degrade" if state.get("last_error") else "execute"


def _after_execute(state: AgentState) -> str:
    return "degrade" if state.get("last_error") else "report"


def build_graph(
    deps: AgentDeps | None = None,
    checkpointer: BaseCheckpointSaver | None = None,
):
    """Build and compile the analysis graph.

    Args:
        deps: Injected resources (runner + settings). Defaults to the real ones.
        checkpointer: Conversation-state store. Defaults to an in-memory saver so
            multi-turn follow-ups work within a session.
    """
    deps = deps or AgentDeps.create()
    if checkpointer is None:
        checkpointer = InMemorySaver()

    builder = StateGraph(AgentState)

    # Nodes that need shared resources are wrapped so each is a plain `state -> dict`.
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
        "validate_sql",
        _after_validate,
        {"execute": "execute_sql", "degrade": "degrade"},
    )
    builder.add_conditional_edges(
        "execute_sql",
        _after_execute,
        {"report": "synthesize_report", "degrade": "degrade"},
    )
    builder.add_edge("synthesize_report", END)
    builder.add_edge("degrade", END)

    return builder.compile(checkpointer=checkpointer)
