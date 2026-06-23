"""The agent's shared state — the single object that flows through the graph.

LangGraph merges each node's returned dict into this state (the ``messages`` key
uses the ``add_messages`` reducer to append rather than overwrite). The schema
below is intentionally the *Phase 2* subset of the full design in plan/005; later
phases add their own fields (contextualization, routing, compound questions, PII,
oversight, persona/prefs) without disturbing these.
"""

from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

from assistant.golden.models import Trio


class AgentState(TypedDict, total=False):
    """Typed conversation + analysis state for one thread.

    ``total=False`` makes every field optional, so nodes only declare the keys
    they actually produce.
    """

    # --- Conversation / identity ---
    messages: Annotated[list[BaseMessage], add_messages]
    user_id: str
    thread_id: str
    run_id: str

    # --- The question under analysis ---
    question: str

    # --- Hybrid Intelligence (Golden Bucket) ---
    retrieved_trios: list[Trio]
    retrieval_cold: bool

    # --- Analysis pipeline ---
    schema_context: str
    generated_sql: str
    sql_attempts: int
    last_error: str | None
    raw_rows: list[dict]
    row_count: int

    # --- Output ---
    report: str
