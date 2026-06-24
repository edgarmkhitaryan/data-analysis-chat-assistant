"""The agent's shared state — the single object that flows through the graph.

LangGraph merges each node's returned dict into this state (the ``messages`` key
uses the ``add_messages`` reducer to append rather than overwrite). The schema
below is intentionally the *Phase 2* subset of the full design in plan/005; later
phases add their own fields (contextualization, routing, compound questions, PII,
oversight, persona/prefs) without disturbing these.
"""

from typing import Annotated, Literal, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

from assistant.golden.models import Trio
from assistant.memory.profiles import UserPrefs
from assistant.persona.loader import Persona


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

    # --- Contextualization (follow-up -> standalone) ---
    raw_question: str
    question: str
    history_used: bool
    needs_clarification: bool
    clarifying_question: str | None

    # --- Routing & preferences ---
    intent: Literal["analysis", "update_preference"]
    pref_update: dict | None

    # --- Persona (org tone) + user preferences (format/verbosity) ---
    persona: Persona
    user_prefs: UserPrefs

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
