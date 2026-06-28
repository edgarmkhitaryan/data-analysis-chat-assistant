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


class SubResult(TypedDict, total=False):
    """The outcome of running one sub-question through the analysis pipeline."""

    sub_question: str
    sub_run_id: str
    sql: str | None
    report: str | None
    row_count: int
    error: str | None
    # Masked rows the sub-report stood on, surfaced to the parent turn to ground the judge.
    masked_rows: list[dict]
    pii_masked_count: int


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
    intent: Literal["analysis", "manage_reports", "update_preference", "rejected"]
    rejection_reason: str | None
    # A standing preference the user stated, in their own words (free-form), to be merged
    # into their stored compact preferences by the update_prefs node.
    pref_update: str | None
    # Combined intent (Phase 6): a standing preference *and* an analysis question in
    # one message persists the pref AND continues into the analysis.
    also_analysis: bool
    # A one-off preference ("...as bullets just this once") applied to this turn only.
    oneoff_preference: str | None
    # Acknowledgement prepended to a combined turn's report ("Saved your preference…").
    pref_saved_note: str | None

    # --- Compound questions (decompose -> run_compound -> synthesize) ---
    is_compound: bool
    sub_questions: list[str]
    sub_results: list["SubResult"]

    # --- Report management / oversight (manage_reports path) ---
    report_action: Literal["save", "list", "view", "delete"] | None
    report_filters: dict | None
    # Parsed-but-unexecuted destructive op, carried across the confirm interrupt:
    # {action, filters, target_ids, summary}.
    pending_action: dict | None

    # --- Persona (org tone) + user preferences (compact free-form text) ---
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
    empty_retried: bool  # whether the one guided "0 rows" reformulation has been used
    raw_rows: list[dict]
    masked_rows: list[dict]
    row_count: int

    # --- Safety: PII masking (deterministic) ---
    pii_masked_count: int
    pii_leak_prevented: int

    # --- Output ---
    report: str
