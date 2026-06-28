"""Compound-question handling: split into sub-questions and run each (plan/005 §3).

``decompose`` decides whether the question contains multiple distinct analytical
asks and, if so, splits it into self-contained sub-questions (a single ask passes
through as one). ``run_compound`` then runs each sub-question through the reusable
analysis pipeline independently — one failing sub-question does not sink the rest
(partial-failure tolerant); ``synthesize`` merges the results afterwards.
"""

import logging

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from assistant.agent.dependencies import AgentDeps
from assistant.agent.state import AgentState, SubResult
from assistant.llm import get_chat_model, resilient_invoke
from assistant.observability import get_tracer, summarize_delta

logger = logging.getLogger(__name__)


def _run_pipeline(pipeline, sub_state: dict, tracer, sub_run_id: str | None) -> dict:
    """Run the analysis pipeline once, recording each node's step into the trace.

    When a tracer is active we ``stream`` the subgraph so every inner node (retrieve,
    generate_sql attempts, execute errors, self_correct, mask, report) becomes an
    ordered trace event with timing (plan/009 §2); otherwise we ``invoke`` plainly.
    The final sub-state is reconstructed from the streamed deltas.
    """
    if tracer is None:
        return pipeline.invoke(sub_state)
    final: dict = {}
    sub_field = {"sub": sub_run_id} if sub_run_id else {}
    for update in pipeline.stream(sub_state, stream_mode="updates"):
        for node, delta in update.items():
            tracer.event(node, **sub_field, **summarize_delta(node, delta))
            if isinstance(delta, dict):
                final.update(delta)
    return final


class Decomposition(BaseModel):
    """Structured result of compound-question detection."""

    is_compound: bool = Field(description="true if the question has multiple distinct asks")
    sub_questions: list[str] = Field(
        default_factory=list, description="self-contained sub-questions (each answerable alone)"
    )


_SYSTEM = (
    "You analyze a retail manager's question and decide whether it contains MULTIPLE "
    "distinct analytical asks that should be answered separately (for example: 'top "
    "products by revenue AND how does California compare to Texas').\n\n"
    "If it is compound, set is_compound=true and split it into 2 or more self-contained "
    "sub-questions, each a complete standalone question answerable on its own. If it is a "
    "single ask, set is_compound=false and return that one question unchanged."
)


def decompose(state: AgentState, deps: AgentDeps) -> dict:
    """Detect a compound question and split it into self-contained sub-questions."""
    question = state["question"]
    chat = get_chat_model(temperature=0.0, settings=deps.settings, cheap=True)
    try:
        result: Decomposition = resilient_invoke(
            chat.with_structured_output(Decomposition),
            [SystemMessage(content=_SYSTEM), HumanMessage(content=question)],
            settings=deps.settings,
        )
    except Exception as exc:  # noqa: BLE001 — fall back to a single question on failure
        logger.warning("Decompose failed (%s); treating as a single question", exc)
        return {"is_compound": False, "sub_questions": [question]}

    subs = [s.strip() for s in result.sub_questions if s.strip()]
    if not result.is_compound or len(subs) <= 1:
        return {"is_compound": False, "sub_questions": [question]}

    capped = subs[: deps.settings.max_sub_questions]
    logger.info("Decomposed into %d sub-questions", len(capped))
    return {"is_compound": True, "sub_questions": capped}


def run_compound(state: AgentState, pipeline) -> dict:
    """Run each sub-question through the analysis pipeline, collecting results.

    ``pipeline`` is the compiled analysis subgraph. Each sub-question runs in an
    isolated state with its own retry budget and ``sub_run_id``; a failure is
    captured (not raised) so the other sub-questions still complete.
    """
    sub_questions = state.get("sub_questions") or [state.get("question", "")]
    run_id = state.get("run_id", "run")
    persona = state.get("persona")
    user_prefs = state.get("user_prefs")
    user_id = state.get("user_id")
    # A one-off preference ("...as bullets just this once") applies to this turn's report.
    # It must be passed into the subgraph so the per-question report node honors it on the
    # common single-question path (the compound merge already runs at the top level).
    oneoff_preference = state.get("oneoff_preference")
    tracer = get_tracer()
    is_compound = bool(state.get("is_compound"))

    results: list[SubResult] = []
    for index, sub_question in enumerate(sub_questions, start=1):
        sub_run_id = f"{run_id}-{index}"
        sub_state = {
            "question": sub_question,
            "raw_question": sub_question,
            "persona": persona,
            "user_prefs": user_prefs,
            "user_id": user_id,
            "oneoff_preference": oneoff_preference,
            "sql_attempts": 0,
            "last_error": None,
            "messages": [],
        }
        try:
            out = _run_pipeline(pipeline, sub_state, tracer, sub_run_id if is_compound else None)
        except Exception as exc:  # noqa: BLE001 — isolate sub-question failures
            logger.warning("Sub-question failed (%r): %s", sub_question, exc)
            results.append(
                SubResult(
                    sub_question=sub_question,
                    sub_run_id=sub_run_id,
                    sql=None,
                    report=None,
                    row_count=0,
                    error=str(exc),
                )
            )
            continue
        results.append(
            SubResult(
                sub_question=sub_question,
                sub_run_id=sub_run_id,
                sql=out.get("generated_sql"),
                report=out.get("report"),
                row_count=out.get("row_count", 0),
                error=out.get("last_error"),
                masked_rows=out.get("masked_rows") or [],
                pii_masked_count=out.get("pii_masked_count") or 0,
            )
        )
    return {"sub_results": results}
