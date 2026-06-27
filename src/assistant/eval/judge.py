"""LLM-as-judge for the eval harness (plan/011 §2.1).

Scores a report against a per-case rubric on two axes — does it answer the intent,
and is every claim supported by the data (no hallucination). Faithfulness is judged
against the **actual rows the report was built from** (already PII-masked), so the
judge can catch a fabricated figure rather than only internal inconsistency. Used
*carefully*: always paired with objective checks in the harness (execution, row
presence, regex for PII, and a `reference_sql` aggregate cross-check where provided),
so subjective scoring never stands alone on what can be measured.
"""

from __future__ import annotations

import json

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from assistant.config import Settings
from assistant.llm import get_chat_model, resilient_invoke

# Keep the data block fed to the judge bounded — analytic result sets are small, but a
# pathological query shouldn't blow up the judge prompt (cost + truncation by the model).
_MAX_DATA_ROWS = 50
_MAX_DATA_CHARS = 6000


class JudgeScore(BaseModel):
    """A structured, traceable judgement."""

    intent_satisfaction: int = Field(ge=0, le=5, description="does the report answer the question?")
    faithfulness: int = Field(ge=0, le=5, description="is every claim supported by the data?")
    justification: str = Field(description="one sentence explaining the scores")


_SYSTEM = (
    "You are a strict evaluator of a data-analysis assistant's answers. You are given the "
    "question, a rubric, the SQL the assistant ran, the ACTUAL DATA ROWS it returned "
    "(already PII-masked), and the report. Score two axes, each 0-5:\n"
    "- intent_satisfaction: does the report actually answer the question per the rubric?\n"
    "- faithfulness: is EVERY figure and claim in the report supported by the provided data "
    "rows? Penalize any number that does not trace to the data, and any place the report "
    "misreads or misattributes the data. If NO data rows are provided, judge faithfulness on "
    "internal consistency only and do not award a 5.\n"
    "Be critical; reserve 5 for genuinely excellent answers. Give a one-sentence justification."
)


def _format_data(rows: list[dict] | None) -> str:
    """Compact, judge-readable serialization of the (masked, PII-free) result rows."""
    if not rows:
        return "(no rows were returned to the report)"
    shown = rows[:_MAX_DATA_ROWS]
    text = json.dumps(shown, default=str, ensure_ascii=False)
    if len(text) > _MAX_DATA_CHARS:
        text = text[:_MAX_DATA_CHARS] + " …(truncated)"
    if len(rows) > _MAX_DATA_ROWS:
        text += f"\n(showing the first {_MAX_DATA_ROWS} of {len(rows)} rows)"
    return text


def judge_report(
    question: str,
    report: str,
    rubric: str,
    settings: Settings | None = None,
    *,
    data: list[dict] | None = None,
    sql: str | None = None,
) -> JudgeScore:
    """Score one report with the LLM judge against its rubric.

    ``data`` are the PII-masked rows the report was grounded in; passing them turns
    faithfulness into a real grounding check. ``sql`` is included for context. Both are
    optional and keyword-only so existing callers keep working (the judge then falls
    back to internal-consistency scoring for faithfulness).
    """
    chat = get_chat_model(temperature=0.0, settings=settings)
    parts = [f"Question: {question}", f"Rubric for a good answer: {rubric}"]
    if sql:
        parts.append(f"SQL the assistant ran:\n{sql}")
    parts.append(f"Data rows the report must be grounded in (PII-masked):\n{_format_data(data)}")
    parts.append(f"Report to evaluate:\n{report}")
    human = "\n\n".join(parts)
    return resilient_invoke(
        chat.with_structured_output(JudgeScore),
        [SystemMessage(content=_SYSTEM), HumanMessage(content=human)],
        settings=settings,
    )
