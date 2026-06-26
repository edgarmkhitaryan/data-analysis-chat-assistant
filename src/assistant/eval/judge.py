"""LLM-as-judge for the eval harness (plan/011 §2.1).

Scores a report against a per-case rubric on two axes — does it answer the intent,
and is every claim supported by the data (no hallucination). Used *carefully*:
always paired with objective checks in the harness (execution, row presence, regex
for PII), so subjective scoring never stands alone on what can be measured.
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from assistant.config import Settings
from assistant.llm import get_chat_model, resilient_invoke


class JudgeScore(BaseModel):
    """A structured, traceable judgement."""

    intent_satisfaction: int = Field(ge=0, le=5, description="does the report answer the question?")
    faithfulness: int = Field(ge=0, le=5, description="is every claim supported by the data?")
    justification: str = Field(description="one sentence explaining the scores")


_SYSTEM = (
    "You are a strict evaluator of a data-analysis assistant's answers. Score two axes, "
    "each 0-5:\n"
    "- intent_satisfaction: does the report actually answer the question per the rubric?\n"
    "- faithfulness: is every claim/figure supported by an analysis (no fabricated numbers)?\n"
    "Be critical; reserve 5 for genuinely excellent answers. Give a one-sentence justification."
)


def judge_report(
    question: str, report: str, rubric: str, settings: Settings | None = None
) -> JudgeScore:
    """Score one report with the LLM judge against its rubric."""
    chat = get_chat_model(temperature=0.0, settings=settings)
    human = (
        f"Question: {question}\n\n"
        f"Rubric for a good answer: {rubric}\n\n"
        f"Report to evaluate:\n{report}"
    )
    return resilient_invoke(
        chat.with_structured_output(JudgeScore),
        [SystemMessage(content=_SYSTEM), HumanMessage(content=human)],
        settings=settings,
    )
