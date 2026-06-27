"""The golden eval set — curated cases spanning every expected capability (plan/011 §2).

Each case carries an ``intent`` (expected guard classification), a ``rubric`` for the
LLM-as-judge, and an optional ``expect`` for objective safety checks (refused / no_pii).
``conversational`` cases provide multi-turn ``turns`` that must resolve via context.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_CASES_PATH = "tests/eval/golden_set.json"


@dataclass
class EvalCase:
    """One golden evaluation case."""

    id: str
    kind: str  # "analysis" | "conversational" | "adversarial"
    intent: str
    rubric: str
    question: str | None = None
    turns: list[str] | None = None
    expect: dict = field(default_factory=dict)
    # Optional objective correctness cross-check: an authoritative query whose numeric
    # aggregates must be reproduced in the agent's result (plan/011 §2 / eval.correctness).
    reference_sql: str | None = None
    tolerance: float = 0.01  # relative tolerance when matching reference aggregates

    @property
    def final_question(self) -> str:
        """The question the judge should score (the last turn for conversational cases)."""
        if self.turns:
            return self.turns[-1]
        return self.question or ""


def load_cases(path: str = DEFAULT_CASES_PATH) -> list[EvalCase]:
    """Load the golden eval set from a JSON file."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return [
        EvalCase(
            id=case["id"],
            kind=case["kind"],
            intent=case.get("intent", "analysis"),
            rubric=case.get("rubric", ""),
            question=case.get("question"),
            turns=case.get("turns"),
            expect=case.get("expect", {}),
            reference_sql=case.get("reference_sql"),
            tolerance=case.get("tolerance", 0.01),
        )
        for case in data
    ]
