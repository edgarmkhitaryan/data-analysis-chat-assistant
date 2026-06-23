"""The Trio — a human-vetted exemplar of how an analyst answered a question.

A Trio captures the *interpretation* (which tables, which revenue/profit
definition, which grouping, and what a good narrative looks like), not a cached
answer. At query time, relevant Trios are injected into the SQL- and
report-generation prompts as few-shot exemplars that the model adapts to the new
question — the heart of Requirement 1 (Hybrid Intelligence).
"""

from typing import Literal

from pydantic import BaseModel, Field


class Trio(BaseModel):
    """A question -> SQL -> report exemplar from the Golden Bucket."""

    id: str
    question: str
    sql: str
    report: str
    tags: list[str] = Field(default_factory=list)
    tables: list[str] = Field(default_factory=list)
    created_by: str = "analyst"
    created_at: str = ""
    quality: Literal["approved", "candidate", "rejected"] = "approved"
    source: Literal["seed", "learned"] = "seed"

    @property
    def embedding_text(self) -> str:
        """The text used to represent this Trio in the vector index."""
        return self.question
