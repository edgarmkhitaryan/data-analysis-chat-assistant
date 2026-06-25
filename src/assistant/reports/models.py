"""The Saved Reports domain model (plan/007 §4).

A ``SavedReport`` is the only thing the agent can *write*, and every read/write is
scoped by ``owner_id`` — a manager can only ever see or delete their own reports.
"""

from pydantic import BaseModel, Field


class SavedReport(BaseModel):
    """One report a manager saved to their library."""

    id: str
    owner_id: str
    title: str
    content: str
    clients: list[str] = Field(default_factory=list)
    created_at: str
