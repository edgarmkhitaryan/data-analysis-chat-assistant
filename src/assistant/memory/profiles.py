"""Per-user preference store (Requirement 4.1 — user-level memory).

A tiny SQLite-backed store of each manager's report preferences (format,
verbosity, favorite metrics). It is read on every analysis turn so reports honor
the user's choices, and written synchronously when the user states a standing
preference — so the change applies on the very next turn and survives restarts.
"""

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


class UserPrefs(BaseModel):
    """A single manager's standing report preferences."""

    user_id: str
    format: Literal["table", "bullets", "prose"] = "prose"
    verbosity: Literal["concise", "detailed"] = "concise"
    favorite_metrics: list[str] = Field(default_factory=list)
    updated_at: str = ""


class ProfileStore:
    """SQLite-backed CRUD for :class:`UserPrefs`, keyed by user id."""

    def __init__(self, db_path: str) -> None:
        self._db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS user_profiles (
                    user_id          TEXT PRIMARY KEY,
                    format           TEXT NOT NULL,
                    verbosity        TEXT NOT NULL,
                    favorite_metrics TEXT NOT NULL,
                    updated_at       TEXT NOT NULL
                )
                """
            )

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def get(self, user_id: str) -> UserPrefs:
        """Return the user's preferences, or sensible defaults if none are stored."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM user_profiles WHERE user_id = ?", (user_id,)
            ).fetchone()
        if row is None:
            return UserPrefs(user_id=user_id)
        return UserPrefs(
            user_id=row["user_id"],
            format=row["format"],
            verbosity=row["verbosity"],
            favorite_metrics=json.loads(row["favorite_metrics"]),
            updated_at=row["updated_at"],
        )

    def update(self, user_id: str, **changes: object) -> UserPrefs:
        """Apply changes to the user's preferences and persist them; returns the result."""
        merged = self.get(user_id).model_copy(update=changes)
        merged.updated_at = datetime.now(UTC).isoformat(timespec="seconds")
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO user_profiles (user_id, format, verbosity, favorite_metrics, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(user_id) DO UPDATE SET
                    format = excluded.format,
                    verbosity = excluded.verbosity,
                    favorite_metrics = excluded.favorite_metrics,
                    updated_at = excluded.updated_at
                """,
                (
                    merged.user_id,
                    merged.format,
                    merged.verbosity,
                    json.dumps(merged.favorite_metrics),
                    merged.updated_at,
                ),
            )
        return merged
