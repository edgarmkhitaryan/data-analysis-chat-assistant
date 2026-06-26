"""Per-user preference store (Requirement 4.1 — user-level memory).

Each manager's report preferences are stored as a single **compact, free-form
description** (e.g. *"Concise bullet points; always show % change vs the prior
period; focus on revenue and margin; currency in USD"*). Free-form text means
*any* kind of preference can be captured and applied — not a fixed set of fields.

It is read on every analysis turn so reports honor the user's choices, and
rewritten synchronously when the user states a standing preference — so the change
applies on the very next question and survives restarts.
"""

import sqlite3
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


class UserPrefs(BaseModel):
    """A single manager's standing report preferences, as compact free-form text."""

    user_id: str
    preferences: str = ""
    updated_at: str = ""


class ProfileStore:
    """SQLite-backed store for :class:`UserPrefs`, keyed by user id."""

    def __init__(self, db_path: str) -> None:
        self._db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            self._ensure_schema(conn)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        return conn

    @staticmethod
    def _ensure_schema(conn: sqlite3.Connection) -> None:
        """Create the table, with a one-time migration from the old structured columns.

        Earlier versions stored ``format``/``verbosity``/``favorite_metrics``. If such a
        table is found we fold each row into a free-form sentence and rebuild, so existing
        preferences are preserved rather than dropped.
        """
        cols = {row[1] for row in conn.execute("PRAGMA table_info(user_profiles)")}
        legacy: list = []
        if cols and "preferences" not in cols:
            legacy = conn.execute("SELECT user_id, format, verbosity FROM user_profiles").fetchall()
            conn.execute("DROP TABLE user_profiles")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS user_profiles (
                user_id     TEXT PRIMARY KEY,
                preferences TEXT NOT NULL,
                updated_at  TEXT NOT NULL
            )
            """
        )
        for row in legacy:
            text = f"Prefers {row['format']} format, {row['verbosity']} verbosity."
            conn.execute(
                "INSERT OR IGNORE INTO user_profiles (user_id, preferences, updated_at) "
                "VALUES (?, ?, ?)",
                (row["user_id"], text, _now()),
            )

    def get(self, user_id: str) -> UserPrefs:
        """Return the user's preferences, or empty defaults if none are stored."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM user_profiles WHERE user_id = ?", (user_id,)
            ).fetchone()
        if row is None:
            return UserPrefs(user_id=user_id)
        return UserPrefs(
            user_id=row["user_id"],
            preferences=row["preferences"] or "",
            updated_at=row["updated_at"],
        )

    def set_preferences(self, user_id: str, preferences: str) -> UserPrefs:
        """Persist the user's compact free-form preferences; returns the stored result."""
        prefs = UserPrefs(user_id=user_id, preferences=preferences.strip(), updated_at=_now())
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO user_profiles (user_id, preferences, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(user_id) DO UPDATE SET
                    preferences = excluded.preferences,
                    updated_at = excluded.updated_at
                """,
                (prefs.user_id, prefs.preferences, prefs.updated_at),
            )
        return prefs
