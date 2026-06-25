"""Saved Reports store + audit log (plan/007 §4) — the agent's only writable store.

SQLite-backed CRUD for :class:`SavedReport`, mirroring the ProfileStore pattern.
**Every read and write is scoped by ``owner_id``** so a manager can never see or
delete another manager's reports (Requirement 3). Destructive actions are recorded
in an append-only ``audit_log`` (who / what / when).

On first use an empty library is seeded from ``data/seed_reports/*.json`` so the
oversight demo has real reports to act on.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import uuid
from datetime import UTC, date, datetime
from pathlib import Path

from assistant.reports.models import SavedReport

logger = logging.getLogger(__name__)


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


class SavedReportStore:
    """SQLite-backed, ownership-scoped store for saved reports + an audit log."""

    def __init__(self, db_path: str, seed_dir: str | None = None) -> None:
        self._db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS saved_reports (
                    id         TEXT PRIMARY KEY,
                    owner_id   TEXT NOT NULL,
                    title      TEXT NOT NULL,
                    content    TEXT NOT NULL,
                    clients    TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS audit_log (
                    id     INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts     TEXT NOT NULL,
                    actor  TEXT NOT NULL,
                    action TEXT NOT NULL,
                    detail TEXT NOT NULL
                )
                """
            )
        if seed_dir:
            self._seed_if_empty(seed_dir)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        return conn

    @staticmethod
    def _row_to_report(row: sqlite3.Row) -> SavedReport:
        return SavedReport(
            id=row["id"],
            owner_id=row["owner_id"],
            title=row["title"],
            content=row["content"],
            clients=json.loads(row["clients"]),
            created_at=row["created_at"],
        )

    # --- Writes ---------------------------------------------------------------

    def save(
        self,
        owner_id: str,
        *,
        title: str,
        content: str,
        clients: list[str] | None = None,
        created_at: str | None = None,
    ) -> SavedReport:
        """Persist a new report for ``owner_id`` and return it."""
        report = SavedReport(
            id=uuid.uuid4().hex[:8],
            owner_id=owner_id,
            title=title,
            content=content,
            clients=clients or [],
            created_at=created_at or _now(),
        )
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO saved_reports (id, owner_id, title, content, clients, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    report.id,
                    report.owner_id,
                    report.title,
                    report.content,
                    json.dumps(report.clients),
                    report.created_at,
                ),
            )
        return report

    def delete(self, ids: list[str], owner_id: str) -> int:
        """Delete the given ids **only if owned by ``owner_id``**; return rows removed."""
        if not ids:
            return 0
        placeholders = ",".join("?" for _ in ids)
        with self._connect() as conn:
            cursor = conn.execute(
                f"DELETE FROM saved_reports WHERE owner_id = ? AND id IN ({placeholders})",
                (owner_id, *ids),
            )
            return cursor.rowcount

    def record_audit(self, actor: str, action: str, detail: str) -> None:
        """Append an audit entry (destructive actions are always traced)."""
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO audit_log (ts, actor, action, detail) VALUES (?, ?, ?, ?)",
                (_now(), actor, action, detail),
            )

    # --- Reads (all ownership-scoped) ----------------------------------------

    def list(self, owner_id: str) -> list[SavedReport]:
        """Return all of the owner's reports, newest first."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM saved_reports WHERE owner_id = ? ORDER BY created_at DESC",
                (owner_id,),
            ).fetchall()
        return [self._row_to_report(row) for row in rows]

    def find(
        self, owner_id: str, *, client: str | None = None, today: bool = False
    ) -> list[SavedReport]:
        """Return the owner's reports matching the filters (client mention and/or today).

        ``client`` matches as a case-insensitive substring of the title, content, or
        the structured ``clients`` list (so "mentioning Client X" works whether or not
        the client was tagged). With no filters, returns all the owner's reports.
        """
        reports = self.list(owner_id)
        if client:
            needle = client.lower()
            reports = [r for r in reports if self._mentions(r, needle)]
        if today:
            stamp = date.today().isoformat()
            reports = [r for r in reports if r.created_at.startswith(stamp)]
        return reports

    @staticmethod
    def _mentions(report: SavedReport, needle: str) -> bool:
        haystack = f"{report.title}\n{report.content}\n{' '.join(report.clients)}".lower()
        return needle in haystack

    def audit_tail(self, limit: int = 20) -> list[dict]:
        """Return the most recent audit entries (newest first) — for tests/inspection."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT ts, actor, action, detail FROM audit_log ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]

    # --- Seeding --------------------------------------------------------------

    def _seed_if_empty(self, seed_dir: str) -> None:
        with self._connect() as conn:
            (count,) = conn.execute("SELECT COUNT(*) FROM saved_reports").fetchone()
        if count:
            return
        directory = Path(seed_dir)
        if not directory.is_dir():
            return
        seeded = 0
        for path in sorted(directory.glob("*.json")):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                self.save(
                    data["owner_id"],
                    title=data["title"],
                    content=data["content"],
                    clients=data.get("clients", []),
                    created_at=data.get("created_at"),
                )
                seeded += 1
            except (OSError, json.JSONDecodeError, KeyError) as exc:
                logger.warning("Skipping bad seed report %s: %s", path.name, exc)
        if seeded:
            logger.info("Seeded %d saved reports from %s", seeded, seed_dir)
