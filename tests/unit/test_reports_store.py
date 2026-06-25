"""Unit tests for the Saved Reports store (plan/007 §4, plan/011 §1.1).

Pure SQLite, no network. Covers ownership scoping (cannot see/delete another
owner's reports), client/today filters, transactional delete, and the audit log.
"""

import json

import pytest

from assistant.reports.store import SavedReportStore


@pytest.fixture
def store(tmp_path):
    return SavedReportStore(str(tmp_path / "reports.db"))


def test_save_and_list_is_owner_scoped(store):
    store.save("manager_a", title="A1", content="hello")
    store.save("manager_b", title="B1", content="world")
    assert [r.title for r in store.list("manager_a")] == ["A1"]
    assert [r.title for r in store.list("manager_b")] == ["B1"]


def test_find_by_client_substring_and_owner_scoped(store):
    store.save("manager_a", title="Acme review", content="acme stuff", clients=["Acme"])
    store.save("manager_a", title="Other", content="nothing relevant")
    store.save("manager_b", title="Acme for B", content="acme", clients=["Acme"])

    res = store.find("manager_a", client="acme")  # case-insensitive
    assert [r.title for r in res] == ["Acme review"]
    assert all(r.owner_id == "manager_a" for r in res)  # never sees B's Acme report


def test_find_today_filters_by_creation_date(store):
    store.save("manager_a", title="old", content="x", created_at="2020-01-01T00:00:00+00:00")
    store.save("manager_a", title="new", content="y")  # created_at defaults to now
    assert [r.title for r in store.find("manager_a", today=True)] == ["new"]


def test_delete_is_owner_scoped(store):
    ra = store.save("manager_a", title="A", content="x")
    rb = store.save("manager_b", title="B", content="y")

    assert store.delete([rb.id], "manager_a") == 0  # cannot delete another owner's report
    assert len(store.list("manager_b")) == 1

    assert store.delete([ra.id], "manager_a") == 1  # can delete own
    assert store.list("manager_a") == []


def test_delete_multiple_returns_count(store):
    ids = [store.save("manager_a", title=f"R{i}", content="x").id for i in range(3)]
    assert store.delete(ids, "manager_a") == 3
    assert store.list("manager_a") == []


def test_delete_empty_is_noop(store):
    assert store.delete([], "manager_a") == 0


def test_audit_log_records_and_reads(store):
    store.record_audit("manager_a", "delete", "deleted=2 ids=['x','y']")
    tail = store.audit_tail()
    assert tail[0]["actor"] == "manager_a"
    assert tail[0]["action"] == "delete"
    assert "deleted=2" in tail[0]["detail"]


def test_seeding_runs_once(tmp_path):
    seed_dir = tmp_path / "seeds"
    seed_dir.mkdir()
    (seed_dir / "r.json").write_text(
        json.dumps(
            {
                "owner_id": "manager_a",
                "title": "Seeded",
                "content": "c",
                "clients": ["Z"],
                "created_at": "2026-01-01T00:00:00+00:00",
            }
        )
    )
    db = str(tmp_path / "reports.db")
    store = SavedReportStore(db, seed_dir=str(seed_dir))
    assert [r.title for r in store.list("manager_a")] == ["Seeded"]

    # Re-opening the same db does not double-seed.
    store2 = SavedReportStore(db, seed_dir=str(seed_dir))
    assert len(store2.list("manager_a")) == 1
