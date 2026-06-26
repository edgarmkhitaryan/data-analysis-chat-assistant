"""Unit tests for free-form user preferences (Requirement 4.1, plan/010 §2).

Pure logic — a temp SQLite store and the pure prompt composer; no LLM/network. The
LLM *merge* in update_prefs is verified live; its deterministic fallback is tested here.
"""

import sqlite3

from assistant.agent.nodes.common import compose_system_prompt
from assistant.agent.nodes.update_prefs import _fallback_merge
from assistant.memory.profiles import ProfileStore, UserPrefs


def test_set_and_get_freeform_preferences(tmp_path):
    store = ProfileStore(str(tmp_path / "app.db"))
    assert store.get("manager_a").preferences == ""  # default: empty
    saved = store.set_preferences("manager_a", "Concise bullets; always show % change.")
    assert saved.preferences == "Concise bullets; always show % change."
    assert saved.updated_at
    assert store.get("manager_a").preferences == "Concise bullets; always show % change."


def test_preferences_are_user_scoped(tmp_path):
    store = ProfileStore(str(tmp_path / "app.db"))
    store.set_preferences("manager_a", "tables please")
    assert store.get("manager_b").preferences == ""  # b unaffected


def test_migrates_legacy_structured_prefs(tmp_path):
    db = str(tmp_path / "app.db")
    with sqlite3.connect(db) as conn:  # simulate the OLD structured schema + a row
        conn.execute(
            "CREATE TABLE user_profiles (user_id TEXT PRIMARY KEY, format TEXT NOT NULL, "
            "verbosity TEXT NOT NULL, favorite_metrics TEXT NOT NULL, updated_at TEXT NOT NULL)"
        )
        conn.execute(
            "INSERT INTO user_profiles VALUES (?, ?, ?, ?, ?)",
            ("manager_a", "bullets", "detailed", "[]", "2026-01-01T00:00:00+00:00"),
        )
    store = ProfileStore(db)  # construction triggers the migration
    migrated = store.get("manager_a").preferences
    assert "bullets" in migrated and "detailed" in migrated
    store.set_preferences("manager_b", "USD currency")  # new schema works for fresh writes
    assert store.get("manager_b").preferences == "USD currency"


def test_compose_injects_preferences_and_oneoff():
    prefs = UserPrefs(user_id="manager_a", preferences="Always include % change vs last quarter.")
    state = {"user_prefs": prefs, "oneoff_preference": "render as a Markdown table"}
    prompt = compose_system_prompt(state, "BASE")
    assert "BASE" in prompt
    assert "% change vs last quarter" in prompt
    assert "render as a Markdown table" in prompt


def test_compose_without_prefs_is_just_base():
    prompt = compose_system_prompt({"user_prefs": UserPrefs(user_id="m")}, "BASE")
    assert prompt == "BASE"  # no prefs, no persona -> base only


def test_fallback_merge():
    assert _fallback_merge("", "use tables") == "use tables"
    assert _fallback_merge("concise", "use tables") == "concise\nuse tables"
