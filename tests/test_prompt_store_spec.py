"""Tests for prompt spec persistence (schema migration + write path)."""

import json

import pytest

from gas.cache.db.connection import ConnectionManager, create_schema
from gas.cache.db.prompt_store import PromptStore


@pytest.fixture()
def store(tmp_path):
    db = ConnectionManager(tmp_path / "prompts.db")
    with db.connect() as conn:
        create_schema(conn)
    return PromptStore(db)


def test_spec_columns_exist(store):
    with store.db.connect() as conn:
        cols = {r[1] for r in conn.execute("PRAGMA table_info(prompts)")}
    for col in ("register", "length_band", "event_count", "scene_json", "spec_json"):
        assert col in cols, f"missing column {col}"


def test_write_with_spec_fields(store):
    pid = store.add_prompt_entry(
        content="A man waits at a bus stop in light rain.",
        modality="video",
        register="phone_casual",
        length_band="short",
        event_count=1,
        scene_json=json.dumps({"subject": "man"}),
        spec_json=json.dumps({"register": "phone_casual"}),
    )
    with store.db.connect() as conn:
        row = conn.execute(
            "SELECT register, length_band, event_count, scene_json, spec_json "
            "FROM prompts WHERE id = ?",
            (pid,),
        ).fetchone()
    assert row[0] == "phone_casual"
    assert row[1] == "short"
    assert row[2] == 1
    assert json.loads(row[3])["subject"] == "man"
    assert json.loads(row[4])["register"] == "phone_casual"


def test_write_without_spec_fields_back_compat(store):
    pid = store.add_prompt_entry(content="A dog runs on a beach.", modality="image")
    with store.db.connect() as conn:
        row = conn.execute(
            "SELECT register, spec_json FROM prompts WHERE id = ?", (pid,)
        ).fetchone()
    assert row[0] is None and row[1] is None
