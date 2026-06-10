"""Tests for near-duplicate rejection at prompt write time."""

import pytest

from gas.cache.db.connection import ConnectionManager, create_schema
from gas.cache.db.prompt_store import PromptStore


@pytest.fixture()
def store(tmp_path):
    db = ConnectionManager(tmp_path / "prompts.db")
    with db.connect() as conn:
        create_schema(conn)
    return PromptStore(db)


BASE = (
    "A cyclist pedals along a wet boardwalk at dawn, gulls scattering ahead "
    "of the front wheel while heavy clouds drift over the bay and the first "
    "joggers pass in the opposite direction wearing bright windbreakers."
)


def test_distinct_prompt_accepted(store):
    a = store.add_prompt_entry(content=BASE, modality="video")
    b = store.add_prompt_entry(
        content=(
            "Inside a cluttered workshop a luthier planes the top of an "
            "unfinished guitar, shavings curling onto the bench as dust "
            "hangs in the light from a single window."
        ),
        modality="video",
    )
    assert a is not None and b is not None and a != b


def test_near_duplicate_rejected(store):
    a = store.add_prompt_entry(content=BASE, modality="video")
    assert a is not None
    # Paraphrase-level duplicate: only a few words changed.
    near = BASE.replace("dawn", "first light").replace("heavy", "low")
    b = store.add_prompt_entry(content=near, modality="video")
    assert b is None


def test_near_duplicate_other_modality_allowed(store):
    a = store.add_prompt_entry(content=BASE, modality="video")
    near = BASE.replace("dawn", "first light")
    b = store.add_prompt_entry(content=near, modality="image")
    assert a is not None and b is not None


def test_exact_duplicate_returns_existing_id(store):
    a = store.add_prompt_entry(content=BASE, modality="video")
    b = store.add_prompt_entry(content=BASE, modality="video")
    assert a == b
