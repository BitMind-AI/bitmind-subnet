"""Tests for the validator's bounded last-known-good KOTH cache."""

import json

from neurons.validator.validator import KINGS_CACHE_TTL_SECONDS, _KingsState


def test_kings_cache_expires_after_24_hours():
    state = _KingsState()
    payload = {"kings": [], "chain": {}, "split": {}}
    state.update(payload, now=100.0)

    assert state.fresh_payload(now=100.0 + KINGS_CACHE_TTL_SECONDS) == payload
    assert state.fresh_payload(now=100.0 + KINGS_CACHE_TTL_SECONDS + 1) is None


def test_kings_cache_persists_timestamp(tmp_path):
    state = _KingsState()
    payload = {"kings": [], "chain": {}, "split": {}}
    state.update(payload, now=123.0)
    state.save_state(str(tmp_path), "kings.json")

    restored = _KingsState()
    assert restored.load_state(str(tmp_path), "kings.json") is True
    assert restored.payload == payload
    assert restored.fetched_at == 123.0


def test_legacy_cache_without_timestamp_is_stale(tmp_path):
    payload = {"kings": []}
    (tmp_path / "kings.json").write_text(json.dumps(payload))

    state = _KingsState()
    assert state.load_state(str(tmp_path), "kings.json") is True
    assert state.payload == payload
    assert state.fresh_payload(now=100.0) is None
