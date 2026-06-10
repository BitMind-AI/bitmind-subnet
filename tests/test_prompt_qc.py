"""Tests for the prompt QC gate."""

from gas.generation.prompts.prompt_qc import validate
from gas.generation.prompts.register_sampler import sample_spec


def _plain_spec():
    for seed in range(200):
        s = sample_spec(modality="video", rng_seed=seed)
        if s.style_strictness == "plain" and s.length_band == "medium":
            return s
    raise AssertionError("no plain/medium spec found")


def _free_spec():
    for seed in range(200):
        s = sample_spec(modality="video", rng_seed=seed)
        if s.style_strictness == "free" and s.length_band == "medium":
            return s
    raise AssertionError("no free/medium spec found")


def _words(n):
    return " ".join(f"word{i}" for i in range(n))


def test_accepts_clean_prompt_in_band():
    spec = _free_spec()
    lo, hi = spec.length_words
    text = "A man crosses a rainy street while " + _words(lo)
    ok, reason = validate(text, spec)
    assert ok, reason


def test_rejects_meta_text():
    spec = _free_spec()
    lo, _ = spec.length_words
    for bad in ("Here is a prompt: ", "COMMITTED SHOT SPEC ", "As an AI "):
        ok, reason = validate(bad + _words(lo + 5), spec)
        assert not ok
        assert "meta" in reason


def test_rejects_structural_markup():
    spec = _free_spec()
    lo, _ = spec.length_words
    ok, reason = validate("- bullet one\n- bullet two " + _words(lo), spec)
    assert not ok


def test_rejects_out_of_band_length():
    spec = _free_spec()
    lo, hi = spec.length_words
    ok, reason = validate(_words(int(hi * 1.4)), spec)
    assert not ok and "length" in reason
    ok, reason = validate(_words(max(3, int(lo * 0.5))), spec)
    assert not ok and "length" in reason


def test_band_tolerance():
    spec = _free_spec()
    lo, hi = spec.length_words
    # 10% over the ceiling is within the +-20% tolerance
    ok, reason = validate(_words(int(hi * 1.1)), spec)
    assert ok, reason


def test_plain_register_bans_enforced():
    spec = _plain_spec()
    lo, _ = spec.length_words
    text = "The hallway sits empty as if holding its breath " + _words(lo)
    ok, reason = validate(text, spec)
    assert not ok and "banned" in reason


def test_free_register_allows_literary_style():
    spec = _free_spec()
    lo, _ = spec.length_words
    text = "The hallway sits empty as if holding its breath " + _words(lo)
    ok, reason = validate(text, spec)
    assert ok, reason


def test_no_spec_minimal_checks():
    ok, reason = validate("A short but valid plain prompt about a dog. " + _words(30), None)
    assert ok
    ok, reason = validate("Here is a prompt: " + _words(30), None)
    assert not ok
