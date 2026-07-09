"""Tests for scene remixing (semi-synthetic scene generation)."""

import random


from gas.generation.prompts.scene import SceneDescription
from gas.generation.prompts.scene_remix import remix


def _scene():
    return SceneDescription(
        caption="A fisherman repairs a net on a wooden dock at golden hour.",
        subject="fisherman repairing net",
        scene_kind="documentary",
        setting="wooden dock",
        environment_type="outdoor",
        time_of_day="golden hour",
        weather="clear",
        lighting="warm low sun",
        color_palette="amber and teal",
        dynamic_candidates=["water", "net", "gulls"],
        plausible_events=["a gull lands on a post"],
    )


SETTING_POOL = ["night market alley", "hospital waiting room", "alpine trailhead"]


def test_remix_changes_one_to_three_fields():
    rng = random.Random(0)
    original = _scene()
    remixed = remix(original, rng, setting_pool=SETTING_POOL)
    mutable = ("time_of_day", "weather", "setting", "environment_type", "mood")
    changed = [f for f in mutable if getattr(remixed, f) != getattr(original, f)]
    assert 1 <= len(changed) <= 3


def test_subject_preserved():
    rng = random.Random(1)
    remixed = remix(_scene(), rng, setting_pool=SETTING_POOL)
    assert remixed.subject == "fisherman repairing net"


def test_caption_replaced_with_stub():
    rng = random.Random(2)
    original = _scene()
    remixed = remix(original, rng, setting_pool=SETTING_POOL)
    assert remixed.caption != original.caption
    assert "fisherman repairing net" in remixed.caption


def test_lighting_cleared_when_time_changes():
    found = False
    for seed in range(100):
        rng = random.Random(seed)
        original = _scene()
        remixed = remix(original, rng, setting_pool=SETTING_POOL)
        if remixed.time_of_day != original.time_of_day:
            assert remixed.lighting == ""
            assert remixed.color_palette == ""
            found = True
            break
    assert found, "time_of_day never changed in 100 seeds"


def test_original_not_mutated():
    rng = random.Random(3)
    original = _scene()
    before = original.to_dict()
    remix(original, rng, setting_pool=SETTING_POOL)
    assert original.to_dict() == before


def test_works_without_setting_pool():
    rng = random.Random(4)
    remixed = remix(_scene(), rng)
    assert remixed.subject == "fisherman repairing net"
