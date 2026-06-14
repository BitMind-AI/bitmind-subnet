"""Tests for PromptSpec injection into PromptGenerator message building.

GPU-free: only exercises the static message-construction path.
"""

from collections import deque

from gas.generation.prompts.prompt_generator import PromptGenerator
from gas.generation.prompts.register_sampler import sample_spec
from gas.generation.prompts.scene import SceneDescription


def _scene():
    return SceneDescription(
        caption="A man sits at a desk near a window.",
        subject="man at desk",
        setting="home office",
        dynamic_candidates=["curtains", "dust motes"],
    )


def test_user_message_carries_committed_spec():
    spec = sample_spec(modality="video", rng_seed=3)
    msg = PromptGenerator._build_user_message(
        _scene(), deque(), "", spec=spec
    )
    assert "COMMITTED SHOT SPEC" in msg
    assert spec.register in msg
    assert str(spec.length_words[0]) in msg and str(spec.length_words[1]) in msg


def test_video_spec_includes_camera_and_events():
    spec = sample_spec(modality="video", rng_seed=3)
    msg = PromptGenerator._build_user_message(_scene(), deque(), "", spec=spec)
    assert spec.camera_motion in msg
    if spec.event_count > 0:
        assert "discrete event" in msg
    else:
        assert "ambient motion only" in msg


def test_plain_register_bans_rendered():
    # find a seed yielding a plain register
    spec = None
    for seed in range(100):
        s = sample_spec(modality="video", rng_seed=seed)
        if s.style_strictness == "plain":
            spec = s
            break
    assert spec is not None
    msg = PromptGenerator._build_user_message(_scene(), deque(), "", spec=spec)
    assert "Forbidden" in msg
    assert "as if" in msg


def test_no_spec_is_backward_compatible():
    msg = PromptGenerator._build_user_message(_scene(), deque(), "100-180 words")
    assert "COMMITTED SHOT SPEC" not in msg
    assert "100-180 words" in msg


def test_plausible_events_listed_when_present():
    scene = _scene()
    scene.plausible_events = ["a courier knocks", "the desk lamp flickers"]
    spec = None
    for seed in range(200):
        s = sample_spec(modality="video", rng_seed=seed)
        if s.event_count > 0:
            spec = s
            break
    assert spec is not None
    msg = PromptGenerator._build_user_message(scene, deque(), "", spec=spec)
    assert "a courier knocks" in msg
