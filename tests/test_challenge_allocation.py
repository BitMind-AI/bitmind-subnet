"""Unit tests for generator challenge slot allocation."""

from types import SimpleNamespace

import numpy as np
import pytest

from gas.config import validate_config_and_neuron_path
from gas.evaluation.challenge_allocation import (
    allocate_challenge_slots,
    allocate_random_slots,
    classify_modality_bucket,
    resolve_challenge_response_stats,
    summarize_assignment_buckets,
)
from gas.evaluation.rewards import (
    GeneratorQualification,
    get_generator_qualification,
    resolve_generator_qualification,
)


def _q(**kwargs):
    return GeneratorQualification(**kwargs)


def test_allocate_random_slots_picks_unique_uids_uniformly():
    assignments, stats = allocate_random_slots(
        list(range(80)),
        ["image", "video"],
        sample_size=50,
        rng=np.random.default_rng(0),
    )
    assert stats["mode"] == "random"
    assert stats["pool"] == 80
    assert len(assignments) == 50
    assert len({uid for uid, _ in assignments}) == 50
    assert {mod for _, mod in assignments} <= {"image", "video"}


def test_allocate_random_slots_includes_every_generator():
    assignments, stats = allocate_random_slots(
        [7],
        ["image"],
        sample_size=50,
        rng=np.random.default_rng(1),
    )
    assert stats["pool"] == 1
    assert assignments == [(7, "image")]


def test_classify_missing_and_short_samples_are_onboarding():
    assert classify_modality_bucket(None, "image") == "onboarding"
    assert classify_modality_bucket(_q(image_n=19, qualified_image=True), "image") == "onboarding"
    assert classify_modality_bucket(_q(image_n=20, qualified_image=True), "image") == "qualified"
    assert classify_modality_bucket(_q(video_n=40, qualified_video=False), "video") == "probe"


def test_missing_qualification_treats_everyone_as_onboarding():
    rng = np.random.default_rng(0)
    assignments, stats = allocate_challenge_slots(
        list(range(60)),
        ["image", "video"],
        None,
        rng=rng,
    )
    assert len(assignments) == 50
    assert len({uid for uid, _ in assignments}) == 50
    assert stats["rolled_onboarding"] is False
    assert stats["onboarding"] == 60
    counts = summarize_assignment_buckets(assignments, None)
    assert counts["onboarding"] == 50


def test_empty_onboarding_rolls_to_40_qualified_10_probe():
    qualification = {
        uid: _q(image_n=40, qualified_image=True, video_n=40, qualified_video=True)
        for uid in range(40)
    }
    for uid in range(40, 55):
        qualification[uid] = _q(image_n=40, qualified_image=False, video_n=40, qualified_video=False)

    assignments, stats = allocate_challenge_slots(
        list(range(55)),
        ["image"],
        qualification,
        rng=np.random.default_rng(1),
    )
    assert stats["rolled_onboarding"] is True
    assert stats["onboarding"] == 0
    counts = summarize_assignment_buckets(assignments, qualification)
    assert counts["onboarding"] == 0
    assert counts["qualified"] == 40
    assert counts["probe"] == 10
    assert len(assignments) == 50


def test_unused_qualified_overflows_to_probe():
    qualification = {
        uid: _q(image_n=40, qualified_image=True)
        for uid in range(10)
    }
    for uid in range(10, 60):
        qualification[uid] = _q(image_n=40, qualified_image=False)

    assignments, stats = allocate_challenge_slots(
        list(range(60)),
        ["image"],
        qualification,
        rng=np.random.default_rng(2),
    )
    assert stats["rolled_onboarding"] is True
    counts = summarize_assignment_buckets(assignments, qualification)
    assert counts["qualified"] == 10
    assert counts["probe"] == 40
    assert len(assignments) == 50


def test_unused_probe_overflows_to_qualified():
    qualification = {
        uid: _q(image_n=40, qualified_image=True)
        for uid in range(48)
    }
    for uid in range(48, 50):
        qualification[uid] = _q(image_n=40, qualified_image=False)

    assignments, _ = allocate_challenge_slots(
        list(range(50)),
        ["image"],
        qualification,
        rng=np.random.default_rng(3),
    )
    counts = summarize_assignment_buckets(assignments, qualification)
    assert counts["probe"] == 2
    assert counts["qualified"] == 48
    assert len(assignments) == 50


def test_partial_onboarding_keeps_36_8_6_then_overflows():
    qualification = {
        uid: _q(image_n=5, qualified_image=False)
        for uid in range(3)
    }
    for uid in range(3, 43):
        qualification[uid] = _q(image_n=40, qualified_image=True)
    for uid in range(43, 60):
        qualification[uid] = _q(image_n=40, qualified_image=False)

    assignments, stats = allocate_challenge_slots(
        list(range(60)),
        ["image"],
        qualification,
        rng=np.random.default_rng(4),
    )
    assert stats["rolled_onboarding"] is False
    assert stats["onboarding"] == 3
    counts = summarize_assignment_buckets(assignments, qualification)
    assert counts["onboarding"] == 3
    assert counts["qualified"] == 40  # 36 + 5 leftover onboarding, capped by 40 qualified miners
    assert counts["probe"] == 7
    assert len(assignments) == 50


def test_pays_only_cleared_modality_in_that_bucket():
    qualification = {
        0: _q(image_n=40, qualified_image=True, video_n=40, qualified_video=False),
        1: _q(image_n=40, qualified_image=False, video_n=40, qualified_video=True),
    }
    # Force image slots only so UID 0 is qualified and UID 1 is probe.
    assignments, _ = allocate_challenge_slots(
        [0, 1],
        ["image"],
        qualification,
        sample_size=2,
        qualified_slots=1,
        onboarding_slots=0,
        probe_slots=1,
        rng=np.random.default_rng(5),
    )
    by_uid = dict(assignments)
    assert by_uid[0] == "image"
    assert by_uid[1] == "image"
    counts = summarize_assignment_buckets(assignments, qualification)
    assert counts["qualified"] == 1
    assert counts["probe"] == 1


def test_unknown_uid_is_onboarding():
    qualification = {0: _q(image_n=40, qualified_image=True)}
    assignments, stats = allocate_challenge_slots(
        [0, 1],
        ["image"],
        qualification,
        sample_size=2,
        qualified_slots=1,
        onboarding_slots=1,
        probe_slots=0,
        rng=np.random.default_rng(6),
    )
    assert stats["onboarding"] == 1
    counts = summarize_assignment_buckets(assignments, qualification)
    assert counts["onboarding"] == 1
    assert counts["qualified"] == 1


def test_never_leaves_slots_empty_when_miners_exist():
    qualification = {uid: _q(image_n=40, qualified_image=True) for uid in range(12)}
    assignments, _ = allocate_challenge_slots(
        list(range(12)),
        ["video"],
        qualification,
        sample_size=12,
        qualified_slots=8,
        onboarding_slots=2,
        probe_slots=2,
        rng=np.random.default_rng(7),
    )
    assert len(assignments) == 12
    assert {uid for uid, _ in assignments} == set(range(12))


def test_slot_counts_must_sum_to_sample_size(tmp_path):
    config = SimpleNamespace(
        logging=SimpleNamespace(logging_dir=str(tmp_path)),
        wallet=SimpleNamespace(name="w", hotkey="h"),
        netuid=1,
        neuron=SimpleNamespace(
            name="bitmind",
            sample_size=50,
            qualified_slots=36,
            onboarding_slots=8,
            probe_slots=5,
        ),
    )
    with pytest.raises(ValueError, match="must equal neuron.sample_size"):
        validate_config_and_neuron_path(config)


def test_fresh_cache_does_not_give_replacement_qualified_challenge_slots():
    metagraph = SimpleNamespace(hotkeys=["old-owner", "unchanged"])
    cached = get_generator_qualification([
        {"ss58_address": hotkey, "modality": "image", "fooled_count": 2, "not_fooled_count": 18}
        for hotkey in metagraph.hotkeys
    ], metagraph)
    before = resolve_generator_qualification(cached, metagraph)
    assert classify_modality_bucket(before[0], "image") == "qualified"

    # No new score update or API failure is required: registrations can change
    # while the challenge manager still considers the last fetch fresh.
    metagraph.hotkeys[0] = "replacement"
    current = resolve_generator_qualification(cached, metagraph)
    assignments, stats = allocate_challenge_slots(
        [0, 1], ["image"], current,
        sample_size=2, qualified_slots=1, onboarding_slots=1, probe_slots=0,
        rng=np.random.default_rng(446),
    )
    assert dict(assignments) == {0: "image", 1: "image"}
    assert classify_modality_bucket(current.get(0), "image") == "onboarding"
    assert classify_modality_bucket(current.get(1), "image") == "qualified"
    assert stats["image_qualified"] == 1
    assert stats["onboarding"] == 1
    assert summarize_assignment_buckets(assignments, current) == {
        "qualified": 1, "onboarding": 1, "probe": 0, "unresponsive": 0,
    }


def test_no_answers_are_unresponsive_not_onboarding():
    q = _q(video_n=40, qualified_video=False)
    stats = {"image": {"answered": 0, "no_answer": 5}, "video": {"answered": 8, "no_answer": 0}}
    assert classify_modality_bucket(q, "image", response_stats=stats) == "unresponsive"
    assert classify_modality_bucket(q, "video", response_stats=stats) == "probe"
    assert classify_modality_bucket(q, "image") == "onboarding"
    few = {"image": {"answered": 0, "no_answer": 4}}
    assert classify_modality_bucket(q, "image", response_stats=few) == "onboarding"
    answered = {"image": {"answered": 1, "no_answer": 20}}
    assert classify_modality_bucket(q, "image", response_stats=answered) == "onboarding"


def test_video_specialist_who_ignores_image_is_not_assigned_image():
    qualification = {
        0: _q(video_n=40, qualified_video=False),
    }
    for uid in range(1, 40):
        qualification[uid] = _q(
            image_n=40, qualified_image=True, video_n=40, qualified_video=True
        )
    response_stats = {
        0: {
            "image": {"answered": 0, "no_answer": 5},
            "video": {"answered": 12, "no_answer": 0},
        }
    }
    video_hits = 0
    for seed in range(40):
        assignments, stats = allocate_challenge_slots(
            list(qualification),
            ["image", "video"],
            qualification,
            response_stats=response_stats,
            rng=np.random.default_rng(seed),
        )
        assert stats["image_unresponsive"] == 1
        assert 0 not in {uid for uid, _ in assignments if _ == "image"}
        if (0, "video") in assignments:
            video_hits += 1
    assert video_hits > 0


def test_unresponsive_still_applies_when_qualification_is_missing():
    response_stats = {0: {"image": {"answered": 0, "no_answer": 5}}}
    assignments, stats = allocate_challenge_slots(
        [0, 1],
        ["image"],
        None,
        sample_size=2,
        qualified_slots=1,
        onboarding_slots=1,
        probe_slots=0,
        response_stats=response_stats,
        rng=np.random.default_rng(8),
    )
    assert stats["image_unresponsive"] == 1
    assert stats["onboarding"] == 1
    assert dict(assignments) == {1: "image"}


def test_response_stats_do_not_follow_recycled_uid():
    metagraph = SimpleNamespace(hotkeys=["replacement", "unchanged"])
    by_hotkey = {
        "old-owner": {"image": {"answered": 0, "no_answer": 9}},
        "unchanged": {"image": {"answered": 3, "no_answer": 0}},
    }
    resolved = resolve_challenge_response_stats(by_hotkey, metagraph)
    assert 0 not in resolved
    assert resolved[1]["image"]["answered"] == 3
    assert classify_modality_bucket(
        None, "image", response_stats=resolved.get(0)
    ) == "onboarding"
